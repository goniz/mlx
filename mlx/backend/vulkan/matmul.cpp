// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/matmul.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace mlx::core {

namespace {

constexpr uint32_t kMaxGridZ = 65535;
constexpr uint32_t kMaxMulMatVecCols = 16;
constexpr char kMatvecMatrixCastScratchLane[] = "matvec.matrix_f16";
constexpr char kMatvecVectorCastScratchLane[] = "matvec.vec_f16";
constexpr char kMatvecOutScratchLane[] = "matvec.out_work";
constexpr char kMatvecScoresVOutScratchLane[] = "matvec.scores_v.out_work";
constexpr char kMatmulZeroScratchLane[] = "matmul.zero.out";
constexpr char kMulMmACastScratchLane[] = "mul_mm.a_f16";
constexpr char kMulMmBCastScratchLane[] = "mul_mm.b_f16";
constexpr char kMulMmOutScratchLane[] = "mul_mm.out_work";
constexpr char kMulMmSplitKScratchLane[] = "mul_mm.split_k";
constexpr char kBlockMaskedMMLhsScratchLane[] = "block_masked_mm.lhs";
constexpr char kBlockMaskedMMRhsScratchLane[] = "block_masked_mm.rhs";

bool is_supported_matmul_dtype(Dtype dtype) {
  return dtype == float32 || dtype == float16 || dtype == bfloat16;
}

void insert_compute_barrier(VkCommandBuffer command_buffer) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask =
      static_cast<VkAccessFlags>(vk::AccessFlagBits::eShaderWrite);
  barrier.dstAccessMask = static_cast<VkAccessFlags>(
      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
  vkCmdPipelineBarrier(
      command_buffer,
      static_cast<VkPipelineStageFlags>(
          vk::PipelineStageFlagBits::eComputeShader),
      static_cast<VkPipelineStageFlags>(
          vk::PipelineStageFlagBits::eComputeShader),
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);
}

bool matmul_debug_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_MATMUL_DEBUG");
    return env != nullptr && std::string(env) == "1";
  }();
  return enabled;
}

bool matvec_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_ENABLE_MATVEC");
    if (env == nullptr) {
      return true;
    }
    return std::string(env) != "0";
  }();
  return enabled;
}

bool prefer_subgroup_matvec() {
  static const bool prefer = []() {
    const char* env = std::getenv("MLX_VULKAN_PREFER_SUBGROUP_MATVEC");
    if (env != nullptr) {
      return std::string(env) != "0";
    }
    return vulkan::VulkanContext::get().architecture() ==
        vulkan::GpuArchitecture::Nvidia;
  }();
  return prefer;
}

bool mul_mm_enabled() {
  static auto& runtime_disabled = []() -> std::atomic<bool>& {
    static std::atomic<bool> disabled{false};
    return disabled;
  }();
  if (runtime_disabled.load(std::memory_order_relaxed)) {
    return false;
  }

  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_ENABLE_MUL_MM");
    if (env == nullptr) {
      return true;
    }
    return std::string(env) != "0";
  }();
  return enabled;
}

void disable_mul_mm_runtime(const std::string& reason) {
  static auto& runtime_disabled = []() -> std::atomic<bool>& {
    static std::atomic<bool> disabled{false};
    return disabled;
  }();

  const bool was_disabled =
      runtime_disabled.exchange(true, std::memory_order_relaxed);
  if (!was_disabled && matmul_debug_enabled()) {
    std::cerr << "[vulkan::mul_mm] disabling mul_mm after failure: " << reason
              << "\n";
  }
}

void log_matmul_path(const std::vector<array>& inputs, const char* path) {
  if (!matmul_debug_enabled() || inputs.size() < 2) {
    return;
  }
  static int printed = 0;
  if (printed >= 32) {
    return;
  }
  ++printed;
  std::cerr << "[vulkan::matmul] path=" << path
            << " a_shape=" << inputs[0].shape()
            << " b_shape=" << inputs[1].shape() << "\n";
}

std::optional<vulkan::StaticShaderId> matvec_shader_name(
    Dtype matrix_dtype,
    Dtype vec_dtype) {
  if (matrix_dtype == float32 && vec_dtype == float32) {
    return vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32;
  }
  if (matrix_dtype == float16 && vec_dtype == float32) {
    return vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32;
  }
  if (matrix_dtype == bfloat16 && vec_dtype == float32) {
    if (!vulkan::VulkanContext::get().shader_bfloat16_supported()) {
      return std::nullopt;
    }
    return vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32;
  }
  if (matrix_dtype == float32 && vec_dtype == float16) {
    return vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32;
  }
  if (matrix_dtype == float16 && vec_dtype == float16) {
    return vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32;
  }
  if (matrix_dtype == bfloat16 && vec_dtype == float16) {
    if (!vulkan::VulkanContext::get().shader_bfloat16_supported()) {
      return std::nullopt;
    }
    return vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32;
  }
  return std::nullopt;
}

std::vector<vulkan::StaticShaderId> matvec_shader_candidates(
    Dtype matrix_dtype,
    Dtype vec_dtype) {
  auto base = matvec_shader_name(matrix_dtype, vec_dtype);
  if (!base.has_value()) {
    return {};
  }
  const auto order = [&](vulkan::StaticShaderId standard,
                         vulkan::StaticShaderId subgroup,
                         vulkan::StaticShaderId subgroup_no_shmem) {
    if (prefer_subgroup_matvec()) {
      return std::vector<vulkan::StaticShaderId>{
          subgroup,
          subgroup_no_shmem,
          standard,
      };
    }
    return std::vector<vulkan::StaticShaderId>{
        standard,
        subgroup,
        subgroup_no_shmem,
    };
  };
  switch (*base) {
    case vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32:
      return order(
          vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32,
          vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_f32_f32_f32_subgroup_no_shmem);
    case vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32:
      return order(
          vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32,
          vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_f16_f32_f32_subgroup_no_shmem);
    case vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32:
      return order(
          vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32,
          vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_bf16_f32_f32_subgroup_no_shmem);
    case vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32:
      return order(
          vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32,
          vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_f32_f16_f32_subgroup_no_shmem);
    case vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32:
      return order(
          vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32,
          vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_f16_f16_f32_subgroup_no_shmem);
    case vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32:
      return order(
          vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32,
          vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32_subgroup,
          vulkan::StaticShaderId::mul_mat_vec_bf16_f16_f32_subgroup_no_shmem);
    case vulkan::StaticShaderId::Count:
      break;
  }
  return {};
}

enum class MatmulFamily : uint8_t {
  Small,
  Medium,
  Large,
};

struct MatmulFamilySpec {
  std::array<uint32_t, 11> spec;
  uint32_t fp32_accum_k_threshold;
};

struct MatmulProfile {
  uint32_t preferred_subgroup_size;
  std::array<MatmulFamilySpec, 3> aligned;
  std::array<MatmulFamilySpec, 3> unaligned;
};

struct MatmulDispatchTuning {
  std::vector<uint32_t> specialization_constants;
  MatmulFamily family{MatmulFamily::Small};
  bool aligned{false};
  bool prefer_fp32_accum{false};
  uint32_t split_k_threshold{0};
};

constexpr std::array<uint32_t, 11> kSafeMatmulSpec =
    {32, 32, 32, 16, 32, 32, 2, 2, 2, 1, 32};
constexpr std::array<uint32_t, 11> kLane64MatmulSpec =
    {64, 32, 32, 16, 32, 32, 2, 2, 2, 1, 64};

bool supports_64_lane_matmul(const vulkan::VulkanContext& ctx) {
  return ctx.subgroup_size() >= 64u ||
      (ctx.subgroup_size_control_supported() &&
       ctx.subgroup_min_size() <= 64u && ctx.subgroup_max_size() >= 64u);
}

std::vector<vulkan::StaticShaderId> mul_mm_shader_candidates(
    Dtype dtype,
    bool prefer_fp32_accum) {
  switch (dtype) {
    case float16:
      return prefer_fp32_accum
          ? std::vector<vulkan::StaticShaderId>{
                vulkan::StaticShaderId::matmul_f16_fp32,
                vulkan::StaticShaderId::matmul_f16,
            }
          : std::vector<vulkan::StaticShaderId>{
                vulkan::StaticShaderId::matmul_f16,
                vulkan::StaticShaderId::matmul_f16_fp32,
            };
    case bfloat16:
      if (!vulkan::VulkanContext::get().shader_bfloat16_supported()) {
        return {};
      }
      return prefer_fp32_accum
          ? std::vector<vulkan::StaticShaderId>{
                vulkan::StaticShaderId::matmul_bf16_fp32,
                vulkan::StaticShaderId::matmul_bf16,
            }
          : std::vector<vulkan::StaticShaderId>{
                vulkan::StaticShaderId::matmul_bf16,
                vulkan::StaticShaderId::matmul_bf16_fp32,
            };
    case float32:
      return {
          vulkan::StaticShaderId::matmul_f32_f32_fp32,
          vulkan::StaticShaderId::matmul_f32_f32,
      };
    default:
      return {};
  }
}

std::vector<vulkan::StaticShaderId> mul_mm_direct_shader_candidates(
    Dtype input_dtype,
    Dtype output_dtype,
    bool prefer_fp32_accum) {
  switch (input_dtype) {
    case float16:
      switch (output_dtype) {
        case float16:
          return prefer_fp32_accum
              ? std::vector<vulkan::StaticShaderId>{
                    vulkan::StaticShaderId::matmul_direct_f16,
                    vulkan::StaticShaderId::matmul_direct_f16_f16acc,
                }
              : std::vector<vulkan::StaticShaderId>{
                    vulkan::StaticShaderId::matmul_direct_f16_f16acc,
                    vulkan::StaticShaderId::matmul_direct_f16,
                };
        case bfloat16:
          return prefer_fp32_accum
              ? std::vector<vulkan::StaticShaderId>{
                    vulkan::StaticShaderId::matmul_direct_f16_bf16,
                    vulkan::StaticShaderId::matmul_direct_f16_bf16_f16acc,
                }
              : std::vector<vulkan::StaticShaderId>{
                    vulkan::StaticShaderId::matmul_direct_f16_bf16_f16acc,
                    vulkan::StaticShaderId::matmul_direct_f16_bf16,
                };
        case float32:
          return mul_mm_shader_candidates(input_dtype, prefer_fp32_accum);
        default:
          return {};
      }
    case bfloat16:
      if (!vulkan::VulkanContext::get().shader_bfloat16_supported()) {
        return {};
      }
      if (output_dtype != bfloat16) {
        return output_dtype == float32
            ? mul_mm_shader_candidates(input_dtype, prefer_fp32_accum)
            : std::vector<vulkan::StaticShaderId>{};
      }
      return prefer_fp32_accum
          ? std::vector<vulkan::StaticShaderId>{
                vulkan::StaticShaderId::matmul_direct_bf16,
                vulkan::StaticShaderId::matmul_direct_bf16_f16acc,
            }
          : std::vector<vulkan::StaticShaderId>{
                vulkan::StaticShaderId::matmul_direct_bf16_f16acc,
                vulkan::StaticShaderId::matmul_direct_bf16,
            };
    case float32:
      return output_dtype == float32
          ? mul_mm_shader_candidates(input_dtype, prefer_fp32_accum)
          : std::vector<vulkan::StaticShaderId>{};
    default:
      return {};
  }
}

vulkan::StaticShaderId split_k_reduce_shader_id(Dtype out_dtype) {
  switch (out_dtype) {
    case float16:
      return vulkan::StaticShaderId::split_k_reduce_f16;
    case bfloat16:
      return vulkan::StaticShaderId::split_k_reduce_bf16;
    case float32:
    default:
      return vulkan::StaticShaderId::split_k_reduce;
  }
}

const char* matmul_family_name(MatmulFamily family) {
  switch (family) {
    case MatmulFamily::Small:
      return "small";
    case MatmulFamily::Medium:
      return "medium";
    case MatmulFamily::Large:
      return "large";
  }
  return "small";
}

MatmulProfile default_matmul_profile(uint32_t subgroup_size) {
  const bool use_64_lane = subgroup_size >= 64u;
  const auto spec = use_64_lane ? kLane64MatmulSpec : kSafeMatmulSpec;
  return {
      use_64_lane ? 64u : 32u,
      {{{spec, 512}, {spec, 1024}, {spec, 2048}}},
      {{{spec, 512}, {spec, 768}, {spec, 1536}}},
  };
}

MatmulProfile matmul_profile_for_device() {
  const auto& ctx = vulkan::VulkanContext::get();
  const uint32_t device_subgroup = std::max(ctx.subgroup_size(), 32u);
  const bool use_64_lane = supports_64_lane_matmul(ctx);
  const auto amd_spec = use_64_lane ? kLane64MatmulSpec : kSafeMatmulSpec;
  const uint32_t amd_subgroup = use_64_lane ? 64u : 32u;

  switch (ctx.architecture()) {
    case vulkan::GpuArchitecture::AmdRdna:
      return {
          amd_subgroup,
          {{{amd_spec, 768}, {amd_spec, 1536}, {amd_spec, 3072}}},
          {{{amd_spec, 512}, {amd_spec, 1024}, {amd_spec, 2048}}},
      };
    case vulkan::GpuArchitecture::Nvidia:
      return {
          32,
          {{{kSafeMatmulSpec, 2048},
            {kSafeMatmulSpec, 4096},
            {kSafeMatmulSpec, 8192}}},
          {{{kSafeMatmulSpec, 1024},
            {kSafeMatmulSpec, 2048},
            {kSafeMatmulSpec, 4096}}},
      };
    case vulkan::GpuArchitecture::Intel:
    case vulkan::GpuArchitecture::Apple:
    case vulkan::GpuArchitecture::Qualcomm:
      return {
          32,
          {{{kSafeMatmulSpec, 768},
            {kSafeMatmulSpec, 1536},
            {kSafeMatmulSpec, 3072}}},
          {{{kSafeMatmulSpec, 512},
            {kSafeMatmulSpec, 1024},
            {kSafeMatmulSpec, 2048}}},
      };
    case vulkan::GpuArchitecture::AmdCdna:
    case vulkan::GpuArchitecture::Unknown:
      return default_matmul_profile(device_subgroup);
  }
  return default_matmul_profile(device_subgroup);
}

MatmulFamily classify_matmul_family(uint32_t m, uint32_t n, uint32_t k) {
  const uint64_t mn = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
  if (m <= 32 || n <= 32 || mn <= 4096 || k <= 256) {
    return MatmulFamily::Small;
  }
  if (mn <= 65536 || k <= 2048) {
    return MatmulFamily::Medium;
  }
  return MatmulFamily::Large;
}

bool matmul_inputs_aligned(uint32_t m, uint32_t n, uint32_t k) {
  return (m % 4u) == 0 && (n % 8u) == 0 && (k % 8u) == 0;
}

uint32_t round_up_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1u) / divisor;
}

uint32_t
choose_split_k(uint32_t k, uint32_t num_batches, uint32_t split_k_threshold) {
  // Latency-sensitive decode/prefill paths on current Vulkan targets perform
  // better without split-K for the common K<=4096 regime.
  if (num_batches == 1u && k <= 4096u) {
    return 1u;
  }

  if (split_k_threshold == 0 || k < split_k_threshold || num_batches > 2u) {
    return 1u;
  }

  uint32_t split_k = round_up_div(k, split_k_threshold);
  split_k = std::clamp(split_k, 1u, 8u);

  if (k / split_k < 64u) {
    return 1u;
  }

  return split_k;
}

MatmulDispatchTuning
select_matmul_dispatch_tuning(Dtype dtype, uint32_t m, uint32_t n, uint32_t k) {
  const auto profile = matmul_profile_for_device();
  const auto& ctx = vulkan::VulkanContext::get();
  const bool aligned = matmul_inputs_aligned(m, n, k);
  const MatmulFamily family = classify_matmul_family(m, n, k);
  const size_t family_index = static_cast<size_t>(family);
  const auto& family_spec =
      aligned ? profile.aligned[family_index] : profile.unaligned[family_index];

  MatmulDispatchTuning tuning;
  tuning.specialization_constants.assign(
      family_spec.spec.begin(), family_spec.spec.end());
  tuning.family = family;
  tuning.aligned = aligned;
  tuning.split_k_threshold = family_spec.fp32_accum_k_threshold;
  tuning.prefer_fp32_accum = dtype == float32 ||
      (dtype != float32 && k >= family_spec.fp32_accum_k_threshold);

  if (!ctx.cooperative_matrix_supported() &&
      ctx.architecture() == vulkan::GpuArchitecture::AmdRdna &&
      tuning.specialization_constants.size() > 0) {
    tuning.specialization_constants[0] =
        std::min(tuning.specialization_constants[0], 64u);
  }

  if (tuning.specialization_constants.size() > 10 &&
      ctx.subgroup_size_control_supported()) {
    uint32_t preferred = std::clamp(
        profile.preferred_subgroup_size,
        std::max(ctx.subgroup_min_size(), 1u),
        std::max(ctx.subgroup_max_size(), 1u));
    // The scalar matmul shader also uses WARP for lane tiling, so it cannot
    // exceed the selected workgroup size.
    preferred = std::min(preferred, tuning.specialization_constants[0]);
    tuning.specialization_constants[10] = preferred;
  }

  if (ctx.shader_core_count() > 0) {
    const uint32_t core_scale =
        std::clamp(ctx.shader_core_count() / 8u, 1u, 4u);
    tuning.split_k_threshold =
        std::max<uint32_t>(tuning.split_k_threshold, 512u * core_scale);
    if (dtype != float32) {
      tuning.prefer_fp32_accum =
          k >= std::max<uint32_t>(tuning.split_k_threshold, 1024u);
    }
  }

  if (ctx.architecture() == vulkan::GpuArchitecture::AmdRdna &&
      dtype == bfloat16) {
    tuning.prefer_fp32_accum = true;
  }

  return tuning;
}

bool is_row_contiguous_zero_offset(const array& arr) {
  return arr.flags().row_contiguous && arr.offset() == 0 &&
      arr.strides(-1) == 1;
}

struct TensorLayout4D {
  uint32_t ne00{1};
  uint32_t ne01{1};
  uint32_t ne02{1};
  uint32_t ne03{1};
};

TensorLayout4D make_tensor_layout_4d(const array& arr) {
  TensorLayout4D layout;
  for (size_t i = 0; i < arr.ndim() && i < 4; ++i) {
    const int src_dim = static_cast<int>(arr.ndim() - 1 - i);
    const uint32_t dim = static_cast<uint32_t>(arr.shape(src_dim));
    switch (i) {
      case 0:
        layout.ne00 = dim;
        break;
      case 1:
        layout.ne01 = dim;
        break;
      case 2:
        layout.ne02 = dim;
        break;
      case 3:
        layout.ne03 = dim;
        break;
    }
  }
  return layout;
}

bool has_vulkan_buffer(const array& arr) {
  auto data = arr.data_shared_ptr();
  return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
}

array cast_to_float16_scratch(const array& arr, Stream s, const char* lane) {
  array out = vulkan::acquire_scratch_array(s, lane, arr.shape(), float16);
  copy_gpu(arr, out, CopyType::General, s);
  vulkan::mark_scratch_array_written(s, lane);
  return out;
}

bool ensure_vulkan_buffer(array& arr, Stream s);

bool try_eval_scores_v_matvec_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array scores = inputs[0];
  array values = inputs[1];
  if (scores.dtype() != float32 || values.dtype() != float16 ||
      out.dtype() != float32) {
    return false;
  }
  if (scores.ndim() != 4 || values.ndim() != 4 || out.ndim() != 4) {
    return false;
  }
  if (scores.shape(-2) != 1 || out.shape(-2) != 1) {
    return false;
  }
  if (scores.shape(-1) != values.shape(-2) ||
      out.shape(-1) != values.shape(-1)) {
    return false;
  }
  if (scores.shape(0) != values.shape(0) || scores.shape(0) != out.shape(0)) {
    return false;
  }
  if (scores.shape(-3) != out.shape(-3)) {
    return false;
  }

  if (!ensure_vulkan_buffer(scores, s) || !ensure_vulkan_buffer(values, s)) {
    return false;
  }

  if (!is_row_contiguous_zero_offset(scores)) {
    scores = contiguous_copy_gpu(scores, s);
  }
  if (!is_row_contiguous_zero_offset(scores)) {
    return false;
  }

  array values_t = swapaxes_in_eval(values, -1, -2);
  if (!ensure_vulkan_buffer(values_t, s)) {
    return false;
  }
  if (!is_row_contiguous_zero_offset(values_t)) {
    values_t = contiguous_copy_gpu(values_t, s);
  }
  if (!is_row_contiguous_zero_offset(values_t)) {
    return false;
  }

  array out_work = out;
  const bool needs_out_copy = !is_row_contiguous_zero_offset(out_work);
  if (needs_out_copy) {
    out_work = vulkan::acquire_scratch_array(
        s, kMatvecScoresVOutScratchLane, out.shape(), float32);
  } else {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  }
  if (!ensure_vulkan_buffer(out_work, s)) {
    return false;
  }

  const auto matrix_layout = make_tensor_layout_4d(values_t);
  const auto vec_layout = make_tensor_layout_4d(scores);
  const uint32_t kv_heads = matrix_layout.ne02;
  const uint32_t q_heads = vec_layout.ne02;
  if (kv_heads == 0 || q_heads == 0 || (q_heads % kv_heads) != 0) {
    return false;
  }
  const uint32_t gqa_ratio = q_heads / kv_heads;
  if (gqa_ratio == 0 || gqa_ratio > 8) {
    return false;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    bool dispatched = false;
    if (matrix_layout.ne03 == 1 && gqa_ratio > 1) {
      for (auto shader_id : {
               vulkan::StaticShaderId::mul_mat_vec_p021_f16_f32_subgroup_add,
               vulkan::StaticShaderId::mul_mat_vec_p021_f16_f32,
           }) {
        try {
          vulkan::dispatch_mul_mat_vec_p021_op(
              values_t,
              scores,
              out_work,
              shader_id,
              command_buffer,
              s,
              {
                  matrix_layout.ne00,
                  matrix_layout.ne01,
                  matrix_layout.ne02,
                  vec_layout.ne02,
                  0,
                  0,
                  0,
              },
              {1u, matrix_layout.ne01, kv_heads},
              {std::max(vulkan::VulkanContext::get().subgroup_size(), 32u),
               gqa_ratio});
          dispatched = true;
          break;
        } catch (const std::runtime_error&) {
          // The subgroup variant is opportunistic; fall back to the scalar
          // p021 shader if the subgroup pipeline is unsupported on this GPU.
        }
      }
    }

    if (!dispatched) {
      vulkan::dispatch_mul_mat_vec_nc_op(
          values_t,
          scores,
          out_work,
          vulkan::StaticShaderId::mul_mat_vec_nc_f16_f32,
          command_buffer,
          s,
          {
              matrix_layout.ne00,
              matrix_layout.ne01,
              static_cast<uint32_t>(values_t.strides(-2)),
              static_cast<uint32_t>(values_t.strides(-3)),
              static_cast<uint32_t>(scores.strides(-3)),
              gqa_ratio,
              vec_layout.ne02,
              0,
              0,
              static_cast<uint32_t>(values_t.strides(-4)),
              static_cast<uint32_t>(scores.strides(-4)),
              static_cast<uint32_t>(out_work.strides(-4)),
              0,
          },
          {matrix_layout.ne03, matrix_layout.ne01, vec_layout.ne02});
    }
    vulkan::end_command_recording(s.index);
  } catch (const std::runtime_error&) {
    return false;
  }

  if (needs_out_copy) {
    // Unlike ggml, MLX matmul can target a non-row-contiguous destination view,
    // so this path stages into scratch and copies back only when needed.
    vulkan::mark_scratch_array_written(s, kMatvecScoresVOutScratchLane);
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool ensure_vulkan_buffer(array& arr, Stream s) {
  if (has_vulkan_buffer(arr)) {
    return true;
  }

  if (arr.has_primitive()) {
    arr = contiguous_copy_gpu(arr, s);
    return has_vulkan_buffer(arr);
  }

  if (!arr.has_primitive()) {
    arr.wait();
  }

  if (has_vulkan_buffer(arr)) {
    return true;
  }

  auto data = arr.data_shared_ptr();
  if (data == nullptr || !vulkan::is_vulkan_buffer(data->buffer)) {
    return false;
  }

  arr = contiguous_copy_gpu(arr, s);
  return has_vulkan_buffer(arr);
}

void zero_initialize_output(array& out, Stream s) {
  if (out.size() == 0) {
    return;
  }

  auto zero_contiguous = [&](array& target) {
    target.set_data(allocator::malloc(target.nbytes()));
    if (target.nbytes() == 0) {
      return;
    }
    auto* target_buf =
        static_cast<vulkan::VulkanBuffer*>(target.buffer().ptr());
    auto cmd_buffer = vulkan::begin_command_recording(s.index);
    cmd_buffer.fillBuffer(target_buf->buffer, 0, target.nbytes(), 0);
    vulkan::end_command_recording(s.index);
  };

  if (is_row_contiguous_zero_offset(out)) {
    zero_contiguous(out);
    return;
  }

  array scratch = vulkan::acquire_scratch_array(
      s, kMatmulZeroScratchLane, out.shape(), out.dtype());
  zero_contiguous(scratch);
  vulkan::mark_scratch_array_written(s, kMatmulZeroScratchLane);
  copy_gpu(scratch, out, CopyType::General, s);
}

bool try_eval_matvec_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) {
    return false;
  }
  if (a.dtype() != b.dtype() || out.dtype() != a.dtype() ||
      !is_supported_matmul_dtype(a.dtype())) {
    return false;
  }

  bool a_is_vec = (a.shape(0) >= 1 && a.shape(0) <= kMaxMulMatVecCols);
  bool b_is_vec = (b.shape(1) == 1);
  bool is_matvec = a_is_vec || b_is_vec;

  if (!is_matvec) {
    return false;
  }

  array vec = a;
  array matrix = b;
  if (a_is_vec && b_is_vec) {
    if (a.shape(1) != b.shape(0)) {
      return false;
    }
    vec = a;
    matrix = swapaxes_in_eval(b, -1, -2);
  } else if (a_is_vec) {
    if (a.shape(1) != b.shape(0)) {
      return false;
    }
    vec = a;
    matrix = swapaxes_in_eval(b, -1, -2);
  } else {
    if (b.shape(1) != a.shape(0)) {
      return false;
    }
    vec = swapaxes_in_eval(b, -1, -2);
    matrix = a;
  }

  if (out.shape(0) != vec.shape(0) || out.shape(1) != matrix.shape(0)) {
    return false;
  }

  if (vec.shape(1) == 0) {
    zero_initialize_output(out, s);
    return true;
  }

  if (!is_row_contiguous_zero_offset(vec)) {
    vec = contiguous_copy_gpu(vec, s);
  }
  if (!is_row_contiguous_zero_offset(vec)) {
    return false;
  }

  if (!is_row_contiguous_zero_offset(matrix)) {
    matrix = contiguous_copy_gpu(matrix, s);
  }
  if (!is_row_contiguous_zero_offset(matrix)) {
    return false;
  }

  if (!ensure_vulkan_buffer(vec, s) || !ensure_vulkan_buffer(matrix, s)) {
    return false;
  }

  if (matrix.dtype() == bfloat16 &&
      !vulkan::VulkanContext::get().shader_bfloat16_supported()) {
    matrix = cast_to_float16_scratch(matrix, s, kMatvecMatrixCastScratchLane);
  }

  Dtype vec_shader_dtype = vec.dtype();
  if (vec_shader_dtype == bfloat16) {
    vec = cast_to_float16_scratch(vec, s, kMatvecVectorCastScratchLane);
    vec_shader_dtype = float16;
  }

  auto shader_candidates =
      matvec_shader_candidates(matrix.dtype(), vec_shader_dtype);
  if (shader_candidates.empty()) {
    return false;
  }

  const bool needs_out_copy =
      out.dtype() != float32 || !is_row_contiguous_zero_offset(out);
  array out_work = out;
  if (needs_out_copy) {
    out_work = vulkan::acquire_scratch_array(
        s, kMatvecOutScratchLane, out.shape(), float32);
  } else {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  }
  if (out_work.size() == 0) {
    if (needs_out_copy) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  for (const auto shader_id : shader_candidates) {
    bool dispatched = false;
    try {
      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_mul_mat_vec_op(
          matrix, vec, out_work, shader_id, command_buffer, s);
      insert_compute_barrier(command_buffer);
      vulkan::end_command_recording(s.index);
      vulkan::retain_array_for_stream(s, matrix);
      vulkan::retain_array_for_stream(s, vec);
      vulkan::retain_array_for_stream(s, out_work);
      dispatched = true;
    } catch (const std::runtime_error& e) {
      if (matmul_debug_enabled()) {
        std::cerr << "[vulkan::matvec] shader="
                  << vulkan::static_shader_name(shader_id)
                  << " failed: " << e.what() << "\n";
      }
    }
    if (dispatched) {
      if (needs_out_copy) {
        vulkan::mark_scratch_array_written(s, kMatvecOutScratchLane);
        copy_gpu(out_work, out, CopyType::General, s);
      }
      return true;
    }
  }
  return false;
}

bool try_eval_mul_mm_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (!mul_mm_enabled()) {
    return false;
  }
  if (inputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (a.ndim() < 2 || b.ndim() < 2 || out.ndim() < 2) {
    return false;
  }
  if (a.shape(-1) != b.shape(-2) || out.shape(-2) != a.shape(-2) ||
      out.shape(-1) != b.shape(-1)) {
    return false;
  }
  if (!is_supported_matmul_dtype(a.dtype()) ||
      !is_supported_matmul_dtype(b.dtype()) ||
      !is_supported_matmul_dtype(out.dtype())) {
    return false;
  }

  if (try_eval_scores_v_matvec_vulkan(inputs, out, s)) {
    return true;
  }

  if (a.ndim() != b.ndim() || a.ndim() != out.ndim()) {
    return false;
  }

  auto materialize_broadcast_input = [&](array& in) {
    Shape target_shape = out.shape();
    target_shape[target_shape.size() - 2] = in.shape(in.ndim() - 2);
    target_shape[target_shape.size() - 1] = in.shape(in.ndim() - 1);
    if (in.shape() == target_shape) {
      return true;
    }
    if (broadcast_shapes(in.shape(), target_shape) != target_shape) {
      return false;
    }
    if (!ensure_vulkan_buffer(in, s)) {
      return false;
    }
    array view(target_shape, in.dtype(), nullptr, {});
    broadcast(in, view);
    in = view;
    return true;
  };

  const bool can_keep_a_broadcast_view = a.ndim() == 4 &&
      a.shape(-2) == out.shape(-2) && a.shape(-1) == b.shape(-2) &&
      (a.shape(-3) == 1 || a.shape(-3) == out.shape(-3)) &&
      (a.shape(-4) == 1 || a.shape(-4) == out.shape(-4));

  if ((!can_keep_a_broadcast_view && !materialize_broadcast_input(a)) ||
      !materialize_broadcast_input(b)) {
    return false;
  }

  if (a.dtype() != b.dtype()) {
    return false;
  }

  if (a.shape(-1) == 0) {
    zero_initialize_output(out, s);
    return true;
  }

  if (a.dtype() == bfloat16 &&
      !vulkan::VulkanContext::get().shader_bfloat16_supported()) {
    a = cast_to_float16_scratch(a, s, kMulMmACastScratchLane);
    b = cast_to_float16_scratch(b, s, kMulMmBCastScratchLane);
  }

  // Keep BF16 inputs in BF16 and dispatch matmul_bf16* directly.
  // This matches ggml's BF16xBF16 path and avoids costly staging casts.

  if (!is_row_contiguous_zero_offset(a)) {
    if (!ensure_vulkan_buffer(a, s)) {
      return false;
    }
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_row_contiguous_zero_offset(a)) {
    return false;
  }

  array b_t = swapaxes_in_eval(b, -1, -2);
  if (!is_row_contiguous_zero_offset(b_t)) {
    if (!ensure_vulkan_buffer(b_t, s)) {
      return false;
    }
    b_t = contiguous_copy_gpu(b_t, s);
  }
  if (!is_row_contiguous_zero_offset(b_t)) {
    return false;
  }

  if (!ensure_vulkan_buffer(a, s) || !ensure_vulkan_buffer(b_t, s)) {
    if (matmul_debug_enabled()) {
      std::cerr << "[vulkan::mul_mm] missing buffer"
                << " a=" << has_vulkan_buffer(a) << " a_status=" << a.status()
                << " a_has_primitive=" << a.has_primitive()
                << " b_t=" << has_vulkan_buffer(b_t)
                << " b_t_status=" << b_t.status()
                << " b_t_has_primitive=" << b_t.has_primitive() << "\n";
    }
    return false;
  }

  const uint32_t m = static_cast<uint32_t>(out.shape(-2));
  const uint32_t n = static_cast<uint32_t>(out.shape(-1));
  const uint32_t k = static_cast<uint32_t>(a.shape(-1));
  auto tuning = select_matmul_dispatch_tuning(a.dtype(), m, n, k);
  if (a.dtype() == float16 && out.dtype() == float16 && k >= 128u &&
      (m <= 32u || n <= 32u)) {
    tuning.prefer_fp32_accum = true;
  }

  const uint32_t batch_stride_a =
      static_cast<uint32_t>(a.shape(-2) * a.shape(-1));
  const uint32_t batch_stride_b =
      static_cast<uint32_t>(b_t.shape(-2) * b_t.shape(-1));

  uint64_t num_batches_u64 = 1;
  for (int i = 0; i < static_cast<int>(out.ndim()) - 2; ++i) {
    num_batches_u64 *= static_cast<uint64_t>(out.shape(i));
  }
  if (num_batches_u64 > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  const uint32_t num_batches = static_cast<uint32_t>(num_batches_u64);
  const uint32_t split_k =
      choose_split_k(k, num_batches, tuning.split_k_threshold);

  const uint32_t tile_m = tuning.specialization_constants.size() > 1
      ? tuning.specialization_constants[1]
      : 32u;
  const uint32_t tile_n = tuning.specialization_constants.size() > 2
      ? tuning.specialization_constants[2]
      : 32u;
  const uint32_t blocks_m = (m + tile_m - 1) / tile_m;
  const uint32_t blocks_n = (n + tile_n - 1) / tile_n;

  auto try_dispatch = [&](const std::vector<vulkan::StaticShaderId>&
                              shader_candidates,
                          array& out_work,
                          bool needs_out_copy) {
    if (shader_candidates.empty()) {
      return false;
    }

    const uint32_t batch_stride_d =
        static_cast<uint32_t>(out_work.shape(-2) * out_work.shape(-1));

    if (num_batches_u64 == 0) {
      if (needs_out_copy) {
        vulkan::mark_scratch_array_written(s, kMulMmOutScratchLane);
        copy_gpu(out_work, out, CopyType::General, s);
      }
      return true;
    }

    std::optional<array> split_k_out;
    if (split_k > 1u) {
      const uint64_t split_k_elems = static_cast<uint64_t>(batch_stride_d) *
          static_cast<uint64_t>(num_batches) * static_cast<uint64_t>(split_k);
      if (split_k_elems >
          static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return false;
      }
      split_k_out = vulkan::acquire_scratch_array(
          s,
          kMulMmSplitKScratchLane,
          {
              static_cast<int>(split_k * num_batches),
              static_cast<int>(m),
              static_cast<int>(n),
          },
          float32);
      if (!ensure_vulkan_buffer(*split_k_out, s)) {
        return false;
      }
    }

    const uint32_t a_heads =
        static_cast<uint32_t>(a.ndim() >= 3 ? a.shape(-3) : 1);
    const uint32_t out_heads =
        static_cast<uint32_t>(out_work.ndim() >= 3 ? out_work.shape(-3) : 1);
    const uint32_t a_batches_outer =
        static_cast<uint32_t>(a.ndim() >= 4 ? a.shape(-4) : 1);
    const uint32_t out_batches_outer =
        static_cast<uint32_t>(out_work.ndim() >= 4 ? out_work.shape(-4) : 1);
    if (a_heads == 0 || out_heads == 0 || a_batches_outer == 0 ||
        out_batches_outer == 0 || (out_heads % a_heads) != 0 ||
        (out_batches_outer % a_batches_outer) != 0) {
      return false;
    }

    vulkan::MatmulPushConstants push_constants{};
    push_constants.M = m;
    push_constants.N = n;
    push_constants.K = k;
    push_constants.stride_a = static_cast<uint32_t>(a.strides(-2));
    push_constants.stride_b = static_cast<uint32_t>(b_t.strides(-2));
    push_constants.stride_d = static_cast<uint32_t>(out_work.strides(-2));
    push_constants.batch_stride_a = batch_stride_a;
    push_constants.batch_stride_b = batch_stride_b;
    push_constants.batch_stride_d = batch_stride_d;
    push_constants.num_batches = num_batches;
    push_constants.k_split = round_up_div(k, split_k);
    push_constants.ne02 = a_heads;
    push_constants.ne12 = out_heads;
    push_constants.broadcast2 = out_heads / a_heads;
    push_constants.broadcast3 = out_batches_outer / a_batches_outer;
    push_constants.padded_N = n;

    if (matmul_debug_enabled()) {
      std::cerr << "[vulkan::mul_mm] a_shape=" << a.shape()
                << " a_strides=" << a.strides() << " b_t_shape=" << b_t.shape()
                << " b_t_strides=" << b_t.strides()
                << " out_shape=" << out_work.shape()
                << " out_strides=" << out_work.strides() << " pc(M,N,K)=" << m
                << "," << n << "," << k
                << " stride(a,b,d)=" << push_constants.stride_a << ","
                << push_constants.stride_b << "," << push_constants.stride_d
                << " batch=" << num_batches
                << " family=" << matmul_family_name(tuning.family)
                << " aligned=" << tuning.aligned
                << " subgroup=" << tuning.specialization_constants[10]
                << " fp32_accum=" << tuning.prefer_fp32_accum
                << " split_k_threshold=" << tuning.split_k_threshold
                << " split_k=" << split_k << "\n";
    }

    for (const auto shader_id : shader_candidates) {
      bool dispatched = true;
      bool should_recover_stream = false;
      try {
        auto command_buffer = vulkan::begin_command_recording(s.index);
        for (uint32_t base_z = 0; base_z < num_batches; base_z += kMaxGridZ) {
          const uint32_t chunk_z = std::min(kMaxGridZ, num_batches - base_z);
          push_constants.base_work_group_z = base_z;
          const std::array<uint32_t, 3> grid = {
              blocks_m * split_k,
              blocks_n,
              chunk_z,
          };

          vulkan::dispatch_mul_mm_op(
              a,
              b_t,
              split_k_out.has_value() ? *split_k_out : out_work,
              shader_id,
              command_buffer,
              s,
              push_constants,
              grid,
              tuning.specialization_constants);
        }

        if (split_k_out.has_value()) {
          insert_compute_barrier(command_buffer);
          vulkan::MatmulSplitKReducePushConstants reduce_push_constants{};
          reduce_push_constants.ne = batch_stride_d * num_batches;
          reduce_push_constants.k_num = split_k;
          vulkan::dispatch_matmul_split_k_reduce_op(
              *split_k_out,
              out_work,
              split_k_reduce_shader_id(out_work.dtype()),
              command_buffer,
              s,
              reduce_push_constants,
              {
                  round_up_div(batch_stride_d * num_batches, 256u * 4u),
                  1u,
                  1u,
              },
              {});
          vulkan::mark_scratch_array_written(s, kMulMmSplitKScratchLane);
        }

        vulkan::end_command_recording(s.index);
      } catch (const std::runtime_error& e) {
        if (matmul_debug_enabled()) {
          std::cerr << "[vulkan::mul_mm] shader="
                    << vulkan::static_shader_name(shader_id)
                    << " failed: " << e.what() << "\n";
        }
        const std::string message = e.what();
        if (message.find("[vulkan::submit_commands]") != std::string::npos ||
            message.find("VkResult=") != std::string::npos) {
          should_recover_stream = true;
        }
        dispatched = false;
      }

      if (dispatched) {
        if (matmul_debug_enabled()) {
          std::cerr << "[vulkan::mul_mm] shader="
                    << vulkan::static_shader_name(shader_id) << " dispatched\n";
        }
        try {
          if (needs_out_copy) {
            vulkan::mark_scratch_array_written(s, kMulMmOutScratchLane);
            if (matmul_debug_enabled()) {
              std::cerr << "[vulkan::mul_mm] begin output copy\n";
            }
            copy_gpu(out_work, out, CopyType::General, s);
            if (matmul_debug_enabled()) {
              std::cerr << "[vulkan::mul_mm] end output copy\n";
            }
          }
          return true;
        } catch (const std::runtime_error& e) {
          if (matmul_debug_enabled()) {
            std::cerr << "[vulkan::mul_mm] output copy failed: " << e.what()
                      << "\n";
          }
          disable_mul_mm_runtime(e.what());
          return false;
        }
      }

      if (should_recover_stream) {
        disable_mul_mm_runtime("submit failure");
        return false;
      }
    }

    return false;
  };

  const bool can_direct_write =
      split_k == 1u && is_row_contiguous_zero_offset(out);
  if (can_direct_write) {
    auto direct_candidates = mul_mm_direct_shader_candidates(
        a.dtype(), out.dtype(), tuning.prefer_fp32_accum);
    array out_direct = out;
    out_direct.set_data(allocator::malloc(out_direct.nbytes()));
    if (ensure_vulkan_buffer(out_direct, s) &&
        try_dispatch(direct_candidates, out_direct, false)) {
      return true;
    }
  }

  auto shader_candidates =
      mul_mm_shader_candidates(a.dtype(), tuning.prefer_fp32_accum);
  array out_work = out;
  const bool can_direct_split_k_reduce =
      split_k > 1u && is_row_contiguous_zero_offset(out_work);
  const bool needs_out_copy = !is_row_contiguous_zero_offset(out_work) ||
      (split_k == 1u && out.dtype() != float32);
  if (needs_out_copy) {
    out_work = vulkan::acquire_scratch_array(
        s, kMulMmOutScratchLane, out.shape(), float32);
  } else {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  }
  if (!ensure_vulkan_buffer(out_work, s)) {
    return false;
  }
  return try_dispatch(shader_candidates, out_work, needs_out_copy);
}

bool fits_u32_complex_matmul_layout(const array& arr) {
  if (arr.offset() / size_of(arr.dtype()) >
      std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  for (auto dim : arr.shape()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  for (auto stride : arr.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

std::string build_complex_matmul_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, false);
  os << "layout(push_constant) uniform Params {\n";
  os << "  uint total;\n";
  os << "  uint M;\n";
  os << "  uint N;\n";
  os << "  uint K;\n";
  os << "  uint a_offset;\n";
  os << "  uint b_offset;\n";
  os << "  uint out_offset;\n";
  os << "  uint a_stride0;\n";
  os << "  uint a_stride1;\n";
  os << "  uint b_stride0;\n";
  os << "  uint b_stride1;\n";
  os << "  uint out_stride0;\n";
  os << "  uint out_stride1;\n";
  os << "  uint batch_count;\n";
  os << "  uint a_batch_stride;\n";
  os << "  uint b_batch_stride;\n";
  os << "  uint out_batch_stride;\n";
  os << "} p;\n";
  os << "layout(set = 0, binding = 0) readonly buffer A { vec2 data[]; } a;\n";
  os << "layout(set = 0, binding = 1) readonly buffer B { vec2 data[]; } b;\n";
  os << "layout(set = 0, binding = 2) buffer Out { vec2 data[]; } out_buf;\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= p.total) return;\n";
  os << "  uint batch_elems = p.M * p.N;\n";
  os << "  uint batch = idx / batch_elems;\n";
  os << "  uint elem = idx - batch * batch_elems;\n";
  os << "  uint row = elem / p.N;\n";
  os << "  uint col = elem - row * p.N;\n";
  os << "  uint a_base = p.a_offset + (p.batch_count == 1 ? 0 : batch) * p.a_batch_stride;\n";
  os << "  uint b_base = p.b_offset + (p.batch_count == 1 ? 0 : batch) * p.b_batch_stride;\n";
  os << "  uint out_base = p.out_offset + batch * p.out_batch_stride;\n";
  os << "  vec2 acc = vec2(0.0, 0.0);\n";
  os << "  for (uint kk = 0; kk < p.K; ++kk) {\n";
  os << "    vec2 av = a.data[a_base + row * p.a_stride0 + kk * p.a_stride1];\n";
  os << "    vec2 bv = b.data[b_base + kk * p.b_stride0 + col * p.b_stride1];\n";
  os << "    acc += vec2(av.x * bv.x - av.y * bv.y, av.x * bv.y + av.y * bv.x);\n";
  os << "  }\n";
  os << "  out_buf.data[out_base + row * p.out_stride0 + col * p.out_stride1] = acc;\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_matmul_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (a.dtype() != complex64 || b.dtype() != complex64 ||
      out.dtype() != complex64 || a.ndim() != b.ndim() ||
      a.ndim() != out.ndim() || (a.ndim() != 2 && a.ndim() != 3)) {
    return false;
  }
  if (a.shape(-1) != b.shape(-2) || out.shape(-2) != a.shape(-2) ||
      out.shape(-1) != b.shape(-1)) {
    return false;
  }
  if (a.ndim() == 3 &&
      !((a.shape(0) == out.shape(0) || a.shape(0) == 1) &&
        (b.shape(0) == out.shape(0) || b.shape(0) == 1))) {
    return false;
  }
  if (out.size() > std::numeric_limits<uint32_t>::max() ||
      !fits_u32_complex_matmul_layout(a) ||
      !fits_u32_complex_matmul_layout(b) ||
      !fits_u32_complex_matmul_layout(out)) {
    return false;
  }

  if (a.shape(-1) == 0) {
    zero_initialize_output(out, s);
    return true;
  }

  if (!ensure_vulkan_buffer(a, s) || !ensure_vulkan_buffer(b, s)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (!ensure_vulkan_buffer(out, s)) {
    return false;
  }

  struct PushConstants {
    uint32_t total;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t out_offset;
    uint32_t a_stride0;
    uint32_t a_stride1;
    uint32_t b_stride0;
    uint32_t b_stride1;
    uint32_t out_stride0;
    uint32_t out_stride1;
    uint32_t batch_count;
    uint32_t a_batch_stride;
    uint32_t b_batch_stride;
    uint32_t out_batch_stride;
  } pc{};
  pc.total = static_cast<uint32_t>(out.size());
  pc.M = static_cast<uint32_t>(out.shape(-2));
  pc.N = static_cast<uint32_t>(out.shape(-1));
  pc.K = static_cast<uint32_t>(a.shape(-1));
  pc.a_offset = static_cast<uint32_t>(a.offset() / size_of(a.dtype()));
  pc.b_offset = static_cast<uint32_t>(b.offset() / size_of(b.dtype()));
  pc.out_offset = static_cast<uint32_t>(out.offset() / size_of(out.dtype()));
  pc.a_stride0 = static_cast<uint32_t>(a.strides(-2));
  pc.a_stride1 = static_cast<uint32_t>(a.strides(-1));
  pc.b_stride0 = static_cast<uint32_t>(b.strides(-2));
  pc.b_stride1 = static_cast<uint32_t>(b.strides(-1));
  pc.out_stride0 = static_cast<uint32_t>(out.strides(-2));
  pc.out_stride1 = static_cast<uint32_t>(out.strides(-1));
  pc.batch_count = static_cast<uint32_t>(out.ndim() == 3 ? out.shape(0) : 1);
  pc.a_batch_stride = static_cast<uint32_t>(
      a.ndim() == 3 && a.shape(0) != 1 ? a.strides(0) : 0);
  pc.b_batch_stride = static_cast<uint32_t>(
      b.ndim() == 3 && b.shape(0) != 1 ? b.strides(0) : 0);
  pc.out_batch_stride =
      static_cast<uint32_t>(out.ndim() == 3 ? out.strides(0) : 0);

  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out, 2}};
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_matmul_c64_2d",
      build_complex_matmul_shader(),
      3,
      arrays,
      sizeof(PushConstants),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(PushConstants),
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool fits_u32_gather_mm_layout(const array& arr) {
  if (arr.offset() / size_of(arr.dtype()) >
      std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  for (auto dim : arr.shape()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  for (auto stride : arr.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

std::string
build_gather_mm_f32_shader(int index_ndim, int a_batch_ndim, int b_batch_ndim) {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(float32, false);
  os << "#define INDEX_NDIM " << index_ndim << "\n";
  os << "#define A_BATCH_NDIM " << a_batch_ndim << "\n";
  os << "#define B_BATCH_NDIM " << b_batch_ndim << "\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "  uint total;\n";
  os << "  uint M;\n";
  os << "  uint N;\n";
  os << "  uint K;\n";
  os << "  uint a_offset;\n";
  os << "  uint b_offset;\n";
  os << "  uint lhs_offset;\n";
  os << "  uint rhs_offset;\n";
  os << "  uint out_offset;\n";
  os << "  uint index_shape[4];\n";
  os << "  uint lhs_strides[4];\n";
  os << "  uint rhs_strides[4];\n";
  os << "  uint out_batch_strides[4];\n";
  os << "  uint a_batch_shape[4];\n";
  os << "  uint a_batch_strides[4];\n";
  os << "  uint b_batch_shape[4];\n";
  os << "  uint b_batch_strides[4];\n";
  os << "  uint a_row_stride;\n";
  os << "  uint a_col_stride;\n";
  os << "  uint b_row_stride;\n";
  os << "  uint b_col_stride;\n";
  os << "  uint out_row_stride;\n";
  os << "  uint out_col_stride;\n";
  os << "} p;\n";
  os << "layout(set = 0, binding = 0) readonly buffer A { float data[]; } a;\n";
  os << "layout(set = 0, binding = 1) readonly buffer B { float data[]; } b;\n";
  os << "layout(set = 0, binding = 2) readonly buffer LHS { uint data[]; } lhs;\n";
  os << "layout(set = 0, binding = 3) readonly buffer RHS { uint data[]; } rhs;\n";
  os << "layout(set = 0, binding = 4) buffer Out { float data[]; } out_buf;\n";
  os << "uint batch_offset(uint batch, uint shape[4], uint strides[4], int ndim) {\n";
  os << "  uint offset = 0;\n";
  os << "  uint rem = batch;\n";
  os << "  for (int d = ndim - 1; d >= 0; --d) {\n";
  os << "    uint coord = rem % shape[d];\n";
  os << "    rem /= shape[d];\n";
  os << "    offset += coord * strides[d];\n";
  os << "  }\n";
  os << "  return offset;\n";
  os << "}\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= p.total) return;\n";
  os << "  uint matrix_size = p.M * p.N;\n";
  os << "  uint batch = idx / matrix_size;\n";
  os << "  uint within = idx - batch * matrix_size;\n";
  os << "  uint row = within / p.N;\n";
  os << "  uint col = within - row * p.N;\n";
  os << "  uint lhs_pos = p.lhs_offset + batch_offset(batch, p.index_shape, p.lhs_strides, INDEX_NDIM);\n";
  os << "  uint rhs_pos = p.rhs_offset + batch_offset(batch, p.index_shape, p.rhs_strides, INDEX_NDIM);\n";
  os << "  uint lhs_batch = lhs.data[lhs_pos];\n";
  os << "  uint rhs_batch = rhs.data[rhs_pos];\n";
  os << "  uint a_base = p.a_offset + batch_offset(lhs_batch, p.a_batch_shape, p.a_batch_strides, A_BATCH_NDIM);\n";
  os << "  uint b_base = p.b_offset + batch_offset(rhs_batch, p.b_batch_shape, p.b_batch_strides, B_BATCH_NDIM);\n";
  os << "  float acc = 0.0;\n";
  os << "  for (uint kk = 0; kk < p.K; ++kk) {\n";
  os << "    float av = a.data[a_base + row * p.a_row_stride + kk * p.a_col_stride];\n";
  os << "    float bv = b.data[b_base + kk * p.b_row_stride + col * p.b_col_stride];\n";
  os << "    acc = fma(av, bv, acc);\n";
  os << "  }\n";
  os << "  uint out_base = p.out_offset + batch_offset(batch, p.index_shape, p.out_batch_strides, INDEX_NDIM);\n";
  os << "  out_buf.data[out_base + row * p.out_row_stride + col * p.out_col_stride] = acc;\n";
  os << "}\n";
  return os.str();
}

bool try_eval_gather_mm_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 4) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  array lhs_indices = inputs[2];
  array rhs_indices = inputs[3];
  if (a.dtype() != float32 || b.dtype() != float32 || out.dtype() != float32 ||
      lhs_indices.dtype() != uint32 || rhs_indices.dtype() != uint32 ||
      a.ndim() < 2 || b.ndim() < 2 || out.ndim() < 2 || a.ndim() > 4 ||
      b.ndim() > 4 || out.ndim() > 4 || lhs_indices.ndim() > 2 ||
      rhs_indices.ndim() > 2) {
    return false;
  }
  if (a.shape(-1) != b.shape(-2) || out.shape(-2) != a.shape(-2) ||
      out.shape(-1) != b.shape(-1) ||
      lhs_indices.shape() != rhs_indices.shape()) {
    return false;
  }
  if (out.ndim() != lhs_indices.ndim() + 2) {
    return false;
  }
  for (int i = 0; i < lhs_indices.ndim(); ++i) {
    if (out.shape(i) != lhs_indices.shape(i)) {
      return false;
    }
  }
  if (out.size() > std::numeric_limits<uint32_t>::max() ||
      !fits_u32_gather_mm_layout(a) || !fits_u32_gather_mm_layout(b) ||
      !fits_u32_gather_mm_layout(lhs_indices) ||
      !fits_u32_gather_mm_layout(rhs_indices) ||
      !fits_u32_gather_mm_layout(out)) {
    return false;
  }

  if (a.size() == 0 || b.size() == 0 || a.shape(-1) == 0) {
    zero_initialize_output(out, s);
    return true;
  }

  if (!ensure_vulkan_buffer(a, s) || !ensure_vulkan_buffer(b, s) ||
      !ensure_vulkan_buffer(lhs_indices, s) ||
      !ensure_vulkan_buffer(rhs_indices, s)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (!ensure_vulkan_buffer(out, s)) {
    return false;
  }

  struct PushConstants {
    uint32_t total;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t lhs_offset;
    uint32_t rhs_offset;
    uint32_t out_offset;
    uint32_t index_shape[4];
    uint32_t lhs_strides[4];
    uint32_t rhs_strides[4];
    uint32_t out_batch_strides[4];
    uint32_t a_batch_shape[4];
    uint32_t a_batch_strides[4];
    uint32_t b_batch_shape[4];
    uint32_t b_batch_strides[4];
    uint32_t a_row_stride;
    uint32_t a_col_stride;
    uint32_t b_row_stride;
    uint32_t b_col_stride;
    uint32_t out_row_stride;
    uint32_t out_col_stride;
  } pc{};

  const int index_ndim = lhs_indices.ndim();
  const int a_batch_ndim = a.ndim() - 2;
  const int b_batch_ndim = b.ndim() - 2;
  pc.total = static_cast<uint32_t>(out.size());
  pc.M = static_cast<uint32_t>(out.shape(-2));
  pc.N = static_cast<uint32_t>(out.shape(-1));
  pc.K = static_cast<uint32_t>(a.shape(-1));
  pc.a_offset = static_cast<uint32_t>(a.offset() / size_of(a.dtype()));
  pc.b_offset = static_cast<uint32_t>(b.offset() / size_of(b.dtype()));
  pc.lhs_offset = static_cast<uint32_t>(
      lhs_indices.offset() / size_of(lhs_indices.dtype()));
  pc.rhs_offset = static_cast<uint32_t>(
      rhs_indices.offset() / size_of(rhs_indices.dtype()));
  pc.out_offset = static_cast<uint32_t>(out.offset() / size_of(out.dtype()));
  for (int i = 0; i < index_ndim; ++i) {
    pc.index_shape[i] = static_cast<uint32_t>(lhs_indices.shape(i));
    pc.lhs_strides[i] = static_cast<uint32_t>(lhs_indices.strides(i));
    pc.rhs_strides[i] = static_cast<uint32_t>(rhs_indices.strides(i));
    pc.out_batch_strides[i] = static_cast<uint32_t>(out.strides(i));
  }
  for (int i = 0; i < a_batch_ndim; ++i) {
    pc.a_batch_shape[i] = static_cast<uint32_t>(a.shape(i));
    pc.a_batch_strides[i] = static_cast<uint32_t>(a.strides(i));
  }
  for (int i = 0; i < b_batch_ndim; ++i) {
    pc.b_batch_shape[i] = static_cast<uint32_t>(b.shape(i));
    pc.b_batch_strides[i] = static_cast<uint32_t>(b.strides(i));
  }
  pc.a_row_stride = static_cast<uint32_t>(a.strides(-2));
  pc.a_col_stride = static_cast<uint32_t>(a.strides(-1));
  pc.b_row_stride = static_cast<uint32_t>(b.strides(-2));
  pc.b_col_stride = static_cast<uint32_t>(b.strides(-1));
  pc.out_row_stride = static_cast<uint32_t>(out.strides(-2));
  pc.out_col_stride = static_cast<uint32_t>(out.strides(-1));

  vulkan::DynamicArrayRef arrays[] = {
      {&a, 0}, {&b, 1}, {&lhs_indices, 2}, {&rhs_indices, 3}, {&out, 4}};
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_gather_mm_f32_" + std::to_string(index_ndim) + "d_" +
          std::to_string(a_batch_ndim) + "ad_" + std::to_string(b_batch_ndim) +
          "bd",
      build_gather_mm_f32_shader(index_ndim, a_batch_ndim, b_batch_ndim),
      5,
      arrays,
      sizeof(PushConstants),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(PushConstants),
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

std::string build_segmented_mm_f32_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(float32, false);
  os << "layout(push_constant) uniform Params {\n";
  os << "  uint total;\n";
  os << "  uint M;\n";
  os << "  uint N;\n";
  os << "  uint a_offset;\n";
  os << "  uint b_offset;\n";
  os << "  uint segments_offset;\n";
  os << "  uint out_offset;\n";
  os << "  uint a_row_stride;\n";
  os << "  uint a_col_stride;\n";
  os << "  uint b_row_stride;\n";
  os << "  uint b_col_stride;\n";
  os << "  uint segments_row_stride;\n";
  os << "  uint segments_col_stride;\n";
  os << "  uint out_segment_stride;\n";
  os << "  uint out_row_stride;\n";
  os << "  uint out_col_stride;\n";
  os << "} p;\n";
  os << "layout(set = 0, binding = 0) readonly buffer A { float data[]; } a;\n";
  os << "layout(set = 0, binding = 1) readonly buffer B { float data[]; } b;\n";
  os << "layout(set = 0, binding = 2) readonly buffer Segments { uint data[]; } segments;\n";
  os << "layout(set = 0, binding = 3) buffer Out { float data[]; } out_buf;\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= p.total) return;\n";
  os << "  uint matrix_size = p.M * p.N;\n";
  os << "  uint segment = idx / matrix_size;\n";
  os << "  uint within = idx - segment * matrix_size;\n";
  os << "  uint row = within / p.N;\n";
  os << "  uint col = within - row * p.N;\n";
  os << "  uint seg_base = p.segments_offset + segment * p.segments_row_stride;\n";
  os << "  uint k_start = segments.data[seg_base];\n";
  os << "  uint k_end = segments.data[seg_base + p.segments_col_stride];\n";
  os << "  float acc = 0.0;\n";
  os << "  if (k_end > k_start) {\n";
  os << "    for (uint kk = k_start; kk < k_end; ++kk) {\n";
  os << "      float av = a.data[p.a_offset + row * p.a_row_stride + kk * p.a_col_stride];\n";
  os << "      float bv = b.data[p.b_offset + kk * p.b_row_stride + col * p.b_col_stride];\n";
  os << "      acc = fma(av, bv, acc);\n";
  os << "    }\n";
  os << "  }\n";
  os << "  out_buf.data[p.out_offset + segment * p.out_segment_stride + row * p.out_row_stride + col * p.out_col_stride] = acc;\n";
  os << "}\n";
  return os.str();
}

bool try_eval_segmented_mm_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 3) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  array segments = inputs[2];
  if (a.dtype() != float32 || b.dtype() != float32 || out.dtype() != float32 ||
      segments.dtype() != uint32 || a.ndim() != 2 || b.ndim() != 2 ||
      segments.ndim() != 2 || out.ndim() != 3 || segments.shape(-1) != 2) {
    return false;
  }
  if (a.shape(-1) != b.shape(-2) || out.shape(-3) != segments.shape(0) ||
      out.shape(-2) != a.shape(-2) || out.shape(-1) != b.shape(-1)) {
    return false;
  }
  if (out.size() > std::numeric_limits<uint32_t>::max() ||
      !fits_u32_gather_mm_layout(a) || !fits_u32_gather_mm_layout(b) ||
      !fits_u32_gather_mm_layout(segments) || !fits_u32_gather_mm_layout(out)) {
    return false;
  }

  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return true;
  }
  if (a.size() == 0 || b.size() == 0) {
    zero_initialize_output(out, s);
    return true;
  }

  if (!ensure_vulkan_buffer(a, s) || !ensure_vulkan_buffer(b, s) ||
      !ensure_vulkan_buffer(segments, s)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (!ensure_vulkan_buffer(out, s)) {
    return false;
  }

  struct PushConstants {
    uint32_t total;
    uint32_t M;
    uint32_t N;
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t segments_offset;
    uint32_t out_offset;
    uint32_t a_row_stride;
    uint32_t a_col_stride;
    uint32_t b_row_stride;
    uint32_t b_col_stride;
    uint32_t segments_row_stride;
    uint32_t segments_col_stride;
    uint32_t out_segment_stride;
    uint32_t out_row_stride;
    uint32_t out_col_stride;
  } pc{};

  pc.total = static_cast<uint32_t>(out.size());
  pc.M = static_cast<uint32_t>(a.shape(-2));
  pc.N = static_cast<uint32_t>(b.shape(-1));
  pc.a_offset = static_cast<uint32_t>(a.offset() / size_of(a.dtype()));
  pc.b_offset = static_cast<uint32_t>(b.offset() / size_of(b.dtype()));
  pc.segments_offset =
      static_cast<uint32_t>(segments.offset() / size_of(segments.dtype()));
  pc.out_offset = static_cast<uint32_t>(out.offset() / size_of(out.dtype()));
  pc.a_row_stride = static_cast<uint32_t>(a.strides(-2));
  pc.a_col_stride = static_cast<uint32_t>(a.strides(-1));
  pc.b_row_stride = static_cast<uint32_t>(b.strides(-2));
  pc.b_col_stride = static_cast<uint32_t>(b.strides(-1));
  pc.segments_row_stride = static_cast<uint32_t>(segments.strides(-2));
  pc.segments_col_stride = static_cast<uint32_t>(segments.strides(-1));
  pc.out_segment_stride = static_cast<uint32_t>(out.strides(-3));
  pc.out_row_stride = static_cast<uint32_t>(out.strides(-2));
  pc.out_col_stride = static_cast<uint32_t>(out.strides(-1));

  vulkan::DynamicArrayRef arrays[] = {
      {&a, 0}, {&b, 1}, {&segments, 2}, {&out, 3}};
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_segmented_mm_f32_2d",
      build_segmented_mm_f32_shader(),
      4,
      arrays,
      sizeof(PushConstants),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(PushConstants),
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool try_eval_matmul_vulkan_impl(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    bool* used_matvec_fast_path) {
  if (used_matvec_fast_path) {
    *used_matvec_fast_path = false;
  }
  if (inputs.size() == 2 && (inputs[0].size() == 0 || inputs[1].size() == 0)) {
    zero_initialize_output(out, s);
    return true;
  }
  if (try_eval_complex_matmul_vulkan(inputs, out, s)) {
    return true;
  }
  if (matvec_enabled() && try_eval_matvec_vulkan(inputs, out, s)) {
    if (used_matvec_fast_path) {
      *used_matvec_fast_path = true;
    }
    return true;
  }
  return try_eval_mul_mm_vulkan(inputs, out, s);
}

bool is_supported_addmm_epilogue_dtype(Dtype dtype) {
  return dtype == float32 || dtype == float16 || dtype == bfloat16 ||
      dtype == complex64;
}

const char* addmm_epilogue_dtype_name(Dtype dtype) {
  switch (dtype) {
    case float32:
      return "f32";
    case float16:
      return "f16";
    case bfloat16:
      return "bf16";
    case complex64:
      return "c64";
    default:
      return "unsupported";
  }
}

std::string emit_addmm_bf16_helpers() {
  return R"(
uint fp32_to_bf16(float f) {
  uint u = floatBitsToUint(f);
  u = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
  return u;
}

float bf16_to_fp32(uint u) {
  return uintBitsToFloat(u << 16);
}

)";
}

std::string addmm_load_expr(
    const std::string& buffer,
    const std::string& index,
    Dtype dtype) {
  if (dtype == bfloat16) {
    return "bf16_to_fp32(uint(" + buffer + ".data[" + index + "]))";
  }
  if (dtype == float16) {
    return "float(" + buffer + ".data[" + index + "])";
  }
  return buffer + ".data[" + index + "]";
}

std::string addmm_store_expr(const std::string& value, Dtype dtype) {
  if (dtype == bfloat16) {
    return "uint16_t(fp32_to_bf16(" + value + "))";
  }
  if (dtype == float16) {
    return "float16_t(" + value + ")";
  }
  return value;
}

std::string build_addmm_epilogue_shader(Dtype dtype, int ndim) {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(dtype, false);
  if (dtype == bfloat16) {
    os << emit_addmm_bf16_helpers();
  }
  os << "#define NDIM " << ndim << "\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "  uint total;\n";
  os << "  uint mm_offset;\n";
  os << "  uint c_offset;\n";
  os << "  uint out_offset;\n";
  os << "  float alpha;\n";
  os << "  float beta;\n";
  os << "  uint shape[4];\n";
  os << "  uint c_strides[4];\n";
  os << "} p;\n";
  const auto storage_type = vulkan::dtype_to_glsl_storage_type(dtype);
  os << "layout(set = 0, binding = 0) readonly buffer MM { " << storage_type
     << " data[]; } mm;\n";
  os << "layout(set = 0, binding = 1) readonly buffer C { " << storage_type
     << " data[]; } c;\n";
  os << "layout(set = 0, binding = 2) buffer Out { " << storage_type
     << " data[]; } out_buf;\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= p.total) return;\n";
  os << "  uint c_idx = p.c_offset;\n";
  os << "  uint rem = idx;\n";
  os << "  for (int d = NDIM - 1; d >= 0; --d) {\n";
  os << "    uint coord = rem % p.shape[d];\n";
  os << "    rem /= p.shape[d];\n";
  os << "    c_idx += coord * p.c_strides[d];\n";
  os << "  }\n";
  os << "  " << (dtype == complex64 ? "vec2" : "float") << " value = p.alpha * "
     << addmm_load_expr("mm", "idx + p.mm_offset", dtype) << " + p.beta * "
     << addmm_load_expr("c", "c_idx", dtype) << ";\n";
  os << "  out_buf.data[idx + p.out_offset] = "
     << addmm_store_expr("value", dtype) << ";\n";
  os << "}\n";
  return os.str();
}

bool try_eval_addmm_epilogue_vulkan(
    const array& mm,
    array c,
    array& out,
    float alpha,
    float beta,
    Stream s) {
  const Dtype dtype = out.dtype();
  if (!is_supported_addmm_epilogue_dtype(dtype) || mm.dtype() != dtype ||
      c.dtype() != dtype || out.ndim() > 4 || mm.shape() != out.shape()) {
    return false;
  }

  if (c.shape() != out.shape()) {
    if (broadcast_shapes(c.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer(c, s)) {
      return false;
    }
    array view(out.shape(), c.dtype(), nullptr, {});
    broadcast(c, view);
    c = view;
  }
  if (!ensure_vulkan_buffer(c, s)) {
    return false;
  }
  for (auto stride : c.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }

  const auto total = static_cast<uint64_t>(out.size());
  const auto mm_offset =
      static_cast<uint64_t>(mm.offset() / size_of(mm.dtype()));
  const auto c_offset = static_cast<uint64_t>(c.offset() / size_of(c.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out.offset() / size_of(out.dtype()));
  if (total > std::numeric_limits<uint32_t>::max() ||
      mm_offset > std::numeric_limits<uint32_t>::max() ||
      c_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (!ensure_vulkan_buffer(out, s)) {
    return false;
  }

  struct PushConstants {
    uint32_t total;
    uint32_t mm_offset;
    uint32_t c_offset;
    uint32_t out_offset;
    float alpha;
    float beta;
    uint32_t shape[4];
    uint32_t c_strides[4];
  } pc{};
  pc.total = static_cast<uint32_t>(total);
  pc.mm_offset = static_cast<uint32_t>(mm_offset);
  pc.c_offset = static_cast<uint32_t>(c_offset);
  pc.out_offset = static_cast<uint32_t>(out_offset);
  pc.alpha = alpha;
  pc.beta = beta;
  for (int i = 0; i < out.ndim(); ++i) {
    pc.shape[i] = static_cast<uint32_t>(out.shape(i));
    pc.c_strides[i] = static_cast<uint32_t>(c.strides(i));
  }

  vulkan::DynamicArrayRef arrays[] = {{&mm, 0}, {&c, 1}, {&out, 2}};
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      std::string("dynamic_addmm_epilogue_") +
          addmm_epilogue_dtype_name(dtype) + "_" + std::to_string(out.ndim()) +
          "d",
      build_addmm_epilogue_shader(dtype, static_cast<int>(out.ndim())),
      3,
      arrays,
      sizeof(PushConstants),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(PushConstants),
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool is_supported_block_mask_dtype(Dtype dtype) {
  return dtype == float32 || dtype == float16 || dtype == bfloat16;
}

bool is_supported_block_mask_dtype(Dtype data_dtype, Dtype mask_dtype) {
  return is_supported_block_mask_dtype(data_dtype) &&
      (mask_dtype == bool_ || mask_dtype == data_dtype);
}

const char* block_mask_dtype_name(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return "bool";
    case float32:
      return "f32";
    case float16:
      return "f16";
    case bfloat16:
      return "bf16";
    default:
      return "unsupported";
  }
}

std::string build_block_mask_copy_shader(
    Dtype data_dtype,
    Dtype mask_dtype,
    int ndim,
    int mask_ndim) {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(data_dtype, mask_dtype, false);
  if (data_dtype == bfloat16 || mask_dtype == bfloat16) {
    os << emit_addmm_bf16_helpers();
  }
  os << "#define NDIM " << ndim << "\n";
  os << "#define MASK_NDIM " << mask_ndim << "\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "  uint total;\n";
  os << "  uint block_size;\n";
  os << "  uint rows;\n";
  os << "  uint cols;\n";
  os << "  uint src_offset;\n";
  os << "  uint dst_offset;\n";
  os << "  uint mask_offset;\n";
  os << "  uint shape[4];\n";
  os << "  uint src_strides[4];\n";
  os << "  uint mask_shape[4];\n";
  os << "  uint mask_strides[4];\n";
  os << "} p;\n";
  const auto data_storage = vulkan::dtype_to_glsl_storage_type(data_dtype);
  const auto mask_storage = vulkan::dtype_to_glsl_storage_type(mask_dtype);
  os << "layout(set = 0, binding = 0) readonly buffer Src { " << data_storage
     << " data[]; } src;\n";
  os << "layout(set = 0, binding = 1) readonly buffer Mask { " << mask_storage
     << " data[]; } mask;\n";
  os << "layout(set = 0, binding = 2) buffer Dst { " << data_storage
     << " data[]; } dst;\n";
  os << "float load_data(uint idx) { return "
     << addmm_load_expr("src", "idx", data_dtype) << "; }\n";
  if (mask_dtype != bool_) {
    os << "float load_mask(uint idx) { return "
       << addmm_load_expr("mask", "idx", mask_dtype) << "; }\n";
  }
  os << "void store_data(uint idx, float value) { dst.data[idx] = "
     << addmm_store_expr("value", data_dtype) << "; }\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= p.total) return;\n";
  os << "  uint src_idx = p.src_offset;\n";
  os << "  uint rem = idx;\n";
  os << "  for (int d = NDIM - 1; d >= 0; --d) {\n";
  os << "    uint coord = rem % p.shape[d];\n";
  os << "    rem /= p.shape[d];\n";
  os << "    src_idx += coord * p.src_strides[d];\n";
  os << "  }\n";
  os << "  uint mat_size = p.rows * p.cols;\n";
  os << "  uint batch = idx / mat_size;\n";
  os << "  uint within = idx - batch * mat_size;\n";
  os << "  uint row = within / p.cols;\n";
  os << "  uint col = within - row * p.cols;\n";
  os << "  uint mask_rows = p.mask_shape[MASK_NDIM - 2];\n";
  os << "  uint mask_cols = p.mask_shape[MASK_NDIM - 1];\n";
  os << "  uint mask_linear = batch * mask_rows * mask_cols + "
        "(row / p.block_size) * mask_cols + (col / p.block_size);\n";
  os << "  uint mask_idx = p.mask_offset;\n";
  os << "  rem = mask_linear;\n";
  os << "  for (int d = MASK_NDIM - 1; d >= 0; --d) {\n";
  os << "    uint coord = rem % p.mask_shape[d];\n";
  os << "    rem /= p.mask_shape[d];\n";
  os << "    mask_idx += coord * p.mask_strides[d];\n";
  os << "  }\n";
  if (mask_dtype == bool_) {
    os << "  float value = mask.data[mask_idx] != uint8_t(0) ? load_data(src_idx) : 0.0;\n";
  } else {
    os << "  float value = load_data(src_idx) * load_mask(mask_idx);\n";
  }
  os << "  store_data(p.dst_offset + idx, value);\n";
  os << "}\n";
  return os.str();
}

bool fits_u32_offset(const array& arr) {
  return static_cast<uint64_t>(arr.offset() / size_of(arr.dtype())) <=
      std::numeric_limits<uint32_t>::max();
}

bool fill_shape_and_strides(
    const array& arr,
    uint32_t shape[4],
    uint32_t strides[4]) {
  if (arr.ndim() > 4) {
    return false;
  }
  for (int i = 0; i < arr.ndim(); ++i) {
    if (arr.shape(i) < 0 ||
        arr.shape(i) > std::numeric_limits<uint32_t>::max() ||
        arr.strides(i) < 0 ||
        arr.strides(i) > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    shape[i] = static_cast<uint32_t>(arr.shape(i));
    strides[i] = static_cast<uint32_t>(arr.strides(i));
  }
  return true;
}

bool dispatch_block_mask_copy_vulkan(
    array src,
    const array& mask,
    array& dst,
    int block_size,
    int64_t rows,
    int64_t cols,
    Stream s) {
  if (src.shape() != dst.shape() || src.dtype() != dst.dtype() ||
      src.ndim() != mask.ndim() || src.ndim() > 4 || src.size() == 0 ||
      rows <= 0 || cols <= 0 ||
      !is_supported_block_mask_dtype(src.dtype(), mask.dtype())) {
    return false;
  }
  if (src.shape(-2) != rows || src.shape(-1) != cols) {
    return false;
  }
  const int64_t mask_rows = (rows + block_size - 1) / block_size;
  const int64_t mask_cols = (cols + block_size - 1) / block_size;
  if (mask.shape(-2) != mask_rows || mask.shape(-1) != mask_cols) {
    return false;
  }
  if (src.size() > std::numeric_limits<uint32_t>::max() ||
      !fits_u32_offset(src) || !fits_u32_offset(mask) ||
      !fits_u32_offset(dst)) {
    return false;
  }

  if (!ensure_vulkan_buffer(src, s)) {
    return false;
  }
  array mask_arr = mask;
  if (!ensure_vulkan_buffer(mask_arr, s)) {
    return false;
  }
  if (!ensure_vulkan_buffer(dst, s)) {
    return false;
  }

  struct PushConstants {
    uint32_t total;
    uint32_t block_size;
    uint32_t rows;
    uint32_t cols;
    uint32_t src_offset;
    uint32_t dst_offset;
    uint32_t mask_offset;
    uint32_t shape[4];
    uint32_t src_strides[4];
    uint32_t mask_shape[4];
    uint32_t mask_strides[4];
  } pc{};
  pc.total = static_cast<uint32_t>(src.size());
  pc.block_size = static_cast<uint32_t>(block_size);
  pc.rows = static_cast<uint32_t>(rows);
  pc.cols = static_cast<uint32_t>(cols);
  pc.src_offset = static_cast<uint32_t>(src.offset() / size_of(src.dtype()));
  pc.dst_offset = static_cast<uint32_t>(dst.offset() / size_of(dst.dtype()));
  pc.mask_offset =
      static_cast<uint32_t>(mask_arr.offset() / size_of(mask_arr.dtype()));
  if (!fill_shape_and_strides(src, pc.shape, pc.src_strides) ||
      !fill_shape_and_strides(mask_arr, pc.mask_shape, pc.mask_strides)) {
    return false;
  }

  vulkan::DynamicArrayRef arrays[] = {{&src, 0}, {&mask_arr, 1}, {&dst, 2}};
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      std::string("dynamic_block_mask_copy_") +
          block_mask_dtype_name(src.dtype()) + "_" +
          block_mask_dtype_name(mask_arr.dtype()) + "_" +
          std::to_string(src.ndim()) + "d",
      build_block_mask_copy_shader(
          src.dtype(), mask_arr.dtype(), src.ndim(), mask_arr.ndim()),
      3,
      arrays,
      sizeof(PushConstants),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(PushConstants),
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

std::optional<array> copy_with_block_mask_vulkan(
    array src,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    Stream s,
    const char* scratch_lane) {
  array dst =
      vulkan::acquire_scratch_array(s, scratch_lane, src.shape(), src.dtype());
  if (!dispatch_block_mask_copy_vulkan(
          src, mask, dst, block_size, rows, cols, s)) {
    return std::nullopt;
  }
  vulkan::mark_scratch_array_written(s, scratch_lane);
  return dst;
}

bool apply_block_mask_vulkan(
    array& data,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    Stream s) {
  return dispatch_block_mask_copy_vulkan(
      data, mask, data, block_size, rows, cols, s);
}

} // namespace

bool try_eval_matmul_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  return try_eval_matmul_vulkan_impl(inputs, out, s, nullptr);
}

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  bool used_matvec_fast_path = false;
  if (try_eval_matmul_vulkan_impl(
          inputs, out, stream(), &used_matvec_fast_path)) {
    log_matmul_path(inputs, used_matvec_fast_path ? "matvec" : "mul_mm");
    return;
  }
  throw std::runtime_error(
      "Matmul operation failed on Vulkan (unsupported dtype or layout).");
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }

  auto& s = stream();
  array mm(out.shape(), out.dtype(), nullptr, {});
  if (!try_eval_matmul_vulkan_impl({inputs[0], inputs[1]}, mm, s, nullptr)) {
    throw std::runtime_error(
        "AddMM operation failed on Vulkan (unsupported matmul input).");
  }
  mm.set_status(array::Status::evaluated);

  if (!try_eval_addmm_epilogue_vulkan(mm, inputs[2], out, alpha_, beta_, s)) {
    throw std::runtime_error(
        "AddMM operation failed on Vulkan (unsupported epilogue input).");
  }
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_gather_mm_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "GatherMM operation failed on Vulkan (unsupported dtype or layout).");
  }
}

void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_segmented_mm_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "SegmentedMM operation failed on Vulkan (unsupported dtype or layout).");
  }
}

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[BlockMaskedMM] Does not yet support non-floating point types.");
  }

  auto& s = stream();
  array a = inputs[0];
  array b = inputs[1];

  if (a.size() == 0 || b.size() == 0) {
    zero_initialize_output(out, s);
    return;
  }

  const int64_t M = a.shape(-2);
  const int64_t N = b.shape(-1);
  const int64_t K = a.shape(-1);
  if (M == 0 || N == 0) {
    return;
  }
  if (K == 0) {
    zero_initialize_output(out, s);
    return;
  }

  const bool has_op_mask = inputs.size() > 3;
  const bool has_out_mask = inputs.size() == 3 || inputs.size() == 5;

  if (has_op_mask) {
    const array& lhs_mask = inputs[inputs.size() - 2];
    const array& rhs_mask = inputs[inputs.size() - 1];
    auto masked_a = copy_with_block_mask_vulkan(
        a, lhs_mask, block_size_, M, K, s, kBlockMaskedMMLhsScratchLane);
    auto masked_b = copy_with_block_mask_vulkan(
        b, rhs_mask, block_size_, K, N, s, kBlockMaskedMMRhsScratchLane);
    if (!masked_a.has_value() || !masked_b.has_value()) {
      throw std::runtime_error(
          "BlockMaskedMM operation failed on Vulkan (unsupported operand mask input).");
    }
    a = *masked_a;
    b = *masked_b;
  }

  if (!try_eval_matmul_vulkan_impl({a, b}, out, s, nullptr)) {
    throw std::runtime_error(
        "BlockMaskedMM operation failed on Vulkan (unsupported matmul input).");
  }

  if (has_out_mask &&
      !apply_block_mask_vulkan(out, inputs[2], block_size_, M, N, s)) {
    throw std::runtime_error(
        "BlockMaskedMM operation failed on Vulkan (unsupported output mask input).");
  }
}

} // namespace mlx::core
