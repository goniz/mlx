// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/matmul.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/quantized.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

using namespace vk;

namespace fast {

namespace {

constexpr char kFlashAttnMaskOptScratchLane[] = "flash_attn.mask_opt";
constexpr char kFlashAttnSplitKScratchLane[] = "flash_attn.split_k";
constexpr char kFlashAttnOutScratchLane[] = "flash_attn.out_storage";
constexpr char kFlashAttnCausalMaskScratchLane[] = "flash_attn.causal_mask";
constexpr char kFlashAttnQCastScratchLane[] = "flash_attn.q_cast";
constexpr char kFlashAttnKCastScratchLane[] = "flash_attn.k_cast";
constexpr char kFlashAttnVCastScratchLane[] = "flash_attn.v_cast";
constexpr char kFlashAttnMaskCastScratchLane[] = "flash_attn.mask_cast";
constexpr char kFlashAttnSinksCastScratchLane[] = "flash_attn.sinks_cast";
constexpr uint32_t kFlashAttnSinkEnableBit = 1u << 24;

bool needs_host_row_contiguous(const array& arr) {
  return !arr.flags().row_contiguous || arr.offset() != 0;
}

array ensure_host_readable_row_contiguous(array arr, Stream s) {
  if (needs_host_row_contiguous(arr) || arr.has_primitive()) {
    arr = contiguous_copy_gpu(arr, s);
  }
  arr.wait();
  return arr;
}

template <typename T>
T bit_cast_scalar(const auto& value) {
  return std::bit_cast<T>(value);
}

uint8_t to_fp8_e4m3_scalar(float x) {
  constexpr uint32_t fp8_max = 543u << 21;
  constexpr uint32_t denorm_mask = 141u << 23;
  uint32_t f_bits = bit_cast_scalar<uint32_t>(x);
  uint32_t sign = f_bits & 0x80000000u;
  f_bits ^= sign;

  uint32_t f_bits_low = bit_cast_scalar<uint32_t>(
      bit_cast_scalar<float>(f_bits) + bit_cast_scalar<float>(denorm_mask));
  uint8_t result_low = static_cast<uint8_t>(f_bits_low - denorm_mask);

  uint8_t mant_odd = static_cast<uint8_t>((f_bits >> 20) & 1u);
  uint32_t f_bits_high = f_bits + (((7u - 127u) << 23) + 0x7FFFFu);
  f_bits_high += mant_odd;
  uint8_t result_high = static_cast<uint8_t>(f_bits_high >> 20);

  uint8_t result = f_bits < (121u << 23) ? result_low : result_high;
  if (f_bits >= fp8_max) {
    result = 0x7E;
  }
  return result | static_cast<uint8_t>(sign >> 24);
}

float from_fp8_e4m3_scalar(uint8_t x) {
  uint16_t bits = static_cast<uint16_t>(x & 127u) << 7;
  float16_t half = bit_cast_scalar<float16_t>(bits);
  float out = static_cast<float>(half) * 256.0f;
  return (x & 128u) ? -out : out;
}

std::string build_to_fp8_f32_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(float32, uint8, false);
  os << R"(
layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;
layout(set = 0, binding = 0) readonly buffer Input { float data[]; } in_buf;
)";
  os << vulkan::storage_buffer_layout_for_dtype(uint8, 1)
     << " buffer Output { uint8_t data[]; } out_buf;\n";
  os << R"(

uint8_t to_fp8_e4m3(float x) {
  uint f_bits = floatBitsToUint(x);
  uint sign = f_bits & 0x80000000u;
  f_bits ^= sign;

  uint f_bits_low = floatBitsToUint(uintBitsToFloat(f_bits) + uintBitsToFloat(141u << 23));
  uint result_low = f_bits_low - (141u << 23);

  uint mant_odd = (f_bits >> 20) & 1u;
  uint f_bits_high = f_bits + (((7u - 127u) << 23) + 0x7FFFFu);
  f_bits_high += mant_odd;
  uint result_high = f_bits_high >> 20;

  uint result = f_bits < (121u << 23) ? result_low : result_high;
  if (f_bits >= (543u << 21)) {
    result = 0x7Eu;
  }
  return uint8_t(result | (sign >> 24));
}

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;
  out_buf.data[idx + pc.out_offset] = to_fp8_e4m3(in_buf.data[idx + pc.in_offset]);
}
)";
  return os.str();
}

std::string build_from_fp8_shader(Dtype out_dtype) {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(uint8, out_dtype, false);
  if (out_dtype == bfloat16) {
    os << "uint fp32_to_bf16(float f) {\n";
    os << "  uint u = floatBitsToUint(f);\n";
    os << "  u = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;\n";
    os << "  return u;\n";
    os << "}\n\n";
  }
  const char* out_type = out_dtype == float16 ? "float16_t"
      : out_dtype == bfloat16                 ? "uint16_t"
                                              : "float";
  os << R"(
layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;
)";
  os << vulkan::storage_buffer_layout_for_dtype(uint8, 0)
     << " readonly buffer Input { uint8_t data[]; } in_buf;\n";
  os << "layout(set = 0, binding = 1) buffer Output { " << out_type
     << " data[]; } out_buf;\n";
  os << R"(

float from_fp8_e4m3(uint8_t x8) {
  uint x = uint(x8);
  uint exponent = (x >> 3u) & 15u;
  uint mantissa = x & 7u;
  float outv = 0.0;
  if (exponent == 0u) {
    outv = float(mantissa) * 0.001953125;
  } else {
    outv = exp2(float(int(exponent) - 7)) *
        (1.0 + float(mantissa) * 0.125);
  }
  return (x & 128u) != 0u ? -outv : outv;
}

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;
)";
  if (out_dtype == bfloat16) {
    os << "  out_buf.data[idx + pc.out_offset] = uint16_t(fp32_to_bf16(from_fp8_e4m3(in_buf.data[idx + pc.in_offset])));\n";
  } else {
    os << "  out_buf.data[idx + pc.out_offset] = " << out_type
       << "(from_fp8_e4m3(in_buf.data[idx + pc.in_offset]));\n";
  }
  os << R"(
}
)";
  return os.str();
}

bool try_convert_fp8_f32_gpu(
    const array& input,
    array& out,
    bool to_fp8,
    Stream s) {
  if (to_fp8) {
    if (input.dtype() != float32 || out.dtype() != uint8) {
      return false;
    }
  } else if (
      input.dtype() != uint8 ||
      (out.dtype() != float32 && out.dtype() != float16 &&
       out.dtype() != bfloat16)) {
    return false;
  }

  array in = input;
  if (!in.flags().row_contiguous || in.offset() != 0 || in.has_primitive()) {
    in = contiguous_copy_gpu(in, s);
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  const auto in_offset =
      static_cast<uint64_t>(in.offset() / size_of(in.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out.offset() / size_of(out.dtype()));
  const auto total = static_cast<uint64_t>(out.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  vulkan::DynamicArrayRef arrays[] = {{&in, 0}, {&out, 1}};
  struct PushConstants {
    uint32_t in_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(in_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };

  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      to_fp8 ? "dynamic_to_fp8_f32"
             : (out.dtype() == bfloat16      ? "dynamic_from_fp8_bf16"
                    : out.dtype() == float16 ? "dynamic_from_fp8_f16"
                                             : "dynamic_from_fp8_f32"),
      to_fp8 ? build_to_fp8_f32_shader() : build_from_fp8_shader(out.dtype()),
      2,
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
  vkCmdDispatch(
      dispatch.command_buffer, (pc.total_elements + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

array apply_diag_mask_inf_vulkan(const array& scores, int n_past, Stream s);
array ensure_sdpa_rowwise_layout(array arr, Stream s);
void eval_sdpa_binary_add_vulkan(array lhs, array rhs, array& out, Stream s);

struct FlashAttentionCausalMaskCacheEntry {
  Shape shape;
  bool valid{false};
};

std::mutex flash_attention_causal_mask_cache_mutex;
std::unordered_map<int, FlashAttentionCausalMaskCacheEntry>
    flash_attention_causal_mask_cache;

std::optional<bool> experimental_flash_attention_override() {
  static const std::optional<bool> enabled = []() -> std::optional<bool> {
    if (const char* env = std::getenv("MLX_VULKAN_EXPERIMENTAL_FLASH_ATTN");
        env != nullptr) {
      return std::string(env) != "0";
    }
    return std::nullopt;
  }();
  return enabled;
}

bool flash_attention_enabled_for(const array& k, const array& v) {
  if (auto override = experimental_flash_attention_override();
      override.has_value()) {
    return *override;
  }
  const bool supported_k_dtype = k.dtype() == float16 || k.dtype() == bfloat16;
  const bool supported_v_dtype = v.dtype() == float16 || v.dtype() == bfloat16;
  return supported_k_dtype && supported_v_dtype;
}

bool trace_flash_attention_debug_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_TRACE_FLASH_ATTN");
        env != nullptr) {
      return std::string(env) != "0";
    }
    return false;
  }();
  return enabled;
}

void trace_flash_attention_debug(const std::string& msg) {
  if (!trace_flash_attention_debug_enabled()) {
    return;
  }
  std::cerr << "[vulkan-fa-debug] " << msg << std::endl;
}

void trace_flash_attention_array(const char* label, const array& arr) {
  if (!trace_flash_attention_debug_enabled()) {
    return;
  }
  std::ostringstream oss;
  oss << label << " shape=" << arr.shape() << " dtype=" << arr.dtype()
      << " status=" << arr.status() << " has_primitive=" << arr.has_primitive()
      << " row_contig=" << arr.flags().row_contiguous
      << " offset=" << arr.offset();
  trace_flash_attention_debug(oss.str());
}

std::optional<vulkan::StaticShaderId> rms_norm_shader_id(Dtype dtype) {
  switch (dtype) {
    case float32:
      return vulkan::StaticShaderId::rms_norm_f32;
    case float16:
      return vulkan::StaticShaderId::rms_norm_f16;
    case bfloat16:
      return vulkan::StaticShaderId::rms_norm_bf16;
    default:
      return std::nullopt;
  }
}

void begin_tracked_manual_op(
    Stream s,
    const char* name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  vulkan::record_primitive_for_stream(s, name);
  vulkan::begin_primitive_tracking(s, inputs, outputs);
}

void end_tracked_manual_op(
    Stream s,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  vulkan::end_primitive_tracking(s, inputs, outputs);
}

array cast_flash_attention_kv_to_f16(
    const array& x,
    const char* scratch_lane,
    Stream s) {
  auto data = x.data_shared_ptr();
  if (x.dtype() == float16 && x.offset() == 0 && x.flags().row_contiguous &&
      x.strides().back() == 1 && data != nullptr &&
      data->buffer.ptr() != nullptr) {
    return x;
  }
  array out =
      vulkan::acquire_scratch_array(s, scratch_lane, x.shape(), float16);
  copy_gpu(x, out, CopyType::General, s);
  vulkan::mark_scratch_array_written(s, scratch_lane);
  return out;
}

array cast_flash_attention_mask_to_f16(const array& x, Stream s) {
  auto data = x.data_shared_ptr();
  if (x.dtype() == float16 && x.offset() == 0 && x.strides().back() == 1 &&
      data != nullptr && data->buffer.ptr() != nullptr) {
    return x;
  }
  array out = vulkan::acquire_scratch_array(
      s, kFlashAttnMaskCastScratchLane, x.shape(), float16);
  copy_gpu(x, out, CopyType::General, s);
  vulkan::mark_scratch_array_written(s, kFlashAttnMaskCastScratchLane);
  return out;
}

array cast_flash_attention_sinks_to_f32(const array& x, Stream s) {
  auto data = x.data_shared_ptr();
  if (x.dtype() == float32 && x.offset() == 0 && x.flags().row_contiguous &&
      x.strides().back() == 1 && data != nullptr &&
      data->buffer.ptr() != nullptr) {
    return x;
  }
  array out = vulkan::acquire_scratch_array(
      s, kFlashAttnSinksCastScratchLane, x.shape(), float32);
  copy_gpu(x, out, CopyType::General, s);
  vulkan::mark_scratch_array_written(s, kFlashAttnSinksCastScratchLane);
  return out;
}

array cast_flash_attention_q_to_f32(const array& x, Stream s) {
  auto data = x.data_shared_ptr();
  if (x.dtype() == float32 && x.offset() == 0 && x.flags().row_contiguous &&
      x.strides().back() == 1 && data != nullptr &&
      data->buffer.ptr() != nullptr) {
    return x;
  }
  array out = vulkan::acquire_scratch_array(
      s, kFlashAttnQCastScratchLane, x.shape(), float32);
  copy_gpu(x, out, CopyType::General, s);
  vulkan::mark_scratch_array_written(s, kFlashAttnQCastScratchLane);
  return out;
}

struct FlashAttentionTuningParams {
  enum class Path {
    Scalar,
    CoopMat1,
  };

  Path path;
  uint32_t workgroup_size;
  uint32_t block_rows;
  uint32_t block_cols;
  uint32_t d_split;
  uint32_t row_split;
  uint32_t subgroup_size;
  uint32_t shmem_staging;
  bool disable_subgroups;
  uint32_t limit_occupancy_shmem;
};

vulkan::StaticShaderId flash_attention_main_shader(
    FlashAttentionTuningParams::Path path,
    bool use_native_bf16_kv) {
  const bool kv_bf16 = use_native_bf16_kv;
  if (const char* env = std::getenv("MLX_VULKAN_FLASH_ATTN_SHADER");
      env != nullptr) {
    const std::string value(env);
    if (value == "cm1") {
      if (kv_bf16) {
        return vulkan::StaticShaderId::flash_attn_f32_f16_bf16;
      }
      return vulkan::StaticShaderId::flash_attn_f32_f16_f16_cm1;
    }
    if (value == "fp32") {
      if (kv_bf16) {
        return vulkan::StaticShaderId::flash_attn_f32_f16_bf16_fp32;
      }
      return vulkan::StaticShaderId::flash_attn_f32_f16_f16_fp32;
    }
    if (value == "f16acc") {
      if (kv_bf16) {
        return vulkan::StaticShaderId::flash_attn_f32_f16_bf16_f16acc;
      }
      return path == FlashAttentionTuningParams::Path::CoopMat1
          ? vulkan::StaticShaderId::flash_attn_f32_f16_f16_f16acc_cm1
          : vulkan::StaticShaderId::flash_attn_f32_f16_f16_f16acc;
    }
  }
  if (kv_bf16) {
    if (vulkan::VulkanContext::get().shader_float16_supported()) {
      return vulkan::StaticShaderId::flash_attn_f32_f16_bf16;
    }
    return vulkan::StaticShaderId::flash_attn_f32_f16_bf16_fp32;
  }
  if (path == FlashAttentionTuningParams::Path::CoopMat1) {
    return vulkan::StaticShaderId::flash_attn_f32_f16_f16_cm1;
  }
  if (vulkan::VulkanContext::get().shader_float16_supported()) {
    return vulkan::StaticShaderId::flash_attn_f32_f16_f16;
  }
  return vulkan::StaticShaderId::flash_attn_f32_f16_f16_fp32;
}

struct FlashAttentionExecutionPlan {
  FlashAttentionTuningParams tuning;
  bool aligned;
  bool use_mask_opt;
  uint32_t gqa_ratio;
  uint32_t n_rows;
  uint32_t workgroups_y;
  uint32_t split_kv;
  uint32_t split_k;
};

constexpr uint32_t kVendorIdAmd = 0x1002u;
constexpr uint32_t kVendorIdIntel = 0x8086u;
constexpr uint32_t kVendorIdNvidia = 0x10DEu;

uint32_t lowest_set_bit(uint32_t value) {
  return value & (~value + 1u);
}

std::pair<uint32_t, uint32_t> vulkan_device_vendor_and_subgroup_size() {
  vk::PhysicalDeviceProperties props =
      vulkan::VulkanContext::get().physical_device().getProperties();

  VkPhysicalDeviceSubgroupProperties subgroup_props{};
  subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  PhysicalDeviceProperties2 props2{};
  props2.pNext = &subgroup_props;
  vulkan::VulkanContext::get().physical_device().getProperties2(&props2);

  return {props.vendorID, subgroup_props.subgroupSize};
}

uint32_t max_compute_shared_memory_size() {
  vk::PhysicalDeviceProperties props =
      vulkan::VulkanContext::get().physical_device().getProperties();
  return props.limits.maxComputeSharedMemorySize;
}

bool flash_attention_scalar_shmem_supported(
    const FlashAttentionTuningParams& params,
    uint32_t hsk,
    uint32_t hsv) {
  const uint32_t float_type_size =
      vulkan::VulkanContext::get().shader_float16_supported() ? 2u : 4u;
  const uint32_t tmpsh = params.workgroup_size * sizeof(float);
  const uint32_t tmpshv4 = params.workgroup_size * 4u * float_type_size;
  const uint32_t masksh =
      params.block_cols * (params.block_rows + 1u) * float_type_size;
  const uint32_t qf =
      params.block_rows * (hsk / 4u + 1u) * 4u * float_type_size;
  const uint32_t d = std::max(hsk, hsv);
  const uint32_t kvsh = params.shmem_staging
      ? params.block_cols * (d / 4u + 1u) * 4u * float_type_size
      : 4u * float_type_size;
  return tmpsh + tmpshv4 + masksh + qf + kvsh <=
      max_compute_shared_memory_size();
}

bool flash_attention_coopmat_shmem_supported(
    const FlashAttentionTuningParams& params,
    uint32_t hsk,
    uint32_t hsv,
    bool f32acc) {
  const uint32_t block_rows = params.block_rows;
  const uint32_t block_cols = params.block_cols;
  const uint32_t mat_block_rows = 16u;
  const uint32_t mat_block_cols = 16u;
  const uint32_t row_split = block_cols / mat_block_cols;
  const uint32_t hsk_pad = ((hsk + 15u) / 16u) * 16u;
  const uint32_t hsv_pad = ((hsv + 15u) / 16u) * 16u;
  const uint32_t acc_type_size = f32acc ? 4u : 2u;
  const uint32_t f16vec4_size = 8u;
  const uint32_t tmpsh = (block_cols / mat_block_cols) * sizeof(float);
  const uint32_t qstride = hsk_pad / 4u + 2u;
  const uint32_t qf = block_rows * qstride * f16vec4_size;
  const uint32_t psh_stride = block_rows / 4u + 2u;
  const uint32_t psh = block_cols * psh_stride * f16vec4_size;
  const uint32_t sfshstride = (hsk <= 128u) ? (block_rows + 8u) : block_rows;
  const uint32_t sfsh = block_cols * sfshstride * acc_type_size;
  const uint32_t kvshstride =
      (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : mat_block_rows) /
          4u +
      2u;
  const uint32_t vsh_stride = mat_block_cols / 4u * row_split;
  const uint32_t ksh =
      ((kvshstride >= vsh_stride) ? (block_cols * kvshstride)
                                  : (block_cols * vsh_stride)) *
      f16vec4_size;
  const uint32_t osh_stride = params.row_split * mat_block_rows / 4u;
  const uint32_t pvsh = mat_block_cols * osh_stride * f16vec4_size;
  const uint32_t slope = block_rows * acc_type_size;
  return tmpsh + qf + psh + sfsh + ksh + pvsh + slope <=
      max_compute_shared_memory_size();
}

bool intel_is_battlemage(uint32_t device_id) {
  switch (device_id) {
    case 0xE212u:
    case 0xE20Cu:
    case 0xE20Bu:
      return true;
    default:
      return false;
  }
}

FlashAttentionTuningParams get_flash_attention_tuning_params_scalar(
    uint32_t hsk,
    uint32_t hsv,
    uint32_t n_rows,
    uint32_t n_kv) {
  auto [vendor_id, device_subgroup_size] =
      vulkan_device_vendor_and_subgroup_size();
  const uint32_t subgroup_size = std::max(device_subgroup_size, 1u);
  const bool unified_memory = vulkan::VulkanContext::get().is_unified_memory();
  const uint32_t d = hsk | hsv;
  const auto architecture = vulkan::VulkanContext::get().architecture();

  FlashAttentionTuningParams result{};
  result.path = FlashAttentionTuningParams::Path::Scalar;
  result.disable_subgroups = false;

  if (vendor_id == kVendorIdIntel) {
    result.subgroup_size = 32u;
    result.disable_subgroups = true;
  } else if (
      vendor_id == kVendorIdAmd &&
      architecture == vulkan::GpuArchitecture::AmdRdna) {
    result.subgroup_size = n_rows < 4u ? 32u : subgroup_size;
  } else {
    result.subgroup_size = subgroup_size;
  }

  uint32_t row_split_max_hsk = 64u;
  if (vendor_id == kVendorIdAmd && !unified_memory) {
    row_split_max_hsk = n_rows <= 8 ? 64u : 128u;
  }
  result.row_split = (n_rows < 4 || hsk <= row_split_max_hsk) ? 1u : 4u;

  if (result.subgroup_size > 32u &&
      (n_rows < 4 || hsk < (result.row_split == 1 ? 128u : 64u))) {
    result.workgroup_size = result.subgroup_size * 2u;
  } else {
    result.workgroup_size = result.subgroup_size * 4u;
  }
  result.workgroup_size = std::clamp(result.workgroup_size, 32u, 256u);

  const bool reduce_block_rows =
      (d & 8u) != 0 || n_kv < 1024u || vendor_id == kVendorIdIntel;
  if (n_rows == 1) {
    result.block_rows = 1u;
    result.block_cols = 64u;
  } else if (result.row_split == 1u) {
    result.block_rows =
        n_rows == 2 ? 2u : ((n_rows <= 4 || reduce_block_rows) ? 4u : 8u);
    result.block_cols = (d & 8u) ? 64u : 32u;
  } else {
    result.block_rows =
        n_rows <= 4 ? 4u : ((n_rows <= 8 || reduce_block_rows) ? 8u : 16u);
    result.block_cols = (d & 8u) ? 64u : 32u;
  }

  const uint32_t d_lsb = lowest_set_bit(d);
  result.d_split = std::min(std::min(result.subgroup_size, 8u), d_lsb / 4u);
  result.shmem_staging =
      (vendor_id == kVendorIdNvidia && hsk < 256u && hsv < 256u) ? 1u : 0u;

  if (const char* env = std::getenv("MLX_VULKAN_FLASH_ATTN_OCCUPANCY_LIMIT");
      env != nullptr) {
    result.limit_occupancy_shmem = std::stoul(env);
  } else {
    result.limit_occupancy_shmem = 0u;
    if (vendor_id == kVendorIdAmd && n_rows >= 64u && hsk <= 128u) {
      // Subgroups per WG
      const uint32_t num_subgroups =
          result.workgroup_size / result.subgroup_size;
      // Approx. maximum limit 4 subgroups per SIMD via dummy shmem allocation
      const uint32_t target_subgroups_per_simd = 4;
      result.limit_occupancy_shmem =
          30 * 1024 / target_subgroups_per_simd / num_subgroups;
    }
  }

  if (!reduce_block_rows &&
      !flash_attention_scalar_shmem_supported(result, hsk, hsv)) {
    result.block_rows = std::max(result.block_rows / 2u, 1u);
  }

  return result;
}

FlashAttentionTuningParams get_flash_attention_tuning_params_coopmat1(
    uint32_t hsk,
    uint32_t hsv) {
  auto [vendor_id, device_subgroup_size] =
      vulkan_device_vendor_and_subgroup_size();
  const uint32_t subgroup_size = std::max(device_subgroup_size, 1u);
  const uint32_t d = hsk | hsv;

  FlashAttentionTuningParams result{};
  result.path = FlashAttentionTuningParams::Path::CoopMat1;
  result.block_rows = 16u;
  result.block_cols = 64u;
  result.row_split = 4u;
  result.subgroup_size = subgroup_size;
  result.disable_subgroups = false;
  result.workgroup_size = 4u * subgroup_size;
  result.d_split =
      std::min(std::min(subgroup_size, 8u), lowest_set_bit(d) / 4u);
  result.shmem_staging =
      (vendor_id == kVendorIdNvidia && hsk < 256u && hsv < 256u) ? 1u : 0u;

  if (const char* env = std::getenv("MLX_VULKAN_FLASH_ATTN_OCCUPANCY_LIMIT");
      env != nullptr) {
    result.limit_occupancy_shmem = std::stoul(env);
  } else {
    result.limit_occupancy_shmem = 0u;
  }
  return result;
}

FlashAttentionTuningParams get_flash_attention_tuning_params(
    uint32_t hsk,
    uint32_t hsv,
    uint32_t n_rows,
    uint32_t n_kv) {
  if (n_rows > 1 &&
      vulkan::VulkanContext::get().coopmat_flash_attention_f32acc_supported()) {
    auto coopmat = get_flash_attention_tuning_params_coopmat1(hsk, hsv);
    if (flash_attention_coopmat_shmem_supported(coopmat, hsk, hsv, true)) {
      return coopmat;
    }
  }
  return get_flash_attention_tuning_params_scalar(hsk, hsv, n_rows, n_kv);
}

uint32_t round_up_multiple(uint32_t value, uint32_t multiple) {
  if (multiple == 0) {
    return value;
  }
  return ((value + multiple - 1u) / multiple) * multiple;
}

FlashAttentionExecutionPlan make_flash_attention_execution_plan(
    uint32_t hsk,
    uint32_t hsv,
    uint32_t q_len,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t kv_len,
    uint32_t batch,
    bool has_mask,
    bool do_causal,
    uint32_t mask_heads,
    bool use_native_bf16_kv,
    uint32_t q_stride,
    uint32_t k_stride,
    uint32_t v_stride) {
  uint32_t gqa_ratio = 1u;
  uint32_t n_rows = q_len;
  uint32_t workgroups_y = q_heads;
  const uint32_t qk_ratio = kv_heads == 0u ? 0u : q_heads / kv_heads;

  auto tuning = use_native_bf16_kv
      ? get_flash_attention_tuning_params_scalar(hsk, hsv, n_rows, kv_len)
      : get_flash_attention_tuning_params(hsk, hsv, n_rows, kv_len);

  if (!use_native_bf16_kv && q_len <= 8u && qk_ratio > 1u &&
      qk_ratio <= tuning.block_rows && qk_ratio * kv_heads == q_heads &&
      mask_heads <= 1u) {
    gqa_ratio = qk_ratio;
    n_rows = gqa_ratio;
    workgroups_y /= gqa_ratio;
    tuning = use_native_bf16_kv
        ? get_flash_attention_tuning_params_scalar(hsk, hsv, n_rows, kv_len)
        : get_flash_attention_tuning_params(hsk, hsv, n_rows, kv_len);
  }

  const bool aligned = (kv_len % tuning.block_cols) == 0 &&
      (q_stride & 7u) == 0 && (k_stride & 7u) == 0 && (v_stride & 7u) == 0;

  FlashAttentionExecutionPlan plan{
      tuning,
      aligned,
      false,
      gqa_ratio,
      n_rows,
      workgroups_y,
      kv_len,
      1u,
  };

  if (has_mask && !do_causal) {
    plan.use_mask_opt = q_len >= 32u && kv_len >= 128u &&
        static_cast<uint64_t>(q_len) * static_cast<uint64_t>(kv_len) >= 32768u;
  }

  const uint32_t tr = (n_rows + tuning.block_rows - 1u) / tuning.block_rows;
  const uint32_t total_wgs_no_split = std::max(tr * workgroups_y * batch, 1u);
  uint32_t shader_core_count = vulkan::VulkanContext::get().shader_core_count();
  if (shader_core_count == 0u) {
    shader_core_count = 16u;
  }
  if (vulkan::VulkanContext::get().vendor_id() == kVendorIdIntel &&
      !intel_is_battlemage(vulkan::VulkanContext::get().device_id())) {
    shader_core_count *= 2u;
  }
  const uint32_t target_workgroups = shader_core_count * 2u;
  uint32_t split_k = total_wgs_no_split < target_workgroups
      ? (target_workgroups + total_wgs_no_split - 1u) / total_wgs_no_split
      : 1u;
  split_k = std::min(split_k, 8u);
  if (has_mask && split_k > 1u) {
    split_k = std::min(split_k, 4u);
  }
  if (split_k > 1u && kv_len >= 2u * tuning.block_cols) {
    uint32_t split_kv =
        round_up_multiple(std::max(1u, kv_len / split_k), tuning.block_cols);
    split_k = (kv_len + split_kv - 1u) / split_kv;
    if (split_k > 1u) {
      plan.split_kv = split_kv;
      plan.split_k = split_k;
    }
  }

  return plan;
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

array make_flash_attention_causal_mask(
    const array& q,
    const array& k,
    Stream s) {
  Shape shape = {q.shape(0), 1, q.shape(2), k.shape(2)};
  array mask = vulkan::acquire_scratch_array(
      s, kFlashAttnCausalMaskScratchLane, shape, float16);

  bool needs_refresh = true;
  {
    std::lock_guard<std::mutex> lock(flash_attention_causal_mask_cache_mutex);
    auto& entry = flash_attention_causal_mask_cache[s.index];
    needs_refresh = !entry.valid || entry.shape != shape;
    entry.shape = shape;
    entry.valid = true;
  }

  if (!needs_refresh) {
    mask.set_status(array::Status::available);
    return mask;
  }

  array mask_f16 = zeros(shape, float16, s);
  if (mask_f16.dtype() != float16) {
    throw std::runtime_error("Unexpected causal mask dtype.");
  }
  const int n_past = k.shape(2) - q.shape(2);
  mask_f16 = apply_diag_mask_inf_vulkan(mask_f16, n_past, s);
  eval(mask_f16);
  copy_gpu_inplace(mask_f16, mask, CopyType::General, s);
  mask.set_status(array::Status::evaluated);
  vulkan::mark_scratch_array_written(s, kFlashAttnCausalMaskScratchLane);
  return mask;
}

bool try_dispatch_flash_attention_native_vulkan(
    const array& q,
    const array& k,
    const array& v,
    const array* mask,
    const array* sinks,
    bool do_causal,
    array& out_storage,
    Stream s,
    bool use_native_bf16_kv,
    float scale) {
  const uint32_t batch = checked_u32_size(q.shape(0), "flash_attn batch");
  const uint32_t q_heads = checked_u32_size(q.shape(1), "flash_attn q_heads");
  const uint32_t q_len = checked_u32_size(q.shape(2), "flash_attn q_len");
  const uint32_t hsk = checked_u32_size(q.shape(3), "flash_attn hsk");
  const uint32_t kv_heads = checked_u32_size(k.shape(1), "flash_attn kv_heads");
  const uint32_t kv_len = checked_u32_size(k.shape(2), "flash_attn kv_len");
  const uint32_t hsv = checked_u32_size(v.shape(3), "flash_attn hsv");
  const bool has_mask = mask != nullptr;
  const uint32_t q_stride =
      checked_u32_size(q.strides(2), "flash_attn q_stride");
  const uint32_t k_stride =
      checked_u32_size(k.strides(2), "flash_attn k_stride");
  const uint32_t v_stride =
      checked_u32_size(v.strides(2), "flash_attn v_stride");

  const auto plan = make_flash_attention_execution_plan(
      hsk,
      hsv,
      q_len,
      q_heads,
      kv_heads,
      kv_len,
      batch,
      has_mask,
      do_causal,
      has_mask ? checked_u32_size((*mask).shape(1), "flash_attn mask_heads")
               : 1u,
      use_native_bf16_kv,
      q_stride,
      k_stride,
      v_stride);
  const auto& tuning = plan.tuning;
  const auto shader_id =
      flash_attention_main_shader(tuning.path, use_native_bf16_kv);
  if (tuning.d_split == 0 || tuning.block_rows == 0 || tuning.block_cols == 0 ||
      tuning.row_split == 0 || hsk % tuning.d_split != 0 ||
      hsv % tuning.d_split != 0 ||
      tuning.block_cols %
              (tuning.workgroup_size / tuning.d_split / tuning.row_split) !=
          0) {
    return false;
  }

  auto stride_bytes = [](const array& arr, int axis, const char* name) {
    return checked_u32_size(
        static_cast<int64_t>(arr.strides(axis)) * size_of(arr.dtype()), name);
  };

  vulkan::FlashAttentionPushConstants push_constants{};
  push_constants.N = plan.n_rows;
  push_constants.KV = kv_len;
  push_constants.ne1 = q_heads;
  push_constants.ne2 = q_len;
  push_constants.ne3 = batch;
  push_constants.neq2 = q_heads;
  push_constants.neq3 = batch;
  push_constants.nek2 = kv_heads;
  push_constants.nek3 = batch;
  push_constants.nev2 = checked_u32_size(v.shape(1), "flash_attn v_heads");
  push_constants.nev3 = checked_u32_size(v.shape(0), "flash_attn v_batch");
  push_constants.nem1 = has_mask
      ? checked_u32_size((*mask).shape(2), "flash_attn mask_rows")
      : 1u;
  push_constants.nem2 = has_mask
      ? checked_u32_size((*mask).shape(1), "flash_attn mask_heads")
      : 1u;
  push_constants.nem3 = has_mask
      ? checked_u32_size((*mask).shape(0), "flash_attn mask_batch")
      : 1u;
  push_constants.nb01 = q_stride;
  push_constants.nb02 = stride_bytes(q, 1, "flash_attn q_nb02");
  push_constants.nb03 = stride_bytes(q, 0, "flash_attn q_nb03");
  push_constants.nb11 = k_stride;
  push_constants.nb12 = stride_bytes(k, 1, "flash_attn k_nb12");
  push_constants.nb13 = stride_bytes(k, 0, "flash_attn k_nb13");
  push_constants.nb21 = v_stride;
  push_constants.nb22 = stride_bytes(v, 1, "flash_attn v_nb22");
  push_constants.nb23 = stride_bytes(v, 0, "flash_attn v_nb23");
  push_constants.scale = scale;
  push_constants.max_bias = 0.0f;
  push_constants.logit_softcap = 0.0f;
  push_constants.mask_n_head_log2 =
      sinks != nullptr ? kFlashAttnSinkEnableBit : 0u;
  push_constants.m0 = 0.0f;
  push_constants.m1 = 0.0f;
  push_constants.gqa_ratio = plan.gqa_ratio;
  push_constants.split_kv = plan.split_kv;
  push_constants.k_num = plan.split_k;

  std::optional<array> mask_opt;
  if (plan.use_mask_opt) {
    const uint32_t mask_opt_num_dwords =
        (kv_len + 16u * tuning.block_cols - 1u) / (16u * tuning.block_cols);
    const uint32_t mask_rows =
        (push_constants.nem1 + tuning.block_rows - 1u) / tuning.block_rows;
    const uint64_t mask_opt_elems = static_cast<uint64_t>(mask_opt_num_dwords) *
        mask_rows * push_constants.nem2 * push_constants.nem3;
    if (mask_opt_elems >
        static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    mask_opt = vulkan::acquire_scratch_array(
        s,
        kFlashAttnMaskOptScratchLane,
        {static_cast<int>(mask_opt_elems)},
        uint32);
  }

  std::optional<array> split_k_tmp;
  if (plan.split_k > 1u) {
    const uint64_t out_elems =
        static_cast<uint64_t>(hsv) * q_heads * q_len * batch;
    const uint64_t lm_elems =
        static_cast<uint64_t>(q_heads) * 2u * q_len * batch;
    const uint64_t split_k_elems = (out_elems + lm_elems) * plan.split_k;
    if (split_k_elems >
        static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    split_k_tmp = vulkan::acquire_scratch_array(
        s,
        kFlashAttnSplitKScratchLane,
        {static_cast<int>(split_k_elems)},
        float32);
  }

  const std::vector<uint32_t> specialization_constants = {
      tuning.workgroup_size,
      tuning.block_rows,
      tuning.block_cols,
      hsk,
      hsv,
      plan.aligned ? 0u : 1u,
      tuning.d_split,
      tuning.row_split,
      tuning.disable_subgroups ? 0u : tuning.subgroup_size,
      tuning.shmem_staging,
      (plan.use_mask_opt ? 1u : 0u) | (has_mask ? 2u : 0u) |
          (do_causal ? 8u : 0u),
      tuning.limit_occupancy_shmem,
  };

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);

    if (plan.use_mask_opt) {
      const uint32_t mask_opt_subgroup_size =
          std::max(vulkan::VulkanContext::get().subgroup_size(), 1u);
      vulkan::FlashAttentionMaskOptPushConstants mask_opt_push_constants{};
      mask_opt_push_constants.nem0 = kv_len;
      mask_opt_push_constants.nem1 = push_constants.nem1;
      mask_opt_push_constants.nem2 = push_constants.nem2;
      mask_opt_push_constants.nbm1 =
          checked_u32_size((*mask).strides(2), "flash_attn mask_nbm1");
      mask_opt_push_constants.nbm2 =
          checked_u32_size((*mask).strides(1), "flash_attn mask_nbm2");
      mask_opt_push_constants.nbm3 =
          checked_u32_size((*mask).strides(0), "flash_attn mask_nbm3");
      mask_opt_push_constants.nbd1 =
          (kv_len + 16u * tuning.block_cols - 1u) / (16u * tuning.block_cols);
      mask_opt_push_constants.nbd2 = mask_opt_push_constants.nbd1 *
          ((push_constants.nem1 + tuning.block_rows - 1u) / tuning.block_rows);
      mask_opt_push_constants.nbd3 =
          mask_opt_push_constants.nbd2 * push_constants.nem2;

      vulkan::dispatch_flash_attention_mask_opt_op(
          *mask,
          *mask_opt,
          vulkan::StaticShaderId::fa_mask_opt,
          command_buffer,
          s,
          mask_opt_push_constants,
          {
              mask_opt_push_constants.nbd1,
              (push_constants.nem1 + tuning.block_rows - 1u) /
                  tuning.block_rows,
              push_constants.nem2 * push_constants.nem3,
          },
          {
              128u,
              std::max(1u, 128u / mask_opt_subgroup_size),
              tuning.block_rows,
              tuning.block_cols,
          });
      insert_compute_barrier(command_buffer);
      vulkan::mark_scratch_array_written(s, kFlashAttnMaskOptScratchLane);
    }

    array& flash_output = plan.split_k > 1u ? *split_k_tmp : out_storage;
    vulkan::dispatch_flash_attention_op(
        q,
        k,
        v,
        has_mask ? *mask : q,
        sinks != nullptr ? *sinks : q,
        flash_output,
        plan.use_mask_opt ? *mask_opt : q,
        shader_id,
        command_buffer,
        s,
        push_constants,
        {
            ((plan.n_rows + tuning.block_rows - 1u) / tuning.block_rows) *
                plan.split_k,
            plan.workgroups_y,
            batch,
        },
        specialization_constants);

    if (plan.split_k > 1u) {
      insert_compute_barrier(command_buffer);
      vulkan::FlashAttentionSplitKReducePushConstants reduce_push_constants{};
      reduce_push_constants.D = hsv;
      reduce_push_constants.ne1 = q_heads;
      reduce_push_constants.ne2 = q_len;
      reduce_push_constants.ne3 = batch;
      reduce_push_constants.k_num = plan.split_k;
      reduce_push_constants.sinks = sinks != nullptr ? 1u : 0u;
      vulkan::dispatch_flash_attention_split_k_reduce_op(
          *split_k_tmp,
          sinks != nullptr ? *sinks : q,
          out_storage,
          vulkan::StaticShaderId::fa_split_k_reduce,
          command_buffer,
          s,
          reduce_push_constants,
          {q_heads, (hsv + 31u) / 32u, q_len * batch},
          {32u});
      vulkan::mark_scratch_array_written(s, kFlashAttnSplitKScratchLane);
    }

    vulkan::set_force_immediate_submit(s);
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "flash_attn_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_eval_flash_attention_vulkan(
    const std::vector<array>& inputs,
    array& out,
    float scale,
    bool do_causal,
    bool has_sinks,
    Stream s) {
  const bool has_arr_mask = inputs.size() > static_cast<size_t>(3 + has_sinks);
  const bool has_bool_mask = has_arr_mask && inputs[3].dtype() == bool_;
  if (inputs.size() != static_cast<size_t>(3 + has_sinks + has_arr_mask)) {
    return false;
  }
  array q = inputs[0];
  array k = inputs[1];
  array v = inputs[2];

  if (has_bool_mask) {
    return false;
  }

  if (!flash_attention_enabled_for(k, v)) {
    return false;
  }

  const bool supported_kv_dtype =
      (k.dtype() == float16 || k.dtype() == bfloat16) &&
      (v.dtype() == float16 || v.dtype() == bfloat16);
  if (!supported_kv_dtype || out.ndim() != 4 || q.ndim() != 4 ||
      k.ndim() != 4 || v.ndim() != 4) {
    return false;
  }

  const uint32_t batch = checked_u32_size(q.shape(0), "flash_attn batch");
  const uint32_t q_heads = checked_u32_size(q.shape(1), "flash_attn q_heads");
  const uint32_t q_len = checked_u32_size(q.shape(2), "flash_attn q_len");
  const uint32_t hsk = checked_u32_size(q.shape(3), "flash_attn hsk");
  const uint32_t kv_heads = checked_u32_size(k.shape(1), "flash_attn kv_heads");
  const uint32_t kv_len = checked_u32_size(k.shape(2), "flash_attn kv_len");
  const uint32_t hsv = checked_u32_size(v.shape(3), "flash_attn hsv");
  const bool is_lowp_attention =
      q.dtype() == float16 && k.dtype() == float16 && v.dtype() == float16;

  // Keep low-precision prefill on the manual path for now, but allow the
  // native flash kernel on decode shapes where generation throughput matters
  // most and the path was originally introduced. GPT-OSS prefill uses bf16 K/V
  // and is handled by the native bf16 path below.
  if (is_lowp_attention && q_len > 1) {
    return false;
  }

  // Use native bf16 KV when:
  // 1. bf16 is supported by the shader
  // 2. K and V are already bf16 (no cast needed)
  // 3. Either:
  //    a. Prefill mode: causal mask with q_len > 1 and q_len == kv_len
  //    b. Decode mode: q_len == 1 (K/V are appended during decode, so kv_len >=
  //    1)
  const bool use_native_bf16_kv =
      vulkan::VulkanContext::get().shader_bfloat16_supported() &&
      k.dtype() == bfloat16 && v.dtype() == bfloat16 &&
      ((do_causal && q_len > 1 && q_len == kv_len) ||
       (do_causal && q_len == 1));

  if (batch == 0 || q_heads == 0 || q_len == 0 || hsk == 0 || kv_heads == 0 ||
      kv_len == 0 || hsv == 0 || hsk % 4 != 0 || hsv % 4 != 0 ||
      q_heads % kv_heads != 0) {
    return false;
  }

  q = cast_flash_attention_q_to_f32(q, s);

  // Q casting already materializes dense storage; native bf16 K/V may still
  // arrive as sliced views and need an explicit contiguous copy here.
  auto ensure_flash_attention_kv_layout = [&](array x) {
    auto data = x.data_shared_ptr();
    if (data == nullptr || data->buffer.ptr() == nullptr ||
        !x.flags().row_contiguous || x.strides().back() != 1 ||
        x.offset() != 0) {
      x = contiguous_copy_gpu(x, s);
    }
    return x;
  };

  // Note: q is already processed by cast_flash_attention_q_to_f32 which
  // ensures proper layout. Only k and v need layout enforcement here.
  k = ensure_flash_attention_kv_layout(k);
  v = ensure_flash_attention_kv_layout(v);

  // Cast K/V to appropriate dtype if needed. When use_native_bf16_kv is true,
  // we skip casting since K/V are already bfloat16.
  if (!use_native_bf16_kv) {
    k = cast_flash_attention_kv_to_f16(k, kFlashAttnKCastScratchLane, s);
    v = cast_flash_attention_kv_to_f16(v, kFlashAttnVCastScratchLane, s);
  }
  std::optional<array> mask;
  if (has_arr_mask) {
    mask = cast_flash_attention_mask_to_f16(inputs[3], s);
  }
  std::optional<array> sinks;
  if (has_sinks) {
    sinks = cast_flash_attention_sinks_to_f32(inputs.back(), s);
  }
  trace_flash_attention_array("q_ready", q);
  trace_flash_attention_array("k_ready", k);
  trace_flash_attention_array("v_ready", v);
  if (mask.has_value()) {
    trace_flash_attention_array("mask_ready", *mask);
  }
  if (sinks.has_value()) {
    trace_flash_attention_array("sinks_ready", *sinks);
  }

  if (q.dtype() != float32) {
    return false;
  }
  if (use_native_bf16_kv) {
    if (k.dtype() != bfloat16 || v.dtype() != bfloat16) {
      return false;
    }
  } else if (k.dtype() != float16 || v.dtype() != float16) {
    return false;
  }

  try {
    const bool use_causal_shader = do_causal && q_len > 1;

    array out_storage = vulkan::acquire_scratch_array(
        s,
        kFlashAttnOutScratchLane,
        {out.shape(0), out.shape(2), out.shape(1), out.shape(3)},
        float32);

    if (!try_dispatch_flash_attention_native_vulkan(
            q,
            k,
            v,
            mask.has_value() ? &*mask : nullptr,
            sinks.has_value() ? &*sinks : nullptr,
            use_causal_shader,
            out_storage,
            s,
            use_native_bf16_kv,
            scale)) {
      return false;
    }
    vulkan::mark_scratch_array_written(s, kFlashAttnOutScratchLane);

    array out_transposed = swapaxes_in_eval(out_storage, 1, 2);
    trace_flash_attention_array("out_storage", out_storage);
    trace_flash_attention_array("out_transposed", out_transposed);
    copy_gpu(out_transposed, out, CopyType::General, s);
    out.set_status(array::Status::evaluated);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "flash_attn_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool sdpa_vulkan_supported(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    bool has_sinks,
    Stream s,
    std::string* reason = nullptr) {
  if (s.device == Device::cpu) {
    if (reason != nullptr) {
      *reason = "cpu stream";
    }
    return false;
  }
  if (has_mask && !do_causal && !has_arr_mask) {
    if (reason != nullptr) {
      *reason = "non-causal mask unsupported";
    }
    return false;
  }
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    if (reason != nullptr) {
      *reason = "rank != 4";
    }
    return false;
  }
  if (!is_vulkan_float_dtype(q.dtype()) || !is_vulkan_float_dtype(k.dtype()) ||
      !is_vulkan_float_dtype(v.dtype())) {
    if (reason != nullptr) {
      *reason = "unsupported dtype";
    }
    return false;
  }
  if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0) ||
      q.shape(-1) != k.shape(-1) || k.shape(1) != v.shape(1) ||
      q.shape(1) % k.shape(1) != 0 || k.shape(2) != v.shape(2) ||
      v.shape(-1) <= 0) {
    if (reason != nullptr) {
      *reason = "incompatible attention shapes";
    }
    return false;
  }
  return true;
}

bool is_supported_sdpa_rowwise_layout(const array& arr) {
  return arr.flags().contiguous && arr.offset() == 0 && arr.ndim() > 0 &&
      arr.strides().back() == 1;
}

bool has_vulkan_buffer_fast(const array& arr) {
  auto data = arr.data_shared_ptr();
  return data != nullptr && data->buffer.ptr() != nullptr;
}

array ensure_sdpa_rowwise_layout(array arr, Stream s) {
  if (!has_vulkan_buffer_fast(arr) || !is_supported_sdpa_rowwise_layout(arr)) {
    arr = contiguous_copy_gpu(arr, s);
  }
  return arr;
}

array reshape_sdpa_contiguous_view(const array& arr, const Shape& shape) {
  array out(shape, arr.dtype(), nullptr, {});
  out.copy_shared_buffer(
      arr,
      make_contiguous_strides(shape),
      {true, true, false},
      arr.data_size(),
      arr.offset());
  return out;
}

array collapse_sdpa_matmul_view(const array& arr) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  Shape shape = arr.shape();
  int64_t leading = 1;
  for (size_t i = 0; i < shape.size() - 3; ++i) {
    leading *= shape[i];
  }
  return reshape_sdpa_contiguous_view(
      arr,
      {
          static_cast<int>(leading),
          static_cast<int>(shape[shape.size() - 3]),
          static_cast<int>(shape[shape.size() - 2]),
          static_cast<int>(shape[shape.size() - 1]),
      });
}

array collapse_sdpa_scores_view(const array& arr) {
  if (arr.ndim() <= 3) {
    return arr;
  }
  Shape shape = arr.shape();
  int64_t leading = 1;
  for (size_t i = 0; i < shape.size() - 2; ++i) {
    leading *= shape[i];
  }
  return reshape_sdpa_contiguous_view(
      arr,
      {
          static_cast<int>(leading),
          static_cast<int>(shape[shape.size() - 2]),
          static_cast<int>(shape[shape.size() - 1]),
      });
}

array expand_sdpa_dim_view(const array& arr, int axis) {
  int ndim = arr.ndim();
  int normalized_axis = axis < 0 ? axis + ndim + 1 : axis;
  Shape shape = arr.shape();
  Strides strides = arr.strides();
  shape.insert(shape.begin() + normalized_axis, 1);
  strides.insert(strides.begin() + normalized_axis, 1);
  array out(shape, arr.dtype(), nullptr, {});
  out.copy_shared_buffer(
      arr, strides, arr.flags(), arr.data_size(), arr.offset());
  return out;
}

array broadcast_sdpa_view(const array& arr, const Shape& shape) {
  if (arr.shape() == shape) {
    return arr;
  }
  array out(shape, arr.dtype(), nullptr, {});
  broadcast(arr, out);
  return out;
}

array swap_sdpa_last_two_dims_view(const array& arr) {
  if (arr.ndim() < 2) {
    return arr;
  }
  Shape shape = arr.shape();
  Strides strides = arr.strides();
  std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);
  std::swap(strides[strides.size() - 1], strides[strides.size() - 2]);
  array out(shape, arr.dtype(), nullptr, {});
  out.copy_shared_buffer(
      arr, strides, arr.flags(), arr.data_size(), arr.offset());
  return out;
}

array cast_to_dtype_sdpa(const array& arr, Dtype dtype, Stream s) {
  if (arr.dtype() == dtype && has_vulkan_buffer_fast(arr) &&
      is_supported_sdpa_rowwise_layout(arr)) {
    return arr;
  }
  array out(arr.shape(), dtype, nullptr, {});
  copy_gpu(arr, out, CopyType::General, s);
  out.set_status(array::Status::evaluated);
  return out;
}

array cast_to_f32_sdpa(const array& arr, Stream s) {
  return cast_to_dtype_sdpa(arr, float32, s);
}

array prepare_sdpa_mask_vulkan(
    array mask,
    bool is_bool_mask,
    Dtype additive_dtype,
    const Shape& broadcast_shape,
    const Shape& scores_shape,
    Stream s) {
  if (!is_bool_mask) {
    mask = cast_to_dtype_sdpa(mask, additive_dtype, s);
  }
  if (mask.shape() != broadcast_shape) {
    mask = broadcast_to(mask, broadcast_shape, s);
  }
  mask = ensure_sdpa_rowwise_layout(mask, s);
  mask = reshape_sdpa_contiguous_view(mask, scores_shape);
  mask = ensure_sdpa_rowwise_layout(mask, s);
  mask.set_status(array::Status::evaluated);
  return mask;
}

array apply_sdpa_mask_vulkan(
    array scores,
    const array& mask,
    bool is_bool_mask,
    Stream s) {
  if (is_bool_mask) {
    scores = where(
        mask, scores, array(finfo(scores.dtype()).min, scores.dtype()), s);
    scores = ensure_sdpa_rowwise_layout(scores, s);
    scores.set_status(array::Status::evaluated);
    return scores;
  }

  array masked(scores.shape(), scores.dtype(), nullptr, {});
  eval_sdpa_binary_add_vulkan(scores, mask, masked, s);
  return masked;
}

void eval_sdpa_binary_vulkan(
    array lhs,
    array rhs,
    array& out,
    vulkan::StaticShaderId shader_id,
    Stream s) {
  lhs = ensure_sdpa_rowwise_layout(lhs, s);
  rhs = ensure_sdpa_rowwise_layout(rhs, s);
  if (lhs.shape() != rhs.shape() || lhs.shape() != out.shape()) {
    throw std::runtime_error(
        "SDPA binary dispatch received incompatible shapes.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  const std::vector<array> tracked_inputs = {lhs, rhs};
  const std::vector<array> tracked_outputs = {out};
  begin_tracked_manual_op(s, "sdpa.binary", tracked_inputs, tracked_outputs);
  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_binary_op(
        lhs,
        rhs,
        out,
        shader_id,
        command_buffer,
        s,
        vulkan::BinaryDispatchVariant::Standard);
    vulkan::end_command_recording(s.index);
  } catch (...) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    throw;
  }
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
  out.set_status(array::Status::evaluated);
}

vulkan::StaticShaderId sdpa_add_shader(Dtype dtype) {
  switch (dtype) {
    case float16:
      return vulkan::StaticShaderId::add_f16_f16_f16;
    case float32:
      return vulkan::StaticShaderId::add_f32_f32_f32;
    default:
      throw std::runtime_error("Unsupported SDPA add dtype.");
  }
}

vulkan::StaticShaderId sdpa_mul_shader(Dtype dtype) {
  switch (dtype) {
    case float16:
      return vulkan::StaticShaderId::mul_f16_f16_f16;
    case float32:
      return vulkan::StaticShaderId::mul_f32_f32_f32;
    default:
      throw std::runtime_error("Unsupported SDPA mul dtype.");
  }
}

vulkan::StaticShaderId sdpa_sub_shader(Dtype dtype) {
  switch (dtype) {
    case float16:
      return vulkan::StaticShaderId::sub_f16_f16_f16;
    case float32:
      return vulkan::StaticShaderId::sub_f32_f32_f32;
    default:
      throw std::runtime_error("Unsupported SDPA sub dtype.");
  }
}

void eval_sdpa_binary_add_vulkan(array lhs, array rhs, array& out, Stream s) {
  eval_sdpa_binary_vulkan(
      std::move(lhs), std::move(rhs), out, sdpa_add_shader(out.dtype()), s);
}

void eval_sdpa_binary_mul_vulkan(array lhs, array rhs, array& out, Stream s) {
  eval_sdpa_binary_vulkan(
      std::move(lhs), std::move(rhs), out, sdpa_mul_shader(out.dtype()), s);
}

void eval_sdpa_binary_sub_vulkan(array lhs, array rhs, array& out, Stream s) {
  eval_sdpa_binary_vulkan(
      std::move(lhs), std::move(rhs), out, sdpa_sub_shader(out.dtype()), s);
}

void eval_sdpa_scale_vulkan(
    const array& in,
    array& out,
    float scale,
    Stream s) {
  array scale_host(scale, float32);
  array scale_dev(scale_host.shape(), float32, nullptr, {});
  copy_gpu(scale_host, scale_dev, CopyType::General, s);
  scale_dev.set_status(array::Status::evaluated);
  eval_sdpa_binary_mul_vulkan(
      in, broadcast_sdpa_view(scale_dev, in.shape()), out, s);
}

void eval_sdpa_softmax_vulkan(array in, array& out, Stream s) {
  const bool use_f16_variant = in.dtype() == float16 && out.dtype() == float16;
  if (!(use_f16_variant || (in.dtype() == float32 && out.dtype() == float32))) {
    throw std::runtime_error(
        "SDPA softmax requires float16 or float32 tensors.");
  }

  array in_work = use_f16_variant ? cast_to_f32_sdpa(in, s) : in;
  array out_work =
      use_f16_variant ? array(out.shape(), float32, nullptr, {}) : out;

  in_work = ensure_sdpa_rowwise_layout(in_work, s);
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  const std::vector<array> tracked_inputs = {in_work};
  const std::vector<array> tracked_outputs = {out_work};
  begin_tracked_manual_op(s, "sdpa.softmax", tracked_inputs, tracked_outputs);

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    const uint32_t row_width = checked_u32_size(
        in_work.shape(in_work.ndim() - 1), "sdpa softmax width");
    if (row_width > 16384u) {
      vulkan::dispatch_softmax_large_op(
          in_work,
          out_work,
          use_f16_variant ? vulkan::StaticShaderId::soft_max_large1_f32_f16
                          : vulkan::StaticShaderId::soft_max_large1_f32,
          use_f16_variant ? vulkan::StaticShaderId::soft_max_large2_f32_f16
                          : vulkan::StaticShaderId::soft_max_large2_f32,
          use_f16_variant ? vulkan::StaticShaderId::soft_max_large3_f32_f16
                          : vulkan::StaticShaderId::soft_max_large3_f32,
          command_buffer,
          s);
    } else {
      vulkan::dispatch_softmax_op(
          in_work,
          out_work,
          use_f16_variant ? vulkan::StaticShaderId::soft_max_f32_f16
                          : vulkan::StaticShaderId::soft_max_f32,
          command_buffer,
          s);
    }
    vulkan::end_command_recording(s.index);
  } catch (...) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    throw;
  }
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
  if (use_f16_variant) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  out.set_status(array::Status::evaluated);
}

void eval_sdpa_logsumexp_vulkan(array in, array& out, Stream s) {
  in = ensure_sdpa_rowwise_layout(in, s);
  out.set_data(allocator::malloc(out.nbytes()));
  const std::vector<array> tracked_inputs = {in};
  const std::vector<array> tracked_outputs = {out};
  begin_tracked_manual_op(s, "sdpa.logsumexp", tracked_inputs, tracked_outputs);
  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_sum_rows_op(
        in, out, vulkan::StaticShaderId::logsumexp_f32, command_buffer, s);
    vulkan::end_command_recording(s.index);
  } catch (...) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    throw;
  }
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
  out.set_status(array::Status::evaluated);
}

void eval_sdpa_softmax_back_vulkan(array grad, array y, array& out, Stream s) {
  grad = ensure_sdpa_rowwise_layout(grad, s);
  y = ensure_sdpa_rowwise_layout(y, s);
  out.set_data(allocator::malloc(out.nbytes()));
  const std::vector<array> tracked_inputs = {grad, y};
  const std::vector<array> tracked_outputs = {out};
  begin_tracked_manual_op(
      s, "sdpa.softmax_back", tracked_inputs, tracked_outputs);
  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_softmax_back_op(
        grad,
        y,
        out,
        vulkan::StaticShaderId::soft_max_back_f32,
        command_buffer,
        s);
    vulkan::end_command_recording(s.index);
  } catch (...) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    throw;
  }
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
  out.set_status(array::Status::evaluated);
}

void eval_sdpa_repeat_back_vulkan(array in, array& out, Stream s) {
  in = ensure_sdpa_rowwise_layout(in, s);
  out.set_data(allocator::malloc(out.nbytes()));
  const std::vector<array> tracked_inputs = {in};
  const std::vector<array> tracked_outputs = {out};
  begin_tracked_manual_op(
      s, "sdpa.repeat_back", tracked_inputs, tracked_outputs);
  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_unary_op(
        in, out, vulkan::StaticShaderId::repeat_back_f32, command_buffer, s);
    vulkan::end_command_recording(s.index);
  } catch (...) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    throw;
  }
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
  out.set_status(array::Status::evaluated);
}

void eval_sdpa_sum_rows_vulkan(array in, array& out, Stream s) {
  in = ensure_sdpa_rowwise_layout(in, s);
  out.set_data(allocator::malloc(out.nbytes()));
  const std::vector<array> tracked_inputs = {in};
  const std::vector<array> tracked_outputs = {out};
  begin_tracked_manual_op(s, "sdpa.sum_rows", tracked_inputs, tracked_outputs);
  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_sum_rows_op(
        in, out, vulkan::StaticShaderId::sum_rows_f32, command_buffer, s);
    vulkan::end_command_recording(s.index);
  } catch (...) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    throw;
  }
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
  out.set_status(array::Status::evaluated);
}

void eval_sdpa_matmul_vulkan(
    const array& a,
    const array& b,
    array& out,
    Stream s) {
  array a_work = a;
  array b_work = b;

  if (a_work.ndim() > 2) {
    Shape a_shape = out.shape();
    a_shape[a_shape.size() - 1] = a_work.shape(a_work.ndim() - 1);
    a_shape[a_shape.size() - 2] = a_work.shape(a_work.ndim() - 2);
    if (a_work.shape() != a_shape) {
      a_work = broadcast_sdpa_view(a_work, a_shape);
    }
  }
  if (b_work.ndim() > 2) {
    Shape b_shape = out.shape();
    b_shape[b_shape.size() - 1] = b_work.shape(b_work.ndim() - 1);
    b_shape[b_shape.size() - 2] = b_work.shape(b_work.ndim() - 2);
    if (b_work.shape() != b_shape) {
      b_work = broadcast_sdpa_view(b_work, b_shape);
    }
  }

  a_work = ensure_sdpa_rowwise_layout(a_work, s);
  b_work = ensure_sdpa_rowwise_layout(b_work, s);

  array out_work = out;
  if (out_work.ndim() > 4) {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
    a_work = collapse_sdpa_matmul_view(a_work);
    b_work = collapse_sdpa_matmul_view(b_work);
    out_work = collapse_sdpa_matmul_view(out_work);
  } else if (
      out_work.data_shared_ptr() == nullptr ||
      out_work.data_shared_ptr()->buffer.ptr() == nullptr) {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  }

  const std::vector<array> tracked_inputs = {a_work, b_work};
  const std::vector<array> tracked_outputs = {out_work};
  begin_tracked_manual_op(s, "sdpa.matmul", tracked_inputs, tracked_outputs);
  try {
    if (!try_eval_matmul_vulkan({a_work, b_work}, out_work, s)) {
      throw std::runtime_error(
          "SDPA Vulkan matmul dispatch failed for broadcasted attention shapes.");
    }
  } catch (...) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    throw;
  }
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
  out.set_status(array::Status::evaluated);
}

bool try_eval_sdpa_heads_vulkan(
    const std::vector<array>& inputs,
    array& out,
    array* logsumexp_out,
    float scale,
    bool do_causal,
    Stream s) {
  const char* stage = "validate";
  try {
    if (inputs.size() < 3) {
      return false;
    }

    const array& q_in = inputs[0];
    const array& k_in = inputs[1];
    const array& v_in = inputs[2];
    if (q_in.ndim() != 4 || k_in.ndim() != 4 || v_in.ndim() != 4) {
      return false;
    }

    const int n_q_heads = q_in.shape(1);
    const int n_kv_heads = k_in.shape(1);
    const int n_repeats = n_q_heads / n_kv_heads;
    if (n_repeats <= 0) {
      return false;
    }

    const int batch = q_in.shape(0);
    const int q_len = q_in.shape(2);
    const int kv_len = k_in.shape(2);
    const int q_dim = q_in.shape(3);
    const int v_dim = v_in.shape(3);
    const int batch_heads = batch * n_kv_heads;
    const bool has_arr_mask = inputs.size() > 3;
    const bool has_bool_mask = has_arr_mask && inputs[3].dtype() == bool_;

    const bool is_decode = (q_len == 1 && n_repeats > 1);

    // Build output shape in head layout (not collapsed batch_heads layout).
    const Shape out_head_shape = {batch, n_q_heads, q_len, v_dim};
    const Shape score_head_shape = {batch, n_q_heads, q_len, kv_len};

    std::vector<array> tracked_outputs = {out};
    if (logsumexp_out != nullptr) {
      tracked_outputs.push_back(*logsumexp_out);
    }
    vulkan::record_primitive_for_stream(s, "sdpa_manual_heads");
    vulkan::ScopedPrimitiveTracking tracking_scope(s, inputs, tracked_outputs);

    if (false && is_decode && k_in.dtype() != float32 &&
        v_in.dtype() != float32) {
      // --- Decode GQA path: avoid broadcast copies on K/V
      // ----------------------- Use NC matvec with f16 matrix + f32 vector, no
      // repeated-head broadcast.

      stage = "cast_inputs_decode";
      // q: scale + cast to f32
      array q_f32 =
          cast_to_f32_sdpa(multiply(array(scale, q_in.dtype()), q_in, s), s);
      // K/V: cast original (bf16/f16) directly to f16 for NC matvec.
      array k_f16(k_in.shape(), float16, nullptr, {});
      copy_gpu(k_in, k_f16, CopyType::General, s);
      k_f16.set_status(array::Status::evaluated);

      array v_f16(v_in.shape(), float16, nullptr, {});
      copy_gpu(v_in, v_f16, CopyType::General, s);
      v_f16.set_status(array::Status::evaluated);

      // q in head layout, must be contiguous for matvec
      stage = "reshape_q_decode";
      array q_heads =
          reshape_sdpa_contiguous_view(q_f32, {batch, n_q_heads, q_len, q_dim});
      q_heads = ensure_sdpa_rowwise_layout(q_heads, s);
      q_heads.set_status(array::Status::evaluated);

      // k is already in [batch, n_kv_heads, kv_len, q_dim] f16.
      // Ensure k is row-contiguous so matvec strides are trivial.
      if (!is_supported_sdpa_rowwise_layout(k_f16)) {
        k_f16 = contiguous_copy_gpu(k_f16, s);
      }
      k_f16.set_status(array::Status::evaluated);

      // --- QK matvec via NC dispatch (matrix = k_f16, vec = q_heads)
      // ----------
      stage = "scores_matmul_decode";
      array scores_head(score_head_shape, float32, nullptr, {});
      scores_head.set_data(allocator::malloc(scores_head.nbytes()));

      {
        const uint32_t gqa = static_cast<uint32_t>(n_repeats);
        auto cmd = vulkan::begin_command_recording(s.index);
        vulkan::dispatch_mul_mat_vec_nc_op(
            // matrix: k_f16 [batch, n_kv_heads, kv_len, q_dim]
            k_f16,
            // vec:    q_heads [batch, n_q_heads, 1, q_dim]
            q_heads,
            scores_head,
            vulkan::StaticShaderId::mul_mat_vec_nc_f16_f32,
            cmd,
            s,
            {
                static_cast<uint32_t>(q_dim), // ncols_x = inner dim
                static_cast<uint32_t>(kv_len), // nrows_x = output dim
                static_cast<uint32_t>(k_f16.strides(-2)), // row_stride_x
                static_cast<uint32_t>(k_f16.strides(-3)), // channel_stride_x
                static_cast<uint32_t>(q_heads.strides(-3)), // channel_stride_y
                gqa, // channel_x_divisor
                static_cast<uint32_t>(n_q_heads), // ne12
                0,
                0,
                static_cast<uint32_t>(k_f16.strides(-4)), // nb03
                static_cast<uint32_t>(q_heads.strides(-4)), // nb13
                static_cast<uint32_t>(scores_head.strides(-4)), // nb23
                0,
            },
            {
                static_cast<uint32_t>(batch),
                static_cast<uint32_t>(kv_len),
                static_cast<uint32_t>(n_q_heads),
            });
        vulkan::end_command_recording(s.index);
      }
      scores_head.set_status(array::Status::evaluated);

      // Cast scores to the f32 type expected by downstream stages.
      // (scores_head is already f32, but output dtype may differ; keep as is.)

      // Mask
      std::optional<array> mask_work;
      if (inputs.size() > 3) {
        stage = "prepare_mask_decode";
        array mask = cast_to_f32_sdpa(inputs[3], s);
        if (mask.ndim() == 4 && mask.shape(1) == 1) {
          mask = broadcast_to(mask, score_head_shape, s);
        }
        mask = ensure_sdpa_rowwise_layout(mask, s);
        mask = reshape_sdpa_contiguous_view(mask, score_head_shape);
        mask = ensure_sdpa_rowwise_layout(mask, s);
        mask.set_status(array::Status::evaluated);
        mask_work = mask;
      }

      if (do_causal) {
        stage = "causal_mask_decode";
        const int n_past = k_in.shape(2) - q_in.shape(2);
        scores_head = apply_diag_mask_inf_vulkan(scores_head, n_past, s);
      }

      if (mask_work.has_value()) {
        stage = "add_mask_decode";
        array masked(score_head_shape, float32, nullptr, {});
        eval_sdpa_binary_add_vulkan(scores_head, *mask_work, masked, s);
        scores_head = masked;
      }

      if (logsumexp_out != nullptr) {
        stage = "logsumexp_decode";
        Shape logsumexp_shape = score_head_shape;
        logsumexp_shape.back() = 1;
        array logsumexp_f32(logsumexp_shape, float32, nullptr, {});
        eval_sdpa_logsumexp_vulkan(scores_head, logsumexp_f32, s);
        copy_gpu(
            reshape_sdpa_contiguous_view(logsumexp_f32, logsumexp_out->shape()),
            *logsumexp_out,
            CopyType::General,
            s);
        logsumexp_out->set_status(array::Status::evaluated);
      }

      stage = "softmax_decode";
      array probs_head(score_head_shape, float32, nullptr, {});
      eval_sdpa_softmax_vulkan(scores_head, probs_head, s);

      // --- Scores×V matvec via NC dispatch (matrix = v_f16, vec = probs)
      // ------
      stage = "result_matmul_decode";
      // Ensure v_f16 is contiguous before swapaxes
      if (!is_supported_sdpa_rowwise_layout(v_f16)) {
        v_f16 = contiguous_copy_gpu(v_f16, s);
      }
      v_f16.set_status(array::Status::evaluated);

      array v_t = swapaxes_in_eval(v_f16, -1, -2);
      if (!is_supported_sdpa_rowwise_layout(v_t)) {
        v_t = contiguous_copy_gpu(v_t, s);
      }
      v_t.set_status(array::Status::evaluated);

      {
        array result_heads(out_head_shape, float32, nullptr, {});
        result_heads.set_data(allocator::malloc(result_heads.nbytes()));

        const uint32_t gqa = static_cast<uint32_t>(n_repeats);
        auto cmd = vulkan::begin_command_recording(s.index);
        vulkan::dispatch_mul_mat_vec_nc_op(
            v_t,
            probs_head,
            result_heads,
            vulkan::StaticShaderId::mul_mat_vec_nc_f16_f32,
            cmd,
            s,
            {
                static_cast<uint32_t>(kv_len), // ncols_x = inner dim
                static_cast<uint32_t>(v_dim), // nrows_x = output dim
                static_cast<uint32_t>(v_t.strides(-2)), // row_stride_x
                static_cast<uint32_t>(v_t.strides(-3)), // channel_stride_x
                static_cast<uint32_t>(
                    probs_head.strides(-3)), // channel_stride_y
                gqa, // channel_x_divisor
                static_cast<uint32_t>(n_q_heads), // ne12
                0,
                0,
                static_cast<uint32_t>(v_t.strides(-4)), // nb03
                static_cast<uint32_t>(probs_head.strides(-4)), // nb13
                static_cast<uint32_t>(result_heads.strides(-4)), // nb23
                0,
            },
            {
                static_cast<uint32_t>(batch),
                static_cast<uint32_t>(v_dim),
                static_cast<uint32_t>(n_q_heads),
            });
        vulkan::end_command_recording(s.index);

        result_heads.set_status(array::Status::evaluated);
        stage = "copy_out_decode";
        copy_gpu(
            reshape_sdpa_contiguous_view(result_heads, out.shape()),
            out,
            CopyType::General,
            s);
      }
      out.set_status(array::Status::evaluated);
      return true;
    }

    // ==========================================================================
    // Prefill (or non-decode) path: original broadcast + materialize logic
    // ==========================================================================

    const bool use_precise_masked_gqa_path = logsumexp_out == nullptr &&
        has_arr_mask && !has_bool_mask && !do_causal && n_repeats > 1 &&
        q_len > 1;
    if (use_precise_masked_gqa_path) {
      // Additive-mask GQA is more reliable when we reuse the primitive op
      // sequence directly instead of the flattened manual helper path.
      stage = "precise_masked_gqa";
      array q = multiply(array(scale, q_in.dtype()), q_in, s);
      array k = k_in;
      array v = v_in;

      q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
      k = expand_dims(k, 2, s);
      v = expand_dims(v, 2, s);

      array scores = matmul(q, swapaxes(k, -1, -2, s), s);
      array mask = inputs[3];
      if (mask.ndim() >= 3) {
        if (mask.shape(-3) == 1) {
          mask = expand_dims(mask, -3, s);
        } else {
          mask = unflatten(mask, -3, {n_kv_heads, n_repeats}, s);
        }
      }
      if (mask.shape() != scores.shape()) {
        mask = broadcast_to(mask, scores.shape(), s);
      }
      scores = add(scores, mask, s);
      scores = softmax(scores, std::vector<int>{-1}, true, s);

      copy_gpu(
          flatten(matmul(scores, v, s), 1, 2, s), out, CopyType::General, s);
      out.set_status(array::Status::evaluated);
      return true;
    }

    const bool use_lowp_path = logsumexp_out == nullptr &&
        q_in.dtype() == float16 && k_in.dtype() == float16 &&
        v_in.dtype() == float16 &&
        !(has_arr_mask && n_repeats > 1 && kv_len >= 8192);
    const Dtype compute_dtype = use_lowp_path ? float16 : float32;

    stage = "cast_inputs";
    array q = cast_to_dtype_sdpa(
        multiply(array(scale, q_in.dtype()), q_in, s), compute_dtype, s);
    array k = cast_to_dtype_sdpa(k_in, compute_dtype, s);
    array v = cast_to_dtype_sdpa(v_in, compute_dtype, s);

    stage = "reshape_q";
    q = reshape(q, {batch_heads, n_repeats, q_len, q_dim}, s);
    q = ensure_sdpa_rowwise_layout(q, s);
    q.set_status(array::Status::evaluated);

    stage = "prepare_k";
    k = broadcast_to(
        expand_dims(k, 2, s), {batch, n_kv_heads, n_repeats, kv_len, q_dim}, s);
    k = reshape(k, {batch_heads, n_repeats, kv_len, q_dim}, s);
    k = ensure_sdpa_rowwise_layout(k, s);
    k.set_status(array::Status::evaluated);

    stage = "scores_matmul";
    Shape scores_shape = {batch_heads, n_repeats, q_len, kv_len};
    array scores(scores_shape, compute_dtype, nullptr, {});
    eval_sdpa_matmul_vulkan(q, swap_sdpa_last_two_dims_view(k), scores, s);

    std::optional<array> mask_work;
    if (has_arr_mask) {
      stage = "prepare_mask";
      array mask = inputs[3];
      if (has_bool_mask) {
        mask = where(
            mask,
            array(0.0f, compute_dtype),
            array(finfo(compute_dtype).min, compute_dtype),
            s);
      }
      mask_work = prepare_sdpa_mask_vulkan(
          mask,
          false,
          compute_dtype,
          {batch, n_q_heads, q_len, kv_len},
          scores.shape(),
          s);
    }

    if (do_causal) {
      stage = "causal_mask";
      const int n_past = k_in.shape(2) - q_in.shape(2);
      scores = apply_diag_mask_inf_vulkan(scores, n_past, s);
    }

    if (mask_work.has_value()) {
      stage = "add_mask";
      scores = apply_sdpa_mask_vulkan(scores, *mask_work, false, s);
    }

    if (logsumexp_out != nullptr) {
      stage = "logsumexp";
      Shape logsumexp_shape = scores.shape();
      logsumexp_shape.back() = 1;
      array logsumexp_f32(logsumexp_shape, float32, nullptr, {});
      eval_sdpa_logsumexp_vulkan(scores, logsumexp_f32, s);
      copy_gpu(
          reshape_sdpa_contiguous_view(logsumexp_f32, logsumexp_out->shape()),
          *logsumexp_out,
          CopyType::General,
          s);
      logsumexp_out->set_status(array::Status::evaluated);
    }

    stage = "softmax";
    array probs(scores.shape(), compute_dtype, nullptr, {});
    eval_sdpa_softmax_vulkan(scores, probs, s);

    stage = "prepare_v";
    array v_work = broadcast_to(
        expand_dims(v, 2, s), {batch, n_kv_heads, n_repeats, kv_len, v_dim}, s);
    v_work = reshape(v_work, {batch_heads, n_repeats, kv_len, v_dim}, s);
    v_work = ensure_sdpa_rowwise_layout(v_work, s);

    stage = "result_matmul";
    array result(
        {batch_heads, n_repeats, q_len, v_dim}, compute_dtype, nullptr, {});
    eval_sdpa_matmul_vulkan(probs, v_work, result, s);
    stage = "copy_out";
    copy_gpu(
        reshape_sdpa_contiguous_view(result, out.shape()),
        out,
        CopyType::General,
        s);
    out.set_status(array::Status::evaluated);
    return true;
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("ScaledDotProductAttention GQA stage failed at ") + stage +
        ": " + e.what());
  }
}

array apply_diag_mask_inf_vulkan(const array& scores, int n_past, Stream s) {
  array in = ensure_sdpa_rowwise_layout(scores, s);

  array masked(in.shape(), in.dtype(), nullptr, {});
  masked.set_status(array::Status::available);
  masked.set_data(allocator::malloc(masked.nbytes()));

  auto shader_id = [&]() {
    switch (in.dtype()) {
      case float32:
        return vulkan::StaticShaderId::diag_mask_inf_f32;
      case float16:
        return vulkan::StaticShaderId::diag_mask_inf_f16;
      case bfloat16:
        return vulkan::StaticShaderId::diag_mask_inf_bf16;
      default:
        throw std::runtime_error("DiagMaskInf unsupported dtype on Vulkan.");
    }
  }();

  const std::vector<array> tracked_inputs = {in};
  const std::vector<array> tracked_outputs = {masked};
  begin_tracked_manual_op(s, "diag_mask_inf", tracked_inputs, tracked_outputs);

  auto command_buffer = vulkan::begin_command_recording(s.index);
  vulkan::dispatch_diag_mask_inf_op(
      in,
      masked,
      shader_id,
      command_buffer,
      s,
      checked_u32_size(in.shape(in.ndim() - 2), "rows_per_channel"),
      n_past);
  vulkan::end_command_recording(s.index);
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);

  return masked;
}

bool try_eval_rms_norm_vulkan(
    const std::vector<array>& inputs,
    array& out,
    float eps,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array x = inputs[0];
  array w = inputs[1];
  if (x.ndim() == 0 || x.shape() != out.shape()) {
    return false;
  }

  const bool has_weight = w.ndim() != 0;

  if (!is_vulkan_float_dtype(x.dtype()) || !is_vulkan_float_dtype(w.dtype()) ||
      !is_vulkan_float_dtype(out.dtype())) {
    return false;
  }

  const bool native_lowp_2048 = x.shape(-1) == 2048 &&
      (x.dtype() == float16 || x.dtype() == bfloat16) &&
      x.dtype() == w.dtype() && x.dtype() == out.dtype();
  auto shader_id = (x.dtype() == float32 && w.dtype() == float32 &&
                    out.dtype() == float32) ||
          native_lowp_2048
      ? rms_norm_shader_id(x.dtype())
      : std::nullopt;
  const bool use_f32_staging_io = !shader_id.has_value();
  if (use_f32_staging_io) {
    array x_f32(x.shape(), float32, nullptr, {});
    array w_f32(w.shape(), float32, nullptr, {});
    copy_gpu(x, x_f32, CopyType::General, s);
    copy_gpu(w, w_f32, CopyType::General, s);
    x = x_f32;
    w = w_f32;
    shader_id = vulkan::StaticShaderId::rms_norm_f32;
  }

  if (w.ndim() > 4) {
    return false;
  }

  const uint32_t axis_size =
      checked_u32_size(x.shape(x.ndim() - 1), "axis_size");
  if (axis_size == 0 || axis_size > 32u * 512u) {
    return false;
  }

  if (!x.flags().contiguous || x.strides().back() != 1) {
    x = contiguous_copy_gpu(x, s);
  }
  if (has_weight && (!w.flags().row_contiguous || w.offset() != 0)) {
    w = contiguous_copy_gpu(w, s);
  }

  if (!is_supported_unary_layout(x) || !is_supported_unary_layout(w)) {
    return false;
  }

  const bool staged_output = use_f32_staging_io || !out.flags().contiguous ||
      out.offset() != 0 || out.strides().back() != 1;
  array out_work =
      staged_output ? array(out.shape(), x.dtype(), nullptr, {}) : out;
  set_unary_output_data(x, out_work);
  if (!out_work.flags().contiguous || out_work.offset() != 0 ||
      out_work.strides().back() != 1 || !is_supported_unary_layout(out_work)) {
    return false;
  }

  if (x.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const uint32_t nrows =
      x.ndim() >= 2 ? checked_u32_size(x.shape(x.ndim() - 2), "nrows") : 1u;
  const uint32_t nchannels =
      x.ndim() >= 3 ? checked_u32_size(x.shape(x.ndim() - 3), "nchannels") : 1u;
  const uint32_t nsamples =
      x.ndim() >= 4 ? checked_u32_size(x.shape(x.ndim() - 4), "nsamples") : 1u;

  if (nrows > std::numeric_limits<uint32_t>::max() ||
      nchannels > std::numeric_limits<uint32_t>::max() ||
      nsamples > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const std::vector<array> tracked_inputs =
      has_weight ? std::vector<array>{x, w} : std::vector<array>{x};
  const std::vector<array> tracked_outputs = {out_work};
  begin_tracked_manual_op(s, "rms_norm", tracked_inputs, tracked_outputs);

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_binary_op(
        x,
        w,
        out_work,
        *shader_id,
        command_buffer,
        s,
        vulkan::BinaryDispatchVariant::Standard,
        std::array<uint32_t, 3>{nrows, nchannels, nsamples},
        {0u, has_weight ? 1u : 0u},
        eps);
    vulkan::end_command_recording(s.index);
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "rms_norm_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_eval_layer_norm_vulkan(
    const std::vector<array>& inputs,
    array& out,
    float eps,
    Stream s) {
  if (inputs.size() != 3) {
    return false;
  }

  array x = inputs[0];
  array w = inputs[1];
  array b = inputs[2];
  if (x.ndim() == 0 || x.shape() != out.shape()) {
    return false;
  }

  const bool has_weight = w.ndim() != 0;
  const bool has_bias = b.ndim() != 0;
  if ((has_weight && (w.ndim() != 1 || w.shape(0) != x.shape(x.ndim() - 1))) ||
      (has_bias && (b.ndim() != 1 || b.shape(0) != x.shape(x.ndim() - 1)))) {
    return false;
  }

  if (!is_vulkan_float_dtype(x.dtype()) ||
      (has_weight && !is_vulkan_float_dtype(w.dtype())) ||
      (has_bias && !is_vulkan_float_dtype(b.dtype())) ||
      !is_vulkan_float_dtype(out.dtype())) {
    return false;
  }

  if (x.dtype() != float32) {
    array x_f32(x.shape(), float32, nullptr, {});
    copy_gpu(x, x_f32, CopyType::General, s);
    x = x_f32;
  }
  if (!x.flags().contiguous || x.offset() != 0 || x.strides().back() != 1) {
    x = contiguous_copy_gpu(x, s);
  }

  if (!x.flags().contiguous || x.offset() != 0 || x.strides().back() != 1 ||
      !is_supported_unary_layout(x)) {
    return false;
  }

  auto materialize_param = [&](array param) {
    if (param.dtype() != float32) {
      array param_f32(param.shape(), float32, nullptr, {});
      copy_gpu(param, param_f32, CopyType::General, s);
      param = param_f32;
    }
    if (!param.flags().row_contiguous || param.offset() != 0) {
      param = contiguous_copy_gpu(param, s);
    }
    return param;
  };
  auto make_default_affine_param = [&](float value) {
    array host(value, float32);
    array dev({1}, float32, nullptr, {});
    dev.set_data(allocator::malloc(dev.nbytes()));
    copy_gpu(host, dev, CopyType::Scalar, s);
    return dev;
  };

  const bool needs_affine = has_weight || has_bias;
  array out_target = needs_affine
      ? (out.dtype() == float16 ? array(out.shape(), float32, nullptr, {})
                                : out)
      : (out.dtype() == float32 ? out
                                : array(out.shape(), float32, nullptr, {}));
  const bool staged_output = needs_affine
      ? !is_supported_elementwise_layout(out_target)
      : !is_supported_unary_layout(out_target);
  array final_work = staged_output
      ? array(out_target.shape(), out_target.dtype(), nullptr, {})
      : out_target;

  array norm_out =
      needs_affine ? array(x.shape(), float32, nullptr, {}) : final_work;
  norm_out.set_data(allocator::malloc(norm_out.nbytes()));
  if (!is_supported_unary_layout(norm_out)) {
    return false;
  }

  std::optional<array> w_work;
  std::optional<array> b_work;

  if (needs_affine) {
    w_work =
        has_weight ? materialize_param(w) : make_default_affine_param(1.0f);
    b_work = has_bias ? materialize_param(b) : make_default_affine_param(0.0f);

    if (((w_work->ndim() == 1) && !w_work->flags().row_contiguous) ||
        w_work->offset() != 0) {
      return false;
    }
    if (((b_work->ndim() == 1) && !b_work->flags().row_contiguous) ||
        b_work->offset() != 0) {
      return false;
    }
  }

  if (needs_affine || staged_output) {
    final_work.set_data(allocator::malloc(final_work.nbytes()));
  }

  if (!is_supported_elementwise_layout(final_work)) {
    return false;
  }

  const std::vector<array> tracked_inputs = needs_affine
      ? std::vector<array>{x, *w_work, *b_work}
      : std::vector<array>{x};
  const std::vector<array> tracked_outputs = needs_affine
      ? std::vector<array>{norm_out, final_work}
      : std::vector<array>{final_work};
  begin_tracked_manual_op(s, "layer_norm", tracked_inputs, tracked_outputs);

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_norm_op(
        x, norm_out, vulkan::StaticShaderId::norm_f32, command_buffer, s, eps);

    if (needs_affine) {
      insert_compute_barrier(command_buffer);
      vulkan::LayerNormAffinePushConstants affine_push_constants{};
      affine_push_constants.ne =
          checked_u32_size(final_work.size(), "layer_norm_affine elements");
      affine_push_constants.axis_size =
          checked_u32_size(x.shape(x.ndim() - 1), "layer_norm_affine axis");
      affine_push_constants.w_stride = has_weight && w_work->ndim() == 1
          ? checked_u32_size(w_work->strides(0), "layer_norm_affine w_stride")
          : 0u;
      affine_push_constants.b_stride = has_bias && b_work->ndim() == 1
          ? checked_u32_size(b_work->strides(0), "layer_norm_affine b_stride")
          : 0u;
      const auto affine_shader_id = out_target.dtype() == bfloat16
          ? vulkan::StaticShaderId::layer_norm_affine_f32_bf16
          : vulkan::StaticShaderId::layer_norm_affine_f32;
      vulkan::dispatch_layer_norm_affine_op(
          norm_out,
          *w_work,
          *b_work,
          final_work,
          affine_shader_id,
          command_buffer,
          s,
          affine_push_constants);
    }

    vulkan::end_command_recording(s.index);
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
  } catch (const std::runtime_error& e) {
    end_tracked_manual_op(s, tracked_inputs, tracked_outputs);
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "layer_norm_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }

  if (staged_output) {
    copy_gpu(final_work, out_target, CopyType::GeneralGeneral, s);
  }
  if (out_target.id() != out.id()) {
    copy_gpu(out_target, out, CopyType::GeneralGeneral, s);
  }
  return true;
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  if (s.device == Device::cpu) {
    trace_use_fallback("ScaledDotProductAttention", s, "CPU stream");
    return true;
  }
  const bool lowp_attention =
      q.dtype() != float32 || k.dtype() != float32 || v.dtype() != float32;
  const bool bf16_attention =
      q.dtype() == bfloat16 && k.dtype() == bfloat16 && v.dtype() == bfloat16;
  const bool fp16_attention =
      q.dtype() == float16 && k.dtype() == float16 && v.dtype() == float16;
  const bool bf16_masked_gqa_prefill = bf16_attention && has_arr_mask &&
      q.shape(2) > 1 && q.shape(1) != k.shape(1);
  const bool lowp_masked =
      lowp_attention && has_arr_mask && !bf16_masked_gqa_prefill;
  const bool lowp_causal_gqa_prefill =
      fp16_attention && do_causal && q.shape(2) > 1 && q.shape(1) != k.shape(1);
  const bool lowp_decode_mha =
      lowp_attention && q.shape(2) == 1 && q.shape(1) == k.shape(1);
  if (!output_logsumexp &&
      (lowp_masked || lowp_causal_gqa_prefill || lowp_decode_mha)) {
    std::ostringstream details;
    details << "q_shape=" << q.shape() << " k_shape=" << k.shape()
            << " v_shape=" << v.shape() << " q_dtype=" << q.dtype()
            << " k_dtype=" << k.dtype() << " v_dtype=" << v.dtype()
            << " has_mask=" << has_mask << " has_arr_mask=" << has_arr_mask
            << " do_causal=" << do_causal
            << " output_logsumexp=" << output_logsumexp
            << " fp16_attention=" << fp16_attention
            << " lowp_masked=" << lowp_masked
            << " lowp_causal_gqa_prefill=" << lowp_causal_gqa_prefill
            << " lowp_decode_mha=" << lowp_decode_mha;
    trace_use_fallback(
        "ScaledDotProductAttention", s, "low_precision_guard", details.str());
    return true;
  }
  return false;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return true;
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  return s.device == Device::cpu;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() < 3 || outputs.empty()) {
    throw std::runtime_error(
        "ScaledDotProductAttention expects at least 3 inputs and 1 output.");
  }

  const array& q_in = inputs[0];
  const array& k_in = inputs[1];
  const array& v_in = inputs[2];
  const bool has_arr_mask = inputs.size() > static_cast<size_t>(3 + has_sinks_);
  const bool has_bool_mask = has_arr_mask && inputs[3].dtype() == bool_;

  std::string reason;
  if (!sdpa_vulkan_supported(
          q_in,
          k_in,
          v_in,
          has_arr_mask || do_causal_,
          has_arr_mask,
          do_causal_,
          false,
          output_logsumexp_,
          has_sinks_,
          stream(),
          &reason)) {
    throw std::runtime_error(
        std::string("ScaledDotProductAttention not supported on Vulkan: ") +
        reason);
  }

  auto s = stream();
  array* logsumexp_out =
      output_logsumexp_ && outputs.size() > 1 ? &outputs[1] : nullptr;

  if (!output_logsumexp_ &&
      try_eval_flash_attention_vulkan(
          inputs, outputs[0], scale_, do_causal_, has_sinks_, s)) {
    return;
  }

  if (!has_sinks_) {
    if (try_eval_sdpa_heads_vulkan(
            inputs, outputs[0], logsumexp_out, scale_, do_causal_, s)) {
      return;
    }
    throw std::runtime_error(
        output_logsumexp_
            ? "ScaledDotProductAttention logsumexp path failed on Vulkan."
            : "ScaledDotProductAttention path failed on Vulkan.");
  }

  if (output_logsumexp_) {
    throw std::runtime_error(
        "ScaledDotProductAttention with logsumexp output and attention sinks is not supported on Vulkan.");
  }

  array q = multiply(array(scale_, q_in.dtype()), q_in, s);
  array k = k_in;
  array v = v_in;

  const int n_q_heads = q.shape(1);
  const int n_kv_heads = k.shape(1);
  const int n_repeats = n_q_heads / n_kv_heads;

  if (n_repeats > 1) {
    q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
    k = expand_dims(k, 2, s);
    v = expand_dims(v, 2, s);
  }

  auto scores = matmul(q, swapaxes(k, -1, -2, s), s);
  if (has_arr_mask) {
    if (scores.dtype() != float32) {
      scores = astype(scores, float32, s);
    }
    array mask = inputs[3];
    if (n_repeats > 1 && mask.ndim() >= 3) {
      if (mask.shape(-3) == 1) {
        mask = expand_dims(mask, -3, s);
      } else {
        mask = unflatten(mask, -3, {n_kv_heads, n_repeats}, s);
      }
    }
    if (mask.shape() != scores.shape()) {
      mask = broadcast_to(mask, scores.shape(), s);
    }
    if (has_bool_mask) {
      scores = where(
          mask, scores, array(finfo(scores.dtype()).min, scores.dtype()), s);
    } else if (mask.dtype() != scores.dtype()) {
      mask = astype(mask, scores.dtype(), s);
      scores = add(scores, mask, s);
    } else {
      scores = add(scores, mask, s);
    }
    if (!scores.flags().row_contiguous || scores.offset() != 0 ||
        scores.strides().back() != 1) {
      scores = contiguous_copy_gpu(scores, s);
    }
  }
  if (do_causal_) {
    if (scores.dtype() != float32) {
      scores = astype(scores, float32, s);
    }
    const int n_past = k_in.shape(2) - q_in.shape(2);
    scores = apply_diag_mask_inf_vulkan(scores, n_past, s);
  }

  if (has_sinks_) {
    array sinks = inputs.back();
    sinks = expand_dims(sinks, {0, 2, 3}, s);
    if (n_repeats > 1) {
      sinks = unflatten(sinks, 1, {n_kv_heads, n_repeats}, s);
    }
    Shape sinks_shape = scores.shape();
    sinks_shape.back() = 1;
    scores = concatenate({broadcast_to(sinks, sinks_shape, s), scores}, -1, s);
  }

  const Shape scores_shape = scores.shape();
  const bool collapsed_scores = scores.ndim() > 4;
  if (collapsed_scores) {
    scores = flatten(scores, 0, scores.ndim() - 3, s);
  }
  scores = softmax(scores, std::vector<int>{-1}, true, s);
  if (has_sinks_) {
    auto start = Shape(scores.ndim(), 0);
    start.back() = 1;
    auto stop = scores.shape();
    scores = slice(scores, std::move(start), std::move(stop), s);
  }
  if (collapsed_scores) {
    scores = unflatten(
        scores, 0, Shape(scores_shape.begin(), scores_shape.end() - 2), s);
  }

  array v_work = v;
  if (v_work.dtype() != scores.dtype()) {
    v_work = astype(v_work, scores.dtype(), s);
  }

  array scores_work = scores;
  if (n_repeats > 1) {
    copy_gpu(
        flatten(matmul(scores_work, v_work, s), 1, 2, s),
        outputs[0],
        CopyType::General,
        s);
  } else {
    copy_gpu(matmul(scores_work, v_work, s), outputs[0], CopyType::General, s);
  }

  if (output_logsumexp_) {
    throw std::runtime_error(
        "ScaledDotProductAttention with logsumexp output is not supported on Vulkan.");
  }
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (outputs.size() != 3) {
    throw std::runtime_error(
        "ScaledDotProductAttentionVJP expects three outputs.");
  }
  if (inputs.size() < 6) {
    throw std::runtime_error(
        "ScaledDotProductAttentionVJP expects at least six inputs.");
  }
  if (has_sinks_) {
    throw std::runtime_error(
        "ScaledDotProductAttentionVJP with attention sinks is not supported on Vulkan.");
  }

  auto s = stream();
  const int primals_size = static_cast<int>(inputs.size()) - 3;
  const bool has_arr_mask = primals_size > 3;
  const bool has_bool_mask = has_arr_mask && inputs[3].dtype() == bool_;

  const array& q_in = inputs[0];
  const array& k_in = inputs[1];
  const array& v_in = inputs[2];

  const int n_q_heads = q_in.shape(1);
  const int n_kv_heads = k_in.shape(1);
  const int n_repeats = n_q_heads / n_kv_heads;
  const int batch = q_in.shape(0);
  const int q_len = q_in.shape(2);
  const int kv_len = k_in.shape(2);
  const int q_dim = q_in.shape(3);
  const int v_dim = v_in.shape(3);
  const int batch_heads = batch * n_kv_heads;

  auto prepare_q = [&](const array& src) {
    array q4 = src;
    q4 = reshape(q4, {batch_heads, n_repeats, q_len, q_dim}, s);
    q4 = ensure_sdpa_rowwise_layout(q4, s);
    q4.set_status(array::Status::evaluated);
    return q4;
  };

  auto prepare_kv = [&](const array& src, int width) {
    array out4 = src;
    out4 = broadcast_to(
        expand_dims(out4, 2, s),
        {batch, n_kv_heads, n_repeats, kv_len, width},
        s);
    out4 = reshape(out4, {batch_heads, n_repeats, kv_len, width}, s);
    out4 = ensure_sdpa_rowwise_layout(out4, s);
    out4.set_status(array::Status::evaluated);
    return out4;
  };

  array q_unscaled_h = prepare_q(q_in);
  array q_scaled_h = prepare_q(multiply(array(scale_, q_in.dtype()), q_in, s));
  array k4_h = prepare_kv(k_in, q_dim);
  array v4_h = prepare_kv(v_in, v_dim);

  array q_unscaled = cast_to_f32_sdpa(q_unscaled_h, s);
  array q_scaled = cast_to_f32_sdpa(q_scaled_h, s);
  array k4 = cast_to_f32_sdpa(k4_h, s);
  array v4 = cast_to_f32_sdpa(v4_h, s);

  Shape scores_shape = {batch_heads, n_repeats, q_len, kv_len};
  array scores(scores_shape, float32, nullptr, {});
  eval_sdpa_matmul_vulkan(
      q_scaled_h, swap_sdpa_last_two_dims_view(k4_h), scores, s);

  if (has_arr_mask) {
    array mask = prepare_sdpa_mask_vulkan(
        inputs[3],
        has_bool_mask,
        float32,
        {batch, n_q_heads, q_len, kv_len},
        scores_shape,
        s);
    scores = apply_sdpa_mask_vulkan(scores, mask, has_bool_mask, s);
  }

  if (do_causal_) {
    const int n_past = kv_len - q_len;
    scores = apply_diag_mask_inf_vulkan(scores, n_past, s);
  }

  array probs(scores_shape, float32, nullptr, {});
  eval_sdpa_softmax_vulkan(scores, probs, s);

  array probs_h(scores_shape, q_in.dtype(), nullptr, {});
  copy_gpu(probs, probs_h, CopyType::General, s);
  probs_h = ensure_sdpa_rowwise_layout(probs_h, s);
  probs_h.set_status(array::Status::evaluated);

  array d_o_h = inputs.back();
  d_o_h = reshape(d_o_h, {batch_heads, n_repeats, q_len, v_dim}, s);
  d_o_h = ensure_sdpa_rowwise_layout(d_o_h, s);
  d_o_h.set_status(array::Status::evaluated);

  array d_v_full_h(
      {batch_heads, n_repeats, kv_len, v_dim}, q_in.dtype(), nullptr, {});
  eval_sdpa_matmul_vulkan(
      swap_sdpa_last_two_dims_view(probs_h), d_o_h, d_v_full_h, s);
  array d_v_full = cast_to_f32_sdpa(d_v_full_h, s);

  array d_p_h(scores_shape, q_in.dtype(), nullptr, {});
  eval_sdpa_matmul_vulkan(d_o_h, swap_sdpa_last_two_dims_view(v4_h), d_p_h, s);

  array sv_h(scores_shape, q_in.dtype(), nullptr, {});
  eval_sdpa_binary_mul_vulkan(probs_h, d_p_h, sv_h, s);
  array sv_f32 = cast_to_f32_sdpa(sv_h, s);
  array row_sum_f32({batch_heads, n_repeats, q_len, 1}, float32, nullptr, {});
  eval_sdpa_sum_rows_vulkan(sv_f32, row_sum_f32, s);
  array row_sum_h(row_sum_f32.shape(), q_in.dtype(), nullptr, {});
  copy_gpu(row_sum_f32, row_sum_h, CopyType::General, s);
  row_sum_h.set_status(array::Status::evaluated);

  array row_bcast_h = broadcast_sdpa_view(row_sum_h, scores_shape);
  row_bcast_h = ensure_sdpa_rowwise_layout(row_bcast_h, s);
  row_bcast_h.set_status(array::Status::evaluated);

  array d_p_minus_row_h(scores_shape, q_in.dtype(), nullptr, {});
  eval_sdpa_binary_sub_vulkan(d_p_h, row_bcast_h, d_p_minus_row_h, s);
  array d_s_h(scores_shape, q_in.dtype(), nullptr, {});
  eval_sdpa_binary_mul_vulkan(probs_h, d_p_minus_row_h, d_s_h, s);

  array d_q_scaled_h(
      {batch_heads, n_repeats, q_len, q_dim}, q_in.dtype(), nullptr, {});
  eval_sdpa_matmul_vulkan(d_s_h, k4_h, d_q_scaled_h, s);
  array d_q_scaled = cast_to_f32_sdpa(d_q_scaled_h, s);
  array d_q({batch_heads, n_repeats, q_len, q_dim}, float32, nullptr, {});
  eval_sdpa_scale_vulkan(d_q_scaled, d_q, scale_, s);

  array d_k_full_h(
      {batch_heads, n_repeats, kv_len, q_dim}, q_in.dtype(), nullptr, {});
  eval_sdpa_matmul_vulkan(
      swap_sdpa_last_two_dims_view(d_s_h), q_scaled_h, d_k_full_h, s);
  array d_k_full = cast_to_f32_sdpa(d_k_full_h, s);

  array d_k = d_k_full;
  array d_v = d_v_full;
  if (n_repeats > 1) {
    array d_k_reduce_out({batch_heads, 1, kv_len, q_dim}, float32, nullptr, {});
    eval_sdpa_repeat_back_vulkan(d_k_full, d_k_reduce_out, s);
    d_k = reshape_sdpa_contiguous_view(d_k_reduce_out, k_in.shape());

    array d_v_reduce_out({batch_heads, 1, kv_len, v_dim}, float32, nullptr, {});
    eval_sdpa_repeat_back_vulkan(d_v_full, d_v_reduce_out, s);
    d_v = reshape_sdpa_contiguous_view(d_v_reduce_out, v_in.shape());
  } else {
    d_k = reshape_sdpa_contiguous_view(d_k_full, k_in.shape());
    d_v = reshape_sdpa_contiguous_view(d_v_full, v_in.shape());
  }

  d_q = reshape_sdpa_contiguous_view(d_q, q_in.shape());

  copy_gpu(d_q, outputs[0], CopyType::General, s);
  copy_gpu(d_k, outputs[1], CopyType::General, s);
  copy_gpu(d_v, outputs[2], CopyType::General, s);
  outputs[0].set_status(array::Status::evaluated);
  outputs[1].set_status(array::Status::evaluated);
  outputs[2].set_status(array::Status::evaluated);
}

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (outputs.size() != 1) {
    throw std::runtime_error("LayerNorm expects a single output.");
  }
  if (!try_eval_layer_norm_vulkan(inputs, outputs[0], eps_, stream())) {
    throw std::runtime_error("LayerNorm failed on Vulkan.");
  }
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 4 || outputs.size() != 3) {
    throw std::runtime_error("LayerNormVJP expects 4 inputs and 3 outputs.");
  }

  auto s = stream();
  const array& x = inputs[0];
  const array& w = inputs[1];
  const array& b = inputs[2];
  const array& g = inputs[3];

  array norm = number_of_elements(x, {-1}, true, x.dtype(), s);
  array sumx = sum(x, /* axis= */ -1, /* keepdims= */ true, s);
  array sumx2 = sum(square(x, s), /* axis= */ -1, /* keepdims= */ true, s);
  array mu = multiply(sumx, norm, s);
  array mu2 = multiply(sumx2, norm, s);
  array var = subtract(mu2, square(mu, s), s);
  array n = rsqrt(add(var, array(eps_, x.dtype()), s), s);
  array n3 = power(n, array(3, x.dtype()), s);
  array x_c = subtract(x, mu, s);

  array wg = multiply(w, g, s);
  array sumwg =
      multiply(sum(wg, /* axis= */ -1, /* keepdims= */ true, s), norm, s);
  array sumwgxc = multiply(
      sum(multiply(wg, x_c, s), /* axis= */ -1, /* keepdims= */ true, s),
      norm,
      s);
  array t1 = multiply(multiply(x_c, sumwgxc, s), n3, s);
  array t2 = multiply(subtract(wg, sumwg, s), n, s);
  array gx = subtract(t2, t1, s);

  std::vector<int> axes(g.ndim() - 1);
  std::iota(axes.begin(), axes.end(), 0);

  array gw = (w.ndim() == 0) ? zeros_like(w, s)
                             : sum(multiply(g, multiply(x_c, n, s), s),
                                   axes,
                                   /* keepdims= */ false,
                                   s);
  array gb = (b.ndim() == 0) ? zeros_like(b, s)
                             : sum(g, axes, /* keepdims= */ false, s);

  copy_gpu(gx, outputs[0], CopyType::General, stream());
  copy_gpu(gw, outputs[1], CopyType::General, stream());
  copy_gpu(gb, outputs[2], CopyType::General, stream());
  outputs[0].set_status(array::Status::evaluated);
  outputs[1].set_status(array::Status::evaluated);
  outputs[2].set_status(array::Status::evaluated);
}

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (outputs.size() != 1) {
    throw std::runtime_error("RMSNorm expects a single output.");
  }
  if (!try_eval_rms_norm_vulkan(inputs, outputs[0], eps_, stream())) {
    throw std::runtime_error("RMSNorm failed on Vulkan.");
  }
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 3 || outputs.size() != 2) {
    throw std::runtime_error("RMSNormVJP expects 3 inputs and 2 outputs.");
  }

  auto s = stream();
  array x = inputs[0];
  const array& w = inputs[1];
  array g = inputs[2];
  array& gx = outputs[0];
  array& gw = outputs[1];

  const bool has_w = w.ndim() != 0;
  array wg = has_w ? multiply(w, g, s) : g;

  if (!x.flags().row_contiguous || x.offset() != 0) {
    x = contiguous_copy_gpu(x, s);
  }
  if (has_w) {
    array wg_work(wg.shape(), wg.dtype(), nullptr, {});
    copy_gpu(wg, wg_work, CopyType::General, s);
    wg = wg_work;
  } else if (!wg.flags().row_contiguous || wg.offset() != 0) {
    wg = contiguous_copy_gpu(wg, s);
  }

  const uint32_t axis_size =
      checked_u32_size(x.shape(x.ndim() - 1), "rms_norm_vjp axis_size");
  if (axis_size == 0 || axis_size > 32u * 512u || x.dtype() != float32 ||
      wg.dtype() != float32 || gx.dtype() != float32) {
    throw std::runtime_error(
        "RMSNormVJP unsupported dtype or shape on Vulkan.");
  }
  const uint32_t nrows = checked_u32_size(
      x.data_size() / x.shape(x.ndim() - 1), "rms_norm_vjp nrows");

  gx.set_data(allocator::malloc(gx.nbytes()));
  const std::vector<array> tracked_inputs = {wg, x};
  const std::vector<array> tracked_outputs = {gx};
  begin_tracked_manual_op(s, "rms_norm_vjp", tracked_inputs, tracked_outputs);
  auto command_buffer = vulkan::begin_command_recording(s.index);
  vulkan::dispatch_rms_norm_back_op(
      wg,
      x,
      gx,
      vulkan::StaticShaderId::rms_norm_back_f32,
      command_buffer,
      s,
      axis_size,
      nrows,
      eps_);
  vulkan::end_command_recording(s.index);
  end_tracked_manual_op(s, tracked_inputs, tracked_outputs);

  array norm = number_of_elements(x, {-1}, true, x.dtype(), s);
  array mean_x2 = multiply(sum(square(x, s), -1, true, s), norm, s);
  array scale = rsqrt(add(mean_x2, array(eps_, x.dtype()), s), s);

  array gw_tmp =
      has_w ? multiply(g, multiply(x, scale, s), s) : zeros_like(w, s);
  if (has_w) {
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    gw_tmp = sum(gw_tmp, axes, false, s);
  }
  copy_gpu(gw_tmp, gw, CopyType::General, s);
  gx.set_status(array::Status::evaluated);
  gw.set_status(array::Status::evaluated);
}

void ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    throw std::runtime_error(
        "[ConvertFP8::eval_gpu] Expected one input and one output.");
  }

  if (try_convert_fp8_f32_gpu(inputs[0], outputs[0], to_fp8_, stream())) {
    return;
  }

  auto in = ensure_host_readable_row_contiguous(inputs[0], stream());
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  if (out.size() == 0) {
    return;
  }

  if (to_fp8_) {
    auto* dst = out.data<uint8_t>();
    switch (in.dtype()) {
      case float16: {
        auto* src = in.data<float16_t>();
        for (int i = 0; i < in.size(); ++i) {
          dst[i] = to_fp8_e4m3_scalar(static_cast<float>(src[i]));
        }
        return;
      }
      case bfloat16: {
        auto* src = in.data<bfloat16_t>();
        for (int i = 0; i < in.size(); ++i) {
          dst[i] = to_fp8_e4m3_scalar(static_cast<float>(src[i]));
        }
        return;
      }
      case float32: {
        auto* src = in.data<float>();
        for (int i = 0; i < in.size(); ++i) {
          dst[i] = to_fp8_e4m3_scalar(src[i]);
        }
        return;
      }
      default:
        throw std::runtime_error(
            "[ConvertFP8::eval_gpu] Unsupported input dtype.");
    }
  }

  auto* src = in.data<uint8_t>();
  switch (out.dtype()) {
    case float16: {
      auto* dst = out.data<float16_t>();
      for (int i = 0; i < in.size(); ++i) {
        dst[i] = float16_t(from_fp8_e4m3_scalar(src[i]));
      }
      return;
    }
    case bfloat16: {
      auto* dst = out.data<bfloat16_t>();
      for (int i = 0; i < in.size(); ++i) {
        dst[i] = bfloat16_t(from_fp8_e4m3_scalar(src[i]));
      }
      return;
    }
    case float32: {
      auto* dst = out.data<float>();
      for (int i = 0; i < in.size(); ++i) {
        dst[i] = from_fp8_e4m3_scalar(src[i]);
      }
      return;
    }
    default:
      throw std::runtime_error(
          "[ConvertFP8::eval_gpu] Unsupported output dtype.");
  }
}

void Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto copy_fallback_outputs = [&](std::vector<array> fallback_outputs) {
    auto& s = stream();
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].set_data(allocator::malloc(outputs[i].nbytes()));
      copy_gpu(fallback_outputs[i], outputs[i], CopyType::General, s);
    }
  };

  if (dequantize_) {
    if (mode_ != QuantizationMode::Affine) {
      if ((mode_ == QuantizationMode::Mxfp4 ||
           mode_ == QuantizationMode::Mxfp8) &&
          inputs.size() == 2 && outputs.size() == 1) {
        auto& out = outputs[0];
        array out_f32(out.shape(), float32, nullptr, {});
        if (!vulkan::fp_dequantize_to_float32(
                inputs[0], inputs[1], out_f32, stream(), group_size_, bits_)) {
          throw std::runtime_error(
              "[Quantize::eval_gpu] FP dequantize failed on Vulkan.");
        }

        out.set_data(allocator::malloc(out.nbytes()));
        copy_gpu(out_f32, out, CopyType::General, stream());
        return;
      }
      if (mode_ == QuantizationMode::Nvfp4 &&
          (inputs.size() == 2 || inputs.size() == 3) && outputs.size() == 1) {
        auto& out = outputs[0];
        array out_f32(out.shape(), float32, nullptr, {});
        std::optional<array> global_scale =
            inputs.size() == 3 ? std::make_optional(inputs[2]) : std::nullopt;
        if (!vulkan::nvfp4_dequantize_to_float32(
                inputs[0], inputs[1], global_scale, out_f32, stream())) {
          throw std::runtime_error(
              "[Quantize::eval_gpu] Nvfp4 dequantize failed on Vulkan.");
        }

        out.set_data(allocator::malloc(out.nbytes()));
        copy_gpu(out_f32, out, CopyType::General, stream());
        return;
      }

      throw std::runtime_error(
          "Quantize dequantize only supports Affine and Nvfp4 modes on Vulkan.");
    }
    if (inputs.size() != 3) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Expected affine dequantize inputs to include biases.");
    }

    auto& wq = inputs[0];
    auto& scales = inputs[1];
    auto& biases = inputs[2];
    auto& out = outputs[0];

    auto& s = stream();

    array scales_f32 = scales;
    if (scales_f32.dtype() != float32) {
      scales_f32 = array(scales.shape(), float32, nullptr, {});
      scales_f32.set_data(allocator::malloc(scales_f32.nbytes()));
      copy_gpu(scales, scales_f32, CopyType::General, s);
    }

    array biases_f32 = biases;
    if (biases_f32.dtype() != float32) {
      biases_f32 = array(biases.shape(), float32, nullptr, {});
      biases_f32.set_data(allocator::malloc(biases_f32.nbytes()));
      copy_gpu(biases, biases_f32, CopyType::General, s);
    }

    array out_f32(out.shape(), float32, nullptr, {});
    if (!vulkan::affine_dequantize_to_float32(
            wq, scales_f32, biases_f32, out_f32, s, group_size_, bits_)) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Affine dequantize failed on Vulkan.");
    }

    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(out_f32, out, CopyType::General, s);
  } else {
    if (mode_ != QuantizationMode::Affine) {
      if (mode_ == QuantizationMode::Mxfp4 ||
          mode_ == QuantizationMode::Mxfp8) {
        auto& s = stream();
        std::vector<array> fallback_inputs = inputs;
        if (fallback_inputs[0].dtype() != float32) {
          fallback_inputs[0] = astype(fallback_inputs[0], float32, s);
        }
        copy_fallback_outputs(fallback_(fallback_inputs));
        return;
      }
      if (mode_ == QuantizationMode::Nvfp4 &&
          (inputs.size() == 1 || inputs.size() == 2) && outputs.size() == 2) {
        auto& s = stream();
        array in_f32 = inputs[0];
        if (in_f32.dtype() != float32) {
          in_f32 = array(inputs[0].shape(), float32, nullptr, {});
          in_f32.set_data(allocator::malloc(in_f32.nbytes()));
          copy_gpu(inputs[0], in_f32, CopyType::General, s);
        }
        std::optional<array> global_scale =
            inputs.size() == 2 ? std::make_optional(inputs[1]) : std::nullopt;
        if (!vulkan::nvfp4_quantize_from_float32(
                in_f32, outputs[0], outputs[1], global_scale, s)) {
          throw std::runtime_error(
              "[Quantize::eval_gpu] Nvfp4 quantize failed on Vulkan.");
        }
        return;
      }
      throw std::runtime_error(
          "Quantize encode only supports Affine and Nvfp4 modes on Vulkan.");
    }
    if (outputs.size() != 3 || inputs.size() != 1) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Expected affine quantize outputs for weights, scales, and biases.");
    }

    auto& s = stream();
    array in_f32 = inputs[0];
    if (in_f32.dtype() != float32) {
      in_f32 = array(inputs[0].shape(), float32, nullptr, {});
      in_f32.set_data(allocator::malloc(in_f32.nbytes()));
      copy_gpu(inputs[0], in_f32, CopyType::General, s);
    }

    array scales_f32(outputs[1].shape(), float32, nullptr, {});
    array biases_f32(outputs[2].shape(), float32, nullptr, {});
    if (!vulkan::affine_quantize_from_float32(
            in_f32,
            outputs[0],
            scales_f32,
            biases_f32,
            s,
            group_size_,
            bits_)) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Affine quantize failed on Vulkan.");
    }

    outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
    outputs[2].set_data(allocator::malloc(outputs[2].nbytes()));
    copy_gpu(scales_f32, outputs[1], CopyType::General, s);
    copy_gpu(biases_f32, outputs[2], CopyType::General, s);
  }
}

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("CustomKernel has no Vulkan implementation.");
}

} // namespace fast

} // namespace mlx::core
