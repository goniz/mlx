// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/kernels.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"

namespace mlx::core::vulkan {

constexpr uint32_t kMaxMulMatVecCols = 8;

namespace {

constexpr char kSoftmaxLargeMaxScratchLane[] = "softmax.large.tmp_max";
constexpr char kSoftmaxLargeSumScratchLane[] = "softmax.large.tmp_sum";
constexpr char kCumsumMultipassScratchLane[] = "cumsum.multipass.tmp";
constexpr uint32_t kDescriptorSetBatchSize = 32;
constexpr uint32_t kVendorIdAmd = 0x1002u;
constexpr uint32_t kVendorIdIntel = 0x8086u;
constexpr uint32_t kVendorIdNvidia = 0x10DEu;

bool trace_descriptor_epochs_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_TRACE_DESCRIPTORS");
        env != nullptr) {
      return std::string(env) != "0";
    }
    if (const char* env = std::getenv("MLX_VULKAN_TRACE_SYNC");
        env != nullptr) {
      return std::string(env) != "0";
    }
    return false;
  }();
  return enabled;
}

void trace_descriptor_epochs(const std::string& msg) {
  if (!trace_descriptor_epochs_enabled()) {
    return;
  }

  static std::mutex trace_mutex;
  std::lock_guard<std::mutex> lock(trace_mutex);
  std::cerr << "[vulkan-descriptor] " << msg << std::endl;
}

template <typename T>
void hash_combine(size_t& seed, const T& value) {
  seed ^=
      std::hash<T>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

std::vector<uint32_t> matmul_specialization_constants(
    const std::vector<uint32_t>& selected_spec) {
  // Conservative matmul tile that avoids relying on device-specific tuning.
  // constant_id mapping in mul_mm.comp:
  //   0: BLOCK_SIZE, 1: BM, 2: BN, 3: BK, 4: WM, 5: WN,
  //   6: WMITER, 7: TM, 8: TN, 9: TK, 10: WARP
  static const std::vector<uint32_t> kDefaultSpec = {
      32, 32, 32, 16, 32, 32, 2, 2, 2, 1, 32};
  static const std::vector<uint32_t> kEnvSpec = []() {
    const char* env = std::getenv("MLX_VULKAN_MATMUL_SPEC");
    if (env == nullptr || std::strlen(env) == 0) {
      return std::vector<uint32_t>{};
    }

    std::vector<uint32_t> parsed;
    parsed.reserve(kDefaultSpec.size());
    std::stringstream ss(env);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (item.empty()) {
        return kDefaultSpec;
      }
      try {
        parsed.push_back(static_cast<uint32_t>(std::stoul(item)));
      } catch (...) {
        return kDefaultSpec;
      }
    }
    if (parsed.size() != kDefaultSpec.size()) {
      return std::vector<uint32_t>{};
    }
    return parsed;
  }();
  if (!kEnvSpec.empty()) {
    return kEnvSpec;
  }
  if (selected_spec.size() == kDefaultSpec.size()) {
    return selected_spec;
  }
  return kDefaultSpec;
}

VkDescriptorSetLayoutBinding make_storage_buffer_binding(uint32_t binding) {
  VkDescriptorSetLayoutBinding layout_binding{};
  layout_binding.binding = binding;
  layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  layout_binding.descriptorCount = 1;
  layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  layout_binding.pImmutableSamplers = nullptr;
  return layout_binding;
}

uint32_t checked_u32(int64_t value, const char* name) {
  if (value < 0 || value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + name + " is out of uint32 range.");
  }
  return static_cast<uint32_t>(value);
}

uint32_t checked_mul_u32(uint32_t a, uint32_t b, const char* name) {
  const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  if (product > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + name + " is out of uint32 range.");
  }
  return static_cast<uint32_t>(product);
}

uint32_t next_power_of_two_u32(uint32_t value) {
  if (value <= 1u) {
    return 1u;
  }
  if (value > (1u << 31)) {
    throw std::runtime_error(
        "[vulkan::kernels] value is too large for power-of-two padding.");
  }
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value + 1;
}

uint32_t log2_exact_u32(uint32_t value) {
  uint32_t result = 0;
  while (value > 1u) {
    value >>= 1;
    result++;
  }
  return result;
}

std::vector<uint32_t> copy_spirv_code(const void* data, size_t size_bytes) {
  if ((size_bytes % sizeof(uint32_t)) != 0) {
    throw std::runtime_error(
        "[vulkan::kernels] SPIR-V payload size must be a multiple of 4 bytes.");
  }

  std::vector<uint32_t> spirv_code(size_bytes / sizeof(uint32_t));
  if (!spirv_code.empty()) {
    std::memcpy(spirv_code.data(), data, size_bytes);
  }
  return spirv_code;
}

struct TensorLayout4D {
  uint32_t ne00{1};
  uint32_t ne01{1};
  uint32_t ne02{1};
  uint32_t ne03{1};
  uint32_t nb00{0};
  uint32_t nb01{0};
  uint32_t nb02{0};
  uint32_t nb03{0};
};

struct PipelineCreationOptions {
  bool disable_robustness{false};
  bool require_full_subgroups{false};
  uint32_t required_subgroup_size{0};
};

bool starts_with(const std::string& str, const char* prefix) {
  return str.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& str, const char* suffix) {
  const size_t suffix_len = std::strlen(suffix);
  return str.size() >= suffix_len &&
      str.compare(str.size() - suffix_len, suffix_len, suffix) == 0;
}

std::optional<StaticShaderId> static_shader_id_by_name(
    const std::string& shader_name) {
  for (size_t i = 0; i < static_cast<size_t>(StaticShaderId::Count); ++i) {
    const auto shader_id = static_cast<StaticShaderId>(i);
    if (shader_name == static_shader_name(shader_id)) {
      return shader_id;
    }
  }
  return std::nullopt;
}

PipelineCreationOptions pipeline_creation_options(
    const std::string& shader_name,
    const std::vector<uint32_t>& specialization_constants) {
  PipelineCreationOptions options;

  if ((shader_name == "soft_max_f32" || shader_name == "soft_max_f32_f16") &&
      !specialization_constants.empty()) {
    options.required_subgroup_size = specialization_constants[0];
    return options;
  }

  if (shader_name == "fa_mask_opt") {
    options.disable_robustness = true;
    if (specialization_constants.size() >= 2 &&
        specialization_constants[1] != 0) {
      options.require_full_subgroups = true;
      options.required_subgroup_size =
          specialization_constants[0] / specialization_constants[1];
    }
    return options;
  }

  if (shader_name == "fa_split_k_reduce") {
    options.disable_robustness = true;
    if (!specialization_constants.empty()) {
      options.required_subgroup_size = specialization_constants[0];
    }
    return options;
  }

  if (starts_with(shader_name, "flash_attn_")) {
    options.disable_robustness = true;
    if (!ends_with(shader_name, "_cm2") &&
        specialization_constants.size() > 8 &&
        specialization_constants[8] != 0) {
      options.require_full_subgroups = true;
      options.required_subgroup_size = specialization_constants[8];
    }
  }

  if (starts_with(shader_name, "matmul_") &&
      specialization_constants.size() > 10 &&
      specialization_constants[10] != 0) {
    options.require_full_subgroups = true;
    options.required_subgroup_size = specialization_constants[10];
  }

  return options;
}

PipelineCreationOptions pipeline_creation_options(
    StaticShaderId shader_id,
    const std::vector<uint32_t>& specialization_constants) {
  return pipeline_creation_options(
      static_shader_name(shader_id), specialization_constants);
}

TensorLayout4D make_tensor_layout_4d(
    const array& arr,
    const char* tensor_name) {
  if (arr.ndim() > 4) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + tensor_name +
        " rank > 4 is not supported.");
  }

  TensorLayout4D layout;
  for (size_t i = 0; i < arr.ndim(); ++i) {
    const int src_dim = static_cast<int>(arr.ndim() - 1 - i);
    const auto dim_size = checked_u32(arr.shape(src_dim), "shape");
    const auto stride = checked_u32(arr.strides(src_dim), "stride");

    switch (i) {
      case 0:
        layout.ne00 = dim_size;
        layout.nb00 = stride;
        break;
      case 1:
        layout.ne01 = dim_size;
        layout.nb01 = stride;
        break;
      case 2:
        layout.ne02 = dim_size;
        layout.nb02 = stride;
        break;
      case 3:
        layout.ne03 = dim_size;
        layout.nb03 = stride;
        break;
    }
  }

  return layout;
}

uint32_t
checked_offset(const array& arr, const char* tensor_name, uint32_t max_offset) {
  const int64_t byte_offset = arr.offset();
  const int64_t item_size = static_cast<int64_t>(size_of(arr.dtype()));
  if (item_size <= 0 || byte_offset < 0 || (byte_offset % item_size) != 0) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + tensor_name +
        " offset is misaligned.");
  }

  const int64_t elem_offset = byte_offset / item_size;
  if (elem_offset > static_cast<int64_t>(max_offset)) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] ") + tensor_name +
        " offset is out of supported range.");
  }
  return static_cast<uint32_t>(elem_offset);
}

void init_fastdiv_values(uint32_t d, uint32_t& mp, uint32_t& L) {
  if (d == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] fastdiv divisor must be non-zero.");
  }

  L = 0;
  while (L < 32 && (uint32_t{1} << L) < d) {
    L++;
  }
  mp = static_cast<uint32_t>(
      ((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d) + 1);
}

void init_unary_fastdiv(UnaryPushConstants& p) {
  init_fastdiv_values(p.ne02 * p.ne01 * p.ne00, p.ne0_012mp, p.ne0_012L);
  init_fastdiv_values(p.ne01 * p.ne00, p.ne0_01mp, p.ne0_01L);
  init_fastdiv_values(p.ne00, p.ne0_0mp, p.ne0_0L);
  init_fastdiv_values(p.ne12 * p.ne11 * p.ne10, p.ne1_012mp, p.ne1_012L);
  init_fastdiv_values(p.ne11 * p.ne10, p.ne1_01mp, p.ne1_01L);
  init_fastdiv_values(p.ne10, p.ne1_0mp, p.ne1_0L);
}

VkDescriptorBufferInfo make_buffer_info(const array& arr, const char* name) {
  auto* vulkan_buffer = static_cast<const VulkanBuffer*>(
      static_cast<const void*>(arr.buffer().ptr()));
  if (vulkan_buffer == nullptr || vulkan_buffer->buffer == VK_NULL_HANDLE) {
    throw std::runtime_error(
        std::string("[vulkan::kernels] Missing Vulkan buffer for ") + name +
        ".");
  }

  VkDescriptorBufferInfo info{};
  info.buffer = vulkan_buffer->buffer;
  info.offset = 0;
  info.range = VK_WHOLE_SIZE;
  return info;
}

enum class DispatchGridKind {
  ElementWise,
  Linear1D,
  RowWise,
};

enum class KernelSpecId {
  Binary,
  BinaryAddWithPartials,
  Ternary,
  Unary,
  GenericUnary,
  Arange,
  SumRows,
  Argmax,
  SoftmaxBack,
  Softmax,
  SoftmaxLarge,
  DiagMaskInf,
  CumsumMultipass,
  MatVec,
  MatVecP021,
  MatVecNc,
  Matmul,
  MatmulSplitKReduce,
  RandomBits,
  Gather,
  GatherAxis,
  GatherPair,
  MaskedScatter,
  Rope,
  FlashAttention,
  FlashAttentionSplitKReduce,
  FlashAttentionMaskOpt,
  AffineDequant,
  AffineQuant,
  Nvfp4Dequant,
  Nvfp4Quant,
  FusedAffineMatmul,
  GatherAffineMatmul,
  Nvfp4QMatmul,
  LayerNormAffine,
  Argsort,
  ArgsortLarge,
  FFT,
};

struct KernelSpec {
  std::vector<uint32_t> bindings;
  uint32_t binding_count{0};
  uint32_t push_constant_size{0};
  DispatchGridKind grid_kind{DispatchGridKind::ElementWise};
};

struct BoundArray {
  const array* arr;
  const char* name;
};

KernelSpec make_kernel_spec(
    std::initializer_list<uint32_t> bindings,
    uint32_t push_constant_size,
    DispatchGridKind grid_kind) {
  return {
      std::vector<uint32_t>(bindings),
      static_cast<uint32_t>(bindings.size()),
      push_constant_size,
      grid_kind};
}

const std::array<KernelSpec, 38> kKernelSpecs = {
    make_kernel_spec(
        {0, 1, 2},
        sizeof(BinaryPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(BinaryPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(TernaryPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1},
        sizeof(UnaryPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1},
        sizeof(GenericPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0},
        sizeof(ArangePushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1},
        sizeof(SumRowsPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1},
        sizeof(GenericPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1, 2},
        sizeof(GenericPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(SoftmaxPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1, 2, 3, 4, 5},
        sizeof(SoftmaxPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1},
        sizeof(DiagMaskInfPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2},
        sizeof(SumRowsPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1, 2, 3, 4},
        sizeof(MatVecPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3, 4},
        sizeof(MatVecP021PushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3, 4},
        sizeof(MatVecNcPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2},
        sizeof(MatmulPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1},
        sizeof(MatmulSplitKReducePushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1},
        sizeof(RandomBitsPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2},
        sizeof(GatherPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2},
        sizeof(GatherAxisPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(GatherPairPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2},
        sizeof(MaskedScatterPushConstants),
        DispatchGridKind::ElementWise),
    make_kernel_spec(
        {0, 1, 2, 3, 4},
        sizeof(RopePushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3, 4, 5, 6},
        sizeof(FlashAttentionPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2},
        sizeof(FlashAttentionSplitKReducePushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1},
        sizeof(FlashAttentionMaskOptPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(AffineDequantPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(AffineQuantPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(Nvfp4DequantPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(Nvfp4QuantPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3, 4},
        sizeof(FusedAffineMatmulPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3, 4, 5, 6},
        sizeof(GatherAffineMatmulPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3, 4, 5},
        sizeof(Nvfp4QMatmulPushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 1, 2, 3},
        sizeof(LayerNormAffinePushConstants),
        DispatchGridKind::Linear1D),
    make_kernel_spec(
        {0, 2},
        sizeof(ArgsortPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1, 2},
        sizeof(ArgsortPushConstants),
        DispatchGridKind::RowWise),
    make_kernel_spec(
        {0, 1},
        sizeof(FFTPushConstants),
        DispatchGridKind::Linear1D),
};

size_t kernel_spec_index(KernelSpecId id) {
  return static_cast<size_t>(id);
}

const KernelSpec& get_kernel_spec(KernelSpecId id) {
  return kKernelSpecs[kernel_spec_index(id)];
}

KernelSpecId kernel_spec_id_for_binary_variant(BinaryDispatchVariant variant) {
  switch (variant) {
    case BinaryDispatchVariant::Standard:
      return KernelSpecId::Binary;
    case BinaryDispatchVariant::AddWithPartials:
      return KernelSpecId::BinaryAddWithPartials;
  }

  throw std::runtime_error(
      "[vulkan::kernels] Unsupported binary dispatch variant.");
}

std::vector<VkDescriptorSetLayoutBinding> make_layout_bindings(
    const KernelSpec& spec) {
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  bindings.reserve(spec.binding_count);
  for (size_t i = 0; i < spec.binding_count; ++i) {
    bindings.push_back(make_storage_buffer_binding(spec.bindings[i]));
  }
  return bindings;
}

std::tuple<uint32_t, uint32_t, uint32_t> get_dispatch_grid_dims(
    DispatchGridKind grid_kind,
    uint32_t num_elements) {
  switch (grid_kind) {
    case DispatchGridKind::ElementWise:
      return get_element_wise_grid_dims(num_elements, VULKAN_INDEX_TILE_SIZE);
    case DispatchGridKind::Linear1D:
      return {
          (num_elements + VULKAN_INDEX_TILE_SIZE - 1) / VULKAN_INDEX_TILE_SIZE,
          1,
          1};
    case DispatchGridKind::RowWise: {
      if (num_elements == 0) {
        return {0, 0, 0};
      }

      // Row-wise kernels index rows as:
      //   row = wg.z * (512 * 512) + wg.y * 512 + wg.x
      // so dispatch must tile rows across x/y/z in that order.
      const uint64_t rows = num_elements;
      const uint32_t x = static_cast<uint32_t>(std::min<uint64_t>(rows, 512));
      const uint64_t yz_tiles = (rows + 511) / 512;
      const uint32_t y =
          static_cast<uint32_t>(std::min<uint64_t>(yz_tiles, 512));
      const uint32_t z = static_cast<uint32_t>((yz_tiles + 511) / 512);
      return {x, y, z};
    }
  }

  throw std::runtime_error("[vulkan::kernels] Unsupported dispatch grid kind.");
}

BinaryPushConstants make_binary_push_constants(
    const array& a,
    const array& b,
    const array& out,
    float param1 = 0.0f,
    float param2 = 0.0f,
    int32_t param3 = 0) {
  const auto a_layout = make_tensor_layout_4d(a, "src0");
  const auto b_layout = make_tensor_layout_4d(b, "src1");
  const auto d_layout = make_tensor_layout_4d(out, "dst");

  BinaryPushConstants push_constants{};
  push_constants.ne = checked_u32(out.size(), "binary element count");
  push_constants.ne00 = a_layout.ne00;
  push_constants.ne01 = a_layout.ne01;
  push_constants.ne02 = a_layout.ne02;
  push_constants.ne03 = a_layout.ne03;
  push_constants.nb00 = a_layout.nb00;
  push_constants.nb01 = a_layout.nb01;
  push_constants.nb02 = a_layout.nb02;
  push_constants.nb03 = a_layout.nb03;
  push_constants.ne10 = b_layout.ne00;
  push_constants.ne11 = b_layout.ne01;
  push_constants.ne12 = b_layout.ne02;
  push_constants.ne13 = b_layout.ne03;
  push_constants.nb10 = b_layout.nb00;
  push_constants.nb11 = b_layout.nb01;
  push_constants.nb12 = b_layout.nb02;
  push_constants.nb13 = b_layout.nb03;
  push_constants.ne20 = d_layout.ne00;
  push_constants.ne21 = d_layout.ne01;
  push_constants.ne22 = d_layout.ne02;
  push_constants.ne23 = d_layout.ne03;
  push_constants.nb20 = d_layout.nb00;
  push_constants.nb21 = d_layout.nb01;
  push_constants.nb22 = d_layout.nb02;
  push_constants.nb23 = d_layout.nb03;
  const uint32_t a_offset = checked_offset(a, "src0", 0xFFFFu);
  const uint32_t b_offset = checked_offset(b, "src1", 0xFFu);
  const uint32_t d_offset = checked_offset(out, "dst", 0xFFu);
  push_constants.misalign_offsets =
      (a_offset << 16) | (b_offset << 8) | d_offset;
  push_constants.param1 = param1;
  push_constants.param2 = param2;
  push_constants.param3 = param3;

  return push_constants;
}

UnaryPushConstants make_unary_push_constants(
    const array& in,
    const array& out,
    float param1,
    float param2) {
  const auto in_layout = make_tensor_layout_4d(in, "src0");
  const auto out_layout = make_tensor_layout_4d(out, "dst");

  UnaryPushConstants push_constants{};
  push_constants.ne = checked_u32(out.size(), "unary element count");
  push_constants.ne00 = in_layout.ne00;
  push_constants.ne01 = in_layout.ne01;
  push_constants.ne02 = in_layout.ne02;
  push_constants.ne03 = in_layout.ne03;
  push_constants.nb00 = in_layout.nb00;
  push_constants.nb01 = in_layout.nb01;
  push_constants.nb02 = in_layout.nb02;
  push_constants.nb03 = in_layout.nb03;
  push_constants.ne10 = out_layout.ne00;
  push_constants.ne11 = out_layout.ne01;
  push_constants.ne12 = out_layout.ne02;
  push_constants.ne13 = out_layout.ne03;
  push_constants.nb10 = out_layout.nb00;
  push_constants.nb11 = out_layout.nb01;
  push_constants.nb12 = out_layout.nb02;
  push_constants.nb13 = out_layout.nb03;
  const uint32_t a_offset = checked_offset(in, "src0", 0xFFFFu);
  const uint32_t d_offset = checked_offset(out, "dst", 0xFFFFu);
  push_constants.misalign_offsets = (a_offset << 16) | d_offset;
  push_constants.param1 = param1;
  push_constants.param2 = param2;
  init_unary_fastdiv(push_constants);

  return push_constants;
}

GenericPushConstants make_generic_push_constants(
    uint32_t kx,
    float param1,
    float param2,
    float param3,
    float param4) {
  GenericPushConstants push_constants{};
  push_constants.KX = kx;
  push_constants.KY = 1;
  push_constants.param1 = param1;
  push_constants.param2 = param2;
  push_constants.param3 = param3;
  push_constants.param4 = param4;
  return push_constants;
}

template <typename T>
ArangePushConstants
make_arange_push_constants_t(uint32_t num_elements, double start, double step) {
  const T start_t = static_cast<T>(start);
  const T next_t = static_cast<T>(start + step);
  const T step_t = static_cast<T>(next_t - start_t);

  ArangePushConstants push_constants{};
  push_constants.KX = num_elements;
  push_constants.KY = 1;
  push_constants.start_i64 = static_cast<int64_t>(start_t);
  push_constants.step_i64 = static_cast<int64_t>(step_t);
  push_constants.start_f32 = static_cast<float>(start_t);
  push_constants.step_f32 = static_cast<float>(step_t);
  return push_constants;
}

ArangePushConstants make_arange_push_constants(
    const array& out,
    uint32_t num_elements,
    double start,
    double step) {
  switch (out.dtype()) {
    case uint8:
      return make_arange_push_constants_t<uint8_t>(num_elements, start, step);
    case uint16:
      return make_arange_push_constants_t<uint16_t>(num_elements, start, step);
    case uint32:
      return make_arange_push_constants_t<uint32_t>(num_elements, start, step);
    case int8:
      return make_arange_push_constants_t<int8_t>(num_elements, start, step);
    case int16:
      return make_arange_push_constants_t<int16_t>(num_elements, start, step);
    case int32:
      return make_arange_push_constants_t<int32_t>(num_elements, start, step);
    case int64:
      return make_arange_push_constants_t<int64_t>(num_elements, start, step);
    case float16:
      return make_arange_push_constants_t<float16_t>(num_elements, start, step);
    case bfloat16:
      return make_arange_push_constants_t<bfloat16_t>(
          num_elements, start, step);
    case float32:
      return make_arange_push_constants_t<float>(num_elements, start, step);
    default:
      throw std::runtime_error("[vulkan::kernels] Unsupported arange dtype.");
  }
}

ArangePushConstants make_fill_push_constants(
    uint32_t num_elements,
    float value) {
  ArangePushConstants push_constants{};
  push_constants.KX = num_elements;
  push_constants.KY = 1;
  push_constants.start_f32 = value;
  return push_constants;
}

SumRowsPushConstants
make_sum_rows_push_constants(const array& in, const array& out, float weight) {
  const auto in_layout = make_tensor_layout_4d(in, "src0");
  const auto out_layout = make_tensor_layout_4d(out, "dst");

  SumRowsPushConstants push_constants{};
  push_constants.n_cols = in_layout.ne00;
  push_constants.ne01 = in_layout.ne01;
  push_constants.ne02 = in_layout.ne02;
  push_constants.nb01 = in_layout.nb01;
  push_constants.nb02 = in_layout.nb02;
  push_constants.nb03 = in_layout.nb03;
  push_constants.nb11 = out_layout.nb01;
  push_constants.nb12 = out_layout.nb02;
  push_constants.nb13 = out_layout.nb03;
  push_constants.weight = weight;

  const uint32_t a_offset = checked_offset(in, "src0", 0xFFFFu);
  const uint32_t d_offset = checked_offset(out, "dst", 0xFFFFu);
  push_constants.misalign_offsets = (a_offset << 16) | d_offset;

  const uint32_t ne0_12 = checked_mul_u32(
      push_constants.ne01, push_constants.ne02, "sum_rows ne01*ne02");
  init_fastdiv_values(ne0_12, push_constants.ne0_12mp, push_constants.ne0_12L);
  init_fastdiv_values(
      push_constants.ne01, push_constants.ne0_1mp, push_constants.ne0_1L);

  return push_constants;
}

uint32_t select_softmax_block_size(uint32_t row_width) {
  if (row_width > 1024u) {
    return 512u;
  }

  const auto& context = VulkanContext::get();
  const uint32_t subgroup_size = std::max(context.subgroup_size(), 1u);

  switch (context.vendor_id()) {
    case kVendorIdAmd:
      if (context.subgroup_size_control_supported() &&
          context.subgroup_min_size() <= 64u &&
          context.subgroup_max_size() >= 64u) {
        return 64u;
      }
      return std::min(64u, subgroup_size);
    case kVendorIdNvidia:
      return 32u;
    case kVendorIdIntel:
      return row_width <= 64u ? 8u : 16u;
    default:
      return std::min(32u, subgroup_size);
  }
}

SoftmaxPushConstants make_softmax_push_constants(
    const array& /*in*/,
    uint32_t row_width,
    uint32_t row_count) {
  SoftmaxPushConstants push_constants{};
  push_constants.KX = row_width;
  push_constants.KY = 0;
  push_constants.ne00 = row_width;
  push_constants.ne01 = 1;
  push_constants.ne02 = 1;
  push_constants.ne12 = 1;
  push_constants.ne13 = 1;
  push_constants.nb11 = 0;
  push_constants.nb12 = 0;
  push_constants.nb13 = 0;
  push_constants.scale = 1.0f;
  push_constants.max_bias = 0.0f;
  push_constants.m0 = 0.0f;
  push_constants.m1 = 0.0f;
  push_constants.n_head_log2 = 0;
  push_constants.nrows_x = row_count;
  push_constants.has_sinks = 0;

  // Plain Vulkan softmax indexes rows linearly from row_count and only uses the
  // last dimension width. When the logical tensor rank is greater than 4 we can
  // safely treat the leading dimensions as already collapsed into row_count.

  return push_constants;
}

template <typename PushConstants, size_t N>
void dispatch_with_spec(
    StaticShaderId shader_id,
    KernelSpecId spec_id,
    const std::array<BoundArray, N>& bound_arrays,
    const PushConstants& push_constants,
    uint32_t num_elements,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    std::optional<std::array<uint32_t, 3>> explicit_grid = std::nullopt,
    const std::vector<uint32_t>& specialization_constants = {}) {
  if (num_elements == 0) {
    return;
  }

  const auto& spec = get_kernel_spec(spec_id);
  if (bound_arrays.size() != spec.binding_count) {
    throw std::runtime_error(
        "[vulkan::kernels] Kernel bindings do not match registered "
        "KernelSpec.");
  }
  if (sizeof(PushConstants) != spec.push_constant_size) {
    throw std::runtime_error(
        "[vulkan::kernels] Push constant size does not match registered "
        "KernelSpec.");
  }

  auto bindings = make_layout_bindings(spec);

  auto& manager = KernelManager::get();
  auto* pipeline = manager.get_pipeline(
      shader_id,
      bindings,
      static_cast<uint32_t>(sizeof(PushConstants)),
      specialization_constants);

  // Prepare descriptor writes (used for both push descriptor and traditional
  // path)
  std::vector<VkDescriptorBufferInfo> descriptor_infos(bound_arrays.size());
  std::vector<VkWriteDescriptorSet> descriptor_writes(bound_arrays.size());

  for (size_t i = 0; i < bound_arrays.size(); ++i) {
    if (bound_arrays[i].arr == nullptr) {
      throw std::runtime_error("[vulkan::kernels] Missing bound array.");
    }

    retain_array_for_stream(s, *bound_arrays[i].arr);

    descriptor_infos[i] =
        make_buffer_info(*bound_arrays[i].arr, bound_arrays[i].name);
    auto& write = descriptor_writes[i];
    write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstBinding = spec.bindings[i];
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &descriptor_infos[i];
  }

  vkCmdBindPipeline(
      cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);

  if (pipeline->supports_push_descriptor) {
    // Use push descriptors: avoid allocation, update, and bind
    // vkCmdPushDescriptorSetKHR combines update and bind in one call
    auto push_fn = VulkanContext::get().push_descriptor_fn();
    if (push_fn != nullptr) {
      push_fn(
          cmd_buffer,
          VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline->layout,
          0, // set index
          static_cast<uint32_t>(bound_arrays.size()),
          descriptor_writes.data());
    }
  } else {
    // Traditional path: allocate, update, and bind descriptor set
    VkDescriptorSet descriptor_set =
        manager.allocate_descriptor_set(pipeline->descriptor_layout);
    const uint64_t descriptor_epoch = descriptor_epoch_for_stream(s);
    manager.defer_descriptor_set_free(
        s.index, descriptor_epoch, descriptor_set);
    if (trace_descriptor_epochs_enabled()) {
      std::ostringstream oss;
      oss << "defer stream=" << s.index << " epoch=" << descriptor_epoch
          << " set=0x" << std::hex
          << reinterpret_cast<uintptr_t>(descriptor_set) << std::dec;
      trace_descriptor_epochs(oss.str());
    }

    // Set descriptor set handle in writes
    for (size_t i = 0; i < bound_arrays.size(); ++i) {
      descriptor_writes[i].dstSet = descriptor_set;
    }

    vkUpdateDescriptorSets(
        VulkanContext::get().device(),
        static_cast<uint32_t>(bound_arrays.size()),
        descriptor_writes.data(),
        0,
        nullptr);

    vkCmdBindDescriptorSets(
        cmd_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline->layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
  }

  vkCmdPushConstants(
      cmd_buffer,
      pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(sizeof(PushConstants)),
      &push_constants);

  uint32_t grid_x = 0;
  uint32_t grid_y = 0;
  uint32_t grid_z = 0;
  if (explicit_grid.has_value()) {
    grid_x = (*explicit_grid)[0];
    grid_y = (*explicit_grid)[1];
    grid_z = (*explicit_grid)[2];
  } else {
    auto dims = get_dispatch_grid_dims(spec.grid_kind, num_elements);
    grid_x = std::get<0>(dims);
    grid_y = std::get<1>(dims);
    grid_z = std::get<2>(dims);
  }
  vkCmdDispatch(cmd_buffer, grid_x, grid_y, grid_z);
}
} // namespace

ShaderModule::~ShaderModule() {
  if (module != VK_NULL_HANDLE) {
    vk::Device device = VulkanContext::get().device();
    device.destroyShaderModule(module, nullptr);
  }
}

ComputePipeline::~ComputePipeline() {
  vk::Device device = VulkanContext::get().device();
  if (pipeline != VK_NULL_HANDLE) {
    device.destroyPipeline(pipeline, nullptr);
  }
  if (layout != VK_NULL_HANDLE) {
    device.destroyPipelineLayout(layout, nullptr);
  }
  if (descriptor_layout != VK_NULL_HANDLE) {
    device.destroyDescriptorSetLayout(descriptor_layout, nullptr);
  }
}

KernelManager::KernelManager() : static_shaders_(kStaticShaderCount) {}

KernelManager::~KernelManager() {
  cleanup();
}

KernelManager& KernelManager::get() {
  static auto* manager = new KernelManager();
  return *manager;
}

void KernelManager::initialize_static_registry() {
  ensure_static_registry_initialized();
}

size_t KernelManager::PipelineKeyHash::operator()(
    const PipelineKey& key) const {
  size_t seed = 0;
  hash_combine(seed, key.is_dynamic);
  hash_combine(seed, static_cast<uint32_t>(key.static_shader_id));
  hash_combine(seed, key.dynamic_shader_name);
  hash_combine(seed, key.push_constant_size);
  for (const auto& binding : key.bindings) {
    hash_combine(seed, binding.binding);
    hash_combine(seed, binding.descriptor_type);
    hash_combine(seed, binding.descriptor_count);
    hash_combine(seed, binding.stage_flags);
    hash_combine(seed, binding.has_immutable_samplers);
  }
  for (uint32_t value : key.specialization_constants) {
    hash_combine(seed, value);
  }
  return seed;
}

KernelManager::DescriptorBindingKey KernelManager::make_descriptor_binding_key(
    const vk::DescriptorSetLayoutBinding& binding) {
  return {
      binding.binding,
      static_cast<uint32_t>(binding.descriptorType),
      binding.descriptorCount,
      static_cast<VkShaderStageFlags>(binding.stageFlags),
      binding.pImmutableSamplers != nullptr,
  };
}

void KernelManager::register_static_shader(
    StaticShaderId id,
    const void* data,
    size_t size_bytes) {
  const size_t index = static_cast<size_t>(id);
  auto& shader = static_shaders_[index];
  if (!shader) {
    shader = std::make_unique<ShaderModule>();
  } else if (shader->module != VK_NULL_HANDLE) {
    vkDestroyShaderModule(
        VulkanContext::get().device(), shader->module, nullptr);
  }
  shader->debug_name = static_shader_name(id);
  shader->spirv_code = copy_spirv_code(data, size_bytes);
  shader->compiled = false;
  shader->module = VK_NULL_HANDLE;
}

void KernelManager::register_shader(
    const std::string& name,
    const void* data,
    size_t size_bytes) {
  std::scoped_lock lock(shader_cache_mutex_, pipeline_cache_mutex_);

  auto& shader = dynamic_shaders_[name];
  if (!shader) {
    shader = std::make_unique<ShaderModule>();
  } else if (shader->module != VK_NULL_HANDLE) {
    vkDestroyShaderModule(
        VulkanContext::get().device(), shader->module, nullptr);
  }

  std::unordered_set<vk::DescriptorSetLayout, VulkanHandleHash> evicted_layouts;
  for (const auto& [key, pipeline] : pipelines_) {
    if (!key.is_dynamic || key.dynamic_shader_name != name ||
        pipeline == nullptr || !pipeline->descriptor_layout) {
      continue;
    }
    evicted_layouts.insert(pipeline->descriptor_layout);
  }
  purge_descriptor_sets_for_layouts(evicted_layouts);

  for (auto it = pipelines_.begin(); it != pipelines_.end();) {
    if (it->first.is_dynamic && it->first.dynamic_shader_name == name) {
      it = pipelines_.erase(it);
    } else {
      ++it;
    }
  }

  shader->debug_name = name;
  shader->spirv_code = copy_spirv_code(data, size_bytes);
  shader->compiled = false;
  shader->module = VK_NULL_HANDLE;
}

void KernelManager::ensure_static_registry_initialized() {
  if (static_registry_initialized_) {
    return;
  }

  std::lock_guard<std::mutex> lock(static_registry_mutex_);
  if (static_registry_initialized_) {
    return;
  }

  const auto registry = make_static_shader_registry();
  for (const auto& record : registry) {
    register_static_shader(record.id, record.data, record.len);
  }
  static_registry_initialized_ = true;
}

vk::ShaderModule KernelManager::compile_shader(
    const std::vector<uint32_t>& spirv) {
  vk::Device device = VulkanContext::get().device();

  vk::ShaderModuleCreateInfo createInfo;
  createInfo.setCode(spirv);

  try {
    return device.createShaderModule(createInfo);
  } catch (const vk::SystemError& e) {
    throw std::runtime_error(
        "Failed to create shader module: " + std::string(e.what()));
  }
}

ShaderModule* KernelManager::get_shader(StaticShaderId id) {
  ensure_static_registry_initialized();

  std::lock_guard<std::mutex> lock(shader_cache_mutex_);
  const size_t index = static_cast<size_t>(id);
  if (index >= static_shaders_.size()) {
    return nullptr;
  }

  auto* shader = static_shaders_[index].get();
  if (shader == nullptr) {
    return nullptr;
  }
  if (!shader->compiled) {
    shader->module = compile_shader(shader->spirv_code);
    shader->compiled = true;
  }

  return shader;
}

mlx::core::vulkan::ShaderModule* mlx::core::vulkan::KernelManager::get_shader(
    const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(shader_cache_mutex_);
    auto it = dynamic_shaders_.find(name);
    if (it != dynamic_shaders_.end()) {
      auto* shader = it->second.get();
      if (!shader->compiled) {
        shader->module = compile_shader(shader->spirv_code);
        shader->compiled = true;
      }

      return shader;
    }
  }

  ensure_static_registry_initialized();
  const auto shader_id = static_shader_id_by_name(name);
  if (!shader_id.has_value()) {
    return nullptr;
  }

  return get_shader(*shader_id);
}

ComputePipeline* KernelManager::get_pipeline(
    StaticShaderId shader_id,
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
    uint32_t push_constant_size,
    const std::vector<uint32_t>& specialization_constants) {
  // Convert C++ bindings to C bindings for internal use
  std::vector<VkDescriptorSetLayoutBinding> c_bindings;
  for (const auto& binding : bindings) {
    VkDescriptorSetLayoutBinding c_binding{};
    c_binding.binding = binding.binding;
    c_binding.descriptorType =
        static_cast<VkDescriptorType>(binding.descriptorType);
    c_binding.descriptorCount = binding.descriptorCount;
    c_binding.stageFlags = static_cast<VkShaderStageFlags>(binding.stageFlags);
    c_binding.pImmutableSamplers = binding.pImmutableSamplers
        ? reinterpret_cast<const VkSampler*>(binding.pImmutableSamplers)
        : nullptr;
    c_bindings.push_back(c_binding);
  }

  return get_pipeline(
      shader_id, c_bindings, push_constant_size, specialization_constants);
}

ComputePipeline* KernelManager::get_pipeline(
    StaticShaderId shader_id,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t push_constant_size,
    const std::vector<uint32_t>& specialization_constants) {
  ensure_static_registry_initialized();

  PipelineKey pipeline_key;
  pipeline_key.is_dynamic = false;
  pipeline_key.static_shader_id = shader_id;
  pipeline_key.push_constant_size = push_constant_size;
  pipeline_key.specialization_constants = specialization_constants;
  pipeline_key.bindings.reserve(bindings.size());
  for (const auto& binding : bindings) {
    pipeline_key.bindings.push_back(make_descriptor_binding_key(binding));
  }

  {
    std::lock_guard<std::mutex> lock(pipeline_cache_mutex_);
    auto it = pipelines_.find(pipeline_key);
    if (it != pipelines_.end()) {
      return it->second.get();
    }
  }

  VkDevice device = VulkanContext::get().device();
  const auto pipeline_options =
      pipeline_creation_options(shader_id, specialization_constants);
  ShaderModule* shader = get_shader(shader_id);
  if (!shader) {
    throw std::runtime_error(
        std::string("Shader not found: ") + static_shader_name(shader_id));
  }

  const bool use_push_descriptor =
      VulkanContext::get().push_descriptor_supported();
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  if (!bindings.empty()) {
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (use_push_descriptor) {
      layoutInfo.flags =
          VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }

    if (vkCreateDescriptorSetLayout(
            device, &layoutInfo, nullptr, &descriptor_layout) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor set layout");
    }
  }

  VkPipelineLayout pipeline_layout;
  VkPushConstantRange push_constant_range{};

  if (push_constant_size > 0) {
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = push_constant_size;
  }

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  if (descriptor_layout != VK_NULL_HANDLE) {
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_layout;
  }
  if (push_constant_size > 0) {
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push_constant_range;
  }

  if (vkCreatePipelineLayout(
          device, &pipelineLayoutInfo, nullptr, &pipeline_layout) !=
      VK_SUCCESS) {
    if (descriptor_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    }
    throw std::runtime_error("Failed to create pipeline layout");
  }

  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  if (VulkanContext::get().subgroup_require_full_support() &&
      pipeline_options.require_full_subgroups) {
    pipelineInfo.stage.flags =
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
  }
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shader->module;
  pipelineInfo.stage.pName = "main";

  std::vector<VkSpecializationMapEntry> specialization_entries;
  VkSpecializationInfo specialization_info{};
  if (!specialization_constants.empty()) {
    specialization_entries.reserve(specialization_constants.size());
    for (uint32_t i = 0; i < specialization_constants.size(); ++i) {
      VkSpecializationMapEntry entry{};
      entry.constantID = i;
      entry.offset = i * sizeof(uint32_t);
      entry.size = sizeof(uint32_t);
      specialization_entries.push_back(entry);
    }
    specialization_info.mapEntryCount =
        static_cast<uint32_t>(specialization_entries.size());
    specialization_info.pMapEntries = specialization_entries.data();
    specialization_info.dataSize =
        specialization_constants.size() * sizeof(uint32_t);
    specialization_info.pData = specialization_constants.data();
    pipelineInfo.stage.pSpecializationInfo = &specialization_info;
  }

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_info{};
  if (VulkanContext::get().subgroup_size_control_supported() &&
      pipeline_options.required_subgroup_size >=
          VulkanContext::get().subgroup_min_size() &&
      pipeline_options.required_subgroup_size <=
          VulkanContext::get().subgroup_max_size() &&
      pipeline_options.required_subgroup_size > 0) {
    subgroup_size_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup_size_info.requiredSubgroupSize =
        pipeline_options.required_subgroup_size;
    pipelineInfo.stage.pNext = &subgroup_size_info;
  }

  pipelineInfo.layout = pipeline_layout;

  VkPipelineRobustnessCreateInfoEXT pipeline_robustness_info{};
  if (VulkanContext::get().pipeline_robustness_supported() &&
      pipeline_options.disable_robustness) {
    pipeline_robustness_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT;
    pipeline_robustness_info.storageBuffers =
        VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT;
    pipeline_robustness_info.uniformBuffers =
        VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT;
    pipelineInfo.pNext = &pipeline_robustness_info;
  }

  VkPipeline pipeline;
  if (vkCreateComputePipelines(
          device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) !=
      VK_SUCCESS) {
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (descriptor_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    }
    throw std::runtime_error("Failed to create compute pipeline");
  }

  auto pipeline_ptr = std::make_unique<ComputePipeline>();
  pipeline_ptr->pipeline = pipeline;
  pipeline_ptr->layout = pipeline_layout;
  pipeline_ptr->descriptor_layout = descriptor_layout;
  pipeline_ptr->push_constant_size = push_constant_size;
  pipeline_ptr->supports_push_descriptor = use_push_descriptor;

  auto* result = pipeline_ptr.get();
  {
    std::lock_guard<std::mutex> lock(pipeline_cache_mutex_);
    auto it = pipelines_.find(pipeline_key);
    if (it != pipelines_.end()) {
      vkDestroyPipeline(device, pipeline, nullptr);
      vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      if (descriptor_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
      }
      return it->second.get();
    }
    pipelines_.emplace(std::move(pipeline_key), std::move(pipeline_ptr));
  }

  return result;
}

ComputePipeline* KernelManager::get_pipeline(
    const std::string& shader_name,
    const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
    uint32_t push_constant_size,
    const std::vector<uint32_t>& specialization_constants) {
  // Convert C++ bindings to C bindings for internal use
  std::vector<VkDescriptorSetLayoutBinding> c_bindings;
  for (const auto& binding : bindings) {
    VkDescriptorSetLayoutBinding c_binding{};
    c_binding.binding = binding.binding;
    c_binding.descriptorType =
        static_cast<VkDescriptorType>(binding.descriptorType);
    c_binding.descriptorCount = binding.descriptorCount;
    c_binding.stageFlags = static_cast<VkShaderStageFlags>(binding.stageFlags);
    c_binding.pImmutableSamplers = binding.pImmutableSamplers
        ? reinterpret_cast<const VkSampler*>(binding.pImmutableSamplers)
        : nullptr;
    c_bindings.push_back(c_binding);
  }

  return get_pipeline(
      shader_name, c_bindings, push_constant_size, specialization_constants);
}

ComputePipeline* KernelManager::get_pipeline(
    const std::string& shader_name,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t push_constant_size,
    const std::vector<uint32_t>& specialization_constants) {
  PipelineKey pipeline_key;
  pipeline_key.is_dynamic = true;
  pipeline_key.dynamic_shader_name = shader_name;
  pipeline_key.push_constant_size = push_constant_size;
  pipeline_key.specialization_constants = specialization_constants;
  pipeline_key.bindings.reserve(bindings.size());
  for (const auto& binding : bindings) {
    pipeline_key.bindings.push_back(make_descriptor_binding_key(binding));
  }

  {
    std::lock_guard<std::mutex> lock(pipeline_cache_mutex_);
    auto it = pipelines_.find(pipeline_key);
    if (it != pipelines_.end()) {
      return it->second.get();
    }
  }

  VkDevice device = VulkanContext::get().device();
  const auto pipeline_options =
      pipeline_creation_options(shader_name, specialization_constants);

  // Get or compile shader
  ShaderModule* shader = get_shader(shader_name);
  if (!shader) {
    throw std::runtime_error("Shader not found: " + shader_name);
  }

  // Create descriptor set layout
  const bool use_push_descriptor =
      VulkanContext::get().push_descriptor_supported();
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  if (!bindings.empty()) {
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (use_push_descriptor) {
      layoutInfo.flags =
          VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }

    if (vkCreateDescriptorSetLayout(
            device, &layoutInfo, nullptr, &descriptor_layout) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor set layout");
    }
  }

  // Create pipeline layout
  VkPipelineLayout pipeline_layout;
  VkPushConstantRange push_constant_range{};

  if (push_constant_size > 0) {
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = push_constant_size;
  }

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  if (descriptor_layout != VK_NULL_HANDLE) {
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_layout;
  }
  if (push_constant_size > 0) {
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &push_constant_range;
  }

  if (vkCreatePipelineLayout(
          device, &pipelineLayoutInfo, nullptr, &pipeline_layout) !=
      VK_SUCCESS) {
    if (descriptor_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    }
    throw std::runtime_error("Failed to create pipeline layout");
  }

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  if (VulkanContext::get().subgroup_require_full_support() &&
      pipeline_options.require_full_subgroups) {
    pipelineInfo.stage.flags =
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
  }
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shader->module;
  pipelineInfo.stage.pName = "main";

  std::vector<VkSpecializationMapEntry> specialization_entries;
  VkSpecializationInfo specialization_info{};
  if (!specialization_constants.empty()) {
    specialization_entries.reserve(specialization_constants.size());
    for (uint32_t i = 0; i < specialization_constants.size(); ++i) {
      VkSpecializationMapEntry entry{};
      entry.constantID = i;
      entry.offset = i * sizeof(uint32_t);
      entry.size = sizeof(uint32_t);
      specialization_entries.push_back(entry);
    }
    specialization_info.mapEntryCount =
        static_cast<uint32_t>(specialization_entries.size());
    specialization_info.pMapEntries = specialization_entries.data();
    specialization_info.dataSize =
        specialization_constants.size() * sizeof(uint32_t);
    specialization_info.pData = specialization_constants.data();
    pipelineInfo.stage.pSpecializationInfo = &specialization_info;
  }

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_info{};
  if (VulkanContext::get().subgroup_size_control_supported() &&
      pipeline_options.required_subgroup_size >=
          VulkanContext::get().subgroup_min_size() &&
      pipeline_options.required_subgroup_size <=
          VulkanContext::get().subgroup_max_size() &&
      pipeline_options.required_subgroup_size > 0) {
    subgroup_size_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroup_size_info.requiredSubgroupSize =
        pipeline_options.required_subgroup_size;
    pipelineInfo.stage.pNext = &subgroup_size_info;
  }

  pipelineInfo.layout = pipeline_layout;

  VkPipelineRobustnessCreateInfoEXT pipeline_robustness_info{};
  if (VulkanContext::get().pipeline_robustness_supported() &&
      pipeline_options.disable_robustness) {
    pipeline_robustness_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_ROBUSTNESS_CREATE_INFO_EXT;
    pipeline_robustness_info.storageBuffers =
        VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT;
    pipeline_robustness_info.uniformBuffers =
        VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT;
    pipelineInfo.pNext = &pipeline_robustness_info;
  }

  VkPipeline pipeline;
  if (vkCreateComputePipelines(
          device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) !=
      VK_SUCCESS) {
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    if (descriptor_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    }
    throw std::runtime_error("Failed to create compute pipeline");
  }

  auto pipeline_ptr = std::make_unique<ComputePipeline>();
  pipeline_ptr->pipeline = pipeline;
  pipeline_ptr->layout = pipeline_layout;
  pipeline_ptr->descriptor_layout = descriptor_layout;
  pipeline_ptr->push_constant_size = push_constant_size;
  pipeline_ptr->supports_push_descriptor = use_push_descriptor;

  auto* result = pipeline_ptr.get();
  {
    std::lock_guard<std::mutex> lock(pipeline_cache_mutex_);
    auto it = pipelines_.find(pipeline_key);
    if (it != pipelines_.end()) {
      vkDestroyPipeline(device, pipeline, nullptr);
      vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      if (descriptor_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
      }
      return it->second.get();
    }
    pipelines_.emplace(std::move(pipeline_key), std::move(pipeline_ptr));
  }

  return result;
}

void KernelManager::init_descriptor_pool() {
  std::lock_guard<std::mutex> lock(descriptor_pool_mutex_);
  if (descriptor_pool_initialized_) {
    return;
  }

  VkDevice device = VulkanContext::get().device();

  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 65536;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = 16384;
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  VkDescriptorPool c_pool;
  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &c_pool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create descriptor pool");
  }
  descriptor_pool_ = c_pool;

  descriptor_pool_initialized_ = true;
}

vk::DescriptorSet KernelManager::allocate_descriptor_set(
    vk::DescriptorSetLayout layout) {
  {
    std::lock_guard<std::mutex> lock(descriptor_sets_mutex_);
    auto reusable_it = reusable_descriptor_sets_.find(layout);
    if (reusable_it != reusable_descriptor_sets_.end() &&
        !reusable_it->second.empty()) {
      vk::DescriptorSet descriptor_set = reusable_it->second.back();
      reusable_it->second.pop_back();
      if (reusable_it->second.empty()) {
        reusable_descriptor_sets_.erase(reusable_it);
      }
      if (trace_descriptor_epochs_enabled()) {
        std::ostringstream oss;
        oss << "reuse layout=0x" << std::hex
            << reinterpret_cast<uintptr_t>(
                   static_cast<VkDescriptorSetLayout>(layout))
            << " set=0x"
            << reinterpret_cast<uintptr_t>(
                   static_cast<VkDescriptorSet>(descriptor_set))
            << std::dec;
        trace_descriptor_epochs(oss.str());
      }
      return descriptor_set;
    }
  }

  if (!descriptor_pool_initialized_) {
    init_descriptor_pool();
  }

  vk::Device device = VulkanContext::get().device();

  std::optional<vk::SystemError> last_error;
  for (uint32_t batch_size = kDescriptorSetBatchSize; batch_size >= 1;
       batch_size /= 2) {
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.setDescriptorPool(descriptor_pool_);
    std::vector<vk::DescriptorSetLayout> layouts(batch_size, layout);
    allocInfo.setSetLayouts(layouts);

    try {
      auto descriptor_sets = device.allocateDescriptorSets(allocInfo);
      if (descriptor_sets.empty()) {
        throw std::runtime_error("Failed to allocate descriptor set batch.");
      }

      vk::DescriptorSet descriptor_set = descriptor_sets.back();
      descriptor_sets.pop_back();

      {
        std::lock_guard<std::mutex> lock(descriptor_sets_mutex_);
        auto& reusable_sets = reusable_descriptor_sets_[layout];
        reusable_sets.insert(
            reusable_sets.end(),
            descriptor_sets.begin(),
            descriptor_sets.end());
        descriptor_set_layouts_[descriptor_set] = layout;
        for (const auto& reusable_set : descriptor_sets) {
          descriptor_set_layouts_[reusable_set] = layout;
        }
      }

      if (trace_descriptor_epochs_enabled()) {
        std::ostringstream oss;
        oss << "batch_allocate layout=0x" << std::hex
            << reinterpret_cast<uintptr_t>(
                   static_cast<VkDescriptorSetLayout>(layout))
            << " count=" << std::dec << (descriptor_sets.size() + 1)
            << " set=0x" << std::hex
            << reinterpret_cast<uintptr_t>(
                   static_cast<VkDescriptorSet>(descriptor_set))
            << std::dec;
        trace_descriptor_epochs(oss.str());
      }

      return descriptor_set;
    } catch (const vk::SystemError& e) {
      last_error = e;
      if (batch_size == 1) {
        break;
      }
    }
  }

  if (last_error.has_value()) {
    throw std::runtime_error(
        "Failed to allocate descriptor set: " +
        std::string(last_error->what()));
  }

  throw std::runtime_error("Failed to allocate descriptor set.");
}

void KernelManager::free_descriptor_set(VkDescriptorSet set) {
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    {
      std::lock_guard<std::mutex> lock(descriptor_sets_mutex_);
      descriptor_set_layouts_.erase(set);
    }
    VkDevice device = VulkanContext::get().device();
    vkFreeDescriptorSets(device, descriptor_pool_, 1, &set);
  }
}

void KernelManager::defer_descriptor_set_free(
    int stream_index,
    vk::DescriptorSet set) {
  defer_descriptor_set_free(stream_index, 0, set);
}

void KernelManager::defer_descriptor_set_free(
    int stream_index,
    uint64_t submission_epoch,
    vk::DescriptorSet set) {
  if (!set) {
    return;
  }
  vk::DescriptorSetLayout layout;
  {
    std::lock_guard<std::mutex> lock(descriptor_sets_mutex_);
    auto it = descriptor_set_layouts_.find(set);
    if (it != descriptor_set_layouts_.end()) {
      layout = it->second;
    }
  }
  std::lock_guard<std::mutex> lock(deferred_descriptor_sets_mutex_);
  deferred_descriptor_sets_[stream_index][submission_epoch].push_back(
      DescriptorSetRecord{set, layout});
  if (trace_descriptor_epochs_enabled()) {
    std::ostringstream oss;
    oss << "enqueue stream=" << stream_index << " epoch=" << submission_epoch
        << " queued="
        << deferred_descriptor_sets_[stream_index][submission_epoch].size();
    trace_descriptor_epochs(oss.str());
  }
}

// Backward compatibility overloads for C API
void KernelManager::defer_descriptor_set_free(
    int stream_index,
    VkDescriptorSet set) {
  defer_descriptor_set_free(stream_index, 0, set);
}

void KernelManager::defer_descriptor_set_free(
    int stream_index,
    uint64_t submission_epoch,
    VkDescriptorSet set) {
  defer_descriptor_set_free(
      stream_index, submission_epoch, vk::DescriptorSet(set));
}

void KernelManager::reclaim_descriptor_sets(int stream_index) {
  reclaim_descriptor_sets(stream_index, std::numeric_limits<uint64_t>::max());
}

void KernelManager::reclaim_descriptor_set_epoch(
    int stream_index,
    uint64_t submission_epoch) {
  std::vector<DescriptorSetRecord> sets;
  {
    std::lock_guard<std::mutex> lock(deferred_descriptor_sets_mutex_);
    auto it = deferred_descriptor_sets_.find(stream_index);
    if (it == deferred_descriptor_sets_.end()) {
      return;
    }

    auto& epoch_map = it->second;
    auto epoch_it = epoch_map.find(submission_epoch);
    if (epoch_it == epoch_map.end()) {
      return;
    }

    auto& epoch_sets = epoch_it->second;
    sets.insert(
        sets.end(),
        std::make_move_iterator(epoch_sets.begin()),
        std::make_move_iterator(epoch_sets.end()));
    epoch_map.erase(epoch_it);
    if (epoch_map.empty()) {
      deferred_descriptor_sets_.erase(it);
    }
  }

  if (sets.empty() || descriptor_pool_ == VK_NULL_HANDLE) {
    return;
  }

  size_t reusable_count = 0;
  size_t freed_count = 0;
  std::vector<VkDescriptorSet> freeable_sets;
  {
    std::lock_guard<std::mutex> lock(descriptor_sets_mutex_);
    for (const auto& record : sets) {
      if (record.set == VK_NULL_HANDLE) {
        continue;
      }
      if (record.layout != VK_NULL_HANDLE) {
        reusable_descriptor_sets_[record.layout].push_back(record.set);
        ++reusable_count;
      } else {
        descriptor_set_layouts_.erase(record.set);
        freeable_sets.push_back(record.set);
        ++freed_count;
      }
    }
  }
  if (!freeable_sets.empty()) {
    VkDevice device = VulkanContext::get().device();
    vkFreeDescriptorSets(
        device,
        descriptor_pool_,
        static_cast<uint32_t>(freeable_sets.size()),
        freeable_sets.data());
  }

  if (trace_descriptor_epochs_enabled()) {
    std::ostringstream oss;
    oss << "reclaim_epoch stream=" << stream_index
        << " epoch=" << submission_epoch << " reusable_sets=" << reusable_count
        << " freed_sets=" << freed_count;
    trace_descriptor_epochs(oss.str());
  }
}

void KernelManager::reclaim_descriptor_sets(
    int stream_index,
    uint64_t completed_epoch) {
  std::vector<DescriptorSetRecord> sets;
  size_t remaining_epochs = 0;
  {
    std::lock_guard<std::mutex> lock(deferred_descriptor_sets_mutex_);
    auto it = deferred_descriptor_sets_.find(stream_index);
    if (it == deferred_descriptor_sets_.end()) {
      return;
    }

    auto& epoch_map = it->second;
    for (auto epoch_it = epoch_map.begin(); epoch_it != epoch_map.end();) {
      if (epoch_it->first <= completed_epoch) {
        auto& epoch_sets = epoch_it->second;
        sets.insert(
            sets.end(),
            std::make_move_iterator(epoch_sets.begin()),
            std::make_move_iterator(epoch_sets.end()));
        epoch_it = epoch_map.erase(epoch_it);
      } else {
        ++epoch_it;
      }
    }
    if (epoch_map.empty()) {
      deferred_descriptor_sets_.erase(it);
      remaining_epochs = 0;
    } else {
      remaining_epochs = epoch_map.size();
    }
  }

  if (sets.empty() || descriptor_pool_ == VK_NULL_HANDLE) {
    return;
  }

  size_t reusable_count = 0;
  size_t freed_count = 0;
  std::vector<VkDescriptorSet> freeable_sets;
  {
    std::lock_guard<std::mutex> lock(descriptor_sets_mutex_);
    for (const auto& record : sets) {
      if (record.set == VK_NULL_HANDLE) {
        continue;
      }
      if (record.layout != VK_NULL_HANDLE) {
        reusable_descriptor_sets_[record.layout].push_back(record.set);
        ++reusable_count;
      } else {
        descriptor_set_layouts_.erase(record.set);
        freeable_sets.push_back(record.set);
        ++freed_count;
      }
    }
  }
  if (!freeable_sets.empty()) {
    VkDevice device = VulkanContext::get().device();
    vkFreeDescriptorSets(
        device,
        descriptor_pool_,
        static_cast<uint32_t>(freeable_sets.size()),
        freeable_sets.data());
  }

  if (trace_descriptor_epochs_enabled()) {
    std::ostringstream oss;
    oss << "reclaim stream=" << stream_index
        << " completed_epoch=" << completed_epoch
        << " reusable_sets=" << reusable_count << " freed_sets=" << freed_count
        << " remaining_epochs=" << remaining_epochs;
    trace_descriptor_epochs(oss.str());
  }
}

void KernelManager::reclaim_all_descriptor_sets() {
  std::vector<VkDescriptorSet> all_sets;
  {
    std::lock_guard<std::mutex> lock(deferred_descriptor_sets_mutex_);
    for (auto& [_, epoch_map] : deferred_descriptor_sets_) {
      for (auto& [_, sets] : epoch_map) {
        for (auto& record : sets) {
          if (record.set != VK_NULL_HANDLE) {
            all_sets.push_back(record.set);
          }
        }
      }
    }
    deferred_descriptor_sets_.clear();
  }

  {
    std::lock_guard<std::mutex> lock(descriptor_sets_mutex_);
    for (auto& [_, sets] : reusable_descriptor_sets_) {
      all_sets.insert(all_sets.end(), sets.begin(), sets.end());
    }
    reusable_descriptor_sets_.clear();
    descriptor_set_layouts_.clear();
  }

  if (all_sets.empty() || descriptor_pool_ == VK_NULL_HANDLE) {
    return;
  }

  VkDevice device = VulkanContext::get().device();
  vkFreeDescriptorSets(
      device,
      descriptor_pool_,
      static_cast<uint32_t>(all_sets.size()),
      all_sets.data());

  if (trace_descriptor_epochs_enabled()) {
    std::ostringstream oss;
    oss << "reclaim_all freed_sets=" << all_sets.size();
    trace_descriptor_epochs(oss.str());
  }
}

void KernelManager::purge_descriptor_sets_for_layouts(
    const std::unordered_set<vk::DescriptorSetLayout, VulkanHandleHash>&
        layouts) {
  if (layouts.empty()) {
    return;
  }

  std::vector<VkDescriptorSet> freeable_sets;
  {
    std::lock_guard<std::mutex> lock(deferred_descriptor_sets_mutex_);
    for (auto deferred_it = deferred_descriptor_sets_.begin();
         deferred_it != deferred_descriptor_sets_.end();) {
      auto& epoch_map = deferred_it->second;
      for (auto epoch_it = epoch_map.begin(); epoch_it != epoch_map.end();) {
        auto& records = epoch_it->second;
        for (auto record_it = records.begin(); record_it != records.end();) {
          if (record_it->layout != VK_NULL_HANDLE &&
              layouts.contains(record_it->layout)) {
            freeable_sets.push_back(record_it->set);
            record_it = records.erase(record_it);
          } else {
            ++record_it;
          }
        }
        if (records.empty()) {
          epoch_it = epoch_map.erase(epoch_it);
        } else {
          ++epoch_it;
        }
      }
      if (epoch_map.empty()) {
        deferred_it = deferred_descriptor_sets_.erase(deferred_it);
      } else {
        ++deferred_it;
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(descriptor_sets_mutex_);
    for (auto layout : layouts) {
      auto reusable_it = reusable_descriptor_sets_.find(layout);
      if (reusable_it != reusable_descriptor_sets_.end()) {
        freeable_sets.insert(
            freeable_sets.end(),
            reusable_it->second.begin(),
            reusable_it->second.end());
        reusable_descriptor_sets_.erase(reusable_it);
      }
    }

    for (auto it = descriptor_set_layouts_.begin();
         it != descriptor_set_layouts_.end();) {
      if (layouts.contains(it->second)) {
        it = descriptor_set_layouts_.erase(it);
      } else {
        ++it;
      }
    }
  }

  if (freeable_sets.empty() || descriptor_pool_ == VK_NULL_HANDLE) {
    return;
  }

  VkDevice device = VulkanContext::get().device();
  vkFreeDescriptorSets(
      device,
      descriptor_pool_,
      static_cast<uint32_t>(freeable_sets.size()),
      freeable_sets.data());
}

void KernelManager::cleanup() {
  reclaim_all_descriptor_sets();
  {
    std::lock_guard<std::mutex> pipeline_lock(pipeline_cache_mutex_);
    pipelines_.clear();
  }
  {
    std::lock_guard<std::mutex> shader_lock(shader_cache_mutex_);
    dynamic_shaders_.clear();
    for (auto& shader : static_shaders_) {
      shader.reset();
    }
  }
  {
    std::lock_guard<std::mutex> lock(static_registry_mutex_);
    static_registry_initialized_ = false;
  }

  {
    std::lock_guard<std::mutex> lock(descriptor_pool_mutex_);
    if (descriptor_pool_ != VK_NULL_HANDLE) {
      VkDevice device = VulkanContext::get().device();
      vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
      descriptor_pool_ = VK_NULL_HANDLE;
      descriptor_pool_initialized_ = false;
    }
  }
}

std::tuple<uint32_t, uint32_t, uint32_t> get_element_wise_grid_dims(
    size_t num_elements,
    uint32_t tile_size) {
  if (num_elements == 0) {
    return {0, 0, 0};
  }

  const uint64_t tiles =
      (static_cast<uint64_t>(num_elements) + tile_size - 1) / tile_size;
  if (tiles >
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) * 512ULL) {
    throw std::runtime_error(
        "[vulkan::kernels] Elementwise dispatch exceeds Vulkan limits.");
  }

  const uint32_t x = 1;
  const uint32_t y = static_cast<uint32_t>(std::min<uint64_t>(tiles, 512));
  const uint32_t z = static_cast<uint32_t>((tiles + 511) / 512);
  return {x, y, z};
}

void dispatch_binary_op(
    const array& a,
    const array& b,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    BinaryDispatchVariant variant,
    std::optional<std::array<uint32_t, 3>> explicit_grid,
    const std::vector<uint32_t>& specialization_constants) {
  const auto push_constants = make_binary_push_constants(a, b, out);
  const auto spec_id = kernel_spec_id_for_binary_variant(variant);

  if (variant == BinaryDispatchVariant::AddWithPartials) {
    const std::array<BoundArray, 4> bound_arrays = {{
        {&a, "src0"},
        {&b, "src1"},
        {&out, "dst"},
        {&out, "partial"},
    }};
    dispatch_with_spec(
        shader_id,
        spec_id,
        bound_arrays,
        push_constants,
        push_constants.ne,
        cmd_buffer,
        s,
        explicit_grid,
        specialization_constants);
    return;
  }

  const std::array<BoundArray, 3> bound_arrays = {{
      {&a, "src0"},
      {&b, "src1"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      spec_id,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s,
      explicit_grid,
      specialization_constants);
}

void dispatch_binary_op(
    const array& a,
    const array& b,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    BinaryDispatchVariant variant,
    std::optional<std::array<uint32_t, 3>> explicit_grid,
    const std::vector<uint32_t>& specialization_constants,
    float param1,
    float param2,
    int32_t param3) {
  const auto push_constants =
      make_binary_push_constants(a, b, out, param1, param2, param3);
  const auto spec_id = kernel_spec_id_for_binary_variant(variant);

  if (variant == BinaryDispatchVariant::AddWithPartials) {
    const std::array<BoundArray, 4> bound_arrays = {{
        {&a, "src0"},
        {&b, "src1"},
        {&out, "dst"},
        {&out, "partial"},
    }};
    dispatch_with_spec(
        shader_id,
        spec_id,
        bound_arrays,
        push_constants,
        push_constants.ne,
        cmd_buffer,
        s,
        explicit_grid,
        specialization_constants);
    return;
  }

  const std::array<BoundArray, 3> bound_arrays = {{
      {&a, "src0"},
      {&b, "src1"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      spec_id,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s,
      explicit_grid,
      specialization_constants);
}

void dispatch_ternary_op(
    const array& cond,
    const array& x,
    const array& y,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s) {
  if (cond.shape() != out.shape() || x.shape() != out.shape() ||
      y.shape() != out.shape()) {
    throw std::runtime_error(
        "[vulkan::kernels] Ternary dispatch received incompatible shapes.");
  }

  TernaryPushConstants push_constants{};
  push_constants.ne = checked_u32(out.size(), "ternary elements");

  const std::array<BoundArray, 4> bound_arrays = {{
      {&cond, "src0"},
      {&x, "src1"},
      {&y, "src2"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Ternary,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_unary_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float param1,
    float param2) {
  const auto push_constants =
      make_unary_push_constants(in, out, param1, param2);
  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Unary,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_norm_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float eps) {
  const uint32_t axis_size = checked_u32(in.shape(in.ndim() - 1), "norm axis");
  const uint32_t row_count =
      axis_size == 0 ? 0 : checked_u32(out.size() / axis_size, "norm rows");
  const auto push_constants =
      make_generic_push_constants(axis_size, eps, 0.0f, 0.0f, 0.0f);
  const uint32_t grid_x = std::min<uint32_t>(row_count, 512u);
  const uint32_t grid_y = std::min<uint32_t>((row_count + 511u) / 512u, 512u);
  const uint32_t grid_z = (row_count + 262144u - 1u) / 262144u;
  const std::array<uint32_t, 3> grid = {grid_x, grid_y, grid_z};
  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GenericUnary,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      grid);
}

void dispatch_layer_norm_affine_op(
    const array& x,
    const array& weight,
    const array& bias,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const LayerNormAffinePushConstants& push_constants) {
  const std::array<BoundArray, 4> bound_arrays = {{
      {&x, "src0"},
      {&weight, "src1"},
      {&bias, "src2"},
      {&out, "dst"},
  }};
  const std::array<uint32_t, 3> grid = {
      (push_constants.ne + 255u) / 256u,
      1u,
      1u,
  };
  dispatch_with_spec(
      shader_id,
      KernelSpecId::LayerNormAffine,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s,
      grid);
}

void dispatch_generic_unary_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float param1,
    float param2,
    float param3,
    float param4) {
  const auto element_count =
      checked_u32(out.size(), "generic unary element count");
  const auto push_constants = make_generic_push_constants(
      element_count, param1, param2, param3, param4);

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GenericUnary,
      bound_arrays,
      push_constants,
      push_constants.KX,
      cmd_buffer,
      s);
}

void dispatch_arange_op(
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    double start,
    double step) {
  const auto num_elements = checked_u32(out.size(), "arange element count");
  const auto push_constants =
      make_arange_push_constants(out, num_elements, start, step);
  const std::array<BoundArray, 1> bound_arrays = {{{&out, "dst"}}};

  dispatch_with_spec(
      shader_id,
      KernelSpecId::Arange,
      bound_arrays,
      push_constants,
      push_constants.KX,
      cmd_buffer,
      s);
}

void dispatch_fill_op(
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float value) {
  const auto num_elements = checked_u32(out.size(), "fill element count");
  const auto push_constants = make_fill_push_constants(num_elements, value);
  const std::array<BoundArray, 1> bound_arrays = {{{&out, "dst"}}};

  dispatch_with_spec(
      shader_id,
      KernelSpecId::Arange,
      bound_arrays,
      push_constants,
      push_constants.KX,
      cmd_buffer,
      s);
}

void dispatch_sum_rows_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float weight) {
  if (out.size() == 0) {
    return;
  }

  const auto push_constants = make_sum_rows_push_constants(in, out, weight);
  const auto row_count = checked_u32(out.size(), "sum_rows output rows");
  const uint32_t block_size = push_constants.n_cols <= 32u ? 32u
      : push_constants.n_cols <= 64u                       ? 64u
                                                           : 128u;

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::SumRows,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      std::nullopt,
      {block_size});
}

void dispatch_argmax_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] ArgMax requires input rank >= 1.");
  }

  const uint32_t row_width =
      checked_u32(in.shape(in.ndim() - 1), "argmax reduction width");
  const uint32_t row_count = checked_u32(out.size(), "argmax row count");
  auto push_constants =
      make_generic_push_constants(row_width, 0.0f, 0.0f, 0.0f, 0.0f);
  push_constants.KY = row_count;

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Argmax,
      bound_arrays,
      push_constants,
      push_constants.KY,
      cmd_buffer,
      s,
      std::nullopt,
      {32u});
}

void dispatch_argsort_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t order) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Argsort requires input rank >= 1.");
  }

  const uint32_t ncols =
      checked_u32(in.shape(in.ndim() - 1), "argsort column count");
  constexpr uint32_t kMaxLargeArgsortCols = 262144u;
  if (ncols > kMaxLargeArgsortCols) {
    throw std::runtime_error(
        "[vulkan::kernels] Argsort requires ncols <= 262144.");
  }

  const uint32_t nrows = checked_u32(out.size() / ncols, "argsort row count");

  constexpr uint32_t signed_min_as_u32 = 1u << 31;
  const bool topk_suffix_partition = order > signed_min_as_u32 && ncols == 256u;
  const bool large_sort = ncols > 1024u;
  const uint32_t large_ncols_padded =
      large_sort ? next_power_of_two_u32(ncols) : 0u;
  const uint32_t ncols_padded = topk_suffix_partition ? 256u
      : large_sort                                    ? large_ncols_padded
                                                      : 1024u;
  const uint32_t ncols_padded_log2 = topk_suffix_partition ? 8u
      : large_sort ? log2_exact_u32(ncols_padded)
                   : 10u;
  ArgsortPushConstants push_constants{};
  push_constants.ncols = ncols;
  push_constants.ncols_padded = ncols_padded;
  push_constants.ncols_padded_log2 = ncols_padded_log2;
  push_constants.nrows = nrows;
  push_constants.order = order;
  push_constants.outer_start = 0u;
  push_constants.outer_end = ncols_padded_log2;
  push_constants.inner_start = 0u;
  push_constants.inner_end = ncols_padded_log2 + 1u;

  const auto row_groups = std::min(nrows, 65535u);
  const std::array<uint32_t, 3> grid = {1u, row_groups, 1u};

  if (large_sort) {
    const uint32_t wg_unroll = ncols_padded / 1024u;
    array tmp(
        {static_cast<int>(nrows), static_cast<int>(ncols_padded), 2},
        int32,
        nullptr,
        {});
    tmp.set_data(allocator::malloc(tmp.nbytes()));

    const std::array<BoundArray, 3> bound_arrays = {{
        {&in, "src0"},
        {&tmp, "tmp"},
        {&out, "dst0"},
    }};
    dispatch_with_spec(
        StaticShaderId::argsort_large_f32,
        KernelSpecId::ArgsortLarge,
        bound_arrays,
        push_constants,
        nrows,
        cmd_buffer,
        s,
        grid,
        {1024u, wg_unroll});
    return;
  }

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst0"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Argsort,
      bound_arrays,
      push_constants,
      nrows,
      cmd_buffer,
      s,
      grid,
      {ncols_padded, ncols_padded_log2});
}

void dispatch_fft_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const FFTPushConstants& push_constants,
    const std::vector<uint32_t>& specialization_constants) {
  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::FFT,
      bound_arrays,
      push_constants,
      push_constants.batch_count,
      cmd_buffer,
      s,
      std::array<uint32_t, 3>{push_constants.batch_count, 1, 1},
      specialization_constants);
}

void dispatch_softmax_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax requires input rank >= 1.");
  }

  const uint32_t row_width = checked_u32(in.shape(in.ndim() - 1), "softmax KX");
  if (row_width == 0) {
    throw std::runtime_error("[vulkan::kernels] Softmax requires non-zero KX.");
  }

  const uint32_t total_elements = checked_u32(out.size(), "softmax elements");
  if (total_elements % row_width != 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax elements are not divisible by KX.");
  }
  const uint32_t row_count = total_elements / row_width;
  const uint32_t block_size = select_softmax_block_size(row_width);
  const auto push_constants =
      make_softmax_push_constants(in, row_width, row_count);

  const std::array<BoundArray, 4> bound_arrays = {{
      {&in, "src0"},
      {&in, "src1"},
      {&in, "src2"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Softmax,
      bound_arrays,
      push_constants,
      push_constants.nrows_x,
      cmd_buffer,
      s,
      std::nullopt,
      {block_size});
}

void dispatch_softmax_back_op(
    const array& grad,
    const array& y,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    float scale) {
  if (out.size() == 0) {
    return;
  }

  if (grad.ndim() == 0 || grad.shape() != y.shape() ||
      grad.shape() != out.shape()) {
    throw std::runtime_error(
        "[vulkan::kernels] SoftmaxBack requires matching non-scalar shapes.");
  }

  const uint32_t row_width =
      checked_u32(out.shape(out.ndim() - 1), "softmax_back KX");
  if (row_width == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] SoftmaxBack requires non-zero KX.");
  }

  const uint32_t total_elements =
      checked_u32(out.size(), "softmax_back elements");
  if (total_elements % row_width != 0) {
    throw std::runtime_error(
        "[vulkan::kernels] SoftmaxBack elements are not divisible by KX.");
  }
  const uint32_t row_count = total_elements / row_width;
  auto push_constants =
      make_generic_push_constants(row_width, scale, 0.0f, 0.0f, 0.0f);
  push_constants.KY = row_count;

  const std::array<BoundArray, 3> bound_arrays = {{
      {&grad, "src0"},
      {&y, "src1"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::SoftmaxBack,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      std::nullopt,
      {32u});
}

void dispatch_softmax_large_op(
    const array& in,
    array& out,
    StaticShaderId shader_id_pass1,
    StaticShaderId shader_id_pass2,
    StaticShaderId shader_id_pass3,
    vk::CommandBuffer cmd_buffer,
    const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax requires input rank >= 1.");
  }

  const uint32_t row_width = checked_u32(in.shape(in.ndim() - 1), "softmax KX");
  if (row_width == 0) {
    throw std::runtime_error("[vulkan::kernels] Softmax requires non-zero KX.");
  }

  const uint32_t total_elements = checked_u32(out.size(), "softmax elements");
  if (total_elements % row_width != 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax elements are not divisible by KX.");
  }

  const uint32_t row_count = total_elements / row_width;
  const auto push_constants =
      make_softmax_push_constants(in, row_width, row_count);

  const uint32_t block_size = 128u;
  const uint32_t elems_per_workgroup = block_size * 4u;
  const uint32_t num_workgroups_x = (row_width + block_size - 1) / block_size;
  if (num_workgroups_x == 0) {
    return;
  }

  const uint64_t tmp_elements_u64 = static_cast<uint64_t>(num_workgroups_x) *
      static_cast<uint64_t>(row_count);
  if (tmp_elements_u64 > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax large temporary size exceeds uint32 range.");
  }
  if (tmp_elements_u64 >
      static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "[vulkan::kernels] Softmax large temporary shape exceeds int range.");
  }
  const int tmp_elements = static_cast<int>(tmp_elements_u64);

  array temp_max = acquire_scratch_array(
      s, kSoftmaxLargeMaxScratchLane, {tmp_elements}, float32);
  array temp_sum = acquire_scratch_array(
      s, kSoftmaxLargeSumScratchLane, {tmp_elements}, float32);

  const std::array<BoundArray, 6> bound_arrays = {{
      {&in, "src0"},
      {&in, "src1"},
      {&in, "src2"},
      {&out, "dst"},
      {&temp_max, "tmp_max"},
      {&temp_sum, "tmp_sum"},
  }};

  const std::array<uint32_t, 3> grid = {num_workgroups_x, row_count, 1};

  dispatch_with_spec(
      shader_id_pass1,
      KernelSpecId::SoftmaxLarge,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      grid,
      {128u, 4u});

  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(
      cmd_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);

  dispatch_with_spec(
      shader_id_pass2,
      KernelSpecId::SoftmaxLarge,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      grid,
      {128u, 4u});

  vkCmdPipelineBarrier(
      cmd_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);

  dispatch_with_spec(
      shader_id_pass3,
      KernelSpecId::SoftmaxLarge,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      grid,
      {128u, 4u});

  mark_scratch_array_written(s, kSoftmaxLargeMaxScratchLane);
  mark_scratch_array_written(s, kSoftmaxLargeSumScratchLane);
}

void dispatch_diag_mask_inf_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t rows_per_channel,
    uint32_t n_past) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] DiagMaskInf requires input rank >= 1.");
  }
  const bool f32_io = in.dtype() == float32 && out.dtype() == float32;
  const bool f16_io = in.dtype() == float16 && out.dtype() == float16;
  const bool bf16_io = in.dtype() == bfloat16 && out.dtype() == bfloat16;
  if (!f32_io && !f16_io && !bf16_io) {
    throw std::runtime_error(
        "[vulkan::kernels] DiagMaskInf currently requires matching f32/f16/bf16 IO.");
  }

  const uint32_t ncols =
      checked_u32(in.shape(in.ndim() - 1), "diag_mask_inf ncols");
  if (ncols == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] DiagMaskInf requires non-zero column count.");
  }

  const uint32_t total_elements =
      checked_u32(out.size(), "diag_mask_inf elements");
  if (total_elements % ncols != 0) {
    throw std::runtime_error(
        "[vulkan::kernels] DiagMaskInf elements are not divisible by ncols.");
  }

  DiagMaskInfPushConstants push_constants{};
  push_constants.ncols = ncols;
  push_constants.rows_per_channel = rows_per_channel;
  push_constants.n_past = n_past;

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};

  const uint32_t row_count = total_elements / ncols;
  const std::array<uint32_t, 3> grid = {
      row_count,
      (ncols + 511u) / 512u,
      1u,
  };

  dispatch_with_spec(
      shader_id,
      KernelSpecId::DiagMaskInf,
      bound_arrays,
      push_constants,
      total_elements,
      cmd_buffer,
      s,
      grid);
}

void dispatch_flash_attention_op(
    const array& q,
    const array& k,
    const array& v,
    const array& mask,
    const array& sinks,
    array& out,
    const array& mask_opt,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const FlashAttentionPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants) {
  const std::array<BoundArray, 7> bound_arrays = {{
      {&q, "q"},
      {&k, "k"},
      {&v, "v"},
      {&mask, "mask"},
      {&sinks, "sinks"},
      {&out, "dst"},
      {&mask_opt, "mask_opt"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::FlashAttention,
      bound_arrays,
      push_constants,
      checked_mul_u32(push_constants.N, push_constants.KV, "flash_attn ne"),
      cmd_buffer,
      s,
      grid,
      specialization_constants);
}

void dispatch_flash_attention_split_k_reduce_op(
    const array& in,
    const array& sinks,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const FlashAttentionSplitKReducePushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants) {
  const std::array<BoundArray, 3> bound_arrays = {{
      {&in, "in"},
      {&sinks, "sinks"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::FlashAttentionSplitKReduce,
      bound_arrays,
      push_constants,
      checked_mul_u32(
          push_constants.ne1, push_constants.D, "fa_split_reduce ne"),
      cmd_buffer,
      s,
      grid,
      specialization_constants);
}

void dispatch_flash_attention_mask_opt_op(
    const array& mask,
    array& mask_opt,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const FlashAttentionMaskOptPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants) {
  const std::array<BoundArray, 2> bound_arrays = {{
      {&mask, "mask"},
      {&mask_opt, "mask_opt"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::FlashAttentionMaskOpt,
      bound_arrays,
      push_constants,
      checked_mul_u32(
          push_constants.nem0, push_constants.nem1, "fa_mask_opt ne"),
      cmd_buffer,
      s,
      grid,
      specialization_constants);
}

void dispatch_cumsum_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    bool reverse,
    bool inclusive) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Cumsum requires input rank >= 1.");
  }

  const uint32_t row_width =
      checked_u32(in.shape(in.ndim() - 1), "cumsum n_cols");
  if (row_width == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Cumsum requires non-zero row width.");
  }

  const uint32_t total_elements = checked_u32(out.size(), "cumsum elements");
  if (total_elements % row_width != 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Cumsum elements are not divisible by row width.");
  }
  const uint32_t row_count = total_elements / row_width;

  const auto push_constants = make_sum_rows_push_constants(in, out, 1.0f);

  const uint32_t block_size = 128u;
  const uint32_t elems_per_workgroup = block_size * 4u;
  const uint32_t num_workgroups_x = (row_width + block_size - 1) / block_size;

  if (num_workgroups_x <= 1) {
    const std::array<BoundArray, 2> bound_arrays = {{
        {&in, "src0"},
        {&out, "dst"},
    }};
    dispatch_with_spec(
        shader_id,
        KernelSpecId::SumRows,
        bound_arrays,
        push_constants,
        row_count,
        cmd_buffer,
        s,
        std::nullopt,
        {128u, 32u, 4u, reverse ? 1u : 0u, inclusive ? 1u : 0u});
    return;
  }

  if (reverse || !inclusive) {
    throw std::runtime_error(
        "[vulkan::kernels] Cumsum reverse/exclusive requires single-pass row width.");
  }

  const uint64_t tmp_elements_u64 = static_cast<uint64_t>(num_workgroups_x) *
      static_cast<uint64_t>(row_count);
  if (tmp_elements_u64 >
      static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "[vulkan::kernels] Cumsum multipass temporary shape exceeds int range.");
  }
  const int tmp_elements = static_cast<int>(tmp_elements_u64);

  array temp = acquire_scratch_array(
      s, kCumsumMultipassScratchLane, {tmp_elements}, float32);

  const std::array<BoundArray, 3> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
      {&temp, "tmp"},
  }};

  const std::array<uint32_t, 3> grid = {num_workgroups_x, row_count, 1};

  dispatch_with_spec(
      StaticShaderId::cumsum_multipass1_f32,
      KernelSpecId::CumsumMultipass,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      grid,
      {128u, 32u});

  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(
      cmd_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);

  dispatch_with_spec(
      StaticShaderId::cumsum_multipass2_f32,
      KernelSpecId::CumsumMultipass,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      grid,
      {128u, 32u});

  mark_scratch_array_written(s, kCumsumMultipassScratchLane);
}

void dispatch_scan_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    bool reverse,
    bool inclusive) {
  if (out.size() == 0) {
    return;
  }

  if (in.ndim() == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Scan requires input rank >= 1.");
  }

  const uint32_t row_width =
      checked_u32(in.shape(in.ndim() - 1), "scan n_cols");
  if (row_width == 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Scan requires non-zero row width.");
  }

  const uint32_t total_elements = checked_u32(out.size(), "scan elements");
  if (total_elements % row_width != 0) {
    throw std::runtime_error(
        "[vulkan::kernels] Scan elements are not divisible by row width.");
  }
  const uint32_t row_count = total_elements / row_width;
  const uint32_t block_size = 128u;
  const uint32_t num_workgroups_x = (row_width + block_size - 1) / block_size;
  if (num_workgroups_x > 1) {
    throw std::runtime_error(
        "[vulkan::kernels] Scan multipass is not implemented for this shader.");
  }

  const auto push_constants = make_sum_rows_push_constants(in, out, 1.0f);
  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src0"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::SumRows,
      bound_arrays,
      push_constants,
      row_count,
      cmd_buffer,
      s,
      std::nullopt,
      {128u, 32u, 4u, reverse ? 1u : 0u, inclusive ? 1u : 0u});
}

void dispatch_mul_mm_op(
    const array& a,
    const array& b,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const MatmulPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants) {
  if (a.ndim() < 2 || b.ndim() < 2 || out.ndim() < 2) {
    throw std::runtime_error(
        "[vulkan::kernels] mul_mm dispatch requires rank >= 2 tensors.");
  }

  const uint32_t m = checked_u32(out.shape(-2), "mul_mm M");
  const uint32_t n = checked_u32(out.shape(-1), "mul_mm N");
  const uint32_t k = checked_u32(a.shape(-1), "mul_mm K");
  if (checked_u32(a.shape(-2), "mul_mm A M") != m ||
      checked_u32(b.shape(-2), "mul_mm B N") != n ||
      checked_u32(b.shape(-1), "mul_mm B K") != k) {
    throw std::runtime_error(
        "[vulkan::kernels] mul_mm dispatch received incompatible shapes.");
  }

  const std::array<BoundArray, 3> bound_arrays = {{
      {&a, "src0"},
      {&b, "src1"},
      {&out, "dst"},
  }};
  const uint32_t num_elements = checked_mul_u32(m, n, "mul_mm output elements");
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Matmul,
      bound_arrays,
      push_constants,
      num_elements,
      cmd_buffer,
      s,
      grid,
      matmul_specialization_constants(specialization_constants));
}

void dispatch_matmul_split_k_reduce_op(
    const array& in,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const MatmulSplitKReducePushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants) {
  if (in.nbytes() == 0 || out.nbytes() == 0) {
    return;
  }

  const std::array<BoundArray, 2> bound_arrays = {{
      {&in, "src"},
      {&out, "dst"},
  }};
  const uint32_t num_elements = checked_mul_u32(
      push_constants.ne, 1u, "matmul split-k reduce output elements");
  dispatch_with_spec(
      shader_id,
      KernelSpecId::MatmulSplitKReduce,
      bound_arrays,
      push_constants,
      num_elements,
      cmd_buffer,
      s,
      grid,
      specialization_constants);
}

void dispatch_mul_mat_vec_op(
    const array& matrix,
    const array& vec,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s) {
  if (matrix.ndim() != 2 || vec.ndim() != 2 || out.ndim() != 2) {
    throw std::runtime_error(
        "[vulkan::kernels] Mat-vec dispatch requires rank-2 tensors.");
  }
  if (vec.shape(0) != out.shape(0)) {
    throw std::runtime_error(
        "[vulkan::kernels] Mat-vec dispatch expects matching batch rows.");
  }

  const uint32_t batch_rows = checked_u32(vec.shape(0), "matvec batch rows");
  const uint32_t ncols = checked_u32(vec.shape(1), "matvec ncols");
  const uint32_t nrows = checked_u32(out.shape(1), "matvec nrows");
  if (checked_u32(matrix.shape(0), "matvec matrix N") != nrows ||
      checked_u32(matrix.shape(1), "matvec matrix K") != ncols) {
    throw std::runtime_error(
        "[vulkan::kernels] Mat-vec dispatch received incompatible shapes.");
  }

  MatVecPushConstants push_constants{};
  push_constants.ncols = ncols;
  push_constants.stride_a = ncols;
  push_constants.stride_b = ncols;
  push_constants.stride_d = nrows;
  push_constants.batch_stride_a =
      checked_mul_u32(ncols, nrows, "matvec batch_stride_a");
  push_constants.batch_stride_b = ncols;
  push_constants.batch_stride_d = nrows;
  push_constants.fusion_flags = 0;
  push_constants.ne02 = 1;
  push_constants.ne12 = batch_rows;
  push_constants.broadcast2 = batch_rows;
  push_constants.broadcast3 = 1;

  const std::array<BoundArray, 5> bound_arrays = {{
      {&matrix, "src0"},
      {&vec, "src1"},
      {&out, "dst"},
      {&out, "fuse0"},
      {&out, "fuse1"},
  }};

  constexpr uint32_t kMaxWorkgroupsX = 65535u;
  const uint32_t groups_z = (nrows + kMaxWorkgroupsX - 1u) / kMaxWorkgroupsX;
  const uint32_t groups_x = (nrows + groups_z - 1u) / groups_z;

  const bool force_single_column_dispatch =
      VulkanContext::get().architecture() == GpuArchitecture::AmdRdna;
  const uint32_t col_chunk =
      force_single_column_dispatch ? 1u : kMaxMulMatVecCols;

  for (uint32_t base_work_group_y = 0; base_work_group_y < batch_rows;
       base_work_group_y += col_chunk) {
    push_constants.base_work_group_y = base_work_group_y;
    const uint32_t num_cols =
        std::min(col_chunk, batch_rows - base_work_group_y);
    const std::array<uint32_t, 3> grid = {groups_x, 1u, groups_z};
    const std::vector<uint32_t> specialization_constants = {
        32u,
        1u,
        num_cols,
    };

    dispatch_with_spec(
        shader_id,
        KernelSpecId::MatVec,
        bound_arrays,
        push_constants,
        nrows,
        cmd_buffer,
        s,
        grid,
        specialization_constants);
  }
}

void dispatch_mul_mat_vec_p021_op(
    const array& matrix,
    const array& vec,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const MatVecP021PushConstants& push_constants,
    const std::array<uint32_t, 3>& grid,
    const std::vector<uint32_t>& specialization_constants) {
  const std::array<BoundArray, 5> bound_arrays = {{
      {&matrix, "src0"},
      {&vec, "src1"},
      {&out, "dst"},
      {&out, "fuse0"},
      {&out, "fuse1"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::MatVecP021,
      bound_arrays,
      push_constants,
      checked_u32(out.size(), "mul_mat_vec_p021 output elements"),
      cmd_buffer,
      s,
      grid,
      specialization_constants);
}

void dispatch_mul_mat_vec_nc_op(
    const array& matrix,
    const array& vec,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const MatVecNcPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 5> bound_arrays = {{
      {&matrix, "src0"},
      {&vec, "src1"},
      {&out, "dst"},
      {&out, "fuse0"},
      {&out, "fuse1"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::MatVecNc,
      bound_arrays,
      push_constants,
      checked_u32(out.size(), "mul_mat_vec_nc output elements"),
      cmd_buffer,
      s,
      grid);
}

void dispatch_random_bits_op(
    const array& keys,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const RandomBitsPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 2> bound_arrays = {{
      {&keys, "keys"},
      {&out, "out"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::RandomBits,
      bound_arrays,
      push_constants,
      push_constants.num_keys,
      cmd_buffer,
      s,
      grid);
}

void dispatch_gather_op(
    const array& src,
    const array& indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t slice_size,
    uint32_t axis_size,
    uint32_t index_count) {
  GatherPushConstants push_constants{};
  push_constants.ne =
      checked_mul_u32(index_count, slice_size, "gather elements");
  push_constants.slice_size = slice_size;
  push_constants.axis_size = axis_size;
  push_constants.index_count = index_count;

  const std::array<BoundArray, 3> bound_arrays = {{
      {&src, "src"},
      {&indices, "indices"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Gather,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_gather_axis_op(
    const array& src,
    const array& indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t size_pre,
    uint32_t size_axis,
    uint32_t size_post,
    uint32_t idx_axis_size) {
  GatherAxisPushConstants push_constants{};
  const uint32_t elems_per_prefix =
      checked_mul_u32(idx_axis_size, size_post, "gather_axis elems_per_prefix");
  push_constants.ne =
      checked_mul_u32(size_pre, elems_per_prefix, "gather_axis elements");
  push_constants.size_pre = size_pre;
  push_constants.size_axis = size_axis;
  push_constants.size_post = size_post;
  push_constants.idx_axis_size = idx_axis_size;

  const std::array<BoundArray, 3> bound_arrays = {{
      {&src, "src"},
      {&indices, "indices"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GatherAxis,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_gather_take_op(
    const array& src,
    const array& indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t size_pre,
    uint32_t size_axis,
    uint32_t size_post,
    uint32_t index_count) {
  GatherAxisPushConstants push_constants{};
  const uint32_t elems_per_prefix =
      checked_mul_u32(index_count, size_post, "gather_take elems_per_prefix");
  push_constants.ne =
      checked_mul_u32(size_pre, elems_per_prefix, "gather_take elements");
  push_constants.size_pre = size_pre;
  push_constants.size_axis = size_axis;
  push_constants.size_post = size_post;
  push_constants.idx_axis_size = index_count;

  const std::array<BoundArray, 3> bound_arrays = {{
      {&src, "src"},
      {&indices, "indices"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GatherAxis,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_gather_pair_op(
    const array& src,
    const array& idx0,
    const array& idx1,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t outer_size,
    uint32_t axis0_size,
    uint32_t slice0_size,
    uint32_t middle_size,
    uint32_t axis1_size,
    uint32_t slice1_size,
    uint32_t inner_size,
    uint32_t index_count) {
  GatherPairPushConstants push_constants{};
  const uint32_t slice_size = checked_mul_u32(
      checked_mul_u32(
          checked_mul_u32(
              checked_mul_u32(
                  outer_size, slice0_size, "gather_pair outer_slice0"),
              middle_size,
              "gather_pair outer_middle"),
          slice1_size,
          "gather_pair middle_slice1"),
      inner_size,
      "gather_pair slice_size");
  push_constants.ne =
      checked_mul_u32(index_count, slice_size, "gather_pair elements");
  push_constants.outer_size = outer_size;
  push_constants.axis0_size = axis0_size;
  push_constants.slice0_size = slice0_size;
  push_constants.middle_size = middle_size;
  push_constants.axis1_size = axis1_size;
  push_constants.slice1_size = slice1_size;
  push_constants.inner_size = inner_size;
  push_constants.index_count = index_count;

  const std::array<BoundArray, 4> bound_arrays = {{
      {&src, "src"},
      {&idx0, "idx0"},
      {&idx1, "idx1"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GatherPair,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_scatter_axis_op(
    const array& updates,
    const array& indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t size_pre,
    uint32_t size_axis,
    uint32_t size_post,
    uint32_t idx_axis_size) {
  GatherAxisPushConstants push_constants{};
  const uint32_t elems_per_prefix = checked_mul_u32(
      idx_axis_size, size_post, "scatter_axis elems_per_prefix");
  push_constants.ne =
      checked_mul_u32(size_pre, elems_per_prefix, "scatter_axis elements");
  push_constants.size_pre = size_pre;
  push_constants.size_axis = size_axis;
  push_constants.size_post = size_post;
  push_constants.idx_axis_size = idx_axis_size;

  const std::array<BoundArray, 3> bound_arrays = {{
      {&updates, "updates"},
      {&indices, "indices"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GatherAxis,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_scatter_take_op(
    const array& updates,
    const array& indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t size_pre,
    uint32_t size_axis,
    uint32_t size_post,
    uint32_t index_count) {
  GatherAxisPushConstants push_constants{};
  const uint32_t elems_per_prefix =
      checked_mul_u32(index_count, size_post, "scatter_take elems_per_prefix");
  push_constants.ne =
      checked_mul_u32(size_pre, elems_per_prefix, "scatter_take elements");
  push_constants.size_pre = size_pre;
  push_constants.size_axis = size_axis;
  push_constants.size_post = size_post;
  push_constants.idx_axis_size = index_count;

  const std::array<BoundArray, 3> bound_arrays = {{
      {&updates, "updates"},
      {&indices, "indices"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GatherAxis,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_scatter_pair_op(
    const array& updates,
    const array& idx0,
    const array& idx1,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t outer_size,
    uint32_t axis0_size,
    uint32_t slice0_size,
    uint32_t middle_size,
    uint32_t axis1_size,
    uint32_t slice1_size,
    uint32_t inner_size,
    uint32_t index_count) {
  GatherPairPushConstants push_constants{};
  const uint32_t slice_size = checked_mul_u32(
      checked_mul_u32(
          checked_mul_u32(
              checked_mul_u32(
                  outer_size, slice0_size, "scatter_pair outer_slice0"),
              middle_size,
              "scatter_pair outer_middle"),
          slice1_size,
          "scatter_pair middle_slice1"),
      inner_size,
      "scatter_pair slice_size");
  push_constants.ne =
      checked_mul_u32(index_count, slice_size, "scatter_pair elements");
  push_constants.outer_size = outer_size;
  push_constants.axis0_size = axis0_size;
  push_constants.slice0_size = slice0_size;
  push_constants.middle_size = middle_size;
  push_constants.axis1_size = axis1_size;
  push_constants.slice1_size = slice1_size;
  push_constants.inner_size = inner_size;
  push_constants.index_count = index_count;

  const std::array<BoundArray, 4> bound_arrays = {{
      {&updates, "updates"},
      {&idx0, "idx0"},
      {&idx1, "idx1"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GatherPair,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_masked_scatter_op(
    const array& mask,
    const array& src,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    uint32_t src_batch_size,
    uint32_t mask_batch_size) {
  MaskedScatterPushConstants push_constants{};
  push_constants.ne = checked_u32(mask.size(), "masked_scatter elements");
  push_constants.src_batch_size = src_batch_size;
  push_constants.mask_batch_size = mask_batch_size;

  const std::array<BoundArray, 3> bound_arrays = {{
      {&mask, "mask"},
      {&src, "src"},
      {&out, "dst"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::MaskedScatter,
      bound_arrays,
      push_constants,
      push_constants.ne,
      cmd_buffer,
      s);
}

void dispatch_rope_op(
    const array& in,
    const array& positions,
    const array& freqs,
    array& out,
    const array& indices,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const RopePushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 5> bound_arrays = {{
      {&in, "src0"},
      {&positions, "positions"},
      {&freqs, "freqs"},
      {&out, "dst"},
      {&indices, "indices"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Rope,
      bound_arrays,
      push_constants,
      push_constants.nrows,
      cmd_buffer,
      s,
      grid);
}

void dispatch_affine_dequant_op(
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const AffineDequantPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 4> bound_arrays = {{
      {&w, "W"},
      {&scales, "S"},
      {&biases, "B"},
      {&out, "D"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::AffineDequant,
      bound_arrays,
      push_constants,
      grid[0],
      cmd_buffer,
      s,
      grid);
}

void dispatch_affine_quant_op(
    const array& in,
    array& w,
    array& scales,
    array& biases,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const AffineQuantPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 4> bound_arrays = {{
      {&in, "A"},
      {&w, "W"},
      {&scales, "S"},
      {&biases, "B"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::AffineQuant,
      bound_arrays,
      push_constants,
      grid[0],
      cmd_buffer,
      s,
      grid);
}

void dispatch_nvfp4_dequant_op(
    const array& w,
    const array& scales,
    const array& global_scale,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const Nvfp4DequantPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 4> bound_arrays = {{
      {&w, "W"},
      {&scales, "S"},
      {&out, "D"},
      {&global_scale, "G"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Nvfp4Dequant,
      bound_arrays,
      push_constants,
      grid[0],
      cmd_buffer,
      s,
      grid);
}

void dispatch_nvfp4_quant_op(
    const array& in,
    array& w,
    array& scales,
    const array& global_scale,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const Nvfp4QuantPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 4> bound_arrays = {{
      {&in, "A"},
      {&w, "W"},
      {&scales, "S"},
      {&global_scale, "G"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Nvfp4Quant,
      bound_arrays,
      push_constants,
      grid[0],
      cmd_buffer,
      s,
      grid);
}

void dispatch_fused_affine_matmul_op(
    const array& w,
    const array& scales,
    const array& biases,
    const array& x,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const FusedAffineMatmulPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  if (w.ndim() != 2 || scales.ndim() != 2 || biases.ndim() != 2 ||
      x.ndim() != 2 || out.ndim() != 2) {
    throw std::runtime_error(
        "[vulkan::kernels] fused_affine_matmul dispatch requires 2D tensors.");
  }

  const uint32_t rows = checked_u32(out.shape(-2), "fused_affine_matmul rows");
  const uint32_t cols = checked_u32(out.shape(-1), "fused_affine_matmul cols");
  const uint32_t k = push_constants.K;
  const uint32_t num_groups = push_constants.num_groups;

  if (checked_u32(x.shape(-2), "fused_affine_matmul X rows") != rows ||
      checked_u32(x.shape(-1), "fused_affine_matmul X K") != k ||
      checked_u32(w.shape(-2), "fused_affine_matmul W rows") != cols ||
      checked_u32(scales.shape(-2), "fused_affine_matmul scales rows") !=
          cols ||
      checked_u32(biases.shape(-2), "fused_affine_matmul biases rows") !=
          cols ||
      checked_u32(scales.shape(-1), "fused_affine_matmul scales groups") !=
          num_groups ||
      checked_u32(biases.shape(-1), "fused_affine_matmul biases groups") !=
          num_groups) {
    throw std::runtime_error(
        "[vulkan::kernels] fused_affine_matmul dispatch received incompatible shapes.");
  }

  const std::array<BoundArray, 5> bound_arrays = {{
      {&w, "W"},
      {&scales, "SCALES"},
      {&biases, "BIASES"},
      {&x, "X"},
      {&out, "OUT"},
  }};
  const uint32_t num_elements =
      checked_mul_u32(rows, cols, "fused_affine_matmul output elements");
  dispatch_with_spec(
      shader_id,
      KernelSpecId::FusedAffineMatmul,
      bound_arrays,
      push_constants,
      num_elements,
      cmd_buffer,
      s,
      grid,
      matmul_specialization_constants({}));
}

void dispatch_gather_affine_matmul_op(
    const array& w,
    const array& scales,
    const array& biases,
    const array& x,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const GatherAffineMatmulPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 7> bound_arrays = {{
      {&w, "W"},
      {&scales, "SCALES"},
      {&biases, "BIASES"},
      {&x, "X"},
      {&lhs_indices, "LHS_INDICES"},
      {&rhs_indices, "RHS_INDICES"},
      {&out, "OUT"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::GatherAffineMatmul,
      bound_arrays,
      push_constants,
      checked_mul_u32(
          checked_mul_u32(
              push_constants.rows, push_constants.cols, "gather qmm matrix"),
          grid[2],
          "gather qmm elements"),
      cmd_buffer,
      s,
      grid,
      matmul_specialization_constants({}));
}

void dispatch_nvfp4_qmatmul_op(
    const array& w,
    const array& scales,
    const array& x,
    const array& global_scale_x,
    const array& global_scale_w,
    array& out,
    StaticShaderId shader_id,
    vk::CommandBuffer cmd_buffer,
    const Stream& s,
    const Nvfp4QMatmulPushConstants& push_constants,
    const std::array<uint32_t, 3>& grid) {
  const std::array<BoundArray, 6> bound_arrays = {{
      {&w, "W"},
      {&scales, "S"},
      {&x, "X"},
      {&global_scale_x, "GX"},
      {&global_scale_w, "GW"},
      {&out, "OUT"},
  }};
  dispatch_with_spec(
      shader_id,
      KernelSpecId::Nvfp4QMatmul,
      bound_arrays,
      push_constants,
      checked_mul_u32(
          push_constants.rows, push_constants.cols, "nvfp4 qmatmul elements"),
      cmd_buffer,
      s,
      grid);
}

void write_descriptor_binding(
    std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t binding) {
  bindings.push_back(
      {binding,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       1,
       VK_SHADER_STAGE_COMPUTE_BIT,
       nullptr});
}

void write_descriptor_buffer(
    const array& arr,
    uint32_t binding,
    VkDescriptorSet descriptor_set,
    std::vector<VkDescriptorBufferInfo>& infos,
    std::vector<VkWriteDescriptorSet>& writes) {
  auto* vulkan_buffer = static_cast<const VulkanBuffer*>(
      static_cast<const void*>(arr.buffer().ptr()));
  if (vulkan_buffer == nullptr || vulkan_buffer->buffer == VK_NULL_HANDLE) {
    throw std::runtime_error("Missing Vulkan buffer for dynamic copy shader.");
  }

  VkDescriptorBufferInfo info{};
  info.buffer = vulkan_buffer->buffer;
  info.offset = 0;
  info.range = VK_WHOLE_SIZE;
  infos.push_back(info);

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = descriptor_set;
  write.dstBinding = binding;
  write.dstArrayElement = 0;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.descriptorCount = 1;
  write.pBufferInfo = &infos.back();
  writes.push_back(write);
}

void ensure_dynamic_shader_registered(
    const std::string& shader_name,
    const std::string& glsl_source) {
  auto& manager = KernelManager::get();
  if (manager.get_shader(shader_name) != nullptr) {
    return;
  }

  auto spirv = compile_glsl_to_spirv(glsl_source, shader_name);
  manager.register_shader(
      shader_name, spirv.data(), spirv.size() * sizeof(uint32_t));
}

bool is_vulkan_storage_array(const array& arr) {
  return arr.data_shared_ptr() != nullptr && is_vulkan_buffer(arr.buffer());
}

DynamicComputeDispatch dispatch_dynamic_compute_begin(
    const std::string& shader_name,
    const std::string& glsl_source,
    uint32_t num_bindings,
    const DynamicArrayRef* arrays,
    uint32_t push_constant_size,
    const Stream& s) {
  ensure_dynamic_shader_registered(shader_name, glsl_source);

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for (uint32_t i = 0; i < num_bindings; ++i) {
    write_descriptor_binding(bindings, arrays[i].binding);
  }

  auto& manager = KernelManager::get();
  auto* pipeline =
      manager.get_pipeline(shader_name, bindings, push_constant_size);
  auto command_buffer = begin_command_recording(s.index);

  // Prepare descriptor writes (used for both push descriptor and traditional
  // path)
  std::vector<VkDescriptorBufferInfo> infos;
  std::vector<VkWriteDescriptorSet> writes;
  infos.reserve(num_bindings);
  writes.reserve(num_bindings);

  vk::DescriptorSet descriptor_set = nullptr;
  if (!pipeline->supports_push_descriptor) {
    // Traditional path: allocate descriptor set
    const uint64_t descriptor_epoch = descriptor_epoch_for_stream(s);
    descriptor_set =
        manager.allocate_descriptor_set(pipeline->descriptor_layout);
    manager.defer_descriptor_set_free(
        s.index, descriptor_epoch, descriptor_set);
  }

  for (uint32_t i = 0; i < num_bindings; ++i) {
    retain_array_for_stream(s, *arrays[i].arr);
    write_descriptor_buffer(
        *arrays[i].arr, arrays[i].binding, descriptor_set, infos, writes);
  }

  vkCmdBindPipeline(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);

  if (pipeline->supports_push_descriptor) {
    // Use push descriptors: single call to update and bind
    auto push_fn = VulkanContext::get().push_descriptor_fn();
    if (push_fn != nullptr) {
      push_fn(
          command_buffer,
          VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline->layout,
          0, // set index
          static_cast<uint32_t>(writes.size()),
          writes.data());
    }
  } else {
    // Traditional path: update and bind descriptor set
    vkUpdateDescriptorSets(
        VulkanContext::get().device(),
        static_cast<uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);

    VkDescriptorSet vk_descriptor_set =
        static_cast<VkDescriptorSet>(descriptor_set);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline->layout,
        0,
        1,
        &vk_descriptor_set,
        0,
        nullptr);
  }

  return DynamicComputeDispatch{
      command_buffer,
      pipeline,
      std::move(infos),
      std::move(writes),
      descriptor_set};
}

void dispatch_dynamic_compute(
    const std::string& shader_name,
    const std::string& glsl_source,
    uint32_t num_bindings,
    const DynamicArrayRef* arrays,
    uint32_t workgroup_x,
    uint32_t workgroup_y,
    uint32_t workgroup_z,
    const Stream& s) {
  auto dispatch = dispatch_dynamic_compute_begin(
      shader_name, glsl_source, num_bindings, arrays, 0, s);
  vkCmdDispatch(dispatch.command_buffer, workgroup_x, workgroup_y, workgroup_z);
  end_command_recording(s.index);
}

} // namespace mlx::core::vulkan
