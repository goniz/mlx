#include "mlx/backend/vulkan/quantized.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/matmul.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

#include <deque>
#include <limits>
#include <mutex>

namespace mlx::core {
namespace {

bool is_supported_quantized_bits(int bits) {
  return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
      bits == 8;
}

bool is_supported_quantized_output_dtype(Dtype dtype) {
  return dtype == float16 || dtype == bfloat16 || dtype == float32;
}

// Match matmul.cpp: Vulkan requires workgroup counts <= 65535 on common GPUs.
constexpr uint32_t kMaxComputeWorkGroupCount = 65535;

bool dispatch_grid_within_limits(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
  const auto limits = vulkan::VulkanContext::get()
                          .physical_device()
                          .getProperties()
                          .limits;
  const uint32_t max_x = std::min(
      kMaxComputeWorkGroupCount, limits.maxComputeWorkGroupCount[0]);
  const uint32_t max_y = std::min(
      kMaxComputeWorkGroupCount, limits.maxComputeWorkGroupCount[1]);
  const uint32_t max_z = std::min(
      kMaxComputeWorkGroupCount, limits.maxComputeWorkGroupCount[2]);
  return x > 0 && y > 0 && z > 0 && x <= max_x && y <= max_y && z <= max_z;
}

constexpr size_t kDequantizedWeightCacheLimit = 8;

struct CachedArrayIdentity {
  std::weak_ptr<array::Data> data;
  const array::Data* id{nullptr};
  Shape shape;
  Strides strides;
  int64_t offset{0};
  Dtype dtype;
};

struct DequantizedWeightCacheEntry {
  CachedArrayIdentity w;
  CachedArrayIdentity scales;
  std::optional<CachedArrayIdentity> biases;
  QuantizationMode mode;
  int group_size{0};
  int bits{0};
  array dequantized;
};

std::mutex& dequantized_weight_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::deque<DequantizedWeightCacheEntry>& dequantized_weight_cache() {
  static std::deque<DequantizedWeightCacheEntry> cache;
  return cache;
}

bool cacheable_dequant_source(const array& arr) {
  return !arr.has_primitive() && !arr.is_donatable() &&
      arr.status() == array::Status::available &&
      arr.data_shared_ptr() != nullptr;
}

std::optional<CachedArrayIdentity> make_cached_array_identity(
    const array& arr) {
  if (!cacheable_dequant_source(arr)) {
    return std::nullopt;
  }
  auto data = arr.data_shared_ptr();
  return CachedArrayIdentity{
      data, data.get(), arr.shape(), arr.strides(), arr.offset(), arr.dtype()};
}

bool same_cached_array_identity(
    const CachedArrayIdentity& cached,
    const array& arr) {
  auto data = arr.data_shared_ptr();
  auto locked = cached.data.lock();
  return data != nullptr && locked.get() == data.get() &&
      cached.id == data.get() && cached.shape == arr.shape() &&
      cached.strides == arr.strides() && cached.offset == arr.offset() &&
      cached.dtype == arr.dtype();
}

bool dequantized_weight_cache_entry_alive(
    const DequantizedWeightCacheEntry& entry) {
  return !entry.w.data.expired() && !entry.scales.data.expired() &&
      (!entry.biases.has_value() || !entry.biases->data.expired());
}

std::optional<array> find_cached_dequantized_weight(
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    QuantizationMode mode,
    int group_size,
    int bits) {
  if (detail::in_tracing() || detail::retain_graph()) {
    return std::nullopt;
  }
  std::lock_guard<std::mutex> lock(dequantized_weight_cache_mutex());
  auto& cache = dequantized_weight_cache();
  for (auto it = cache.begin(); it != cache.end();) {
    if (!dequantized_weight_cache_entry_alive(*it)) {
      it = cache.erase(it);
      continue;
    }
    const bool biases_match =
        (!biases.has_value() && !it->biases.has_value()) ||
        (biases.has_value() && it->biases.has_value() &&
         same_cached_array_identity(*it->biases, *biases));
    if (it->mode == mode && it->group_size == group_size &&
        it->bits == bits && biases_match &&
        same_cached_array_identity(it->w, w) &&
        same_cached_array_identity(it->scales, scales)) {
      auto cached = it->dequantized;
      auto entry = std::move(*it);
      cache.erase(it);
      cache.push_back(std::move(entry));
      return cached;
    }
    ++it;
  }
  return std::nullopt;
}

void cache_dequantized_weight(
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    QuantizationMode mode,
    int group_size,
    int bits,
    const array& dequantized) {
  if (detail::in_tracing() || detail::retain_graph() ||
      dequantized.data_shared_ptr() == nullptr) {
    return;
  }
  auto w_id = make_cached_array_identity(w);
  auto scales_id = make_cached_array_identity(scales);
  std::optional<CachedArrayIdentity> biases_id;
  if (biases.has_value()) {
    biases_id = make_cached_array_identity(*biases);
  }
  if (!w_id.has_value() || !scales_id.has_value() ||
      (biases.has_value() && !biases_id.has_value())) {
    return;
  }

  std::lock_guard<std::mutex> lock(dequantized_weight_cache_mutex());
  auto& cache = dequantized_weight_cache();
  cache.push_back(DequantizedWeightCacheEntry{
      *w_id,
      *scales_id,
      biases_id,
      mode,
      group_size,
      bits,
      dequantized});
  while (cache.size() > kDequantizedWeightCacheLimit) {
    cache.pop_front();
  }
}

bool fused_nvfp4_qqmm_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_NVFP4_QQMM");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();
  return enabled;
}

std::optional<vulkan::StaticShaderId> fused_affine_matmul_shader_id(
    Dtype x_dtype) {
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::fused_affine_matmul_f32_f32;
    case float16:
      return vulkan::StaticShaderId::fused_affine_matmul_f16_f32;
    default:
      return std::nullopt;
  }
}

std::optional<vulkan::StaticShaderId> fused_affine_matvec8_shader_id(
    Dtype x_dtype) {
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::fused_affine_matvec8_f32_f32;
    case float16:
      return vulkan::StaticShaderId::fused_affine_matvec8_f16_f32;
    default:
      return std::nullopt;
  }
}

std::optional<vulkan::StaticShaderId> fused_affine_matvec_shader_id(
    Dtype x_dtype) {
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::fused_affine_matvec_f32_f32;
    case float16:
      return vulkan::StaticShaderId::fused_affine_matvec_f16_f32;
    default:
      return std::nullopt;
  }
}

std::optional<vulkan::StaticShaderId> fused_affine_qmm_shader_id(
    Dtype x_dtype) {
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::fused_affine_qmm_f32_f32;
    case float16:
      return vulkan::StaticShaderId::fused_affine_qmm_f16_f32;
    default:
      return std::nullopt;
  }
}

std::optional<vulkan::StaticShaderId> gather_affine_qmm_shader_id(
    Dtype x_dtype,
    Dtype out_dtype) {
  if (x_dtype == bfloat16 && out_dtype == bfloat16) {
    return vulkan::StaticShaderId::gather_affine_qmm_bf16_bf16;
  }
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::gather_affine_qmm_f32_f32;
    case float16:
      return vulkan::StaticShaderId::gather_affine_qmm_f16_f32;
    default:
      return std::nullopt;
  }
}

std::optional<vulkan::StaticShaderId> gather_affine_qmm_rhs_shader_id(
    Dtype x_dtype,
    Dtype out_dtype) {
  if (x_dtype == bfloat16 && out_dtype == bfloat16) {
    return vulkan::StaticShaderId::gather_affine_qmm_rhs_bf16_bf16;
  }
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::gather_affine_qmm_rhs_f32_f32;
    case float16:
      return vulkan::StaticShaderId::gather_affine_qmm_rhs_f16_f32;
    default:
      return std::nullopt;
  }
}

std::optional<vulkan::StaticShaderId> gather_affine_matvec8_shader_id(
    Dtype x_dtype,
    Dtype out_dtype) {
  if (x_dtype == bfloat16 && out_dtype == bfloat16) {
    return vulkan::StaticShaderId::gather_affine_matvec8_bf16_bf16;
  }
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::gather_affine_matvec8_f32_f32;
    case float16:
      return vulkan::StaticShaderId::gather_affine_matvec8_f16_f32;
    default:
      return std::nullopt;
  }
}

std::optional<vulkan::StaticShaderId> gather_affine_matvec8_smallk_shader_id(
    Dtype x_dtype,
    Dtype out_dtype) {
  if (x_dtype == bfloat16 && out_dtype == bfloat16) {
    return vulkan::StaticShaderId::gather_affine_matvec8_smallk_bf16_bf16;
  }
  switch (x_dtype) {
    case float32:
      return vulkan::StaticShaderId::gather_affine_matvec8_smallk_f32_f32;
    case float16:
      return vulkan::StaticShaderId::gather_affine_matvec8_smallk_f16_f32;
    default:
      return std::nullopt;
  }
}

bool fused_affine_qmm_prefill_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_AFFINE_QMM");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();
  return enabled;
}

bool gather_affine_matvec8_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_GATHER_MATVEC8");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();
  return enabled;
}

bool gather_affine_matvec8_smallk_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_GATHER_MATVEC8_SMALLK");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();
  return enabled;
}

bool gather_affine_coop_prefill_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_GATHER_QMM_COOP");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();
  return enabled;
}

bool fused_affine_bf16_tiled_prefill_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_AFFINE_BF16_TILED_PREFILL");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();
  return enabled;
}

bool dequantized_bf16_prefill_enabled() {
  static const bool enabled = []() {
    if (const char* env = std::getenv("MLX_VULKAN_DEQUANT_BF16_PREFILL");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();
  return enabled;
}

bool is_row_contiguous_zero_offset(const array& arr) {
  if (arr.ndim() == 0) {
    return arr.offset() == 0;
  }
  return arr.flags().row_contiguous && arr.offset() == 0 &&
      arr.strides(-1) == 1;
}

array ensure_row_contiguous_zero_offset(const array& arr, Stream s) {
  if (is_row_contiguous_zero_offset(arr)) {
    return arr;
  }
  return contiguous_copy_gpu(arr, s);
}

array ensure_float32_row_contiguous(const array& arr, Stream s) {
  if (arr.dtype() == float32 && is_row_contiguous_zero_offset(arr)) {
    return arr;
  }
  array out(arr.shape(), float32, nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(arr, out, CopyType::General, s);
  if (!is_row_contiguous_zero_offset(out)) {
    out = contiguous_copy_gpu(out, s);
  }
  return out;
}

array ensure_float16_row_contiguous(const array& arr, Stream s) {
  if (arr.dtype() == float16 && is_row_contiguous_zero_offset(arr)) {
    return arr;
  }
  array out(arr.shape(), float16, nullptr, {});
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(arr, out, CopyType::General, s);
  if (!is_row_contiguous_zero_offset(out)) {
    out = contiguous_copy_gpu(out, s);
  }
  return out;
}

Shape expanded_quantized_shape(const array& w, int bits) {
  auto out_shape = w.shape();
  out_shape.back() = w.shape(-1) * 32 / bits;
  return out_shape;
}

Shape packed_quantized_shape(const array& x, int bits) {
  auto out_shape = x.shape();
  out_shape.back() = x.shape(-1) * bits / 32;
  return out_shape;
}

Shape quantized_scales_shape(const array& x, int group_size) {
  auto out_shape = x.shape();
  out_shape.back() = x.shape(-1) / group_size;
  return out_shape;
}

bool fp_dequantize_to_float32_fallback(
    const array& w,
    const array& scales,
    array& out,
    Stream s,
    int group_size,
    int bits) {
  if (w.dtype() != uint32 || scales.dtype() != uint8 ||
      out.dtype() != float32) {
    return false;
  }
  if (group_size != 32 || (bits != 4 && bits != 8)) {
    return false;
  }

  array w_work = ensure_row_contiguous_zero_offset(w, s);
  array scales_work = ensure_row_contiguous_zero_offset(scales, s);
  if (!is_row_contiguous_zero_offset(w_work) ||
      !is_row_contiguous_zero_offset(scales_work)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(uint8, float32, false);
  os << R"(
layout(push_constant) uniform PushConstants { uint total_elements; } pc;
layout(set = 0, binding = 0) readonly buffer Packed { uint data[]; } packed_buf;
)";
  os << vulkan::storage_buffer_layout_for_dtype(uint8, 1)
     << " readonly buffer Scales { uint8_t data[]; } scale_buf;\n";
  os << R"(
layout(set = 0, binding = 2) buffer Output { float data[]; } out_buf;

float fp8_e4m3_to_fp32(uint8_t x) {
  uint ux = uint(x);
  uint exponent = (ux >> 3u) & 15u;
  uint mantissa = ux & 7u;
  float result = 0.0;
  if (exponent == 0u) {
    result = float(mantissa) * 0.001953125;
  } else {
    result = exp2(float(int(exponent) - 7)) *
        (1.0 + float(mantissa) * 0.125);
  }
  return (ux & 128u) != 0u ? -result : result;
}

float e8m0_to_fp32(uint8_t x) {
  uint bits = x == uint8_t(0) ? 0x00400000u : (uint(x) << 23);
  return uintBitsToFloat(bits);
}

float fp4_to_float(uint q) {
  switch (q & 15u) {
    case 0u: return 0.0;
    case 1u: return 0.5;
    case 2u: return 1.0;
    case 3u: return 1.5;
    case 4u: return 2.0;
    case 5u: return 3.0;
    case 6u: return 4.0;
    case 7u: return 6.0;
    case 8u: return -0.0;
    case 9u: return -0.5;
    case 10u: return -1.0;
    case 11u: return -1.5;
    case 12u: return -2.0;
    case 13u: return -3.0;
    case 14u: return -4.0;
    default: return -6.0;
  }
}

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;
  uint group_idx = idx / 32u;
  float scale = e8m0_to_fp32(scale_buf.data[group_idx]);
)";
  if (bits == 4) {
    os << R"(
  uint packed = packed_buf.data[idx >> 3u];
  uint shift = (idx & 7u) * 4u;
  out_buf.data[idx] = fp4_to_float((packed >> shift) & 15u) * scale;
}
)";
  } else {
    os << R"(
  uint packed = packed_buf.data[idx >> 2u];
  uint shift = (idx & 3u) * 8u;
  uint8_t q = uint8_t((packed >> shift) & 255u);
  out_buf.data[idx] = fp8_e4m3_to_fp32(q) * scale;
}
)";
  }

  vulkan::DynamicArrayRef arrays[] = {
      {&w_work, 0},
      {&scales_work, 1},
      {&out, 2},
  };
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      bits == 4 ? "dynamic_mxfp4_dequant_f32" : "dynamic_mxfp8_dequant_f32",
      os.str(),
      3,
      arrays,
      kPushConstantSize,
      s);
  const uint32_t total_elements = static_cast<uint32_t>(out.size());
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &total_elements);
  vkCmdDispatch(dispatch.command_buffer, (total_elements + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool fp_gather_qmm_fused(
    const array& w,
    const array& scales,
    const array& x,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    Stream s,
    int bits) {
  if (w.dtype() != uint32 || scales.dtype() != uint8 ||
      !is_supported_quantized_output_dtype(x.dtype()) ||
      lhs_indices.dtype() != uint32 || rhs_indices.dtype() != uint32 ||
      !is_supported_quantized_output_dtype(out.dtype()) ||
      (bits != 4 && bits != 8)) {
    return false;
  }
  if (!is_row_contiguous_zero_offset(w) ||
      !is_row_contiguous_zero_offset(scales) ||
      !is_row_contiguous_zero_offset(x) ||
      !is_row_contiguous_zero_offset(lhs_indices) ||
      !is_row_contiguous_zero_offset(rhs_indices)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  struct PushConstants {
    uint32_t total;
    uint32_t rows;
    uint32_t cols;
    uint32_t K;
    uint32_t packed_row_words;
    uint32_t x_batch_stride;
    uint32_t x_row_stride;
    uint32_t out_batch_stride;
    uint32_t out_row_stride;
    uint32_t scale_matrix_stride;
    uint32_t scale_row_stride;
    uint32_t w_matrix_stride_words;
    uint32_t bits;
  } pc{};
  pc.total = static_cast<uint32_t>(out.size());
  pc.rows = static_cast<uint32_t>(out.shape(-2));
  pc.cols = static_cast<uint32_t>(out.shape(-1));
  pc.K = static_cast<uint32_t>(x.shape(-1));
  pc.packed_row_words = static_cast<uint32_t>(w.strides(-2));
  pc.x_batch_stride = static_cast<uint32_t>(x.shape(-2) * x.shape(-1));
  pc.x_row_stride = static_cast<uint32_t>(x.strides(-2));
  pc.out_batch_stride = pc.rows * pc.cols;
  pc.out_row_stride = static_cast<uint32_t>(out.strides(-2));
  pc.scale_matrix_stride = static_cast<uint32_t>(scales.strides(-3));
  pc.scale_row_stride = static_cast<uint32_t>(scales.strides(-2));
  pc.w_matrix_stride_words = static_cast<uint32_t>(w.strides(-3));
  pc.bits = static_cast<uint32_t>(bits);

  std::ostringstream os;
  vulkan::DynamicShaderPreambleOptions preamble_options;
  preamble_options.dtypes = {uint8, x.dtype(), out.dtype()};
  os << vulkan::emit_dynamic_shader_preamble(preamble_options);
  os << "#define X_TYPE " << vulkan::dtype_to_glsl_storage_type(x.dtype())
     << "\n";
  os << "#define OUT_TYPE " << vulkan::dtype_to_glsl_storage_type(out.dtype())
     << "\n";
  if (bits == 4) {
    os << "#define FP_BITS_4 1\n";
  } else {
    os << "#define FP_BITS_8 1\n";
  }
  if (x.dtype() == bfloat16) {
    os << "#define X_BF16 1\n";
  }
  if (out.dtype() == bfloat16) {
    os << "#define OUT_BF16 1\n";
  }
  os << R"(
layout(push_constant) uniform PushConstants {
  uint total;
  uint rows;
  uint cols;
  uint K;
  uint packed_row_words;
  uint x_batch_stride;
  uint x_row_stride;
  uint out_batch_stride;
  uint out_row_stride;
  uint scale_matrix_stride;
  uint scale_row_stride;
  uint w_matrix_stride_words;
  uint bits;
} p;
layout(set = 0, binding = 0) readonly buffer W { uint data[]; } w;
)";
  os << vulkan::storage_buffer_layout_for_dtype(uint8, 1)
     << " readonly buffer Scales { uint8_t data[]; } scales;\n";
  os << vulkan::storage_buffer_layout_for_dtype(x.dtype(), 2)
     << " readonly buffer X { X_TYPE data[]; } x;\n";
  os << R"(
layout(set = 0, binding = 3) readonly buffer LhsIndices { uint data[]; } lhs;
layout(set = 0, binding = 4) readonly buffer RhsIndices { uint data[]; } rhs;
)";
  os << vulkan::storage_buffer_layout_for_dtype(out.dtype(), 5)
     << " buffer Output { OUT_TYPE data[]; } out_buf;\n";
  os << R"(

float bf16_to_fp32(uint v) {
  return uintBitsToFloat(v << 16u);
}

uint fp32_to_bf16(float v) {
  uint u = floatBitsToUint(v);
  return (u >> 16u) + (((u & 0xFFFFu) + 0x7FFFu) >> 16u);
}

float load_x(uint idx) {
#if defined(X_BF16)
  return bf16_to_fp32(uint(x.data[idx]));
#else
  return float(x.data[idx]);
#endif
}

void store_out(uint idx, float v) {
#if defined(OUT_BF16)
  out_buf.data[idx] = uint16_t(fp32_to_bf16(v));
#else
  out_buf.data[idx] = OUT_TYPE(v);
#endif
}

float fp8_e4m3_to_fp32(uint8_t v) {
  uint ux = uint(v);
  uint exponent = (ux >> 3u) & 15u;
  uint mantissa = ux & 7u;
  float result = exponent == 0u
      ? float(mantissa) * 0.001953125
      : exp2(float(int(exponent) - 7)) * (1.0 + float(mantissa) * 0.125);
  return (ux & 128u) != 0u ? -result : result;
}

float e8m0_to_fp32(uint8_t v) {
  uint bits = v == uint8_t(0) ? 0x00400000u : (uint(v) << 23);
  return uintBitsToFloat(bits);
}

float fp4_to_float(uint q) {
  switch (q & 15u) {
    case 0u: return 0.0;
    case 1u: return 0.5;
    case 2u: return 1.0;
    case 3u: return 1.5;
    case 4u: return 2.0;
    case 5u: return 3.0;
    case 6u: return 4.0;
    case 7u: return 6.0;
    case 8u: return -0.0;
    case 9u: return -0.5;
    case 10u: return -1.0;
    case 11u: return -1.5;
    case 12u: return -2.0;
    case 13u: return -3.0;
    case 14u: return -4.0;
    default: return -6.0;
  }
}

float read_weight(uint row_base, uint k, float scale) {
#if defined(FP_BITS_4)
  uint packed = w.data[row_base + (k >> 3u)];
  uint q = (packed >> ((k & 7u) * 4u)) & 15u;
  return fp4_to_float(q) * scale;
#else
  uint packed = w.data[row_base + (k >> 2u)];
  uint8_t q = uint8_t((packed >> ((k & 3u) * 8u)) & 255u);
  return fp8_e4m3_to_fp32(q) * scale;
#endif
}

void accumulate_word_fp4(inout float acc, uint x_row_base, uint w_row_base, uint word, float scale) {
  uint packed = w.data[w_row_base + word];
  uint k_base = word << 3u;
  if (k_base + 7u < p.K) {
    acc = fma(load_x(x_row_base + k_base + 0u), fp4_to_float((packed >>  0u) & 15u) * scale, acc);
    acc = fma(load_x(x_row_base + k_base + 1u), fp4_to_float((packed >>  4u) & 15u) * scale, acc);
    acc = fma(load_x(x_row_base + k_base + 2u), fp4_to_float((packed >>  8u) & 15u) * scale, acc);
    acc = fma(load_x(x_row_base + k_base + 3u), fp4_to_float((packed >> 12u) & 15u) * scale, acc);
    acc = fma(load_x(x_row_base + k_base + 4u), fp4_to_float((packed >> 16u) & 15u) * scale, acc);
    acc = fma(load_x(x_row_base + k_base + 5u), fp4_to_float((packed >> 20u) & 15u) * scale, acc);
    acc = fma(load_x(x_row_base + k_base + 6u), fp4_to_float((packed >> 24u) & 15u) * scale, acc);
    acc = fma(load_x(x_row_base + k_base + 7u), fp4_to_float((packed >> 28u) & 15u) * scale, acc);
  } else {
    for (uint lane = 0u; lane < 8u && k_base + lane < p.K; ++lane) {
      acc = fma(load_x(x_row_base + k_base + lane), fp4_to_float((packed >> (lane * 4u)) & 15u) * scale, acc);
    }
  }
}

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= p.total) return;

  uint matrix_size = p.rows * p.cols;
  uint batch = idx / matrix_size;
  uint within = idx - batch * matrix_size;
  uint row = within / p.cols;
  uint col = within - row * p.cols;

  uint lhs_batch = lhs.data[batch];
  uint rhs_batch = rhs.data[batch];
  uint w_row_base = rhs_batch * p.w_matrix_stride_words + col * p.packed_row_words;
  uint scale_row_base = rhs_batch * p.scale_matrix_stride + col * p.scale_row_stride;
  uint x_row_base = lhs_batch * p.x_batch_stride + row * p.x_row_stride;

  float acc = 0.0;
  uint num_groups = (p.K + 31u) >> 5u;
  for (uint group = 0u; group < num_groups; ++group) {
    float scale = e8m0_to_fp32(scales.data[scale_row_base + group]);
#if defined(FP_BITS_4)
    uint word_start = group << 2u;
    accumulate_word_fp4(acc, x_row_base, w_row_base, word_start + 0u, scale);
    accumulate_word_fp4(acc, x_row_base, w_row_base, word_start + 1u, scale);
    accumulate_word_fp4(acc, x_row_base, w_row_base, word_start + 2u, scale);
    accumulate_word_fp4(acc, x_row_base, w_row_base, word_start + 3u, scale);
#else
    uint group_start = group << 5u;
    uint group_end = min(group_start + 32u, p.K);
    for (uint k = group_start; k < group_end; ++k) {
      acc = fma(load_x(x_row_base + k), read_weight(w_row_base, k, scale), acc);
    }
#endif
  }
  store_out(batch * p.out_batch_stride + row * p.out_row_stride + col, acc);
}
)";

  vulkan::DynamicArrayRef arrays[] = {
      {&w, 0},
      {&scales, 1},
      {&x, 2},
      {&lhs_indices, 3},
      {&rhs_indices, 4},
      {&out, 5},
  };
  const std::string shader_name =
      std::string(
          bits == 4 ? "dynamic_gather_mxfp4_qmm" : "dynamic_gather_mxfp8_qmm") +
      "_x" + std::to_string(static_cast<int>(x.dtype().val())) + "_o" +
      std::to_string(static_cast<int>(out.dtype().val()));
  const uint32_t grid_x = (pc.total + 255u) / 256u;
  if (!dispatch_grid_within_limits(grid_x)) {
    return false;
  }
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, os.str(), 6, arrays, sizeof(PushConstants), s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(PushConstants),
      &pc);
  vkCmdDispatch(dispatch.command_buffer, grid_x, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool fp_gather_qmm_fused_matvec(
    const array& w,
    const array& scales,
    const array& x,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    Stream s,
    int bits) {
  if (w.dtype() != uint32 || scales.dtype() != uint8 ||
      !is_supported_quantized_output_dtype(x.dtype()) ||
      lhs_indices.dtype() != uint32 || rhs_indices.dtype() != uint32 ||
      !is_supported_quantized_output_dtype(out.dtype()) ||
      (bits != 4 && bits != 8)) {
    return false;
  }
  if (!is_row_contiguous_zero_offset(w) ||
      !is_row_contiguous_zero_offset(scales) ||
      !is_row_contiguous_zero_offset(x) ||
      !is_row_contiguous_zero_offset(lhs_indices) ||
      !is_row_contiguous_zero_offset(rhs_indices)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  const uint32_t rows = static_cast<uint32_t>(out.shape(-2));
  const uint32_t cols = static_cast<uint32_t>(out.shape(-1));
  const uint32_t K = static_cast<uint32_t>(x.shape(-1));
  const uint32_t batches = static_cast<uint32_t>(out.size() / (rows * cols));

  struct PushConstants {
    uint32_t rows;
    uint32_t cols;
    uint32_t K;
    uint32_t packed_row_words;
    uint32_t x_batch_stride;
    uint32_t x_row_stride;
    uint32_t out_batch_stride;
    uint32_t out_row_stride;
    uint32_t scale_matrix_stride;
    uint32_t scale_row_stride;
    uint32_t w_matrix_stride_words;
    uint32_t bits;
  } pc{};
  pc.rows = rows;
  pc.cols = cols;
  pc.K = K;
  pc.packed_row_words = static_cast<uint32_t>(w.strides(-2));
  pc.x_batch_stride = static_cast<uint32_t>(x.shape(-2) * x.shape(-1));
  pc.x_row_stride = static_cast<uint32_t>(x.strides(-2));
  pc.out_batch_stride = rows * cols;
  pc.out_row_stride = static_cast<uint32_t>(out.strides(-2));
  pc.scale_matrix_stride = static_cast<uint32_t>(scales.strides(-3));
  pc.scale_row_stride = static_cast<uint32_t>(scales.strides(-2));
  pc.w_matrix_stride_words = static_cast<uint32_t>(w.strides(-3));
  pc.bits = static_cast<uint32_t>(bits);

  std::ostringstream os;
  vulkan::DynamicShaderPreambleOptions preamble_options;
  preamble_options.dtypes = {uint8, x.dtype(), out.dtype()};
  preamble_options.local_size_x = 64; // One AMD wavefront for better occupancy
  os << vulkan::emit_dynamic_shader_preamble(preamble_options);
  os << "#define X_TYPE " << vulkan::dtype_to_glsl_storage_type(x.dtype())
     << "\n";
  os << "#define OUT_TYPE " << vulkan::dtype_to_glsl_storage_type(out.dtype())
     << "\n";
  if (bits == 4) {
    os << "#define FP_BITS_4 1\n";
  } else {
    os << "#define FP_BITS_8 1\n";
  }
  if (x.dtype() == bfloat16) {
    os << "#define X_BF16 1\n";
  }
  if (out.dtype() == bfloat16) {
    os << "#define OUT_BF16 1\n";
  }
  os << R"(

layout(push_constant) uniform PushConstants {
  uint rows;
  uint cols;
  uint K;
  uint packed_row_words;
  uint x_batch_stride;
  uint x_row_stride;
  uint out_batch_stride;
  uint out_row_stride;
  uint scale_matrix_stride;
  uint scale_row_stride;
  uint w_matrix_stride_words;
  uint bits;
} p;
layout(set = 0, binding = 0) readonly buffer W { uint data[]; } w;
)";
  os << vulkan::storage_buffer_layout_for_dtype(uint8, 1)
     << " readonly buffer Scales { uint8_t data[]; } scales;\n";
  os << vulkan::storage_buffer_layout_for_dtype(x.dtype(), 2)
     << " readonly buffer X { X_TYPE data[]; } x;\n";
  os << R"(
layout(set = 0, binding = 3) readonly buffer LhsIndices { uint data[]; } lhs;
layout(set = 0, binding = 4) readonly buffer RhsIndices { uint data[]; } rhs;
)";
  os << vulkan::storage_buffer_layout_for_dtype(out.dtype(), 5)
     << " buffer Output { OUT_TYPE data[]; } out_buf;\n";
  os << R"(

shared float partial[64];

float bf16_to_fp32(uint v) {
  return uintBitsToFloat(v << 16u);
}

uint fp32_to_bf16(float v) {
  uint u = floatBitsToUint(v);
  return (u >> 16u) + (((u & 0xFFFFu) + 0x7FFFu) >> 16u);
}

float load_x_val(uint idx) {
#if defined(X_BF16)
  return bf16_to_fp32(uint(x.data[idx]));
#else
  return float(x.data[idx]);
#endif
}

void store_out(uint idx, float v) {
#if defined(OUT_BF16)
  out_buf.data[idx] = uint16_t(fp32_to_bf16(v));
#else
  out_buf.data[idx] = OUT_TYPE(v);
#endif
}

float e8m0_to_fp32(uint8_t v) {
  uint bits = v == uint8_t(0) ? 0x00400000u : (uint(v) << 23);
  return uintBitsToFloat(bits);
}

float fp4_to_float(uint q) {
  switch (q & 15u) {
    case 0u: return 0.0;
    case 1u: return 0.5;
    case 2u: return 1.0;
    case 3u: return 1.5;
    case 4u: return 2.0;
    case 5u: return 3.0;
    case 6u: return 4.0;
    case 7u: return 6.0;
    case 8u: return -0.0;
    case 9u: return -0.5;
    case 10u: return -1.0;
    case 11u: return -1.5;
    case 12u: return -2.0;
    case 13u: return -3.0;
    case 14u: return -4.0;
    default: return -6.0;
  }
}

float fp8_e4m3_to_fp32(uint8_t v) {
  uint ux = uint(v);
  uint exponent = (ux >> 3u) & 15u;
  uint mantissa = ux & 7u;
  float result = exponent == 0u
      ? float(mantissa) * 0.001953125
      : exp2(float(int(exponent) - 7)) * (1.0 + float(mantissa) * 0.125);
  return (ux & 128u) != 0u ? -result : result;
}

float decode_weight_at(uint w_row_base, uint k, float scale) {
#if defined(FP_BITS_4)
  uint packed = w.data[w_row_base + (k >> 3u)];
  uint q = (packed >> ((k & 7u) * 4u)) & 15u;
  return fp4_to_float(q) * scale;
#else
  uint packed = w.data[w_row_base + (k >> 2u)];
  uint8_t q = uint8_t((packed >> ((k & 3u) * 8u)) & 255u);
  return fp8_e4m3_to_fp32(q) * scale;
#endif
}

void main() {
  const uint col = gl_WorkGroupID.x;
  const uint row = gl_WorkGroupID.y;
  const uint batch = gl_WorkGroupID.z;
  const uint tid = gl_LocalInvocationID.x;

  if (row >= p.rows || col >= p.cols) {
    return;
  }

  const uint lhs_batch = lhs.data[batch];
  const uint rhs_batch = rhs.data[batch];
  const uint w_row_base = rhs_batch * p.w_matrix_stride_words + col * p.packed_row_words;
  const uint scale_row_base = rhs_batch * p.scale_matrix_stride + col * p.scale_row_stride;
  const uint x_row_base = lhs_batch * p.x_batch_stride + row * p.x_row_stride;

  float acc = 0.0f;
  for (uint k = tid; k < p.K; k += gl_WorkGroupSize.x) {
    const uint group = k >> 5u;
    const float scale = e8m0_to_fp32(scales.data[scale_row_base + group]);
    acc = fma(load_x_val(x_row_base + k), decode_weight_at(w_row_base, k, scale), acc);
  }

  partial[tid] = acc;
  barrier();
  for (uint stride = gl_WorkGroupSize.x >> 1u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    barrier();
  }

  if (tid == 0u) {
    store_out(batch * p.out_batch_stride + row * p.out_row_stride + col, partial[0]);
  }
}
)";

  vulkan::DynamicArrayRef arrays[] = {
      {&w, 0},
      {&scales, 1},
      {&x, 2},
      {&lhs_indices, 3},
      {&rhs_indices, 4},
      {&out, 5},
  };
  const std::string shader_name =
      std::string(
          bits == 4 ? "dynamic_gather_mxfp4_qmm_matvec_wg64"
                    : "dynamic_gather_mxfp8_qmm_matvec_wg64") +
      "_x" + std::to_string(static_cast<int>(x.dtype().val())) + "_o" +
      std::to_string(static_cast<int>(out.dtype().val()));
  // One workgroup per output element (col, row, batch)
  if (!dispatch_grid_within_limits(cols, rows, batches)) {
    return false;
  }
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, os.str(), 6, arrays, sizeof(PushConstants), s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(PushConstants),
      &pc);
  vkCmdDispatch(dispatch.command_buffer, cols, rows, batches);
  vulkan::end_command_recording(s.index);
  return true;
}

} // namespace

namespace vulkan {

void clear_dequantized_weight_cache() {
  std::lock_guard<std::mutex> lock(dequantized_weight_cache_mutex());
  dequantized_weight_cache().clear();
}

bool affine_quantize_from_float32(
    const array& in,
    array& w,
    array& scales,
    array& biases,
    Stream s,
    int group_size,
    int bits) {
  if (in.dtype() != float32 || w.dtype() != uint32 ||
      scales.dtype() != float32 || biases.dtype() != float32) {
    return false;
  }
  if (!is_supported_quantized_bits(bits)) {
    return false;
  }

  array in_work = ensure_row_contiguous_zero_offset(in, s);
  if (!is_row_contiguous_zero_offset(in_work)) {
    return false;
  }

  w.set_data(allocator::malloc(w.nbytes()));
  scales.set_data(allocator::malloc(scales.nbytes()));
  biases.set_data(allocator::malloc(biases.nbytes()));
  if (in.size() == 0) {
    return true;
  }

  AffineQuantPushConstants push_constants{};
  push_constants.ne = static_cast<uint32_t>(in.size());
  push_constants.bits = static_cast<uint32_t>(bits);
  push_constants.group_size = static_cast<uint32_t>(group_size);
  const uint32_t num_groups = static_cast<uint32_t>(scales.size());

  auto command_buffer = vulkan::begin_command_recording(s.index);
  dispatch_affine_quant_op(
      in_work,
      w,
      scales,
      biases,
      StaticShaderId::affine_quantize_f32,
      command_buffer,
      s,
      push_constants,
      {(num_groups + 255u) / 256u, 1, 1});
  vulkan::end_command_recording(s.index);
  return true;
}

bool affine_dequantize_to_float32(
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    Stream s,
    int group_size,
    int bits) {
  if (w.dtype() != uint32 || scales.dtype() != float32 ||
      biases.dtype() != float32 || out.dtype() != float32) {
    return false;
  }
  if (!is_supported_quantized_bits(bits)) {
    return false;
  }

  array w_work = ensure_row_contiguous_zero_offset(w, s);
  array scales_work = ensure_row_contiguous_zero_offset(scales, s);
  array biases_work = ensure_row_contiguous_zero_offset(biases, s);

  if (!is_row_contiguous_zero_offset(w_work) ||
      !is_row_contiguous_zero_offset(scales_work) ||
      !is_row_contiguous_zero_offset(biases_work)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  AffineDequantPushConstants push_constants{};
  push_constants.ne = static_cast<uint32_t>(out.size());
  push_constants.bits = static_cast<uint32_t>(bits);
  push_constants.group_size = static_cast<uint32_t>(group_size);

  auto command_buffer = vulkan::begin_command_recording(s.index);
  dispatch_affine_dequant_op(
      w_work,
      scales_work,
      biases_work,
      out,
      StaticShaderId::affine_dequantize_f32,
      command_buffer,
      s,
      push_constants,
      {(push_constants.ne + 255u) / 256u, 1, 1});
  vulkan::end_command_recording(s.index);
  return true;
}

bool affine_dequantize_to_bfloat16(
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    Stream s,
    int group_size,
    int bits) {
  if (w.dtype() != uint32 || scales.dtype() != bfloat16 ||
      biases.dtype() != bfloat16 || out.dtype() != bfloat16) {
    return false;
  }
  if (!is_supported_quantized_bits(bits)) {
    return false;
  }

  array w_work = ensure_row_contiguous_zero_offset(w, s);
  array scales_work = ensure_row_contiguous_zero_offset(scales, s);
  array biases_work = ensure_row_contiguous_zero_offset(biases, s);

  if (!is_row_contiguous_zero_offset(w_work) ||
      !is_row_contiguous_zero_offset(scales_work) ||
      !is_row_contiguous_zero_offset(biases_work)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  AffineDequantPushConstants push_constants{};
  push_constants.ne = static_cast<uint32_t>(out.size());
  push_constants.bits = static_cast<uint32_t>(bits);
  push_constants.group_size = static_cast<uint32_t>(group_size);

  auto command_buffer = vulkan::begin_command_recording(s.index);
  dispatch_affine_dequant_op(
      w_work,
      scales_work,
      biases_work,
      out,
      StaticShaderId::affine_dequantize_bf16,
      command_buffer,
      s,
      push_constants,
      {(push_constants.ne + 255u) / 256u, 1, 1});
  vulkan::end_command_recording(s.index);
  return true;
}

bool nvfp4_dequantize_to_float32(
    const array& w,
    const array& scales,
    const std::optional<array>& global_scale,
    array& out,
    Stream s) {
  if (w.dtype() != uint32 || scales.dtype() != uint8 ||
      out.dtype() != float32) {
    return false;
  }
  if (global_scale.has_value() && global_scale->dtype() != float32) {
    return false;
  }

  array w_work = ensure_row_contiguous_zero_offset(w, s);
  array scales_work = ensure_row_contiguous_zero_offset(scales, s);
  array global_scale_work = global_scale.has_value()
      ? ensure_float32_row_contiguous(*global_scale, s)
      : scales_work;
  if (!is_row_contiguous_zero_offset(w_work) ||
      !is_row_contiguous_zero_offset(scales_work) ||
      !is_row_contiguous_zero_offset(global_scale_work)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  Nvfp4DequantPushConstants push_constants{};
  push_constants.ne = static_cast<uint32_t>(out.size());
  push_constants.has_global_scale = global_scale.has_value() ? 1u : 0u;

  auto command_buffer = vulkan::begin_command_recording(s.index);
  dispatch_nvfp4_dequant_op(
      w_work,
      scales_work,
      global_scale_work,
      out,
      StaticShaderId::dequant_nvfp4_f32,
      command_buffer,
      s,
      push_constants,
      {(push_constants.ne + 255u) / 256u, 1, 1});
  vulkan::end_command_recording(s.index);
  return true;
}

bool fp_dequantize_to_float32(
    const array& w,
    const array& scales,
    array& out,
    Stream s,
    int group_size,
    int bits) {
  return fp_dequantize_to_float32_fallback(w, scales, out, s, group_size, bits);
}

bool fp_quantize_from_float32(
    const array& in,
    array& w,
    array& scales,
    Stream s,
    int group_size,
    int bits) {
  if (in.dtype() != float32 || w.dtype() != uint32 || scales.dtype() != uint8) {
    return false;
  }
  if (group_size != 32 || (bits != 4 && bits != 8)) {
    return false;
  }
  if ((in.size() % group_size) != 0) {
    return false;
  }
  if (w.size() != in.size() * bits / 32 ||
      scales.size() != in.size() / group_size) {
    return false;
  }

  array in_work = ensure_row_contiguous_zero_offset(in, s);
  if (!is_row_contiguous_zero_offset(in_work)) {
    return false;
  }

  w.set_data(allocator::malloc(w.nbytes()));
  scales.set_data(allocator::malloc(scales.nbytes()));
  if (in.size() == 0) {
    return true;
  }

  const uint32_t num_groups = static_cast<uint32_t>(scales.size());
  if (static_cast<size_t>(num_groups) != scales.size()) {
    return false;
  }

  vulkan::DynamicShaderPreambleOptions preamble_options;
  preamble_options.dtypes = {float32, uint8};
  preamble_options.local_size_x = 32;
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(preamble_options);
  os << R"(
layout(push_constant) uniform PushConstants { uint num_groups; } pc;
layout(set = 0, binding = 0) readonly buffer Input { float data[]; } in_buf;
layout(set = 0, binding = 1) buffer Packed { uint data[]; } packed_buf;
)";
  os << vulkan::storage_buffer_layout_for_dtype(uint8, 2)
     << " buffer Scales { uint8_t data[]; } scale_buf;\n";
  os << R"(
shared float group_abs[32];
shared uint group_q[32];

uint fp32_to_e8m0(float x) {
  if (isinf(x)) {
    return 255u;
  }
  if (!(x > 0.0)) {
    return 0u;
  }
  int n = int(round(log2(x)));
  n = clamp(n, -127, 127);
  return uint(n + 127);
}

float e8m0_to_fp32(uint x) {
  uint bits = x == 0u ? 0x00400000u : (x << 23);
  return uintBitsToFloat(bits);
}

uint to_fp8_e4m3(float x) {
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
  return result | (sign >> 24);
}

uint fp32_to_fp4_e2m1(float x) {
  uint sign_bit = (floatBitsToUint(x) & 0x80000000u) != 0u ? 8u : 0u;
  x = abs(x);

  uint bits;
  if (x > 5.0) {
    bits = 7u;
  } else if (x >= 3.5) {
    bits = 6u;
  } else if (x > 2.5) {
    bits = 5u;
  } else if (x >= 1.75) {
    bits = 4u;
  } else if (x > 1.25) {
    bits = 3u;
  } else if (x >= 0.75) {
    bits = 2u;
  } else if (x > 0.25) {
    bits = 1u;
  } else {
    bits = 0u;
  }
  return bits | sign_bit;
}

void main() {
  const uint group = gl_WorkGroupID.x;
  const uint tid = gl_LocalInvocationID.x;
  if (group >= pc.num_groups) {
    return;
  }

  const uint base = group * 32u;
  const float x = in_buf.data[base + tid];
  group_abs[tid] = abs(x);
  barrier();

  for (uint stride = 16u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      group_abs[tid] = max(group_abs[tid], group_abs[tid + stride]);
    }
    barrier();
  }

)";
  os << "  const float scale_base = group_abs[0] / "
     << (bits == 4 ? "6.0" : "448.0") << ";\n";
  os << R"(
  const uint scale_enc = fp32_to_e8m0(scale_base);
  const float scale = e8m0_to_fp32(scale_enc);
  if (tid == 0u) {
    scale_buf.data[group] = uint8_t(scale_enc);
  }

  const float normalized = scale == 0.0 ? 0.0 : x / scale;
)";
  if (bits == 4) {
    os << R"(
  group_q[tid] = fp32_to_fp4_e2m1(normalized);
  barrier();

  if (tid < 4u) {
    const uint qbase = tid * 8u;
    uint packed = 0u;
    for (uint i = 0u; i < 8u; ++i) {
      packed |= (group_q[qbase + i] & 15u) << (i * 4u);
    }
    packed_buf.data[group * 4u + tid] = packed;
  }
}
)";
  } else {
    os << R"(
  group_q[tid] = to_fp8_e4m3(normalized);
  barrier();

  if (tid < 8u) {
    const uint qbase = tid * 4u;
    uint packed = 0u;
    for (uint i = 0u; i < 4u; ++i) {
      packed |= (group_q[qbase + i] & 255u) << (i * 8u);
    }
    packed_buf.data[group * 8u + tid] = packed;
  }
}
)";
  }

  vulkan::DynamicArrayRef arrays[] = {
      {&in_work, 0},
      {&w, 1},
      {&scales, 2},
  };
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      bits == 4 ? "dynamic_mxfp4_quant_f32" : "dynamic_mxfp8_quant_f32",
      os.str(),
      3,
      arrays,
      kPushConstantSize,
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &num_groups);
  vkCmdDispatch(dispatch.command_buffer, num_groups, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool fp_quantize_dequantize_to_float32(
    const array& in,
    array& out,
    Stream s,
    int group_size,
    int bits) {
  if (in.dtype() != float32 || out.dtype() != float32) {
    return false;
  }
  if (group_size != 32 || (bits != 4 && bits != 8)) {
    return false;
  }
  if (in.shape() != out.shape() || (in.size() % group_size) != 0) {
    return false;
  }

  array in_work = ensure_row_contiguous_zero_offset(in, s);
  if (!is_row_contiguous_zero_offset(in_work)) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(float32, float32, false);
  os << R"(
layout(push_constant) uniform PushConstants { uint total_elements; } pc;
layout(set = 0, binding = 0) readonly buffer Input { float data[]; } in_buf;
layout(set = 0, binding = 1) buffer Output { float data[]; } out_buf;

uint to_fp8_e4m3(float x) {
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
  return result | (sign >> 24);
}

float fp8_e4m3_to_fp32(uint x) {
  uint exponent = (x >> 3u) & 15u;
  uint mantissa = x & 7u;
  float result = 0.0;
  if (exponent == 0u) {
    result = float(mantissa) * 0.001953125;
  } else {
    result = exp2(float(int(exponent) - 7)) *
        (1.0 + float(mantissa) * 0.125);
  }
  return (x & 128u) != 0u ? -result : result;
}

float fp4_to_float(uint q) {
  switch (q & 15u) {
    case 0u: return 0.0;
    case 1u: return 0.5;
    case 2u: return 1.0;
    case 3u: return 1.5;
    case 4u: return 2.0;
    case 5u: return 3.0;
    case 6u: return 4.0;
    case 7u: return 6.0;
    case 8u: return -0.0;
    case 9u: return -0.5;
    case 10u: return -1.0;
    case 11u: return -1.5;
    case 12u: return -2.0;
    case 13u: return -3.0;
    case 14u: return -4.0;
    default: return -6.0;
  }
}

uint fp32_to_fp4_e2m1(float x) {
  uint sign_bit = (floatBitsToUint(x) & 0x80000000u) != 0u ? 8u : 0u;
  x = abs(x);

  uint bits;
  if (x > 5.0) {
    bits = 7u;
  } else if (x >= 3.5) {
    bits = 6u;
  } else if (x > 2.5) {
    bits = 5u;
  } else if (x >= 1.75) {
    bits = 4u;
  } else if (x > 1.25) {
    bits = 3u;
  } else if (x >= 0.75) {
    bits = 2u;
  } else if (x > 0.25) {
    bits = 1u;
  } else {
    bits = 0u;
  }
  return bits | sign_bit;
}

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;

  uint group_base = (idx / 32u) * 32u;
  float max_abs = 0.0;
  for (uint i = 0u; i < 32u; ++i) {
    max_abs = max(max_abs, abs(in_buf.data[group_base + i]));
  }
)";
  if (bits == 4) {
    os << R"(
  float scale = max_abs == 0.0 ? 1.0 : exp2(round(log2(max_abs / 6.0)));
  float normalized = in_buf.data[idx] / scale;
  out_buf.data[idx] = scale * fp4_to_float(fp32_to_fp4_e2m1(normalized));
}
)";
  } else {
    os << R"(
  float scale = max_abs == 0.0 ? 1.0 : exp2(round(log2(max_abs / 448.0)));
  float normalized = in_buf.data[idx] / scale;
  out_buf.data[idx] = scale * fp8_e4m3_to_fp32(to_fp8_e4m3(normalized));
}
)";
  }

  vulkan::DynamicArrayRef arrays[] = {
      {&in_work, 0},
      {&out, 1},
  };
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      bits == 4 ? "dynamic_mxfp4_qdq_f32" : "dynamic_mxfp8_qdq_f32",
      os.str(),
      2,
      arrays,
      kPushConstantSize,
      s);
  const uint32_t total_elements = static_cast<uint32_t>(out.size());
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &total_elements);
  vkCmdDispatch(dispatch.command_buffer, (total_elements + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool nvfp4_quantize_from_float32(
    const array& in,
    array& w,
    array& scales,
    const std::optional<array>& global_scale,
    Stream s) {
  if (in.dtype() != float32 || w.dtype() != uint32 || scales.dtype() != uint8) {
    return false;
  }
  if (global_scale.has_value() && global_scale->dtype() != float32) {
    return false;
  }

  array in_work = ensure_row_contiguous_zero_offset(in, s);
  array global_scale_work = global_scale.has_value()
      ? ensure_float32_row_contiguous(*global_scale, s)
      : in_work;
  if (!is_row_contiguous_zero_offset(in_work) ||
      !is_row_contiguous_zero_offset(global_scale_work)) {
    return false;
  }

  w.set_data(allocator::malloc(w.nbytes()));
  scales.set_data(allocator::malloc(scales.nbytes()));
  if (in.size() == 0) {
    return true;
  }

  Nvfp4QuantPushConstants push_constants{};
  push_constants.ne = static_cast<uint32_t>(in.size());
  push_constants.has_global_scale = global_scale.has_value() ? 1u : 0u;
  const uint32_t num_groups = static_cast<uint32_t>(scales.size());
  if (static_cast<size_t>(num_groups) != scales.size()) {
    return false;
  }

  const auto limits = VulkanContext::get()
                          .physical_device()
                          .getProperties()
                          .limits;
  const uint32_t max_groups_x = std::min(
      kMaxComputeWorkGroupCount, limits.maxComputeWorkGroupCount[0]);

  auto command_buffer = begin_command_recording(s.index);
  for (uint32_t base_group = 0; base_group < num_groups;
       base_group += max_groups_x) {
    const uint32_t chunk =
        std::min(max_groups_x, num_groups - base_group);
    push_constants.base_group = base_group;
    dispatch_nvfp4_quant_op(
        in_work,
        w,
        scales,
        global_scale_work,
        StaticShaderId::quantize_nvfp4_f32,
        command_buffer,
        s,
        push_constants,
        {chunk, 1, 1});
  }
  end_command_recording(s.index);
  return true;
}

} // namespace vulkan

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (mode_ == QuantizationMode::Affine) {
    if (inputs.size() != 4) {
      throw std::runtime_error(
          "[QuantizedMatmul::eval_gpu] Expected x, w, scales, biases.");
    }
  } else if (inputs.size() != 3) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Expected x, w, scales.");
  }
  if (!is_supported_quantized_bits(bits_)) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Unsupported quantization bits on Vulkan.");
  }
  if (!is_supported_quantized_output_dtype(inputs[0].dtype()) ||
      !is_supported_quantized_output_dtype(out.dtype())) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Only float16, bfloat16, and float32 are supported.");
  }

  auto& s = stream();
  auto trace_qmm = [&](std::string_view kind, std::string_view detail) {
    if (!trace_fallback_enabled()) {
      return;
    }
    std::ostringstream oss;
    oss << "primitive=QuantizedMatmul kind=" << kind
        << " x_shape=" << inputs[0].shape() << " x_dtype=" << inputs[0].dtype()
        << " w_shape=" << inputs[1].shape()
        << " scales_dtype=" << inputs[2].dtype() << " out_shape=" << out.shape()
        << " out_dtype=" << out.dtype() << " bits=" << bits_
        << " group_size=" << group_size_ << " transpose=" << transpose_;
    if (!detail.empty()) {
      oss << ' ' << detail;
    }
    trace_fallback(oss.str());
  };
  array x = ensure_row_contiguous_zero_offset(inputs[0], s);
  array w = ensure_row_contiguous_zero_offset(inputs[1], s);

  const bool vector_lhs = x.ndim() == 1;
  const bool flatten_lhs_batches = x.ndim() > 2 && w.ndim() == 2;
  const bool decode_lhs = flatten_lhs_batches && x.shape(-2) == 1;
  array x_mat = x;
  if (vector_lhs) {
    Shape mat_shape = {1, x.shape(0)};
    x_mat = array(mat_shape, x.dtype(), nullptr, {});
    x_mat.copy_shared_buffer(
        x, make_contiguous_strides(mat_shape), x.flags(), x.size());
  } else if (flatten_lhs_batches) {
    Shape flat_shape = {static_cast<int>(x.size() / x.shape(-1)), x.shape(-1)};
    x_mat = array(flat_shape, x.dtype(), nullptr, {});
    x_mat.copy_shared_buffer(
        x, make_contiguous_strides(flat_shape), x.flags(), x.size());
  }

  const uint32_t qmm_rows =
      x_mat.ndim() == 2 ? static_cast<uint32_t>(x_mat.shape(-2)) : 0u;

  auto finalize_bf16_output = [&](array& out_work) {
    if (vector_lhs || flatten_lhs_batches) {
      array::Flags flags = out.flags();
      flags.contiguous = true;
      flags.row_contiguous = true;
      if (vector_lhs) {
        flags.col_contiguous = true;
      } else {
        auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
        flags.col_contiguous =
            out.size() <= 1 || out.size() == *max_dim;
      }
      out.copy_shared_buffer(
          out_work,
          make_contiguous_strides(out.shape()),
          flags,
          out.size());
      if (!detail::in_tracing() && !detail::retain_graph()) {
        out.detach();
      }
      out.set_status(array::Status::evaluated);
      return;
    }

    out.copy_shared_buffer(out_work);
    out.set_status(array::Status::evaluated);
  };

  const bool enable_fused_decode_qmm = []() {
    if (const char* env = std::getenv("MLX_VULKAN_FUSED_AFFINE_QMM");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();

  if (mode_ == QuantizationMode::Affine && enable_fused_decode_qmm &&
      transpose_ && bits_ == 8 && x_mat.dtype() == bfloat16 &&
      out.dtype() == bfloat16 && inputs[2].dtype() == bfloat16 &&
      inputs[3].dtype() == bfloat16 && x_mat.ndim() == 2 && w.ndim() == 2) {
    array scales_bf16 = ensure_row_contiguous_zero_offset(inputs[2], s);
    array biases_bf16 = ensure_row_contiguous_zero_offset(inputs[3], s);
    if (scales_bf16.ndim() == 2 && biases_bf16.ndim() == 2 &&
        is_row_contiguous_zero_offset(x_mat) &&
        is_row_contiguous_zero_offset(w) &&
        is_row_contiguous_zero_offset(scales_bf16) &&
        is_row_contiguous_zero_offset(biases_bf16)) {
      const uint32_t rows = static_cast<uint32_t>(x_mat.shape(-2));
      const uint32_t cols = static_cast<uint32_t>(w.shape(-2));
      const uint32_t k = static_cast<uint32_t>(x_mat.shape(-1));
      const uint32_t num_groups = static_cast<uint32_t>(scales_bf16.shape(-1));
      if (cols == static_cast<uint32_t>(out.shape(-1)) &&
          static_cast<uint32_t>(w.shape(-1) * 32 / bits_) == k &&
          num_groups ==
              static_cast<uint32_t>((k + group_size_ - 1) / group_size_)) {
        const bool use_dequantized_prefill = rows >= 256 && !decode_lhs &&
            dequantized_bf16_prefill_enabled();
        if (use_dequantized_prefill) {
          array w_deq(
              expanded_quantized_shape(w, bits_), bfloat16, nullptr, {});
          if (!vulkan::affine_dequantize_to_bfloat16(
                  w,
                  scales_bf16,
                  biases_bf16,
                  w_deq,
                  s,
                  group_size_,
                  bits_)) {
            throw std::runtime_error(
                "[QuantizedMatmul::eval_gpu] Failed to dequantize BF16 weights on Vulkan.");
          }
          w_deq.set_status(array::Status::evaluated);

          array rhs_bf16 = swapaxes_in_eval(w_deq, -1, -2);
          array out_work(
              (vector_lhs || flatten_lhs_batches)
                  ? Shape{static_cast<int>(rows), out.shape(-1)}
                  : out.shape(),
              bfloat16,
              nullptr,
              {});
          if (!try_eval_matmul_vulkan({x_mat, rhs_bf16}, out_work, s)) {
            throw std::runtime_error(
                "[QuantizedMatmul::eval_gpu] Failed to dispatch BF16 prefill matmul.");
          }
          finalize_bf16_output(out_work);
          trace_qmm(
              "dequant_bf16_prefill",
              (vector_lhs || flatten_lhs_batches) ? "reshaped_output=1"
                                                  : "reshaped_output=0");
          return;
        }

        array out_work(
            (vector_lhs || flatten_lhs_batches)
                ? Shape{static_cast<int>(rows), out.shape(-1)}
                : out.shape(),
            bfloat16,
            nullptr,
            {});
        out_work.set_data(allocator::malloc(out_work.nbytes()));
        if (out_work.size() != 0) {
          vulkan::FusedAffineMatmulPushConstants push_constants{};
          push_constants.rows = rows;
          push_constants.cols = cols;
          push_constants.K = k;
          push_constants.packed_row_bytes =
              static_cast<uint32_t>(w.strides(-2) * sizeof(uint32_t));
          push_constants.x_row_stride =
              static_cast<uint32_t>(x_mat.strides(-2));
          push_constants.out_row_stride =
              static_cast<uint32_t>(out_work.strides(-2));
          push_constants.scale_row_stride =
              static_cast<uint32_t>(scales_bf16.strides(-2));
          push_constants.bias_row_stride =
              static_cast<uint32_t>(biases_bf16.strides(-2));
          push_constants.bits = static_cast<uint32_t>(bits_);
          push_constants.group_size = static_cast<uint32_t>(group_size_);
          push_constants.num_groups = num_groups;

          const bool use_decode_matvec = rows == 1 || decode_lhs;
          const bool use_tiled_prefill = rows > 1 && !decode_lhs &&
              group_size_ >= 32 && (group_size_ % 32) == 0 &&
              fused_affine_bf16_tiled_prefill_enabled();
          const bool use_large_n_tile = use_tiled_prefill && cols >= 1024;
          const auto shader_id = use_decode_matvec
              ? vulkan::StaticShaderId::fused_affine_matvec8_bf16_bf16
              : use_large_n_tile
              ? vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16_tiled_n32
              : use_tiled_prefill
              ? vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16_tiled
              : vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16;
          const std::array<uint32_t, 3> grid = use_decode_matvec
              ? std::array<uint32_t, 3>{cols, rows, 1u}
              : shader_id ==
                      vulkan::StaticShaderId::
                          fused_affine_qmm_bf16_bf16_tiled_n32
              ? std::array<
                    uint32_t,
                    3>{(cols + 31u) / 32u, (rows + 31u) / 32u, 1u}
              : shader_id ==
                      vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16_tiled
              ? std::array<
                    uint32_t,
                    3>{(cols + 15u) / 16u, (rows + 31u) / 32u, 1u}
              : std::array<uint32_t, 3>{
                    (cols + 15u) / 16u, (rows + 15u) / 16u, 1u};

          auto command_buffer = vulkan::begin_command_recording(s.index);
          vulkan::dispatch_fused_affine_matmul_op(
              w,
              scales_bf16,
              biases_bf16,
              x_mat,
              out_work,
              shader_id,
              command_buffer,
              s,
              push_constants,
              grid);
          vulkan::end_command_recording(s.index);
        }

        finalize_bf16_output(out_work);
        trace_qmm(
            "fused_bf16",
            (vector_lhs || flatten_lhs_batches) ? "reshaped_output=1"
                                                : "reshaped_output=0");
        return;
      }
    }
  }

  array scales = mode_ == QuantizationMode::Affine
      ? ensure_float32_row_contiguous(inputs[2], s)
      : inputs[2];
  std::optional<array> biases = mode_ == QuantizationMode::Affine
      ? std::make_optional(ensure_float32_row_contiguous(inputs[3], s))
      : std::nullopt;

  if (x_mat.dtype() == bfloat16) {
    x_mat = ensure_float32_row_contiguous(x_mat, s);
  }
  const bool large_qmm_edge_dim = x_mat.ndim() == 2 &&
      (x_mat.shape(-2) > std::numeric_limits<int16_t>::max() ||
       out.shape(-1) > std::numeric_limits<int16_t>::max());
  auto fused_shader = large_qmm_edge_dim
      ? std::optional<vulkan::StaticShaderId>{}
      : decode_lhs ? (bits_ == 8 ? fused_affine_matvec8_shader_id(x_mat.dtype())
                                 : fused_affine_matvec_shader_id(x_mat.dtype()))
      : qmm_rows > 1 && fused_affine_qmm_prefill_enabled()
      ? fused_affine_qmm_shader_id(x_mat.dtype())
      : (bits_ == 8 ? fused_affine_matvec8_shader_id(x_mat.dtype())
                    : fused_affine_matvec_shader_id(x_mat.dtype()));
  const Dtype out_work_dtype = float32;

  array out_work(
      (vector_lhs || flatten_lhs_batches)
          ? Shape{static_cast<int>(x_mat.shape(0)), out.shape(-1)}
          : out.shape(),
      out_work_dtype,
      nullptr,
      {});

  bool fused_dispatched = false;
  if (mode_ == QuantizationMode::Affine && enable_fused_decode_qmm &&
      transpose_ && fused_shader.has_value() && x_mat.ndim() == 2 &&
      w.ndim() == 2 && scales.ndim() == 2 && biases->ndim() == 2 &&
      is_row_contiguous_zero_offset(x_mat) &&
      is_row_contiguous_zero_offset(w) &&
      is_row_contiguous_zero_offset(scales) &&
      is_row_contiguous_zero_offset(*biases)) {
    const uint32_t rows = static_cast<uint32_t>(out_work.shape(-2));
    const uint32_t cols = static_cast<uint32_t>(out_work.shape(-1));
    const uint32_t k = static_cast<uint32_t>(x_mat.shape(-1));
    const uint32_t num_groups = static_cast<uint32_t>(scales.shape(-1));
    const bool decode_like_rows = rows == 1 || decode_lhs;
    const bool prefill_like_rows =
        rows > 1 && !decode_lhs && fused_affine_qmm_prefill_enabled();

    if ((decode_like_rows || prefill_like_rows) &&
        rows == static_cast<uint32_t>(x_mat.shape(-2)) &&
        cols == static_cast<uint32_t>(w.shape(-2)) &&
        num_groups ==
            static_cast<uint32_t>((k + group_size_ - 1) / group_size_)) {
      try {
        out_work.set_data(allocator::malloc(out_work.nbytes()));
        if (out_work.size() != 0) {
          vulkan::FusedAffineMatmulPushConstants push_constants{};
          push_constants.rows = rows;
          push_constants.cols = cols;
          push_constants.K = k;
          push_constants.packed_row_bytes =
              static_cast<uint32_t>(w.strides(-2) * sizeof(uint32_t));
          push_constants.x_row_stride =
              static_cast<uint32_t>(x_mat.strides(-2));
          push_constants.out_row_stride =
              static_cast<uint32_t>(out_work.strides(-2));
          push_constants.scale_row_stride =
              static_cast<uint32_t>(scales.strides(-2));
          push_constants.bias_row_stride =
              static_cast<uint32_t>(biases->strides(-2));
          push_constants.bits = static_cast<uint32_t>(bits_);
          push_constants.group_size = static_cast<uint32_t>(group_size_);
          push_constants.num_groups = num_groups;

          const std::array<uint32_t, 3> grid = prefill_like_rows
              ? std::array<
                    uint32_t,
                    3>{(cols + 15u) / 16u, (rows + 31u) / 32u, 1u}
              : std::array<uint32_t, 3>{cols, rows, 1u};

          auto command_buffer = vulkan::begin_command_recording(s.index);
          vulkan::dispatch_fused_affine_matmul_op(
              w,
              scales,
              *biases,
              x_mat,
              out_work,
              *fused_shader,
              command_buffer,
              s,
              push_constants,
              grid);
          vulkan::end_command_recording(s.index);
        }
        fused_dispatched = true;
        trace_qmm(
            prefill_like_rows ? "fused_prefill" : "fused_decode",
            "staged_float32=1");
      } catch (const std::runtime_error&) {
        fused_dispatched = false;
      }
    }
  }

  if (!fused_dispatched) {
    trace_qmm("dequant_fallback", "reason=fused_conditions_not_met");
    std::optional<array> cache_biases = mode_ == QuantizationMode::Affine
        ? std::make_optional(inputs[3])
        : std::nullopt;
    auto cached_w_deq = find_cached_dequantized_weight(
        w, inputs[2], cache_biases, mode_, group_size_, bits_);
    array w_deq = cached_w_deq.value_or(
        array(expanded_quantized_shape(w, bits_), float32, nullptr, {}));
    if (!cached_w_deq.has_value()) {
      if (mode_ == QuantizationMode::Affine &&
          !vulkan::affine_dequantize_to_float32(
              w, scales, *biases, w_deq, s, group_size_, bits_)) {
        throw std::runtime_error(
            "[QuantizedMatmul::eval_gpu] Failed to dequantize weights on Vulkan.");
      }
      if (mode_ == QuantizationMode::Nvfp4 &&
          !vulkan::nvfp4_dequantize_to_float32(
              w, inputs[2], std::nullopt, w_deq, s)) {
        throw std::runtime_error(
            "[QuantizedMatmul::eval_gpu] Failed to dequantize FP weights on Vulkan.");
      }
      if ((mode_ == QuantizationMode::Mxfp4 ||
           mode_ == QuantizationMode::Mxfp8) &&
          !vulkan::fp_dequantize_to_float32(
              w, inputs[2], w_deq, s, group_size_, bits_)) {
        throw std::runtime_error(
            "[QuantizedMatmul::eval_gpu] Failed to dequantize FP weights on Vulkan.");
      }
      w_deq.set_status(array::Status::evaluated);
      cache_dequantized_weight(
          w, inputs[2], cache_biases, mode_, group_size_, bits_, w_deq);
    }

    array rhs_f32 = transpose_ ? swapaxes_in_eval(w_deq, -1, -2) : w_deq;
    rhs_f32 = ensure_row_contiguous_zero_offset(rhs_f32, s);

    bool lowp_dispatched = false;
    if (x_mat.dtype() != float32) {
      array lhs_lowp = ensure_row_contiguous_zero_offset(x_mat, s);
      array rhs_lowp(rhs_f32.shape(), x_mat.dtype(), nullptr, {});
      rhs_lowp.set_data(allocator::malloc(rhs_lowp.nbytes()));
      copy_gpu(rhs_f32, rhs_lowp, CopyType::General, s);

      array out_lowp(out_work.shape(), x_mat.dtype(), nullptr, {});
      if (try_eval_matmul_vulkan({lhs_lowp, rhs_lowp}, out_lowp, s)) {
        out_work.set_data(allocator::malloc(out_work.nbytes()));
        copy_gpu(out_lowp, out_work, CopyType::General, s);
        lowp_dispatched = true;
      }
    }

    if (!lowp_dispatched) {
      array x_mat_f32 = ensure_float32_row_contiguous(x_mat, s);
      if (!try_eval_matmul_vulkan({x_mat_f32, rhs_f32}, out_work, s)) {
        throw std::runtime_error(
            "[QuantizedMatmul::eval_gpu] Failed to dispatch Vulkan matmul.");
      }
    }

    if (out_work.dtype() != float32) {
      throw std::runtime_error(
          "[QuantizedMatmul::eval_gpu] Internal error: expected float32 work buffer.");
    }
  }

  if (vector_lhs || flatten_lhs_batches) {
    array::Flags flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    if (vector_lhs) {
      flags.col_contiguous = true;
    } else {
      auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
      flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    }

    if (out_work.dtype() == out.dtype()) {
      out.copy_shared_buffer(
          out_work, make_contiguous_strides(out.shape()), flags, out.size());
      if (!detail::in_tracing() && !detail::retain_graph()) {
        out.detach();
      }
      out.set_status(array::Status::evaluated);
      return;
    }

    array out_view(out.shape(), out_work.dtype(), nullptr, {});
    out_view.copy_shared_buffer(
        out_work, make_contiguous_strides(out.shape()), flags, out.size());
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(out_view, out, CopyType::GeneralGeneral, s);
    if (!detail::in_tracing() && !detail::retain_graph()) {
      out.detach();
    }
    out.set_status(array::Status::evaluated);
    return;
  }

  if (out_work.dtype() == out.dtype()) {
    out.copy_shared_buffer(out_work);
    out.set_status(array::Status::evaluated);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(out_work, out, CopyType::General, s);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  bool w_quantized = inputs[1].dtype() == uint32;
  if (!w_quantized) {
    throw std::runtime_error("[QQMatmul::eval_gpu] Not implemented on Vulkan.");
  }

  auto& s = stream();
  auto mode = quantization_mode_to_string(mode_);

  std::optional<array> global_scale_x = std::nullopt;
  std::optional<array> global_scale_w = std::nullopt;
  if (mode_ == QuantizationMode::Nvfp4 && inputs.size() >= 5) {
    global_scale_x = inputs[3];
    global_scale_w = inputs[4];
  }

  if (mode_ == QuantizationMode::Mxfp4 || mode_ == QuantizationMode::Mxfp8) {
    array x_f32 = ensure_float32_row_contiguous(inputs[0], s);
    array xhat(x_f32.shape(), float32, nullptr, {});
    if (!vulkan::fp_quantize_dequantize_to_float32(
            x_f32, xhat, s, group_size_, bits_)) {
      throw std::runtime_error(
          "[QQMatmul::eval_gpu] Failed to quantize-dequantize lhs on Vulkan.");
    }

    array what(
        expanded_quantized_shape(inputs[1], bits_), float32, nullptr, {});
    if (!vulkan::fp_dequantize_to_float32(
            inputs[1], inputs[2], what, s, group_size_, bits_)) {
      throw std::runtime_error(
          "[QQMatmul::eval_gpu] Failed to dequantize rhs on Vulkan.");
    }

    array rhs =
        ensure_row_contiguous_zero_offset(swapaxes_in_eval(what, -1, -2), s);
    array result(out.shape(), float32, nullptr, {});
    if (!try_eval_matmul_vulkan({xhat, rhs}, result, s)) {
      throw std::runtime_error(
          "[QQMatmul::eval_gpu] Failed to dispatch Vulkan fallback matmul.");
    }

    if (result.dtype() == out.dtype()) {
      out.copy_shared_buffer(result);
      return;
    }
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(result, out, CopyType::General, s);
    return;
  }

  if (mode_ != QuantizationMode::Nvfp4) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Only nvfp4, mxfp4, and mxfp8 modes are implemented on Vulkan.");
  }

  if (fused_nvfp4_qqmm_enabled() && inputs[0].ndim() == 2 &&
      inputs[1].ndim() == 2 && inputs[2].ndim() == 2) {
    array x = ensure_float32_row_contiguous(inputs[0], s);
    array w = ensure_row_contiguous_zero_offset(inputs[1], s);
    array scales = ensure_row_contiguous_zero_offset(inputs[2], s);
    array global_x = global_scale_x.has_value()
        ? ensure_float32_row_contiguous(*global_scale_x, s)
        : x;
    array global_w = global_scale_w.has_value()
        ? ensure_float32_row_contiguous(*global_scale_w, s)
        : x;
    const uint32_t rows = static_cast<uint32_t>(x.shape(0));
    const uint32_t cols = static_cast<uint32_t>(w.shape(0));
    const uint32_t k = static_cast<uint32_t>(x.shape(1));
    if (w.dtype() == uint32 && scales.dtype() == uint8 &&
        w.shape(1) * 8 == x.shape(1) && scales.shape(0) == w.shape(0) &&
        scales.shape(1) * 16 == x.shape(1) && out.shape(0) == x.shape(0) &&
        out.shape(1) == w.shape(0) && is_row_contiguous_zero_offset(x) &&
        is_row_contiguous_zero_offset(w) &&
        is_row_contiguous_zero_offset(scales) &&
        is_row_contiguous_zero_offset(global_x) &&
        is_row_contiguous_zero_offset(global_w)) {
      array out_work(out.shape(), float32, nullptr, {});
      out_work.set_data(allocator::malloc(out_work.nbytes()));
      if (out_work.size() != 0) {
        vulkan::Nvfp4QMatmulPushConstants push_constants{};
        push_constants.rows = rows;
        push_constants.cols = cols;
        push_constants.K = k;
        push_constants.packed_row_words = static_cast<uint32_t>(w.strides(-2));
        push_constants.x_row_stride = static_cast<uint32_t>(x.strides(-2));
        push_constants.out_row_stride =
            static_cast<uint32_t>(out_work.strides(-2));
        push_constants.scale_row_stride =
            static_cast<uint32_t>(scales.strides(-2));
        push_constants.has_global_scale_x =
            global_scale_x.has_value() ? 1u : 0u;
        push_constants.has_global_scale_w =
            global_scale_w.has_value() ? 1u : 0u;
        auto command_buffer = vulkan::begin_command_recording(s.index);
        vulkan::dispatch_nvfp4_qmatmul_op(
            w,
            scales,
            x,
            global_x,
            global_w,
            out_work,
            vulkan::StaticShaderId::mul_mm_nvfp4_f32,
            command_buffer,
            s,
            push_constants,
            {cols, rows, 1u});
        vulkan::end_command_recording(s.index);
      }
      if (out_work.dtype() == out.dtype()) {
        out.copy_shared_buffer(out_work);
        out.set_status(array::Status::evaluated);
        return;
      }
      out.set_data(allocator::malloc(out.nbytes()));
      copy_gpu(out_work, out, CopyType::General, s);
      return;
    }
  }

  array x_f32 = ensure_float32_row_contiguous(inputs[0], s);
  array x_packed(packed_quantized_shape(x_f32, bits_), uint32, nullptr, {});
  array x_scales(quantized_scales_shape(x_f32, group_size_), uint8, nullptr, {});
  if (!vulkan::nvfp4_quantize_from_float32(
          x_f32, x_packed, x_scales, global_scale_x, s)) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Failed to quantize lhs on Vulkan.");
  }
  array xhat(x_f32.shape(), float32, nullptr, {});
  if (!vulkan::nvfp4_dequantize_to_float32(
          x_packed, x_scales, global_scale_x, xhat, s)) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Failed to quantize-dequantize lhs on Vulkan.");
  }

  array what(expanded_quantized_shape(inputs[1], bits_), float32, nullptr, {});
  if (!vulkan::nvfp4_dequantize_to_float32(
          inputs[1], inputs[2], global_scale_w, what, s)) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Failed to dequantize rhs on Vulkan.");
  }

  array rhs =
      ensure_row_contiguous_zero_offset(swapaxes_in_eval(what, -1, -2), s);
  array result(out.shape(), float32, nullptr, {});
  if (!try_eval_matmul_vulkan({xhat, rhs}, result, s)) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Failed to dispatch Vulkan fallback matmul.");
  }

  if (result.dtype() == out.dtype()) {
    out.copy_shared_buffer(result);
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(result, out, CopyType::General, s);
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!is_supported_quantized_bits(bits_)) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Unsupported quantization bits on Vulkan.");
  }

  if (!is_supported_quantized_output_dtype(inputs[0].dtype()) ||
      !is_supported_quantized_output_dtype(out.dtype())) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Only float16, bfloat16, and float32 are supported.");
  }

  auto& s = stream();
  array x = ensure_row_contiguous_zero_offset(inputs[0], s);
  const bool affine_mode = mode_ == QuantizationMode::Affine;
  const bool native_bf16 = affine_mode && x.dtype() == bfloat16 &&
      out.dtype() == bfloat16 && inputs[2].dtype() == bfloat16 &&
      inputs[3].dtype() == bfloat16;
  if (x.dtype() == bfloat16 && !native_bf16) {
    x = ensure_float32_row_contiguous(x, s);
  }
  array w = ensure_row_contiguous_zero_offset(inputs[1], s);
  array scales = affine_mode
      ? (native_bf16 ? ensure_row_contiguous_zero_offset(inputs[2], s)
                     : ensure_float32_row_contiguous(inputs[2], s))
      : ensure_row_contiguous_zero_offset(inputs[2], s);
  std::optional<array> biases = affine_mode
      ? std::make_optional(
            native_bf16 ? ensure_row_contiguous_zero_offset(inputs[3], s)
                        : ensure_float32_row_contiguous(inputs[3], s))
      : std::nullopt;
  array lhs_indices =
      ensure_row_contiguous_zero_offset(inputs[inputs.size() - 2], s);
  array rhs_indices =
      ensure_row_contiguous_zero_offset(inputs[inputs.size() - 1], s);

  if (x.ndim() < 3 || w.ndim() < 3 || scales.ndim() < 3 ||
      (affine_mode && biases->ndim() < 3) ||
      lhs_indices.shape() != rhs_indices.shape() ||
      out.ndim() != lhs_indices.ndim() + 2) {
    std::ostringstream msg;
    msg << "[GatherQMM::eval_gpu] Expected rank-compatible x/w/scales/biases, "
        << "matching indices, and output rank indices+2 but got x=" << x.shape()
        << " w=" << w.shape() << " scales=" << scales.shape() << " biases=";
    if (biases.has_value()) {
      msg << biases->shape();
    } else {
      msg << "none";
    }
    msg << " lhs_indices=" << lhs_indices.shape()
        << " rhs_indices=" << rhs_indices.shape() << " out=" << out.shape()
        << ".";
    throw std::runtime_error(msg.str());
  }
  if (lhs_indices.dtype() != uint32 || rhs_indices.dtype() != uint32) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Expected uint32 gather indices.");
  }

  if ((mode_ == QuantizationMode::Mxfp4 || mode_ == QuantizationMode::Mxfp8) &&
      transpose_) {
    array x_work = ensure_row_contiguous_zero_offset(x, s);
    const uint32_t rows = static_cast<uint32_t>(out.shape(-2));
    const uint32_t cols = static_cast<uint32_t>(out.shape(-1));
    const uint32_t k = static_cast<uint32_t>(x_work.shape(-1));
    const uint32_t num_groups = static_cast<uint32_t>(scales.shape(-1));
    if (rows != static_cast<uint32_t>(x_work.shape(-2)) ||
        cols != static_cast<uint32_t>(w.shape(-2)) ||
        static_cast<uint32_t>(w.shape(-1) * 32 / bits_) != k ||
        num_groups !=
            static_cast<uint32_t>((k + group_size_ - 1) / group_size_) ||
        group_size_ != 32) {
      throw std::runtime_error(
          "[GatherQMM::eval_gpu] Incompatible gather FP qmm shapes.");
    }

    // Use matvec kernel with cooperative K-reduction for better utilization
    if (fp_gather_qmm_fused_matvec(
            w, scales, x_work, lhs_indices, rhs_indices, out, s, bits_)) {
      return;
    }

    if (!fp_gather_qmm_fused(
            w, scales, x_work, lhs_indices, rhs_indices, out, s, bits_)) {
      throw std::runtime_error(
          "[GatherQMM::eval_gpu] Failed to dispatch Vulkan gather FP matmul.");
    }
    return;
  }

  if (!affine_mode || !transpose_) {
    array w_deq(expanded_quantized_shape(w, bits_), float32, nullptr, {});
    if (affine_mode &&
        !vulkan::affine_dequantize_to_float32(
            w, scales, *biases, w_deq, s, group_size_, bits_)) {
      throw std::runtime_error(
          "[GatherQMM::eval_gpu] Failed to dequantize weights on Vulkan.");
    }
    if (mode_ == QuantizationMode::Nvfp4 &&
        !vulkan::nvfp4_dequantize_to_float32(
            w, scales, std::nullopt, w_deq, s)) {
      throw std::runtime_error(
          "[GatherQMM::eval_gpu] Failed to dequantize FP weights on Vulkan.");
    }
    if ((mode_ == QuantizationMode::Mxfp4 ||
         mode_ == QuantizationMode::Mxfp8) &&
        !vulkan::fp_dequantize_to_float32(
            w, scales, w_deq, s, group_size_, bits_)) {
      throw std::runtime_error(
          "[GatherQMM::eval_gpu] Failed to dequantize FP weights on Vulkan.");
    }

    array rhs_f32 = transpose_ ? swapaxes_in_eval(w_deq, -1, -2) : w_deq;
    rhs_f32 = ensure_row_contiguous_zero_offset(rhs_f32, s);
    array result(out.shape(), float32, nullptr, {});
    if (!try_eval_gather_mm_vulkan(
            {ensure_float32_row_contiguous(x, s),
             rhs_f32,
             lhs_indices,
             rhs_indices},
            result,
            s)) {
      throw std::runtime_error(
          "[GatherQMM::eval_gpu] Failed to dispatch Vulkan gather matmul.");
    }
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(result, out, CopyType::General, s);
    return;
  }

  const uint32_t batches = static_cast<uint32_t>(lhs_indices.size());
  const uint32_t rows = static_cast<uint32_t>(out.shape(-2));
  const uint32_t cols = static_cast<uint32_t>(out.shape(-1));
  const uint32_t k = static_cast<uint32_t>(x.shape(-1));
  const uint32_t num_groups = static_cast<uint32_t>(scales.shape(-1));

  const bool gather_decode_rows = rows <= 8;
  const bool use_smallk_matvec8 = bits_ == 8 && gather_decode_rows &&
      k <= 512 && gather_affine_matvec8_enabled() &&
      gather_affine_matvec8_smallk_enabled();
  auto matvec8_shader = use_smallk_matvec8
      ? gather_affine_matvec8_smallk_shader_id(x.dtype(), out.dtype())
      : (bits_ == 8 && gather_decode_rows && gather_affine_matvec8_enabled())
      ? gather_affine_matvec8_shader_id(x.dtype(), out.dtype())
      : std::optional<vulkan::StaticShaderId>{};
  const auto shader_id = matvec8_shader.has_value()
      ? matvec8_shader
      : gather_affine_qmm_shader_id(x.dtype(), out.dtype());
  if (!shader_id.has_value()) {
    std::string mode = matvec8_shader.has_value() ? "matvec8" : "qmm";
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Unsupported activation dtype for Vulkan gather " +
        mode + ".");
  }
  if (rows != static_cast<uint32_t>(x.shape(-2)) ||
      cols != static_cast<uint32_t>(w.shape(-2)) ||
      static_cast<uint32_t>(w.shape(-1) * 32 / bits_) != k ||
      num_groups !=
          static_cast<uint32_t>((k + group_size_ - 1) / group_size_) ||
      x.size() / (x.shape(-2) * x.shape(-1)) <= 0) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Incompatible gather qmm shapes.");
  }

  const auto x_batch_count = x.size() / (x.shape(-2) * x.shape(-1));
  if (x_batch_count <= 0) {
    throw std::runtime_error("[GatherQMM::eval_gpu] Invalid x batch count.");
  }

  const auto expert_count = w.size() / (w.shape(-2) * w.shape(-1));
  const bool use_sorted_rhs_qmm = rows == 1 && batches >= 16 && right_sorted_ &&
      x_batch_count == batches && expert_count > 0 &&
      batches / expert_count >= 4;
  const auto sorted_rhs_shader = use_sorted_rhs_qmm
      ? gather_affine_qmm_rhs_shader_id(x.dtype(), out.dtype())
      : std::optional<vulkan::StaticShaderId>{};

  const auto& context = vulkan::VulkanContext::get();
  const auto device_limits = context.physical_device().getProperties().limits;
  const bool supports_64_lane_subgroups = context.subgroup_size() == 64u ||
      (context.subgroup_size_control_supported() &&
       context.subgroup_min_size() <= 64u &&
       context.subgroup_max_size() >= 64u);
  const bool supports_expert_coop_workgroup =
      device_limits.maxComputeWorkGroupInvocations >= 512u &&
      device_limits.maxComputeWorkGroupSize[0] >= 512u &&
      device_limits.maxComputeSharedMemorySize >= 27648u;
#if defined(MLX_VULKAN_COOPMAT_GLSLC_SUPPORT)
  const bool use_expert_coop_qmm = use_sorted_rhs_qmm && native_bf16 &&
      bits_ == 8 && group_size_ == 64 && expert_count <= 256 &&
      (k % 16u) == 0u && context.coopmat_f16acc_supported() &&
      supports_64_lane_subgroups && supports_expert_coop_workgroup &&
      gather_affine_coop_prefill_enabled();
#else
  const bool use_expert_coop_qmm = false;
#endif

  array out_work(out.shape(), native_bf16 ? bfloat16 : float32, nullptr, {});
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() != 0) {
    if (use_expert_coop_qmm) {
#if defined(MLX_VULKAN_COOPMAT_GLSLC_SUPPORT)
      const uint32_t max_tiles =
          (batches + 63u) / 64u + static_cast<uint32_t>(expert_count);
      const uint32_t metadata_elements = 1u + 3u * max_tiles + batches;
      array metadata(
          {static_cast<int>(metadata_elements)}, uint32, nullptr, {});
      metadata.set_data(allocator::malloc(metadata.nbytes()));

      vulkan::GatherAffineTileMetadataPushConstants metadata_push_constants{};
      metadata_push_constants.rows = batches;
      metadata_push_constants.expert_count =
          static_cast<uint32_t>(expert_count);
      metadata_push_constants.max_tiles = max_tiles;
      metadata_push_constants.K = k;
      metadata_push_constants.x_row_stride =
          static_cast<uint32_t>(x.strides(-2));
      metadata_push_constants.scan_ranges = 0u;

      vulkan::GatherAffineCoopMatmulPushConstants push_constants{};
      push_constants.rows = batches;
      push_constants.cols = cols;
      push_constants.K = k;
      push_constants.packed_row_bytes =
          static_cast<uint32_t>(w.strides(-2) * sizeof(uint32_t));
      push_constants.x_row_stride = static_cast<uint32_t>(x.strides(-2));
      push_constants.out_row_stride =
          static_cast<uint32_t>(out_work.strides(-2));
      push_constants.scale_matrix_stride =
          static_cast<uint32_t>(scales.strides(-3));
      push_constants.scale_row_stride =
          static_cast<uint32_t>(scales.strides(-2));
      push_constants.bias_matrix_stride =
          static_cast<uint32_t>(biases->strides(-3));
      push_constants.bias_row_stride =
          static_cast<uint32_t>(biases->strides(-2));
      push_constants.w_matrix_stride_bytes =
          static_cast<uint32_t>(w.strides(-3) * sizeof(uint32_t));
      push_constants.group_size = static_cast<uint32_t>(group_size_);
      push_constants.max_tiles = max_tiles;

      const std::array<uint32_t, 3> grid = {(cols + 127u) / 128u, max_tiles, 1u};
      if (!dispatch_grid_within_limits(grid[0], grid[1], grid[2])) {
        throw std::runtime_error(
            "[GatherQMM::eval_gpu] Cooperative gather dispatch grid exceeds Vulkan workgroup count limits.");
      }

      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_gather_affine_coop_matmul_op(
          w,
          scales,
          *biases,
          x,
          rhs_indices,
          metadata,
          out_work,
          vulkan::StaticShaderId::gather_affine_qmm_rhs_bf16_bf16_cm1,
          command_buffer,
          s,
          metadata_push_constants,
          push_constants,
          grid);
      vulkan::end_command_recording(s.index);
#endif
    } else {
      vulkan::GatherAffineMatmulPushConstants push_constants{};
      push_constants.rows = sorted_rhs_shader.has_value() ? batches : rows;
      push_constants.cols = cols;
      push_constants.K = k;
      push_constants.packed_row_bytes =
          static_cast<uint32_t>(w.strides(-2) * sizeof(uint32_t));
      push_constants.x_batch_stride =
          static_cast<uint32_t>(x.shape(-2) * x.shape(-1));
      push_constants.x_row_stride = static_cast<uint32_t>(x.strides(-2));
      push_constants.out_batch_stride = rows * cols;
      push_constants.out_row_stride =
          static_cast<uint32_t>(out_work.strides(-2));
      push_constants.scale_matrix_stride =
          static_cast<uint32_t>(scales.strides(-3));
      push_constants.scale_row_stride =
          static_cast<uint32_t>(scales.strides(-2));
      push_constants.bias_matrix_stride =
          static_cast<uint32_t>(biases->strides(-3));
      push_constants.bias_row_stride =
          static_cast<uint32_t>(biases->strides(-2));
      push_constants.w_matrix_stride_bytes =
          static_cast<uint32_t>(w.strides(-3) * sizeof(uint32_t));
      push_constants.bits = static_cast<uint32_t>(bits_);
      push_constants.group_size = static_cast<uint32_t>(group_size_);
      push_constants.num_groups = num_groups;

      const std::array<uint32_t, 3> grid = sorted_rhs_shader.has_value()
          ? std::array<
                uint32_t,
                3>{(cols + 15u) / 16u, (batches + 31u) / 32u, 1u}
          : matvec8_shader.has_value()
          ? std::array<uint32_t, 3>{(cols + 7u) / 8u, rows, batches}
          : std::array<uint32_t, 3>{
                (cols + 15u) / 16u, (rows + 15u) / 16u, batches};
      if (!dispatch_grid_within_limits(grid[0], grid[1], grid[2])) {
        throw std::runtime_error(
            "[GatherQMM::eval_gpu] Gather dispatch grid exceeds Vulkan workgroup count limits.");
      }

      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_gather_affine_matmul_op(
          w,
          scales,
          *biases,
          x,
          lhs_indices,
          rhs_indices,
          out_work,
          sorted_rhs_shader.value_or(*shader_id),
          command_buffer,
          s,
          push_constants,
          grid);
      vulkan::end_command_recording(s.index);
    }
  }

  if (out.dtype() == out_work.dtype()) {
    out.copy_shared_buffer(out_work);
    out.set_status(array::Status::evaluated);
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(out_work, out, CopyType::General, s);
}

} // namespace mlx::core
