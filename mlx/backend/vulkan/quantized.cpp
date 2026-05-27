// Copyright © 2026 Apple Inc.

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

namespace mlx::core {
namespace {

bool is_supported_quantized_bits(int bits) {
  return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
      bits == 8;
}

bool is_supported_quantized_output_dtype(Dtype dtype) {
  return dtype == float16 || dtype == bfloat16 || dtype == float32;
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

array nvfp4_quantize_dequantize(
    const array& in,
    Stream s,
    const std::optional<array>& global_scale) {
  array xhat = reshape(in, {-1, 16}, s);
  array scales =
      divide(max(abs(xhat, s), -1, true, s), array(6.0f, in.dtype()), s);
  array scale_encode = global_scale.has_value()
      ? divide(array(448.0f * 6.0f, float32), *global_scale, s)
      : array(1.0f, float32);
  array scales_enc = to_fp8(multiply(scales, scale_encode, s), s);
  array scale = divide(from_fp8(scales_enc, in.dtype(), s), scale_encode, s);

  array lut = astype(
      array(
          {+0.0f,
           +0.5f,
           +1.0f,
           +1.5f,
           +2.0f,
           +3.0f,
           +4.0f,
           +6.0f,
           -0.0f,
           -0.5f,
           -1.0f,
           -1.5f,
           -2.0f,
           -3.0f,
           -4.0f,
           -6.0f}),
      in.dtype(),
      s);
  array idx = argmin(
      abs(subtract(expand_dims(divide(xhat, scale, s), -1, s), lut, s), s),
      -1,
      false,
      s);
  array dequant = multiply(gather(lut, idx, 0, {1}, s), scale, s);
  return reshape(dequant, in.shape(), s);
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
      std::string(bits == 4 ? "dynamic_gather_mxfp4_qmm"
                            : "dynamic_gather_mxfp8_qmm") +
      "_x" + std::to_string(static_cast<int>(x.dtype().val())) + "_o" +
      std::to_string(static_cast<int>(out.dtype().val()));
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      os.str(),
      6,
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

} // namespace

namespace vulkan {

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

  auto command_buffer = vulkan::begin_command_recording(s.index);
  dispatch_nvfp4_quant_op(
      in_work,
      w,
      scales,
      global_scale_work,
      StaticShaderId::quantize_nvfp4_f32,
      command_buffer,
      s,
      push_constants,
      {num_groups, 1, 1});
  vulkan::end_command_recording(s.index);
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
  array scales = mode_ == QuantizationMode::Affine
      ? ensure_float32_row_contiguous(inputs[2], s)
      : inputs[2];
  std::optional<array> biases = mode_ == QuantizationMode::Affine
      ? std::make_optional(ensure_float32_row_contiguous(inputs[3], s))
      : std::nullopt;

  const bool vector_lhs = x.ndim() == 1;
  const bool flatten_lhs_batches = x.ndim() > 2 && w.ndim() == 2;
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

          const bool use_tiled_prefill = rows > 1 && group_size_ >= 32 &&
              (group_size_ % 32) == 0 &&
              fused_affine_bf16_tiled_prefill_enabled();
          const auto shader_id = use_tiled_prefill
              ? vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16_tiled
              : vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16;
          const std::array<uint32_t, 3> grid = shader_id ==
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

        if (vector_lhs || flatten_lhs_batches) {
          array::Flags flags = out.flags();
          flags.contiguous = true;
          flags.row_contiguous = true;
          if (vector_lhs) {
            flags.col_contiguous = true;
          } else {
            auto max_dim =
                std::max_element(out.shape().begin(), out.shape().end());
            flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
          }
          out.copy_shared_buffer(
              out_work,
              make_contiguous_strides(out.shape()),
              flags,
              out.size());
          out.detach();
          out.set_status(array::Status::evaluated);
          trace_qmm("fused_bf16", "reshaped_output=1");
          return;
        }

        out.copy_shared_buffer(out_work);
        trace_qmm("fused_bf16", "reshaped_output=0");
        return;
      }
    }
  }

  if (x_mat.dtype() == bfloat16) {
    x_mat = ensure_float32_row_contiguous(x_mat, s);
  }
  auto fused_shader = qmm_rows > 1 && fused_affine_qmm_prefill_enabled()
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
    const bool decode_like_rows = rows == 1;
    const bool prefill_like_rows =
        rows > 1 && fused_affine_qmm_prefill_enabled();

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
    array w_deq(expanded_quantized_shape(w, bits_), float32, nullptr, {});
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
      out.detach();
      out.set_status(array::Status::evaluated);
      return;
    }

    array out_view(out.shape(), out_work.dtype(), nullptr, {});
    out_view.copy_shared_buffer(
        out_work, make_contiguous_strides(out.shape()), flags, out.size());
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu(out_view, out, CopyType::GeneralGeneral, s);
    out.detach();
    out.set_status(array::Status::evaluated);
    return;
  }

  if (out_work.dtype() == out.dtype()) {
    out.copy_shared_buffer(out_work);
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
        return;
      }
      out.set_data(allocator::malloc(out.nbytes()));
      copy_gpu(out_work, out, CopyType::General, s);
      return;
    }
  }

  array x_f32 = ensure_float32_row_contiguous(inputs[0], s);
  Shape xq_shape = x_f32.shape();
  xq_shape.back() = xq_shape.back() * bits_ / 32;
  Shape x_scales_shape = x_f32.shape();
  x_scales_shape.back() = x_scales_shape.back() / group_size_;

  array x_q(xq_shape, uint32, nullptr, {});
  array x_scales(x_scales_shape, uint8, nullptr, {});
  if (!vulkan::nvfp4_quantize_from_float32(
          x_f32, x_q, x_scales, global_scale_x, s)) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Failed to quantize lhs on Vulkan.");
  }

  array xhat(x_f32.shape(), float32, nullptr, {});
  if (!vulkan::nvfp4_dequantize_to_float32(
          x_q, x_scales, global_scale_x, xhat, s)) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Failed to dequantize lhs on Vulkan.");
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

  array out_work(out.shape(), native_bf16 ? bfloat16 : float32, nullptr, {});
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() != 0) {
    vulkan::GatherAffineMatmulPushConstants push_constants{};
    push_constants.rows = rows;
    push_constants.cols = cols;
    push_constants.K = k;
    push_constants.packed_row_bytes =
        static_cast<uint32_t>(w.strides(-2) * sizeof(uint32_t));
    push_constants.x_batch_stride =
        static_cast<uint32_t>(x.shape(-2) * x.shape(-1));
    push_constants.x_row_stride = static_cast<uint32_t>(x.strides(-2));
    push_constants.out_batch_stride = rows * cols;
    push_constants.out_row_stride = static_cast<uint32_t>(out_work.strides(-2));
    push_constants.scale_matrix_stride =
        static_cast<uint32_t>(scales.strides(-3));
    push_constants.scale_row_stride = static_cast<uint32_t>(scales.strides(-2));
    push_constants.bias_matrix_stride =
        static_cast<uint32_t>(biases->strides(-3));
    push_constants.bias_row_stride = static_cast<uint32_t>(biases->strides(-2));
    push_constants.w_matrix_stride_bytes =
        static_cast<uint32_t>(w.strides(-3) * sizeof(uint32_t));
    push_constants.bits = static_cast<uint32_t>(bits_);
    push_constants.group_size = static_cast<uint32_t>(group_size_);
    push_constants.num_groups = num_groups;

    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_gather_affine_matmul_op(
        w,
        scales,
        *biases,
        x,
        lhs_indices,
        rhs_indices,
        out_work,
        *shader_id,
        command_buffer,
        s,
        push_constants,
        matvec8_shader.has_value()
            ? std::array<uint32_t, 3>{cols, rows, batches}
            : std::array<uint32_t, 3>{
                  (cols + 15u) / 16u, (rows + 15u) / 16u, batches});
    vulkan::end_command_recording(s.index);
  }

  if (out.dtype() == out_work.dtype()) {
    out.copy_shared_buffer(out_work);
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu(out_work, out, CopyType::General, s);
}

} // namespace mlx::core
