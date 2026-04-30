// Copyright © 2026 Apple Inc.

#include "mlx/backend/vulkan/quantized.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/matmul.h"
#include "mlx/backend/vulkan/primitives_utils.h"
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
  if (mode_ != QuantizationMode::Affine) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Only affine mode is implemented on Vulkan.");
  }
  if (inputs.size() != 4) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Expected x, w, scales, biases.");
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
        << " w_shape=" << inputs[1].shape() << " scales_dtype="
        << inputs[2].dtype() << " out_shape=" << out.shape()
        << " out_dtype=" << out.dtype() << " bits=" << bits_
        << " group_size=" << group_size_ << " transpose=" << transpose_;
    if (!detail.empty()) {
      oss << ' ' << detail;
    }
    trace_fallback(oss.str());
  };
  array x = ensure_row_contiguous_zero_offset(inputs[0], s);
  array w = ensure_row_contiguous_zero_offset(inputs[1], s);
  array scales = ensure_float32_row_contiguous(inputs[2], s);
  array biases = ensure_float32_row_contiguous(inputs[3], s);

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

  const uint32_t qmm_rows = x_mat.ndim() == 2
      ? static_cast<uint32_t>(x_mat.shape(-2))
      : 0u;

  const bool enable_fused_decode_qmm = []() {
    if (const char* env = std::getenv("MLX_VULKAN_FUSED_AFFINE_QMM");
        env != nullptr) {
      return std::string_view(env) != "0";
    }
    return true;
  }();

  if (enable_fused_decode_qmm && transpose_ && bits_ == 8 &&
      x_mat.dtype() == bfloat16 && out.dtype() == bfloat16 &&
      inputs[2].dtype() == bfloat16 && inputs[3].dtype() == bfloat16 &&
      x_mat.ndim() == 2 && w.ndim() == 2) {
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
          push_constants.x_row_stride = static_cast<uint32_t>(x_mat.strides(-2));
          push_constants.out_row_stride =
              static_cast<uint32_t>(out_work.strides(-2));
          push_constants.scale_row_stride =
              static_cast<uint32_t>(scales_bf16.strides(-2));
          push_constants.bias_row_stride =
              static_cast<uint32_t>(biases_bf16.strides(-2));
          push_constants.bits = static_cast<uint32_t>(bits_);
          push_constants.group_size = static_cast<uint32_t>(group_size_);
          push_constants.num_groups = num_groups;

          const bool use_tiled_prefill = rows > 1 &&
              group_size_ >= 32 && (group_size_ % 32) == 0 &&
              fused_affine_bf16_tiled_prefill_enabled();
          const auto shader_id = use_tiled_prefill
              ? vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16_tiled
              : vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16;
          const std::array<uint32_t, 3> grid = shader_id ==
                  vulkan::StaticShaderId::fused_affine_qmm_bf16_bf16_tiled
              ? std::array<uint32_t, 3>{
                    (cols + 15u) / 16u, (rows + 31u) / 32u, 1u}
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
              out_work, make_contiguous_strides(out.shape()), flags, out.size());
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
  if (enable_fused_decode_qmm && transpose_ && fused_shader.has_value() &&
      x_mat.ndim() == 2 && w.ndim() == 2 && scales.ndim() == 2 &&
      biases.ndim() == 2 && is_row_contiguous_zero_offset(x_mat) &&
      is_row_contiguous_zero_offset(w) &&
      is_row_contiguous_zero_offset(scales) &&
      is_row_contiguous_zero_offset(biases)) {
    const uint32_t rows = static_cast<uint32_t>(out_work.shape(-2));
    const uint32_t cols = static_cast<uint32_t>(out_work.shape(-1));
    const uint32_t k = static_cast<uint32_t>(x_mat.shape(-1));
    const uint32_t num_groups = static_cast<uint32_t>(scales.shape(-1));
    const bool decode_like_rows = rows == 1;
    const bool prefill_like_rows = rows > 1 && fused_affine_qmm_prefill_enabled();

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
              static_cast<uint32_t>(biases.strides(-2));
          push_constants.bits = static_cast<uint32_t>(bits_);
          push_constants.group_size = static_cast<uint32_t>(group_size_);
          push_constants.num_groups = num_groups;

          const std::array<uint32_t, 3> grid = prefill_like_rows
              ? std::array<uint32_t, 3>{(cols + 15u) / 16u,
                                        (rows + 31u) / 32u,
                                        1u}
              : std::array<uint32_t, 3>{cols, rows, 1u};

          auto command_buffer = vulkan::begin_command_recording(s.index);
          vulkan::dispatch_fused_affine_matmul_op(
              w,
              scales,
              biases,
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
    if (!vulkan::affine_dequantize_to_float32(
            w, scales, biases, w_deq, s, group_size_, bits_)) {
      throw std::runtime_error(
          "[QuantizedMatmul::eval_gpu] Failed to dequantize weights on Vulkan.");
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

  if (mode_ != QuantizationMode::Nvfp4) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] Only nvfp4 mode is implemented on Vulkan.");
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
        push_constants.out_row_stride = static_cast<uint32_t>(out_work.strides(-2));
        push_constants.scale_row_stride =
            static_cast<uint32_t>(scales.strides(-2));
        push_constants.has_global_scales =
            global_scale_x.has_value() && global_scale_w.has_value() ? 1u : 0u;
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
      out.set_data(allocator::malloc(out.nbytes()));
      copy_gpu(out_work, out, CopyType::General, s);
      return;
    }
  }

  array xhat = nvfp4_quantize_dequantize(inputs[0], s, global_scale_x);
  array w_hat = dequantize(
      inputs[1],
      inputs[2],
      std::nullopt,
      group_size_,
      bits_,
      mode,
      global_scale_w,
      inputs[0].dtype(),
      s);

  array result = matmul(xhat, swapaxes(w_hat, -1, -2, s), s);
  if (out.dtype() != result.dtype()) {
    result = astype(result, out.dtype(), s);
  }
  eval(result);
  out.copy_shared_buffer(result);
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (mode_ != QuantizationMode::Affine) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Only affine mode is implemented on Vulkan.");
  }
  if (!transpose_) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Only transposed weights are implemented on Vulkan.");
  }
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
  const bool native_bf16 = x.dtype() == bfloat16 && out.dtype() == bfloat16 &&
      inputs[2].dtype() == bfloat16 && inputs[3].dtype() == bfloat16;
  if (x.dtype() == bfloat16 && !native_bf16) {
    x = ensure_float32_row_contiguous(x, s);
  }
  array w = ensure_row_contiguous_zero_offset(inputs[1], s);
  array scales = native_bf16 ? ensure_row_contiguous_zero_offset(inputs[2], s)
                             : ensure_float32_row_contiguous(inputs[2], s);
  array biases = native_bf16 ? ensure_row_contiguous_zero_offset(inputs[3], s)
                             : ensure_float32_row_contiguous(inputs[3], s);
  array lhs_indices = ensure_row_contiguous_zero_offset(inputs[inputs.size() - 2], s);
  array rhs_indices = ensure_row_contiguous_zero_offset(inputs[inputs.size() - 1], s);

  if (x.ndim() < 3 || w.ndim() != 3 || scales.ndim() != 3 ||
      biases.ndim() != 3 || lhs_indices.shape() != rhs_indices.shape() ||
      out.ndim() != lhs_indices.ndim() + 2) {
    std::ostringstream msg;
    msg << "[GatherQMM::eval_gpu] Expected rank-compatible x/w/scales/biases, "
        << "matching indices, and output rank indices+2 but got x=" << x.shape()
        << " w=" << w.shape() << " scales=" << scales.shape()
        << " biases=" << biases.shape() << " lhs_indices="
        << lhs_indices.shape() << " rhs_indices=" << rhs_indices.shape()
        << " out=" << out.shape() << ".";
    throw std::runtime_error(msg.str());
  }
  if (lhs_indices.dtype() != uint32 || rhs_indices.dtype() != uint32) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Expected uint32 gather indices.");
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
      num_groups != static_cast<uint32_t>((k + group_size_ - 1) / group_size_) ||
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
    push_constants.scale_matrix_stride = static_cast<uint32_t>(scales.strides(0));
    push_constants.scale_row_stride = static_cast<uint32_t>(scales.strides(-2));
    push_constants.bias_matrix_stride = static_cast<uint32_t>(biases.strides(0));
    push_constants.bias_row_stride = static_cast<uint32_t>(biases.strides(-2));
    push_constants.w_matrix_stride_bytes =
        static_cast<uint32_t>(w.strides(0) * sizeof(uint32_t));
    push_constants.bits = static_cast<uint32_t>(bits_);
    push_constants.group_size = static_cast<uint32_t>(group_size_);
    push_constants.num_groups = num_groups;

    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_gather_affine_matmul_op(
        w,
        scales,
        biases,
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
