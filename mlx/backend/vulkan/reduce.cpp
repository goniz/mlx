// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"

#include <cstring>
#include <optional>
#include <vector>

namespace mlx::core {

namespace {

array stage_zero_offset_row_contiguous(const array& in, Stream s) {
  array staged(in.shape(), in.dtype(), nullptr, {});
  staged.set_data(allocator::malloc(staged.nbytes()));
  copy_gpu_inplace(in, staged, CopyType::Vector, s);
  return staged;
}

std::optional<vulkan::StaticShaderId> integer_reduce_rows_shader(
    Reduce::ReduceType reduce_type,
    Dtype in_dtype,
    Dtype out_dtype) {
  if (reduce_type == Reduce::Sum) {
    if (in_dtype == uint8 && out_dtype == uint32) {
      return vulkan::StaticShaderId::sum_rows_u8_u32;
    }
    if (in_dtype == uint16 && out_dtype == uint32) {
      return vulkan::StaticShaderId::sum_rows_u16_u32;
    }
    if (in_dtype == uint32 && out_dtype == uint32) {
      return vulkan::StaticShaderId::sum_rows_u32;
    }
    if (in_dtype == int32 && out_dtype == int32) {
      return vulkan::StaticShaderId::sum_rows_i32;
    }
  }
  if (reduce_type == Reduce::Prod) {
    if (in_dtype == uint8 && out_dtype == uint32) {
      return vulkan::StaticShaderId::prod_rows_u8_u32;
    }
    if (in_dtype == uint16 && out_dtype == uint32) {
      return vulkan::StaticShaderId::prod_rows_u16_u32;
    }
    if (in_dtype == uint32 && out_dtype == uint32) {
      return vulkan::StaticShaderId::prod_rows_u32;
    }
    if (in_dtype == int32 && out_dtype == int32) {
      return vulkan::StaticShaderId::prod_rows_i32;
    }
  }
  if (reduce_type == Reduce::Max) {
    if (in_dtype == uint32 && out_dtype == uint32) {
      return vulkan::StaticShaderId::max_rows_u32;
    }
    if (in_dtype == int32 && out_dtype == int32) {
      return vulkan::StaticShaderId::max_rows_i32;
    }
  }
  if (reduce_type == Reduce::Min) {
    if (in_dtype == uint32 && out_dtype == uint32) {
      return vulkan::StaticShaderId::min_rows_u32;
    }
    if (in_dtype == int32 && out_dtype == int32) {
      return vulkan::StaticShaderId::min_rows_i32;
    }
  }
  return std::nullopt;
}

bool try_eval_reduce_sum_rows_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    Stream s) {
  if (inputs.size() != 1 || axes.empty()) {
    return false;
  }

  array in = inputs[0];
  const bool sum_reduce = reduce_type == Reduce::Sum;
  const bool prod_reduce = reduce_type == Reduce::Prod;
  const bool bool_prod = reduce_type == Reduce::Prod && in.dtype() == bool_ &&
      out.dtype() == int32;
  const bool max_reduce = reduce_type == Reduce::Max;
  const bool min_reduce = reduce_type == Reduce::Min;
  const bool logic_reduce =
      reduce_type == Reduce::And || reduce_type == Reduce::Or;
  if (!sum_reduce && !prod_reduce && !max_reduce && !min_reduce &&
      !logic_reduce) {
    return false;
  }

  const bool f32_io = in.dtype() == float32 && out.dtype() == float32;
  const bool f16_io = in.dtype() == float16 && out.dtype() == float16;
  const bool bf16_io = in.dtype() == bfloat16 && out.dtype() == bfloat16;
  const bool bool_sum =
      sum_reduce && in.dtype() == bool_ && out.dtype() == int32;
  const bool bool_io = in.dtype() == bool_ && out.dtype() == bool_;
  const bool bool_min_max = (max_reduce || min_reduce) && bool_io;
  const bool integer_sum_prod = (sum_reduce || prod_reduce) &&
      ((in.dtype() == uint8 && out.dtype() == uint32) ||
       (in.dtype() == uint16 && out.dtype() == uint32) ||
       (in.dtype() == uint32 && out.dtype() == uint32) ||
       (in.dtype() == int32 && out.dtype() == int32));
  const bool integer_min_max = (max_reduce || min_reduce) &&
      ((in.dtype() == uint32 && out.dtype() == uint32) ||
       (in.dtype() == int32 && out.dtype() == int32));
  if ((sum_reduce || prod_reduce || max_reduce || min_reduce) && !f32_io &&
      !f16_io && !bf16_io && !bool_sum && !bool_prod && !bool_min_max &&
      !integer_sum_prod && !integer_min_max) {
    return false;
  }
  if (logic_reduce && !bool_prod && out.dtype() != bool_) {
    return false;
  }

  const bool use_f32_staging_io =
      (sum_reduce || prod_reduce || max_reduce || min_reduce) &&
      (f16_io || bf16_io || bool_sum || bool_prod);
  const bool use_u8_staging_io = logic_reduce || bool_min_max;
  if (use_f32_staging_io) {
    if (in.offset() > 0xFFFF && in.flags().row_contiguous) {
      in = stage_zero_offset_row_contiguous(in, s);
    }
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }
  if (use_u8_staging_io) {
    if (in.dtype() != bool_) {
      array in_bool(in.shape(), bool_, nullptr, {});
      copy_gpu(in, in_bool, CopyType::General, s);
      in = in_bool;
    }
    array in_u8(in.shape(), uint8, nullptr, {});
    in_u8.copy_shared_buffer(
        in, in.strides(), in.flags(), in.data_size(), in.offset());
    in = in_u8;
  }

  array reduce_out_target = use_f32_staging_io
      ? array(out.shape(), float32, nullptr, {})
      : use_u8_staging_io
      ? array(out.shape(), bool_prod ? int32 : bool_, nullptr, {})
      : out;
  if (use_u8_staging_io) {
    reduce_out_target.set_data(allocator::malloc(reduce_out_target.nbytes()));
  }

  if ((logic_reduce || bool_min_max) && in.ndim() == 0) {
    copy_gpu(in, out, CopyType::General, s);
    return true;
  }

  if (in.size() == 0 && (sum_reduce || prod_reduce)) {
    array fill_value(sum_reduce ? 0.0f : 1.0f, reduce_out_target.dtype());
    fill_gpu(fill_value, reduce_out_target, s);
    if (use_f32_staging_io) {
      copy_gpu(reduce_out_target, out, CopyType::General, s);
    }
    return true;
  }

  if (in.ndim() == 0) {
    return false;
  }

  std::vector<int> normalized_axes;
  if (!normalize_unique_axes(axes, in.ndim(), normalized_axes)) {
    return false;
  }
  const bool full_reduce = normalized_axes.size() == in.ndim();

  if (in.ndim() > 4 && full_reduce) {
    Shape flattened = {
        static_cast<ShapeElem>(in.shape(0) * in.shape(1)),
        in.ndim() > 2 ? in.shape(2) : 1,
        in.ndim() > 3 ? in.shape(3) : 1,
        in.ndim() > 4 ? in.shape(4) : 1,
    };
    for (int i = 5; i < in.ndim(); ++i) {
      flattened[0] *= in.shape(i);
    }
    in = reshape_in_eval(in, flattened, s);
    normalized_axes = {0, 1, 2, 3};
  }

  const bool staged_full_reduce_output =
      reduce_out_target.ndim() > 4 && full_reduce;
  if (staged_full_reduce_output) {
    reduce_out_target =
        array(Shape{1, 1, 1, 1}, reduce_out_target.dtype(), nullptr, {});
    if (use_u8_staging_io) {
      reduce_out_target.set_data(allocator::malloc(reduce_out_target.nbytes()));
    }
  }

  if (in.ndim() > 4 || reduce_out_target.ndim() > 4) {
    return false;
  }

  const bool out_is_keepdims = staged_full_reduce_output ||
      has_keepdims_axes_shape(in, reduce_out_target, normalized_axes);
  const bool out_is_squeezed =
      has_squeezed_axes_shape(in, reduce_out_target, normalized_axes);
  if (!out_is_keepdims && !out_is_squeezed) {
    return false;
  }

  if (logic_reduce && in.size() == 0) {
    array fill_value(reduce_type == Reduce::And ? true : false, bool_);
    fill_gpu(fill_value, out, s);
    return true;
  }

  array reduced = in;
  for (int axis : normalized_axes) {
    array in_kernel = reduced;
    if (axis != reduced.ndim() - 1) {
      in_kernel = swapaxes_in_eval(reduced, axis, reduced.ndim() - 1);
    }

    if (in_kernel.ndim() == 0 || in_kernel.strides(-1) != 1 ||
        !is_supported_unary_layout(in_kernel)) {
      in_kernel = contiguous_copy_gpu(in_kernel, s);
    }

    const auto integer_shader =
        integer_reduce_rows_shader(reduce_type, in_kernel.dtype(), out.dtype());
    const auto kernel_out_dtype =
        integer_shader ? out.dtype() : in_kernel.dtype();
    array kernel_out(
        keepdims_shape_for_axis(in_kernel, in_kernel.ndim() - 1),
        kernel_out_dtype,
        nullptr,
        {});

    const bool staged_output = !kernel_out.flags().row_contiguous ||
        !is_supported_unary_layout(kernel_out);
    array out_work = staged_output
        ? array(kernel_out.shape(), kernel_out.dtype(), nullptr, {})
        : kernel_out;

    out_work.set_data(allocator::malloc(out_work.nbytes()));
    if (out_work.size() == 0) {
      if (staged_output) {
        copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
      }
    } else {
      try {
        auto command_buffer = vulkan::begin_command_recording(s.index);
        const auto shader_id = integer_shader ? *integer_shader
            : sum_reduce   ? vulkan::StaticShaderId::sum_rows_f32
            : prod_reduce  ? vulkan::StaticShaderId::prod_rows_f32
            : bool_min_max ? (max_reduce ? vulkan::StaticShaderId::any_rows_u8
                                         : vulkan::StaticShaderId::all_rows_u8)
            : max_reduce   ? vulkan::StaticShaderId::max_rows_f32
            : min_reduce   ? vulkan::StaticShaderId::min_rows_f32
            : reduce_type == Reduce::And ? vulkan::StaticShaderId::all_rows_u8
                                         : vulkan::StaticShaderId::any_rows_u8;
        vulkan::dispatch_sum_rows_op(
            in_kernel, out_work, shader_id, command_buffer, s, 1.0f);
        vulkan::end_command_recording(s.index);
      } catch (const std::runtime_error&) {
        return false;
      }
      if (staged_output) {
        copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
      }
    }

    reduced = (axis == reduced.ndim() - 1)
        ? kernel_out
        : swapaxes_in_eval(kernel_out, axis, reduced.ndim() - 1);
  }

  array final_result = out_is_squeezed
      ? reshape_in_eval(reduced, out.shape(), s)
      : staged_full_reduce_output ? reshape_in_eval(reduced, out.shape(), s)
                                  : reduced;

  if (use_f32_staging_io) {
    copy_gpu(final_result, reduce_out_target, CopyType::General, s);
    copy_gpu(reduce_out_target, out, CopyType::General, s);
  } else if (use_u8_staging_io) {
    out.copy_shared_buffer(
        final_result,
        final_result.strides(),
        final_result.flags(),
        final_result.data_size(),
        final_result.offset());
  } else if (final_result.ndim() == 0 && out.ndim() == 0) {
    copy_gpu(final_result, out, CopyType::Scalar, s);
  } else {
    copy_gpu(final_result, out, CopyType::GeneralGeneral, s);
  }

  return true;
}

bool try_eval_arg_reduce_vulkan(
    const std::vector<array>& inputs,
    array& out,
    ArgReduce::ReduceType reduce_type,
    int axis,
    Stream s) {
  if (inputs.size() != 1 ||
      (reduce_type != ArgReduce::ArgMax && reduce_type != ArgReduce::ArgMin)) {
    return false;
  }

  array in = inputs[0];
  const bool f32_input = in.dtype() == float32;
  const bool f16_input = in.dtype() == float16;
  const bool bf16_input = in.dtype() == bfloat16;
  if (in.ndim() == 0 || (!f32_input && !f16_input && !bf16_input) ||
      out.dtype() != uint32) {
    return false;
  }

  if (f16_input || bf16_input) {
    if (in.offset() > 0xFFFF && in.flags().row_contiguous) {
      in = stage_zero_offset_row_contiguous(in, s);
    }
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  axis = normalize_axis(axis, in.ndim());

  const bool out_is_keepdims = has_keepdims_axis_shape(in, out, axis);
  const bool out_is_squeezed = has_squeezed_axis_shape(in, out, axis);
  if (!out_is_keepdims && !out_is_squeezed) {
    return false;
  }

  array in_kernel = in;
  if (axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, axis, in.ndim() - 1);
  }

  if (in_kernel.size() > std::numeric_limits<uint32_t>::max() ||
      out.size() > std::numeric_limits<uint32_t>::max() ||
      in_kernel.shape(in_kernel.ndim() - 1) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  if (in_kernel.ndim() == 0 || in_kernel.strides(-1) != 1 ||
      in_kernel.offset() != 0 || !is_supported_unary_layout(in_kernel)) {
    in_kernel = contiguous_copy_gpu(in_kernel, s);
  }

  array kernel_out(
      keepdims_shape_for_axis(in_kernel, in_kernel.ndim() - 1),
      out.dtype(),
      nullptr,
      {});

  const bool staged_output = !kernel_out.flags().row_contiguous ||
      kernel_out.offset() != 0 || !is_supported_unary_layout(kernel_out);
  array out_work = staged_output
      ? array(kernel_out.shape(), kernel_out.dtype(), nullptr, {})
      : kernel_out;

  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
    }
    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(kernel_out, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    const auto shader_id = reduce_type == ArgReduce::ArgMin
        ? vulkan::StaticShaderId::argmin_f32
        : vulkan::StaticShaderId::argmax_f32;
    vulkan::dispatch_argmax_op(
        in_kernel, out_work, shader_id, command_buffer, s);
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
    }

    array restored_keepdims = kernel_out;
    if (axis != in.ndim() - 1) {
      restored_keepdims = swapaxes_in_eval(kernel_out, axis, in.ndim() - 1);
    }

    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(restored_keepdims, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    } else {
      copy_gpu(restored_keepdims, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

} // namespace

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  if (!try_eval_arg_reduce_vulkan(inputs, out, reduce_type, axis, stream())) {
    throw std::runtime_error(
        "ArgReduce has no Vulkan implementation for this input.");
  }
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axes] = state();
  if (!try_eval_reduce_sum_rows_vulkan(
          inputs, out, reduce_type, axes, stream())) {
    throw std::runtime_error(
        "Reduce has no Vulkan implementation for this input.");
  }
}

} // namespace mlx::core
