// Copyright © 2024 Apple Inc.

#include <utility>

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/ops.h"

namespace mlx::core {

namespace {

bool needs_row_contiguous(const array& arr) {
  return !arr.flags().row_contiguous || arr.offset() != 0;
}

array ensure_row_contiguous(array arr, Stream s) {
  if (needs_row_contiguous(arr)) {
    arr = contiguous_copy_gpu(arr, s);
  }
  return arr;
}

CopyType source_copy_type(const array& src) {
  if (src.data_size() == 1) {
    return CopyType::Scalar;
  }
  if (src.flags().row_contiguous && src.offset() == 0) {
    return CopyType::Vector;
  }
  return CopyType::General;
}

std::pair<array, bool> make_output_work(array& out) {
  const bool staged_output = needs_row_contiguous(out);
  array out_work =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  return {out_work, staged_output};
}

uint32_t
checked_shape_product(const array& arr, int begin, int end, const char* label) {
  uint32_t product = 1;
  for (int i = begin; i < end; ++i) {
    product =
        checked_mul_u32(product, checked_u32_size(arr.shape(i), label), label);
  }
  return product;
}

bool try_eval_scatter_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Scatter::ReduceType reduce_type,
    const std::vector<int>& axes,
    Stream s) {
  if (inputs.size() < 3 || inputs.size() != axes.size() + 2) {
    return false;
  }

  const auto& src = inputs[0];
  if (src.ndim() == 0 || out.shape() != src.shape() ||
      out.dtype() != src.dtype()) {
    trace_vulkan_unsupported(
        "Scatter", "source/output shape or dtype mismatch");
    return false;
  }

  if (axes.size() == 2) {
    if (reduce_type != Scatter::None) {
      return false;
    }

    int axis0 = normalize_axis(axes[0], src.ndim());
    int axis1 = normalize_axis(axes[1], src.ndim());
    array idx0 = inputs[1];
    array idx1 = inputs[2];
    array upd = inputs.back();
    if (idx0.shape() != idx1.shape() || idx0.dtype() != idx1.dtype()) {
      return false;
    }
    if (axis0 < 0 || axis1 < 0 || axis0 >= src.ndim() || axis1 >= src.ndim() ||
        axis0 == axis1) {
      return false;
    }
    if (axis0 > axis1) {
      std::swap(axis0, axis1);
      std::swap(idx0, idx1);
    }

    const int idx_ndim = idx0.ndim();
    const uint32_t outer_size =
        checked_shape_product(src, 0, axis0, "scatter_pair outer_size");
    const uint32_t axis0_size =
        checked_u32_size(src.shape(axis0), "scatter_pair axis0_size");
    const uint32_t slice0_size = checked_u32_size(
        upd.shape(idx_ndim + axis0), "scatter_pair slice0_size");
    const uint32_t middle_size = checked_shape_product(
        src, axis0 + 1, axis1, "scatter_pair middle_size");
    const uint32_t axis1_size =
        checked_u32_size(src.shape(axis1), "scatter_pair axis1_size");
    const uint32_t slice1_size = checked_u32_size(
        upd.shape(idx_ndim + axis1), "scatter_pair slice1_size");
    const uint32_t inner_size = checked_shape_product(
        src, axis1 + 1, src.ndim(), "scatter_pair inner_size");
    const uint32_t index_count =
        checked_u32_size(idx0.size(), "scatter_pair index_count");
    const uint32_t slice_size = checked_mul_u32(
        checked_mul_u32(
            checked_mul_u32(
                checked_mul_u32(
                    outer_size,
                    slice0_size,
                    "scatter_pair expected_update_size"),
                middle_size,
                "scatter_pair expected_update_size"),
            slice1_size,
            "scatter_pair expected_update_size"),
        inner_size,
        "scatter_pair expected_update_size");
    const uint32_t expected_update_size = checked_mul_u32(
        index_count, slice_size, "scatter_pair expected_update_size");
    if (upd.size() != expected_update_size) {
      return false;
    }
    for (int i = 0; i < src.ndim(); ++i) {
      const int64_t expected = (i == axis0) ? slice0_size
          : (i == axis1)                    ? slice1_size
                                            : src.shape(i);
      if (upd.shape(idx_ndim + i) != expected) {
        return false;
      }
    }

    const auto shader_id = scatter_pair_shader_id(out.dtype(), idx0.dtype());
    if (!shader_id.has_value()) {
      return false;
    }

    idx0 = ensure_row_contiguous(idx0, s);
    idx1 = ensure_row_contiguous(idx1, s);
    upd = ensure_row_contiguous(upd, s);

    auto [out_work, staged_output] = make_output_work(out);
    copy_gpu(src, out_work, source_copy_type(src), s);

    if (upd.size() == 0) {
      if (staged_output) {
        copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
      }
      return true;
    }

    try {
      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_scatter_pair_op(
          upd,
          idx0,
          idx1,
          out_work,
          *shader_id,
          command_buffer,
          s,
          outer_size,
          axis0_size,
          slice0_size,
          middle_size,
          axis1_size,
          slice1_size,
          inner_size,
          index_count);
      vulkan::end_command_recording(s.index);

      if (staged_output) {
        copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
      }
      return true;
    } catch (const std::runtime_error& e) {
      if (trace_fallback_enabled()) {
        std::ostringstream oss;
        oss << "scatter_pair_dispatch_failed reason=" << e.what();
        trace_fallback(oss.str());
      }
      return false;
    }
  }

  if ((reduce_type != Scatter::None && reduce_type != Scatter::Sum) ||
      axes.size() != 1) {
    trace_vulkan_unsupported("Scatter", "scatter mode or axes unsupported");
    return false;
  }

  int axis = normalize_axis(axes[0], src.ndim());
  array idx = inputs[1];
  array upd = inputs[2];
  if (axis < 0 || axis >= src.ndim()) {
    trace_vulkan_unsupported("Scatter", "axis is out of range");
    return false;
  }
  const auto shader_id = reduce_type == Scatter::Sum
      ? scatter_sum_shader_id(out.dtype(), idx.dtype())
      : scatter_shader_id(out.dtype(), idx.dtype());
  if (!shader_id.has_value()) {
    trace_vulkan_unsupported(
        "Scatter", "value/index dtype combination is not supported");
    return false;
  }

  idx = ensure_row_contiguous(idx, s);
  upd = ensure_row_contiguous(upd, s);

  const uint32_t size_pre =
      checked_shape_product(src, 0, axis, "scatter_take size_pre");
  const uint32_t size_axis =
      checked_u32_size(src.shape(axis), "scatter_take size_axis");
  const uint32_t size_post = checked_shape_product(
      src, axis + 1, src.ndim(), "scatter_take size_post");
  const uint32_t index_count =
      checked_u32_size(idx.size(), "scatter_take index_count");
  const uint32_t slice_size = checked_mul_u32(
      size_pre, size_post, "scatter_take slice_size");
  const uint32_t expected_update_size = checked_mul_u32(
      index_count, slice_size, "scatter_take expected_update_size");
  if (upd.size() != expected_update_size) {
    trace_vulkan_unsupported("Scatter", "update size does not match indices");
    return false;
  }

  auto [out_work, staged_output] = make_output_work(out);
  copy_gpu(src, out_work, source_copy_type(src), s);

  if (upd.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_scatter_take_op(
        upd,
        idx,
        out_work,
        *shader_id,
        command_buffer,
        s,
        size_pre,
        size_axis,
        size_post,
        index_count);
    vulkan::end_command_recording(s.index);

    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "scatter_take_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_eval_scatter_axis_vulkan(
    const std::vector<array>& inputs,
    array& out,
    ScatterAxis::ReduceType reduce_type,
    int axis,
    Stream s) {
  if (inputs.size() != 3) {
    return false;
  }

  if (reduce_type != ScatterAxis::None) {
    trace_vulkan_unsupported(
        "ScatterAxis", "only non-reducing put_along_axis is implemented");
    return false;
  }

  const auto& src = inputs[0];
  array idx = inputs[1];
  array upd = inputs[2];

  if (src.ndim() == 0 || idx.ndim() != src.ndim() ||
      upd.shape() != idx.shape() || out.shape() != src.shape() ||
      out.dtype() != src.dtype()) {
    return false;
  }

  axis = normalize_axis(axis, src.ndim());
  if (axis < 0 || axis >= src.ndim()) {
    trace_vulkan_unsupported("ScatterAxis", "axis is out of range");
    return false;
  }

  const auto shader_id = scatter_axis_shader_id(out.dtype(), idx.dtype());
  if (!shader_id.has_value()) {
    trace_vulkan_unsupported(
        "ScatterAxis",
        "value/index dtype combination is not supported by Vulkan scatter_axis");
    return false;
  }

  idx = ensure_row_contiguous(idx, s);
  upd = ensure_row_contiguous(upd, s);

  const uint32_t size_pre =
      checked_shape_product(src, 0, axis, "scatter_axis size_pre");
  const uint32_t size_axis =
      checked_u32_size(src.shape(axis), "scatter_axis size_axis");
  const uint32_t size_post = checked_shape_product(
      src, axis + 1, src.ndim(), "scatter_axis size_post");
  const uint32_t idx_axis_size =
      checked_u32_size(idx.shape(axis), "scatter_axis idx_axis_size");

  auto [out_work, staged_output] = make_output_work(out);
  copy_gpu(src, out_work, source_copy_type(src), s);

  if (upd.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_scatter_axis_op(
        upd,
        idx,
        out_work,
        *shader_id,
        command_buffer,
        s,
        size_pre,
        size_axis,
        size_post,
        idx_axis_size);
    vulkan::end_command_recording(s.index);

    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "scatter_axis_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axes] = state();
  if (!try_eval_scatter_vulkan(inputs, out, reduce_type, axes, stream())) {
    throw std::runtime_error(
        "Scatter operation failed on Vulkan (unsupported dtype or layout).");
  }
}

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  if (!try_eval_scatter_axis_vulkan(inputs, out, reduce_type, axis, stream())) {
    throw std::runtime_error(
        "ScatterAxis operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
