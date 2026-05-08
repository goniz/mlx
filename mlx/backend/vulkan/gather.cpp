// Copyright © 2024 Apple Inc.

#include <utility>
#include <limits>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/ops.h"

namespace mlx::core {

namespace {

uint32_t checked_shape_product(
    const array& arr,
    int begin,
    int end,
    const char* label);

bool needs_row_contiguous(const array& arr) {
  return !arr.flags().row_contiguous || arr.offset() != 0;
}

array ensure_row_contiguous(array arr, Stream s) {
  if (needs_row_contiguous(arr)) {
    arr = contiguous_copy_gpu(arr, s);
  }
  return arr;
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

std::optional<int64_t> scalar_index_value(const array& idx) {
  if (idx.ndim() != 0) {
    return std::nullopt;
  }
  switch (idx.dtype()) {
    case int32:
      return idx.item<int32_t>();
    case int64:
      return idx.item<int64_t>();
    case uint32:
      return static_cast<int64_t>(idx.item<uint32_t>());
    case uint64: {
      auto value = idx.item<uint64_t>();
      if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::nullopt;
      }
      return static_cast<int64_t>(value);
    }
    default:
      return std::nullopt;
  }
}

std::optional<int64_t> singleton_index_value(const array& idx) {
  if (idx.size() != 1) {
    return std::nullopt;
  }
  switch (idx.dtype()) {
    case int32:
      return idx.item<int32_t>();
    case int64:
      return idx.item<int64_t>();
    case uint32:
      return static_cast<int64_t>(idx.item<uint32_t>());
    case uint64: {
      auto value = idx.item<uint64_t>();
      if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::nullopt;
      }
      return static_cast<int64_t>(value);
    }
    default:
      return std::nullopt;
  }
}

int64_t normalize_gather_index(int64_t idx, int64_t axis_size) {
  return idx < 0 ? idx + axis_size : idx;
}

bool try_eval_gather_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::vector<int>& axes,
    const Shape& slice_sizes,
    Stream s) {
  if (inputs.empty()) {
    return false;
  }

  const auto& src_input = inputs[0];
  if (src_input.ndim() == 0 || out.dtype() != src_input.dtype()) {
    return false;
  }

  if (inputs.size() == 1 && axes.empty()) {
    if (slice_sizes.size() != src_input.ndim()) {
      return false;
    }
    Shape start(src_input.ndim(), 0);
    Shape stop = slice_sizes;
    Shape strides(src_input.ndim(), 1);
    copy_gpu(
        slice(src_input, std::move(start), std::move(stop), std::move(strides), s),
        out,
        CopyType::GeneralGeneral,
        s);
    return true;
  }

  if (inputs.size() < 2 || inputs.size() != axes.size() + 1) {
    return false;
  }

  if (axes.size() == 2) {
    int axis0 = normalize_axis(axes[0], src_input.ndim());
    int axis1 = normalize_axis(axes[1], src_input.ndim());
    array idx0 = inputs[1];
    array idx1 = inputs[2];
    if (idx0.shape() != idx1.shape() || idx0.dtype() != idx1.dtype()) {
      return false;
    }
    if (axis0 < 0 || axis1 < 0 || axis0 >= src_input.ndim() ||
        axis1 >= src_input.ndim() || axis0 == axis1) {
      return false;
    }
    if (axis0 > axis1) {
      std::swap(axis0, axis1);
      std::swap(idx0, idx1);
    }

    for (int i = 0; i < src_input.ndim(); ++i) {
      const int64_t expected = src_input.shape(i);
      if (slice_sizes[i] != expected && i != axis0 && i != axis1) {
        return false;
      }
      if (slice_sizes[i] < 0 || slice_sizes[i] > expected) {
        return false;
      }
    }

    const auto shader_id =
        gather_pair_shader_id(src_input.dtype(), idx0.dtype());
    if (!shader_id.has_value()) {
      return false;
    }

    array src = ensure_row_contiguous(src_input, s);
    idx0 = ensure_row_contiguous(idx0, s);
    idx1 = ensure_row_contiguous(idx1, s);

    const uint32_t outer_size =
        checked_shape_product(src_input, 0, axis0, "gather_pair outer_size");
    const uint32_t axis0_size =
        checked_u32_size(src_input.shape(axis0), "gather_pair axis0_size");
    const uint32_t slice0_size =
        checked_u32_size(slice_sizes[axis0], "gather_pair slice0_size");
    const uint32_t middle_size = checked_shape_product(
        src_input, axis0 + 1, axis1, "gather_pair middle_size");
    const uint32_t axis1_size =
        checked_u32_size(src_input.shape(axis1), "gather_pair axis1_size");
    const uint32_t slice1_size =
        checked_u32_size(slice_sizes[axis1], "gather_pair slice1_size");
    const uint32_t inner_size = checked_shape_product(
        src_input, axis1 + 1, src_input.ndim(), "gather_pair inner_size");
    const uint32_t index_count =
        checked_u32_size(idx0.size(), "gather_pair index_count");

    auto [out_work, staged_output] = make_output_work(out);
    if (out_work.size() == 0) {
      if (staged_output) {
        copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
      }
      return true;
    }

    try {
      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_gather_pair_op(
          src,
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
        oss << "gather_pair_dispatch_failed reason=" << e.what();
        trace_fallback(oss.str());
      }
      return false;
    }
  }

  if (axes.size() != 1) {
    return false;
  }

  array idx = inputs[1];
  const int axis = normalize_axis(axes[0], src_input.ndim());
  if (axis < 0 || axis >= src_input.ndim()) {
    trace_vulkan_unsupported("Gather", "axis is out of range");
    return false;
  }
  if (auto scalar_index = scalar_index_value(idx); scalar_index.has_value()) {
    Shape start(src_input.ndim(), 0);
    Shape stop = slice_sizes;
    Shape strides(src_input.ndim(), 1);
    auto normalized_index =
        normalize_gather_index(*scalar_index, src_input.shape(axis));
    start[axis] = normalized_index;
    stop[axis] += normalized_index;
    copy_gpu(
        slice(src_input, std::move(start), std::move(stop), std::move(strides), s),
        out,
        CopyType::GeneralGeneral,
        s);
    return true;
  }
  if (auto singleton_index = singleton_index_value(idx);
      singleton_index.has_value()) {
    Shape start(src_input.ndim(), 0);
    Shape stop = slice_sizes;
    Shape strides(src_input.ndim(), 1);
    auto normalized_index =
        normalize_gather_index(*singleton_index, src_input.shape(axis));
    start[axis] = normalized_index;
    stop[axis] += normalized_index;
    copy_gpu(
        reshape(
            slice(
                src_input,
                std::move(start),
                std::move(stop),
                std::move(strides),
                s),
            out.shape(),
            s),
        out,
        CopyType::GeneralGeneral,
        s);
    return true;
  }
  for (int i = 0; i < src_input.ndim(); ++i) {
    const int64_t expected = (i == axis) ? 1 : src_input.shape(i);
    if (slice_sizes[i] != expected) {
      trace_vulkan_unsupported(
          "Gather", "only take-like single-axis gathers are supported");
      return false;
    }
  }

  const auto shader_id = gather_shader_id(src_input.dtype(), idx.dtype());
  if (!shader_id.has_value()) {
    trace_vulkan_unsupported(
        "Gather",
        "value/index dtype combination is not supported by Vulkan gather");
    return false;
  }

  array src = ensure_row_contiguous(src_input, s);
  idx = ensure_row_contiguous(idx, s);

  const uint32_t size_pre =
      checked_shape_product(src_input, 0, axis, "gather_take size_pre");
  const uint32_t size_axis =
      checked_u32_size(src_input.shape(axis), "gather_take size_axis");
  const uint32_t size_post = checked_shape_product(
      src_input, axis + 1, src_input.ndim(), "gather_take size_post");
  const uint32_t index_count =
      checked_u32_size(idx.size(), "gather_take index_count");

  auto [out_work, staged_output] = make_output_work(out);
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_gather_take_op(
        src,
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
      oss << "gather_take_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_eval_gather_axis_vulkan(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }

  array src = inputs[0];
  array idx = inputs[1];
  if (src.ndim() == 0 || idx.ndim() != src.ndim() ||
      out.shape() != idx.shape() || out.dtype() != src.dtype()) {
    return false;
  }
  axis = normalize_axis(axis, src.ndim());
  if (axis < 0 || axis >= src.ndim()) {
    trace_vulkan_unsupported("GatherAxis", "axis is out of range");
    return false;
  }

  const auto shader_id = gather_axis_shader_id(src.dtype(), idx.dtype());
  if (!shader_id.has_value()) {
    trace_vulkan_unsupported(
        "GatherAxis",
        "value/index dtype combination is not supported by Vulkan gather_axis");
    return false;
  }

  src = ensure_row_contiguous(src, s);
  idx = ensure_row_contiguous(idx, s);

  const uint32_t size_pre =
      checked_shape_product(src, 0, axis, "gather_axis size_pre");
  const uint32_t size_axis =
      checked_u32_size(src.shape(axis), "gather_axis size_axis");
  const uint32_t size_post =
      checked_shape_product(src, axis + 1, src.ndim(), "gather_axis size_post");
  const uint32_t idx_axis_size =
      checked_u32_size(idx.shape(axis), "gather_axis idx_axis_size");

  auto [out_work, staged_output] = make_output_work(out);
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_gather_axis_op(
        src,
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
      oss << "gather_axis_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [axes, slice_sizes] = state();
  if (!try_eval_gather_vulkan(inputs, out, axes, slice_sizes, stream())) {
    throw std::runtime_error(
        "Gather operation failed on Vulkan (unsupported dtype or layout).");
  }
}

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_gather_axis_vulkan(inputs, out, state(), stream())) {
    throw std::runtime_error(
        "GatherAxis operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
