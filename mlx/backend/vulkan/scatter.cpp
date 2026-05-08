// Copyright © 2024 Apple Inc.

#include <cstdlib>
#include <cstring>
#include <utility>

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

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

array ensure_host_readable_row_contiguous(array arr, Stream s) {
  if (arr.has_primitive()) {
    arr.eval();
  }
  if (needs_row_contiguous(arr)) {
    arr = contiguous_copy_gpu(arr, s);
  }
  arr.wait();
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
  auto scalar = idx.ndim() == 0 ? idx : slice(idx, Shape(idx.ndim(), 0), Shape(idx.ndim(), 1));
  switch (idx.dtype()) {
    case int32:
      return scalar.item<int32_t>();
    case int64:
      return scalar.item<int64_t>();
    case uint32:
      return static_cast<int64_t>(scalar.item<uint32_t>());
    case uint64: {
      auto value = scalar.item<uint64_t>();
      if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return std::nullopt;
      }
      return static_cast<int64_t>(value);
    }
    default:
      return std::nullopt;
  }
}

int64_t normalize_scatter_index(int64_t idx, int64_t axis_size) {
  if (idx < 0) {
    idx += axis_size;
  }
  if (idx < 0 || idx >= axis_size) {
    throw std::out_of_range(
        "scatter index " + std::to_string(idx) + " out of bounds " +
        std::to_string(axis_size));
  }
  return idx;
}

int64_t read_flat_index_item(const array& idx, int i, Stream s) {
  auto scalar = slice(
      idx,
      {static_cast<ShapeElem>(i)},
      {static_cast<ShapeElem>(i + 1)},
      s);
  switch (idx.dtype()) {
    case int32:
      return scalar.item<int32_t>();
    case int64:
      return scalar.item<int64_t>();
    case uint32:
      return static_cast<int64_t>(scalar.item<uint32_t>());
    case uint64:
      return static_cast<int64_t>(scalar.item<uint64_t>());
    default:
      throw std::runtime_error("Unsupported index dtype for Vulkan scatter.");
  }
}

int64_t read_contiguous_index(const array& idx, int i) {
  switch (idx.dtype()) {
    case int32:
      return idx.data<int32_t>()[i];
    case int64:
      return idx.data<int64_t>()[i];
    case uint32:
      return idx.data<uint32_t>()[i];
    case uint64: {
      auto val = idx.data<uint64_t>()[i];
      if (val > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("uint64 index exceeds max int64_t value");
      }
      return static_cast<int64_t>(val);
    }
    default:
      throw std::runtime_error("Unsupported index dtype for Vulkan scatter.");
  }
}

template <typename T>
bool try_host_scatter_sum_single_axis_typed(
    const array& src,
    const array& idx,
    const array& upd,
    array& out,
    int axis,
    const Shape& update_shape,
    uint32_t slice_elems,
    Stream s) {
  const uint32_t index_count =
      checked_u32_size(idx.size(), "scatter_host index_count");
  array src_host = ensure_host_readable_row_contiguous(src, s);
  array idx_host = ensure_host_readable_row_contiguous(
      reshape(idx, {static_cast<ShapeElem>(index_count)}, s),
      s);
  array upd_host = ensure_host_readable_row_contiguous(
      add(upd, array(0.0f, upd.dtype()), s),
      s);

  auto src_strides = make_contiguous_strides(src.shape());
  auto update_strides = make_contiguous_strides(update_shape);
  std::vector<T> result(src.size());
  std::copy(
      src_host.data<T>(),
      src_host.data<T>() + src.size(),
      result.begin());

  for (uint32_t i = 0; i < index_count; ++i) {
    auto normalized_index =
        normalize_scatter_index(read_contiguous_index(idx_host, i), src.shape(axis));
    const size_t update_base = static_cast<size_t>(i) * slice_elems;
    for (uint32_t linear = 0; linear < slice_elems; ++linear) {
      size_t remainder = linear;
      size_t dst_offset = 0;
      for (int d = 0; d < src.ndim(); ++d) {
        size_t coord = 0;
        if (!update_shape.empty()) {
          coord = remainder / update_strides[d];
          remainder %= update_strides[d];
        }
        if (d == axis) {
          coord += normalized_index;
        }
        dst_offset += coord * src_strides[d];
      }
      result[dst_offset] += upd_host.data<T>()[update_base + linear];
    }
  }

  array host_result(result.begin(), src.shape(), src.dtype());
  out.copy_shared_buffer(host_result);
  return true;
}

bool try_host_scatter_sum_single_axis(
    const array& src,
    const array& idx,
    const array& upd,
    array& out,
    int axis,
    const Shape& update_shape,
    uint32_t slice_elems,
    Stream s) {
  if (src.dtype() != out.dtype() || upd.dtype() != src.dtype()) {
    return false;
  }
  switch (src.dtype()) {
    case float32:
      return try_host_scatter_sum_single_axis_typed<float>(
          src, idx, upd, out, axis, update_shape, slice_elems, s);
    case int32:
      return try_host_scatter_sum_single_axis_typed<int32_t>(
          src, idx, upd, out, axis, update_shape, slice_elems, s);
    case int64:
      return try_host_scatter_sum_single_axis_typed<int64_t>(
          src, idx, upd, out, axis, update_shape, slice_elems, s);
    case uint32:
      return try_host_scatter_sum_single_axis_typed<uint32_t>(
          src, idx, upd, out, axis, update_shape, slice_elems, s);
    case uint64:
      return try_host_scatter_sum_single_axis_typed<uint64_t>(
          src, idx, upd, out, axis, update_shape, slice_elems, s);
    default:
      return false;
  }
}

bool try_host_scatter_none_single_axis(
    const array& src,
    int64_t normalized_index,
    const array& upd,
    array& out,
    int axis,
    const Shape& update_shape,
    uint32_t slice_elems,
    Stream s) {
  array src_host = ensure_host_readable_row_contiguous(src, s);
  array upd_host =
      ensure_host_readable_row_contiguous(reshape(upd, update_shape, s), s);

  auto src_strides = make_contiguous_strides(src.shape());
  auto update_strides = make_contiguous_strides(update_shape);
  const size_t item_size = size_of(src.dtype());
  std::vector<uint8_t> result(src.nbytes());
  std::memcpy(result.data(), src_host.data<void>(), src.nbytes());
  const auto* upd_bytes = static_cast<const uint8_t*>(upd_host.data<void>());

  for (uint32_t linear = 0; linear < slice_elems; ++linear) {
    size_t remainder = linear;
    size_t dst_offset = 0;
    for (int d = 0; d < src.ndim(); ++d) {
      size_t coord = 0;
      if (!update_shape.empty()) {
        coord = remainder / update_strides[d];
        remainder %= update_strides[d];
      }
      if (d == axis) {
        coord += normalized_index;
      }
      dst_offset += coord * src_strides[d];
    }
    std::memcpy(
        result.data() + dst_offset * item_size,
        upd_bytes + static_cast<size_t>(linear) * item_size,
        item_size);
  }

  void* host_result = std::malloc(result.size());
  if (host_result == nullptr) {
    throw std::bad_alloc();
  }
  std::memcpy(host_result, result.data(), result.size());
  array host_array(
      host_result,
      src.shape(),
      src.dtype(),
      [](void* ptr) { std::free(ptr); });
  out.set_data(allocator::malloc(out.nbytes()));
  copy_gpu_inplace(host_array, out, CopyType::GeneralGeneral, s);
  return true;
}

bool try_eval_scatter_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Scatter::ReduceType reduce_type,
    const std::vector<int>& axes,
    Stream s) {
  if (axes.empty() && inputs.size() == 2) {
    const auto& src = inputs[0];
    const auto& upd = inputs[1];
    if (src.ndim() != 0 || upd.shape() != src.shape() || upd.dtype() != src.dtype() ||
        out.shape() != src.shape() || out.dtype() != src.dtype()) {
      return false;
    }

    array result = src;
    switch (reduce_type) {
      case Scatter::None:
        result = upd;
        break;
      case Scatter::Sum:
        result = add(src, upd, s);
        break;
      case Scatter::Max:
        result = maximum(src, upd, s);
        break;
      case Scatter::Min:
        result = minimum(src, upd, s);
        break;
      default:
        return false;
    }
    copy_gpu(result, out, CopyType::GeneralGeneral, s);
    return true;
  }

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

  {
    std::vector<int> norm_axes;
    norm_axes.reserve(axes.size());
    for (int ax : axes) {
      int norm = normalize_axis(ax, src.ndim());
      if (norm < 0 || norm >= src.ndim()) {
        return false;
      }
      norm_axes.push_back(norm);
    }
    bool duplicate_axes = false;
    for (int i = 0; i < norm_axes.size(); ++i) {
      for (int j = i + 1; j < norm_axes.size(); ++j) {
        if (norm_axes[i] == norm_axes[j]) {
          duplicate_axes = true;
          break;
        }
      }
      if (duplicate_axes) {
        break;
      }
    }

    if (
        !duplicate_axes && norm_axes.size() > 2 &&
        (reduce_type == Scatter::None || reduce_type == Scatter::Sum ||
         reduce_type == Scatter::Prod || reduce_type == Scatter::Max ||
         reduce_type == Scatter::Min)) {
      const int idx_ndim = inputs[1].ndim();
      bool consistent_indices = true;
      for (int i = 2; i < inputs.size() - 1; ++i) {
        if (inputs[i].shape() != inputs[1].shape() || inputs[i].dtype() != inputs[1].dtype()) {
          consistent_indices = false;
          break;
        }
      }
      array upd = inputs.back();
      if (consistent_indices && upd.ndim() == idx_ndim + src.ndim()) {
        Shape update_shape(upd.shape().begin() + idx_ndim, upd.shape().end());
        uint32_t slice_elems = 1;
        for (auto dim : update_shape) {
          slice_elems = checked_mul_u32(
              slice_elems,
              checked_u32_size(dim, "scatter_generic slice_elems"),
              "scatter_generic slice_elems");
        }
        if (update_shape.size() == src.ndim() &&
            checked_mul_u32(
                checked_u32_size(inputs[1].size(), "scatter_generic index_count"),
                slice_elems,
                "scatter_generic update_size") == upd.size()) {
          std::vector<array> flat_indices;
          flat_indices.reserve(inputs.size() - 2);
          for (int i = 1; i < inputs.size() - 1; ++i) {
            flat_indices.push_back(
                reshape(
                    inputs[i],
                    {static_cast<ShapeElem>(inputs[i].size())},
                    s));
          }

          std::vector<int> reduced_axes;
          std::vector<array> reduced_indices;
          reduced_axes.reserve(norm_axes.size());
          reduced_indices.reserve(norm_axes.size());
          bool dropped_full_slice_axis = false;
          for (int j = 0; j < norm_axes.size(); ++j) {
            const int axis = norm_axes[j];
            bool can_drop_axis = update_shape[axis] == src.shape(axis);
            if (can_drop_axis) {
              auto flat_idx = ensure_host_readable_row_contiguous(flat_indices[j], s);
              for (int i = 0; i < flat_idx.size(); ++i) {
                if (read_contiguous_index(flat_idx, i) != 0) {
                  can_drop_axis = false;
                  break;
                }
              }
            }

            if (can_drop_axis) {
              dropped_full_slice_axis = true;
              continue;
            }

            reduced_axes.push_back(axis);
            reduced_indices.push_back(inputs[j + 1]);
          }

          if (
              dropped_full_slice_axis && reduced_axes.size() == 2 &&
              (reduce_type == Scatter::Prod || reduce_type == Scatter::Max ||
               reduce_type == Scatter::Min)) {
            std::vector<array> reduced_inputs;
            reduced_inputs.reserve(reduced_indices.size() + 2);
            reduced_inputs.push_back(src);
            reduced_inputs.insert(
                reduced_inputs.end(), reduced_indices.begin(), reduced_indices.end());
            reduced_inputs.push_back(upd);
            if (try_eval_scatter_vulkan(
                    reduced_inputs, out, reduce_type, reduced_axes, s)) {
              return true;
            }
          }

          if (reduce_type != Scatter::None && reduce_type != Scatter::Sum) {
            return false;
          }

          upd = ensure_host_readable_row_contiguous(upd, s);

          array result = contiguous_copy_gpu(src, s);
          result.set_status(array::Status::available);
          Shape flat_shape = {
              static_cast<ShapeElem>(checked_u32_size(inputs[1].size(), "scatter_generic index_count")),
              static_cast<ShapeElem>(slice_elems)};
          array flat_upd = reshape(upd, flat_shape, s);

          for (int i = 0; i < flat_indices[0].size(); ++i) {
            Shape start(src.ndim(), 0);
            Shape stop = update_shape;
            Shape unit_strides(src.ndim(), 1);
            for (int j = 0; j < norm_axes.size(); ++j) {
              int axis = norm_axes[j];
              start[axis] = normalize_scatter_index(
                  read_flat_index_item(flat_indices[j], i, s), src.shape(axis));
              stop[axis] += start[axis];
              if (stop[axis] > src.shape(axis)) {
                return false;
              }
            }

          array update_view = reshape(
              slice(
                  flat_upd,
                  {static_cast<ShapeElem>(i), 0},
                  {static_cast<ShapeElem>(i + 1),
                   static_cast<ShapeElem>(slice_elems)},
                  s),
              update_shape,
              s);

            array next(src.shape(), src.dtype(), nullptr, {});
            next.set_data(allocator::malloc(next.nbytes()));
            next.set_status(array::Status::available);
            SliceUpdate::ReduceType slice_reduce = SliceUpdate::None;
            if (reduce_type == Scatter::Sum) {
              slice_reduce = SliceUpdate::Sum;
            }
            SliceUpdate op(s, slice_reduce, start, stop, unit_strides);
            op.eval_gpu({result, update_view}, next);
            result = std::move(next);
          }
          out.copy_shared_buffer(result);
          return true;
        }
      }
    }
  }

  if (axes.size() == 2) {
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
    if (upd.ndim() == idx_ndim + src.ndim()) {
      Shape update_shape(upd.shape().begin() + idx_ndim, upd.shape().end());
      Shape target_slice_shape = update_shape;
      uint32_t slice_elems = 1;
      for (auto dim : update_shape) {
        slice_elems = checked_mul_u32(
            slice_elems,
            checked_u32_size(dim, "scatter_pair slice_elems"),
            "scatter_pair slice_elems");
      }
      if (update_shape.size() == src.ndim() &&
          checked_mul_u32(
              checked_u32_size(idx0.size(), "scatter_pair composed index_count"),
              slice_elems,
              "scatter_pair composed update_size") == upd.size() &&
          !(reduce_type == Scatter::Sum && slice_elems == 1) &&
          (reduce_type == Scatter::None || reduce_type == Scatter::Sum ||
           reduce_type == Scatter::Prod || reduce_type == Scatter::Max ||
           reduce_type == Scatter::Min)) {
        idx0 = ensure_host_readable_row_contiguous(idx0, s);
        idx1 = ensure_host_readable_row_contiguous(idx1, s);
        upd = ensure_row_contiguous(upd, s);
        array flat_idx0 =
            reshape(idx0, {static_cast<ShapeElem>(idx0.size())}, s);
        array flat_idx1 =
            reshape(idx1, {static_cast<ShapeElem>(idx1.size())}, s);
        Shape flat_shape = {
            static_cast<ShapeElem>(index_count),
            static_cast<ShapeElem>(slice_elems)};
        array flat_upd = reshape(upd, flat_shape, s);
        array result = contiguous_copy_gpu(src, s);
        result.set_status(array::Status::available);
        for (int i = 0; i < idx0.size(); ++i) {
          Shape start(src.ndim(), 0);
          Shape stop = target_slice_shape;
          Shape unit_strides(src.ndim(), 1);
          auto idx0_value =
              normalize_scatter_index(
                  read_contiguous_index(flat_idx0, i), src.shape(axis0));
          auto idx1_value =
              normalize_scatter_index(
                  read_contiguous_index(flat_idx1, i), src.shape(axis1));
          start[axis0] = idx0_value;
          start[axis1] = idx1_value;
          stop[axis0] += start[axis0];
          stop[axis1] += start[axis1];
          if (stop[axis0] > src.shape(axis0) || stop[axis1] > src.shape(axis1)) {
            return false;
          }

          array update_value = reshape(
              slice(
                  flat_upd,
                  {static_cast<ShapeElem>(i), 0},
                  {static_cast<ShapeElem>(i + 1),
                   static_cast<ShapeElem>(slice_elems)},
                  s),
              update_shape,
              s);

          switch (reduce_type) {
            case Scatter::None:
              {
              array next(src.shape(), src.dtype(), nullptr, {});
              next.set_data(allocator::malloc(next.nbytes()));
              next.set_status(array::Status::available);
              SliceUpdate op(
                  s, SliceUpdate::None, start, stop, unit_strides);
              op.eval_gpu({result, update_value}, next);
              result = std::move(next);
              break;
              }
            case Scatter::Sum: {
              array next(src.shape(), src.dtype(), nullptr, {});
              next.set_data(allocator::malloc(next.nbytes()));
              next.set_status(array::Status::available);
              SliceUpdate op(
                  s, SliceUpdate::Sum, start, stop, unit_strides);
              op.eval_gpu({result, update_value}, next);
              result = std::move(next);
              break;
            }
            case Scatter::Prod:
              {
              array next(src.shape(), src.dtype(), nullptr, {});
              next.set_data(allocator::malloc(next.nbytes()));
              next.set_status(array::Status::available);
              SliceUpdate op(
                  s, SliceUpdate::Prod, start, stop, unit_strides);
              op.eval_gpu({result, update_value}, next);
              result = std::move(next);
              break;
              }
            case Scatter::Max:
              {
              array next(src.shape(), src.dtype(), nullptr, {});
              next.set_data(allocator::malloc(next.nbytes()));
              next.set_status(array::Status::available);
              SliceUpdate op(
                  s, SliceUpdate::Max, start, stop, unit_strides);
              op.eval_gpu({result, update_value}, next);
              result = std::move(next);
              break;
              }
            case Scatter::Min:
              {
              array next(src.shape(), src.dtype(), nullptr, {});
              next.set_data(allocator::malloc(next.nbytes()));
              next.set_status(array::Status::available);
              SliceUpdate op(
                  s, SliceUpdate::Min, start, stop, unit_strides);
              op.eval_gpu({result, update_value}, next);
              result = std::move(next);
              break;
              }
          }
        }
        out.copy_shared_buffer(result);
        return true;
      }
    }
    if (reduce_type != Scatter::None && reduce_type != Scatter::Sum) {
      return false;
    }
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

    const auto shader_id = reduce_type == Scatter::Sum
        ? scatter_sum_pair_shader_id(out.dtype(), idx0.dtype())
        : scatter_pair_shader_id(out.dtype(), idx0.dtype());
    if (!shader_id.has_value()) {
      Shape update_shape(
          upd.shape().begin() + idx_ndim,
          upd.shape().end());
      if (reduce_type != Scatter::None) {
        return false;
      }

      idx0 = ensure_host_readable_row_contiguous(idx0, s);
      idx1 = ensure_host_readable_row_contiguous(idx1, s);
      upd = ensure_host_readable_row_contiguous(upd, s);
      array flat_idx0 =
          reshape(idx0, {static_cast<ShapeElem>(idx0.size())}, s);
      array flat_idx1 =
          reshape(idx1, {static_cast<ShapeElem>(idx1.size())}, s);

      array result = src;

      Strides update_strides(update_shape.size(), 1);
      for (int d = static_cast<int>(update_shape.size()) - 2; d >= 0; --d) {
        update_strides[d] = update_strides[d + 1] * update_shape[d + 1];
      }
      auto [update_data_size, update_row_contig, update_col_contig] =
          check_contiguity(update_shape, update_strides);

      for (int i = 0; i < idx0.size(); ++i) {
        auto normalized_index0 =
            normalize_scatter_index(
                read_flat_index_item(flat_idx0, i, s), src.shape(axis0));
        auto normalized_index1 =
            normalize_scatter_index(
                read_flat_index_item(flat_idx1, i, s), src.shape(axis1));
        Shape start(src.ndim(), 0);
        Shape stop = update_shape;
        Shape unit_strides(src.ndim(), 1);
        start[axis0] = normalized_index0;
        stop[axis0] += normalized_index0;
        start[axis1] = normalized_index1;
        stop[axis1] += normalized_index1;
        if (stop[axis0] > src.shape(axis0) || stop[axis1] > src.shape(axis1)) {
          return false;
        }

        array update_view(update_shape, upd.dtype(), nullptr, {});
        update_view.copy_shared_buffer(
            upd,
            update_strides,
            {update_data_size == slice_size, update_row_contig, update_col_contig},
            slice_size,
            static_cast<int64_t>(i) * slice_size);
        update_view.set_status(array::Status::available);

        array next(src.shape(), src.dtype(), nullptr, {});
        next.set_data(allocator::malloc(next.nbytes()));
        next.set_status(array::Status::available);
        SliceUpdate op(
            s,
            SliceUpdate::None,
            start,
            stop,
            unit_strides);
        op.eval_gpu({result, update_view}, next);
        result = std::move(next);
      }

      if (result.has_primitive()) {
        result.eval();
      }
      copy_gpu(result, out, CopyType::GeneralGeneral, s);
      return true;
    }

    idx0 = ensure_row_contiguous(idx0, s);
    idx1 = ensure_row_contiguous(idx1, s);
    upd = ensure_row_contiguous(upd, s);

    if (reduce_type == Scatter::Sum && out.dtype() == float32 &&
        !vulkan::VulkanContext::get().shader_buffer_atomic_float32_supported()) {
      trace_vulkan_unsupported(
          "Scatter", "float32 scatter-sum pair requires shader atomic float support");
      return false;
    }
    if (reduce_type == Scatter::Sum &&
        (out.dtype() == float16 || out.dtype() == bfloat16)) {
      if ((out.offset() % 4) != 0 || (out.size() % 2) != 0) {
        trace_vulkan_unsupported(
            "Scatter",
            "float16/bfloat16 scatter-sum pair requires 4-byte-aligned offset and even element count");
        return false;
      }
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

  if (axes.size() != 1) {
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
  const uint32_t size_pre =
      checked_shape_product(src, 0, axis, "scatter_take size_pre");
  const uint32_t size_axis =
      checked_u32_size(src.shape(axis), "scatter_take size_axis");
  const uint32_t size_post = checked_shape_product(
      src, axis + 1, src.ndim(), "scatter_take size_post");
  const uint32_t index_count =
      checked_u32_size(idx.size(), "scatter_take index_count");
  const uint32_t take_slice_size = checked_mul_u32(
      size_pre, size_post, "scatter_take slice_size");
  if (reduce_type == Scatter::Sum && take_slice_size == 1 && out.dtype() == float32) {
    array src_host = ensure_host_readable_row_contiguous(src, s);
    array idx_host =
        ensure_host_readable_row_contiguous(reshape(idx, {static_cast<ShapeElem>(index_count)}, s), s);
    array upd_host =
        ensure_host_readable_row_contiguous(reshape(upd, {static_cast<ShapeElem>(index_count)}, s), s);
    std::vector<float> result(src.size());
    std::copy(
        src_host.data<float>(),
        src_host.data<float>() + src.size(),
        result.begin());
    for (uint32_t i = 0; i < index_count; ++i) {
      auto normalized_index =
          normalize_scatter_index(read_contiguous_index(idx_host, i), src.shape(axis));
      result[normalized_index] += upd_host.data<float>()[i];
    }
    array host_result(result.begin(), src.shape(), float32);
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(host_result, out, CopyType::GeneralGeneral, s);
    return true;
  }
  if (upd.ndim() == idx.ndim() + src.ndim()) {
    Shape update_shape(
        upd.shape().begin() + idx.ndim(),
        upd.shape().end());
    Shape target_slice_shape = update_shape;
    uint32_t slice_elems = 1;
    for (auto dim : update_shape) {
      slice_elems =
          checked_mul_u32(slice_elems, checked_u32_size(dim, "scatter slice_elems"),
                          "scatter slice_elems");
    }
    if (reduce_type == Scatter::None && idx.size() == 1 &&
        update_shape.size() == src.ndim()) {
      array flat_idx = reshape(idx, {1}, s);
      auto singleton_index =
          normalize_scatter_index(read_flat_index_item(flat_idx, 0, s), src.shape(axis));
      Shape start(src.ndim(), 0);
      Shape stop = target_slice_shape;
      Shape unit_strides(src.ndim(), 1);
      start[axis] = singleton_index;
      stop[axis] += singleton_index;
      if (stop[axis] > src.shape(axis)) {
        return false;
      }
      return try_host_scatter_none_single_axis(
          src, singleton_index, upd, out, axis, update_shape, slice_elems, s);
    }
    if (update_shape.size() == src.ndim() &&
        checked_mul_u32(
            checked_u32_size(idx.size(), "scatter composed index_count"),
            slice_elems,
            "scatter composed update_size") == upd.size() &&
        !(reduce_type == Scatter::None && idx.size() == 1) &&
        (reduce_type == Scatter::None || reduce_type == Scatter::Sum)) {
      if (reduce_type == Scatter::Sum &&
          try_host_scatter_sum_single_axis(
              src, idx, upd, out, axis, update_shape, slice_elems, s)) {
        return true;
      }
      Shape flat_shape = {
          static_cast<ShapeElem>(index_count),
          static_cast<ShapeElem>(slice_elems)};
      array flat_idx =
          reshape(idx, {static_cast<ShapeElem>(index_count)}, s);
      array flat_upd = reshape(upd, flat_shape, s);
      array result(src.shape(), src.dtype(), nullptr, {});
      result.set_data(allocator::malloc(result.nbytes()));
      copy_gpu(src, result, source_copy_type(src), s);
      for (int i = 0; i < idx.size(); ++i) {
        auto normalized_index =
            normalize_scatter_index(
                read_flat_index_item(flat_idx, i, s), src.shape(axis));
        Shape start(src.ndim(), 0);
        Shape stop = target_slice_shape;
        Shape unit_strides(src.ndim(), 1);
        start[axis] = normalized_index;
        stop[axis] += normalized_index;
        if (stop[axis] > src.shape(axis)) {
          return false;
        }
        array update_value = reshape(
            slice(
                flat_upd,
                {static_cast<ShapeElem>(i), 0},
                {static_cast<ShapeElem>(i + 1),
                 static_cast<ShapeElem>(slice_elems)},
                s),
            update_shape,
            s);

        if (reduce_type == Scatter::None) {
          array next(src.shape(), src.dtype(), nullptr, {});
          next.set_data(allocator::malloc(next.nbytes()));
          next.set_status(array::Status::available);
          SliceUpdate op(
              s,
              SliceUpdate::None,
              start,
              stop,
              unit_strides);
          op.eval_gpu({result, update_value}, next);
          result = std::move(next);
          continue;
        }

        if (reduce_type == Scatter::Sum) {
          auto current = slice(result, start, stop, unit_strides, s);
          auto merged = add(current, update_value, s);
          result = slice_update(
              result,
              merged,
              std::move(start),
              std::move(stop),
              std::move(unit_strides),
              s);
          continue;
        }

        auto current = slice(result, start, stop, unit_strides, s);
        array merged = current;
        switch (reduce_type) {
          case Scatter::Prod:
            merged = multiply(current, update_value, s);
            break;
          case Scatter::Max:
            merged = maximum(current, update_value, s);
            break;
          case Scatter::Min:
            merged = minimum(current, update_value, s);
            break;
          default:
            break;
        }
        result = slice_update(result, merged, std::move(start), std::move(stop), std::move(unit_strides), s);
      }
      if (result.has_primitive()) {
        result.eval();
      }
      copy_gpu(result, out, CopyType::GeneralGeneral, s);
      return true;
    }
  }
  if (reduce_type != Scatter::None && reduce_type != Scatter::Sum) {
    const uint32_t slice_size = take_slice_size;
    const uint32_t expected_update_size = checked_mul_u32(
        index_count, slice_size, "scatter_take expected_update_size");
    if (upd.size() != expected_update_size) {
      trace_vulkan_unsupported("Scatter", "update size does not match indices");
      return false;
    }
    Shape flat_shape = {
        static_cast<ShapeElem>(index_count),
        static_cast<ShapeElem>(slice_size)};
    array flat_idx =
        reshape(idx, {static_cast<ShapeElem>(index_count)}, s);
    array flat_upd = reshape(upd, flat_shape, s);
    Shape slice_shape = src.shape();
    slice_shape[axis] = 1;
    array result = src;
    for (uint32_t i = 0; i < index_count; ++i) {
      Shape unit_strides(src.ndim(), 1);
      auto idx_value =
          normalize_scatter_index(
              read_flat_index_item(flat_idx, i, s),
              src.shape(axis));
      Shape start(src.ndim(), 0);
      Shape stop = slice_shape;
      start[axis] = idx_value;
      stop[axis] += idx_value;
      auto current = slice(result, start, stop, unit_strides, s);
      auto update = reshape(
          slice(
              flat_upd,
              {static_cast<ShapeElem>(i), 0},
              {static_cast<ShapeElem>(i + 1), static_cast<ShapeElem>(slice_size)},
              s),
          slice_shape,
          s);
      array reduced = current;
      switch (reduce_type) {
        case Scatter::Prod:
          reduced = multiply(current, update, s);
          break;
        case Scatter::Max:
          reduced = maximum(current, update, s);
          break;
        case Scatter::Min:
          reduced = minimum(current, update, s);
          break;
        default:
          return false;
      }
      result = slice_update(
          result,
          reduced,
          std::move(start),
          std::move(stop),
          std::move(unit_strides),
          s);
    }
    copy_gpu(result, out, CopyType::GeneralGeneral, s);
    return true;
  }

  if (reduce_type == Scatter::None) {
    if (auto scalar_index = scalar_index_value(idx); scalar_index.has_value()) {
      Shape start(src.ndim(), 0);
      Shape stop = upd.shape();
      Shape strides(src.ndim(), 1);
      auto normalized_index =
          normalize_scatter_index(*scalar_index, src.shape(axis));
      start[axis] = normalized_index;
      stop[axis] += normalized_index;
      array update_value = ensure_row_contiguous(upd, s);
      array next(src.shape(), src.dtype(), nullptr, {});
      next.set_data(allocator::malloc(next.nbytes()));
      next.set_status(array::Status::available);
      SliceUpdate op(s, SliceUpdate::None, start, stop, strides);
      op.eval_gpu({src, update_value}, next);
      out.copy_shared_buffer(next);
      return true;
    }
    if (idx.size() == 1) {
      if (auto single_idx = singleton_index_value(idx); single_idx.has_value()) {
        Shape slice_shape(
            upd.shape().begin() + idx.ndim(),
            upd.shape().end());
        Shape start(src.ndim(), 0);
        Shape stop = slice_shape;
        Shape strides(src.ndim(), 1);
        auto normalized_index =
            normalize_scatter_index(*single_idx, src.shape(axis));
        start[axis] = normalized_index;
        stop[axis] += normalized_index;
        array update_value =
            ensure_row_contiguous(reshape(upd, slice_shape, s), s);
        array next(src.shape(), src.dtype(), nullptr, {});
        next.set_data(allocator::malloc(next.nbytes()));
        next.set_status(array::Status::available);
        SliceUpdate op(s, SliceUpdate::None, start, stop, strides);
        op.eval_gpu({src, update_value}, next);
        out.copy_shared_buffer(next);
        return true;
      }
    }
  }
  const uint32_t slice_size = take_slice_size;
  const uint32_t expected_update_size = checked_mul_u32(
      index_count, slice_size, "scatter_take expected_update_size");
  if (upd.size() != expected_update_size) {
    trace_vulkan_unsupported("Scatter", "update size does not match indices");
    return false;
  }
  if (reduce_type == Scatter::Sum && out.dtype() == float32 &&
      !vulkan::VulkanContext::get().shader_buffer_atomic_float32_supported()) {
    trace_vulkan_unsupported(
        "Scatter", "float32 scatter-sum requires shader atomic float support");
    return false;
  }
  if (reduce_type == Scatter::Sum &&
      (out.dtype() == float16 || out.dtype() == bfloat16)) {
    if ((out.offset() % 4) != 0 || (out.size() % 2) != 0) {
      trace_vulkan_unsupported(
          "Scatter",
          "float16/bfloat16 scatter-sum requires 4-byte-aligned offset and even element count");
      return false;
    }
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

void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);

  const auto& dst = inputs[0];
  array mask = inputs[1];
  array src = inputs[2];
  auto& s = stream();

  const size_t total = mask.size();
  auto [out_work, staged_output] = make_output_work(out);
  copy_gpu(
      dst,
      out_work,
      total == 1 ? CopyType::Scalar : source_copy_type(dst),
      s);
  if (total == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return;
  }

  auto shader_id = masked_scatter_shader_id(out.dtype());
  if (!shader_id.has_value() || mask.dtype() != bool_) {
    throw std::runtime_error(
        "MaskedScatter operation failed on Vulkan (unsupported dtype).");
  }

  mask = flatten_in_eval(mask, 1, -1, s);
  mask = ensure_row_contiguous(mask, s);
  // src is already shaped as [batch, values_per_batch] by masked_scatter.
  src = ensure_row_contiguous(src, s);

  const uint32_t batch_count =
      checked_u32_size(mask.shape(0), "masked_scatter batch_count");
  const uint32_t mask_batch_size = checked_u32_size(
      mask.size() / batch_count, "masked_scatter mask_batch_size");
  const uint32_t src_batch_size = checked_u32_size(
      src.size() / src.shape(0), "masked_scatter src_batch_size");

  auto command_buffer = vulkan::begin_command_recording(s.index);
  vulkan::dispatch_masked_scatter_op(
      mask,
      src,
      out_work,
      *shader_id,
      command_buffer,
      s,
      src_batch_size,
      mask_batch_size);
  vulkan::end_command_recording(s.index);

  if (staged_output) {
    copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
  }
}

} // namespace mlx::core
