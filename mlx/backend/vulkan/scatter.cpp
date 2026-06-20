// Copyright © 2024 Apple Inc.

#include <utility>

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

bool needs_row_contiguous(const array& arr) {
  return !arr.flags().row_contiguous || arr.offset() != 0 ||
      arr.data_size() != arr.size();
}

array ensure_row_contiguous(array arr, Stream s) {
  if (needs_row_contiguous(arr)) {
    arr = contiguous_copy_gpu(arr, s);
  }
  return arr;
}

bool ensure_vulkan_storage(array& arr, Stream s) {
  if (vulkan::is_vulkan_storage_array(arr)) {
    return true;
  }
  array storage(arr.shape(), arr.dtype(), nullptr, {});
  storage.set_data(allocator::malloc(storage.nbytes()));
  storage.set_status(array::Status::available);
  copy_gpu(arr, storage, CopyType::General, s);
  arr = std::move(storage);
  return vulkan::is_vulkan_storage_array(arr);
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

constexpr uint32_t kMaxScatterPushConstants = 128;

bool supports_dynamic_scatter_sum_dtype(Dtype dtype) {
  return dtype == float32 || dtype == int32 || dtype == uint32;
}

bool supports_dynamic_scatter_reduction_dtype(
    Dtype dtype,
    Scatter::ReduceType reduce_type) {
  switch (reduce_type) {
    case Scatter::None:
      return true;
    case Scatter::Sum:
      return supports_dynamic_scatter_sum_dtype(dtype);
    case Scatter::Prod:
    case Scatter::Max:
    case Scatter::Min:
      return dtype == float32 || dtype == int32 || dtype == uint32;
  }
  return false;
}

std::string build_generic_scatter_shader(
    Dtype value_dtype,
    Dtype index_dtype,
    int ndim,
    int nidx,
    Scatter::ReduceType reduce_type) {
  std::ostringstream os;
  const bool use_float_atomic_cas = value_dtype == float32 &&
      (reduce_type == Scatter::Prod || reduce_type == Scatter::Max ||
       reduce_type == Scatter::Min);
  const bool use_integer_atomic_cas =
      (value_dtype == int32 || value_dtype == uint32) &&
      (reduce_type == Scatter::Prod || reduce_type == Scatter::Max ||
       reduce_type == Scatter::Min);

  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  os << "#extension GL_EXT_scalar_block_layout : require\n";

  bool needs_int64 = index_dtype == int64 || index_dtype == uint64 ||
      value_dtype == int64 || value_dtype == uint64;
  if (needs_int64) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  if (vulkan::uses_float16_extension(value_dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
  }
  if (vulkan::uses_16bit_storage(value_dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  if (vulkan::uses_8bit_storage(value_dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
    os << "#extension GL_EXT_shader_8bit_storage : require\n";
  }
  if (reduce_type == Scatter::Sum && value_dtype == float32) {
    os << "#extension GL_EXT_shader_atomic_float : require\n";
  }

  os << "\n#define SRC_NDIM " << ndim << "\n";
  os << "#define NIDX " << nidx << "\n";

  if (index_dtype == int64) {
    os << "#define INDEX_IS_I64\n";
    os << "#define INDEX_TYPE int64_t\n";
  } else if (index_dtype == uint64) {
    os << "#define INDEX_IS_I64\n";
    os << "#define INDEX_IS_UNSIGNED\n";
    os << "#define INDEX_TYPE uint64_t\n";
  } else if (index_dtype == uint32) {
    os << "#define INDEX_IS_UNSIGNED\n";
    os << "#define INDEX_TYPE uint\n";
  } else {
    os << "#define INDEX_TYPE int\n";
  }

  os << "#define VALUE_TYPE "
     << (use_float_atomic_cas ? "uint"
                              : vulkan::dtype_to_glsl_storage_type(value_dtype))
     << "\n";

  os << "\nlayout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;\n\n";

  os << "layout(push_constant) uniform Params {\n";
  os << "    uint ne;\n";
  os << "    uint slice_size;\n";
  os << "    uint src_shape[SRC_NDIM];\n";
  os << "    uint src_strides[SRC_NDIM];\n";
  os << "    uint slice_sizes[SRC_NDIM];\n";
  os << "    uint axes[NIDX];\n";
  os << "} p;\n\n";

  os << "layout(scalar, binding = 0) readonly buffer Updates { VALUE_TYPE upd_data[]; };\n";
  for (int j = 0; j < nidx; ++j) {
    os << "layout(scalar, binding = " << (1 + j) << ") readonly buffer Idx" << j
       << " { INDEX_TYPE idx" << j << "_data[]; };\n";
  }
  os << "layout(scalar, binding = " << (1 + nidx)
     << ") buffer Out { VALUE_TYPE out_data[]; };\n\n";

  os << "uint normalize_index(INDEX_TYPE idx, uint axis_size) {\n";
  os << "#if defined(INDEX_IS_UNSIGNED)\n";
  os << "    return uint(idx);\n";
  os << "#elif defined(INDEX_IS_I64)\n";
  os << "    return uint(idx < int64_t(0) ? idx + int64_t(axis_size) : idx);\n";
  os << "#else\n";
  os << "    return uint(idx < 0 ? idx + int(axis_size) : idx);\n";
  os << "#endif\n";
  os << "}\n\n";

  if (use_float_atomic_cas) {
    os << "float read_update(uint idx) { return uintBitsToFloat(upd_data[idx]); }\n";
    os << "void atomic_reduce(uint dst_offset, float value) {\n";
    os << "    uint old_bits = out_data[dst_offset];\n";
    os << "    while (true) {\n";
    os << "        float old_value = uintBitsToFloat(old_bits);\n";
    if (reduce_type == Scatter::Prod) {
      os << "        float new_value = old_value * value;\n";
    } else if (reduce_type == Scatter::Max) {
      os << "        float new_value = max(old_value, value);\n";
    } else {
      os << "        float new_value = min(old_value, value);\n";
    }
    os << "        uint new_bits = floatBitsToUint(new_value);\n";
    os << "        uint prev_bits = atomicCompSwap(out_data[dst_offset], old_bits, new_bits);\n";
    os << "        if (prev_bits == old_bits) break;\n";
    os << "        old_bits = prev_bits;\n";
    os << "    }\n";
    os << "}\n\n";
  } else if (use_integer_atomic_cas) {
    os << "void atomic_reduce(uint dst_offset, VALUE_TYPE value) {\n";
    os << "    VALUE_TYPE old_value = out_data[dst_offset];\n";
    os << "    while (true) {\n";
    if (reduce_type == Scatter::Prod) {
      os << "        VALUE_TYPE new_value = old_value * value;\n";
    } else if (reduce_type == Scatter::Max) {
      os << "        VALUE_TYPE new_value = max(old_value, value);\n";
    } else {
      os << "        VALUE_TYPE new_value = min(old_value, value);\n";
    }
    os << "        VALUE_TYPE prev_value = atomicCompSwap(out_data[dst_offset], old_value, new_value);\n";
    os << "        if (prev_value == old_value) break;\n";
    os << "        old_value = prev_value;\n";
    os << "    }\n";
    os << "}\n\n";
  }

  os << "void main() {\n";
  os << "    uint linear_idx = gl_GlobalInvocationID.x;\n";
  os << "    if (linear_idx >= p.ne) return;\n\n";
  os << "    uint slice_elem = linear_idx % p.slice_size;\n";
  os << "    uint idx_elem = linear_idx / p.slice_size;\n\n";
  os << "    uint dst_offset = 0;\n";
  os << "    uint rem = slice_elem;\n";
  os << "    for (int d = SRC_NDIM - 1; d >= 0; --d) {\n";
  os << "        uint sz = max(p.slice_sizes[d], 1u);\n";
  os << "        dst_offset += (rem % sz) * p.src_strides[d];\n";
  os << "        rem /= sz;\n";
  os << "    }\n\n";

  for (int j = 0; j < nidx; ++j) {
    os << "    dst_offset += normalize_index(idx" << j
       << "_data[idx_elem], p.src_shape[p.axes[" << j << "]])"
       << " * p.src_strides[p.axes[" << j << "]];\n";
  }

  if (reduce_type == Scatter::Sum) {
    os << "\n    atomicAdd(out_data[dst_offset], upd_data[linear_idx]);\n";
  } else if (use_float_atomic_cas) {
    os << "\n    atomic_reduce(dst_offset, read_update(linear_idx));\n";
  } else if (use_integer_atomic_cas) {
    os << "\n    atomic_reduce(dst_offset, upd_data[linear_idx]);\n";
  } else if (reduce_type == Scatter::Prod) {
    os << "\n    out_data[dst_offset] *= upd_data[linear_idx];\n";
  } else if (reduce_type == Scatter::Max) {
    os << "\n    out_data[dst_offset] = max(out_data[dst_offset], upd_data[linear_idx]);\n";
  } else if (reduce_type == Scatter::Min) {
    os << "\n    out_data[dst_offset] = min(out_data[dst_offset], upd_data[linear_idx]);\n";
  } else {
    os << "\n    out_data[dst_offset] = upd_data[linear_idx];\n";
  }
  os << "}\n";

  return os.str();
}

bool try_dispatch_generic_scatter(
    const std::vector<array>& inputs,
    const std::vector<int>& norm_axes,
    const Shape& update_shape,
    uint32_t index_count,
    array& out,
    Stream s,
    Scatter::ReduceType reduce_type) {
  if (reduce_type != Scatter::None && reduce_type != Scatter::Sum &&
      reduce_type != Scatter::Prod && reduce_type != Scatter::Max &&
      reduce_type != Scatter::Min) {
    return false;
  }

  const auto& src_input = inputs[0];
  const int ndim = static_cast<int>(src_input.ndim());
  const int nidx = static_cast<int>(norm_axes.size());
  const Dtype value_dtype = src_input.dtype();
  const Dtype index_dtype = inputs[1].dtype();

  if (!supports_dynamic_scatter_reduction_dtype(value_dtype, reduce_type)) {
    return false;
  }

  std::vector<array> flat_indices;
  flat_indices.reserve(nidx);
  for (int i = 0; i < nidx; ++i) {
    auto flat_idx = ensure_row_contiguous(
        reshape(inputs[1 + i], {static_cast<ShapeElem>(index_count)}, s), s);
    if (!ensure_vulkan_storage(flat_idx, s)) {
      return false;
    }
    flat_indices.push_back(flat_idx);
  }

  array upd = ensure_row_contiguous(inputs.back(), s);
  if (!ensure_vulkan_storage(upd, s)) {
    return false;
  }
  auto [out_work, staged_output] = make_output_work(out);
  copy_gpu(src_input, out_work, source_copy_type(src_input), s);
  if (!ensure_vulkan_storage(out_work, s)) {
    return false;
  }

  if (upd.size() == 0 || out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  std::vector<uint32_t> src_shape(ndim);
  std::vector<uint32_t> src_strides(ndim);
  for (int d = 0; d < ndim; ++d) {
    src_shape[d] = static_cast<uint32_t>(src_input.shape(d));
  }
  for (int d = ndim - 1; d >= 0; --d) {
    src_strides[d] =
        (d == ndim - 1) ? 1u : src_strides[d + 1] * src_shape[d + 1];
  }

  std::vector<uint32_t> slice_sizes(ndim);
  uint32_t slice_size = 1;
  for (int d = 0; d < ndim; ++d) {
    slice_sizes[d] = static_cast<uint32_t>(update_shape[d]);
    slice_size = checked_mul_u32(
        slice_size, slice_sizes[d], "generic_scatter slice_size");
  }

  std::vector<uint32_t> px_axes(nidx);
  for (int a = 0; a < nidx; ++a) {
    px_axes[a] = static_cast<uint32_t>(norm_axes[a]);
  }

  const uint32_t ne =
      checked_mul_u32(index_count, slice_size, "generic_scatter ne");
  std::vector<uint32_t> pc;
  pc.push_back(ne);
  pc.push_back(slice_size);
  pc.insert(pc.end(), src_shape.begin(), src_shape.end());
  pc.insert(pc.end(), src_strides.begin(), src_strides.end());
  pc.insert(pc.end(), slice_sizes.begin(), slice_sizes.end());
  pc.insert(pc.end(), px_axes.begin(), px_axes.end());

  const uint32_t pc_size = static_cast<uint32_t>(pc.size() * sizeof(uint32_t));
  if (pc_size > kMaxScatterPushConstants) {
    return false;
  }

  std::string shader_name =
      std::string(
          reduce_type == Scatter::Sum        ? "dynamic_scatter_sum_generic_"
              : reduce_type == Scatter::Prod ? "dynamic_scatter_prod_generic_"
              : reduce_type == Scatter::Max  ? "dynamic_scatter_max_generic_"
              : reduce_type == Scatter::Min  ? "dynamic_scatter_min_generic_"
                                             : "dynamic_scatter_generic_") +
      dtype_suffix(value_dtype) + "_" + gather_index_suffix(index_dtype) + "_" +
      std::to_string(ndim) + "d_" + std::to_string(nidx) + "a";
  std::string glsl = build_generic_scatter_shader(
      value_dtype, index_dtype, ndim, nidx, reduce_type);

  const uint32_t num_bindings = static_cast<uint32_t>(2 + nidx);
  std::vector<vulkan::DynamicArrayRef> refs;
  refs.reserve(num_bindings);
  refs.push_back({&upd, 0});
  for (int j = 0; j < nidx; ++j) {
    refs.push_back({&flat_indices[j], static_cast<uint32_t>(1 + j)});
  }
  refs.push_back({&out_work, static_cast<uint32_t>(1 + nidx)});

  try {
    auto dispatch = vulkan::dispatch_dynamic_compute_begin(
        shader_name, glsl, num_bindings, refs.data(), pc_size, s);

    vkCmdPushConstants(
        dispatch.command_buffer,
        dispatch.pipeline->layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        pc_size,
        pc.data());
    vkCmdDispatch(dispatch.command_buffer, (ne + 511u) / 512u, 1, 1);
    vulkan::end_command_recording(s.index);

    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    if (trace_fallback_enabled()) {
      trace_fallback(
          "generic_scatter_dispatch_failed reason=dynamic_shader_error");
    }
    return false;
  }
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
    if (src.ndim() != 0 || upd.shape() != src.shape() ||
        upd.dtype() != src.dtype() || out.shape() != src.shape() ||
        out.dtype() != src.dtype()) {
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

    if (!duplicate_axes && norm_axes.size() > 2 &&
        (reduce_type == Scatter::None || reduce_type == Scatter::Sum ||
         reduce_type == Scatter::Prod || reduce_type == Scatter::Max ||
         reduce_type == Scatter::Min)) {
      const int idx_ndim = inputs[1].ndim();
      bool consistent_indices = true;
      for (int i = 2; i < inputs.size() - 1; ++i) {
        if (inputs[i].shape() != inputs[1].shape() ||
            inputs[i].dtype() != inputs[1].dtype()) {
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
                checked_u32_size(
                    inputs[1].size(), "scatter_generic index_count"),
                slice_elems,
                "scatter_generic update_size") == upd.size()) {
          if (try_dispatch_generic_scatter(
                  inputs,
                  norm_axes,
                  update_shape,
                  checked_u32_size(
                      inputs[1].size(), "scatter_generic index_count"),
                  out,
                  s,
                  reduce_type)) {
            return true;
          }
          return false;
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
      uint32_t slice_elems = 1;
      for (auto dim : update_shape) {
        slice_elems = checked_mul_u32(
            slice_elems,
            checked_u32_size(dim, "scatter_pair slice_elems"),
            "scatter_pair slice_elems");
      }
      if (update_shape.size() == src.ndim() &&
          checked_mul_u32(
              checked_u32_size(
                  idx0.size(), "scatter_pair composed index_count"),
              slice_elems,
              "scatter_pair composed update_size") == upd.size() &&
          (reduce_type == Scatter::None || reduce_type == Scatter::Sum ||
           reduce_type == Scatter::Prod || reduce_type == Scatter::Max ||
           reduce_type == Scatter::Min)) {
        std::vector<int> norm_axes = {axis0, axis1};
        std::vector<array> generic_inputs = {src, idx0, idx1, upd};
        if (try_dispatch_generic_scatter(
                generic_inputs,
                norm_axes,
                update_shape,
                index_count,
                out,
                s,
                reduce_type)) {
          return true;
        }
        return false;
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
      return false;
    }

    idx0 = ensure_row_contiguous(idx0, s);
    idx1 = ensure_row_contiguous(idx1, s);
    upd = ensure_row_contiguous(upd, s);

    if (reduce_type == Scatter::Sum && out.dtype() == float32 &&
        !vulkan::VulkanContext::get()
             .shader_buffer_atomic_float32_supported()) {
      trace_vulkan_unsupported(
          "Scatter",
          "float32 scatter-sum pair requires shader atomic float support");
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
  const uint32_t take_slice_size =
      checked_mul_u32(size_pre, size_post, "scatter_take slice_size");

  if (idx.size() == 0 || upd.size() == 0) {
    copy_gpu(src, out, source_copy_type(src), s);
    return true;
  }

  if (upd.ndim() == idx.ndim() + src.ndim()) {
    Shape update_shape(upd.shape().begin() + idx.ndim(), upd.shape().end());
    uint32_t slice_elems = 1;
    for (auto dim : update_shape) {
      slice_elems = checked_mul_u32(
          slice_elems,
          checked_u32_size(dim, "scatter slice_elems"),
          "scatter slice_elems");
    }
    if (update_shape.size() == src.ndim() &&
        checked_mul_u32(
            checked_u32_size(idx.size(), "scatter composed index_count"),
            slice_elems,
            "scatter composed update_size") == upd.size() &&
        (reduce_type == Scatter::None || reduce_type == Scatter::Sum ||
         reduce_type == Scatter::Prod || reduce_type == Scatter::Max ||
         reduce_type == Scatter::Min)) {
      std::vector<int> norm_axes = {axis};
      if (try_dispatch_generic_scatter(
              inputs,
              norm_axes,
              update_shape,
              index_count,
              out,
              s,
              reduce_type)) {
        return true;
      }
      return false;
    }
  }
  const uint32_t slice_size = take_slice_size;
  const uint32_t expected_update_size = checked_mul_u32(
      index_count, slice_size, "scatter_take expected_update_size");
  if (upd.size() != expected_update_size) {
    trace_vulkan_unsupported("Scatter", "update size does not match indices");
    return false;
  }
  if (reduce_type != Scatter::None && reduce_type != Scatter::Sum) {
    trace_vulkan_unsupported(
        "Scatter",
        "single-axis non-composed reduction is not supported by Vulkan");
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

  if (reduce_type != ScatterAxis::None && reduce_type != ScatterAxis::Sum) {
    trace_vulkan_unsupported(
        "ScatterAxis",
        "only put_along_axis and scatter_add_axis are implemented");
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

  if (reduce_type == ScatterAxis::Sum && out.dtype() == float32 &&
      !vulkan::VulkanContext::get().shader_buffer_atomic_float32_supported()) {
    trace_vulkan_unsupported(
        "ScatterAxis",
        "float32 scatter_add_axis requires shader atomic float support");
    return false;
  }

  const auto shader_id = reduce_type == ScatterAxis::Sum
      ? scatter_sum_axis_shader_id(out.dtype(), idx.dtype())
      : scatter_axis_shader_id(out.dtype(), idx.dtype());
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
      dst, out_work, total == 1 ? CopyType::Scalar : source_copy_type(dst), s);
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
