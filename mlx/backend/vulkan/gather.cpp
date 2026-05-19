// Copyright © 2024 Apple Inc.

#include <limits>
#include <utility>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/ops.h"

namespace mlx::core {

namespace {

uint32_t
checked_shape_product(const array& arr, int begin, int end, const char* label);

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
  if (idx < 0) {
    idx += axis_size;
  }
  if (idx < 0 || idx >= axis_size) {
    throw std::out_of_range(
        "gather index " + std::to_string(idx) + " out of bounds " +
        std::to_string(axis_size));
  }
  return idx;
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
      throw std::runtime_error("Unsupported index dtype for Vulkan gather.");
  }
}

bool is_full_range_index_for_axis(
    const array& idx,
    int64_t axis_size,
    Stream s) {
  if (axis_size <= 0 || idx.size() == 0 || (idx.size() % axis_size) != 0) {
    return false;
  }
  auto flat_idx = ensure_row_contiguous(
      reshape(idx, {static_cast<ShapeElem>(idx.size())}, s), s);
  flat_idx.eval();
  for (int i = 0; i < flat_idx.size(); ++i) {
    if (read_contiguous_index(flat_idx, i) != (i % axis_size)) {
      return false;
    }
  }
  return true;
}

constexpr uint32_t kMaxGatherPushConstants = 128;

std::string build_generic_gather_shader(
    Dtype value_dtype,
    Dtype index_dtype,
    int ndim,
    int nidx) {
  std::ostringstream os;

  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  os << "#extension GL_EXT_scalar_block_layout : require\n";

  bool needs_int64 =
      (index_dtype == int64 || index_dtype == uint64 || value_dtype == int64 ||
       value_dtype == uint64);
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

  os << "#define VALUE_TYPE " << vulkan::dtype_to_glsl_storage_type(value_dtype)
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

  os << "layout(scalar, binding = 0) readonly buffer Src { VALUE_TYPE src_data[]; };\n";
  for (int j = 0; j < nidx; ++j) {
    os << "layout(scalar, binding = " << (1 + j) << ") readonly buffer Idx" << j
       << " { INDEX_TYPE idx" << j << "_data[]; };\n";
  }
  os << "layout(scalar, binding = " << (1 + nidx)
     << ") writeonly buffer Out { VALUE_TYPE out_data[]; };\n\n";

  os << "uint normalize_index(INDEX_TYPE idx, uint axis_size) {\n";
  os << "#if defined(INDEX_IS_UNSIGNED)\n";
  os << "    return uint(idx);\n";
  os << "#elif defined(INDEX_IS_I64)\n";
  os << "    return uint(idx < int64_t(0) ? idx + int64_t(axis_size) : idx);\n";
  os << "#else\n";
  os << "    return uint(idx < 0 ? idx + int(axis_size) : idx);\n";
  os << "#endif\n";
  os << "}\n\n";

  os << "void main() {\n";
  os << "    uint linear_idx = gl_GlobalInvocationID.x;\n";
  os << "    if (linear_idx >= p.ne) return;\n\n";
  os << "    uint src_elem = linear_idx % p.slice_size;\n";
  os << "    uint idx_elem = linear_idx / p.slice_size;\n\n";
  os << "    uint src_offset = 0;\n";
  os << "    uint rem = src_elem;\n";
  os << "    for (int d = SRC_NDIM - 1; d >= 0; --d) {\n";
  os << "        uint sz = uint(max(p.slice_sizes[d], 1));\n";
  os << "        src_offset += (rem % sz) * p.src_strides[d];\n";
  os << "        rem /= sz;\n";
  os << "    }\n\n";

  for (int j = 0; j < nidx; ++j) {
    os << "    src_offset += normalize_index(idx" << j
       << "_data[idx_elem], p.src_shape[p.axes[" << j << "]]) "
       << "* p.src_strides[p.axes[" << j << "]];\n";
  }

  os << "\n    out_data[linear_idx] = src_data[src_offset];\n";
  os << "}\n";

  return os.str();
}

bool try_dispatch_generic_gather(
    const std::vector<array>& inputs,
    const std::vector<int>& norm_axes,
    const Shape& slice_sizes,
    int idx_ndim,
    uint32_t index_count,
    array& out,
    Stream s) {
  const auto& src_input = inputs[0];
  const int ndim = static_cast<int>(src_input.ndim());
  const int nidx = static_cast<int>(norm_axes.size());
  const Dtype value_dtype = src_input.dtype();
  const Dtype index_dtype = inputs[1].dtype();

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

  array src = ensure_row_contiguous(src_input, s);
  if (!ensure_vulkan_storage(src, s)) {
    return false;
  }

  auto [out_work, staged_output] = make_output_work(out);
  if (!ensure_vulkan_storage(out_work, s)) {
    return false;
  }
  if (out_work.size() == 0) {
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

  std::vector<uint32_t> ss(ndim);
  for (int d = 0; d < ndim; ++d) {
    ss[d] = static_cast<uint32_t>(slice_sizes[d]);
  }

  std::vector<uint32_t> px_axes(nidx);
  for (int a = 0; a < nidx; ++a) {
    px_axes[a] = static_cast<uint32_t>(norm_axes[a]);
  }

  const uint32_t slice_size =
      std::accumulate(ss.begin(), ss.end(), 1u, std::multiplies<uint32_t>());
  const uint32_t ne = index_count * slice_size;

  std::vector<uint32_t> pc;
  pc.push_back(ne);
  pc.push_back(slice_size);
  pc.insert(pc.end(), src_shape.begin(), src_shape.end());
  pc.insert(pc.end(), src_strides.begin(), src_strides.end());
  pc.insert(pc.end(), ss.begin(), ss.end());
  pc.insert(pc.end(), px_axes.begin(), px_axes.end());

  const uint32_t pc_size = static_cast<uint32_t>(pc.size() * sizeof(uint32_t));
  if (pc_size > kMaxGatherPushConstants) {
    return false;
  }

  std::string shader_name = "dynamic_gather_generic_" +
      dtype_suffix(value_dtype) + "_" + gather_index_suffix(index_dtype) + "_" +
      std::to_string(ndim) + "d_" + std::to_string(nidx) + "a_" +
      std::to_string(index_count) + "i";

  std::string glsl =
      build_generic_gather_shader(value_dtype, index_dtype, ndim, nidx);

  const uint32_t num_bindings = static_cast<uint32_t>(2 + nidx);
  std::vector<vulkan::DynamicArrayRef> refs;
  refs.reserve(num_bindings);
  refs.push_back({&src, 0});
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

    vulkan::retain_array_for_stream(s, src);
    for (const auto& flat_idx : flat_indices) {
      vulkan::retain_array_for_stream(s, flat_idx);
    }
    vulkan::retain_array_for_stream(s, out_work);

    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    if (trace_fallback_enabled()) {
      trace_fallback(
          "generic_gather_dispatch_failed reason=dynamic_shader_error");
    }
    return false;
  }
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
        slice(
            src_input,
            std::move(start),
            std::move(stop),
            std::move(strides),
            s),
        out,
        CopyType::GeneralGeneral,
        s);
    return true;
  }

  if (inputs.size() < 2 || inputs.size() != axes.size() + 1) {
    return false;
  }

  if (axes.size() > 2) {
    std::vector<int> norm_axes;
    norm_axes.reserve(axes.size());
    for (int ax : axes) {
      int norm = normalize_axis(ax, src_input.ndim());
      if (norm < 0 || norm >= src_input.ndim()) {
        return false;
      }
      norm_axes.push_back(norm);
    }
    for (int i = 0; i < norm_axes.size(); ++i) {
      for (int j = i + 1; j < norm_axes.size(); ++j) {
        if (norm_axes[i] == norm_axes[j]) {
          return false;
        }
      }
    }

    const int idx_ndim = inputs[1].ndim();
    for (int i = 2; i < inputs.size(); ++i) {
      if (inputs[i].shape() != inputs[1].shape() ||
          inputs[i].dtype() != inputs[1].dtype()) {
        return false;
      }
    }

    if (out.ndim() != idx_ndim + src_input.ndim()) {
      return false;
    }

    Shape out_slice_shape(out.shape().begin() + idx_ndim, out.shape().end());
    if (out_slice_shape != slice_sizes) {
      return false;
    }

    const auto index_count =
        checked_u32_size(inputs[1].size(), "gather_generic index_count");

    if (try_dispatch_generic_gather(
            inputs, norm_axes, slice_sizes, idx_ndim, index_count, out, s)) {
      return true;
    }

    if (trace_fallback_enabled()) {
      trace_fallback("generic_gather_gpu_unavailable fallback=host_loop");
    }

    std::vector<array> flat_indices;
    flat_indices.reserve(inputs.size() - 1);
    for (int i = 1; i < inputs.size(); ++i) {
      flat_indices.push_back(ensure_host_readable_row_contiguous(
          reshape(inputs[i], {static_cast<ShapeElem>(index_count)}, s), s));
    }

    auto [out_work, staged_output] = make_output_work(out);
    if (out_work.size() == 0) {
      if (staged_output) {
        copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
      }
      return true;
    }

    Strides out_slice_strides(
        out_work.strides().begin() + idx_ndim, out_work.strides().end());
    size_t out_slice_elems = 1;
    for (auto dim : out_slice_shape) {
      out_slice_elems *= static_cast<size_t>(dim);
    }
    auto [out_slice_data_size, out_slice_row_contig, out_slice_col_contig] =
        check_contiguity(out_slice_shape, out_slice_strides);
    array::Flags out_slice_flags = {
        out_slice_data_size == out_slice_elems,
        out_slice_row_contig,
        out_slice_col_contig};

    Strides index_shape_strides(idx_ndim, 1);
    for (int i = idx_ndim - 2; i >= 0; --i) {
      index_shape_strides[i] =
          index_shape_strides[i + 1] * inputs[1].shape(i + 1);
    }

    for (uint32_t i = 0; i < index_count; ++i) {
      Shape start(src_input.ndim(), 0);
      Shape stop = slice_sizes;
      Shape unit_strides(src_input.ndim(), 1);
      for (int j = 0; j < norm_axes.size(); ++j) {
        const int axis = norm_axes[j];
        start[axis] = normalize_gather_index(
            read_contiguous_index(flat_indices[j], i), src_input.shape(axis));
        stop[axis] += start[axis];
        if (stop[axis] > src_input.shape(axis)) {
          return false;
        }
      }

      array gathered = slice(src_input, start, stop, unit_strides, s);

      int64_t out_offset = 0;
      size_t remainder = i;
      for (int d = 0; d < idx_ndim; ++d) {
        const size_t coord = remainder / index_shape_strides[d];
        remainder %= index_shape_strides[d];
        out_offset += coord * out_work.strides(d);
      }

      array out_slice(out_slice_shape, out.dtype(), nullptr, {});
      out_slice.copy_shared_buffer(
          out_work,
          out_slice_strides,
          out_slice_flags,
          out_slice_data_size,
          out_offset);
      out_slice.set_status(array::Status::available);
      copy_gpu_inplace(gathered, out_slice, CopyType::GeneralGeneral, s);
    }

    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
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
      const int idx_ndim = idx0.ndim();
      if (out.ndim() != idx_ndim + src_input.ndim()) {
        return false;
      }
      Shape out_slice_shape(out.shape().begin() + idx_ndim, out.shape().end());
      if (out_slice_shape != slice_sizes) {
        return false;
      }

      std::vector<array> generic_inputs = {src_input, idx0, idx1};
      std::vector<int> norm_axes = {axis0, axis1};
      const auto index_count =
          checked_u32_size(idx0.size(), "gather_pair fallback index_count");
      if (try_dispatch_generic_gather(
              generic_inputs,
              norm_axes,
              slice_sizes,
              idx_ndim,
              index_count,
              out,
              s)) {
        return true;
      }

      idx0 = ensure_host_readable_row_contiguous(
          reshape(idx0, {static_cast<ShapeElem>(idx0.size())}, s), s);
      idx1 = ensure_host_readable_row_contiguous(
          reshape(idx1, {static_cast<ShapeElem>(idx1.size())}, s), s);

      auto [out_work, staged_output] = make_output_work(out);
      if (out_work.size() == 0) {
        if (staged_output) {
          copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
        }
        return true;
      }

      Strides out_slice_strides(
          out_work.strides().begin() + idx_ndim, out_work.strides().end());
      size_t out_slice_elems = 1;
      for (auto dim : out_slice_shape) {
        out_slice_elems *= static_cast<size_t>(dim);
      }
      auto [out_slice_data_size, out_slice_row_contig, out_slice_col_contig] =
          check_contiguity(out_slice_shape, out_slice_strides);
      array::Flags out_slice_flags = {
          out_slice_data_size == out_slice_elems,
          out_slice_row_contig,
          out_slice_col_contig};

      Strides index_shape_strides(idx_ndim, 1);
      for (int i = idx_ndim - 2; i >= 0; --i) {
        index_shape_strides[i] =
            index_shape_strides[i + 1] * inputs[1].shape(i + 1);
      }

      for (uint32_t i = 0; i < index_count; ++i) {
        Shape start(src_input.ndim(), 0);
        Shape stop = slice_sizes;
        Shape unit_strides(src_input.ndim(), 1);
        start[axis0] = normalize_gather_index(
            read_contiguous_index(idx0, i), src_input.shape(axis0));
        start[axis1] = normalize_gather_index(
            read_contiguous_index(idx1, i), src_input.shape(axis1));
        stop[axis0] += start[axis0];
        stop[axis1] += start[axis1];
        if (stop[axis0] > src_input.shape(axis0) ||
            stop[axis1] > src_input.shape(axis1)) {
          return false;
        }

        array gathered = slice(src_input, start, stop, unit_strides, s);

        int64_t out_offset = 0;
        size_t remainder = i;
        for (int d = 0; d < idx_ndim; ++d) {
          const size_t coord = remainder / index_shape_strides[d];
          remainder %= index_shape_strides[d];
          out_offset += coord * out_work.strides(d);
        }

        array out_slice(out_slice_shape, out.dtype(), nullptr, {});
        out_slice.copy_shared_buffer(
            out_work,
            out_slice_strides,
            out_slice_flags,
            out_slice_data_size,
            out_offset);
        out_slice.set_status(array::Status::available);
        copy_gpu_inplace(gathered, out_slice, CopyType::GeneralGeneral, s);
      }

      if (staged_output) {
        copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
      }
      return true;
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
        slice(
            src_input,
            std::move(start),
            std::move(stop),
            std::move(strides),
            s),
        out,
        CopyType::GeneralGeneral,
        s);
    return true;
  }
  if (auto singleton_index = singleton_index_value(idx);
      singleton_index.has_value() &&
      out.size() ==
          static_cast<size_t>(std::accumulate(
              slice_sizes.begin(),
              slice_sizes.end(),
              int64_t{1},
              std::multiplies<int64_t>()))) {
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
