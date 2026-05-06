// Copyright © 2024 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

#include "mlx/backend/vulkan/shader_compiler.h"

namespace {

using mlx::core::array;
using mlx::core::Dtype;
using mlx::core::Shape;
using mlx::core::Strides;
namespace vulkan = mlx::core::vulkan;

using vulkan::cast_expr_for_dtype;
using vulkan::dtype_to_glsl_storage_type;
using vulkan::emit_dynamic_shader_preamble;
using vulkan::is_vulkan_storage_array;
using vulkan::zero_literal_for_dtype;

constexpr size_t kMinTransferQueueCopyBytes = 256 * 1024;

bool has_vulkan_storage(const array& arr) {
  return arr.data_shared_ptr() != nullptr &&
      vulkan::is_vulkan_buffer(arr.buffer());
}

std::string copy_dtype_suffix(Dtype dtype);
bool needs_bf16_helpers(Dtype in_dtype, Dtype out_dtype);
std::string emit_bf16_conversion_helpers();
std::string
bf16_cast_expr(const std::string& expr, Dtype in_dtype, Dtype out_dtype);

bool has_row_contiguous_strides(const mlx::core::array& arr) {
  if (arr.ndim() == 0) {
    return true;
  }
  int64_t expected = 1;
  for (int i = static_cast<int>(arr.ndim()) - 1; i >= 0; --i) {
    if (arr.shape(i) == 1) {
      continue;
    }
    if (arr.strides(i) != expected) {
      return false;
    }
    expected *= arr.shape(i);
  }
  return true;
}

int64_t element_offset(const mlx::core::array& arr) {
  const auto item_size = static_cast<int64_t>(mlx::core::size_of(arr.dtype()));
  return item_size == 0 ? 0 : arr.offset() / item_size;
}

bool has_large_element_offset(const mlx::core::array& arr) {
  return element_offset(arr) > static_cast<int64_t>(0xFFFFu);
}

std::vector<VkBufferCopy> make_strided_copy_regions(
    const mlx::core::array& in,
    const mlx::core::array& out) {
  if (in.shape() != out.shape() || in.dtype() != out.dtype()) {
    return {};
  }

  const auto& shape = in.shape();
  const auto& in_strides = in.strides();
  const auto& out_strides = out.strides();
  if (shape.empty()) {
    return {
        {static_cast<VkDeviceSize>(in.offset()),
         static_cast<VkDeviceSize>(out.offset()),
         static_cast<VkDeviceSize>(mlx::core::size_of(in.dtype()))}};
  }

  size_t block_elems = 1;
  size_t suffix_begin = shape.size();
  for (size_t dim = shape.size(); dim-- > 0;) {
    if (shape[dim] == 1) {
      suffix_begin = dim;
      continue;
    }
    if (in_strides[dim] != static_cast<int64_t>(block_elems) ||
        out_strides[dim] != static_cast<int64_t>(block_elems)) {
      break;
    }
    block_elems *= static_cast<size_t>(shape[dim]);
    suffix_begin = dim;
  }

  if (block_elems == 0) {
    return {};
  }

  const auto item_size =
      static_cast<VkDeviceSize>(mlx::core::size_of(in.dtype()));
  std::vector<VkBufferCopy> regions;

  std::function<void(size_t, int64_t, int64_t)> emit_regions =
      [&](size_t dim, int64_t in_elem_offset, int64_t out_elem_offset) {
        if (dim == suffix_begin) {
          VkBufferCopy region{};
          region.srcOffset = static_cast<VkDeviceSize>(in.offset()) +
              static_cast<VkDeviceSize>(in_elem_offset) * item_size;
          region.dstOffset = static_cast<VkDeviceSize>(out.offset()) +
              static_cast<VkDeviceSize>(out_elem_offset) * item_size;
          region.size = static_cast<VkDeviceSize>(block_elems) * item_size;
          regions.push_back(region);
          return;
        }

        for (int64_t i = 0; i < shape[dim]; ++i) {
          emit_regions(
              dim + 1,
              in_elem_offset + i * in_strides[dim],
              out_elem_offset + i * out_strides[dim]);
        }
      };

  emit_regions(0, 0, 0);
  return regions;
}

size_t num_elements(const Shape& shape) {
  return std::accumulate(
      shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
}

bool supports_dynamic_gpu_cast_dtype(Dtype dtype) {
  switch (dtype) {
    case mlx::core::bool_:
    case mlx::core::uint8:
    case mlx::core::uint16:
    case mlx::core::uint32:
    case mlx::core::uint64:
    case mlx::core::int8:
    case mlx::core::int16:
    case mlx::core::int32:
    case mlx::core::int64:
    case mlx::core::float16:
    case mlx::core::float32:
    case mlx::core::bfloat16:
    case mlx::core::complex64:
      return true;
    default:
      return false;
  }
}

bool supports_dynamic_gpu_cast_pair(Dtype in_dtype, Dtype out_dtype) {
  if (in_dtype == mlx::core::complex64 || out_dtype == mlx::core::complex64) {
    return in_dtype == out_dtype ||
        (in_dtype == mlx::core::complex64 && out_dtype == mlx::core::float32) ||
        (in_dtype != mlx::core::complex64 && out_dtype == mlx::core::complex64);
  }
  return supports_dynamic_gpu_cast_dtype(in_dtype) &&
      supports_dynamic_gpu_cast_dtype(out_dtype);
}

void append_layout_key(std::ostringstream& os, const Shape& shape) {
  for (auto dim : shape) {
    os << dim << ',';
  }
}

void append_layout_key(std::ostringstream& os, const Strides& strides) {
  for (auto stride : strides) {
    os << stride << ',';
  }
}

void validate_dynamic_offset_array(
    const std::optional<mlx::core::array>& offset) {
  if (!offset.has_value()) {
    return;
  }
  if (offset->size() != 1 || offset->dtype() != mlx::core::int64) {
    throw std::runtime_error(
        "Dynamic Vulkan copy offsets must be int64 scalars.");
  }
}

std::string build_dynamic_offset_shader(
    Dtype dtype,
    int64_t indices_base_offset,
    const std::vector<int64_t>& stride_terms) {
  std::ostringstream os;
  os << emit_dynamic_shader_preamble(dtype, true);
  os << "layout(set = 0, binding = 0) readonly buffer IndicesBuffer {"
     << dtype_to_glsl_storage_type(dtype) << " data[];} indices_buf;\n";
  os << "layout(set = 0, binding = 1) buffer OffsetBuffer {int64_t data[];} offset_buf;\n\n";
  os << "void main() {\n";
  os << "  if (gl_GlobalInvocationID.x != 0u) {\n";
  os << "    return;\n";
  os << "  }\n";
  os << "  int64_t acc = 0;\n";
  for (size_t i = 0; i < stride_terms.size(); ++i) {
    os << "  acc += int64_t(indices_buf.data["
       << (indices_base_offset + static_cast<int64_t>(i)) << "]) * int64_t("
       << stride_terms[i] << ");\n";
  }
  os << "  offset_buf.data[0] = acc;\n";
  os << "}\n";
  return os.str();
}

std::string build_dynamic_general_copy_shader(
    Dtype in_dtype,
    Dtype out_dtype,
    const Shape& shape,
    const Strides& i_strides,
    const Strides& o_strides,
    bool has_dynamic_i_offset,
    bool has_dynamic_o_offset) {
  std::ostringstream os;
  os << emit_dynamic_shader_preamble(in_dtype, out_dtype, true);
  if (needs_bf16_helpers(in_dtype, out_dtype)) {
    os << emit_bf16_conversion_helpers();
  }
  os << "layout(push_constant) uniform PushConstants {\n";
  os << "  uint total_elements;\n";
  os << "  int64_t input_base;\n";
  os << "  int64_t output_base;\n";
  os << "  int64_t dynamic_i_base;\n";
  os << "  int64_t dynamic_o_base;\n";
  os << "} pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputBuffer {"
     << dtype_to_glsl_storage_type(in_dtype) << " data[];} input_buf;\n";
  os << "layout(set = 0, binding = 1) buffer OutputBuffer {"
     << dtype_to_glsl_storage_type(out_dtype) << " data[];} output_buf;\n";
  if (has_dynamic_i_offset) {
    os << "layout(set = 0, binding = 2) readonly buffer DynamicInputOffset {int64_t data[];} dynamic_i_offset_buf;\n";
  }
  if (has_dynamic_o_offset) {
    os << "layout(set = 0, binding = 3) readonly buffer DynamicOutputOffset {int64_t data[];} dynamic_o_offset_buf;\n";
  }
  os << "\nvoid main() {\n";
  os << "  uint linear_idx = gl_GlobalInvocationID.x;\n";
  os << "  if (linear_idx >= pc.total_elements) {\n";
  os << "    return;\n";
  os << "  }\n";
  os << "  int64_t input_index = pc.input_base;\n";
  os << "  int64_t output_index = pc.output_base;\n";
  if (has_dynamic_i_offset) {
    os << "  input_index += dynamic_i_offset_buf.data[uint(pc.dynamic_i_base)];\n";
  }
  if (has_dynamic_o_offset) {
    os << "  output_index += dynamic_o_offset_buf.data[uint(pc.dynamic_o_base)];\n";
  }
  if (!shape.empty()) {
    os << "  uint remaining = linear_idx;\n";
    for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
      os << "  {\n";
      os << "    uint coord = remaining % " << static_cast<uint32_t>(shape[dim])
         << "u;\n";
      os << "    remaining /= " << static_cast<uint32_t>(shape[dim]) << "u;\n";
      os << "    input_index += int64_t(coord) * int64_t(" << i_strides[dim]
         << ");\n";
      os << "    output_index += int64_t(coord) * int64_t(" << o_strides[dim]
         << ");\n";
      os << "  }\n";
    }
  }
  std::string cast_expr;
  if (needs_bf16_helpers(in_dtype, out_dtype)) {
    cast_expr = bf16_cast_expr(
        "input_buf.data[uint(input_index)]", in_dtype, out_dtype);
  } else if (
      in_dtype == mlx::core::complex64 && out_dtype == mlx::core::float32) {
    cast_expr = "input_buf.data[input_index].x";
  } else if (
      in_dtype == mlx::core::float32 && out_dtype == mlx::core::complex64) {
    cast_expr =
        "vec2(" + std::string("input_buf.data[uint(input_index)]") + ", 0.0)";
  } else {
    cast_expr = cast_expr_for_dtype(
        "input_buf.data[uint(input_index)]", in_dtype, out_dtype);
  }
  os << "  output_buf.data[uint(output_index)] = " << cast_expr << ";\n";
  os << "}\n";
  return os.str();
}

void insert_copy_memory_barrier(vk::CommandBuffer command_buffer) {
  vk::MemoryBarrier barrier;
  barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead |
      vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead |
      vk::AccessFlagBits::eTransferWrite;
  barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead |
      vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead |
      vk::AccessFlagBits::eTransferWrite;

  command_buffer.pipelineBarrier(
      vk::PipelineStageFlagBits::eComputeShader |
          vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eComputeShader |
          vk::PipelineStageFlagBits::eTransfer,
      {},
      {barrier},
      {},
      {});
}

void dispatch_dynamic_offset_kernel(
    const mlx::core::array& indices,
    mlx::core::array& offset,
    const std::vector<int64_t>& stride_terms,
    const mlx::core::Stream& s) {
  if (!is_vulkan_storage_array(indices) || !is_vulkan_storage_array(offset)) {
    throw std::runtime_error(
        "compute_dynamic_offset requires Vulkan-backed arrays.");
  }

  const int64_t indices_base_offset = element_offset(indices);
  std::ostringstream layout_key;
  layout_key << static_cast<int>(indices.dtype().val()) << ':'
             << indices_base_offset << ':';
  append_layout_key(
      layout_key, Strides(stride_terms.begin(), stride_terms.end()));
  const std::string shader_name = "compute_dynamic_offset_" +
      copy_dtype_suffix(indices.dtype()) + "_" +
      std::to_string(std::hash<std::string>{}(layout_key.str()));

  const std::string glsl_source = build_dynamic_offset_shader(
      indices.dtype(), indices_base_offset, stride_terms);

  vulkan::DynamicArrayRef arrays[] = {
      {&indices, 0},
      {&offset, 1},
  };
  vulkan::dispatch_dynamic_compute(
      shader_name, glsl_source, 2, arrays, 1, 1, 1, s);
}

bool dispatch_dynamic_general_copy(
    const mlx::core::array& in,
    mlx::core::array& out,
    const Shape& shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    const mlx::core::Stream& s,
    const std::optional<mlx::core::array>& dynamic_i_offset,
    const std::optional<mlx::core::array>& dynamic_o_offset,
    bool insert_barrier_after_dispatch = false) {
  validate_dynamic_offset_array(dynamic_i_offset);
  validate_dynamic_offset_array(dynamic_o_offset);

  if (!is_vulkan_storage_array(in) || !is_vulkan_storage_array(out)) {
    throw std::runtime_error(
        "Dynamic Vulkan copy requires Vulkan-backed input and output arrays.");
  }

  const size_t total_elements = num_elements(shape);
  if (total_elements == 0) {
    return true;
  }
  if (total_elements > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "Dynamic Vulkan copy does not support tensors with more than 2^32 elements.");
  }
  if (shape.size() > 4) {
    return false;
  }

  const int64_t in_base_offset = element_offset(in);
  const int64_t out_base_offset = element_offset(out);
  const int64_t dynamic_i_base_offset =
      dynamic_i_offset.has_value() ? element_offset(*dynamic_i_offset) : 0;
  const int64_t dynamic_o_base_offset =
      dynamic_o_offset.has_value() ? element_offset(*dynamic_o_offset) : 0;

  std::ostringstream layout_key;
  layout_key << static_cast<int>(in.dtype().val()) << ':'
             << dynamic_i_offset.has_value() << ':'
             << dynamic_o_offset.has_value() << ':';
  append_layout_key(layout_key, shape);
  layout_key << ':';
  append_layout_key(layout_key, i_strides);
  layout_key << ':';
  append_layout_key(layout_key, o_strides);

  const std::string shader_name = "dynamic_general_copy_" +
      copy_dtype_suffix(in.dtype()) + "_" + copy_dtype_suffix(out.dtype()) +
      "_" + std::to_string(std::hash<std::string>{}(layout_key.str()));

  const std::string glsl_source = build_dynamic_general_copy_shader(
      in.dtype(),
      out.dtype(),
      shape,
      i_strides,
      o_strides,
      dynamic_i_offset.has_value(),
      dynamic_o_offset.has_value());

  std::vector<vulkan::DynamicArrayRef> arrays;
  arrays.push_back({&in, 0});
  arrays.push_back({&out, 1});
  if (dynamic_i_offset.has_value()) {
    arrays.push_back({&*dynamic_i_offset, 2});
  }
  if (dynamic_o_offset.has_value()) {
    arrays.push_back({&*dynamic_o_offset, 3});
  }

  struct PushConstants {
    uint32_t total_elements;
    int64_t input_base;
    int64_t output_base;
    int64_t dynamic_i_base;
    int64_t dynamic_o_base;
  };
  constexpr uint32_t kPushConstantSize = sizeof(PushConstants);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      glsl_source,
      static_cast<uint32_t>(arrays.size()),
      arrays.data(),
      kPushConstantSize,
      s);

  PushConstants pc{};
  pc.total_elements = static_cast<uint32_t>(total_elements);
  pc.input_base = in_base_offset + i_offset;
  pc.output_base = out_base_offset + o_offset;
  pc.dynamic_i_base = dynamic_i_base_offset;
  pc.dynamic_o_base = dynamic_o_base_offset;

  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &pc);

  const uint32_t workgroups = std::max<uint32_t>(
      (static_cast<uint32_t>(total_elements) + 255u) / 256u, 1u);
  vkCmdDispatch(dispatch.command_buffer, workgroups, 1, 1);
  if (insert_barrier_after_dispatch) {
    insert_copy_memory_barrier(dispatch.command_buffer);
  }
  vulkan::end_command_recording(s.index);
  return true;
}

bool dispatch_dynamic_complex_scalar_fill(
    const mlx::core::array& in,
    mlx::core::array& out,
    const mlx::core::Stream& s) {
  if (!is_vulkan_storage_array(in) || !is_vulkan_storage_array(out) ||
      in.dtype() != mlx::core::complex64 ||
      out.dtype() != mlx::core::complex64 || in.data_size() != 1 ||
      out.size() == 0) {
    return false;
  }

  const uint64_t in_offset = static_cast<uint64_t>(element_offset(in));
  const uint64_t out_offset = static_cast<uint64_t>(element_offset(out));
  const uint64_t total = static_cast<uint64_t>(out.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  std::ostringstream os;
  os << emit_dynamic_shader_preamble(mlx::core::complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputBuffer {vec2 data[];} input_buf;\n";
  os << "layout(set = 0, binding = 1) buffer OutputBuffer {vec2 data[];} output_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  output_buf.data[idx + pc.out_offset] = input_buf.data[pc.in_offset];\n";
  os << "}\n";

  vulkan::DynamicArrayRef arrays[] = {
      {&in, 0},
      {&out, 1},
  };
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 3;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_fill_c64_scalar", os.str(), 2, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t in_off;
    uint32_t out_off;
    uint32_t total;
  } pc{
      static_cast<uint32_t>(in_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &pc);
  const uint32_t workgroups =
      std::max<uint32_t>((static_cast<uint32_t>(total) + 255u) / 256u, 1u);
  vkCmdDispatch(dispatch.command_buffer, workgroups, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool needs_bf16_helpers(Dtype in_dtype, Dtype out_dtype) {
  return in_dtype == mlx::core::bfloat16 || out_dtype == mlx::core::bfloat16;
}

std::string emit_bf16_conversion_helpers() {
  std::ostringstream os;
  os << "uint fp32_to_bf16(float f) {\n";
  os << "  uint u = floatBitsToUint(f);\n";
  os << "  u = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;\n";
  os << "  return u;\n";
  os << "}\n\n";
  os << "float bf16_to_fp32(uint u) {\n";
  os << "  return uintBitsToFloat(u << 16);\n";
  os << "}\n\n";
  return os.str();
}

std::string
bf16_cast_expr(const std::string& expr, Dtype in_dtype, Dtype out_dtype) {
  const bool in_is_bf16 = in_dtype == mlx::core::bfloat16;
  const bool out_is_bf16 = out_dtype == mlx::core::bfloat16;

  if (in_is_bf16 && out_is_bf16) {
    return expr;
  }

  if (in_is_bf16 && !out_is_bf16) {
    std::string as_float = "bf16_to_fp32(uint(" + expr + "))";
    if (out_dtype == mlx::core::bool_) {
      return "(" + as_float + " != 0.0 ? uint8_t(1) : uint8_t(0))";
    }
    return dtype_to_glsl_storage_type(out_dtype) + "(" + as_float + ")";
  }

  if (!in_is_bf16 && out_is_bf16) {
    std::string intermediate;
    if (in_dtype == mlx::core::bool_) {
      intermediate = "float(" + expr + ")";
    } else {
      intermediate = "float(" + expr + ")";
    }
    return "uint16_t(fp32_to_bf16(" + intermediate + "))";
  }

  return {};
}

std::string build_dynamic_vector_cast_shader(Dtype in_dtype, Dtype out_dtype) {
  std::ostringstream os;
  os << emit_dynamic_shader_preamble(in_dtype, out_dtype, false);

  if (needs_bf16_helpers(in_dtype, out_dtype)) {
    os << emit_bf16_conversion_helpers();
  }

  os << "layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputBuffer {"
     << dtype_to_glsl_storage_type(in_dtype) << " data[];} input_buf;\n";
  os << "layout(set = 0, binding = 1) buffer OutputBuffer {"
     << dtype_to_glsl_storage_type(out_dtype) << " data[];} output_buf;\n\n";
  os << "void main() {\n";
  os << "  uint linear_idx = gl_GlobalInvocationID.x;\n";
  os << "  if (linear_idx >= pc.total_elements) {\n";
  os << "    return;\n";
  os << "  }\n";
  os << "  uint input_index = linear_idx + pc.in_offset;\n";
  os << "  uint output_index = linear_idx + pc.out_offset;\n";

  std::string cast_expr;
  if (needs_bf16_helpers(in_dtype, out_dtype)) {
    cast_expr =
        bf16_cast_expr("input_buf.data[input_index]", in_dtype, out_dtype);
  } else {
    cast_expr =
        cast_expr_for_dtype("input_buf.data[input_index]", in_dtype, out_dtype);
  }

  os << "  output_buf.data[output_index] = " << cast_expr << ";\n";
  os << "}\n";
  return os.str();
}

bool dispatch_dynamic_vector_cast_copy(
    const mlx::core::array& in,
    mlx::core::array& out,
    size_t size,
    int64_t in_offset,
    int64_t out_offset,
    const mlx::core::Stream& s) {
  if (!is_vulkan_storage_array(in) || !is_vulkan_storage_array(out)) {
    return false;
  }
  if (!supports_dynamic_gpu_cast_pair(in.dtype(), out.dtype())) {
    return false;
  }
  if (size == 0) {
    return true;
  }
  if (size > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const int64_t in_base_offset = in_offset;
  const int64_t out_base_offset = out_offset;
  if (in_base_offset < 0 || out_base_offset < 0) {
    return false;
  }
  if (static_cast<uint64_t>(in_base_offset) >
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
      static_cast<uint64_t>(out_base_offset) >
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  const std::string shader_name = "dynamic_vector_cast_copy_" +
      copy_dtype_suffix(in.dtype()) + "_" + copy_dtype_suffix(out.dtype());

  const std::string glsl_source =
      build_dynamic_vector_cast_shader(in.dtype(), out.dtype());

  vulkan::DynamicArrayRef arrays[] = {
      {&in, 0},
      {&out, 1},
  };

  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 3;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, glsl_source, 2, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t in_off;
    uint32_t out_off;
    uint32_t total;
  };
  PushConstants pc{};
  pc.in_off = static_cast<uint32_t>(in_base_offset);
  pc.out_off = static_cast<uint32_t>(out_base_offset);
  pc.total = static_cast<uint32_t>(size);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &pc);

  const uint32_t workgroups =
      std::max<uint32_t>((static_cast<uint32_t>(size) + 255u) / 256u, 1u);
  vkCmdDispatch(dispatch.command_buffer, workgroups, 1, 1);

  vulkan::end_command_recording(s.index);
  return true;
}

mlx::core::array make_copy_view(
    const mlx::core::array& base,
    const Shape& shape,
    const Strides& strides,
    int64_t elem_offset) {
  mlx::core::array view(shape, base.dtype(), nullptr, {});
  const auto [data_size, row_contiguous, col_contiguous] =
      mlx::core::check_contiguity(shape, strides);
  view.copy_shared_buffer(
      base,
      strides,
      {data_size == num_elements(shape), row_contiguous, col_contiguous},
      data_size,
      elem_offset);
  return view;
}

std::tuple<Shape, Strides, Strides> collapse_copy_dims(
    const Shape& shape,
    const Strides& in_strides,
    const Strides& out_strides) {
  auto [collapsed_shape, collapsed_strides] =
      mlx::core::collapse_contiguous_dims(
          shape,
          {in_strides, out_strides},
          std::numeric_limits<uint32_t>::max());
  return {
      std::move(collapsed_shape),
      std::move(collapsed_strides[0]),
      std::move(collapsed_strides[1])};
}

template <typename SrcT, typename DstT>
void host_cast_copy_vector(const void* src, void* dst, size_t size) {
  auto* src_ptr = static_cast<const SrcT*>(src);
  auto* dst_ptr = static_cast<DstT*>(dst);
  for (size_t i = 0; i < size; ++i) {
    dst_ptr[i] = static_cast<DstT>(src_ptr[i]);
  }
}

template <typename SrcT>
void host_cast_copy_dispatch_dst(
    const void* src,
    void* dst,
    size_t size,
    Dtype dst_dtype) {
  switch (dst_dtype) {
    case mlx::core::bool_:
      host_cast_copy_vector<SrcT, bool>(src, dst, size);
      return;
    case mlx::core::uint8:
      host_cast_copy_vector<SrcT, uint8_t>(src, dst, size);
      return;
    case mlx::core::uint16:
      host_cast_copy_vector<SrcT, uint16_t>(src, dst, size);
      return;
    case mlx::core::uint32:
      host_cast_copy_vector<SrcT, uint32_t>(src, dst, size);
      return;
    case mlx::core::uint64:
      host_cast_copy_vector<SrcT, uint64_t>(src, dst, size);
      return;
    case mlx::core::int8:
      host_cast_copy_vector<SrcT, int8_t>(src, dst, size);
      return;
    case mlx::core::int16:
      host_cast_copy_vector<SrcT, int16_t>(src, dst, size);
      return;
    case mlx::core::int32:
      host_cast_copy_vector<SrcT, int32_t>(src, dst, size);
      return;
    case mlx::core::int64:
      host_cast_copy_vector<SrcT, int64_t>(src, dst, size);
      return;
    case mlx::core::float16:
      host_cast_copy_vector<SrcT, mlx::core::float16_t>(src, dst, size);
      return;
    case mlx::core::float32:
      host_cast_copy_vector<SrcT, float>(src, dst, size);
      return;
    case mlx::core::bfloat16:
      host_cast_copy_vector<SrcT, mlx::core::bfloat16_t>(src, dst, size);
      return;
    case mlx::core::complex64:
      host_cast_copy_vector<SrcT, mlx::core::complex64_t>(src, dst, size);
      return;
    case mlx::core::float64:
      throw std::runtime_error("float64 is not supported on the GPU");
  }
}

void host_cast_copy_dispatch_src(
    const void* src,
    Dtype src_dtype,
    void* dst,
    size_t size,
    Dtype dst_dtype) {
  switch (src_dtype) {
    case mlx::core::bool_:
      host_cast_copy_dispatch_dst<bool>(src, dst, size, dst_dtype);
      return;
    case mlx::core::uint8:
      host_cast_copy_dispatch_dst<uint8_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::uint16:
      host_cast_copy_dispatch_dst<uint16_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::uint32:
      host_cast_copy_dispatch_dst<uint32_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::uint64:
      host_cast_copy_dispatch_dst<uint64_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::int8:
      host_cast_copy_dispatch_dst<int8_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::int16:
      host_cast_copy_dispatch_dst<int16_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::int32:
      host_cast_copy_dispatch_dst<int32_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::int64:
      host_cast_copy_dispatch_dst<int64_t>(src, dst, size, dst_dtype);
      return;
    case mlx::core::float16:
      host_cast_copy_dispatch_dst<mlx::core::float16_t>(
          src, dst, size, dst_dtype);
      return;
    case mlx::core::float32:
      host_cast_copy_dispatch_dst<float>(src, dst, size, dst_dtype);
      return;
    case mlx::core::bfloat16:
      host_cast_copy_dispatch_dst<mlx::core::bfloat16_t>(
          src, dst, size, dst_dtype);
      return;
    case mlx::core::complex64:
      host_cast_copy_dispatch_dst<mlx::core::complex64_t>(
          src, dst, size, dst_dtype);
      return;
    case mlx::core::float64:
      throw std::runtime_error("float64 is not supported on the GPU");
  }
}

std::vector<char> make_host_scalar_fill_buffer(
    const array& fill_val,
    Dtype out_dtype,
    size_t out_elements) {
  std::vector<char> scalar_bytes(size_of(out_dtype));
  host_cast_copy_dispatch_src(
      fill_val.data<void>(),
      fill_val.dtype(),
      scalar_bytes.data(),
      1,
      out_dtype);

  std::vector<char> host_fill(out_elements * size_of(out_dtype));
  for (size_t offset = 0; offset < host_fill.size();
       offset += scalar_bytes.size()) {
    std::memcpy(
        host_fill.data() + offset, scalar_bytes.data(), scalar_bytes.size());
  }
  return host_fill;
}

bool trace_copy_fallback_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COPY_FALLBACK");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

bool try_host_vector_cast_copy(
    const mlx::core::array& in,
    mlx::core::array& out,
    size_t size,
    int64_t in_offset,
    int64_t out_offset,
    const mlx::core::Stream& s) {
  const bool in_is_vulkan = mlx::core::vulkan::is_vulkan_buffer(in.buffer());

  if (trace_copy_fallback_enabled() && in_is_vulkan) {
    std::cerr << "[MLX_VULKAN_COPY_FALLBACK] cast-copy fallback triggered: "
              << "in_dtype=" << copy_dtype_suffix(in.dtype()) << " "
              << "out_dtype=" << copy_dtype_suffix(out.dtype()) << " "
              << "size=" << size << " "
              << "in_offset=" << in_offset << " "
              << "out_offset=" << out_offset << std::endl;
  }

  auto* out_buf =
      static_cast<mlx::core::vulkan::VulkanBuffer*>(out.buffer().ptr());

  auto convert_and_store = [&](const void* src_ptr) {
    std::vector<char> host_out(size * size_of(out.dtype()));
    host_cast_copy_dispatch_src(
        src_ptr, in.dtype(), host_out.data(), size, out.dtype());

    if (mlx::core::vulkan::VulkanContext::get().is_unified_memory() &&
        out_buf->mapped_ptr != nullptr) {
      auto* dst_ptr = static_cast<char*>(out_buf->mapped_ptr) +
          out_offset * size_of(out.dtype());
      std::memcpy(dst_ptr, host_out.data(), host_out.size());
      return;
    }

    mlx::core::vulkan::enqueue_owned_staging_upload(
        s,
        host_out.data(),
        host_out.size(),
        out_buf->buffer,
        out_offset * size_of(out.dtype()),
        out.data_shared_ptr());
    mlx::core::vulkan::retain_array_for_stream(s, in);
    mlx::core::vulkan::retain_array_for_stream(s, out);
  };

  if (!in_is_vulkan) {
    auto* src_ptr = static_cast<const char*>(in.data<void>()) +
        in_offset * size_of(in.dtype());
    convert_and_store(src_ptr);
    return true;
  }

  if (dispatch_dynamic_vector_cast_copy(
          in, out, size, in_offset, out_offset, s)) {
    return true;
  }

  std::ostringstream oss;
  oss << "Vulkan cast-copy fallback is not implemented for dtype pair in="
      << static_cast<int>(in.dtype().val())
      << " out=" << static_cast<int>(out.dtype().val());
  throw std::runtime_error(oss.str());
}

std::string copy_dtype_suffix(Dtype dtype) {
  switch (dtype) {
    case mlx::core::float16:
      return "f16";
    case mlx::core::float32:
      return "f32";
    case mlx::core::bfloat16:
      return "bf16";
    case mlx::core::bool_:
      return "bool";
    case mlx::core::uint16:
      return "u16";
    case mlx::core::uint8:
      return "u8";
    case mlx::core::int8:
      return "i8";
    case mlx::core::int16:
      return "i16";
    case mlx::core::int32:
      return "i32";
    case mlx::core::uint32:
      return "u32";
    case mlx::core::uint64:
      return "u64";
    case mlx::core::int64:
      return "i64";
    case mlx::core::complex64:
      return "c64";
    default:
      return {};
  }
}

const char* copy_type_name(mlx::core::CopyType ctype) {
  switch (ctype) {
    case mlx::core::CopyType::Scalar:
      return "scalar";
    case mlx::core::CopyType::Vector:
      return "vector";
    case mlx::core::CopyType::General:
      return "general";
    case mlx::core::CopyType::GeneralGeneral:
      return "general_general";
  }
  return "unknown";
}

template <typename Seq>
std::string seq_to_string(const Seq& seq) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < seq.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << seq[i];
  }
  oss << "]";
  return oss.str();
}

bool is_supported_copy_layout(const mlx::core::array& arr) {
  // For contiguous arrays, we can handle any number of dimensions
  // by treating them as 1D
  if (arr.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  const auto item_size = static_cast<int64_t>(size_of(arr.dtype()));
  if (item_size <= 0 || arr.offset() < 0 || (arr.offset() % item_size) != 0) {
    return false;
  }
  const auto elem_offset = arr.offset() / item_size;
  if (elem_offset >
      static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  for (auto dim : arr.shape()) {
    if (dim < 0 || dim > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  for (auto stride : arr.strides()) {
    if (stride < 0 || stride > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool trace_copy_dispatch_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COPY_DISPATCH");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

bool should_use_transfer_queue_for_copy(size_t copy_bytes) {
  return copy_bytes >= kMinTransferQueueCopyBytes;
}

std::optional<vulkan::StaticShaderId> get_copy_shader_id(
    const mlx::core::array& in,
    mlx::core::array& out) {
  // Fast transpose path: source is column-contiguous and destination is
  // row-contiguous with identical shape/dtype.
  if (in.dtype() == out.dtype() && in.shape() == out.shape() &&
      in.ndim() >= 2 && in.ndim() <= 4 && in.offset() == 0 &&
      out.offset() == 0 && in.flags().col_contiguous &&
      out.flags().row_contiguous) {
    const size_t item_size = size_of(in.dtype());
    if (item_size == 2) {
      return vulkan::StaticShaderId::cpy_transpose_16;
    }
    if (item_size == 4) {
      return vulkan::StaticShaderId::cpy_transpose_32;
    }
  }

  // Only use contig_cpy shaders when both arrays have the same size
  // (for simple contiguous copies, not for scatter/concatenate operations)
  if (in.offset() == 0 && out.offset() == 0 && in.flags().row_contiguous &&
      out.flags().row_contiguous && in.size() == out.size()) {
    if (in.dtype() == mlx::core::float32 &&
        out.dtype() == mlx::core::bfloat16) {
      return vulkan::StaticShaderId::contig_cpy_f32_bf16;
    }
    if (in.dtype() == mlx::core::bfloat16 &&
        out.dtype() == mlx::core::float32) {
      return vulkan::StaticShaderId::contig_cpy_bf16_f32;
    }
    if (in.dtype() == mlx::core::bfloat16 &&
        out.dtype() == mlx::core::float16) {
      return vulkan::StaticShaderId::contig_cpy_bf16_f16;
    }
    if (in.dtype() == mlx::core::bfloat16 &&
        out.dtype() == mlx::core::bfloat16) {
      return vulkan::StaticShaderId::contig_cpy_bf16_bf16;
    }
    if (in.dtype() == mlx::core::float16 &&
        out.dtype() == mlx::core::bfloat16) {
      return vulkan::StaticShaderId::contig_cpy_f16_bf16;
    }
  }

  const bool supported_pair =
      (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float16) ||
      (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::float16) ||
      (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::bfloat16 &&
       out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::bfloat16 &&
       out.dtype() == mlx::core::float16) ||
      (in.dtype() == mlx::core::bfloat16 &&
       out.dtype() == mlx::core::bfloat16) ||
      (in.dtype() == mlx::core::float16 &&
       out.dtype() == mlx::core::bfloat16) ||
      (in.dtype() == mlx::core::float32 &&
       out.dtype() == mlx::core::bfloat16) ||
      (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::int8 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::int16 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int8) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int16) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int32) ||
      (in.dtype() == mlx::core::int8 && out.dtype() == mlx::core::int8) ||
      (in.dtype() == mlx::core::int16 && out.dtype() == mlx::core::int16) ||
      (in.dtype() == mlx::core::uint16 && out.dtype() == mlx::core::uint16) ||
      (in.dtype() == mlx::core::uint8 && out.dtype() == mlx::core::uint8) ||
      (in.dtype() == mlx::core::uint8 && out.dtype() == mlx::core::uint32) ||
      (in.dtype() == mlx::core::uint16 && out.dtype() == mlx::core::uint32) ||
      (in.dtype() == mlx::core::bool_ && out.dtype() == mlx::core::bool_) ||
      (in.dtype() == mlx::core::int64 && out.dtype() == mlx::core::int64) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint32) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint8) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint16) ||
      (in.dtype() == mlx::core::uint64 && out.dtype() == mlx::core::uint64) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::float32) ||
      (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int64) ||
      (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::int64) ||
      (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::complex64);

  if (!supported_pair) {
    return std::nullopt;
  }

  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_f32_f32;
  }
  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::float16) {
    return vulkan::StaticShaderId::cpy_f32_f16;
  }
  if (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::float16) {
    return vulkan::StaticShaderId::cpy_f16_f16;
  }
  if (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_f16_f32;
  }
  if (in.dtype() == mlx::core::bfloat16 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_bf16_f32;
  }
  if (in.dtype() == mlx::core::bfloat16 && out.dtype() == mlx::core::float16) {
    return vulkan::StaticShaderId::cpy_bf16_f16;
  }
  if (in.dtype() == mlx::core::bfloat16 && out.dtype() == mlx::core::bfloat16) {
    return vulkan::StaticShaderId::cpy_bf16_bf16;
  }
  if (in.dtype() == mlx::core::float16 && out.dtype() == mlx::core::bfloat16) {
    return vulkan::StaticShaderId::cpy_f16_bf16;
  }
  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::bfloat16) {
    return vulkan::StaticShaderId::cpy_f32_bf16;
  }
  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::int32) {
    return vulkan::StaticShaderId::cpy_f32_i32;
  }
  if (in.dtype() == mlx::core::int8 && out.dtype() == mlx::core::int32) {
    return vulkan::StaticShaderId::cpy_i8_i32;
  }
  if (in.dtype() == mlx::core::int16 && out.dtype() == mlx::core::int32) {
    return vulkan::StaticShaderId::cpy_i16_i32;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int8) {
    return vulkan::StaticShaderId::cpy_i32_i8;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int16) {
    return vulkan::StaticShaderId::cpy_i32_i16;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_i32_f32;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int32) {
    return vulkan::StaticShaderId::cpy_i32_i32;
  }
  if (in.dtype() == mlx::core::int8 && out.dtype() == mlx::core::int8) {
    return vulkan::StaticShaderId::cpy_i8_i8;
  }
  if (in.dtype() == mlx::core::int16 && out.dtype() == mlx::core::int16) {
    return vulkan::StaticShaderId::cpy_i16_i16;
  }
  if (in.dtype() == mlx::core::uint16 && out.dtype() == mlx::core::uint16) {
    return vulkan::StaticShaderId::cpy_u16_u16;
  }
  if (in.dtype() == mlx::core::uint8 && out.dtype() == mlx::core::uint8) {
    return vulkan::StaticShaderId::cpy_u8_u8;
  }
  if (in.dtype() == mlx::core::uint8 && out.dtype() == mlx::core::uint32) {
    return vulkan::StaticShaderId::cpy_u8_u32;
  }
  if (in.dtype() == mlx::core::uint16 && out.dtype() == mlx::core::uint32) {
    return vulkan::StaticShaderId::cpy_u16_u32;
  }
  if (in.dtype() == mlx::core::bool_ && out.dtype() == mlx::core::bool_) {
    return vulkan::StaticShaderId::cpy_bool_bool;
  }
  if (in.dtype() == mlx::core::int64 && out.dtype() == mlx::core::int64) {
    return vulkan::StaticShaderId::cpy_i64_i64;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint32) {
    return vulkan::StaticShaderId::cpy_u32_u32;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint8) {
    return vulkan::StaticShaderId::cpy_u32_u8;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::uint16) {
    return vulkan::StaticShaderId::cpy_u32_u16;
  }
  if (in.dtype() == mlx::core::uint64 && out.dtype() == mlx::core::uint64) {
    return vulkan::StaticShaderId::cpy_u64_u64;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::float32) {
    return vulkan::StaticShaderId::cpy_u32_f32;
  }
  if (in.dtype() == mlx::core::int32 && out.dtype() == mlx::core::int64) {
    return vulkan::StaticShaderId::cpy_i32_i64;
  }
  if (in.dtype() == mlx::core::uint32 && out.dtype() == mlx::core::int64) {
    return vulkan::StaticShaderId::cpy_u32_i64;
  }
  if (in.dtype() == mlx::core::float32 && out.dtype() == mlx::core::complex64) {
    return vulkan::StaticShaderId::cpy_f32_c64;
  }

  return std::nullopt;
}

} // namespace

namespace mlx::core {

namespace {

std::optional<vulkan::StaticShaderId> fill_shader_id(Dtype dtype) {
  switch (dtype) {
    case float32:
      return vulkan::StaticShaderId::fill_f32;
    case float16:
      return vulkan::StaticShaderId::fill_f16;
    case bfloat16:
      return vulkan::StaticShaderId::fill_bf16;
    case bool_:
    case uint8:
      return vulkan::StaticShaderId::fill_u8;
    default:
      return std::nullopt;
  }
}

std::optional<float> scalar_fill_value_as_float(const array& val) {
  const void* ptr = val.data<void>();
  if (ptr == nullptr) {
    return std::nullopt;
  }
  switch (val.dtype()) {
    case float32:
      return *static_cast<const float*>(ptr);
    case float16:
      return static_cast<float>(*static_cast<const float16_t*>(ptr));
    case bfloat16:
      return static_cast<float>(*static_cast<const bfloat16_t*>(ptr));
    case bool_:
      return *static_cast<const bool*>(ptr) ? 1.0f : 0.0f;
    case uint8:
      return static_cast<float>(*static_cast<const uint8_t*>(ptr));
    default:
      return std::nullopt;
  }
}

} // namespace

void copy_gpu(const array& src, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(src, out, ctype);
  if (donated && src.dtype() == out.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  if (ctype == CopyType::Scalar) {
    fill_gpu(src, out, s);
    return;
  }
  copy_gpu_inplace(src, out, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  if (out.size() == 0) {
    return;
  }

  auto dispatch_shape = data_shape;
  auto dispatch_i_strides = i_strides;
  auto dispatch_o_strides = o_strides;

  if (dispatch_shape.size() > 4) {
    std::tie(dispatch_shape, dispatch_i_strides, dispatch_o_strides) =
        collapse_copy_dims(
            dispatch_shape, dispatch_i_strides, dispatch_o_strides);
  }

  if (dispatch_shape.size() > 4) {
    Shape sub_shape(dispatch_shape.begin() + 1, dispatch_shape.end());
    Strides sub_i_strides(
        dispatch_i_strides.begin() + 1, dispatch_i_strides.end());
    Strides sub_o_strides(
        dispatch_o_strides.begin() + 1, dispatch_o_strides.end());

    for (int64_t i = 0; i < dispatch_shape[0]; ++i) {
      copy_gpu_inplace(
          in,
          out,
          sub_shape,
          sub_i_strides,
          sub_o_strides,
          i_offset + i * dispatch_i_strides[0],
          o_offset + i * dispatch_o_strides[0],
          ctype,
          s,
          dynamic_i_offset,
          dynamic_o_offset);
    }
    return;
  }

  const auto dispatch_elements = num_elements(dispatch_shape);
  std::optional<array> materialized_in;
  const array* source = &in;
  if (in.has_primitive()) {
    materialized_in.emplace(in);
    if (materialized_in->status() == array::Status::unscheduled) {
      materialized_in->eval();
    }
    source = &*materialized_in;
  } else {
    auto data = in.data_shared_ptr();
    if (data == nullptr || data->buffer.ptr() == nullptr) {
      materialized_in.emplace(in);
      materialized_in->wait();
      source = &*materialized_in;
    }
  }

  if (source != &in && source->shape() == in.shape()) {
    dispatch_shape = source->shape();
    dispatch_i_strides = source->strides();
    if (dispatch_shape.size() > 4) {
      std::tie(dispatch_shape, dispatch_i_strides, dispatch_o_strides) =
          collapse_copy_dims(
              dispatch_shape, dispatch_i_strides, dispatch_o_strides);
    }
  }

  if (dynamic_i_offset.has_value() || dynamic_o_offset.has_value()) {
    if (ctype != CopyType::GeneralGeneral) {
      throw std::runtime_error(
          "Dynamic Vulkan copy offsets require GeneralGeneral copy.");
    }
    if (!dispatch_dynamic_general_copy(
            *source,
            out,
            dispatch_shape,
            dispatch_i_strides,
            dispatch_o_strides,
            i_offset,
            o_offset,
            s,
            dynamic_i_offset,
            dynamic_o_offset)) {
      throw std::runtime_error(
          "Dynamic Vulkan copy does not support tensors with rank greater than 4.");
    }
    return;
  }

  const int64_t resolved_i_offset = i_offset;
  const int64_t resolved_o_offset = o_offset;

  auto in_view = make_copy_view(
      *source, dispatch_shape, dispatch_i_strides, resolved_i_offset);
  auto out_view = make_copy_view(
      out, dispatch_shape, dispatch_o_strides, resolved_o_offset);

  std::vector<array> tracking_inputs;
  std::vector<array> tracking_outputs;
  if (vulkan::is_vulkan_buffer(in_view.buffer())) {
    tracking_inputs.push_back(in_view);
  }
  if (vulkan::is_vulkan_buffer(out_view.buffer())) {
    tracking_outputs.push_back(out_view);
  }
  vulkan::ScopedPrimitiveTracking tracking_scope(
      s, std::move(tracking_inputs), std::move(tracking_outputs));

  const bool same_dtype = source->dtype() == out.dtype();
  const bool raw_buffer_copy = same_dtype && ctype == CopyType::Vector;

  const bool full_tensor_copy = data_shape == source->shape() &&
      data_shape == out.shape() && i_strides == source->strides() &&
      o_strides == out.strides() && resolved_i_offset == 0 &&
      resolved_o_offset == 0;

  const auto shader_id = get_copy_shader_id(in_view, out_view);

  const bool shader_copy_type = ctype == CopyType::General ||
      ctype == CopyType::GeneralGeneral ||
      (ctype == CopyType::Vector && !same_dtype);

  const bool shader_copy = shader_copy_type &&
      is_supported_copy_layout(in_view) && is_supported_copy_layout(out_view) &&
      shader_id.has_value();

  const bool staging_scalar_fill = ctype == CopyType::Scalar &&
      resolved_i_offset == 0 && resolved_o_offset == 0 && full_tensor_copy &&
      (out.flags().contiguous || out.flags().row_contiguous ||
       out.flags().col_contiguous);

  if (staging_scalar_fill) {
    const char* scalar_ptr = static_cast<const char*>(source->data<void>());
    const size_t scalar_size = size_of(source->dtype());
    auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());
    if (out_buf->mapped_ptr != nullptr) {
      char* dst = static_cast<char*>(out_buf->mapped_ptr) + out.offset();
      const bool repeated_byte =
          std::all_of(scalar_ptr + 1, scalar_ptr + scalar_size, [&](char c) {
            return c == scalar_ptr[0];
          });
      if (repeated_byte) {
        std::memset(dst, scalar_ptr[0], out.nbytes());
      } else {
        for (size_t offset = 0; offset < out.nbytes(); offset += scalar_size) {
          std::memcpy(dst + offset, scalar_ptr, scalar_size);
        }
      }
    } else {
      const bool scalar_is_zero = std::all_of(
          scalar_ptr, scalar_ptr + scalar_size, [](char c) { return c == 0; });
      const bool can_use_fill_buffer =
          scalar_is_zero && (out.offset() % 4 == 0) && (out.nbytes() % 4 == 0);
      if (can_use_fill_buffer) {
        auto cmd_buffer = vulkan::begin_transfer_command_recording(s.index);
        cmd_buffer.fillBuffer(
            out_buf->buffer,
            static_cast<VkDeviceSize>(out.offset()),
            static_cast<VkDeviceSize>(out.nbytes()),
            0u);
        vulkan::retain_array_for_stream(s, out);
        vulkan::end_transfer_command_recording(s.index);
      } else {
        std::vector<char> host_fill(out.nbytes());
        for (size_t offset = 0; offset < host_fill.size();
             offset += scalar_size) {
          std::memcpy(host_fill.data() + offset, scalar_ptr, scalar_size);
        }
        vulkan::enqueue_owned_staging_upload(
            s,
            host_fill.data(),
            host_fill.size(),
            out_buf->buffer,
            out.offset(),
            out.data_shared_ptr());
        vulkan::retain_array_for_stream(s, *source);
        vulkan::retain_array_for_stream(s, out);
      }
    }
    return;
  }

  const bool is_slice_copy =
      shader_copy_type && shader_id.has_value() && in.size() != out.size();
  const bool dynamic_general_copy = !raw_buffer_copy && !shader_copy &&
      !is_slice_copy &&
      (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) &&
      dispatch_shape.size() <= 4 && is_vulkan_storage_array(in_view) &&
      is_vulkan_storage_array(out_view) &&
      supports_dynamic_gpu_cast_pair(in_view.dtype(), out_view.dtype());
  const bool large_shader_offset = (shader_copy || is_slice_copy) &&
      (has_large_element_offset(in_view) || has_large_element_offset(out_view));

  const bool segmented_buffer_copy = same_dtype && large_shader_offset;

  const bool host_contiguous_copy = in_view.flags().row_contiguous &&
      out_view.flags().row_contiguous && dispatch_elements == in_view.size() &&
      dispatch_elements == out_view.size();

  if (!raw_buffer_copy && !shader_copy && !is_slice_copy &&
      host_contiguous_copy) {
    if (try_host_vector_cast_copy(
            in_view,
            out_view,
            dispatch_elements,
            element_offset(in_view),
            element_offset(out_view),
            s)) {
      return;
    }
  }

  // Fallback for Vector copies with dtype conversion when no static shader
  // exists. try_host_vector_cast_copy dispatches a dynamic GPU cast shader or
  // falls back to CPU conversion for non-Vulkan inputs.
  if (ctype == CopyType::Vector && !same_dtype && !shader_copy) {
    try {
      if (try_host_vector_cast_copy(
              in_view,
              out_view,
              dispatch_elements,
              element_offset(in_view),
              element_offset(out_view),
              s)) {
        return;
      }
    } catch (const std::exception& e) {
      // Fall through to error message
    }
  }

  if (dynamic_general_copy) {
    if (dispatch_dynamic_general_copy(
            in_view,
            out_view,
            dispatch_shape,
            dispatch_i_strides,
            dispatch_o_strides,
            0,
            0,
            s,
            std::nullopt,
            std::nullopt)) {
      return;
    }
  }

  if (!raw_buffer_copy && !shader_copy && !is_slice_copy) {
    std::ostringstream oss;
    oss << "Copy operation failed on Vulkan (unsupported dtype or layout): "
        << "ctype=" << copy_type_name(ctype) << " "
        << "in_dtype=" << static_cast<int>(in.dtype().val()) << " "
        << "out_dtype=" << static_cast<int>(out.dtype().val()) << " "
        << "in_shape=" << seq_to_string(in_view.shape()) << " "
        << "out_shape=" << seq_to_string(out_view.shape()) << " "
        << "in_strides=" << seq_to_string(in_view.strides()) << " "
        << "out_strides=" << seq_to_string(out_view.strides()) << " "
        << "in_offset=" << in_view.offset() << " "
        << "out_offset=" << out_view.offset() << " "
        << "shader_name="
        << (shader_id.has_value() ? vulkan::static_shader_name(*shader_id)
                                  : "<none>");
    throw std::runtime_error(oss.str());
  }

  const bool source_is_vulkan = source->data_shared_ptr() != nullptr &&
      mlx::core::vulkan::is_vulkan_buffer(source->buffer());
  const bool out_is_vulkan = mlx::core::vulkan::is_vulkan_buffer(out.buffer());

  if (!source_is_vulkan && out_is_vulkan && host_contiguous_copy &&
      same_dtype) {
    auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());
    const auto* src_ptr =
        static_cast<const char*>(source->data<void>()) + in_view.offset();
    if (out_buf->mapped_ptr != nullptr) {
      auto* dst_ptr =
          static_cast<char*>(out_buf->mapped_ptr) + out_view.offset();
      std::memcpy(dst_ptr, src_ptr, in_view.nbytes());
    } else {
      vulkan::enqueue_owned_staging_upload(
          s,
          src_ptr,
          in_view.nbytes(),
          out_buf->buffer,
          out_view.offset(),
          out.data_shared_ptr());
      vulkan::retain_array_for_stream(s, *source);
      vulkan::retain_array_for_stream(s, out);
    }
    return;
  }

  const bool same_buffer = source_is_vulkan && out_is_vulkan &&
      source->buffer().ptr() == out.buffer().ptr();
  if (segmented_buffer_copy && same_buffer) {
    array staged(out_view.shape(), out_view.dtype(), nullptr, {});
    staged.set_data(mlx::core::allocator::malloc(staged.nbytes()));

    const auto stage_in_type =
        has_row_contiguous_strides(in_view) && staged.flags().row_contiguous
        ? CopyType::Vector
        : CopyType::GeneralGeneral;
    copy_gpu_inplace(in_view, staged, stage_in_type, s);

    const auto stage_out_type = has_row_contiguous_strides(staged) &&
            has_row_contiguous_strides(out_view)
        ? CopyType::Vector
        : CopyType::GeneralGeneral;
    copy_gpu_inplace(staged, out_view, stage_out_type, s);
    return;
  }

  const size_t copy_bytes =
      static_cast<size_t>(dispatch_elements) * size_of(out.dtype());
  // Small copies, especially decode-time KV cache updates, are latency
  // sensitive and do better when they stay in the current compute submission.
  const bool use_transfer_queue = (raw_buffer_copy || segmented_buffer_copy) &&
      should_use_transfer_queue_for_copy(copy_bytes);
  vk::CommandBuffer cmd_buffer = use_transfer_queue
      ? vulkan::begin_transfer_command_recording(s.index)
      : vulkan::begin_command_recording(s.index);

  // Get buffer handles
  auto* in_buf = static_cast<vulkan::VulkanBuffer*>(
      const_cast<void*>(static_cast<const void*>(source->buffer().ptr())));
  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());

  if (raw_buffer_copy) {
    // Simple contiguous memory copy using Vulkan command buffer
    VkBufferCopy copy_region{};
    copy_region.srcOffset = static_cast<VkDeviceSize>(in_view.offset());
    copy_region.dstOffset = static_cast<VkDeviceSize>(out_view.offset());
    copy_region.size =
        static_cast<VkDeviceSize>(dispatch_elements * size_of(out.dtype()));

    cmd_buffer.copyBuffer(in_buf->buffer, out_buf->buffer, {copy_region});

    vulkan::retain_array_for_stream(s, *source);
    vulkan::retain_array_for_stream(s, out);

  } else if (segmented_buffer_copy) {
    const auto copy_regions = make_strided_copy_regions(in_view, out_view);
    if (copy_regions.empty()) {
      if (use_transfer_queue) {
        vulkan::end_transfer_command_recording(s.index);
      } else {
        vulkan::end_command_recording(s.index);
      }
      throw std::runtime_error(
          "Copy operation failed on Vulkan: unsupported large-offset strided copy.");
    }

    std::vector<vk::BufferCopy> cpp_copy_regions(
        copy_regions.begin(), copy_regions.end());
    cmd_buffer.copyBuffer(in_buf->buffer, out_buf->buffer, cpp_copy_regions);

    vulkan::retain_array_for_stream(s, in_view);
    vulkan::retain_array_for_stream(s, out_view);
  } else if (shader_copy || is_slice_copy) {
    if (trace_copy_dispatch_enabled() &&
        (*shader_id == vulkan::StaticShaderId::cpy_bf16_f32 ||
         *shader_id == vulkan::StaticShaderId::cpy_bf16_bf16)) {
      std::cerr << "[vulkan-copy] shader="
                << vulkan::static_shader_name(*shader_id)
                << " ctype=" << copy_type_name(ctype)
                << " in_shape=" << seq_to_string(in_view.shape())
                << " out_shape=" << seq_to_string(out_view.shape())
                << " in_offset=" << in_view.offset()
                << " out_offset=" << out_view.offset()
                << " in_strides=" << seq_to_string(in_view.strides())
                << " out_strides=" << seq_to_string(out_view.strides()) << "\n";
    }
    try {
      if (dispatch_shape.size() > 4) {
        throw std::runtime_error(
            "Copy operation failed on Vulkan: >4D non-contiguous arrays not supported");
      }

      if (large_shader_offset || is_slice_copy) {
        const bool copied = dispatch_dynamic_general_copy(
            in_view,
            out_view,
            dispatch_shape,
            dispatch_i_strides,
            dispatch_o_strides,
            0,
            0,
            s,
            std::nullopt,
            std::nullopt,
            is_slice_copy && same_buffer);
        if (!copied) {
          throw std::runtime_error(
              "Large-offset shader copy does not support tensors with rank greater than 4.");
        }
        return;
      }

      vulkan::dispatch_unary_op(in_view, out_view, *shader_id, cmd_buffer, s);
    } catch (const std::runtime_error& e) {
      if (use_transfer_queue) {
        vulkan::end_transfer_command_recording(s.index);
      } else {
        vulkan::end_command_recording(s.index);
      }
      throw std::runtime_error(
          std::string("Copy operation failed on Vulkan: ") + e.what());
    }
  } else {
    throw std::runtime_error("Unsupported Vulkan copy type.");
  }

  if (use_transfer_queue) {
    vulkan::end_transfer_command_recording(s.index);
  } else {
    vulkan::end_command_recording(s.index);
  }
}

// Note: The simpler overload copy_gpu_inplace(in, out, ctype, s) is defined in
// mlx/backend/gpu/copy.cpp and calls the complex version implemented above.

void fill_gpu(const array& val, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }

  array fill_val = val;
  if (fill_val.has_primitive()) {
    fill_val.eval();
  } else {
    auto data = fill_val.data_shared_ptr();
    if (data == nullptr || data->buffer.ptr() == nullptr) {
      fill_val.wait();
    }
  }

  out.set_data(allocator::malloc(out.nbytes()));

  if (dispatch_dynamic_complex_scalar_fill(fill_val, out, s)) {
    return;
  }

  if (auto shader_id = fill_shader_id(out.dtype()); shader_id.has_value()) {
    if (auto fill_value = scalar_fill_value_as_float(fill_val);
        fill_value.has_value()) {
      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_fill_op(out, *shader_id, command_buffer, s, *fill_value);
      vulkan::retain_array_for_stream(s, out);
      vulkan::end_command_recording(s.index);
      return;
    }
  }

  // CPU-side scalar replication must observe the final scalar value, even when
  // it was produced by a prior GPU op.
  fill_val.wait();

  // For unified memory, we can directly fill on CPU
  auto* out_buf = static_cast<vulkan::VulkanBuffer*>(out.buffer().ptr());
  const char* val_ptr = static_cast<const char*>(fill_val.data<void>());
  size_t val_size = size_of(fill_val.dtype());
  complex64_t complex_scalar_value{};
  if (fill_val.size() == 1 && fill_val.dtype() == complex64) {
    complex_scalar_value = fill_val.item<complex64_t>();
    val_ptr = reinterpret_cast<const char*>(&complex_scalar_value);
    val_size = sizeof(complex_scalar_value);
  }

  const bool same_dtype_scalar_bytes = fill_val.dtype() == out.dtype();
  std::vector<char> converted_scalar_bytes;
  if (!same_dtype_scalar_bytes) {
    converted_scalar_bytes =
        make_host_scalar_fill_buffer(fill_val, out.dtype(), 1);
    val_ptr = converted_scalar_bytes.data();
    val_size = converted_scalar_bytes.size();
  }

  if (vulkan::VulkanContext::get().is_unified_memory()) {
    // Direct CPU fill for unified memory
    char* dst_ptr = static_cast<char*>(out_buf->mapped_ptr);
    size_t out_size = out.nbytes();

    const bool repeated_byte =
        std::all_of(val_ptr + 1, val_ptr + val_size, [&](char c) {
          return c == val_ptr[0];
        });
    if (repeated_byte) {
      std::memset(dst_ptr, val_ptr[0], out_size);
    } else {
      for (size_t i = 0; i < out_size; i += val_size) {
        std::memcpy(dst_ptr + i, val_ptr, val_size);
      }
    }
  } else {
    std::vector<char> host_fill(out.nbytes());
    for (size_t i = 0; i < host_fill.size(); i += val_size) {
      std::memcpy(host_fill.data() + i, val_ptr, val_size);
    }
    vulkan::enqueue_owned_staging_upload(
        s,
        host_fill.data(),
        host_fill.size(),
        out_buf->buffer,
        out.offset(),
        out.data_shared_ptr());
    vulkan::retain_array_for_stream(s, fill_val);
    vulkan::retain_array_for_stream(s, out);
  }
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (const auto& in : inputs) {
    sizes.push_back(in.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc(out.nbytes()));

  if (axis == 0 && out.flags().row_contiguous) {
    bool supported_axis0_concat = true;
    for (const auto& in : inputs) {
      if (in.dtype() != out.dtype()) {
        supported_axis0_concat = false;
        break;
      }
    }
    if (supported_axis0_concat) {
      std::vector<array> prepared_inputs;
      prepared_inputs.reserve(inputs.size());
      for (auto in : inputs) {
        if (!has_row_contiguous_strides(in) || !has_vulkan_storage(in)) {
          in = contiguous_copy_gpu(in, s);
        }
        prepared_inputs.push_back(in);
      }

      auto* out_buf =
          static_cast<mlx::core::vulkan::VulkanBuffer*>(out.buffer().ptr());
      auto command_buffer = vulkan::begin_command_recording(s.index);
      size_t dst_offset = 0;
      for (const auto& in : prepared_inputs) {
        auto* in_buf = static_cast<mlx::core::vulkan::VulkanBuffer*>(
            const_cast<void*>(static_cast<const void*>(in.buffer().ptr())));
        VkBufferCopy copy_region{};
        copy_region.srcOffset = static_cast<VkDeviceSize>(in.offset());
        copy_region.dstOffset = static_cast<VkDeviceSize>(dst_offset);
        copy_region.size = static_cast<VkDeviceSize>(in.nbytes());
        command_buffer.copyBuffer(
            in_buf->buffer, out_buf->buffer, {copy_region});
        dst_offset += in.nbytes();
        vulkan::retain_array_for_stream(s, in);
      }
      vulkan::retain_array_for_stream(s, out);
      vulkan::end_command_recording(s.index);
      return;
    }
  }

  if (out.flags().row_contiguous) {
    bool supported_contiguous_concat = true;
    std::vector<array> prepared_inputs;
    for (const auto& in : inputs) {
      if (in.dtype() != out.dtype()) {
        supported_contiguous_concat = false;
        break;
      }
    }
    if (supported_contiguous_concat) {
      prepared_inputs.reserve(inputs.size());
      for (auto in : inputs) {
        if (!has_row_contiguous_strides(in) || !has_vulkan_storage(in)) {
          in = contiguous_copy_gpu(in, s);
        }
        if (!has_row_contiguous_strides(in) || !has_vulkan_storage(in)) {
          supported_contiguous_concat = false;
          break;
        }
        prepared_inputs.push_back(in);
      }
    }
    if (supported_contiguous_concat) {
      size_t size_pre = 1;
      for (int dim = 0; dim < axis; ++dim) {
        size_pre *= out.shape(dim);
      }
      size_t size_post = 1;
      for (int dim = axis + 1; dim < out.ndim(); ++dim) {
        size_post *= out.shape(dim);
      }

      const size_t itemsize = size_of(out.dtype());
      const size_t out_axis_size = out.shape(axis);

      auto* out_buf =
          static_cast<mlx::core::vulkan::VulkanBuffer*>(out.buffer().ptr());
      auto command_buffer = vulkan::begin_command_recording(s.index);
      for (int i = 0; i < prepared_inputs.size(); ++i) {
        const auto& in = prepared_inputs[i];
        const size_t in_axis_size = in.shape(axis);
        const size_t elements = in_axis_size * size_post;
        if (elements == 0) {
          continue;
        }
        auto* in_buf = static_cast<mlx::core::vulkan::VulkanBuffer*>(
            const_cast<void*>(static_cast<const void*>(in.buffer().ptr())));
        std::vector<vk::BufferCopy> copy_regions;
        copy_regions.reserve(size_pre);
        for (size_t prefix = 0; prefix < size_pre; ++prefix) {
          vk::BufferCopy region{};
          region.srcOffset = static_cast<VkDeviceSize>(
              in.offset() + prefix * in_axis_size * size_post * itemsize);
          region.dstOffset = static_cast<VkDeviceSize>(
              ((prefix * out_axis_size + sizes[i]) * size_post) * itemsize);
          region.size = static_cast<VkDeviceSize>(elements * itemsize);
          copy_regions.push_back(region);
        }
        command_buffer.copyBuffer(
            in_buf->buffer, out_buf->buffer, copy_regions);
        vulkan::retain_array_for_stream(s, in);
      }
      vulkan::retain_array_for_stream(s, out);
      vulkan::end_command_recording(s.index);
      return;
    }
  }

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;

  for (int i = 0; i < inputs.size(); ++i) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, s);
  }
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  if (indices.size() != axes.size()) {
    throw std::runtime_error(
        "compute_dynamic_offset expected indices.size() == axes.size().");
  }

  switch (indices.dtype()) {
    case mlx::core::int8:
    case mlx::core::uint8:
    case mlx::core::int16:
    case mlx::core::uint16:
    case mlx::core::int32:
    case mlx::core::uint32:
    case mlx::core::int64:
    case mlx::core::uint64:
      break;
    default:
      throw std::runtime_error(
          "compute_dynamic_offset requires integer index types.");
  }

  std::vector<int64_t> stride_terms;
  stride_terms.reserve(axes.size());
  for (size_t i = 0; i < axes.size(); ++i) {
    int axis = axes[i];
    if (axis < 0) {
      axis += static_cast<int>(strides.size());
    }
    if (axis < 0 || axis >= static_cast<int>(strides.size())) {
      throw std::out_of_range("compute_dynamic_offset axis out of range.");
    }
    stride_terms.push_back(strides[axis]);
  }

  array offset({1}, int64, nullptr, {});
  offset.set_data(allocator::malloc(offset.itemsize()));
  dispatch_dynamic_offset_kernel(indices, offset, stride_terms, s);
  return offset;
}

} // namespace mlx::core
