// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/utils.h"

#include <algorithm>
#include <sstream>

namespace mlx::core {

namespace {

array collapse_binary_leading_dims(const array& arr, Stream s) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  return flatten_in_eval(arr, 0, arr.ndim() - 4, s);
}

bool is_vulkan_integer_dtype(Dtype dtype) {
  switch (dtype) {
    case int8:
    case int16:
    case int32:
    case int64:
    case uint8:
    case uint16:
    case uint32:
    case uint64:
      return true;
    default:
      return false;
  }
}

bool is_vulkan_div_cast_dtype(Dtype dtype) {
  switch (dtype) {
    case bool_:
    case int32:
    case int64:
    case uint32:
    case uint64:
    case float16:
    case float32:
    case bfloat16:
      return true;
    default:
      return false;
  }
}

bool is_vulkan_compare_dtype(Dtype dtype) {
  switch (dtype) {
    case float16:
    case float32:
    case bfloat16:
    case int32:
    case int64:
    case uint32:
    case uint64:
      return true;
    default:
      return false;
  }
}

bool has_vulkan_buffer(const array& arr) {
  auto data = arr.data_shared_ptr();
  return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
}

void ensure_materialized_scalar_input(array& arr) {
  if (arr.data_size() != 1 || !arr.has_primitive()) {
    return;
  }
  auto data = arr.data_shared_ptr();
  if (arr.status() != array::Status::unscheduled &&
      (data == nullptr || data->buffer.ptr() == nullptr)) {
    arr.set_status(array::Status::unscheduled);
  }
  arr.eval();
}

bool ensure_vulkan_buffer(array& arr, Stream s) {
  if (has_vulkan_buffer(arr)) {
    return true;
  }

  if (arr.has_primitive()) {
    if (auto& p = arr.primitive();
        typeid(p) == typeid(Broadcast) || typeid(p) == typeid(BroadcastAxes)) {
      arr.eval();
      return has_vulkan_buffer(arr);
    }
    arr = contiguous_copy_gpu(arr, s);
    return has_vulkan_buffer(arr);
  }

  if (!arr.has_primitive()) {
    arr.wait();
  }

  if (has_vulkan_buffer(arr)) {
    return true;
  }

  auto data = arr.data_shared_ptr();
  if (data == nullptr || data->buffer.ptr() == nullptr) {
    return false;
  }

  arr = contiguous_copy_gpu(arr, s);
  return has_vulkan_buffer(arr);
}

bool is_same_complex_add(const array& a, const array& b, const array& out) {
  return a.dtype() == complex64 && b.dtype() == complex64 &&
      out.dtype() == complex64;
}

bool is_same_complex_sub(const array& a, const array& b, const array& out) {
  return a.dtype() == complex64 && b.dtype() == complex64 &&
      out.dtype() == complex64;
}

bool is_complex_float_mul(const array& a, const array& b, const array& out) {
  return out.dtype() == complex64 &&
      ((a.dtype() == complex64 && b.dtype() == float32) ||
       (a.dtype() == float32 && b.dtype() == complex64));
}

bool is_same_complex_mul(const array& a, const array& b, const array& out) {
  return a.dtype() == complex64 && b.dtype() == complex64 &&
      out.dtype() == complex64;
}

bool is_complex_scalar_mul(const array& a, const array& b, const array& out) {
  return is_same_complex_mul(a, b, out) &&
      ((a.data_size() == 1) != (b.data_size() == 1));
}

bool is_same_complex_div(const array& a, const array& b, const array& out) {
  return a.dtype() == complex64 && b.dtype() == complex64 &&
      out.dtype() == complex64;
}

bool is_same_complex_max(const array& a, const array& b, const array& out) {
  return a.dtype() == complex64 && b.dtype() == complex64 &&
      out.dtype() == complex64;
}

bool is_same_complex_min(const array& a, const array& b, const array& out) {
  return a.dtype() == complex64 && b.dtype() == complex64 &&
      out.dtype() == complex64;
}

std::string build_complex_add_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {vec2 data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {vec2 data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  out_buf.data[idx + pc.out_offset] = "
        "a_buf.data[idx + pc.a_offset] + b_buf.data[idx + pc.b_offset];\n";
  os << "}\n";
  return os.str();
}

struct ComplexBinaryPushConstants {
  uint32_t a_offset;
  uint32_t b_offset;
  uint32_t out_offset;
  uint32_t total_elements;
};

struct ComplexScalarMulPushConstants {
  uint32_t in_offset;
  uint32_t scalar_offset;
  uint32_t out_offset;
  uint32_t total_elements;
};

bool materialize_complex_binary_input(array& in, const array& out, Stream s) {
  if (in.shape() == out.shape()) {
    if (in.data_size() == 1 && in.size() != 1) {
      array materialized(out.shape(), in.dtype(), nullptr, {});
      copy_gpu(in, materialized, CopyType::General, s);
      in = materialized;
      return true;
    }
    return ensure_vulkan_buffer(in, s);
  }
  if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
    return false;
  }
  if (!ensure_vulkan_buffer(in, s)) {
    return false;
  }
  if (in.data_size() == 1) {
    array materialized(out.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, materialized, CopyType::Scalar, s);
    in = materialized;
    return true;
  }
  array view(out.shape(), in.dtype(), nullptr, {});
  broadcast(in, view);
  in = view;
  return true;
}

bool prepare_complex_binary_dispatch(
    array& a,
    array& b,
    array& out,
    array& out_work,
    bool& staged_output,
    Stream s) {
  if (!materialize_complex_binary_input(a, out, s) ||
      !materialize_complex_binary_input(b, out, s)) {
    return false;
  }

  if (!is_supported_elementwise_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }
  staged_output = !is_supported_elementwise_layout(out);
  out_work = staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  if (!is_supported_elementwise_layout(out_work)) {
    return false;
  }

  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out_work, bopt);
  return true;
}

bool make_complex_binary_push_constants(
    const array& a,
    const array& b,
    const array& out,
    ComplexBinaryPushConstants& pc) {
  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out.offset() / size_of(out.dtype()));
  const auto total = static_cast<uint64_t>(out.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  pc = {
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };
  return true;
}

bool dispatch_complex_binary_shader(
    array& a,
    array& b,
    array& out,
    Stream s,
    const char* name,
    const std::string& source) {
  array out_work(out.shape(), out.dtype(), nullptr, {});
  bool staged_output = false;
  if (!prepare_complex_binary_dispatch(a, b, out, out_work, staged_output, s)) {
    return false;
  }

  ComplexBinaryPushConstants pc{};
  if (!make_complex_binary_push_constants(a, b, out_work, pc)) {
    return false;
  }

  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(ComplexBinaryPushConstants);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      name, source, 3, arrays, kPushConstantSize, s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total_elements + 255) / 256, 1, 1);
  vulkan::end_command_recording(s.index);
  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool materialize_complex_vector_input(array& in, const array& out, Stream s) {
  if (in.dtype() != complex64 || in.data_size() == 1) {
    return false;
  }
  if (!materialize_complex_binary_input(in, out, s)) {
    return false;
  }
  if (!is_supported_elementwise_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }
  return is_supported_elementwise_layout(in);
}

bool try_eval_complex_add_vulkan(array& a, array& b, array& out, Stream s) {
  return dispatch_complex_binary_shader(
      a, b, out, s, "dynamic_add_c64_c64_c64", build_complex_add_shader());
}

bool try_eval_complex_mul_vulkan(array& a, array& b, array& out, Stream s);

std::string build_complex_sub_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {vec2 data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {vec2 data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  out_buf.data[idx + pc.out_offset] = "
        "a_buf.data[idx + pc.a_offset] - b_buf.data[idx + pc.b_offset];\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_sub_vulkan(array& a, array& b, array& out, Stream s) {
  return dispatch_complex_binary_shader(
      a, b, out, s, "dynamic_sub_c64_c64_c64", build_complex_sub_shader());
}

std::string build_complex_max_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {vec2 data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {vec2 data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  vec2 x = a_buf.data[idx + pc.a_offset];\n";
  os << "  vec2 y = b_buf.data[idx + pc.b_offset];\n";
  os << "  bool x_nan = isnan(x.x) || isnan(x.y);\n";
  os << "  bool x_gt = (x.x > y.x) || (x.x == y.x && x.y > y.y);\n";
  os << "  out_buf.data[idx + pc.out_offset] = (x_nan || x_gt) ? x : y;\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_max_vulkan(array& a, array& b, array& out, Stream s) {
  return dispatch_complex_binary_shader(
      a, b, out, s, "dynamic_max_c64_c64_c64", build_complex_max_shader());
}

std::string build_complex_min_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {vec2 data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {vec2 data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  vec2 x = a_buf.data[idx + pc.a_offset];\n";
  os << "  vec2 y = b_buf.data[idx + pc.b_offset];\n";
  os << "  bool x_nan = isnan(x.x) || isnan(x.y);\n";
  os << "  bool x_lt = (x.x < y.x) || (x.x == y.x && x.y < y.y);\n";
  os << "  out_buf.data[idx + pc.out_offset] = (x_nan || x_lt) ? x : y;\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_min_vulkan(array& a, array& b, array& out, Stream s) {
  return dispatch_complex_binary_shader(
      a, b, out, s, "dynamic_min_c64_c64_c64", build_complex_min_shader());
}

std::string build_complex_scalar_mul_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint in_offset; uint scalar_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer Input {vec2 data[];} in_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer ScalarInput {vec2 data[];} scalar_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  vec2 x = in_buf.data[idx + pc.in_offset];\n";
  os << "  vec2 y = scalar_buf.data[pc.scalar_offset];\n";
  os << "  precise float real = x.x * y.x - x.y * y.y;\n";
  os << "  precise float imag = x.x * y.y + x.y * y.x;\n";
  os << "  out_buf.data[idx + pc.out_offset] = vec2(real, imag);\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_scalar_mul_vulkan(
    array& a,
    array& b,
    array& out,
    Stream s) {
  array* vector_in = nullptr;
  array* scalar_in = nullptr;
  if (a.data_size() == 1 && b.data_size() != 1) {
    scalar_in = &a;
    vector_in = &b;
  } else if (b.data_size() == 1 && a.data_size() != 1) {
    scalar_in = &b;
    vector_in = &a;
  } else {
    return false;
  }

  if (!materialize_complex_vector_input(*vector_in, out, s)) {
    return false;
  }

  array scalar_value = *scalar_in;
  if (scalar_value.shape() != Shape{} && scalar_value.data_size() == 1) {
    array scalar_base(Shape{}, scalar_value.dtype(), nullptr, {});
    scalar_base.copy_shared_buffer(
        scalar_value, Strides{}, {true, true, true}, 1);
    scalar_value = scalar_base;
  }
  if (!ensure_vulkan_buffer(scalar_value, s)) {
    return false;
  }
  if (!materialize_complex_vector_input(scalar_value, scalar_value, s) &&
      !is_supported_elementwise_layout(scalar_value)) {
    scalar_value = contiguous_copy_gpu(scalar_value, s);
  }

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  if (!is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (!staged_output) {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  } else {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  }

  const auto in_offset =
      static_cast<uint64_t>(vector_in->offset() / size_of(vector_in->dtype()));
  const auto scalar_offset = static_cast<uint64_t>(
      scalar_value.offset() / size_of(scalar_value.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      scalar_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  ComplexScalarMulPushConstants pc{};
  pc.in_offset = static_cast<uint32_t>(in_offset);
  pc.scalar_offset = static_cast<uint32_t>(scalar_offset);
  pc.out_offset = static_cast<uint32_t>(out_offset);
  pc.total_elements = static_cast<uint32_t>(total);

  vulkan::DynamicArrayRef arrays[] = {
      {vector_in, 0}, {&scalar_value, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(ComplexScalarMulPushConstants);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_mul_c64_scalar_c64",
      build_complex_scalar_mul_shader(),
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
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total_elements + 255) / 256, 1, 1);
  vulkan::end_command_recording(s.index);
  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

std::string build_complex_float_mul_shader(bool complex_lhs) {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {";
  os << (complex_lhs ? "vec2" : "float") << " data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {";
  os << (complex_lhs ? "float" : "vec2") << " data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  out_buf.data[idx + pc.out_offset] = "
        "a_buf.data[idx + pc.a_offset] * b_buf.data[idx + pc.b_offset];\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_float_mul_vulkan(
    array& a,
    array& b,
    array& out,
    Stream s) {
  auto materialize_float_operand = [&](array& in) -> bool {
    if (in.dtype() != float32) {
      return true;
    }

    if (in.shape() == out.shape() && in.data_size() != 1) {
      return true;
    }

    if (in.shape() != out.shape() &&
        broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }

    array materialized(out.shape(), float32, nullptr, {});
    if (in.data_size() == 1) {
      copy_gpu(in, materialized, CopyType::Scalar, s);
    } else {
      array view(out.shape(), float32, nullptr, {});
      broadcast(in, view);
      copy_gpu(view, materialized, CopyType::General, s);
    }
    in = materialized;
    return true;
  };

  if (!materialize_float_operand(a) || !materialize_float_operand(b)) {
    return false;
  }

  const bool complex_lhs = a.dtype() == complex64;
  return dispatch_complex_binary_shader(
      a,
      b,
      out,
      s,
      complex_lhs ? "dynamic_mul_c64_f32_c64" : "dynamic_mul_f32_c64_c64",
      build_complex_float_mul_shader(complex_lhs));
}

std::string build_complex_mul_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {vec2 data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {vec2 data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  vec2 x = a_buf.data[idx + pc.a_offset];\n";
  os << "  vec2 y = b_buf.data[idx + pc.b_offset];\n";
  os << "  precise float real = x.x * y.x - x.y * y.y;\n";
  os << "  precise float imag = x.x * y.y + x.y * y.x;\n";
  os << "  out_buf.data[idx + pc.out_offset] = vec2(real, imag);\n";
  os << "}\n";
  return os.str();
}

std::string build_complex_div_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {vec2 data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {vec2 data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  vec2 x = a_buf.data[idx + pc.a_offset];\n";
  os << "  vec2 y = b_buf.data[idx + pc.b_offset];\n";
  os << "  precise float denom = y.x * y.x + y.y * y.y;\n";
  os << "  precise float real_num = x.x * y.x + x.y * y.y;\n";
  os << "  precise float imag_num = x.y * y.x - x.x * y.y;\n";
  os << "  precise float real = real_num / denom;\n";
  os << "  precise float imag = imag_num / denom;\n";
  os << "  real += (real_num - real * denom) / denom;\n";
  os << "  imag += (imag_num - imag * denom) / denom;\n";
  os << "  out_buf.data[idx + pc.out_offset] = vec2(real, imag);\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_mul_vulkan(array& a, array& b, array& out, Stream s) {
  return dispatch_complex_binary_shader(
      a, b, out, s, "dynamic_mul_c64_c64_c64", build_complex_mul_shader());
}

bool try_eval_complex_div_vulkan(array& a, array& b, array& out, Stream s) {
  return dispatch_complex_binary_shader(
      a, b, out, s, "dynamic_div_c64_c64_c64", build_complex_div_shader());
}

std::string build_divmod_shader(Dtype dtype) {
  std::ostringstream os;
  const auto type_name = vulkan::dtype_to_glsl_storage_type(dtype);
  os << vulkan::emit_dynamic_shader_preamble(
      dtype, dtype, dtype == int64 || dtype == uint64);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint q_offset; uint r_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {"
     << type_name << " data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {"
     << type_name << " data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Quotient {"
     << type_name << " data[];} q_buf;\n";
  os << "layout(set = 0, binding = 3) buffer Remainder {"
     << type_name << " data[];} r_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  " << type_name << " x = a_buf.data[idx + pc.a_offset];\n";
  os << "  " << type_name << " y = b_buf.data[idx + pc.b_offset];\n";
  if (dtype == float32) {
    os << "  " << type_name << " q = trunc(x / y);\n";
    os << "  q_buf.data[idx + pc.q_offset] = q;\n";
    os << "  r_buf.data[idx + pc.r_offset] = x - y * q;\n";
  } else {
    os << "  " << type_name << " q = x / y;\n";
    os << "  q_buf.data[idx + pc.q_offset] = q;\n";
    os << "  r_buf.data[idx + pc.r_offset] = x % y;\n";
  }
  os << "}\n";
  return os.str();
}

bool try_eval_divmod_vulkan(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    Stream s) {
  if (inputs.size() != 2 || outputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  auto& quotient = outputs[0];
  auto& remainder = outputs[1];
  const bool float_case =
      a.dtype() == float32 && b.dtype() == float32 &&
      quotient.dtype() == float32 && remainder.dtype() == float32;
  const bool int_case = a.dtype() == b.dtype() && a.dtype() == quotient.dtype() &&
      a.dtype() == remainder.dtype() &&
      (a.dtype() == int32 || a.dtype() == uint32 || a.dtype() == int64 ||
       a.dtype() == uint64);
  if ((!float_case && !int_case) ||
      quotient.shape() != remainder.shape()) {
    return false;
  }

  auto ensure_binary_input = [&](array& in) {
    if (in.data_size() == 1) {
      if (in.has_primitive()) {
        ensure_materialized_scalar_input(in);
      }
      return true;
    }
    return ensure_vulkan_buffer(in, s);
  };

  if (!ensure_binary_input(a) || !ensure_binary_input(b)) {
    return false;
  }

  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == quotient.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), quotient.shape()) != quotient.shape()) {
      return false;
    }
    array view(quotient.shape(), in.dtype(), nullptr, {});
    broadcast(in, view);
    in = view;
    return true;
  };

  if (!materialize_broadcast_input(a) || !materialize_broadcast_input(b)) {
    return false;
  }
  if (!is_supported_elementwise_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }

  const bool staged_quotient = !is_supported_elementwise_layout(quotient);
  const bool staged_remainder = !is_supported_elementwise_layout(remainder);
  array quotient_work = staged_quotient
      ? array(quotient.shape(), quotient.dtype(), nullptr, {})
      : quotient;
  array remainder_work = staged_remainder
      ? array(remainder.shape(), remainder.dtype(), nullptr, {})
      : remainder;
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, quotient_work, bopt);
  set_binary_op_output_data(a, b, remainder_work, bopt);
  if (!is_supported_elementwise_layout(quotient_work) ||
      !is_supported_elementwise_layout(remainder_work)) {
    return false;
  }

  if (quotient_work.size() == 0) {
    if (staged_quotient) {
      copy_gpu(quotient_work, quotient, CopyType::General, s);
    }
    if (staged_remainder) {
      copy_gpu(remainder_work, remainder, CopyType::General, s);
    }
    return true;
  }

  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto q_offset = static_cast<uint64_t>(
      quotient_work.offset() / size_of(quotient_work.dtype()));
  const auto r_offset = static_cast<uint64_t>(
      remainder_work.offset() / size_of(remainder_work.dtype()));
  const auto total = static_cast<uint64_t>(quotient_work.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      q_offset > std::numeric_limits<uint32_t>::max() ||
      r_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  vulkan::DynamicArrayRef arrays[] = {
      {&a, 0}, {&b, 1}, {&quotient_work, 2}, {&remainder_work, 3}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 5;
  const std::string shader_name = "dynamic_divmod_" +
      std::to_string(static_cast<int>(a.dtype().val()));
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      build_divmod_shader(a.dtype()),
      4,
      arrays,
      kPushConstantSize,
      s);
  struct PushConstants {
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t q_offset;
    uint32_t r_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
      static_cast<uint32_t>(q_offset),
      static_cast<uint32_t>(r_offset),
      static_cast<uint32_t>(total),
  };
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total_elements + 255) / 256, 1, 1);
  vulkan::end_command_recording(s.index);

  if (staged_quotient) {
    copy_gpu(quotient_work, quotient, CopyType::General, s);
  }
  if (staged_remainder) {
    copy_gpu(remainder_work, remainder, CopyType::General, s);
  }
  return true;
}

bool try_eval_remainder_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  array quotient(out.shape(), out.dtype(), nullptr, {});
  std::vector<array> outputs = {quotient, out};
  return try_eval_divmod_vulkan(inputs, outputs, s);
}

template <typename Primitive>
constexpr vulkan::BinaryDispatchVariant binary_dispatch_variant() {
  if constexpr (std::is_same_v<Primitive, Add>) {
    return vulkan::BinaryDispatchVariant::AddWithPartials;
  } else {
    return vulkan::BinaryDispatchVariant::Standard;
  }
}

template <typename Primitive>
constexpr BinaryShaderOp binary_shader_op() {
  if constexpr (std::is_same_v<Primitive, Add>) {
    return BinaryShaderOp::Add;
  } else if constexpr (std::is_same_v<Primitive, Divide>) {
    return BinaryShaderOp::Divide;
  } else if constexpr (std::is_same_v<Primitive, Maximum>) {
    return BinaryShaderOp::Maximum;
  } else if constexpr (std::is_same_v<Primitive, Minimum>) {
    return BinaryShaderOp::Minimum;
  } else if constexpr (std::is_same_v<Primitive, Multiply>) {
    return BinaryShaderOp::Multiply;
  } else {
    return BinaryShaderOp::Subtract;
  }
}

template <typename Primitive>
bool try_eval_binary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    Stream s) {
  auto trace_binary_unsupported =
      [&](std::string_view reason, const array& lhs, const array& rhs) {
        if (!trace_fallback_enabled()) {
          return;
        }
        std::ostringstream oss;
        oss << "binary_vulkan_unsupported op=" << op_name
            << " reason=" << reason << " lhs_shape=" << lhs.shape()
            << " lhs_dtype=" << lhs.dtype() << " rhs_shape=" << rhs.shape()
            << " rhs_dtype=" << rhs.dtype() << " out_shape=" << out.shape()
            << " out_dtype=" << out.dtype();
        trace_fallback(oss.str());
      };

  if (inputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  const bool mixed_numeric_div = std::string_view(op_name) == "div" &&
      out.dtype() == float32 && is_vulkan_div_cast_dtype(a.dtype()) &&
      is_vulkan_div_cast_dtype(b.dtype());
  const bool bool_add = std::is_same_v<Primitive, Add> && a.dtype() == bool_ &&
      b.dtype() == bool_ && out.dtype() == bool_;
  const bool small_signed_integer_case = a.dtype() == b.dtype() &&
      a.dtype() == out.dtype() && (a.dtype() == int8 || a.dtype() == int16);
  const bool small_unsigned_integer_case = a.dtype() == b.dtype() &&
      a.dtype() == out.dtype() && (a.dtype() == uint8 || a.dtype() == uint16);
  const bool float_case = is_vulkan_float_dtype(a.dtype()) &&
      is_vulkan_float_dtype(b.dtype()) && is_vulkan_float_dtype(out.dtype());
  const bool integer_case = a.dtype() == b.dtype() &&
      a.dtype() == out.dtype() && is_vulkan_integer_dtype(a.dtype());
  const bool complex_add =
      std::is_same_v<Primitive, Add> && is_same_complex_add(a, b, out);
  const bool complex_sub =
      std::is_same_v<Primitive, Subtract> && is_same_complex_sub(a, b, out);
  const bool complex_scalar_mul =
      std::is_same_v<Primitive, Multiply> && is_complex_scalar_mul(a, b, out);
  const bool complex_float_mul =
      std::is_same_v<Primitive, Multiply> && is_complex_float_mul(a, b, out);
  const bool complex_mul =
      std::is_same_v<Primitive, Multiply> && is_same_complex_mul(a, b, out);
  const bool complex_div =
      std::is_same_v<Primitive, Divide> && is_same_complex_div(a, b, out);
  const bool complex_max =
      std::is_same_v<Primitive, Maximum> && is_same_complex_max(a, b, out);
  const bool complex_min =
      std::is_same_v<Primitive, Minimum> && is_same_complex_min(a, b, out);
  if (!float_case && !integer_case && !bool_add && !mixed_numeric_div &&
      !complex_add && !complex_sub && !complex_scalar_mul &&
      !complex_float_mul && !complex_mul && !complex_div && !complex_max &&
      !complex_min) {
    trace_binary_unsupported("unsupported_dtype_combo", a, b);
    return false;
  }

  if ((a.data_size() == 1 && a.has_primitive()) ||
      (b.data_size() == 1 && b.has_primitive())) {
    if (a.data_size() == 1 && a.has_primitive()) {
      ensure_materialized_scalar_input(a);
    }
    if (b.data_size() == 1 && b.has_primitive()) {
      ensure_materialized_scalar_input(b);
    }
  }

  if (out.size() == 0) {
    return true;
  }

  if (!ensure_vulkan_buffer(a, s) || !ensure_vulkan_buffer(b, s)) {
    return false;
  }

  if (complex_add) {
    return try_eval_complex_add_vulkan(a, b, out, s);
  }

  if (complex_sub) {
    return try_eval_complex_sub_vulkan(a, b, out, s);
  }

  if (complex_scalar_mul) {
    return try_eval_complex_scalar_mul_vulkan(a, b, out, s);
  }

  if (complex_float_mul) {
    return try_eval_complex_float_mul_vulkan(a, b, out, s);
  }

  if (complex_mul) {
    return try_eval_complex_mul_vulkan(a, b, out, s);
  }

  if (complex_div) {
    return try_eval_complex_div_vulkan(a, b, out, s);
  }

  if (complex_max) {
    return try_eval_complex_max_vulkan(a, b, out, s);
  }

  if (complex_min) {
    return try_eval_complex_min_vulkan(a, b, out, s);
  }

  if (bool_add) {
    array a_u32(a.shape(), uint32, nullptr, {});
    array b_u32(b.shape(), uint32, nullptr, {});
    copy_gpu(a, a_u32, CopyType::General, s);
    copy_gpu(b, b_u32, CopyType::General, s);
    a = a_u32;
    b = b_u32;
  }

  if (mixed_numeric_div) {
    array a_f32(a.shape(), float32, nullptr, {});
    array b_f32(b.shape(), float32, nullptr, {});
    copy_gpu(a, a_f32, CopyType::General, s);
    copy_gpu(b, b_f32, CopyType::General, s);
    a = a_f32;
    b = b_f32;
  }

  if (small_signed_integer_case) {
    array a_i32(a.shape(), int32, nullptr, {});
    array b_i32(b.shape(), int32, nullptr, {});
    copy_gpu(a, a_i32, CopyType::General, s);
    copy_gpu(b, b_i32, CopyType::General, s);
    a = a_i32;
    b = b_i32;
  } else if (small_unsigned_integer_case) {
    array a_u32(a.shape(), uint32, nullptr, {});
    array b_u32(b.shape(), uint32, nullptr, {});
    copy_gpu(a, a_u32, CopyType::General, s);
    copy_gpu(b, b_u32, CopyType::General, s);
    a = a_u32;
    b = b_u32;
  }

  const bool mixed_bf16_f16 =
      ((a.dtype() == bfloat16 && b.dtype() == float16) ||
       (a.dtype() == float16 && b.dtype() == bfloat16));
  const bool use_f32_staging_io =
      (std::string_view(op_name) == "div" &&
       (a.dtype() == float16 || b.dtype() == float16 ||
        out.dtype() == float16 || a.dtype() == bfloat16 ||
        b.dtype() == bfloat16 || out.dtype() == bfloat16)) ||
      mixed_bf16_f16;
  if (use_f32_staging_io) {
    array a_f32(a.shape(), float32, nullptr, {});
    array b_f32(b.shape(), float32, nullptr, {});
    copy_gpu(a, a_f32, CopyType::General, s);
    copy_gpu(b, b_f32, CopyType::General, s);
    a = a_f32;
    b = b_f32;
  }

  const bool scalar_vector_case =
      (a.data_size() == 1 && b.size() > 1) || (b.data_size() == 1 && a.size() > 1);
  if (scalar_vector_case) {
    bool scalar_is_a = a.data_size() == 1;
    array scalar = scalar_is_a ? a : b;
    array vec = scalar_is_a ? b : a;

    if (scalar.has_primitive()) {
      ensure_materialized_scalar_input(scalar);
    }
    if (vec.has_primitive() || !has_vulkan_buffer(vec) ||
        !is_supported_elementwise_layout(vec)) {
      vec = contiguous_copy_gpu(vec, s);
    }

    array scalar_broadcast(
        out.shape(),
        scalar.dtype(),
        std::make_shared<Broadcast>(s, out.shape()),
        {scalar});
    ensure_materialized_scalar_input(scalar_broadcast);
    array scalar_full(out.shape(), scalar.dtype(), nullptr, {});
    copy_gpu(scalar_broadcast, scalar_full, CopyType::Scalar, s);

    if (scalar_is_a) {
      a = scalar_full;
      b = vec;
    } else {
      a = vec;
      b = scalar_full;
    }
  }

  bool a_materialized = false;
  bool b_materialized = false;
  auto materialize_broadcast_input = [&](array& in, bool& was_materialized) {
    if (in.shape() == out.shape()) {
      if (in.data_size() == 1 && in.size() != 1) {
        if (in.has_primitive()) {
          ensure_materialized_scalar_input(in);
        }
        array materialized_arr(out.shape(), in.dtype(), nullptr, {});
        copy_gpu(in, materialized_arr, CopyType::Scalar, s);
        in = materialized_arr;
        was_materialized = true;
      }
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (in.data_size() == 1) {
      array broadcast_arr(
          out.shape(),
          in.dtype(),
          std::make_shared<Broadcast>(s, out.shape()),
          {in});
      ensure_materialized_scalar_input(broadcast_arr);
      array materialized_arr(out.shape(), in.dtype(), nullptr, {});
      copy_gpu(broadcast_arr, materialized_arr, CopyType::Scalar, s);
      in = materialized_arr;
      was_materialized = true;
      return true;
    }
    if (!ensure_vulkan_buffer(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
    broadcast(in, view);
    in = view;
    was_materialized = true;
    return true;
  };

  if (!materialize_broadcast_input(a, a_materialized) ||
      !materialize_broadcast_input(b, b_materialized)) {
    trace_binary_unsupported("broadcast_materialization_failed", a, b);
    return false;
  }

  if (!is_supported_elementwise_layout(a) && !is_supported_unary_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b) && !is_supported_unary_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }
  a = collapse_binary_leading_dims(a, s);
  b = collapse_binary_leading_dims(b, s);

  const bool staged_output = use_f32_staging_io || small_signed_integer_case ||
      small_unsigned_integer_case || !is_supported_elementwise_layout(out);
  array out_work = staged_output
      ? array(
            out.shape(),
            use_f32_staging_io
                ? float32
                : (small_signed_integer_case
                       ? int32
                       : (small_unsigned_integer_case ? uint32 : out.dtype())),
            nullptr,
            {})
      : out;

  auto bopt = get_binary_op_type(a, b);
  if (small_signed_integer_case || small_unsigned_integer_case) {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  } else if constexpr (std::is_same_v<Primitive, Multiply>) {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  } else if (a_materialized || b_materialized) {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  } else {
    set_binary_op_output_data(a, b, out_work, bopt);
  }
  array out_kernel = collapse_binary_leading_dims(out_work, s);
  if ((!is_supported_elementwise_layout(a) && !is_supported_unary_layout(a)) ||
      (!is_supported_elementwise_layout(b) && !is_supported_unary_layout(b)) ||
      !is_supported_elementwise_layout(out_kernel)) {
    trace_binary_unsupported("unsupported_elementwise_layout", a, b);
    return false;
  }

  if (out_kernel.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  const auto shader_id = bool_add
      ? std::optional<vulkan::StaticShaderId>(
            vulkan::StaticShaderId::maximum_u32_u32_u8)
      : binary_shader_id(
            binary_shader_op<Primitive>(),
            a.dtype(),
            b.dtype(),
            out_kernel.dtype(),
            out_kernel.dtype() == float16);
  if (!shader_id.has_value()) {
    trace_binary_unsupported("missing_shader", a, b);
    return false;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    auto dispatch_variant = binary_dispatch_variant<Primitive>();
    if constexpr (std::is_same_v<Primitive, Add>) {
      if (use_f32_staging_io || bool_add || integer_case) {
        dispatch_variant = vulkan::BinaryDispatchVariant::Standard;
      } else if (a.data_size() == 1 || b.data_size() == 1) {
        dispatch_variant = vulkan::BinaryDispatchVariant::Standard;
      }
    }
    vulkan::dispatch_binary_op(
        a, b, out_kernel, *shader_id, command_buffer, s, dispatch_variant);
    vulkan::end_command_recording(s.index);
    vulkan::retain_array_for_stream(s, a);
    vulkan::retain_array_for_stream(s, b);
    vulkan::retain_array_for_stream(s, out_kernel);
    if (out_work.data_shared_ptr() != out_kernel.data_shared_ptr()) {
      vulkan::retain_array_for_stream(s, out_work);
    }
    if (staged_output || use_f32_staging_io) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "binary_dispatch_failed op=" << op_name << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_binary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    Stream s) {
  if (!try_eval_binary_op_vulkan<Primitive>(inputs, out, op_name, s)) {
    throw std::runtime_error(
        std::string("Binary operation ") + op_name +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

bool try_eval_greater_equal_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2 || out.dtype() != bool_) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (!is_vulkan_compare_dtype(a.dtype()) ||
      !is_vulkan_compare_dtype(b.dtype())) {
    return false;
  }
  if (a.dtype() != b.dtype() &&
      !(is_vulkan_float_dtype(a.dtype()) && is_vulkan_float_dtype(b.dtype()))) {
    return false;
  }
  if (a.shape() != out.shape() || b.shape() != out.shape()) {
    return false;
  }

  const bool use_f32_staging_io =
      a.dtype() == bfloat16 || b.dtype() == bfloat16;
  if (use_f32_staging_io) {
    array a_f32(a.shape(), float32, nullptr, {});
    array b_f32(b.shape(), float32, nullptr, {});
    copy_gpu(a, a_f32, CopyType::General, s);
    copy_gpu(b, b_f32, CopyType::General, s);
    a = a_f32;
    b = b_f32;
  }

  if (!is_supported_elementwise_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }

  const auto shader_id = binary_shader_id(
      BinaryShaderOp::GreaterEqual, a.dtype(), b.dtype(), uint8, false);
  if (!shader_id.has_value()) {
    return false;
  }

  array out_u8(out.shape(), uint8, nullptr, {});
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out_u8, bopt, [&](size_t n) {
    return vulkan::allocator().malloc(n);
  });
  if (!is_supported_elementwise_layout(out_u8)) {
    return false;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_binary_op(
        a,
        b,
        out_u8,
        *shader_id,
        command_buffer,
        s,
        vulkan::BinaryDispatchVariant::Standard);
    vulkan::end_command_recording(s.index);
    out.copy_shared_buffer(
        out_u8,
        out_u8.strides(),
        out_u8.flags(),
        out_u8.data_size(),
        out_u8.offset());
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "binary_dispatch_failed op=greater_equal reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

#define VULKAN_BINARY_GPU(func, op_name)                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_binary_vulkan<func>(inputs, out, op_name, stream());         \
  }

VULKAN_BINARY_GPU(Add, "add")
VULKAN_BINARY_GPU(Minimum, "minimum")
VULKAN_BINARY_GPU(Maximum, "maximum")
VULKAN_BINARY_GPU(Divide, "div")
VULKAN_BINARY_GPU(Subtract, "sub")
VULKAN_BINARY_GPU(Multiply, "mul")

void Remainder::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_remainder_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "Remainder has no Vulkan implementation.");
  }
}

void GreaterEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_greater_equal_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "GreaterEqual operation failed on Vulkan (unsupported dtype or layout).");
  }
}

void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (!try_eval_divmod_vulkan(inputs, outputs, stream())) {
    throw std::runtime_error(
        "DivMod operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
