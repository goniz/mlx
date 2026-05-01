// Copyright © 2024 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"

namespace mlx::core {

#define NYI_OP(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NYI_OP_STATE(func)                                            \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NYI_OP_MULTI(func)                                              \
  void func::eval_gpu(                                                  \
      const std::vector<array>& inputs, std::vector<array>& outputs) {  \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NYI_OP_MULTI_STATE(func)                                       \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NO_GPU_MULTI_STATE(func)                                       \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NO_GPU_STATE(func)                                            \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

namespace {

array collapse_power_leading_dims(const array& arr, Stream s) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  return flatten_in_eval(arr, 0, arr.ndim() - 4, s);
}

array collapse_compare_leading_dims(const array& arr, Stream s) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  return flatten_in_eval(arr, 0, arr.ndim() - 4, s);
}

bool ensure_vulkan_buffer_power(array& arr, Stream s) {
  auto data = arr.data_shared_ptr();
  if (data != nullptr && vulkan::is_vulkan_buffer(data->buffer)) {
    return true;
  }
  if (arr.has_primitive()) {
    arr = contiguous_copy_gpu(arr, s);
    data = arr.data_shared_ptr();
    return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
  }
  arr.wait();
  data = arr.data_shared_ptr();
  if (data != nullptr && vulkan::is_vulkan_buffer(data->buffer)) {
    return true;
  }
  arr = contiguous_copy_gpu(arr, s);
  data = arr.data_shared_ptr();
  return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
}

bool ensure_vulkan_buffer_compare(array& arr, Stream s) {
  auto data = arr.data_shared_ptr();
  if (data != nullptr && vulkan::is_vulkan_buffer(data->buffer)) {
    return true;
  }
  if (arr.has_primitive()) {
    arr = contiguous_copy_gpu(arr, s);
    data = arr.data_shared_ptr();
    return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
  }
  arr.wait();
  data = arr.data_shared_ptr();
  if (data != nullptr && vulkan::is_vulkan_buffer(data->buffer)) {
    return true;
  }
  arr = contiguous_copy_gpu(arr, s);
  data = arr.data_shared_ptr();
  return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
}

bool is_supported_equal_dtype(Dtype dtype) {
  switch (dtype) {
    case bool_:
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

bool is_supported_ordered_compare_dtype(Dtype dtype) {
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

enum class CompareOp {
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
};

std::string emit_bf16_power_helpers() {
  return R"(
uint fp32_to_bf16(float f) {
  uint u = floatBitsToUint(f);
  u = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
  return u;
}

float bf16_to_fp32(uint u) {
  return uintBitsToFloat(u << 16);
}

)";
}

std::string power_input_expr(
    Dtype dtype,
    const char* buffer_name,
    const char* index_name) {
  if (dtype == bfloat16) {
    return std::string("bf16_to_fp32(uint(") + buffer_name + ".data[" +
        index_name + "]))";
  }
  if (dtype == float16) {
    return std::string("float(") + buffer_name + ".data[" + index_name + "])";
  }
  return std::string(buffer_name) + ".data[" + index_name + "]";
}

std::string power_output_expr(Dtype out_dtype, const std::string& expr) {
  if (out_dtype == bfloat16) {
    return "uint16_t(fp32_to_bf16(" + expr + "))";
  }
  if (out_dtype == float16) {
    return "float16_t(" + expr + ")";
  }
  return expr;
}

std::string build_power_shader(Dtype a_dtype, Dtype b_dtype, Dtype out_dtype) {
  std::ostringstream os;
  const bool uses_bfloat16 =
      a_dtype == bfloat16 || b_dtype == bfloat16 || out_dtype == bfloat16;
  const bool uses_float16 =
      a_dtype == float16 || b_dtype == float16 || out_dtype == float16;
  if (uses_bfloat16 || uses_float16) {
    os << "#version 450\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n\n";
    os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  } else {
    os << vulkan::emit_dynamic_shader_preamble(a_dtype, out_dtype, false);
  }
  if (a_dtype == bfloat16 || b_dtype == bfloat16 || out_dtype == bfloat16) {
    os << emit_bf16_power_helpers();
  }
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {"
     << vulkan::dtype_to_glsl_storage_type(a_dtype) << " data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {"
     << vulkan::dtype_to_glsl_storage_type(b_dtype) << " data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {"
     << vulkan::dtype_to_glsl_storage_type(out_dtype) << " data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint linear_idx = gl_GlobalInvocationID.x;\n";
  os << "  if (linear_idx >= pc.total_elements) return;\n";
  os << "  uint idx = linear_idx;\n";
  os << "  uint a_idx = idx + pc.a_offset;\n";
  os << "  uint b_idx = idx + pc.b_offset;\n";
  os << "  float lhs = " << power_input_expr(a_dtype, "a_buf", "a_idx") << ";\n";
  os << "  float rhs = " << power_input_expr(b_dtype, "b_buf", "b_idx") << ";\n";
  os << "  out_buf.data[idx + pc.out_offset] = "
      << power_output_expr(out_dtype, "pow(lhs, rhs)") << ";\n";
  os << "}\n";
  return os.str();
}

std::string equal_input_expr(
    Dtype dtype,
    const char* buffer_name,
    const char* index_name) {
  if (dtype == bfloat16) {
    return std::string("bf16_to_fp32(uint(") + buffer_name + ".data[" +
        index_name + "]))";
  }
  if (dtype == float16) {
    return std::string("float(") + buffer_name + ".data[" + index_name + "])";
  }
  if (dtype == bool_) {
    return std::string("uint(") + buffer_name + ".data[" + index_name + "])";
  }
  return std::string(buffer_name) + ".data[" + index_name + "]";
}

std::string build_equal_shader(Dtype a_dtype, Dtype b_dtype, bool equal_nan) {
  std::ostringstream os;
  const bool uses_bfloat16 = a_dtype == bfloat16 || b_dtype == bfloat16;
  const bool uses_float16 = a_dtype == float16 || b_dtype == float16;
  const bool uses_int64 =
      a_dtype == int64 || a_dtype == uint64 || b_dtype == int64 ||
      b_dtype == uint64;

  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  if (uses_int64) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  if (uses_bfloat16 || uses_float16) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
  os << "#extension GL_EXT_shader_8bit_storage : require\n";
  os << "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";

  if (uses_bfloat16) {
    os << emit_bf16_power_helpers();
  }

  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {"
     << vulkan::dtype_to_glsl_storage_type(a_dtype) << " data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {"
     << vulkan::dtype_to_glsl_storage_type(b_dtype) << " data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {uint8_t data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint linear_idx = gl_GlobalInvocationID.x;\n";
  os << "  if (linear_idx >= pc.total_elements) return;\n";
  os << "  uint idx = linear_idx;\n";
  os << "  uint a_idx = idx + pc.a_offset;\n";
  os << "  uint b_idx = idx + pc.b_offset;\n";
  os << "  bool is_equal = (" << equal_input_expr(a_dtype, "a_buf", "a_idx")
     << ") == (" << equal_input_expr(b_dtype, "b_buf", "b_idx") << ")";
  if (equal_nan && (uses_bfloat16 || uses_float16 || a_dtype == float32 ||
                    b_dtype == float32)) {
    os << " || (isnan(" << equal_input_expr(a_dtype, "a_buf", "a_idx")
       << ") && isnan(" << equal_input_expr(b_dtype, "b_buf", "b_idx")
       << "))";
  }
  os << ";\n";
  os << "  out_buf.data[idx + pc.out_offset] = is_equal ? uint8_t(1) : uint8_t(0);\n";
  os << "}\n";
  return os.str();
}

std::string build_compare_shader(Dtype a_dtype, Dtype b_dtype, CompareOp op) {
  std::ostringstream os;
  const bool uses_bfloat16 = a_dtype == bfloat16 || b_dtype == bfloat16;
  const bool uses_float16 = a_dtype == float16 || b_dtype == float16;
  const bool uses_int64 =
      a_dtype == int64 || a_dtype == uint64 || b_dtype == int64 ||
      b_dtype == uint64;

  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  if (uses_int64) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  if (uses_bfloat16 || uses_float16) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
  os << "#extension GL_EXT_shader_8bit_storage : require\n";
  os << "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";

  if (uses_bfloat16) {
    os << emit_bf16_power_helpers();
  }

  const char* expr = "==";
  switch (op) {
    case CompareOp::Equal:
      expr = "==";
      break;
    case CompareOp::NotEqual:
      expr = "!=";
      break;
    case CompareOp::Less:
      expr = "<";
      break;
    case CompareOp::LessEqual:
      expr = "<=";
      break;
    case CompareOp::Greater:
      expr = ">";
      break;
  }

  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {"
     << vulkan::dtype_to_glsl_storage_type(a_dtype) << " data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {"
     << vulkan::dtype_to_glsl_storage_type(b_dtype) << " data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {uint8_t data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  uint a_idx = idx + pc.a_offset;\n";
  os << "  uint b_idx = idx + pc.b_offset;\n";
  os << "  bool result = (" << equal_input_expr(a_dtype, "a_buf", "a_idx")
     << ") " << expr << " (" << equal_input_expr(b_dtype, "b_buf", "b_idx")
     << ");\n";
  os << "  out_buf.data[idx + pc.out_offset] = result ? uint8_t(1) : uint8_t(0);\n";
  os << "}\n";
  return os.str();
}

bool try_eval_equal_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    bool equal_nan) {
  if (inputs.size() != 2 || out.dtype() != bool_) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (!is_supported_equal_dtype(a.dtype()) || !is_supported_equal_dtype(b.dtype())) {
    return false;
  }
  if (a.dtype() != b.dtype() &&
      !(is_vulkan_float_dtype(a.dtype()) && is_vulkan_float_dtype(b.dtype()))) {
    return false;
  }

  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == out.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer_compare(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
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

  a = collapse_compare_leading_dims(a, s);
  b = collapse_compare_leading_dims(b, s);

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work = staged_output ? array(out.shape(), bool_, nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  out_work = collapse_compare_leading_dims(out_work, s);

  if (!is_supported_elementwise_layout(a) || !is_supported_elementwise_layout(b) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (!ensure_vulkan_buffer_compare(a, s) || !ensure_vulkan_buffer_compare(b, s) ||
      !ensure_vulkan_buffer_compare(out_work, s)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto out_offset = static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const std::string shader_name = std::string(equal_nan ? "dynamic_nan_equal_" : "dynamic_equal_") +
      std::to_string(static_cast<int>(a.dtype().val())) + "_" +
      std::to_string(static_cast<int>(b.dtype().val()));
  const std::string glsl_source =
      build_equal_shader(a.dtype(), b.dtype(), equal_nan);
  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 4;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, glsl_source, 3, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
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

  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_compare_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    CompareOp op) {
  if (inputs.size() != 2 || out.dtype() != bool_) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  const bool supports_bool =
      op == CompareOp::Equal || op == CompareOp::NotEqual;
  const auto supported_dtype = [&](Dtype dtype) {
    return supports_bool ? is_supported_equal_dtype(dtype)
                         : is_supported_ordered_compare_dtype(dtype);
  };
  if (!supported_dtype(a.dtype()) || !supported_dtype(b.dtype())) {
    return false;
  }
  if (a.dtype() != b.dtype() &&
      !(is_vulkan_float_dtype(a.dtype()) && is_vulkan_float_dtype(b.dtype()))) {
    return false;
  }

  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == out.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer_compare(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
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

  a = collapse_compare_leading_dims(a, s);
  b = collapse_compare_leading_dims(b, s);

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work = staged_output ? array(out.shape(), bool_, nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  out_work = collapse_compare_leading_dims(out_work, s);

  if (!is_supported_elementwise_layout(a) || !is_supported_elementwise_layout(b) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (!ensure_vulkan_buffer_compare(a, s) || !ensure_vulkan_buffer_compare(b, s) ||
      !ensure_vulkan_buffer_compare(out_work, s)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto out_offset = static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const std::string shader_name = std::string("dynamic_compare_") +
      std::to_string(static_cast<int>(op)) + "_" +
      std::to_string(static_cast<int>(a.dtype().val())) + "_" +
      std::to_string(static_cast<int>(b.dtype().val()));
  const std::string glsl_source = build_compare_shader(a.dtype(), b.dtype(), op);
  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 4;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, glsl_source, 3, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
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

  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_power_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }
  array a = inputs[0];
  array b = inputs[1];

  auto is_supported_dtype = [](Dtype dtype) {
    return dtype == float16 || dtype == float32 || dtype == bfloat16;
  };
  if (!is_supported_dtype(a.dtype()) || !is_supported_dtype(b.dtype()) ||
      !is_supported_dtype(out.dtype())) {
    return false;
  }

  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == out.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer_power(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
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

  a = collapse_power_leading_dims(a, s);
  b = collapse_power_leading_dims(b, s);

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work = staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  out_work = collapse_power_leading_dims(out_work, s);

  if (!is_supported_elementwise_layout(a) || !is_supported_elementwise_layout(b) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (!ensure_vulkan_buffer_power(a, s) || !ensure_vulkan_buffer_power(b, s) ||
      !ensure_vulkan_buffer_power(out_work, s)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto out_offset = static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const std::string shader_name = "dynamic_power_" +
      std::to_string(static_cast<int>(a.dtype().val())) + "_" +
      std::to_string(static_cast<int>(b.dtype().val())) + "_" +
      std::to_string(static_cast<int>(out_work.dtype().val()));
  const std::string glsl_source =
      build_power_shader(a.dtype(), b.dtype(), out_work.dtype());
  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 4;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, glsl_source, 3, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
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

  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

std::string bitwise_input_expr(
    Dtype dtype,
    const char* buffer_name,
    const char* index_name) {
  if (dtype == bool_) {
    return std::string("uint8_t(") + buffer_name + ".data[" + index_name + "])";
  }
  return std::string(buffer_name) + ".data[" + index_name + "]";
}

const char* bitwise_op_expr(BitwiseBinary::Op op) {
  switch (op) {
    case BitwiseBinary::And:
      return "&";
    case BitwiseBinary::Or:
      return "|";
    case BitwiseBinary::Xor:
      return "^";
    case BitwiseBinary::LeftShift:
      return "<<";
    case BitwiseBinary::RightShift:
      return ">>";
  }
  return "&";
}

std::string build_bitwise_binary_shader(Dtype dtype, BitwiseBinary::Op op) {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(
      dtype, dtype, dtype == int64 || dtype == uint64);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {"
     << vulkan::dtype_to_glsl_storage_type(dtype) << " data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {"
     << vulkan::dtype_to_glsl_storage_type(dtype) << " data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {"
     << vulkan::dtype_to_glsl_storage_type(dtype) << " data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint linear_idx = gl_GlobalInvocationID.x;\n";
  os << "  if (linear_idx >= pc.total_elements) return;\n";
  os << "  uint idx = linear_idx;\n";
  os << "  uint a_idx = idx + pc.a_offset;\n";
  os << "  uint b_idx = idx + pc.b_offset;\n";
  os << "  out_buf.data[idx + pc.out_offset] = ("
     << bitwise_input_expr(dtype, "a_buf", "a_idx") << ") "
     << bitwise_op_expr(op) << " ("
     << bitwise_input_expr(dtype, "b_buf", "b_idx") << ");\n";
  os << "}\n";
  return os.str();
}

std::string build_logical_not_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(bool_, bool_, false);
  os << "layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer Input {uint8_t data[];} in_buf;\n";
  os << "layout(set = 0, binding = 1) buffer Output {uint8_t data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  out_buf.data[idx + pc.out_offset] = in_buf.data[idx + pc.in_offset] == uint8_t(0) ? uint8_t(1) : uint8_t(0);\n";
  os << "}\n";
  return os.str();
}

std::string build_logical_binary_shader(const char* expr) {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(bool_, bool_, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {uint8_t data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {uint8_t data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {uint8_t data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  bool lhs = a_buf.data[idx + pc.a_offset] != uint8_t(0);\n";
  os << "  bool rhs = b_buf.data[idx + pc.b_offset] != uint8_t(0);\n";
  os << "  out_buf.data[idx + pc.out_offset] = (" << expr
     << ") ? uint8_t(1) : uint8_t(0);\n";
  os << "}\n";
  return os.str();
}

bool try_eval_logical_not_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 1 || inputs[0].dtype() != bool_ || out.dtype() != bool_) {
    return false;
  }

  array in = inputs[0];
  if (!is_supported_elementwise_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }
  in = collapse_compare_leading_dims(in, s);

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work = staged_output ? array(out.shape(), bool_, nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  out_work = collapse_compare_leading_dims(out_work, s);

  if (!is_supported_elementwise_layout(in) || !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (!ensure_vulkan_buffer_compare(in, s) || !ensure_vulkan_buffer_compare(out_work, s)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto in_offset = static_cast<uint64_t>(in.offset() / size_of(in.dtype()));
  const auto out_offset = static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  vulkan::DynamicArrayRef arrays[] = {{&in, 0}, {&out_work, 1}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 3;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_logical_not", build_logical_not_shader(), 2, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t in_offset;
    uint32_t out_offset;
    uint32_t total_elements;
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

  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_logical_binary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    const char* shader_name,
    const char* expr) {
  if (inputs.size() != 2 || inputs[0].dtype() != bool_ || inputs[1].dtype() != bool_ ||
      out.dtype() != bool_) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];

  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == out.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer_compare(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
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
  a = collapse_compare_leading_dims(a, s);
  b = collapse_compare_leading_dims(b, s);

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work = staged_output ? array(out.shape(), bool_, nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  out_work = collapse_compare_leading_dims(out_work, s);

  if (!is_supported_elementwise_layout(a) || !is_supported_elementwise_layout(b) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (!ensure_vulkan_buffer_compare(a, s) || !ensure_vulkan_buffer_compare(b, s) ||
      !ensure_vulkan_buffer_compare(out_work, s)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto out_offset = static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 4;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, build_logical_binary_shader(expr), 3, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
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

  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_bitwise_binary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    BitwiseBinary::Op op) {
  if (inputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (a.dtype() != b.dtype() || a.dtype() != out.dtype()) {
    return false;
  }
  if (!(issubdtype(out.dtype(), integer) || out.dtype() == bool_)) {
    return false;
  }

  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == out.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer_compare(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
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

  a = collapse_compare_leading_dims(a, s);
  b = collapse_compare_leading_dims(b, s);

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work = staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  out_work = collapse_compare_leading_dims(out_work, s);

  if (!is_supported_elementwise_layout(a) || !is_supported_elementwise_layout(b) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (!ensure_vulkan_buffer_compare(a, s) || !ensure_vulkan_buffer_compare(b, s) ||
      !ensure_vulkan_buffer_compare(out_work, s)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto out_offset = static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const std::string shader_name = std::string("dynamic_bitwise_") +
      std::to_string(static_cast<int>(op)) + "_" +
      std::to_string(static_cast<int>(out.dtype().val()));
  const std::string glsl_source = build_bitwise_binary_shader(out.dtype(), op);
  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 4;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, glsl_source, 3, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
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

  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool is_supported_select_layout(const array& arr) {
  return arr.flags().contiguous && arr.offset() == 0 && arr.ndim() > 0 &&
      arr.strides().back() == 1;
}

array collapse_select_leading_dims(const array& arr, Stream s) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  return flatten_in_eval(arr, 0, arr.ndim() - 4, s);
}

std::optional<vulkan::StaticShaderId> select_shader_id(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return vulkan::StaticShaderId::select_bool;
    case float16:
      return vulkan::StaticShaderId::select_f16;
    case float32:
      return vulkan::StaticShaderId::select_f32;
    case bfloat16:
      return vulkan::StaticShaderId::select_bf16;
    default:
      return std::nullopt;
  }
}

std::string build_select_dynamic_shader(Dtype dtype) {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(
      bool_, dtype, dtype == int64 || dtype == uint64);
  os << "layout(push_constant) uniform PushConstants { uint cond_offset; uint x_offset; uint y_offset; uint out_offset; uint total_elements; uint out_ne0; uint out_ne1; uint out_ne2; uint out_ne3; uint cond_s0; uint cond_s1; uint cond_s2; uint cond_s3; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer Condition {uint8_t data[];} cond_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputX {"
     << vulkan::dtype_to_glsl_storage_type(dtype) << " data[];} x_buf;\n";
  os << "layout(set = 0, binding = 2) readonly buffer InputY {"
     << vulkan::dtype_to_glsl_storage_type(dtype) << " data[];} y_buf;\n";
  os << "layout(set = 0, binding = 3) buffer Output {"
     << vulkan::dtype_to_glsl_storage_type(dtype) << " data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  uint rem = idx;\n";
  os << "  uint i3 = rem % pc.out_ne3; rem /= pc.out_ne3;\n";
  os << "  uint i2 = rem % pc.out_ne2; rem /= pc.out_ne2;\n";
  os << "  uint i1 = rem % pc.out_ne1; rem /= pc.out_ne1;\n";
  os << "  uint i0 = rem;\n";
  os << "  uint cond_idx = pc.cond_offset + i0 * pc.cond_s0 + i1 * pc.cond_s1 + i2 * pc.cond_s2 + i3 * pc.cond_s3;\n";
  os << "  bool cond = cond_buf.data[cond_idx] != uint8_t(0);\n";
  os << "  out_buf.data[idx + pc.out_offset] = cond ? x_buf.data[idx + pc.x_offset] : y_buf.data[idx + pc.y_offset];\n";
  os << "}\n";
  return os.str();
}

bool try_eval_select_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 3) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "select_vulkan_unsupported reason=wrong_input_count"
          << " out_shape=" << out.shape() << " out_dtype=" << out.dtype();
      trace_fallback(oss.str());
    }
    return false;
  }

  array condition = inputs[0];
  array x = inputs[1];
  array y = inputs[2];
  auto trace_select_unsupported = [&](std::string_view reason) {
    if (!trace_fallback_enabled()) {
      return;
    }
    std::ostringstream oss;
    oss << "select_vulkan_unsupported reason=" << reason
        << " out_shape=" << out.shape() << " out_dtype=" << out.dtype()
        << " cond_shape=" << condition.shape() << " cond_dtype="
        << condition.dtype() << " x_shape=" << x.shape() << " x_dtype="
        << x.dtype() << " y_shape=" << y.shape() << " y_dtype="
        << y.dtype();
    if (reason == "unsupported_elementwise_layout") {
      oss << " cond_ok=" << is_supported_elementwise_layout(condition)
          << " x_ok=" << is_supported_elementwise_layout(x)
          << " y_ok=" << is_supported_elementwise_layout(y)
          << " out_ok=" << is_supported_elementwise_layout(out)
          << " cond_row=" << condition.flags().row_contiguous
          << " x_row=" << x.flags().row_contiguous
          << " y_row=" << y.flags().row_contiguous
          << " out_row=" << out.flags().row_contiguous
          << " cond_offset=" << condition.offset()
          << " x_offset=" << x.offset()
          << " y_offset=" << y.offset()
          << " out_offset=" << out.offset();
    }
    trace_fallback(oss.str());
  };
  if (condition.dtype() != bool_ || x.dtype() != out.dtype() ||
      y.dtype() != out.dtype()) {
    trace_select_unsupported("dtype_mismatch");
    return false;
  }

  auto materialize_broadcast_input = [&](array& in, Dtype expected_dtype) {
    if (in.dtype() != expected_dtype) {
      return false;
    }
    if (in.shape() == out.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer_compare(in, s)) {
      return false;
    }
    array view(out.shape(), expected_dtype, nullptr, {});
    broadcast(in, view);
    in = view;
    return true;
  };

  if (!materialize_broadcast_input(condition, bool_) ||
      !materialize_broadcast_input(x, out.dtype()) ||
      !materialize_broadcast_input(y, out.dtype())) {
    trace_select_unsupported("broadcast_materialization_failed");
    return false;
  }

  auto materialize = [&](array arr) {
    if (!is_supported_select_layout(arr)) {
      array staged(arr.shape(), arr.dtype(), nullptr, {});
      staged.set_data(allocator::malloc(staged.nbytes()));
      copy_gpu(arr, staged, CopyType::General, s);
      arr = staged;
    }
    return arr;
  };

  condition = materialize(condition);
  x = materialize(x);
  y = materialize(y);

  const bool staged_output = !is_supported_select_layout(out);
  array out_storage =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_storage.set_data(allocator::malloc(out_storage.nbytes()));

  array cond_kernel = collapse_select_leading_dims(condition, s);
  array x_kernel = collapse_select_leading_dims(x, s);
  array y_kernel = collapse_select_leading_dims(y, s);
  array out_kernel = collapse_select_leading_dims(out_storage, s);

  const bool can_use_static_select =
      is_supported_elementwise_layout(cond_kernel) &&
      is_supported_elementwise_layout(x_kernel) &&
      is_supported_elementwise_layout(y_kernel) &&
      is_supported_elementwise_layout(out_kernel);
  const bool can_use_dynamic_select =
      is_supported_elementwise_layout(x_kernel) &&
      is_supported_elementwise_layout(y_kernel) &&
      is_supported_elementwise_layout(out_kernel) && cond_kernel.ndim() <= 4;

  if (!can_use_static_select && !can_use_dynamic_select) {
    trace_select_unsupported("unsupported_elementwise_layout");
    return false;
  }

  if (!is_supported_elementwise_layout(x_kernel) ||
      !is_supported_elementwise_layout(y_kernel) ||
      !is_supported_elementwise_layout(out_kernel)) {
    trace_select_unsupported("unsupported_elementwise_layout");
    return false;
  }

  if (out_kernel.size() == 0) {
    if (staged_output) {
      copy_gpu(out_storage, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto shader_id = select_shader_id(out.dtype());
    if (shader_id.has_value() && can_use_static_select) {
      auto command_buffer = vulkan::begin_command_recording(s.index);
      vulkan::dispatch_ternary_op(
          cond_kernel,
          x_kernel,
          y_kernel,
          out_kernel,
          *shader_id,
          command_buffer,
          s);
      vulkan::end_command_recording(s.index);
    } else {
      const auto cond_offset =
          static_cast<uint64_t>(cond_kernel.offset() / size_of(cond_kernel.dtype()));
      const auto x_offset =
          static_cast<uint64_t>(x_kernel.offset() / size_of(x_kernel.dtype()));
      const auto y_offset =
          static_cast<uint64_t>(y_kernel.offset() / size_of(y_kernel.dtype()));
      const auto out_offset =
          static_cast<uint64_t>(out_kernel.offset() / size_of(out_kernel.dtype()));
      const auto total = static_cast<uint64_t>(out_kernel.data_size());
      if (cond_offset > std::numeric_limits<uint32_t>::max() ||
          x_offset > std::numeric_limits<uint32_t>::max() ||
          y_offset > std::numeric_limits<uint32_t>::max() ||
          out_offset > std::numeric_limits<uint32_t>::max() ||
          total > std::numeric_limits<uint32_t>::max()) {
        return false;
      }

      vulkan::DynamicArrayRef arrays[] = {
          {&cond_kernel, 0}, {&x_kernel, 1}, {&y_kernel, 2}, {&out_kernel, 3}};
      constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 13;
      auto dispatch = vulkan::dispatch_dynamic_compute_begin(
          std::string("dynamic_select_") +
              std::to_string(static_cast<int>(out.dtype().val())),
          build_select_dynamic_shader(out.dtype()),
          4,
          arrays,
          kPushConstantSize,
          s);

      auto shape_dim_or_one = [](const array& arr, int axis) {
        const int padded_axis = axis - (4 - arr.ndim());
        return padded_axis < 0 ? int64_t{1} : arr.shape(padded_axis);
      };
      auto stride_dim_or_zero = [](const array& arr, int axis) {
        const int padded_axis = axis - (4 - arr.ndim());
        return padded_axis < 0 ? int64_t{0} : arr.strides(padded_axis);
      };

      struct PushConstants {
        uint32_t cond_offset;
        uint32_t x_offset;
        uint32_t y_offset;
        uint32_t out_offset;
        uint32_t total_elements;
        uint32_t out_ne0;
        uint32_t out_ne1;
        uint32_t out_ne2;
        uint32_t out_ne3;
        uint32_t cond_s0;
        uint32_t cond_s1;
        uint32_t cond_s2;
        uint32_t cond_s3;
      } pc{
          static_cast<uint32_t>(cond_offset),
          static_cast<uint32_t>(x_offset),
          static_cast<uint32_t>(y_offset),
          static_cast<uint32_t>(out_offset),
          static_cast<uint32_t>(total),
          checked_u32_size(shape_dim_or_one(out_kernel, 0), "select_out_ne0"),
          checked_u32_size(shape_dim_or_one(out_kernel, 1), "select_out_ne1"),
          checked_u32_size(shape_dim_or_one(out_kernel, 2), "select_out_ne2"),
          checked_u32_size(shape_dim_or_one(out_kernel, 3), "select_out_ne3"),
          checked_u32_size(stride_dim_or_zero(cond_kernel, 0), "select_cond_s0"),
          checked_u32_size(stride_dim_or_zero(cond_kernel, 1), "select_cond_s1"),
          checked_u32_size(stride_dim_or_zero(cond_kernel, 2), "select_cond_s2"),
          checked_u32_size(stride_dim_or_zero(cond_kernel, 3), "select_cond_s3"),
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
    }
    if (staged_output) {
      copy_gpu(out_storage, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "select_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_equal_vulkan(inputs, out, stream(), equal_nan_)) {
    throw std::runtime_error(
        std::string(name()) +
        " has no Vulkan implementation for this input.");
  }
}

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_bitwise_binary_vulkan(inputs, out, stream(), op_)) {
    throw std::runtime_error(
        std::string(name()) +
        " has no Vulkan implementation for this input.");
  }
}

void Greater::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_compare_vulkan(inputs, out, stream(), CompareOp::Greater)) {
    throw std::runtime_error("Greater has no Vulkan implementation for this input.");
  }
}

void Less::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_compare_vulkan(inputs, out, stream(), CompareOp::Less)) {
    throw std::runtime_error("Less has no Vulkan implementation for this input.");
  }
}

void LessEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_compare_vulkan(inputs, out, stream(), CompareOp::LessEqual)) {
    throw std::runtime_error(
        "LessEqual has no Vulkan implementation for this input.");
  }
}

void NotEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_compare_vulkan(inputs, out, stream(), CompareOp::NotEqual)) {
    throw std::runtime_error(
        "NotEqual has no Vulkan implementation for this input.");
  }
}

void LogicalNot::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_logical_not_vulkan(inputs, out, stream())) {
    throw std::runtime_error("LogicalNot has no Vulkan implementation for this input.");
  }
}

void LogicalAnd::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_logical_binary_vulkan(inputs, out, stream(), "dynamic_logical_and", "lhs && rhs")) {
    throw std::runtime_error("LogicalAnd has no Vulkan implementation for this input.");
  }
}

void LogicalOr::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_logical_binary_vulkan(inputs, out, stream(), "dynamic_logical_or", "lhs || rhs")) {
    throw std::runtime_error("LogicalOr has no Vulkan implementation for this input.");
  }
}

// Primitives implemented in other files:
// - binary.cpp: Add, Minimum, Maximum, Divide, Subtract, Multiply
// - unary.cpp: Abs, Ceil, Cos, Exp, Erf, ErfInv, Floor, Log, Sin, etc.
// - reduce.cpp: Reduce, ArgReduce
// - softmax.cpp: Softmax, LogSumExp
// - gather.cpp: Gather, GatherAxis
// - scan.cpp: Scan
// - arange.cpp: Arange
// - rope.cpp: RoPE (no-op fallback)
// - fast.cpp: LayerNorm, RMSNorm, Quantize, ConvertFP8, CustomKernel, SDPA
// - random.cpp: RandomBits

// Load primitive - throw NYI like Metal backend
void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[Load::eval_gpu] Not implemented.");
}

// ArgPartition / ArgSort - implemented using bitonic argsort shader
void eval_argpartition_or_argsort_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    int kth,
    Stream s) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Allocate output
  out.set_data(allocator::malloc(out.nbytes()));

  // Convert non-float32 inputs to float32 for the argsort_f32 shader.
  // The indices produced by the sort are type-agnostic (just positions).
  // Complex types are not supported since their ordering is not well-defined.
  if (issubdtype(in.dtype(), complexfloating)) {
    throw std::runtime_error(
        "ArgPartition/ArgSort Vulkan does not support complex inputs.");
  }
  if (in.dtype() != float32) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  if (out.dtype() != uint32) {
    throw std::runtime_error(
        "ArgPartition/ArgSort output must be uint32.");
  }

  // Normalize axis
  axis = normalize_axis(axis, in.ndim());

  // Ensure sort axis is the last dimension
  array in_kernel = in;
  if (axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, axis, in.ndim() - 1);
  }

  // Make input contiguous with row-contiguous layout
  if (!in_kernel.flags().row_contiguous || in_kernel.offset() != 0 ||
      !is_supported_unary_layout(in_kernel)) {
    in_kernel = contiguous_copy_gpu(in_kernel, s);
  }

  const uint32_t ncols = static_cast<uint32_t>(in_kernel.shape().back());
  if (ncols > 262144) {
    throw std::runtime_error(
        "ArgPartition/ArgSort Vulkan requires sort axis <= 262144 elements.");
  }

  const int normalized_kth = kth < 0 ? static_cast<int>(ncols) + kth : kth;
  const bool topk_suffix_partition = kth < 0 && normalized_kth >= 0 &&
      normalized_kth >= static_cast<int>(ncols) - 16 && axis == in.ndim() - 1 &&
      in_kernel.dtype() == float32 && ncols == 256;

  // Prepare output in kernel layout (same shape as in_kernel, dtype uint32)
  array kernel_out(in_kernel.shape(), uint32, nullptr, {});
  const bool staged_output = !kernel_out.flags().row_contiguous ||
      kernel_out.offset() != 0 || !is_supported_unary_layout(kernel_out);
  array out_work = staged_output
      ? array(kernel_out.shape(), kernel_out.dtype(), nullptr, {})
      : kernel_out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu_inplace(out_work, kernel_out, CopyType::Vector, s);
    }
    if (axis != in.ndim() - 1) {
      array restored = swapaxes_in_eval(kernel_out, in.ndim() - 1, axis);
      copy_gpu(restored, out, CopyType::GeneralGeneral, s);
    } else {
      copy_gpu(kernel_out, out, CopyType::GeneralGeneral, s);
    }
    return;
  }

  auto command_buffer = vulkan::begin_command_recording(s.index);
  vulkan::dispatch_argsort_op(
      in_kernel,
      out_work,
      vulkan::StaticShaderId::argsort_partition_f32,
      command_buffer,
      s,
      topk_suffix_partition ? static_cast<uint32_t>(kth) : 0u);
  vulkan::end_command_recording(s.index);

  if (staged_output) {
    copy_gpu_inplace(out_work, kernel_out, CopyType::Vector, s);
  }

  if (axis != in.ndim() - 1) {
    array restored = swapaxes_in_eval(kernel_out, in.ndim() - 1, axis);
    copy_gpu(restored, out, CopyType::GeneralGeneral, s);
  } else {
    copy_gpu(kernel_out, out, CopyType::GeneralGeneral, s);
  }
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [kth, axis] = state();
  eval_argpartition_or_argsort_gpu(inputs, out, axis, kth, stream());
}

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval_argpartition_or_argsort_gpu(inputs, out, axis_, 0, stream());
}

// CPU fallbacks for primitives not implemented on Vulkan
NYI_OP(ArcCos)
NYI_OP(ArcCosh)
NYI_OP(ArcSin)
NYI_OP(ArcSinh)
NYI_OP(ArcTan)
NYI_OP(ArcTan2)
NYI_OP(ArcTanh)
NYI_OP(BitwiseInvert)
NYI_OP(Cosh)
NYI_OP_MULTI(DivMod)
NYI_OP(Remainder)
NYI_OP(Expm1)
NYI_OP_STATE(FFT)
NYI_OP_STATE(GatherMM)
NYI_OP_STATE(Hadamard)
NYI_OP(Imag)
NYI_OP(Log1p)
NYI_OP(LogAddExp)
// Linear algebra operations - throw NYI like Metal backend
NO_GPU_MULTI(LUF)
NO_GPU_MULTI(QRF)
NO_GPU_STATE(Inverse)
NO_GPU_STATE(Cholesky)
NO_GPU_MULTI_STATE(Eigh)
NO_GPU_MULTI_STATE(Eig)
NO_GPU_MULTI_STATE(SVD)

NYI_OP_STATE(Partition)
void Power::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_power_vulkan(inputs, out, stream())) {
    throw std::runtime_error("Power has no Vulkan implementation for this input.");
  }
}
// QuantizedMatmul and QQMatmul are implemented in quantized.cpp.

NYI_OP(Real)
NYI_OP(Sinh)
NYI_OP_STATE(Sort)
NYI_OP(Tan)
NO_GPU(MaskedScatter)
// Scatter and ScatterAxis are implemented in scatter.cpp

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  if (reduce_type_ != SliceUpdate::None) {
    throw std::runtime_error(
        "[SliceUpdate::eval_gpu] Vulkan only supports SliceUpdate::None. "
        "Reduce operations (Sum, Prod, Min, Max) are not yet implemented.");
  }

  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  ctype = in.data_size() == 1 ? CopyType::Scalar : ctype;
  out.set_data(allocator::malloc(ctype == CopyType::Vector
                   ? in.data_size() * out.itemsize()
                   : out.nbytes()));
  copy_gpu_inplace(in, out, ctype, stream());

  auto [data_offset, out_strides] =
      prepare_slice(out, start_indices_, strides_);
  copy_gpu_inplace(
      upd,
      out,
      upd.shape(),
      upd.strides(),
      out_strides,
      0,
      data_offset,
      CopyType::GeneralGeneral,
      stream());
}

NYI_OP(SegmentedMM)

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_select_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "Select has no Vulkan implementation for this input.");
  }
}

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
NO_GPU_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core
