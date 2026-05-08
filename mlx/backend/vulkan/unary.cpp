// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"

namespace mlx::core {

namespace {

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

bool is_vulkan_signed_integer_dtype(Dtype dtype) {
  switch (dtype) {
    case int8:
    case int16:
    case int32:
    case int64:
      return true;
    default:
      return false;
  }
}

std::string unary_glsl_storage_type(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return "uint8_t";
    case uint8:
      return "uint8_t";
    case uint16:
      return "uint16_t";
    case uint32:
      return "uint";
    case uint64:
      return "uint64_t";
    case int8:
      return "int8_t";
    case int16:
      return "int16_t";
    case int32:
      return "int";
    case int64:
      return "int64_t";
    default:
      throw std::runtime_error("Unsupported dtype for Vulkan unary shader.");
  }
}

std::string unary_zero_literal(Dtype dtype) {
  switch (dtype) {
    case bool_:
    case uint8:
      return "uint8_t(0)";
    case uint16:
      return "uint16_t(0)";
    case uint32:
      return "uint(0)";
    case uint64:
      return "uint64_t(0)";
    case int8:
      return "int8_t(0)";
    case int16:
      return "int16_t(0)";
    case int32:
      return "int(0)";
    case int64:
      return "int64_t(0)";
    default:
      throw std::runtime_error("Unsupported dtype for Vulkan unary zero.");
  }
}

std::string unary_one_literal(Dtype dtype) {
  switch (dtype) {
    case bool_:
    case uint8:
      return "uint8_t(1)";
    case uint16:
      return "uint16_t(1)";
    case uint32:
      return "uint(1)";
    case uint64:
      return "uint64_t(1)";
    case int8:
      return "int8_t(1)";
    case int16:
      return "int16_t(1)";
    case int32:
      return "int(1)";
    case int64:
      return "int64_t(1)";
    default:
      throw std::runtime_error("Unsupported dtype for Vulkan unary one.");
  }
}

std::string unary_neg_one_literal(Dtype dtype) {
  switch (dtype) {
    case int8:
      return "int8_t(-1)";
    case int16:
      return "int16_t(-1)";
    case int32:
      return "int(-1)";
    case int64:
      return "int64_t(-1)";
    default:
      throw std::runtime_error("Unsupported dtype for Vulkan unary neg one.");
  }
}

std::string build_integer_sign_shader(Dtype dtype) {
  const auto type = unary_glsl_storage_type(dtype);
  const auto zero = unary_zero_literal(dtype);
  const auto one = unary_one_literal(dtype);
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(dtype, dtype, dtype == int64 || dtype == uint64);
  os << "layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer Input {" << type << " data[];} in_buf;\n";
  os << "layout(set = 0, binding = 1) buffer Output {" << type << " data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  " << type << " x = in_buf.data[idx + pc.in_offset];\n";
  if (is_vulkan_signed_integer_dtype(dtype)) {
    const auto neg_one = unary_neg_one_literal(dtype);
    os << "  out_buf.data[idx + pc.out_offset] = x > " << zero << " ? " << one
       << " : (x < " << zero << " ? " << neg_one << " : " << zero << ");\n";
  } else {
    os << "  out_buf.data[idx + pc.out_offset] = x == " << zero << " ? " << zero
       << " : " << one << ";\n";
  }
  os << "}\n";
  return os.str();
}

std::string build_complex_abs_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << R"(
layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;
layout(set = 0, binding = 0) readonly buffer Input {vec2 data[];} in_buf;
layout(set = 0, binding = 1) buffer Output {vec2 data[];} out_buf;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;
  vec2 x = in_buf.data[idx + pc.in_offset];
  precise float mag = sqrt(x.x * x.x + x.y * x.y);
  out_buf.data[idx + pc.out_offset] = vec2(mag, 0.0);
}
)";
  return os.str();
}

std::string build_complex_exp_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << R"(
layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;
layout(set = 0, binding = 0) readonly buffer Input {vec2 data[];} in_buf;
layout(set = 0, binding = 1) buffer Output {vec2 data[];} out_buf;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;
  vec2 z = in_buf.data[idx + pc.in_offset];
  if (isinf(z.x) && z.x < 0.0) {
    out_buf.data[idx + pc.out_offset] = vec2(0.0, 0.0);
    return;
  }
  float exp_x = exp(z.x);
  out_buf.data[idx + pc.out_offset] = vec2(exp_x * cos(z.y), exp_x * sin(z.y));
}
)";
  return os.str();
}

std::string build_complex_sin_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << R"(
layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;
layout(set = 0, binding = 0) readonly buffer Input {vec2 data[];} in_buf;
layout(set = 0, binding = 1) buffer Output {vec2 data[];} out_buf;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;
  vec2 z = in_buf.data[idx + pc.in_offset];
  out_buf.data[idx + pc.out_offset] = vec2(
      sin(z.x) * cosh(z.y),
      cos(z.x) * sinh(z.y));
}
)";
  return os.str();
}

std::string build_complex_cos_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << R"(
layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;
layout(set = 0, binding = 0) readonly buffer Input {vec2 data[];} in_buf;
layout(set = 0, binding = 1) buffer Output {vec2 data[];} out_buf;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;
  vec2 z = in_buf.data[idx + pc.in_offset];
  out_buf.data[idx + pc.out_offset] = vec2(
      cos(z.x) * cosh(z.y),
      -sin(z.x) * sinh(z.y));
}
)";
  return os.str();
}

std::string build_expm1_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(float32, float32, false);
  os << R"(
layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;
layout(set = 0, binding = 0) readonly buffer Input {float data[];} in_buf;
layout(set = 0, binding = 1) buffer Output {float data[];} out_buf;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= pc.total_elements) return;
  float x = in_buf.data[idx + pc.in_offset];
  out_buf.data[idx + pc.out_offset] = exp(x) - 1.0;
}
)";
  return os.str();
}

std::string build_log_base_shader(Log::Base base) {
  const char* expr = nullptr;
  switch (base) {
    case Log::e:
      expr = "log(x)";
      break;
    case Log::two:
      expr = "log2(x)";
      break;
    case Log::ten:
      expr = "log(x) * 0.4342944819032518";
      break;
  }

  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(float32, float32, false);
  os << "layout(push_constant) uniform PushConstants { uint in_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer Input {float data[];} in_buf;\n";
  os << "layout(set = 0, binding = 1) buffer Output {float data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  float x = in_buf.data[idx + pc.in_offset];\n";
  os << "  out_buf.data[idx + pc.out_offset] = " << expr << ";\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_unary_shader_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::string& shader_name,
    const std::string& shader_source,
    Stream s) {
  if (inputs.size() != 1 || inputs[0].dtype() != complex64 ||
      out.dtype() != complex64) {
    return false;
  }

  array in = inputs[0];
  if (!is_supported_elementwise_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }
  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (!is_supported_elementwise_layout(in) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto in_offset =
      static_cast<uint64_t>(in.offset() / size_of(in.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  struct PushConstants {
    uint32_t in_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(in_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };

  vulkan::DynamicArrayRef arrays[] = {{&in, 0}, {&out_work, 1}};
  constexpr uint32_t kPushConstantSize = sizeof(PushConstants);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      shader_source,
      2,
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
  vkCmdDispatch(
      dispatch.command_buffer,
      (static_cast<uint32_t>(total) + 255u) / 256u,
      1,
      1);
  vulkan::end_command_recording(s.index);
  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_expm1_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 1 ||
      !(out.dtype() == float32 || out.dtype() == float16 || out.dtype() == bfloat16)) {
    return false;
  }

  array in = inputs[0];
  array in_f32(in.shape(), float32, nullptr, {});
  copy_gpu(in, in_f32, CopyType::General, s);
  in = in_f32;

  array out_target = out.dtype() == float32 ? out : array(out.shape(), float32, nullptr, {});
  if (!is_supported_elementwise_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }
  const bool staged_output = !is_supported_elementwise_layout(out_target);
  array out_work = staged_output
      ? array(out_target.shape(), out_target.dtype(), nullptr, {})
      : out_target;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (!is_supported_elementwise_layout(in) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out_target, CopyType::General, s);
    }
    if (out_target.id() != out.id()) {
      copy_gpu(out_target, out, CopyType::General, s);
    }
    return true;
  }

  const auto in_offset =
      static_cast<uint64_t>(in.offset() / size_of(in.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  struct PushConstants {
    uint32_t in_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(in_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };

  vulkan::DynamicArrayRef arrays[] = {{&in, 0}, {&out_work, 1}};
  constexpr uint32_t kPushConstantSize = sizeof(PushConstants);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_expm1_f32",
      build_expm1_shader(),
      2,
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
  vkCmdDispatch(
      dispatch.command_buffer,
      (static_cast<uint32_t>(total) + 255u) / 256u,
      1,
      1);
  vulkan::end_command_recording(s.index);
  if (staged_output) {
    copy_gpu(out_work, out_target, CopyType::General, s);
  }
  if (out_target.id() != out.id()) {
    copy_gpu(out_target, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_log_base_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Log::Base base,
    Stream s) {
  if (inputs.size() != 1 ||
      !(out.dtype() == float32 || out.dtype() == float16 || out.dtype() == bfloat16)) {
    return false;
  }

  array in = inputs[0];
  array in_f32(in.shape(), float32, nullptr, {});
  copy_gpu(in, in_f32, CopyType::General, s);
  in = in_f32;

  array out_target = out.dtype() == float32 ? out : array(out.shape(), float32, nullptr, {});
  if (!is_supported_elementwise_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }
  const bool staged_output = !is_supported_elementwise_layout(out_target);
  array out_work = staged_output
      ? array(out_target.shape(), out_target.dtype(), nullptr, {})
      : out_target;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (!is_supported_elementwise_layout(in) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out_target, CopyType::General, s);
    }
    if (out_target.id() != out.id()) {
      copy_gpu(out_target, out, CopyType::General, s);
    }
    return true;
  }

  const auto in_offset =
      static_cast<uint64_t>(in.offset() / size_of(in.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  struct PushConstants {
    uint32_t in_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(in_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };

  vulkan::DynamicArrayRef arrays[] = {{&in, 0}, {&out_work, 1}};
  constexpr uint32_t kPushConstantSize = sizeof(PushConstants);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      base == Log::two ? "dynamic_log2_f32"
      : base == Log::ten ? "dynamic_log10_f32"
                         : "dynamic_log_f32",
      build_log_base_shader(base),
      2,
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
  vkCmdDispatch(
      dispatch.command_buffer,
      (static_cast<uint32_t>(total) + 255u) / 256u,
      1,
      1);
  vulkan::end_command_recording(s.index);
  if (staged_output) {
    copy_gpu(out_work, out_target, CopyType::General, s);
  }
  if (out_target.id() != out.id()) {
    copy_gpu(out_target, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_f32_unary_with_staging_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
    Stream s) {
  if (inputs.size() != 1 ||
      !(out.dtype() == float32 || out.dtype() == float16 || out.dtype() == bfloat16)) {
    return false;
  }

  array in = inputs[0];
  array in_f32(in.shape(), float32, nullptr, {});
  copy_gpu(in, in_f32, CopyType::General, s);
  in = in_f32;

  array out_target = out.dtype() == float32 ? out : array(out.shape(), float32, nullptr, {});
  if (!is_supported_generic_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }
  const bool staged_output = !is_supported_generic_unary_layout(out_target);
  array out_work = staged_output
      ? array(out_target.shape(), out_target.dtype(), nullptr, {})
      : out_target;

  set_unary_output_data(in, out_work);
  if (!is_supported_generic_unary_layout(in) ||
      !is_supported_generic_unary_layout(out_work)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out_target, CopyType::General, s);
    }
    if (out_target.id() != out.id()) {
      copy_gpu(out_target, out, CopyType::General, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_unary_op(in, out_work, shader_id, command_buffer, s);
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_work, out_target, CopyType::General, s);
    }
    if (out_target.id() != out.id()) {
      copy_gpu(out_target, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

bool try_eval_complex_abs_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 1 || inputs[0].dtype() != complex64 ||
      out.dtype() != complex64) {
    return false;
  }

  array in = inputs[0];
  if (!is_supported_elementwise_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }
  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (!is_supported_elementwise_layout(in) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto in_offset =
      static_cast<uint64_t>(in.offset() / size_of(in.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  struct PushConstants {
    uint32_t in_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(in_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };

  vulkan::DynamicArrayRef arrays[] = {{&in, 0}, {&out_work, 1}};
  constexpr uint32_t kPushConstantSize = sizeof(PushConstants);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_abs_c64",
      build_complex_abs_shader(),
      2,
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
  vkCmdDispatch(
      dispatch.command_buffer,
      (static_cast<uint32_t>(total) + 255u) / 256u,
      1,
      1);
  vulkan::end_command_recording(s.index);
  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool try_eval_integer_sign_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 1 || inputs[0].dtype() != out.dtype() ||
      !is_vulkan_integer_dtype(inputs[0].dtype())) {
    return false;
  }

  array in = inputs[0];
  if (!is_supported_elementwise_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }
  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (!is_supported_elementwise_layout(in) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto in_offset =
      static_cast<uint64_t>(in.offset() / size_of(in.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  struct PushConstants {
    uint32_t in_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(in_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };

  vulkan::DynamicArrayRef arrays[] = {{&in, 0}, {&out_work, 1}};
  constexpr uint32_t kPushConstantSize = sizeof(PushConstants);
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_sign_" + dtype_suffix(in.dtype()),
      build_integer_sign_shader(in.dtype()),
      2,
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
  vkCmdDispatch(
      dispatch.command_buffer,
      (static_cast<uint32_t>(total) + 255u) / 256u,
      1,
      1);
  vulkan::end_command_recording(s.index);
  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

template <typename Primitive>
bool try_eval_unary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f) {
  if (inputs.size() != 1) {
    return false;
  }

  array in = inputs[0];
  const bool complex_io = in.dtype() == complex64 && out.dtype() == complex64;
  if ((!is_vulkan_float_dtype(in.dtype()) && !is_vulkan_integer_dtype(in.dtype()) &&
       !complex_io) ||
      in.dtype() != out.dtype()) {
    return false;
  }

  const bool use_f32_staging_io =
      !complex_io && (in.dtype() == bfloat16 || out.dtype() == bfloat16);
  if (use_f32_staging_io) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  if (!is_supported_generic_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  const bool staged_output =
      use_f32_staging_io || !is_supported_generic_unary_layout(out);
  array out_work = staged_output
      ? array(
            out.shape(),
            use_f32_staging_io ? float32 : out.dtype(),
            nullptr,
            {})
      : out;

  set_unary_output_data(in, out_work);
  if (!is_supported_generic_unary_layout(in) ||
      !is_supported_generic_unary_layout(out_work)) {
    return false;
  }

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_unary_op(
        in, out_work, shader_id, command_buffer, s, param1, param2);
    vulkan::end_command_recording(s.index);
    if (staged_output || use_f32_staging_io) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "unary_dispatch_failed shader="
          << vulkan::static_shader_name(shader_id) << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_unary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f) {
  if (!try_eval_unary_op_vulkan<Primitive>(
          inputs, out, shader_id, s, param1, param2)) {
    throw std::runtime_error(
        std::string("Unary operation ") +
        vulkan::static_shader_name(shader_id) +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

template <typename Primitive>
bool try_eval_generic_unary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f) {
  if (inputs.size() != 1) {
    return false;
  }

  array in = inputs[0];
  const bool complex_io = in.dtype() == complex64 && out.dtype() == complex64;
  if ((!is_vulkan_float_dtype(in.dtype()) && !is_vulkan_integer_dtype(in.dtype()) &&
       !complex_io) ||
      in.dtype() != out.dtype()) {
    return false;
  }

  const bool use_f32_staging_io =
      in.dtype() == bfloat16 || out.dtype() == bfloat16;
  if (use_f32_staging_io) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  if (!is_supported_generic_unary_layout(in)) {
    in = contiguous_copy_gpu(in, s);
  }

  const bool staged_output =
      use_f32_staging_io || !is_supported_generic_unary_layout(out);
  array out_work = staged_output
      ? array(
            out.shape(),
            use_f32_staging_io ? float32 : out.dtype(),
            nullptr,
            {})
      : out;

  set_unary_output_data(in, out_work);
  if (!is_supported_generic_unary_layout(in) ||
      !is_supported_generic_unary_layout(out_work)) {
    return false;
  }

  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_generic_unary_op(
        in,
        out_work,
        shader_id,
        command_buffer,
        s,
        param1,
        param2,
        param3,
        param4);
    vulkan::end_command_recording(s.index);
    if (staged_output || use_f32_staging_io) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "generic_unary_dispatch_failed shader="
          << vulkan::static_shader_name(shader_id) << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_generic_unary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    vulkan::StaticShaderId shader_id,
    Stream s,
    float param1 = 0.0f,
    float param2 = 0.0f,
    float param3 = 0.0f,
    float param4 = 0.0f) {
  if (!try_eval_generic_unary_op_vulkan<Primitive>(
          inputs, out, shader_id, s, param1, param2, param3, param4)) {
    throw std::runtime_error(
        std::string("Unary operation ") +
        vulkan::static_shader_name(shader_id) +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

template <typename Primitive>
void eval_generic_unary_suffix_vulkan(
    const std::vector<array>& inputs,
    array& out,
    GenericUnaryShaderOp op,
    Stream s,
    bool f16_with_rte = false) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    Dtype shader_dtype = out.dtype() == bfloat16 ? float32 : out.dtype();
    auto shader_id = generic_unary_shader_id(
        op, shader_dtype, f16_with_rte && out.dtype() == float16);
    if (shader_id.has_value()) {
      eval_generic_unary_vulkan<Primitive>(inputs, out, *shader_id, s);
      return;
    }
  }
  throw std::runtime_error(
      "Unary operation failed on Vulkan (unsupported dtype or layout).");
}

} // namespace

#define VULKAN_GENERIC_UNARY_GPU(func, op_name)                             \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) {       \
    eval_generic_unary_suffix_vulkan<func>(inputs, out, op_name, stream()); \
  }

#define VULKAN_GENERIC_UNARY_RTE_GPU(func, op_name)                   \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_generic_unary_suffix_vulkan<func>(                           \
        inputs, out, op_name, stream(), true);                        \
  }

// Generic unary ops
void Abs::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_complex_abs_vulkan(inputs, out, stream())) {
    return;
  }
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype() &&
      (issubdtype(inputs[0].dtype(), unsignedinteger) ||
       inputs[0].dtype() == bool_)) {
    copy_gpu(inputs[0], out, CopyType::General, stream());
    return;
  }
  eval_generic_unary_suffix_vulkan<Abs>(
      inputs, out, GenericUnaryShaderOp::Abs, stream());
}
VULKAN_GENERIC_UNARY_GPU(Ceil, GenericUnaryShaderOp::Ceil)
VULKAN_GENERIC_UNARY_GPU(Floor, GenericUnaryShaderOp::Floor)
VULKAN_GENERIC_UNARY_GPU(Negative, GenericUnaryShaderOp::Negative)
VULKAN_GENERIC_UNARY_GPU(Round, GenericUnaryShaderOp::Round)
VULKAN_GENERIC_UNARY_GPU(Sigmoid, GenericUnaryShaderOp::Sigmoid)
VULKAN_GENERIC_UNARY_GPU(Tanh, GenericUnaryShaderOp::Tanh)

void Sign::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype() &&
      inputs[0].dtype() == bool_) {
    copy_gpu(inputs[0], out, CopyType::General, stream());
    return;
  }
  if (try_eval_integer_sign_vulkan(inputs, out, stream())) {
    return;
  }
  eval_generic_unary_suffix_vulkan<Sign>(
      inputs, out, GenericUnaryShaderOp::Sign, stream());
}

// Specialized unary ops
void Conjugate::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto shader_id = unary_shader_id(UnaryShaderOp::Conjugate, out.dtype());
    if (shader_id.has_value()) {
      eval_unary_vulkan<Conjugate>(inputs, out, *shader_id, stream());
      return;
    }
  }
  throw std::runtime_error(
      "Conjugate operation failed on Vulkan (unsupported dtype or layout).");
}

void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_complex_unary_shader_vulkan(
          inputs, out, "dynamic_cos_c64", build_complex_cos_shader(), stream())) {
    return;
  }
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Cos>(
            inputs, out, vulkan::StaticShaderId::cos_f32, stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Cos operation failed on Vulkan (unsupported dtype or layout).");
}

void Erf::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_f32_unary_with_staging_vulkan(
          inputs, out, vulkan::StaticShaderId::erf_f32, stream())) {
    return;
  }
  throw std::runtime_error(
      "Erf operation failed on Vulkan (unsupported dtype or layout).");
}

void ErfInv::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_f32_unary_with_staging_vulkan(
          inputs, out, vulkan::StaticShaderId::erfinv_f32, stream())) {
    return;
  }
  throw std::runtime_error(
      "ErfInv operation failed on Vulkan (unsupported dtype or layout).");
}

void Exp::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_complex_unary_shader_vulkan(
          inputs, out, "dynamic_exp_c64", build_complex_exp_shader(), stream())) {
    return;
  }
  eval_generic_unary_suffix_vulkan<Exp>(
      inputs, out, GenericUnaryShaderOp::Exp, stream(), true);
}

void Expm1::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_expm1_vulkan(inputs, out, stream())) {
    return;
  }
  throw std::runtime_error("Expm1 has no Vulkan implementation.");
}

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (state() == Log::e && inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto shader_id = unary_shader_id(UnaryShaderOp::Log, out.dtype());
    if (shader_id.has_value() &&
        try_eval_unary_op_vulkan<Log>(inputs, out, *shader_id, stream())) {
      return;
    }
  }
  if (try_eval_log_base_vulkan(inputs, out, state(), stream())) {
    return;
  }
  throw std::runtime_error(
      "Log operation failed on Vulkan (unsupported dtype or layout).");
}

void Log1p::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Log1p>(
            inputs, out, vulkan::StaticShaderId::log1p_f32, stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Log1p operation failed on Vulkan (unsupported dtype or layout).");
}

void Sin::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_complex_unary_shader_vulkan(
          inputs, out, "dynamic_sin_c64", build_complex_sin_shader(), stream())) {
    return;
  }
  if (inputs.size() == 1 && inputs[0].dtype() == float32 &&
      out.dtype() == float32) {
    if (try_eval_unary_op_vulkan<Sin>(
            inputs, out, vulkan::StaticShaderId::sin_f32, stream())) {
      return;
    }
  }
  throw std::runtime_error(
      "Sin operation failed on Vulkan (unsupported dtype or layout).");
}

void Square::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 && inputs[0].dtype() == out.dtype()) {
    auto shader_id = unary_shader_id(UnaryShaderOp::Square, out.dtype());
    if (shader_id.has_value()) {
      eval_unary_vulkan<Square>(inputs, out, *shader_id, stream());
      return;
    }
  }
  throw std::runtime_error(
      "Square operation failed on Vulkan (unsupported dtype or layout).");
}

void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.size() == 1 &&
      (inputs[0].dtype() == float32 || inputs[0].dtype() == bfloat16 ||
       inputs[0].dtype() == float16 || inputs[0].dtype() == complex64) &&
      out.dtype() == inputs[0].dtype()) {
    auto shader_id = unary_shader_id(
        state() ? UnaryShaderOp::Rsqrt : UnaryShaderOp::Sqrt, out.dtype());
    if (shader_id.has_value()) {
      eval_unary_vulkan<Sqrt>(inputs, out, *shader_id, stream(), 0.0f, 0.0f);
      return;
    }
  }
  throw std::runtime_error(
      "Sqrt operation failed on Vulkan (unsupported dtype or layout).");
}

} // namespace mlx::core
