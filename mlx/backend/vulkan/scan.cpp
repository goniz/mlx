// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

bool has_vulkan_buffer(const array& arr) {
  auto data = arr.data_shared_ptr();
  return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
}

const char* integral_scan_type(Dtype dtype) {
  switch (dtype) {
    case int32:
      return "int";
    case int64:
      return "int64_t";
    default:
      throw std::runtime_error("Unsupported integral scan dtype.");
  }
}

std::string build_integral_scan_shader(
    Dtype dtype,
    Scan::ReduceType reduce_type,
    bool reverse,
    bool inclusive) {
  const char* type_name = integral_scan_type(dtype);
  const bool prod = reduce_type == Scan::Prod;
  const bool min = reduce_type == Scan::Min;
  const bool max = reduce_type == Scan::Max;
  std::ostringstream os;
  os << "#version 450\n";
  if (dtype == int64) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  os << "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "    uint n_cols;\n";
  os << "    uint row_count;\n";
  os << "} p;\n\n";
  os << "layout(binding = 0) readonly buffer A { " << type_name << " data_a[]; };\n";
  os << "layout(binding = 1) writeonly buffer D { " << type_name << " data_d[]; };\n\n";
  os << "void main() {\n";
  os << "    uint row = gl_GlobalInvocationID.x;\n";
  os << "    if (row >= p.row_count) return;\n";
  os << "    uint base = row * p.n_cols;\n";
  os << "    " << type_name << " acc = " << type_name;
  if (prod) {
    os << "(1);\n";
  } else if (min) {
    os << (dtype == int64 ? "(9223372036854775807L);\n" : "(2147483647);\n");
  } else if (max) {
    os << (dtype == int64 ? "(-9223372036854775807L - 1L);\n" : "(-2147483647 - 1);\n");
  } else {
    os << "(0);\n";
  }
  os << "    for (uint logical_col = 0; logical_col < p.n_cols; ++logical_col) {\n";
  if (reverse) {
    os << "        uint col = p.n_cols - 1 - logical_col;\n";
  } else {
    os << "        uint col = logical_col;\n";
  }
  os << "        " << type_name << " value = data_a[base + col];\n";
  if (inclusive) {
    if (min) {
      os << "        acc = min(acc, value);\n";
    } else if (max) {
      os << "        acc = max(acc, value);\n";
    } else {
      os << "        acc " << (prod ? "*= " : "+= ") << "value;\n";
    }
    os << "        data_d[base + col] = acc;\n";
  } else {
    os << "        data_d[base + col] = acc;\n";
    if (min) {
      os << "        acc = min(acc, value);\n";
    } else if (max) {
      os << "        acc = max(acc, value);\n";
    } else {
      os << "        acc " << (prod ? "*= " : "+= ") << "value;\n";
    }
  }
  os << "    }\n";
  os << "}\n";
  return os.str();
}

std::string build_logaddexp_scan_shader(bool reverse, bool inclusive) {
  std::ostringstream os;
  os << "#version 450\n\n";
  os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "    uint n_cols;\n";
  os << "    uint row_count;\n";
  os << "} p;\n\n";
  os << "layout(binding = 0) readonly buffer A { float data_a[]; };\n";
  os << "layout(binding = 1) writeonly buffer D { float data_d[]; };\n\n";
  os << "float logaddexp_pair(float a, float b) {\n";
  os << "    float maxval = max(a, b);\n";
  os << "    float minval = min(a, b);\n";
  os << "    if (isinf(maxval) || isinf(minval)) return maxval;\n";
  os << "    return maxval + log(1.0 + exp(minval - maxval));\n";
  os << "}\n\n";
  os << "void main() {\n";
  os << "    uint row = gl_GlobalInvocationID.x;\n";
  os << "    if (row >= p.row_count) return;\n";
  os << "    uint base = row * p.n_cols;\n";
  os << "    float acc = -1.0 / 0.0;\n";
  os << "    for (uint logical_col = 0; logical_col < p.n_cols; ++logical_col) {\n";
  if (reverse) {
    os << "        uint col = p.n_cols - 1 - logical_col;\n";
  } else {
    os << "        uint col = logical_col;\n";
  }
  os << "        float value = data_a[base + col];\n";
  if (inclusive) {
    os << "        acc = logaddexp_pair(acc, value);\n";
    os << "        data_d[base + col] = acc;\n";
  } else {
    os << "        data_d[base + col] = acc;\n";
    os << "        acc = logaddexp_pair(acc, value);\n";
  }
  os << "    }\n";
  os << "}\n";
  return os.str();
}

std::string build_complex_logaddexp_scan_shader(bool reverse, bool inclusive) {
  std::ostringstream os;
  os << "#version 450\n\n";
  os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "    uint n_cols;\n";
  os << "    uint row_count;\n";
  os << "} p;\n\n";
  os << "layout(binding = 0) readonly buffer A { vec2 data_a[]; };\n";
  os << "layout(binding = 1) writeonly buffer D { vec2 data_d[]; };\n\n";
  os << "vec2 clogaddexp(vec2 a, vec2 b) {\n";
  os << "    vec2 maxv = (a.x > b.x || (a.x == b.x && a.y >= b.y)) ? a : b;\n";
  os << "    vec2 minv = (maxv.x == a.x && maxv.y == a.y) ? b : a;\n";
  os << "    float expx = exp(minv.x - maxv.x);\n";
  os << "    vec2 delta = vec2(expx * cos(minv.y - maxv.y), expx * sin(minv.y - maxv.y));\n";
  os << "    vec2 one_plus = vec2(1.0 + delta.x, delta.y);\n";
  os << "    return vec2(maxv.x + log(length(one_plus)), maxv.y + atan(one_plus.y, one_plus.x));\n";
  os << "}\n\n";
  os << "void main() {\n";
  os << "    uint row = gl_GlobalInvocationID.x;\n";
  os << "    if (row >= p.row_count) return;\n";
  os << "    uint base = row * p.n_cols;\n";
  os << "    vec2 acc = data_a[base + " << (reverse ? "(p.n_cols - 1u)" : "0u") << "] * 0.0 + vec2(-1.0 / 0.0, 0.0);\n";
  os << "    for (uint logical_col = 0; logical_col < p.n_cols; ++logical_col) {\n";
  if (reverse) {
    os << "        uint col = p.n_cols - 1 - logical_col;\n";
  } else {
    os << "        uint col = logical_col;\n";
  }
  os << "        vec2 value = data_a[base + col];\n";
  if (inclusive) {
    os << "        acc = clogaddexp(acc, value);\n";
    os << "        data_d[base + col] = acc;\n";
  } else {
    os << "        data_d[base + col] = acc;\n";
    os << "        acc = clogaddexp(acc, value);\n";
  }
  os << "    }\n";
  os << "}\n";
  return os.str();
}

std::string build_complex_prod_scan_shader(bool reverse, bool inclusive) {
  std::ostringstream os;
  os << "#version 450\n\n";
  os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "    uint n_cols;\n";
  os << "    uint row_count;\n";
  os << "} p;\n\n";
  os << "layout(binding = 0) readonly buffer A { vec2 data_a[]; };\n";
  os << "layout(binding = 1) writeonly buffer D { vec2 data_d[]; };\n\n";
  os << "vec2 cmul(vec2 a, vec2 b) {\n";
  os << "    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n";
  os << "}\n\n";
  os << "void main() {\n";
  os << "    uint row = gl_GlobalInvocationID.x;\n";
  os << "    if (row >= p.row_count) return;\n";
  os << "    uint base = row * p.n_cols;\n";
  os << "    vec2 acc = vec2(1.0, 0.0);\n";
  os << "    for (uint logical_col = 0; logical_col < p.n_cols; ++logical_col) {\n";
  if (reverse) {
    os << "        uint col = p.n_cols - 1 - logical_col;\n";
  } else {
    os << "        uint col = logical_col;\n";
  }
  os << "        vec2 value = data_a[base + col];\n";
  if (inclusive) {
    os << "        acc = cmul(acc, value);\n";
    os << "        data_d[base + col] = acc;\n";
  } else {
    os << "        data_d[base + col] = acc;\n";
    os << "        acc = cmul(acc, value);\n";
  }
  os << "    }\n";
  os << "}\n";
  return os.str();
}

std::string build_complex_sum_scan_shader(bool reverse, bool inclusive) {
  std::ostringstream os;
  os << "#version 450\n\n";
  os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "    uint n_cols;\n";
  os << "    uint row_count;\n";
  os << "} p;\n\n";
  os << "layout(binding = 0) readonly buffer A { vec2 data_a[]; };\n";
  os << "layout(binding = 1) writeonly buffer D { vec2 data_d[]; };\n\n";
  os << "void main() {\n";
  os << "    uint row = gl_GlobalInvocationID.x;\n";
  os << "    if (row >= p.row_count) return;\n";
  os << "    uint base = row * p.n_cols;\n";
  os << "    vec2 acc = vec2(0.0, 0.0);\n";
  os << "    for (uint logical_col = 0; logical_col < p.n_cols; ++logical_col) {\n";
  if (reverse) {
    os << "        uint col = p.n_cols - 1 - logical_col;\n";
  } else {
    os << "        uint col = logical_col;\n";
  }
  if (inclusive) {
    os << "        acc += data_a[base + col];\n";
    os << "        data_d[base + col] = acc;\n";
  } else {
    os << "        data_d[base + col] = acc;\n";
    os << "        acc += data_a[base + col];\n";
  }
  os << "    }\n";
  os << "}\n";
  return os.str();
}

std::string build_prod_scan_shader(bool reverse, bool inclusive) {
  std::ostringstream os;
  os << "#version 450\n\n";
  os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  os << "layout(push_constant) uniform Params {\n";
  os << "    uint n_cols;\n";
  os << "    uint row_count;\n";
  os << "} p;\n\n";
  os << "layout(binding = 0) readonly buffer A { float data_a[]; };\n";
  os << "layout(binding = 1) writeonly buffer D { float data_d[]; };\n\n";
  os << "void main() {\n";
  os << "    uint row = gl_GlobalInvocationID.x;\n";
  os << "    if (row >= p.row_count) return;\n";
  os << "    uint base = row * p.n_cols;\n";
  os << "    float acc = 1.0;\n";
  os << "    for (uint logical_col = 0; logical_col < p.n_cols; ++logical_col) {\n";
  if (reverse) {
    os << "        uint col = p.n_cols - 1 - logical_col;\n";
  } else {
    os << "        uint col = logical_col;\n";
  }
  if (inclusive) {
    os << "        acc *= data_a[base + col];\n";
    os << "        data_d[base + col] = acc;\n";
  } else {
    os << "        data_d[base + col] = acc;\n";
    os << "        acc *= data_a[base + col];\n";
  }
  os << "    }\n";
  os << "}\n";
  return os.str();
}

bool try_dispatch_complex_prod_scan(
    const array& in,
    array& out,
    Stream s,
    bool reverse,
    bool inclusive) {
  if (in.dtype() != complex64 || out.dtype() != complex64 || in.ndim() == 0 ||
      in.shape(in.ndim() - 1) == 0 || in.size() != out.size()) {
    return false;
  }

  const uint32_t n_cols =
      checked_u32_size(in.shape(in.ndim() - 1), "complex_prod_scan n_cols");
  const uint32_t total = checked_u32_size(out.size(), "complex_prod_scan size");
  if (total % n_cols != 0) {
    return false;
  }
  const uint32_t row_count = total / n_cols;
  if (row_count == 0) {
    return true;
  }

  const std::string shader_name = std::string("dynamic_cumprod_c64_") +
      (reverse ? "rev_" : "fwd_") + (inclusive ? "inc" : "exc");
  const std::string glsl = build_complex_prod_scan_shader(reverse, inclusive);
  const std::array<uint32_t, 2> pc = {n_cols, row_count};
  const std::array<vulkan::DynamicArrayRef, 2> refs = {{{&in, 0}, {&out, 1}}};

  try {
    auto dispatch = vulkan::dispatch_dynamic_compute_begin(
        shader_name,
        glsl,
        static_cast<uint32_t>(refs.size()),
        refs.data(),
        static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
        s);
    vkCmdPushConstants(
        dispatch.command_buffer,
        dispatch.pipeline->layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
        pc.data());
    vkCmdDispatch(dispatch.command_buffer, (row_count + 255u) / 256u, 1, 1);
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "complex_prod_scan_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_dispatch_complex_sum_scan(
    const array& in,
    array& out,
    Stream s,
    bool reverse,
    bool inclusive) {
  if (in.dtype() != complex64 || out.dtype() != complex64 || in.ndim() == 0 ||
      in.shape(in.ndim() - 1) == 0 || in.size() != out.size()) {
    return false;
  }

  const uint32_t n_cols =
      checked_u32_size(in.shape(in.ndim() - 1), "complex_sum_scan n_cols");
  const uint32_t total = checked_u32_size(out.size(), "complex_sum_scan size");
  if (total % n_cols != 0) {
    return false;
  }
  const uint32_t row_count = total / n_cols;
  if (row_count == 0) {
    return true;
  }

  const std::string shader_name = std::string("dynamic_cumsum_c64_") +
      (reverse ? "rev_" : "fwd_") + (inclusive ? "inc" : "exc");
  const std::string glsl = build_complex_sum_scan_shader(reverse, inclusive);
  const std::array<uint32_t, 2> pc = {n_cols, row_count};
  const std::array<vulkan::DynamicArrayRef, 2> refs = {{{&in, 0}, {&out, 1}}};

  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      glsl,
      static_cast<uint32_t>(refs.size()),
      refs.data(),
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      pc.data());
  vkCmdDispatch(dispatch.command_buffer, (row_count + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool try_dispatch_prod_scan(
    const array& in,
    array& out,
    Stream s,
    bool reverse,
    bool inclusive) {
  if (in.dtype() != float32 || out.dtype() != float32 || in.ndim() == 0 ||
      in.shape(in.ndim() - 1) == 0 || in.size() != out.size()) {
    return false;
  }

  const uint32_t n_cols =
      checked_u32_size(in.shape(in.ndim() - 1), "prod_scan n_cols");
  const uint32_t total = checked_u32_size(out.size(), "prod_scan size");
  if (total % n_cols != 0) {
    return false;
  }
  const uint32_t row_count = total / n_cols;
  if (row_count == 0) {
    return true;
  }

  const std::string shader_name = std::string("dynamic_cumprod_f32_") +
      (reverse ? "rev_" : "fwd_") + (inclusive ? "inc" : "exc");
  const std::string glsl = build_prod_scan_shader(reverse, inclusive);
  const std::array<uint32_t, 2> pc = {n_cols, row_count};
  const std::array<vulkan::DynamicArrayRef, 2> refs = {{{&in, 0}, {&out, 1}}};

  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      glsl,
      static_cast<uint32_t>(refs.size()),
      refs.data(),
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      pc.data());
  vkCmdDispatch(dispatch.command_buffer, (row_count + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool try_dispatch_integral_scan(
    const array& in,
    array& out,
    Stream s,
    Scan::ReduceType reduce_type,
    bool reverse,
    bool inclusive) {
  if ((in.dtype() != int32 && in.dtype() != int64) || in.dtype() != out.dtype() ||
      in.ndim() == 0 || in.shape(in.ndim() - 1) == 0 ||
      in.size() != out.size() ||
      (reduce_type != Scan::Sum && reduce_type != Scan::Prod &&
       reduce_type != Scan::Min && reduce_type != Scan::Max)) {
    return false;
  }

  const uint32_t n_cols =
      checked_u32_size(in.shape(in.ndim() - 1), "integral_scan n_cols");
  const uint32_t total = checked_u32_size(out.size(), "integral_scan size");
  if (total % n_cols != 0) {
    return false;
  }
  const uint32_t row_count = total / n_cols;
  if (row_count == 0) {
    return true;
  }

  const std::string shader_name = std::string("dynamic_") +
      (reduce_type == Scan::Prod    ? "cumprod_"
           : reduce_type == Scan::Min ? "cummin_"
           : reduce_type == Scan::Max ? "cummax_"
                                      : "cumsum_") +
      (in.dtype() == int64 ? "i64_" : "i32_") +
      (reverse ? "rev_" : "fwd_") + (inclusive ? "inc" : "exc");
  const std::string glsl =
      build_integral_scan_shader(in.dtype(), reduce_type, reverse, inclusive);
  const std::array<uint32_t, 2> pc = {n_cols, row_count};
  const std::array<vulkan::DynamicArrayRef, 2> refs = {{{&in, 0}, {&out, 1}}};

  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      glsl,
      static_cast<uint32_t>(refs.size()),
      refs.data(),
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      pc.data());
  vkCmdDispatch(dispatch.command_buffer, (row_count + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool try_dispatch_logaddexp_scan(
    const array& in,
    array& out,
    Stream s,
    bool reverse,
    bool inclusive) {
  if (in.dtype() != float32 || out.dtype() != float32 || in.ndim() == 0 ||
      in.shape(in.ndim() - 1) == 0 || in.size() != out.size()) {
    return false;
  }

  const uint32_t n_cols =
      checked_u32_size(in.shape(in.ndim() - 1), "logaddexp_scan n_cols");
  const uint32_t total = checked_u32_size(out.size(), "logaddexp_scan size");
  if (total % n_cols != 0) {
    return false;
  }
  const uint32_t row_count = total / n_cols;
  if (row_count == 0) {
    return true;
  }

  const std::string shader_name = std::string("dynamic_cumlogaddexp_f32_") +
      (reverse ? "rev_" : "fwd_") + (inclusive ? "inc" : "exc");
  const std::string glsl = build_logaddexp_scan_shader(reverse, inclusive);
  const std::array<uint32_t, 2> pc = {n_cols, row_count};
  const std::array<vulkan::DynamicArrayRef, 2> refs = {{{&in, 0}, {&out, 1}}};

  try {
    auto dispatch = vulkan::dispatch_dynamic_compute_begin(
        shader_name,
        glsl,
        static_cast<uint32_t>(refs.size()),
        refs.data(),
        static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
        s);
    vkCmdPushConstants(
        dispatch.command_buffer,
        dispatch.pipeline->layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
        pc.data());
    vkCmdDispatch(dispatch.command_buffer, (row_count + 255u) / 256u, 1, 1);
    vulkan::end_command_recording(s.index);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "logaddexp_scan_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

bool try_dispatch_complex_logaddexp_scan(
    const array& in,
    array& out,
    Stream s,
    bool reverse,
    bool inclusive) {
  if (in.dtype() != complex64 || out.dtype() != complex64 || in.ndim() == 0 ||
      in.shape(in.ndim() - 1) == 0 || in.size() != out.size()) {
    return false;
  }

  const uint32_t n_cols = checked_u32_size(
      in.shape(in.ndim() - 1), "complex_logaddexp_scan n_cols");
  const uint32_t total =
      checked_u32_size(out.size(), "complex_logaddexp_scan size");
  if (total % n_cols != 0) {
    return false;
  }
  const uint32_t row_count = total / n_cols;
  if (row_count == 0) {
    return true;
  }

  const std::string shader_name = std::string("dynamic_cumlogaddexp_c64_") +
      (reverse ? "rev_" : "fwd_") + (inclusive ? "inc" : "exc");
  const std::string glsl = build_complex_logaddexp_scan_shader(reverse, inclusive);
  const std::array<uint32_t, 2> pc = {n_cols, row_count};
  const std::array<vulkan::DynamicArrayRef, 2> refs = {{{&in, 0}, {&out, 1}}};

  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      glsl,
      static_cast<uint32_t>(refs.size()),
      refs.data(),
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      pc.data());
  vkCmdDispatch(dispatch.command_buffer, (row_count + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool try_eval_scan_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    Stream s) {
  if (inputs.size() != 1 ||
      (reduce_type != Scan::Sum && reduce_type != Scan::Prod &&
       reduce_type != Scan::Min && reduce_type != Scan::Max &&
       reduce_type != Scan::LogAddExp)) {
    return false;
  }

  array in = inputs[0];
  const bool cumsum_i32 =
      reduce_type == Scan::Sum && in.dtype() == int32 && out.dtype() == int32;
  const bool cumsum_u32 =
      reduce_type == Scan::Sum && in.dtype() == uint32 && out.dtype() == uint32;
  const bool cumsum_c64 = reduce_type == Scan::Sum &&
      in.dtype() == complex64 && out.dtype() == complex64;
  const bool cumprod_i32 =
      reduce_type == Scan::Prod && in.dtype() == int32 && out.dtype() == int32;
  const bool int_minmax_i32 =
      (reduce_type == Scan::Min || reduce_type == Scan::Max) &&
      in.dtype() == int32 && out.dtype() == int32;
  const bool integral_i64 =
      (reduce_type == Scan::Sum || reduce_type == Scan::Prod) &&
      in.dtype() == int64 && out.dtype() == int64;
  const bool cumprod_c64 = reduce_type == Scan::Prod &&
      in.dtype() == complex64 && out.dtype() == complex64;
  const bool cumlogaddexp_c64 = reduce_type == Scan::LogAddExp &&
      in.dtype() == complex64 && out.dtype() == complex64;
  const bool scan_f32 = in.dtype() == float32 && out.dtype() == float32;
  const bool lowp_cumsum = reduce_type == Scan::Sum &&
      (in.dtype() == float16 || in.dtype() == bfloat16) &&
      out.dtype() == in.dtype();
  if (in.ndim() == 0 ||
      (!scan_f32 && !cumsum_i32 && !cumsum_u32 && !cumsum_c64 && !cumprod_i32 &&
       !int_minmax_i32 && !integral_i64 && !cumprod_c64 && !lowp_cumsum &&
       !cumlogaddexp_c64)) {
    return false;
  }

  int normalized_axis = normalize_axis(axis, in.ndim());
  if (normalized_axis < 0 || normalized_axis >= in.ndim()) {
    return false;
  }

  array in_kernel = in;
  if (normalized_axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, normalized_axis, in.ndim() - 1);
  }

  array scan_input = in_kernel;

  if (!scan_input.flags().contiguous || scan_input.offset() != 0 ||
      scan_input.strides().back() != 1 ||
      !is_supported_unary_layout(scan_input)) {
    scan_input = contiguous_copy_gpu(scan_input, s);
  }
  if (!has_vulkan_buffer(scan_input)) {
    scan_input = contiguous_copy_gpu(scan_input, s);
  }

  Shape scan_shape = out.shape();
  if (normalized_axis != in.ndim() - 1) {
    std::swap(scan_shape[normalized_axis], scan_shape[in.ndim() - 1]);
  }
  if (scan_input.shape() != scan_shape) {
    return false;
  }

  if (scan_input.size() > std::numeric_limits<uint32_t>::max() ||
      scan_input.shape(scan_input.ndim() - 1) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  if (((reduce_type == Scan::Prod && !cumprod_c64 && !scan_f32 &&
        !cumprod_i32 && !integral_i64) ||
       (reduce_type == Scan::Min && !int_minmax_i32) ||
       (reduce_type == Scan::Max && !int_minmax_i32) ||
       reduce_type == Scan::LogAddExp) &&
      scan_input.shape(scan_input.ndim() - 1) > 128) {
    return false;
  }

  if (lowp_cumsum) {
    array scan_input_f32(scan_input.shape(), float32, nullptr, {});
    scan_input_f32.set_data(allocator::malloc(scan_input_f32.nbytes()));
    copy_gpu(scan_input, scan_input_f32, CopyType::General, s);

    array result_f32(scan_input.shape(), float32, nullptr, {});
    result_f32.set_data(allocator::malloc(result_f32.nbytes()));
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_cumsum_op(
        scan_input_f32,
        result_f32,
        vulkan::StaticShaderId::cumsum_f32,
        command_buffer,
        s,
        reverse,
        inclusive);
    vulkan::end_command_recording(s.index);

    array restored = result_f32;
    if (normalized_axis != in.ndim() - 1) {
      restored = swapaxes_in_eval(restored, normalized_axis, in.ndim() - 1);
    }
    if (!is_supported_unary_layout(restored)) {
      restored = contiguous_copy_gpu(restored, s);
    }

    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(restored, out, CopyType::GeneralGeneral, s);
    return true;
  }

  array inclusive_out(scan_input.shape(), scan_input.dtype(), nullptr, {});
  inclusive_out.set_data(allocator::malloc(inclusive_out.nbytes()));
  if (inclusive_out.size() == 0) {
    copy_gpu(inclusive_out, out, CopyType::GeneralGeneral, s);
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);

    switch (reduce_type) {
      case Scan::Sum:
        if (cumsum_c64) {
          vulkan::end_command_recording(s.index);
          if (!try_dispatch_complex_sum_scan(
                  scan_input, inclusive_out, s, reverse, inclusive)) {
            return false;
          }
          command_buffer = vulkan::begin_command_recording(s.index);
        } else if (cumsum_i32 || integral_i64) {
          vulkan::end_command_recording(s.index);
          if (!try_dispatch_integral_scan(
                  scan_input, inclusive_out, s, reduce_type, reverse, inclusive)) {
            return false;
          }
          command_buffer = vulkan::begin_command_recording(s.index);
        } else {
          vulkan::dispatch_cumsum_op(
              scan_input,
              inclusive_out,
              cumsum_i32       ? vulkan::StaticShaderId::cumsum_i32
                  : cumsum_u32 ? vulkan::StaticShaderId::cumsum_u32
                               : vulkan::StaticShaderId::cumsum_f32,
              command_buffer,
              s,
              reverse,
              inclusive);
        }
        break;
      case Scan::Prod:
        if (cumprod_c64) {
          vulkan::end_command_recording(s.index);
          if (!try_dispatch_complex_prod_scan(
                  scan_input, inclusive_out, s, reverse, inclusive)) {
            return false;
          }
          command_buffer = vulkan::begin_command_recording(s.index);
        } else if (cumprod_i32 || integral_i64) {
          vulkan::end_command_recording(s.index);
          if (!try_dispatch_integral_scan(
                  scan_input, inclusive_out, s, reduce_type, reverse, inclusive)) {
            return false;
          }
          command_buffer = vulkan::begin_command_recording(s.index);
        } else if (scan_f32 && scan_input.shape(scan_input.ndim() - 1) > 128) {
          vulkan::end_command_recording(s.index);
          if (!try_dispatch_prod_scan(
                  scan_input, inclusive_out, s, reverse, inclusive)) {
            return false;
          }
          command_buffer = vulkan::begin_command_recording(s.index);
        } else {
          vulkan::dispatch_scan_op(
              scan_input,
              inclusive_out,
              cumprod_i32 ? vulkan::StaticShaderId::cumprod_i32
                          : vulkan::StaticShaderId::cumprod_f32,
              command_buffer,
              s,
              reverse,
              inclusive);
        }
        break;
      case Scan::Min:
        if (int_minmax_i32) {
          vulkan::end_command_recording(s.index);
          if (!try_dispatch_integral_scan(
                  scan_input, inclusive_out, s, reduce_type, reverse, inclusive)) {
            return false;
          }
          command_buffer = vulkan::begin_command_recording(s.index);
        } else {
          vulkan::dispatch_scan_op(
              scan_input,
              inclusive_out,
              vulkan::StaticShaderId::cummin_f32,
              command_buffer,
              s,
              reverse,
              inclusive);
        }
        break;
      case Scan::Max:
        if (int_minmax_i32) {
          vulkan::end_command_recording(s.index);
          if (!try_dispatch_integral_scan(
                  scan_input, inclusive_out, s, reduce_type, reverse, inclusive)) {
            return false;
          }
          command_buffer = vulkan::begin_command_recording(s.index);
        } else {
          vulkan::dispatch_scan_op(
              scan_input,
              inclusive_out,
              vulkan::StaticShaderId::cummax_f32,
              command_buffer,
              s,
              reverse,
              inclusive);
        }
        break;
      case Scan::LogAddExp:
        vulkan::end_command_recording(s.index);
        if (cumlogaddexp_c64) {
          if (!try_dispatch_complex_logaddexp_scan(
                  scan_input, inclusive_out, s, reverse, inclusive)) {
            return false;
          }
        } else if (!try_dispatch_logaddexp_scan(
                       scan_input, inclusive_out, s, reverse, inclusive)) {
          return false;
        }
        command_buffer = vulkan::begin_command_recording(s.index);
        break;
      default:
        throw std::runtime_error("Unsupported Vulkan scan reduce type.");
    }

    array scan_result = inclusive_out;

    vulkan::end_command_recording(s.index);

    array restored = scan_result;
    if (normalized_axis != in.ndim() - 1) {
      restored = swapaxes_in_eval(restored, normalized_axis, in.ndim() - 1);
    }
    if (!is_supported_unary_layout(restored)) {
      restored = contiguous_copy_gpu(restored, s);
    }

    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(restored, out, CopyType::GeneralGeneral, s);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "scan_dispatch_failed reduce_type="
          << static_cast<int>(reduce_type) << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis, reverse, inclusive] = state();
  if (!try_eval_scan_vulkan(
          inputs, out, reduce_type, axis, reverse, inclusive, stream())) {
    throw std::runtime_error(
        "Scan operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
