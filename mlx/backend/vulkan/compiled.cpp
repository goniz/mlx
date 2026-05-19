// Copyright © 2024 Apple Inc.

#include <fmt/format.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/dtype_utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

using namespace vk;

bool trace_compiled_glsl_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COMPILED_GLSL");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

bool trace_compiled_runtime_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COMPILED_RUNTIME");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

bool trace_compiled_compile_flow_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COMPILED_COMPILE_FLOW");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

bool trace_compiled_timing_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_COMPILED_TIMING");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

uint32_t checked_elem_offset(const array& arr, const char* name) {
  const int64_t item_size = static_cast<int64_t>(size_of(arr.dtype()));
  const int64_t byte_offset = arr.offset();
  if (item_size <= 0 || byte_offset < 0 || (byte_offset % item_size) != 0) {
    throw std::runtime_error(
        fmt::format("Invalid element offset for Vulkan compiled {}", name));
  }
  const uint64_t elem_offset = static_cast<uint64_t>(byte_offset / item_size);
  if (elem_offset > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        fmt::format(
            "Element offset out of range for Vulkan compiled {}", name));
  }
  return static_cast<uint32_t>(elem_offset);
}

std::string join_primitive_names(const std::vector<array>& arrays) {
  std::ostringstream os;
  bool first = true;
  for (const auto& x : arrays) {
    if (!first) {
      os << ",";
    }
    first = false;
    os << x.primitive().name();
  }
  return os.str();
}

std::string dtype_to_glsl_storage(Dtype d) {
  switch (d) {
    case float32:
      return "float";
    case float16:
      return "float16_t";
    case bfloat16:
      return "uint16_t";
    case complex64:
      return "vec2";
    case int32:
      return "int";
    case uint32:
      return "uint";
    case int64:
      return "int64_t";
    case uint64:
      return "uint64_t";
    case int16:
      return "int16_t";
    case uint16:
      return "uint16_t";
    case int8:
      return "int8_t";
    case uint8:
      return "uint8_t";
    case bool_:
      return "uint8_t";
    default:
      throw std::runtime_error(
          fmt::format(
              "Unsupported dtype for Vulkan compiled: {}", dtype_to_string(d)));
  }
}

std::string dtype_to_glsl_compute(Dtype d) {
  switch (d) {
    case bfloat16:
      return "float";
    case complex64:
      return "vec2";
    case bool_:
      return "bool";
    default:
      return dtype_to_glsl_storage(d);
  }
}

bool has_dtype(const std::vector<array>& arrays, Dtype dtype) {
  for (const auto& x : arrays) {
    if (x.dtype() == dtype) {
      return true;
    }
  }
  return false;
}

bool has_any_dtype(
    const std::vector<array>& arrays,
    std::initializer_list<Dtype> dtypes) {
  for (const auto& x : arrays) {
    for (auto dtype : dtypes) {
      if (x.dtype() == dtype) {
        return true;
      }
    }
  }
  return false;
}

std::string glsl_float_constant(float v) {
  if (std::isnan(v)) {
    return "uintBitsToFloat(0x7fc00000u)";
  }
  if (std::isinf(v)) {
    return v > 0 ? "uintBitsToFloat(0x7f800000u)"
                 : "uintBitsToFloat(0xff800000u)";
  }
  return fmt::format("{:.9g}", v);
}

std::string glsl_constant(const array& x) {
  switch (x.dtype()) {
    case bool_:
      return x.item<bool>() ? "true" : "false";
    case float32:
      return fmt::format("float({})", glsl_float_constant(x.item<float>()));
    case complex64: {
      auto v = x.item<complex64_t>();
      return fmt::format(
          "vec2({}, {})",
          glsl_float_constant(v.real()),
          glsl_float_constant(v.imag()));
    }
    case bfloat16: {
      auto v = x.item<bfloat16_t>();
      return fmt::format(
          "float({})", glsl_float_constant(static_cast<float>(v)));
    }
    case float16: {
      auto v = x.item<float16_t>();
      return fmt::format(
          "float16_t({})", glsl_float_constant(static_cast<float>(v)));
    }
    default: {
      std::ostringstream ss;
      print_constant(ss, x);
      return fmt::format("{}({})", dtype_to_glsl_compute(x.dtype()), ss.str());
    }
  }
}

std::string glsl_cast_expr(Dtype dst, Dtype src, const std::string& expr) {
  if (dst == bool_ && src != bool_) {
    return fmt::format("bool({})", expr);
  }
  if (dst != bool_ && src == bool_) {
    return fmt::format("{}(({}) ? 1 : 0)", dtype_to_glsl_compute(dst), expr);
  }
  if (dst == complex64 && src != complex64) {
    return fmt::format("vec2({}, 0.0)", expr);
  }
  if (dst != complex64 && src == complex64) {
    return fmt::format("{}(({}).x)", dtype_to_glsl_compute(dst), expr);
  }
  return fmt::format("{}({})", dtype_to_glsl_compute(dst), expr);
}

bool supports_primitive_name(const std::string& prim_name) {
  static const std::unordered_set<std::string> supported = {
       "Abs",         "Add",         "AsType",      "Broadcast",
       "Ceil",        "Conjugate",   "Cos",         "Divide",
       "Equal",       "Erf",         "Exp",         "Floor",
       "Greater",     "GreaterEqual", "Imag",       "Less",
       "LessEqual",   "Log",         "LogAddExp",   "LogicalAnd",
       "LogicalOr",   "Maximum",     "Minimum",     "Multiply",
       "Negative",    "NotEqual",    "Power",       "Real",
       "Round",       "Sigmoid",     "Sin",         "Select",
       "Sign",        "Sqrt",        "Subtract",    "Square",      "Tan",
       "Tanh"};
  return supported.contains(prim_name);
}

bool has_unsupported_bool_tape_op(const std::vector<array>& tape) {
  return std::any_of(tape.begin(), tape.end(), [](const array& x) {
    if (is_static_cast(x.primitive())) {
      return false;
    }
    const bool uses_bool = x.dtype() == bool_ || std::any_of(
        x.inputs().begin(), x.inputs().end(), [](const array& input) {
          return input.dtype() == bool_;
        });
    return uses_bool && !supports_primitive_name(x.primitive().name());
  });
}

std::string emit_glsl_preamble(
    bool uses_bfloat16,
    bool uses_complex64,
    bool uses_float16_types,
    bool uses_int16_types,
    bool uses_int8_types,
    bool uses_erf,
    bool uses_power) {
  std::ostringstream os;
  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  if (uses_float16_types) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
  }
  if (uses_float16_types || uses_int16_types || uses_bfloat16) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  if (uses_int8_types) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
    os << "#extension GL_EXT_shader_8bit_storage : require\n";
    os << "#extension GL_EXT_scalar_block_layout : require\n";
  }
  os << "\n";

  if (uses_bfloat16) {
    os << R"(
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

  if (uses_complex64) {
    os << R"(
vec2 complex_mul(vec2 a, vec2 b) {
  precise float real = a.x * b.x - a.y * b.y;
  precise float imag = a.x * b.y + a.y * b.x;
  return vec2(real, imag);
}

vec2 complex_div(vec2 a, vec2 b) {
  precise float denom = b.x * b.x + b.y * b.y;
  precise float real_num = a.x * b.x + a.y * b.y;
  precise float imag_num = a.y * b.x - a.x * b.y;
  precise float real = real_num / denom;
  precise float imag = imag_num / denom;
  real += (real_num - real * denom) / denom;
  imag += (imag_num - imag * denom) / denom;
  return vec2(real, imag);
}

vec2 complex_exp(vec2 z) {
  float e = exp(z.x);
  return vec2(e * cos(z.y), e * sin(z.y));
}

vec2 complex_log(vec2 z) {
  return vec2(log(length(z)), atan(z.y, z.x));
}

vec2 complex_sin(vec2 z) {
  return vec2(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

vec2 complex_cos(vec2 z) {
  return vec2(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}

vec2 complex_tan(vec2 z) {
  return complex_div(complex_sin(z), complex_cos(z));
}

vec2 complex_sigmoid(vec2 z) {
  return complex_div(vec2(1.0, 0.0), vec2(1.0, 0.0) + complex_exp(-z));
}

float complex_abs(vec2 z) {
  return length(z);
}

vec2 complex_conjugate(vec2 z) {
  return vec2(z.x, -z.y);
}

)";
  }

  if (uses_erf) {
    os << R"(
float erf(float x) {
  const float a1 = 0.254829592;
  const float a2 = -0.284496736;
  const float a3 = 1.421413741;
  const float a4 = -1.453152027;
  const float a5 = 1.061405429;
  const float p = 0.3275911;
  float s = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
  return s * y;
}

)";
  }

  if (uses_power) {
    os << R"(
float safe_real_pow(float x, float y) {
  float yi = round(y);
  if (abs(y - yi) <= 0.00001 && abs(yi) <= 64.0) {
    int n = int(yi);
    float base = abs(x);
    float result = 1.0;
    int exp = abs(n);
    while (exp > 0) {
      if ((exp & 1) != 0) {
        result *= base;
      }
      base *= base;
      exp >>= 1;
    }
    if (n < 0) {
      result = 1.0 / result;
    }
    if (x < 0.0 && (abs(n) & 1) != 0) {
      result = -result;
    }
    return result;
  }
  return pow(x, y);
}

)";
  }

  os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  return os.str();
}

// Map primitive names to GLSL operators/functions
std::string get_glsl_operator(const std::string& primitive_name) {
  static const std::unordered_map<std::string, std::string> op_map = {
      {"Add", "+"},
      {"Subtract", "-"},
      {"Multiply", "*"},
      {"Divide", "/"},
      {"Equal", "=="},
      {"NotEqual", "!="},
      {"Greater", ">"},
      {"Less", "<"},
      {"GreaterEqual", ">="},
      {"LessEqual", "<="},
      {"LogicalAnd", "&&"},
      {"LogicalOr", "||"},
      {"Maximum", "max"},
      {"Minimum", "min"},
      // GLSL built-in functions (lowercase)
      {"Exp", "exp"},
      {"Erf", "erf"},
      {"Log", "log"},
      {"Sin", "sin"},
      {"Cos", "cos"},
      {"Tan", "tan"},
      {"Sqrt", "sqrt"},
      {"Abs", "abs"},
      {"Floor", "floor"},
      {"Ceil", "ceil"},
      {"Round", "round"},
      {"Power", "pow"},
      {"Sign", "sign"},
      {"Sigmoid", "sigmoid"},
      {"Tanh", "tanh"},
  };

  auto it = op_map.find(primitive_name);
  if (it != op_map.end()) {
    return it->second;
  }
  return primitive_name; // Function call style
}

// Build GLSL kernel source for the compiled tape
inline void build_glsl_kernel(
    std::string& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    const std::unordered_set<uintptr_t>& constant_ids,
    bool contiguous,
    int ndim,
    int work_per_thread,
    uint32_t params_binding) {
  // Maps to store array identifiers - use simple valid GLSL names
  std::unordered_map<std::uintptr_t, std::string> var_names;
  int var_counter = 0;

  // Helper to get or create identifier
  auto get_var_name = [&](const array& x) -> const std::string& {
    auto key = x.id();
    auto it = var_names.find(key);
    if (it != var_names.end()) {
      return it->second;
    }
    std::string id = fmt::format("v{}", var_counter++);
    auto [insert_it, _] = var_names.emplace(key, id);
    return insert_it->second;
  };

  // Pre-populate variable names for all arrays we'll reference
  for (const auto& x : inputs) {
    get_var_name(x);
  }
  for (const auto& x : outputs) {
    get_var_name(x);
  }
  for (const auto& x : tape) {
    get_var_name(x);
  }

  bool uses_bfloat16 = has_dtype(inputs, bfloat16) ||
      has_dtype(outputs, bfloat16) || has_dtype(tape, bfloat16);
  bool uses_complex64 = has_dtype(inputs, complex64) ||
      has_dtype(outputs, complex64) || has_dtype(tape, complex64);
  bool uses_float16_types = has_dtype(inputs, float16) ||
      has_dtype(outputs, float16) || has_dtype(tape, float16);
  bool uses_int16_types = has_any_dtype(inputs, {int16, uint16}) ||
      has_any_dtype(outputs, {int16, uint16}) ||
      has_any_dtype(tape, {int16, uint16});
  bool uses_int8_types = has_any_dtype(inputs, {int8, uint8, bool_}) ||
      has_any_dtype(outputs, {int8, uint8, bool_}) ||
      has_any_dtype(tape, {int8, uint8, bool_});
  const bool uses_power =
      std::any_of(tape.begin(), tape.end(), [](const array& x) {
        return x.primitive().name() == "Power";
      });
  const bool uses_erf = std::any_of(
      tape.begin(), tape.end(), [](const array& x) {
        return x.primitive().name() == "Erf";
      });

  // GLSL header
  os = emit_glsl_preamble(
      uses_bfloat16,
      uses_complex64,
      uses_float16_types,
      uses_int16_types,
      uses_int8_types,
      uses_erf,
      uses_power);

  // Determine max work per thread based on output dtype size
  int max_itemsize = 1;
  for (const auto& x : outputs) {
    max_itemsize = std::max(max_itemsize, static_cast<int>(x.itemsize()));
  }
  int wpt = std::min(work_per_thread, 16 / max_itemsize);
  wpt = std::max(wpt, 1);

  const std::string buffer_layout_fmt =
      uses_int8_types ? "layout(scalar, binding = {})" : "layout(binding = {})";

  // Buffer bindings for non-constant inputs
  int binding = 0;
  std::vector<std::pair<int, std::string>> input_bindings; // (index, name)

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i)) {
      continue;
    }
    const auto& x = inputs[i];
    const auto& xname = get_var_name(x);

    os += fmt::format(
        "{} readonly buffer Buf{} {{ {} {}[]; }};\n",
        fmt::format(fmt::runtime(buffer_layout_fmt), binding++),
        i,
        dtype_to_glsl_storage(x.dtype()),
        xname);
    input_bindings.push_back({static_cast<int>(i), xname});
  }

  // Output buffers
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& x = outputs[i];
    const auto& xname = get_var_name(x);
    os += fmt::format(
        "{} buffer OutBuf{} {{ {} {}[]; }};\n",
        fmt::format(fmt::runtime(buffer_layout_fmt), binding++),
        i,
        dtype_to_glsl_storage(x.dtype()),
        xname);
  }

  // Params SSBO buffer (shape, strides, offsets)
  os += fmt::format(
      "{} readonly buffer Params {{ uint p[]; }};\n",
      fmt::format(fmt::runtime(buffer_layout_fmt), params_binding));

  // Compute params buffer layout offsets
  uint32_t shape_base = 0;
  uint32_t output_strides_base = 0;
  uint32_t input_strides_base = 0;
  uint32_t input_offsets_base = 0;
  uint32_t output_offsets_base = 0;

  if (!contiguous) {
    int num_input_stride_sets = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!is_constant(i) && !is_scalar(inputs[i])) {
        num_input_stride_sets++;
      }
    }
    shape_base = 0;
    output_strides_base = static_cast<uint32_t>(ndim);
    input_strides_base = static_cast<uint32_t>(2 * ndim);
    input_offsets_base =
        static_cast<uint32_t>(2 * ndim + num_input_stride_sets * ndim);
    output_offsets_base =
        input_offsets_base + static_cast<uint32_t>(inputs.size());
  } else {
    input_offsets_base = 0;
    output_offsets_base = static_cast<uint32_t>(inputs.size());
  }

  // Push constants
  os += R"(
layout(push_constant) uniform PushConstants {
  uint size;
} pc;
 
)";

  // Main kernel function
  os += "void main() {\n";
  os += fmt::format("  uint base_idx = gl_GlobalInvocationID.x * {};\n", wpt);
  os += "  if (base_idx >= pc.size) return;\n\n";

  // Work loop
  if (wpt > 1) {
    os += fmt::format(
        "  for (int w = 0; w < {} && (base_idx + w) < pc.size; ++w) {{\n", wpt);
    os += "    uint idx = base_idx + w;\n";
  } else {
    os += "  uint idx = base_idx;\n";
  }

  if (!contiguous) {
    os += "    uint rem_idx = idx;\n";
    for (int axis = ndim - 1; axis >= 0; --axis) {
      os += fmt::format(
          "    uint coord_{} = rem_idx % p[{}u];\n", axis, shape_base + axis);
      os += fmt::format("    rem_idx /= p[{}u];\n", shape_base + axis);
    }
  }

  // Declare and load inputs into temps
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& x = inputs[i];
    const auto& xname = get_var_name(x);
    std::string type_str = dtype_to_glsl_compute(x.dtype());
    uint32_t param_offset = input_offsets_base + static_cast<uint32_t>(i);

    if (is_constant(i)) {
      continue;
    } else if (is_scalar(x)) {
      if (x.dtype() == bool_) {
        os += fmt::format(
            "    {} t_{} = ({}[p[{}u]] != uint8_t(0));\n",
            type_str,
            xname,
            xname,
            param_offset);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {} t_{} = bf16_to_fp32(uint({}[p[{}u]]));\n",
            type_str,
            xname,
            xname,
            param_offset);
      } else {
        os += fmt::format(
            "    {} t_{} = {}[p[{}u]];\n",
            type_str,
            xname,
            xname,
            param_offset);
      }
    } else if (contiguous) {
      if (x.dtype() == bool_) {
        os += fmt::format(
            "    {} t_{} = ({}[idx + p[{}u]] != uint8_t(0));\n",
            type_str,
            xname,
            xname,
            param_offset);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {} t_{} = bf16_to_fp32(uint({}[idx + p[{}u]]));\n",
            type_str,
            xname,
            xname,
            param_offset);
      } else {
        os += fmt::format(
            "    {} t_{} = {}[idx + p[{}u]];\n",
            type_str,
            xname,
            xname,
            param_offset);
      }
    } else {
      // Strided: compute location from params buffer
      int stride_set = 0;
      for (size_t j = 0; j < i; ++j) {
        if (!is_constant(j) && !is_scalar(inputs[j])) {
          stride_set++;
        }
      }
      uint32_t input_strides_offset = input_strides_base + stride_set * ndim;
      os += fmt::format("    uint loc_{} = p[{}u];\n", xname, param_offset);
      for (int axis = ndim - 1; axis >= 0; --axis) {
        os += fmt::format(
            "    loc_{0} += coord_{1} * p[{2}u];\n",
            xname,
            axis,
            input_strides_offset + axis);
      }
      if (x.dtype() == bool_) {
        os += fmt::format(
            "    {} t_{} = ({}[loc_{}] != uint8_t(0));\n",
            type_str,
            xname,
            xname,
            xname);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {} t_{} = bf16_to_fp32(uint({}[loc_{}]));\n",
            type_str,
            xname,
            xname,
            xname);
      } else {
        os += fmt::format(
            "    {} t_{} = {}[loc_{}];\n", type_str, xname, xname, xname);
      }
    }
  }

  // Helper to check if an array is a constant
  auto is_constant_array = [&](const array& x) -> bool {
    // Check if this array's id is in constant_ids
    return constant_ids.find(x.id()) != constant_ids.end();
  };

  // Helper to get GLSL expression for an input to a tape operation
  auto get_input_expr = [&](const array& x) -> std::string {
    if (is_constant_array(x)) {
      return glsl_constant(x);
    } else {
      // Use temp variable
      return fmt::format("t_{}", get_var_name(x));
    }
  };

  // Replay tape operations
  for (const auto& x : tape) {
    const auto& xname = get_var_name(x);
    std::string type_str = dtype_to_glsl_compute(x.dtype());
    const auto& prim = x.primitive();
    std::string prim_name = prim.name();

    os += fmt::format("    {} t_{} = ", type_str, xname);

    if (is_static_cast(prim)) {
      // Handle Broadcast/AsType as static casts
      os += fmt::format(
          "{};\n",
          glsl_cast_expr(
              x.dtype(), x.inputs()[0].dtype(), get_input_expr(x.inputs()[0])));
    } else {
      if (!supports_primitive_name(prim_name)) {
        throw std::runtime_error(
            fmt::format(
                "Unsupported primitive '{}' in Vulkan compiled kernel",
                prim_name));
      }

      // Get operator or function name
      std::string op = get_glsl_operator(prim_name);

      // Check if it's a binary operator or function
      bool is_binary_op =
          (op == "+" || op == "-" || op == "*" || op == "/" || op == "==" ||
           op == "!=" || op == ">" || op == "<" || op == ">=" || op == "<=" ||
           op == "&&" || op == "||" || op == "&" || op == "|" || op == "^");

      bool is_complex = x.dtype() == complex64;

      if (prim_name == "Negative" && x.inputs().size() == 1) {
        os += fmt::format("(-{});\n", get_input_expr(x.inputs()[0]));
      } else if (prim_name == "Select" && x.inputs().size() == 3) {
        os += fmt::format(
            "(({}) ? {} : {});\n",
            get_input_expr(x.inputs()[0]),
            get_input_expr(x.inputs()[1]),
            get_input_expr(x.inputs()[2]));
      } else if (prim_name == "Square" && x.inputs().size() == 1) {
        const auto input = get_input_expr(x.inputs()[0]);
        os += fmt::format("({} * {});\n", input, input);
      } else if (prim_name == "Sign" && x.inputs().size() == 1) {
        const auto input = get_input_expr(x.inputs()[0]);
        const auto input_dtype = x.inputs()[0].dtype();
        if (x.dtype() == complex64) {
          os += fmt::format(
              "(({} == vec2(0.0, 0.0)) ? {} : complex_div({}, vec2(complex_abs({}), 0.0)));\n",
              input,
              input,
              input,
              input);
        } else if (x.dtype() == bool_) {
          os += fmt::format("{};\n", input);
        } else if (
            input_dtype == uint8 || input_dtype == uint16 ||
            input_dtype == uint32 || input_dtype == uint64) {
          os += fmt::format("(({} == {}(0)) ? {}(0) : {}(1));\n", input, type_str, type_str, type_str);
        } else {
          os += fmt::format("sign({});\n", input);
        }
      } else if (
          prim_name == "Power" && x.inputs().size() == 2 && !is_complex) {
        const auto lhs = get_input_expr(x.inputs()[0]);
        const auto rhs = get_input_expr(x.inputs()[1]);
        os += fmt::format(
            "{}(safe_real_pow(float({}), float({})));\n", type_str, lhs, rhs);
      } else if (prim_name == "LogAddExp" && x.inputs().size() == 2) {
        auto lhs = get_input_expr(x.inputs()[0]);
        auto rhs = get_input_expr(x.inputs()[1]);
        std::string type_str = dtype_to_glsl_compute(x.dtype());
        os += fmt::format(
            "((min({0}, {1}) == {2}(-1.0 / 0.0) || max({0}, {1}) == {2}(1.0 / 0.0)) ? max({0}, {1}) : (max({0}, {1}) + log({2}(1.0) + exp(min({0}, {1}) - max({0}, {1})))));\n",
            lhs,
            rhs,
            type_str);
      } else if (is_binary_op && x.inputs().size() == 2) {
        if (is_complex && op == "*") {
          os += fmt::format(
              "complex_mul({}, {});\n",
              get_input_expr(x.inputs()[0]),
              get_input_expr(x.inputs()[1]));
        } else if (is_complex && op == "/") {
          os += fmt::format(
              "complex_div({}, {});\n",
              get_input_expr(x.inputs()[0]),
              get_input_expr(x.inputs()[1]));
        } else {
          os += fmt::format(
              "({} {} {});\n",
              get_input_expr(x.inputs()[0]),
              op,
              get_input_expr(x.inputs()[1]));
        }
      } else if (op == "max" || op == "min") {
        // Built-in GLSL functions
        os += fmt::format(
            "{}({}, {});\n",
            op,
            get_input_expr(x.inputs()[0]),
            get_input_expr(x.inputs()[1]));
      } else {
        // Generic function call - use the mapped name (e.g., "exp" instead of
        // "Exp")
        std::string fn = op;
        if (prim_name == "Real" && !x.inputs().empty() &&
            x.inputs()[0].dtype() == complex64) {
          os += fmt::format("({}).x;\n", get_input_expr(x.inputs()[0]));
        } else if (
            prim_name == "Imag" && !x.inputs().empty() &&
            x.inputs()[0].dtype() == complex64) {
          os += fmt::format("({}).y;\n", get_input_expr(x.inputs()[0]));
        } else if (x.dtype() == complex64) {
          if (op == "exp") {
            fn = "complex_exp";
          } else if (op == "log") {
            fn = "complex_log";
          } else if (op == "sin") {
            fn = "complex_sin";
          } else if (op == "cos") {
            fn = "complex_cos";
          } else if (op == "tan") {
            fn = "complex_tan";
          } else if (op == "sigmoid") {
            fn = "complex_sigmoid";
          } else if (op == "abs") {
            fn = "complex_abs";
          } else if (prim_name == "Conjugate") {
            fn = "complex_conjugate";
          }
        } else if (
            prim_name == "Abs" && !x.inputs().empty() &&
            x.inputs()[0].dtype() == complex64) {
          fn = "complex_abs";
        } else if (
            prim_name == "Conjugate" && !x.inputs().empty() &&
            x.inputs()[0].dtype() == complex64) {
          fn = "complex_conjugate";
        } else if (op == "sigmoid") {
          std::string type_str = dtype_to_glsl_compute(x.dtype());
          fn = fmt::format("({0}(1.0) / ({0}(1.0) + exp(-", type_str);
        }

        if ((prim_name == "Real" || prim_name == "Imag") &&
            !x.inputs().empty() && x.inputs()[0].dtype() == complex64) {
          // Already emitted.
        } else if (op == "sigmoid" && x.dtype() != complex64) {
          os += fn;
          os += get_input_expr(x.inputs()[0]);
          os += ")));\n";
        } else {
          os += fmt::format("{}(", fn);
          for (size_t i = 0; i < x.inputs().size(); ++i) {
            if (i > 0)
              os += ", ";
            os += get_input_expr(x.inputs()[i]);
          }
          os += ");\n";
        }
      }
    }
  }

  // Write outputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& x = outputs[i];
    const auto& xname = get_var_name(x);
    uint32_t param_offset = output_offsets_base + static_cast<uint32_t>(i);

    if (contiguous) {
      if (x.dtype() == bool_) {
        os += fmt::format(
            "    {}[idx + p[{}u]] = uint8_t(t_{} ? 1 : 0);\n",
            xname,
            param_offset,
            xname);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {}[idx + p[{}u]] = uint16_t(fp32_to_bf16(t_{}));\n",
            xname,
            param_offset,
            xname);
      } else {
        os += fmt::format(
            "    {}[idx + p[{}u]] = t_{};\n", xname, param_offset, xname);
      }
    } else {
      os += fmt::format("    uint loc_out_{} = p[{}u];\n", xname, param_offset);
      for (int axis = ndim - 1; axis >= 0; --axis) {
        os += fmt::format(
            "    loc_out_{0} += coord_{1} * p[{2}u];\n",
            xname,
            axis,
            output_strides_base + axis);
      }
      if (x.dtype() == bool_) {
        os += fmt::format(
            "    {}[loc_out_{}] = uint8_t(t_{} ? 1 : 0);\n",
            xname,
            xname,
            xname);
      } else if (x.dtype() == bfloat16) {
        os += fmt::format(
            "    {}[loc_out_{}] = uint16_t(fp32_to_bf16(t_{}));\n",
            xname,
            xname,
            xname);
      } else {
        os += fmt::format("    {}[loc_out_{}] = t_{};\n", xname, xname, xname);
      }
    }
  }

  if (wpt > 1) {
    os += "  }\n";
  }

  os += "}\n";
}

} // namespace

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  vulkan::ScopedSyncLabel sync_label("compiled_eval_gpu");
  auto& s = stream();

  // Check for unsupported dtypes.
  bool has_unsupported_dtype = false;
  for (const auto& x : inputs_) {
    if (x.dtype() == float64) {
      has_unsupported_dtype = true;
      break;
    }
  }
  if (!has_unsupported_dtype) {
    for (const auto& x : outputs_) {
      if (x.dtype() == float64) {
        has_unsupported_dtype = true;
        break;
      }
    }
  }

  if (has_unsupported_dtype) {
    throw std::runtime_error(
        "Compiled kernel failed on Vulkan (unsupported dtype).");
  }

  // Determine work per thread based on output dtype size
  int max_itemsize = 1;
  for (const auto& x : outputs_) {
    max_itemsize = std::max(max_itemsize, static_cast<int>(x.itemsize()));
  }
  int work_per_thread = 16 / max_itemsize;
  work_per_thread = std::max(work_per_thread, 1);

  // Collapse contiguous dims to route to a faster kernel if possible
  auto [contiguous, shape, strides] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);
  auto dispatch_inputs = inputs;
  std::vector<array> pending_inputs;
  pending_inputs.reserve(dispatch_inputs.size());
  for (const auto& in : dispatch_inputs) {
    if (in.status() == array::Status::unscheduled) {
      pending_inputs.push_back(in);
    }
  }
  if (!pending_inputs.empty()) {
    async_eval(std::move(pending_inputs));
  }
  std::vector<uint32_t> input_offsets(inputs.size(), 0u);
  for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
    input_offsets[i] = checked_elem_offset(dispatch_inputs[i], "input");
  }
  std::vector<uint32_t> output_offsets(outputs.size(), 0u);
  for (size_t i = 0; i < outputs.size(); ++i) {
    output_offsets[i] = checked_elem_offset(outputs[i], "output");
  }

  if (!contiguous) {
    work_per_thread = 1;
  }

  const auto requires_cpu_fallback = [&]() {
    if (std::any_of(tape_.begin(), tape_.end(), [](const array& x) {
          return x.offset() != 0;
        })) {
      return true;
    }

    if (has_unsupported_bool_tape_op(tape_)) {
      return true;
    }

    return std::any_of(tape_.begin(), tape_.end(), [](const array& x) {
      return !is_static_cast(x.primitive()) &&
          !supports_primitive_name(x.primitive().name());
    });
  }();

  if (requires_cpu_fallback) {
    std::ostringstream msg;
    msg << "Compiled kernel failed on Vulkan (complex tape operations or unsupported layout)."
        << " contiguous=" << contiguous << " inputs_offset="
        << std::any_of(
               inputs.begin(),
               inputs.end(),
               [](const array& x) { return x.offset() != 0; })
        << " outputs_offset="
        << std::any_of(
               outputs.begin(),
               outputs.end(),
               [](const array& x) { return x.offset() != 0; })
        << " tape_offset="
        << std::any_of(
               tape_.begin(),
               tape_.end(),
               [](const array& x) { return x.offset() != 0; })
        << " tape_primitives=" << join_primitive_names(tape_);
    throw std::runtime_error(msg.str());
  }

  // Use large index if needed
  bool large = compiled_use_large_index(dispatch_inputs, outputs, contiguous) ||
      outputs[0].data_size() > std::numeric_limits<uint32_t>::max();
  if (large && !contiguous) {
    throw std::runtime_error(
        "Compiled kernel failed on Vulkan (arrays >2^32 elements are only supported for contiguous layouts).");
  }
  // Build kernel name based on configuration
  std::string kernel_name = kernel_lib_;
  if (contiguous) {
    kernel_name += "_contiguous";
  } else {
    kernel_name += fmt::format("_strided_{}", shape.size());
  }
  // Check if we already have this kernel compiled (simple cache check)
  auto& manager = vulkan::KernelManager::get();
  auto* existing_shader = manager.get_shader(kernel_name);

  if (trace_compiled_compile_flow_enabled()) {
    std::cerr << "[vulkan-compiled-flow] eval_gpu enter: " << kernel_name
              << " tape_size=" << tape_.size() << " contiguous=" << contiguous
              << " large=" << large << " inputs=" << inputs_.size()
              << " outputs=" << outputs_.size()
              << " cached=" << (existing_shader != nullptr) << "\n";
  } else if (trace_compiled_timing_enabled()) {
    std::cerr << "[vulkan-compiled-timing] kernel=" << kernel_name
              << " cached=" << (existing_shader != nullptr)
              << " tape_size=" << tape_.size() << " contiguous=" << contiguous
              << "\n";
  }

  std::vector<uint32_t> spirv;
  if (!existing_shader) {
    const auto compile_start = std::chrono::steady_clock::now();

    // Generate GLSL source
    std::string glsl_source;
    uint32_t params_binding = 0;
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (!is_constant_(i)) {
        params_binding++;
      }
    }
    params_binding += static_cast<uint32_t>(outputs_.size());
    build_glsl_kernel(
        glsl_source,
        kernel_name,
        inputs_,
        outputs_,
        tape_,
        is_constant_,
        constant_ids_,
        contiguous,
        static_cast<int>(shape.size()),
        work_per_thread,
        params_binding);

    if (trace_compiled_glsl_enabled()) {
      std::cerr << "=== Vulkan compiled GLSL: " << kernel_name << " ===\n"
                << glsl_source << "\n=== End GLSL ===\n";
    }

    if (trace_compiled_compile_flow_enabled()) {
      const auto glsl_built_at = std::chrono::steady_clock::now();
      const auto glsl_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              glsl_built_at - compile_start)
              .count();
      std::cerr << "[vulkan-compiled-flow] glsl_generated: " << kernel_name
                << " glsl_bytes=" << glsl_source.size() << " ms=" << glsl_ms
                << "\n";
    }

    const auto glsl_built_at = std::chrono::steady_clock::now();

    // Compile to SPIR-V
    try {
      spirv = vulkan::compile_glsl_to_spirv(glsl_source, kernel_name);
    } catch (const std::exception& e) {
      std::cerr << "=== FAILED GLSL for: " << kernel_name
                << " ===" << std::endl;
      std::cerr << glsl_source << std::endl;
      std::cerr << "=== End GLSL ===" << std::endl;
      throw;
    }

    const auto spirv_compiled_at = std::chrono::steady_clock::now();

    if (trace_compiled_compile_flow_enabled()) {
      const auto spirv_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              spirv_compiled_at - glsl_built_at)
              .count();
      std::cerr << "[vulkan-compiled-flow] spirv_compiled: " << kernel_name
                << " spirv_words=" << spirv.size() << " ms=" << spirv_ms
                << "\n";
    }

    // Register the shader
    manager.register_shader(
        kernel_name, spirv.data(), spirv.size() * sizeof(uint32_t));

    if (trace_compiled_compile_flow_enabled()) {
      const auto registered_at = std::chrono::steady_clock::now();
      const auto register_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              registered_at - spirv_compiled_at)
              .count();
      const auto total_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              registered_at - compile_start)
              .count();
      std::cerr << "[vulkan-compiled-flow] registered: " << kernel_name
                << " register_ms=" << register_ms << " total_ms=" << total_ms
                << "\n";
    }

    if (trace_compiled_timing_enabled()) {
      const auto registered_at = std::chrono::steady_clock::now();
      const auto glsl_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              glsl_built_at - compile_start)
              .count();
      const auto spirv_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              spirv_compiled_at - glsl_built_at)
              .count();
      const auto register_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              registered_at - spirv_compiled_at)
              .count();
      const auto total_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              registered_at - compile_start)
              .count();
      std::cerr << "[vulkan-compiled-timing] kernel=" << kernel_name
                << " glsl_ms=" << glsl_ms << " spirv_ms=" << spirv_ms
                << " register_ms=" << register_ms << " total_ms=" << total_ms
                << " tape_size=" << tape_.size() << " contiguous=" << contiguous
                << " large=" << large << "\n";
    }
  }

  // Allocate outputs with buffer donation
  std::vector<array> allocation_inputs;
  if (contiguous) {
    allocation_inputs = dispatch_inputs;
  }
  compiled_allocate_outputs(
      allocation_inputs, outputs, is_constant_, contiguous, [&](size_t n) {
        return vulkan::allocator().malloc(n);
      });

  // Build dynamic descriptor set bindings
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  uint32_t binding_idx = 0;

  // Input bindings
  for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    bindings.push_back(
        {binding_idx++,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         1,
         VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr});
  }

  // Output bindings
  for (size_t i = 0; i < outputs.size(); ++i) {
    bindings.push_back(
        {binding_idx++,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         1,
         VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr});
  }

  // Params binding
  bindings.push_back(
      {binding_idx++,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       1,
       VK_SHADER_STAGE_COMPUTE_BIT,
       nullptr});

  // Get or create pipeline
  uint32_t push_constant_size = sizeof(uint32_t);

  auto* pipeline =
      manager.get_pipeline(kernel_name, bindings, push_constant_size);

  // Get command buffer
  auto cmd_buffer = vulkan::begin_command_recording(s.index);
  const uint64_t descriptor_epoch = vulkan::descriptor_epoch_for_stream(s);

  const bool use_push_descriptor = pipeline->supports_push_descriptor;

  // Prepare descriptor writes
  std::vector<VkWriteDescriptorSet> writes;
  std::vector<VkDescriptorBufferInfo> buffer_infos;
  buffer_infos.reserve(binding_idx);

  // Helper to add buffer info
  auto add_buffer = [&](const array& arr) {
    vulkan::retain_array_for_stream(s, arr);
    auto* vulkan_buffer = static_cast<const vulkan::VulkanBuffer*>(
        static_cast<const void*>(arr.buffer().ptr()));
    if (!vulkan_buffer || !vulkan_buffer->buffer) {
      throw std::runtime_error("Missing Vulkan buffer for compiled kernel");
    }

    VkDescriptorBufferInfo info{};
    info.buffer = vulkan_buffer->buffer;
    info.offset = 0;
    info.range = VK_WHOLE_SIZE;
    buffer_infos.push_back(info);
  };

  // Input buffers
  uint32_t write_idx = 0;

  for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    add_buffer(dispatch_inputs[i]);
    writes.push_back({});
    writes[write_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[write_idx].dstSet = VK_NULL_HANDLE;
    writes[write_idx].dstBinding = write_idx;
    writes[write_idx].dstArrayElement = 0;
    writes[write_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[write_idx].descriptorCount = 1;
    writes[write_idx].pBufferInfo = &buffer_infos[write_idx];
    ++write_idx;
  }

  // Output buffers
  for (size_t i = 0; i < outputs.size(); ++i) {
    add_buffer(outputs[i]);
    writes.push_back({});
    writes[write_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[write_idx].dstSet = VK_NULL_HANDLE;
    writes[write_idx].dstBinding = write_idx;
    writes[write_idx].dstArrayElement = 0;
    writes[write_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[write_idx].descriptorCount = 1;
    writes[write_idx].pBufferInfo = &buffer_infos[write_idx];
    ++write_idx;
  }

  // Build params data
  std::vector<uint32_t> params_data;
  if (contiguous) {
    for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
      params_data.push_back(input_offsets[i]);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      params_data.push_back(output_offsets[i]);
    }
  } else {
    for (auto dim : shape) {
      params_data.push_back(static_cast<uint32_t>(dim));
    }
    const auto& out_strides = strides[0];
    for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
      params_data.push_back(static_cast<uint32_t>(out_strides[d]));
    }
    for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
      if (is_constant_(i) || is_scalar(dispatch_inputs[i])) {
        continue;
      }
      int stride_idx = 1;
      for (size_t j = 0; j < i; ++j) {
        if (!is_constant_(j) && !is_scalar(dispatch_inputs[j])) {
          stride_idx++;
        }
      }
      const auto& in_strides = strides[stride_idx];
      for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
        params_data.push_back(static_cast<uint32_t>(in_strides[d]));
      }
    }
    for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
      params_data.push_back(input_offsets[i]);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      params_data.push_back(output_offsets[i]);
    }
  }

  if (trace_compiled_runtime_enabled()) {
    std::cerr << "[vulkan-compiled] kernel=" << kernel_name
              << " contiguous=" << contiguous << " shape=" << shape
              << " params_size=" << params_data.size()
              << " out_data_size=" << outputs[0].data_size()
              << " out_size=" << outputs[0].size()
              << " input_offsets=" << input_offsets
              << " output_offsets=" << output_offsets << "\n";
    for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
      if (dispatch_inputs[i].dtype() == float32 &&
          dispatch_inputs[i].size() > 0) {
        auto* ptr =
            static_cast<const float*>(dispatch_inputs[i].buffer().raw_ptr());
        std::cerr << "  input[" << i << "] first=" << ptr[0]
                  << " size=" << dispatch_inputs[i].size() << "\n";
      }
    }
  }

  auto params_array =
      array(params_data.data(), {static_cast<int>(params_data.size())}, uint32);
  params_array.eval();
  add_buffer(params_array);
  vulkan::retain_array_for_stream(s, params_array);
  writes.push_back({});
  writes[write_idx].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[write_idx].dstSet = VK_NULL_HANDLE;
  writes[write_idx].dstBinding = write_idx;
  writes[write_idx].dstArrayElement = 0;
  writes[write_idx].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[write_idx].descriptorCount = 1;
  writes[write_idx].pBufferInfo = &buffer_infos[write_idx];
  ++write_idx;

  if (large) {
    for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
      if (is_constant_(i) || is_scalar(dispatch_inputs[i])) {
        continue;
      }
      if (dispatch_inputs[i].data_size() != outputs[0].data_size()) {
        throw std::runtime_error(
            "Compiled kernel failed on Vulkan (large contiguous compiled kernels require elementwise non-scalar inputs).");
      }
    }
  }

  auto update_descriptor_set_for_chunk = [&](uint64_t chunk_base_elements,
                                             uint64_t chunk_elements) {
    size_t descriptor_binding = 0;

    for (size_t i = 0; i < dispatch_inputs.size(); ++i) {
      if (is_constant_(i)) {
        continue;
      }
      const auto& arr = dispatch_inputs[i];
      const uint64_t item_size = static_cast<uint64_t>(size_of(arr.dtype()));
      uint64_t offset_bytes = 0;
      if (large && !is_scalar(arr)) {
        offset_bytes += chunk_base_elements * item_size;
      }

      buffer_infos[descriptor_binding].offset =
          static_cast<VkDeviceSize>(offset_bytes);
      buffer_infos[descriptor_binding].range = VK_WHOLE_SIZE;
      descriptor_binding++;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& arr = outputs[i];
      const uint64_t item_size = static_cast<uint64_t>(size_of(arr.dtype()));
      uint64_t offset_bytes = 0;
      if (large) {
        offset_bytes += chunk_base_elements * item_size;
      }

      buffer_infos[descriptor_binding].offset =
          static_cast<VkDeviceSize>(offset_bytes);
      buffer_infos[descriptor_binding].range = VK_WHOLE_SIZE;
      descriptor_binding++;
    }

    // Params buffer offset is always 0 (no chunking for params)
    buffer_infos[descriptor_binding].offset = 0;
    buffer_infos[descriptor_binding].range = VK_WHOLE_SIZE;
    descriptor_binding++;
  };

  std::vector<vk::DescriptorSet> descriptor_sets_to_free;
  auto bind_descriptors_for_chunk = [&](uint64_t chunk_base_elements,
                                        uint64_t chunk_elements) {
    update_descriptor_set_for_chunk(chunk_base_elements, chunk_elements);

    if (use_push_descriptor) {
      auto push_fn = vulkan::VulkanContext::get().push_descriptor_fn();
      if (push_fn == nullptr) {
        throw std::runtime_error(
            "Missing Vulkan push descriptor function for compiled kernel");
      }
      for (auto& write : writes) {
        write.dstSet = VK_NULL_HANDLE;
      }
      push_fn(
          cmd_buffer,
          VK_PIPELINE_BIND_POINT_COMPUTE,
          pipeline->layout,
          0,
          static_cast<uint32_t>(writes.size()),
          writes.data());
      return;
    }

    auto descriptor_set =
        manager.allocate_descriptor_set(pipeline->descriptor_layout);
    descriptor_sets_to_free.push_back(descriptor_set);
    for (auto& write : writes) {
      write.dstSet = descriptor_set;
    }
    if (!writes.empty()) {
      vkUpdateDescriptorSets(
          vulkan::VulkanContext::get().device(),
          static_cast<uint32_t>(writes.size()),
          writes.data(),
          0,
          nullptr);
    }

    VkDescriptorSet vk_descriptor_set =
        static_cast<VkDescriptorSet>(descriptor_set);
    vkCmdBindDescriptorSets(
        cmd_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline->layout,
        0,
        1,
        &vk_descriptor_set,
        0,
        nullptr);
  };

  // Bind pipeline and descriptor set
  vkCmdBindPipeline(
      cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);

  // Set push constants
  struct PushConstants {
    uint32_t size;
  } pc;

  // Dispatch
  uint64_t num_elements = outputs[0].data_size();
  if (!large) {
    bind_descriptors_for_chunk(0, num_elements);

    pc.size = static_cast<uint32_t>(num_elements);
    vkCmdPushConstants(
        cmd_buffer,
        pipeline->layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pc),
        &pc);

    uint32_t workgroups = static_cast<uint32_t>(
        (num_elements + 256ULL * static_cast<uint64_t>(work_per_thread) - 1) /
        (256ULL * static_cast<uint64_t>(work_per_thread)));
    workgroups = std::max(workgroups, 1u);
    vkCmdDispatch(cmd_buffer, workgroups, 1, 1);
  } else {
    constexpr uint64_t kMaxChunkElements =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    for (uint64_t chunk_base = 0; chunk_base < num_elements;) {
      const uint64_t chunk_elements =
          std::min(kMaxChunkElements, num_elements - chunk_base);
      bind_descriptors_for_chunk(chunk_base, chunk_elements);

      pc.size = static_cast<uint32_t>(chunk_elements);
      vkCmdPushConstants(
          cmd_buffer,
          pipeline->layout,
          VK_SHADER_STAGE_COMPUTE_BIT,
          0,
          sizeof(pc),
          &pc);

      uint32_t workgroups = static_cast<uint32_t>(
          (chunk_elements + 256ULL * static_cast<uint64_t>(work_per_thread) -
           1) /
          (256ULL * static_cast<uint64_t>(work_per_thread)));
      workgroups = std::max(workgroups, 1u);
      vkCmdDispatch(cmd_buffer, workgroups, 1, 1);
      chunk_base += chunk_elements;
    }
  }

  // Defer descriptor set cleanup
  if (!use_push_descriptor) {
    for (auto descriptor_set : descriptor_sets_to_free) {
      manager.defer_descriptor_set_free(
          s.index, descriptor_epoch, descriptor_set);
    }
  }
}

} // namespace mlx::core
