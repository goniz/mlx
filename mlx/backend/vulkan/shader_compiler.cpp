// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/shader_compiler.h"

#include <fmt/format.h>
#include <shaderc/shaderc.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mlx::core::vulkan {

namespace {

bool trace_shader_compile_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_SHADER_COMPILE");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

} // namespace

std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name) {
  const bool trace = trace_shader_compile_enabled();

  if (trace) {
    std::cerr << "[vulkan-shader-compile] start: " << shader_name
              << " glsl_bytes=" << glsl_source.size() << "\n";
  }
  const auto t0 = trace ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};

  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  options.SetTargetEnvironment(
      shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
  options.SetSourceLanguage(shaderc_source_language_glsl);
  options.SetForcedVersionProfile(450, shaderc_profile_core);
  options.SetOptimizationLevel(shaderc_optimization_level_performance);

  auto result = compiler.CompileGlslToSpv(
      glsl_source.c_str(),
      glsl_source.size(),
      shaderc_compute_shader,
      shader_name.c_str(),
      options);

  if (trace) {
    const auto t1 = std::chrono::steady_clock::now();
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cerr << "[vulkan-shader-compile] done: " << shader_name
              << " spirv_words=" << (result.cend() - result.cbegin())
              << " ms=" << ms;
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
      std::cerr << " FAILED";
    }
    std::cerr << "\n";
  }

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    throw std::runtime_error(
        fmt::format(
            "Failed to compile Vulkan kernel '{}': {}",
            shader_name,
            result.GetErrorMessage()));
  }

  return std::vector<uint32_t>(result.cbegin(), result.cend());
}

std::string dtype_to_glsl_storage_type(Dtype dtype) {
  switch (dtype) {
    case mlx::core::float32:
      return "float";
    case mlx::core::float16:
      return "float16_t";
    case mlx::core::bfloat16:
      return "uint16_t";
    case mlx::core::bool_:
      return "uint8_t";
    case mlx::core::uint16:
      return "uint16_t";
    case mlx::core::uint8:
      return "uint8_t";
    case mlx::core::int8:
      return "int8_t";
    case mlx::core::int16:
      return "int16_t";
    case mlx::core::int32:
      return "int";
    case mlx::core::uint32:
      return "uint";
    case mlx::core::uint64:
      return "uint64_t";
    case mlx::core::int64:
      return "int64_t";
    case mlx::core::complex64:
      return "vec2";
    default:
      throw std::runtime_error(
          "Unsupported dtype for Vulkan dynamic shader generation.");
  }
}

bool uses_float16_extension(Dtype dtype) {
  return dtype == mlx::core::float16;
}

bool uses_16bit_storage(Dtype dtype) {
  return dtype == mlx::core::float16 || dtype == mlx::core::bfloat16 ||
      dtype == mlx::core::int16 || dtype == mlx::core::uint16;
}

bool uses_8bit_storage(Dtype dtype) {
  return dtype == mlx::core::bool_ || dtype == mlx::core::int8 ||
      dtype == mlx::core::uint8;
}

std::string emit_dynamic_shader_preamble(Dtype dtype, bool needs_int64_output) {
  std::ostringstream os;
  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  if (needs_int64_output || dtype == mlx::core::int64 ||
      dtype == mlx::core::uint64) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  if (uses_float16_extension(dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
  }
  if (uses_16bit_storage(dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  if (uses_8bit_storage(dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
    os << "#extension GL_EXT_shader_8bit_storage : require\n";
  }
  os << "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  return os.str();
}

std::string emit_dynamic_shader_preamble(
    Dtype in_dtype,
    Dtype out_dtype,
    bool needs_int64_output) {
  std::ostringstream os;
  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  if (needs_int64_output || in_dtype == mlx::core::int64 ||
      in_dtype == mlx::core::uint64 || out_dtype == mlx::core::int64 ||
      out_dtype == mlx::core::uint64) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  if (uses_float16_extension(in_dtype) || uses_float16_extension(out_dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
  }
  if (uses_16bit_storage(in_dtype) || uses_16bit_storage(out_dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  if (uses_8bit_storage(in_dtype) || uses_8bit_storage(out_dtype)) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
    os << "#extension GL_EXT_shader_8bit_storage : require\n";
  }
  os << "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  return os.str();
}

std::string zero_literal_for_dtype(Dtype dtype) {
  switch (dtype) {
    case mlx::core::float32:
      return "0.0";
    case mlx::core::float16:
      return "float16_t(0.0)";
    case mlx::core::bool_:
      return "uint8_t(0)";
    case mlx::core::uint8:
      return "uint8_t(0)";
    case mlx::core::uint16:
      return "uint16_t(0)";
    case mlx::core::uint32:
      return "uint(0)";
    case mlx::core::uint64:
      return "uint64_t(0)";
    case mlx::core::int8:
      return "int8_t(0)";
    case mlx::core::int16:
      return "int16_t(0)";
    case mlx::core::int32:
      return "int(0)";
    case mlx::core::int64:
      return "int64_t(0)";
    case mlx::core::bfloat16:
      return "uint16_t(0)";
    default:
      throw std::runtime_error(
          "Unsupported dtype for Vulkan dynamic cast shader zero literal.");
  }
}

std::string cast_expr_for_dtype(const std::string& expr, Dtype in, Dtype out) {
  if (out == mlx::core::bool_) {
    return "((" + expr + ") != " + zero_literal_for_dtype(in) +
        " ? uint8_t(1) : uint8_t(0))";
  }
  if (out == mlx::core::complex64) {
    if (in == mlx::core::complex64) {
      return expr;
    }
    return "vec2(float(" + expr + "), 0.0)";
  }
  if (in == mlx::core::complex64) {
    return dtype_to_glsl_storage_type(out) + "(" + expr + ".x)";
  }
  return dtype_to_glsl_storage_type(out) + "(" + expr + ")";
}

} // namespace mlx::core::vulkan
