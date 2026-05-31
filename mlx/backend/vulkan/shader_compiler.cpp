// Copyright © 2024 Apple Inc.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <shaderc/shaderc.hpp>

#include "mlx/backend/vulkan/shader_compiler.h"

namespace mlx::core::vulkan {

namespace {

bool env_flag_enabled(const char* name) {
  const char* env = std::getenv(name);
  return env != nullptr && std::string(env) != "0";
}

bool trace_shader_compile_enabled() {
  static const bool enabled = []() {
    return env_flag_enabled("MLX_VULKAN_TRACE_SHADER_COMPILE");
  }();
  return enabled;
}

bool dump_failed_shader_source_enabled() {
  static const bool enabled = []() {
    return env_flag_enabled("MLX_VULKAN_DUMP_FAILED_SHADER");
  }();
  return enabled;
}

shaderc_optimization_level shaderc_optimization_level_for(
    DynamicShaderOptimization optimization) {
  switch (optimization) {
    case DynamicShaderOptimization::Zero:
      return shaderc_optimization_level_zero;
    case DynamicShaderOptimization::Size:
      return shaderc_optimization_level_size;
    case DynamicShaderOptimization::Performance:
      return shaderc_optimization_level_performance;
  }
  return shaderc_optimization_level_performance;
}

shaderc::CompileOptions make_compile_options(
    DynamicShaderOptimization optimization) {
  shaderc::CompileOptions options;
  options.SetTargetEnvironment(
      shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
  options.SetSourceLanguage(shaderc_source_language_glsl);
  options.SetForcedVersionProfile(450, shaderc_profile_core);
  options.SetOptimizationLevel(shaderc_optimization_level_for(optimization));
  return options;
}

shaderc::Compiler& shaderc_compiler() {
  thread_local shaderc::Compiler compiler;
  return compiler;
}

const shaderc::CompileOptions& compile_options_for(
    DynamicShaderOptimization optimization) {
  thread_local const shaderc::CompileOptions zero_options =
      make_compile_options(DynamicShaderOptimization::Zero);
  thread_local const shaderc::CompileOptions size_options =
      make_compile_options(DynamicShaderOptimization::Size);
  thread_local const shaderc::CompileOptions performance_options =
      make_compile_options(DynamicShaderOptimization::Performance);
  switch (optimization) {
    case DynamicShaderOptimization::Zero:
      return zero_options;
    case DynamicShaderOptimization::Size:
      return size_options;
    case DynamicShaderOptimization::Performance:
      return performance_options;
  }
  return performance_options;
}

std::mutex& spirv_cache_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<std::string, std::vector<uint32_t>>& spirv_cache() {
  static auto* cache =
      new std::unordered_map<std::string, std::vector<uint32_t>>();
  return *cache;
}

std::optional<std::vector<uint32_t>> get_cached_spirv(
    const std::string& cache_key) {
  std::lock_guard<std::mutex> lock(spirv_cache_mutex());
  auto it = spirv_cache().find(cache_key);
  if (it == spirv_cache().end()) {
    return std::nullopt;
  }
  return it->second;
}

void cache_spirv(
    const std::string& cache_key,
    const std::vector<uint32_t>& spirv) {
  std::lock_guard<std::mutex> lock(spirv_cache_mutex());
  spirv_cache().try_emplace(cache_key, spirv);
}

std::string make_spirv_cache_key(
    const std::string& glsl_source,
    const DynamicShaderCompileOptions& options) {
  return fmt::format(
      "optimization:{}\n{}",
      static_cast<int>(options.optimization),
      glsl_source);
}

std::string source_excerpt(const std::string& glsl_source) {
  constexpr size_t kMaxLines = 120;
  std::istringstream input(glsl_source);
  std::ostringstream excerpt;
  std::string line;
  size_t line_number = 1;
  while (line_number <= kMaxLines && std::getline(input, line)) {
    excerpt << line_number++ << ": " << line << '\n';
  }
  if (std::getline(input, line)) {
    excerpt << "... truncated after " << kMaxLines << " lines ...\n";
  }
  return excerpt.str();
}

struct DynamicShaderFeatures {
  bool needs_int64{false};
  bool uses_float16{false};
  bool uses_16bit_storage{false};
  bool uses_8bit_storage{false};
  bool uses_scalar_block_layout{false};
};

DynamicShaderFeatures collect_dynamic_shader_features(
    const DynamicShaderPreambleOptions& options) {
  DynamicShaderFeatures features;
  features.needs_int64 = options.needs_int64;
  for (Dtype dtype : options.dtypes) {
    features.needs_int64 = features.needs_int64 || dtype == mlx::core::int64 ||
        dtype == mlx::core::uint64;
    features.uses_float16 =
        features.uses_float16 || uses_float16_extension(dtype);
    features.uses_16bit_storage =
        features.uses_16bit_storage || uses_16bit_storage(dtype);
    features.uses_8bit_storage =
        features.uses_8bit_storage || uses_8bit_storage(dtype);
  }
  features.uses_scalar_block_layout = options.needs_scalar_block_layout ||
      features.uses_8bit_storage || features.uses_16bit_storage;
  return features;
}

} // namespace

std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name) {
  return compile_glsl_to_spirv(glsl_source, shader_name, {});
}

std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name,
    const DynamicShaderCompileOptions& options) {
  const bool trace = trace_shader_compile_enabled();
  const std::string cache_key = make_spirv_cache_key(glsl_source, options);

  if (auto cached = get_cached_spirv(cache_key)) {
    if (trace) {
      std::cerr << "[vulkan-shader-compile] cache_hit: " << shader_name
                << " spirv_words=" << cached->size() << "\n";
    }
    return *cached;
  }

  if (trace) {
    std::cerr << "[vulkan-shader-compile] start: " << shader_name
              << " glsl_bytes=" << glsl_source.size() << "\n";
  }
  const auto t0 = trace ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};

  auto result = shaderc_compiler().CompileGlslToSpv(
      glsl_source.c_str(),
      glsl_source.size(),
      shaderc_compute_shader,
      shader_name.c_str(),
      compile_options_for(options.optimization));

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
    std::string message = fmt::format(
        "Failed to compile Vulkan kernel '{}': {}",
        shader_name,
        result.GetErrorMessage());
    if (dump_failed_shader_source_enabled()) {
      message += "\nGenerated GLSL excerpt:\n" + source_excerpt(glsl_source);
    }
    throw std::runtime_error(message);
  }

  std::vector<uint32_t> spirv(result.cbegin(), result.cend());
  cache_spirv(cache_key, spirv);
  return spirv;
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

std::string dtype_to_glsl_compute_type(Dtype dtype) {
  switch (dtype) {
    case mlx::core::bfloat16:
      return "float";
    case mlx::core::complex64:
      return "vec2";
    case mlx::core::bool_:
      return "bool";
    default:
      return dtype_to_glsl_storage_type(dtype);
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

bool needs_bf16_conversion_helpers(Dtype in_dtype, Dtype out_dtype) {
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

std::string storage_buffer_layout_for_dtype(Dtype dtype, uint32_t binding) {
  return fmt::format(
      "layout({}set = 0, binding = {})",
      (uses_8bit_storage(dtype) || uses_16bit_storage(dtype)) ? "scalar, " : "",
      binding);
}

std::string emit_dynamic_shader_preamble(
    const DynamicShaderPreambleOptions& options) {
  if (!options.use_local_size_x_id && options.local_size_x == 0) {
    throw std::runtime_error("Dynamic Vulkan shader local_size_x must be > 0.");
  }

  const auto features = collect_dynamic_shader_features(options);
  std::ostringstream os;
  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  if (features.needs_int64) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  if (features.uses_float16) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
  }
  if (features.uses_16bit_storage) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  if (features.uses_8bit_storage) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n";
    os << "#extension GL_EXT_shader_8bit_storage : require\n";
  }
  if (features.uses_scalar_block_layout) {
    os << "#extension GL_EXT_scalar_block_layout : require\n";
  }
  if (options.use_local_size_x_id) {
    os << "\nlayout(local_size_x_id = " << options.local_size_x_id
       << ", local_size_y = 1, local_size_z = 1) in;\n\n";
  } else {
    os << "\nlayout(local_size_x = " << options.local_size_x
       << ", local_size_y = 1, local_size_z = 1) in;\n\n";
  }
  return os.str();
}

std::string emit_dynamic_shader_preamble(Dtype dtype, bool needs_int64) {
  DynamicShaderPreambleOptions options;
  options.dtypes = {dtype};
  options.needs_int64 = needs_int64;
  return emit_dynamic_shader_preamble(options);
}

std::string emit_dynamic_shader_preamble(
    Dtype in_dtype,
    Dtype out_dtype,
    bool needs_int64) {
  DynamicShaderPreambleOptions options;
  options.dtypes = {in_dtype, out_dtype};
  options.needs_int64 = needs_int64;
  return emit_dynamic_shader_preamble(options);
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
  if (in == mlx::core::bfloat16 || out == mlx::core::bfloat16) {
    if (in == mlx::core::bfloat16 && out == mlx::core::bfloat16) {
      return expr;
    }
    auto expr_as_float = [&]() {
      if (in == mlx::core::bfloat16) {
        return "bf16_to_fp32(uint(" + expr + "))";
      }
      if (in == mlx::core::complex64) {
        return expr + ".x";
      }
      return "float(" + expr + ")";
    };
    const std::string as_float = expr_as_float();
    if (out == mlx::core::bfloat16) {
      return "uint16_t(fp32_to_bf16(" + as_float + "))";
    }
    if (out == mlx::core::bool_) {
      return "(" + as_float + " != 0.0 ? uint8_t(1) : uint8_t(0))";
    }
    if (out == mlx::core::complex64) {
      return "vec2(" + as_float + ", 0.0)";
    }
    return dtype_to_glsl_storage_type(out) + "(" + as_float + ")";
  }
  if (out == mlx::core::bool_) {
    if (in == mlx::core::complex64) {
      return "((" + expr + ".x != 0.0 || " + expr +
          ".y != 0.0) ? uint8_t(1) : uint8_t(0))";
    }
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
