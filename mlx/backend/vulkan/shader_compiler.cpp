// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/shader_compiler.h"

#include <fmt/format.h>
#include <shaderc/shaderc.hpp>

namespace mlx::core::vulkan {

std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name) {
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

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    throw std::runtime_error(
        fmt::format(
            "Failed to compile Vulkan kernel '{}': {}",
            shader_name,
            result.GetErrorMessage()));
  }

  return std::vector<uint32_t>(result.cbegin(), result.cend());
}

} // namespace mlx::core::vulkan
