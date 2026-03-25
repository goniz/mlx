// Copyright © 2024 Apple Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mlx::core::vulkan {

// Compile GLSL compute shader source to SPIR-V.
// Throws std::runtime_error on compilation failure.
std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name);

} // namespace mlx::core::vulkan
