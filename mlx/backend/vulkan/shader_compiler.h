// Copyright © 2024 Apple Inc.

#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/dtype.h"

namespace mlx::core::vulkan {

// Compile GLSL compute shader source to SPIR-V.
// Throws std::runtime_error on compilation failure.
std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name);

// Returns the GLSL storage type string for a dtype (e.g. "float", "uint16_t").
std::string dtype_to_glsl_storage_type(Dtype dtype);

// Returns true if the dtype requires the float16 GLSL extension.
bool uses_float16_extension(Dtype dtype);

// Returns true if the dtype requires 16-bit storage extensions.
bool uses_16bit_storage(Dtype dtype);

// Returns true if the dtype requires 8-bit storage extensions.
bool uses_8bit_storage(Dtype dtype);

// Emits a GLSL #version + extension preamble for a single-dtype shader.
std::string emit_dynamic_shader_preamble(Dtype dtype, bool needs_int64_output);

// Emits a GLSL #version + extension preamble for a cast shader (two dtypes).
std::string emit_dynamic_shader_preamble(
    Dtype in_dtype,
    Dtype out_dtype,
    bool needs_int64_output);

// Returns a GLSL zero literal for the given dtype.
std::string zero_literal_for_dtype(Dtype dtype);

// Returns a GLSL cast expression, e.g. "float(x)" or "(x != 0 ? 1 : 0)".
std::string cast_expr_for_dtype(
    const std::string& expr,
    Dtype in,
    Dtype out);

} // namespace mlx::core::vulkan