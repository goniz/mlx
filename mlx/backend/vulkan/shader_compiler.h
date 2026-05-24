// Copyright © 2024 Apple Inc.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/dtype.h"

namespace mlx::core::vulkan {

enum class DynamicShaderOptimization {
  Zero,
  Size,
  Performance,
};

struct DynamicShaderCompileOptions {
  DynamicShaderOptimization optimization{DynamicShaderOptimization::Performance};
};

// Compile GLSL compute shader source to SPIR-V.
// Throws std::runtime_error on compilation failure.
std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name);
std::vector<uint32_t> compile_glsl_to_spirv(
    const std::string& glsl_source,
    const std::string& shader_name,
    const DynamicShaderCompileOptions& options);

struct DynamicShaderPreambleOptions {
  std::vector<Dtype> dtypes;
  bool needs_int64{false};
  bool needs_scalar_block_layout{false};
  bool use_local_size_x_id{false};
  uint32_t local_size_x{256};
  uint32_t local_size_x_id{0};
};

// Returns the GLSL storage type string for a dtype (e.g. "float", "uint16_t").
std::string dtype_to_glsl_storage_type(Dtype dtype);

// Returns the GLSL compute type string for a dtype (e.g. "bool" for bool_).
std::string dtype_to_glsl_compute_type(Dtype dtype);

// Returns true if the dtype requires the float16 GLSL extension.
bool uses_float16_extension(Dtype dtype);

// Returns true if the dtype requires 16-bit storage extensions.
bool uses_16bit_storage(Dtype dtype);

// Returns true if the dtype requires 8-bit storage extensions.
bool uses_8bit_storage(Dtype dtype);

// Returns true if a dtype pair needs bf16 conversion helper functions.
bool needs_bf16_conversion_helpers(Dtype in_dtype, Dtype out_dtype);

// Emits GLSL helpers for converting bf16 storage bits to and from fp32.
std::string emit_bf16_conversion_helpers();

// Emits a storage buffer layout prefix, adding scalar layout for 8-bit types.
std::string storage_buffer_layout_for_dtype(Dtype dtype, uint32_t binding);

// Emits a GLSL #version + extension preamble from explicit options.
std::string emit_dynamic_shader_preamble(
    const DynamicShaderPreambleOptions& options);

// Emits a GLSL #version + extension preamble for a single-dtype shader.
std::string emit_dynamic_shader_preamble(Dtype dtype, bool needs_int64);

// Emits a GLSL #version + extension preamble for a cast shader (two dtypes).
std::string emit_dynamic_shader_preamble(
    Dtype in_dtype,
    Dtype out_dtype,
    bool needs_int64);

// Returns a GLSL zero literal for the given dtype.
std::string zero_literal_for_dtype(Dtype dtype);

// Returns a GLSL cast expression, e.g. "float(x)" or "(x != 0 ? 1 : 0)".
std::string cast_expr_for_dtype(
    const std::string& expr,
    Dtype in,
    Dtype out);

} // namespace mlx::core::vulkan
