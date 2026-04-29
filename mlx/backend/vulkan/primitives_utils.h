// Copyright © 2024 Apple Inc.

#pragma once

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mlx::core {

// Trace fallback utilities
bool trace_fallback_enabled();
void trace_fallback(const std::string& msg);
void trace_use_fallback(
    std::string_view primitive_name,
    Stream s,
    std::string_view reason,
    std::string_view details = {});
void trace_vulkan_unsupported(
    std::string_view primitive_name,
    std::string_view reason);

// Type/size checking utilities
uint32_t checked_u32_size(int64_t value, const char* name);
uint32_t checked_mul_u32(uint32_t a, uint32_t b, const char* name);
uint32_t checked_product_u32(const Shape& shape, const char* name);

// Layout checkers
bool is_vulkan_float_dtype(Dtype dtype);
std::string dtype_suffix(Dtype dtype);
std::string gather_index_suffix(Dtype dtype);

enum class BinaryShaderOp {
  Add,
  Divide,
  GreaterEqual,
  Maximum,
  Minimum,
  Multiply,
  Subtract,
};

enum class GenericUnaryShaderOp {
  Abs,
  Ceil,
  Exp,
  Floor,
  Negative,
  Round,
  Sigmoid,
  Sign,
  Tanh,
};

enum class UnaryShaderOp {
  Cos,
  Erf,
  ErfInv,
  Log,
  Sin,
  Square,
  Sqrt,
  Rsqrt,
};

std::optional<vulkan::StaticShaderId> binary_shader_id(
    BinaryShaderOp op,
    Dtype a_dtype,
    Dtype b_dtype,
    Dtype out_dtype,
    bool rte = false);
std::optional<vulkan::StaticShaderId>
generic_unary_shader_id(GenericUnaryShaderOp op, Dtype dtype, bool rte = false);
std::optional<vulkan::StaticShaderId> unary_shader_id(
    UnaryShaderOp op,
    Dtype dtype);
std::optional<vulkan::StaticShaderId> gather_shader_id(
    Dtype value_dtype,
    Dtype index_dtype);
std::optional<vulkan::StaticShaderId> gather_pair_shader_id(
    Dtype value_dtype,
    Dtype index_dtype);
std::optional<vulkan::StaticShaderId> gather_axis_shader_id(
    Dtype value_dtype,
    Dtype index_dtype);
std::optional<vulkan::StaticShaderId> scatter_shader_id(
    Dtype value_dtype,
    Dtype index_dtype);
std::optional<vulkan::StaticShaderId> scatter_sum_shader_id(
    Dtype value_dtype,
    Dtype index_dtype);
std::optional<vulkan::StaticShaderId> scatter_pair_shader_id(
    Dtype value_dtype,
    Dtype index_dtype);
std::optional<vulkan::StaticShaderId> scatter_axis_shader_id(
    Dtype value_dtype,
    Dtype index_dtype);
bool is_supported_elementwise_layout(const array& arr);
bool is_supported_unary_layout(const array& arr);
bool is_supported_generic_unary_layout(const array& arr);

// Axis utilities
int normalize_axis(int axis, int ndim);
bool normalize_unique_axes(
    const std::vector<int>& axes,
    int ndim,
    std::vector<int>& normalized_axes);
bool has_keepdims_axis_shape(const array& in, const array& out, int axis);
bool has_squeezed_axis_shape(const array& in, const array& out, int axis);
Shape keepdims_shape_for_axis(const array& in, int axis);
Shape keepdims_shape_for_axes(const array& in, const std::vector<int>& axes);
bool has_keepdims_axes_shape(
    const array& in,
    const array& out,
    const std::vector<int>& axes);
bool has_squeezed_axes_shape(
    const array& in,
    const array& out,
    const std::vector<int>& axes);

template <typename T, typename = void>
struct is_tuple_like : std::false_type {};

template <typename T>
struct is_tuple_like<
    T,
    std::void_t<decltype(std::tuple_size<std::decay_t<T>>::value)>>
    : std::true_type {};

} // namespace mlx::core
