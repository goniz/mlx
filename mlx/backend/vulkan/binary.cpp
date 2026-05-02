// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/utils.h"

#include <algorithm>
#include <sstream>

namespace mlx::core {

namespace {

array collapse_binary_leading_dims(const array& arr, Stream s) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  return flatten_in_eval(arr, 0, arr.ndim() - 4, s);
}

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

bool is_vulkan_div_cast_dtype(Dtype dtype) {
  switch (dtype) {
    case bool_:
    case int32:
    case int64:
    case uint32:
    case uint64:
    case float16:
    case float32:
    case bfloat16:
      return true;
    default:
      return false;
  }
}

bool is_vulkan_compare_dtype(Dtype dtype) {
  switch (dtype) {
    case float16:
    case float32:
    case bfloat16:
    case int32:
    case int64:
    case uint32:
    case uint64:
      return true;
    default:
      return false;
  }
}

bool has_vulkan_buffer(const array& arr) {
  auto data = arr.data_shared_ptr();
  return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
}

bool ensure_vulkan_buffer(array& arr, Stream s) {
  if (has_vulkan_buffer(arr)) {
    return true;
  }

  if (arr.has_primitive()) {
    arr = contiguous_copy_gpu(arr, s);
    return has_vulkan_buffer(arr);
  }

  if (!arr.has_primitive()) {
    arr.wait();
  }

  if (has_vulkan_buffer(arr)) {
    return true;
  }

  auto data = arr.data_shared_ptr();
  if (data == nullptr || data->buffer.ptr() == nullptr) {
    return false;
  }

  arr = contiguous_copy_gpu(arr, s);
  return has_vulkan_buffer(arr);
}

bool is_same_complex_add(const array& a, const array& b, const array& out) {
  return a.dtype() == complex64 && b.dtype() == complex64 &&
      out.dtype() == complex64;
}

std::string build_complex_add_shader() {
  std::ostringstream os;
  os << vulkan::emit_dynamic_shader_preamble(complex64, complex64, false);
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {vec2 data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {vec2 data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {vec2 data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total_elements) return;\n";
  os << "  out_buf.data[idx + pc.out_offset] = "
        "a_buf.data[idx + pc.a_offset] + b_buf.data[idx + pc.b_offset];\n";
  os << "}\n";
  return os.str();
}

bool try_eval_complex_add_vulkan(array& a, array& b, array& out, Stream s) {
  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == out.shape()) {
      return ensure_vulkan_buffer(in, s);
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
    broadcast(in, view);
    in = view;
    return true;
  };

  if (!materialize_broadcast_input(a) || !materialize_broadcast_input(b)) {
    return false;
  }

  if (!is_supported_elementwise_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }
  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work = staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  if (!is_supported_elementwise_layout(out_work)) {
    return false;
  }

  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out_work, bopt);

  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto out_offset =
      static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 4;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_add_c64_c64_c64",
      build_complex_add_shader(),
      3,
      arrays,
      kPushConstantSize,
      s);
  struct PushConstants {
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &pc);
  vkCmdDispatch(
      dispatch.command_buffer, (pc.total_elements + 255) / 256, 1, 1);
  vulkan::end_command_recording(s.index);
  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

template <typename Primitive>
constexpr vulkan::BinaryDispatchVariant binary_dispatch_variant() {
  if constexpr (std::is_same_v<Primitive, Add>) {
    return vulkan::BinaryDispatchVariant::AddWithPartials;
  } else {
    return vulkan::BinaryDispatchVariant::Standard;
  }
}

template <typename Primitive>
constexpr BinaryShaderOp binary_shader_op() {
  if constexpr (std::is_same_v<Primitive, Add>) {
    return BinaryShaderOp::Add;
  } else if constexpr (std::is_same_v<Primitive, Divide>) {
    return BinaryShaderOp::Divide;
  } else if constexpr (std::is_same_v<Primitive, Maximum>) {
    return BinaryShaderOp::Maximum;
  } else if constexpr (std::is_same_v<Primitive, Minimum>) {
    return BinaryShaderOp::Minimum;
  } else if constexpr (std::is_same_v<Primitive, Multiply>) {
    return BinaryShaderOp::Multiply;
  } else {
    return BinaryShaderOp::Subtract;
  }
}

template <typename Primitive>
bool try_eval_binary_op_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    Stream s) {
  auto trace_binary_unsupported =
      [&](std::string_view reason, const array& lhs, const array& rhs) {
        if (!trace_fallback_enabled()) {
          return;
        }
        std::ostringstream oss;
        oss << "binary_vulkan_unsupported op=" << op_name
            << " reason=" << reason << " lhs_shape=" << lhs.shape()
            << " lhs_dtype=" << lhs.dtype() << " rhs_shape=" << rhs.shape()
            << " rhs_dtype=" << rhs.dtype() << " out_shape=" << out.shape()
            << " out_dtype=" << out.dtype();
        trace_fallback(oss.str());
      };

  if (inputs.size() != 2) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  const bool mixed_numeric_div = std::string_view(op_name) == "div" &&
      out.dtype() == float32 && is_vulkan_div_cast_dtype(a.dtype()) &&
      is_vulkan_div_cast_dtype(b.dtype());
  const bool bool_add = std::is_same_v<Primitive, Add> && a.dtype() == bool_ &&
      b.dtype() == bool_ && out.dtype() == bool_;
  const bool small_signed_integer_case = a.dtype() == b.dtype() &&
      a.dtype() == out.dtype() && (a.dtype() == int8 || a.dtype() == int16);
  const bool small_unsigned_integer_case = a.dtype() == b.dtype() &&
      a.dtype() == out.dtype() && (a.dtype() == uint8 || a.dtype() == uint16);
  const bool float_case = is_vulkan_float_dtype(a.dtype()) &&
      is_vulkan_float_dtype(b.dtype()) && is_vulkan_float_dtype(out.dtype());
  const bool integer_case = a.dtype() == b.dtype() &&
      a.dtype() == out.dtype() && is_vulkan_integer_dtype(a.dtype());
  const bool complex_add = std::is_same_v<Primitive, Add> &&
      is_same_complex_add(a, b, out);
  if (!float_case && !integer_case && !bool_add && !mixed_numeric_div &&
      !complex_add) {
    trace_binary_unsupported("unsupported_dtype_combo", a, b);
    return false;
  }

  if (!ensure_vulkan_buffer(a, s) || !ensure_vulkan_buffer(b, s)) {
    return false;
  }

  if (complex_add) {
    return try_eval_complex_add_vulkan(a, b, out, s);
  }

  if (bool_add) {
    array a_u32(a.shape(), uint32, nullptr, {});
    array b_u32(b.shape(), uint32, nullptr, {});
    copy_gpu(a, a_u32, CopyType::General, s);
    copy_gpu(b, b_u32, CopyType::General, s);
    a = a_u32;
    b = b_u32;
  }

  if (mixed_numeric_div) {
    array a_f32(a.shape(), float32, nullptr, {});
    array b_f32(b.shape(), float32, nullptr, {});
    copy_gpu(a, a_f32, CopyType::General, s);
    copy_gpu(b, b_f32, CopyType::General, s);
    a = a_f32;
    b = b_f32;
  }

  if (small_signed_integer_case) {
    array a_i32(a.shape(), int32, nullptr, {});
    array b_i32(b.shape(), int32, nullptr, {});
    copy_gpu(a, a_i32, CopyType::General, s);
    copy_gpu(b, b_i32, CopyType::General, s);
    a = a_i32;
    b = b_i32;
  } else if (small_unsigned_integer_case) {
    array a_u32(a.shape(), uint32, nullptr, {});
    array b_u32(b.shape(), uint32, nullptr, {});
    copy_gpu(a, a_u32, CopyType::General, s);
    copy_gpu(b, b_u32, CopyType::General, s);
    a = a_u32;
    b = b_u32;
  }

  const bool mixed_bf16_f16 =
      ((a.dtype() == bfloat16 && b.dtype() == float16) ||
       (a.dtype() == float16 && b.dtype() == bfloat16));
  const bool use_f32_staging_io =
      (std::string_view(op_name) == "div" &&
       (a.dtype() == float16 || b.dtype() == float16 ||
        out.dtype() == float16 || a.dtype() == bfloat16 ||
        b.dtype() == bfloat16 || out.dtype() == bfloat16)) ||
      mixed_bf16_f16;
  if (use_f32_staging_io) {
    array a_f32(a.shape(), float32, nullptr, {});
    array b_f32(b.shape(), float32, nullptr, {});
    copy_gpu(a, a_f32, CopyType::General, s);
    copy_gpu(b, b_f32, CopyType::General, s);
    a = a_f32;
    b = b_f32;
  }

  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == out.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
    broadcast(in, view);
    in = view;
    return true;
  };

  if (!materialize_broadcast_input(a) || !materialize_broadcast_input(b)) {
    trace_binary_unsupported("broadcast_materialization_failed", a, b);
    return false;
  }

  if (!is_supported_elementwise_layout(a) && !is_supported_unary_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b) && !is_supported_unary_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }
  a = collapse_binary_leading_dims(a, s);
  b = collapse_binary_leading_dims(b, s);

  const bool staged_output = use_f32_staging_io || small_signed_integer_case ||
      small_unsigned_integer_case || !is_supported_elementwise_layout(out);
  array out_work = staged_output
      ? array(
            out.shape(),
            use_f32_staging_io
                ? float32
                : (small_signed_integer_case
                       ? int32
                       : (small_unsigned_integer_case ? uint32 : out.dtype())),
            nullptr,
            {})
      : out;

  auto bopt = get_binary_op_type(a, b);
  if (small_signed_integer_case || small_unsigned_integer_case) {
    out_work.set_data(allocator::malloc(out_work.nbytes()));
  } else {
    set_binary_op_output_data(a, b, out_work, bopt);
  }
  array out_kernel = collapse_binary_leading_dims(out_work, s);
  if ((!is_supported_elementwise_layout(a) && !is_supported_unary_layout(a)) ||
      (!is_supported_elementwise_layout(b) && !is_supported_unary_layout(b)) ||
      !is_supported_elementwise_layout(out_kernel)) {
    trace_binary_unsupported("unsupported_elementwise_layout", a, b);
    return false;
  }

  if (out_kernel.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  const auto shader_id = bool_add
      ? std::optional<vulkan::StaticShaderId>(
            vulkan::StaticShaderId::maximum_u32_u32_u8)
      : binary_shader_id(
            binary_shader_op<Primitive>(),
            a.dtype(),
            b.dtype(),
            out_kernel.dtype(),
            out_kernel.dtype() == float16);
  if (!shader_id.has_value()) {
    trace_binary_unsupported("missing_shader", a, b);
    return false;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    auto dispatch_variant = binary_dispatch_variant<Primitive>();
    if constexpr (std::is_same_v<Primitive, Add>) {
      if (use_f32_staging_io || bool_add || integer_case) {
        dispatch_variant = vulkan::BinaryDispatchVariant::Standard;
      }
    }
    vulkan::dispatch_binary_op(
        a, b, out_kernel, *shader_id, command_buffer, s, dispatch_variant);
    vulkan::end_command_recording(s.index);
    if (staged_output || use_f32_staging_io) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "binary_dispatch_failed op=" << op_name << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

template <typename Primitive>
void eval_binary_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const char* op_name,
    Stream s) {
  if (!try_eval_binary_op_vulkan<Primitive>(inputs, out, op_name, s)) {
    throw std::runtime_error(
        std::string("Binary operation ") + op_name +
        " failed on Vulkan (unsupported dtype or layout).");
  }
}

bool try_eval_greater_equal_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2 || out.dtype() != bool_) {
    return false;
  }

  array a = inputs[0];
  array b = inputs[1];
  if (!is_vulkan_compare_dtype(a.dtype()) ||
      !is_vulkan_compare_dtype(b.dtype())) {
    return false;
  }
  if (a.dtype() != b.dtype() &&
      !(is_vulkan_float_dtype(a.dtype()) && is_vulkan_float_dtype(b.dtype()))) {
    return false;
  }
  if (a.shape() != out.shape() || b.shape() != out.shape()) {
    return false;
  }

  const bool use_f32_staging_io =
      a.dtype() == bfloat16 || b.dtype() == bfloat16;
  if (use_f32_staging_io) {
    array a_f32(a.shape(), float32, nullptr, {});
    array b_f32(b.shape(), float32, nullptr, {});
    copy_gpu(a, a_f32, CopyType::General, s);
    copy_gpu(b, b_f32, CopyType::General, s);
    a = a_f32;
    b = b_f32;
  }

  if (!is_supported_elementwise_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }

  const auto shader_id = binary_shader_id(
      BinaryShaderOp::GreaterEqual, a.dtype(), b.dtype(), uint8, false);
  if (!shader_id.has_value()) {
    return false;
  }

  array out_u8(out.shape(), uint8, nullptr, {});
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out_u8, bopt, [&](size_t n) {
    return vulkan::allocator().malloc(n);
  });
  if (!is_supported_elementwise_layout(out_u8)) {
    return false;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_binary_op(
        a,
        b,
        out_u8,
        *shader_id,
        command_buffer,
        s,
        vulkan::BinaryDispatchVariant::Standard);
    vulkan::end_command_recording(s.index);
    out.copy_shared_buffer(
        out_u8,
        out_u8.strides(),
        out_u8.flags(),
        out_u8.data_size(),
        out_u8.offset());
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "binary_dispatch_failed op=greater_equal reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

#define VULKAN_BINARY_GPU(func, op_name)                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    eval_binary_vulkan<func>(inputs, out, op_name, stream());         \
  }

VULKAN_BINARY_GPU(Add, "add")
VULKAN_BINARY_GPU(Minimum, "minimum")
VULKAN_BINARY_GPU(Maximum, "maximum")
VULKAN_BINARY_GPU(Divide, "div")
VULKAN_BINARY_GPU(Subtract, "sub")
VULKAN_BINARY_GPU(Multiply, "mul")

void GreaterEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_greater_equal_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "GreaterEqual operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
