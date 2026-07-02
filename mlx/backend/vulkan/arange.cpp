// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/dtype.h"

namespace mlx::core {

namespace {

std::optional<vulkan::StaticShaderId> arange_shader_id(Dtype dtype) {
  switch (dtype) {
    case uint8:
      return vulkan::StaticShaderId::arange_u8;
    case uint16:
      return vulkan::StaticShaderId::arange_u16;
    case uint32:
      return vulkan::StaticShaderId::arange_u32;
    case int8:
      return vulkan::StaticShaderId::arange_i8;
    case int16:
      return vulkan::StaticShaderId::arange_i16;
    case int32:
      return vulkan::StaticShaderId::arange_i32;
    case int64:
      return vulkan::StaticShaderId::arange_i64;
    case float16:
      return vulkan::StaticShaderId::arange_f16;
    case bfloat16:
      return vulkan::StaticShaderId::arange_bf16;
    case float32:
      return vulkan::StaticShaderId::arange_f32;
    default:
      return std::nullopt;
  }
}

bool try_eval_arange_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s,
    double start,
    double step) {
  if (!inputs.empty() || !is_supported_generic_unary_layout(out)) {
    return false;
  }

  auto shader_id = arange_shader_id(out.dtype());
  if (!shader_id.has_value()) {
    return false;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_arange_op(
        out,
        *shader_id,
        command_buffer,
        s,
        start,
        step);
    vulkan::end_command_recording(s.index);

    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "arange_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [start, stop, step] = state();
  if (!try_eval_arange_vulkan(inputs, out, stream(), start, step)) {
    throw std::runtime_error(
        "Arange operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
