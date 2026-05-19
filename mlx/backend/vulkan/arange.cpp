// Copyright © 2024 Apple Inc.

#include <cmath>
#include <vector>

#include "mlx/backend/gpu/copy.h"
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

  if ((out.dtype() == float16 || out.dtype() == bfloat16) &&
      start == std::trunc(start) && step == std::trunc(step)) {
    // Match CPU low-precision arange semantics by advancing in the target dtype
    // instead of recomputing each element from float32 math in the shader.
    const auto n = out.size();
    out.set_data(allocator::malloc(out.nbytes()));
    if (n == 0) {
      return true;
    }
    if (out.dtype() == float16) {
      auto* dst = out.data<float16_t>();
      float16_t value(static_cast<float>(start));
      const float16_t step_value(static_cast<float>(step));
      for (size_t i = 0; i < n; ++i) {
        dst[i] = value;
        value = float16_t(static_cast<float>(value) + static_cast<float>(step_value));
      }
      return true;
    }

    auto* dst = out.data<bfloat16_t>();
    bfloat16_t value(static_cast<float>(start));
    const bfloat16_t step_value(static_cast<float>(step));
    for (size_t i = 0; i < n; ++i) {
      dst[i] = value;
      value = bfloat16_t(static_cast<float>(value) + static_cast<float>(step_value));
    }
    return true;
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
