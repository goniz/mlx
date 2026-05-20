#pragma once

#include <vector>

#include "mlx/array.h"

namespace mlx::core {

bool try_eval_matmul_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s);

bool try_eval_gather_mm_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s);

} // namespace mlx::core
