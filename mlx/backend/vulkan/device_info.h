// Copyright © 2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::vulkan {

MLX_API const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index);

} // namespace mlx::core::vulkan
