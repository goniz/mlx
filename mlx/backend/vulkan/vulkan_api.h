// Copyright © 2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::vulkan {

/* Check if the Vulkan backend is available. */
MLX_API bool is_available();

/* Get the number of available Vulkan devices. */
MLX_API int device_count();

/* Get information about a Vulkan device. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info(int device_index = 0);

} // namespace mlx::core::vulkan
