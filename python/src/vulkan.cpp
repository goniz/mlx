// Copyright © 2024 Apple Inc.

#include <nanobind/nanobind.h>

#include "mlx/backend/vulkan/vulkan_api.h"

namespace mx = mlx::core;
namespace nb = nanobind;

void init_vulkan(nb::module_& m) {
  nb::module_ vulkan = m.def_submodule("vulkan", "mlx.vulkan");

  vulkan.def(
      "is_available",
      &mx::vulkan::is_available,
      R"pbdoc(
      Check if the Vulkan back-end is available.
      )pbdoc");
}
