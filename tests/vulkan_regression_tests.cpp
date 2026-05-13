// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test vulkan complex scalar view multiply regression") {
  if (!gpu::is_available()) {
    return;
  }

  auto scalar_source = array(
      {complex64_t{9.0f, 9.0f},
       complex64_t{2.0f, -3.0f},
       complex64_t{7.0f, 1.0f}});
  auto scalar = slice(scalar_source, {1}, {2});
  auto vec = array(
      {complex64_t{1.0f, 2.0f},
       complex64_t{-1.0f, 0.5f},
       complex64_t{0.0f, -4.0f}});

  auto expected = multiply(vec, scalar, Device::cpu);
  auto out = multiply(vec, scalar, Device::gpu);
  CHECK(array_equal(out, expected, Device::cpu).item<bool>());
}

TEST_CASE("test vulkan complex abs general layout regression") {
  if (!gpu::is_available()) {
    return;
  }

  array x(
      {
          complex64_t{3.0f, 4.0f},
          complex64_t{9.0f, 9.0f},
          complex64_t{5.0f, 12.0f},
          complex64_t{8.0f, 15.0f},
      },
      {2, 2});
  auto y = slice(transpose(x, {1, 0}), {0, 0}, {2, 1});

  auto expected = abs(y, Device::cpu);
  auto out = abs(y, Device::gpu);
  CHECK_EQ(out.dtype(), float32);
  CHECK(allclose(out, expected, 1e-6f, 1e-6f, false, Device::cpu).item<bool>());
}
