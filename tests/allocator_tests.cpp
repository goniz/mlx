// Copyright © 2023 Apple Inc.

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "doctest/doctest.h"

#ifdef MLX_BUILD_VULKAN
#include "mlx/backend/vulkan/allocator.h"
#endif
#include "mlx/allocator.h"
#include "mlx/mlx.h"

using namespace mlx::core;

#ifdef MLX_BUILD_VULKAN
namespace {

struct ScopedEnvVar {
  explicit ScopedEnvVar(const char* value) {
    const char* current = std::getenv("MLX_VULKAN_TRACE_SYNC");
    if (current) {
      had_old_value = true;
      old_value = current;
    }
    setenv("MLX_VULKAN_TRACE_SYNC", value, 1);
  }

  ~ScopedEnvVar() {
    if (had_old_value) {
      setenv("MLX_VULKAN_TRACE_SYNC", old_value.c_str(), 1);
    } else {
      unsetenv("MLX_VULKAN_TRACE_SYNC");
    }
  }

  bool had_old_value{false};
  std::string old_value;
};

struct ScopedCerrCapture {
  ScopedCerrCapture() : old_buf(std::cerr.rdbuf(stream.rdbuf())) {}

  ~ScopedCerrCapture() {
    std::cerr.rdbuf(old_buf);
  }

  std::string str() const {
    return stream.str();
  }

  std::ostringstream stream;
  std::streambuf* old_buf;
};

} // namespace
#endif

TEST_CASE("test simple allocations") {
  {
    auto buffer = allocator::malloc(sizeof(float));
    auto fptr = static_cast<float*>(buffer.raw_ptr());
    *fptr = 0.5f;
    CHECK_EQ(*fptr, 0.5f);
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(128 * sizeof(int));
    int* ptr = static_cast<int*>(buffer.raw_ptr());
    for (int i = 0; i < 128; ++i) {
      ptr[i] = i;
    }
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(0);
    allocator::free(buffer);
  }
}

TEST_CASE("test large allocations") {
  size_t size = 1 << 30;
  for (int i = 0; i < 100; ++i) {
    auto buffer = allocator::malloc(size);
    allocator::free(buffer);
  }
}

#ifdef MLX_BUILD_VULKAN
TEST_CASE("test vulkan raw_ptr skips global sync on fresh allocation") {
  if (!gpu::is_available()) {
    return;
  }

  ScopedEnvVar trace_sync("1");
  ScopedCerrCapture capture;

  auto buffer = allocator::malloc(64);
  auto* vk_buffer = static_cast<vulkan::VulkanBuffer*>(buffer.ptr());
  REQUIRE(vk_buffer != nullptr);
  CHECK_FALSE(static_cast<bool>(vk_buffer->last_semaphore));
  CHECK_EQ(vk_buffer->last_timeline_value, 0);

  (void)buffer.raw_ptr();

  allocator::free(buffer);

  auto trace = capture.str();
  CHECK_NE(trace.find("raw_ptr action=skip-sync"), std::string::npos);
  CHECK_EQ(trace.find("sync(all) begin"), std::string::npos);
}
#endif
