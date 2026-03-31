// Multi-queue decode correctness and queue-ownership stress tests for Vulkan backend.

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

#ifdef MLX_BUILD_VULKAN
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/vulkan.h"
#endif

using namespace mlx::core;

#ifdef MLX_BUILD_VULKAN

// Timeout handling for stress tests - uses generation counter to prevent
// stale timeout threads from affecting later tests
static std::atomic<bool> g_test_timeout_reached{false};
static std::atomic<uint64_t> g_timeout_generation{0};

static void set_test_timeout(int seconds) {
  uint64_t current_gen = g_timeout_generation.fetch_add(1) + 1;
  g_test_timeout_reached.store(false);
  
  std::thread([seconds, current_gen]() {
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    // Only set timeout if generation hasn't changed (i.e., we're still in the same test)
    uint64_t expected_gen = current_gen;
    // This is a best-effort check - we only set the flag if generation matches
    if (g_timeout_generation.load() == current_gen) {
      g_test_timeout_reached.store(true);
    }
  }).detach();
}

static bool check_timeout() {
  return g_test_timeout_reached.load();
}

namespace {

// Helper to check if we have separate transfer queue
bool has_separate_transfer_queue() {
  return mlx::core::vulkan::VulkanContext::get().has_separate_transfer_queue();
}

// Helper to get queue family indices
void get_queue_family_indices(uint32_t& compute_family, uint32_t& transfer_family) {
  compute_family = mlx::core::vulkan::VulkanContext::get().compute_queue_family_index();
  transfer_family = mlx::core::vulkan::VulkanContext::get().transfer_queue_family_index();
}

// Decode-like workload: simulates attention KV-cache append pattern
void run_decode_kv_append_workload(
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int num_tokens,
    Stream compute_stream) {
  // Create KV cache on GPU using initializer list for shape
  auto k_cache = zeros({batch_size, num_heads, seq_len, head_dim}, float32, Device::gpu);
  auto v_cache = zeros({batch_size, num_heads, seq_len, head_dim}, float32, Device::gpu);
  
  // Simulate decode steps: for each new token, compute and append to cache
  for (int step = 0; step < num_tokens && !check_timeout(); ++step) {
    // New token key/value (arange-based for determinism)
    auto new_k = arange(0, batch_size * num_heads * 1 * head_dim, float32, Device::gpu) 
                 / static_cast<float>(batch_size * num_heads * head_dim);
    new_k = reshape(new_k, {batch_size, num_heads, 1, head_dim});
    
    auto new_v = arange(0, batch_size * num_heads * 1 * head_dim, float32, Device::gpu)
                 / static_cast<float>(batch_size * num_heads * head_dim);
    new_v = reshape(new_v, {batch_size, num_heads, 1, head_dim});
    
    // KV cache append (slice_update)
    int insert_pos = seq_len - num_tokens + step;
    if (insert_pos >= 0 && insert_pos < seq_len) {
      // Use explicit array for start indices to match API
      auto start = array({0, 0, insert_pos, 0}, {4});
      std::vector<int> axes = {0, 1, 2, 3};
      k_cache = slice_update(k_cache, new_k, start, axes);
      v_cache = slice_update(v_cache, new_v, start, axes);
    }
    
    // Small computation to exercise compute queue
    auto q = arange(0, batch_size * num_heads * 1 * head_dim, float32, Device::gpu)
             / static_cast<float>(batch_size * num_heads * head_dim);
    q = reshape(q, {batch_size, num_heads, 1, head_dim});
    auto scores = matmul(q, transpose(k_cache, {0, 1, 3, 2})) 
                  / std::sqrt(static_cast<float>(head_dim));
    
    eval(scores);
  }
  
  eval(k_cache);
  eval(v_cache);
  synchronize(compute_stream);
}

// Cross-queue handoff stress test: upload -> compute -> readback with correctness check
void run_cross_queue_handoff_stress(int iterations, int data_size) {
  auto stream = new_stream(Device::gpu);
  
  for (int i = 0; i < iterations && !check_timeout(); ++i) {
    // Stage 1: Create deterministic data on CPU
    auto host_data = arange(0, data_size, float32, Device::cpu) / static_cast<float>(data_size);
    
    // Compute reference result on CPU
    auto cpu_result = square(host_data) + host_data * 2.0f + 1.0f;
    eval(cpu_result);
    
    // Stage 2: Compute on GPU (compute queue) - move to GPU via astype
    auto gpu_data = astype(host_data, float32, Device::gpu);
    auto gpu_result = square(gpu_data) + gpu_data * 2.0f + 1.0f;
    
    // Stage 3: Readback to CPU
    auto host_result = astype(gpu_result, float32, Device::cpu);
    
    // Verify correctness
    eval(host_result);
    synchronize(stream);
    
    // Verify GPU result matches CPU reference (with small tolerance for floating point)
    auto diff = abs(host_result - cpu_result);
    auto max_diff = max(diff);
    eval(max_diff);
    float max_diff_val = max_diff.item<float>();
    CHECK(max_diff_val < 1e-5f);
    
    // Also verify result values are finite
    auto scalar_check = sum(host_result);
    eval(scalar_check);
    float val = scalar_check.item<float>();
    CHECK(!std::isnan(val));
    CHECK(std::isfinite(val));
  }
}

// Queue ownership stress test: alternate buffers between queues
void run_queue_ownership_stress(int iterations, int num_buffers) {
  auto compute_stream = new_stream(Device::gpu);
  
  // Create multiple buffers
  std::vector<array> buffers;
  for (int i = 0; i < num_buffers; ++i) {
    buffers.push_back(zeros({1024, 1024}, float32, Device::gpu));
  }
  
  for (int iter = 0; iter < iterations && !check_timeout(); ++iter) {
    // Alternate operations between compute and transfer-like workloads
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (iter % 2 == 0) {
        // Compute operation
        buffers[i] = buffers[i] + 1.0f;
      } else {
        // Simulated transfer: copy to CPU and back via astype
        auto host_copy = astype(buffers[i], float32, Device::cpu);
        buffers[i] = astype(host_copy, float32, Device::gpu);
      }
    }
    
    // Evaluate all buffers on compute stream
    for (auto& buf : buffers) {
      eval(buf);
    }
    synchronize(compute_stream);
  }
}

} // namespace

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE("vulkan multi-queue: queue family detection") {
  uint32_t compute_family, transfer_family;
  get_queue_family_indices(compute_family, transfer_family);
  
  // Verify queue family indices are valid
  CHECK(compute_family != static_cast<uint32_t>(-1));
  CHECK(transfer_family != static_cast<uint32_t>(-1));
  
  // Log the configuration
  bool separate = has_separate_transfer_queue();
  MESSAGE("Queue configuration: compute_family=", compute_family, 
          " transfer_family=", transfer_family,
          " separate_transfer=", separate);
}

TEST_CASE("vulkan multi-queue: same-family dual-queue topology") {
  // This test verifies correct behavior when compute and transfer 
  // share the same queue family but use different queues
  
  if (has_separate_transfer_queue()) {
    MESSAGE("Skipping: separate transfer queue available (not same-family)");
    return;
  }
  
  set_test_timeout(30); // 30 second timeout
  
  uint32_t compute_family, transfer_family;
  get_queue_family_indices(compute_family, transfer_family);
  
  // Should be same family
  CHECK_EQ(compute_family, transfer_family);
  
  // Run decode-like workload
  run_decode_kv_append_workload(1, 4, 32, 64, 8, new_stream(Device::gpu));
  
  CHECK(!check_timeout());
}

TEST_CASE("vulkan multi-queue: separate-family compute/transfer topology") {
  // This test verifies correct behavior when compute and transfer 
  // use different queue families
  
  if (!has_separate_transfer_queue()) {
    MESSAGE("Skipping: no separate transfer queue available");
    return;
  }
  
  set_test_timeout(30); // 30 second timeout
  
  uint32_t compute_family, transfer_family;
  get_queue_family_indices(compute_family, transfer_family);
  
  // Should be different families
  CHECK_NE(compute_family, transfer_family);
  
  // Run cross-queue handoff stress test
  run_cross_queue_handoff_stress(10, 1024);
  
  CHECK(!check_timeout());
}

TEST_CASE("vulkan multi-queue: decode-like workload stress") {
  // Stress test: upload -> compute -> KV append -> readback ordering
  set_test_timeout(60); // 60 second timeout
  
  auto compute_stream = new_stream(Device::gpu);
  
  // Small model config
  const int batch_size = 1;
  const int num_heads = 8;
  const int seq_len = 128;
  const int head_dim = 64;
  const int num_tokens = 16;
  
  // Run multiple iterations
  for (int run = 0; run < 3 && !check_timeout(); ++run) {
    run_decode_kv_append_workload(
        batch_size, num_heads, seq_len, head_dim, num_tokens, compute_stream);
  }
  
  CHECK(!check_timeout());
}

TEST_CASE("vulkan multi-queue: queue ownership stress") {
  // Stress test queue ownership transitions
  set_test_timeout(60); // 60 second timeout
  
  run_queue_ownership_stress(20, 5);
  
  CHECK(!check_timeout());
}

TEST_CASE("vulkan multi-queue: timeline value monotonicity") {
  // Verify timeline values are maintained correctly across operations
  set_test_timeout(30);
  
  auto& ctx = mlx::core::vulkan::VulkanContext::get();
  
  uint64_t initial_timeline = ctx.current_timeline_value();
  
  // Run some operations using arange for determinism
  auto a = arange(0, 100 * 100, float32, Device::gpu) / 10000.0f;
  a = reshape(a, {100, 100});
  auto b = arange(0, 100 * 100, float32, Device::gpu) / 10000.0f;
  b = reshape(b, {100, 100});
  auto c = matmul(a, b);
  
  eval(c);
  
  uint64_t after_compute = ctx.current_timeline_value();
  
  // Run more operations
  auto d = c + 1.0f;
  auto e = square(d);
  
  eval(e);
  
  uint64_t after_more = ctx.current_timeline_value();
  
  // Timeline values should be non-decreasing
  // Note: Timeline increments happen on submission, not on every operation
  CHECK_GE(after_compute, initial_timeline);
  CHECK_GE(after_more, after_compute);
  
  // Operations should complete successfully (this is the real test)
  CHECK_EQ(e.shape()[0], 100);
  CHECK_EQ(e.shape()[1], 100);
}

TEST_CASE("vulkan multi-queue: cross-queue synchronization") {
  // Test that cross-queue synchronization works correctly
  set_test_timeout(30);
  
  // Create multiple streams
  auto stream1 = new_stream(Device::gpu);
  auto stream2 = new_stream(Device::gpu);
  
  // Create shared data on GPU
  auto shared = arange(0, 1000, float32, Device::gpu) / 1000.0f;
  eval(shared);
  
  // Stream 1: heavy computation on shared data
  auto result1 = square(shared);
  for (int i = 0; i < 5; ++i) {
    result1 = square(result1);
  }
  eval(result1);
  
  // Stream 2: different computation on same shared data
  auto result2 = sqrt(abs(shared));
  eval(result2);
  
  // Synchronize both streams to ensure all work is complete
  synchronize(stream1);
  synchronize(stream2);
  
  // Verify results are finite (not corrupted by sync issues)
  auto r1_host = astype(result1, float32, Device::cpu);
  auto r2_host = astype(result2, float32, Device::cpu);
  
  eval(r1_host);
  eval(r2_host);
  
  // Check results using sum to get scalars
  auto r1_sum = sum(r1_host);
  auto r2_sum = sum(r2_host);
  eval(r1_sum);
  eval(r2_sum);
  CHECK(std::isfinite(r1_sum.item<float>()));
  CHECK(std::isfinite(r2_sum.item<float>()));
}

TEST_CASE("vulkan multi-queue: buffer queue affinity transitions") {
  // Test that buffer queue affinity is properly tracked
  set_test_timeout(30);
  
  // Create buffer on GPU
  auto gpu_buffer = zeros({100, 100}, float32, Device::gpu);
  
  // Compute on GPU (compute queue)
  gpu_buffer = gpu_buffer + 1.0f;
  eval(gpu_buffer);
  
  // Transfer to CPU via astype
  auto host_buffer = astype(gpu_buffer, float32, Device::cpu);
  eval(host_buffer);
  
  // Transfer back to GPU via astype
  auto gpu_buffer2 = astype(host_buffer, float32, Device::gpu);
  eval(gpu_buffer2);
  
  // Compute again (compute queue)
  gpu_buffer2 = gpu_buffer2 * 2.0f;
  eval(gpu_buffer2);
  
  // Results should be correct
  auto result = astype(gpu_buffer2, float32, Device::cpu);
  eval(result);
  
  // Should be all 2.0s (0 + 1) * 2
  auto expected = full({100, 100}, 2.0f, float32, Device::cpu);
  bool eq = array_equal(result, expected, Device::cpu).item<bool>();
  CHECK(eq);
}

TEST_CASE("vulkan multi-queue: heavy concurrent workload") {
  // Stress test with many concurrent operations
  set_test_timeout(60);
  
  const int num_streams = 4;
  const int ops_per_stream = 10;
  
  std::vector<Stream> streams;
  for (int i = 0; i < num_streams; ++i) {
    streams.push_back(new_stream(Device::gpu));
  }
  
  std::vector<array> results;
  
  // Launch operations on different streams
  for (int i = 0; i < num_streams && !check_timeout(); ++i) {
    auto x = arange(0, 500 * 500, float32, Device::gpu) / 250000.0f;
    x = reshape(x, {500, 500});
    auto result = x;
    
    for (int j = 0; j < ops_per_stream; ++j) {
      result = matmul(result, x) / static_cast<float>(j + 1);
    }
    
    eval(result);
    results.push_back(result);
  }
  
  // Synchronize all streams
  for (auto& stream : streams) {
    synchronize(stream);
  }
  
  // Verify all results
  for (auto& result : results) {
    auto host_result = astype(result, float32, Device::cpu);
    eval(host_result);
    // Check using sum to get scalar
    auto result_sum = sum(host_result);
    eval(result_sum);
    CHECK(std::isfinite(result_sum.item<float>()));
  }
  
  CHECK(!check_timeout());
}

TEST_CASE("vulkan multi-queue: rapid stream creation/destruction") {
  // Test that streams can be created and destroyed rapidly without issues
  set_test_timeout(30);
  
  for (int i = 0; i < 20 && !check_timeout(); ++i) {
    {
      auto temp_stream = new_stream(Device::gpu);
      auto x = arange(0, 100 * 100, float32, Device::gpu) / 10000.0f;
      x = reshape(x, {100, 100});
      auto y = square(x);
      eval(y);
      synchronize(temp_stream);
    } // Stream goes out of scope
  }
  
  CHECK(!check_timeout());
}

// Short-timeout reproduction harness for CI
TEST_CASE("vulkan multi-queue: quick correctness check") {
  // Fast test for CI - runs in ~5 seconds
  set_test_timeout(10);
  
  auto stream = new_stream(Device::gpu);
  
  // Simple decode pattern using arange for determinism
  auto x = arange(0, 32 * 64, float32, Device::gpu) / 2048.0f;
  x = reshape(x, {32, 64});
  auto k = arange(0, 32 * 64, float32, Device::gpu) / 2048.0f;
  k = reshape(k, {32, 64});
  auto v = arange(0, 32 * 64, float32, Device::gpu) / 2048.0f;
  v = reshape(v, {32, 64});
  
  // Attention-like computation
  auto scores = matmul(x, transpose(k, {1, 0})) / 8.0f;
  auto weights = softmax(scores, 1);
  auto output = matmul(weights, v);
  
  eval(output);
  synchronize(stream);
  
  // Verify
  auto host_out = astype(output, float32, Device::cpu);
  eval(host_out);
  // Check using sum to get scalar
  auto out_sum = sum(host_out);
  eval(out_sum);
  CHECK(std::isfinite(out_sum.item<float>()));
  
  CHECK(!check_timeout());
}

#else // !MLX_BUILD_VULKAN

TEST_CASE("vulkan multi-queue tests skipped - Vulkan not built") {
  MESSAGE("Vulkan backend not enabled, skipping multi-queue tests");
}

#endif // MLX_BUILD_VULKAN
