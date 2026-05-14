// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/vulkan/allocator.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/memory.h"

namespace mlx::core {

namespace allocator {

namespace {

using namespace vk;

bool trace_cpu_access_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_TRACE_CPU_ACCESS");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

} // namespace

Allocator& allocator() {
  return vulkan::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto* buf = static_cast<vulkan::VulkanBuffer*>(ptr_);
  if (trace_cpu_access_enabled()) {
    // Convert memory flags to uint32_t for printing
    uint32_t flags = static_cast<uint32_t>(buf->memory_flags);
    std::cerr << "[vulkan-cpu-access] raw_ptr buffer=" << buf->buffer
              << " mapped=" << buf->mapped_ptr << " size=" << buf->size
              << " alloc=" << buf->allocation_size << " flags=0x" << std::hex
              << flags << std::dec << "\n";
  }

  vulkan::synchronize_buffer_for_host_access(buf);

  return buf->mapped_ptr;
}

} // namespace allocator

namespace vulkan {

namespace {

constexpr size_t min_cache_pool_size = 4096 * 4;
constexpr size_t default_cache_pool_size = 1 << 20;
constexpr size_t opportunistic_decode_pool_size = 4 << 20;
constexpr size_t opportunistic_decode_cache_size = 256 * 1024;
constexpr size_t opportunistic_decode_active_limit = 1152ULL << 20;

struct AllocTraceStats {
  std::string primitive;
  size_t size{0};
  uint64_t mallocs{0};
  uint64_t cache_hits{0};
  uint64_t new_allocs{0};
  uint64_t cached_frees{0};
  uint64_t destroyed_frees{0};
  uint64_t requested_bytes{0};
  uint64_t allocation_bytes{0};
  size_t max_allocation_size{0};
};

bool alloc_summary_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_ALLOC_SUMMARY");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

std::mutex& alloc_trace_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<std::string, AllocTraceStats>& alloc_trace_stats() {
  static auto* stats = new std::unordered_map<std::string, AllocTraceStats>();
  return *stats;
}

thread_local const char* current_alloc_trace_primitive = nullptr;

std::string alloc_trace_key(size_t requested_size) {
  std::string primitive = current_alloc_trace_primitive != nullptr
      ? current_alloc_trace_primitive
      : "(none)";
  return primitive + ":" + std::to_string(requested_size);
}

void trace_alloc(
    size_t requested_size,
    size_t allocation_size,
    bool cache_hit) {
  if (!alloc_summary_enabled()) {
    return;
  }
  std::lock_guard lk(alloc_trace_mutex());
  auto primitive = current_alloc_trace_primitive != nullptr
      ? current_alloc_trace_primitive
      : "(none)";
  auto& stats = alloc_trace_stats()[alloc_trace_key(requested_size)];
  stats.primitive = primitive;
  stats.size = requested_size;
  stats.mallocs++;
  stats.cache_hits += cache_hit ? 1 : 0;
  stats.new_allocs += cache_hit ? 0 : 1;
  stats.requested_bytes += requested_size;
  stats.allocation_bytes += allocation_size;
  stats.max_allocation_size =
      std::max(stats.max_allocation_size, allocation_size);
}

void trace_free(size_t requested_size, bool cached) {
  if (!alloc_summary_enabled()) {
    return;
  }
  std::lock_guard lk(alloc_trace_mutex());
  auto primitive = current_alloc_trace_primitive != nullptr
      ? current_alloc_trace_primitive
      : "(none)";
  auto& stats = alloc_trace_stats()[alloc_trace_key(requested_size)];
  stats.primitive = primitive;
  stats.size = requested_size;
  stats.cached_frees += cached ? 1 : 0;
  stats.destroyed_frees += cached ? 0 : 1;
}

struct AllocTracePrinter {
  ~AllocTracePrinter() {
    if (!alloc_summary_enabled()) {
      return;
    }
    std::vector<AllocTraceStats> rows;
    {
      std::lock_guard lk(alloc_trace_mutex());
      rows.reserve(alloc_trace_stats().size());
      for (const auto& [_, stats] : alloc_trace_stats()) {
        rows.push_back(stats);
      }
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
      return a.requested_bytes > b.requested_bytes;
    });

    std::cerr
        << "[vulkan-alloc-summary] primitive size mallocs hits new cached_free "
        << "destroyed_free total_requested total_allocation max_allocation\n";
    const size_t limit = std::min<size_t>(rows.size(), 64);
    for (size_t i = 0; i < limit; ++i) {
      const auto& stats = rows[i];
      std::cerr << "[vulkan-alloc-summary] " << stats.primitive << " "
                << stats.size << " " << stats.mallocs << " " << stats.cache_hits
                << " " << stats.new_allocs << " " << stats.cached_frees << " "
                << stats.destroyed_frees << " " << stats.requested_bytes << " "
                << stats.allocation_bytes << " " << stats.max_allocation_size
                << "\n";
    }
  }
};

AllocTracePrinter alloc_trace_printer;

size_t query_page_size() {
  auto props = VulkanContext::get().physical_device().getProperties();
  return std::max<size_t>(props.limits.nonCoherentAtomSize, 1);
}

size_t default_max_cacheable_size(size_t max_pool_size) {
  return std::max(min_cache_pool_size, max_pool_size / 8);
}

size_t cacheable_size_limit(size_t active_memory, size_t max_cacheable_size) {
  if (active_memory <= opportunistic_decode_active_limit) {
    return std::max(max_cacheable_size, opportunistic_decode_cache_size);
  }
  return max_cacheable_size;
}

size_t cache_pool_limit(size_t active_memory, size_t max_pool_size) {
  if (max_pool_size == 0) {
    return 0;
  }
  if (active_memory <= opportunistic_decode_active_limit) {
    return std::max(max_pool_size, opportunistic_decode_pool_size);
  }
  return max_pool_size;
}

uint32_t find_memory_type_index(
    const VulkanContext& ctx,
    uint32_t type_filter,
    const std::vector<vk::MemoryPropertyFlags>& preferred_flags) {
  for (auto flags : preferred_flags) {
    try {
      // Convert vk::MemoryPropertyFlags to VkMemoryPropertyFlags
      // This is safe since they're the same underlying type
      VkMemoryPropertyFlags vk_flags =
          static_cast<VkMemoryPropertyFlags>(flags);
      return ctx.find_memory_type(type_filter, vk_flags);
    } catch (const std::runtime_error&) {
    }
  }
  throw std::runtime_error("[vulkan::malloc] No suitable memory type found.");
}

} // namespace

void set_alloc_trace_primitive(const char* name) {
  current_alloc_trace_primitive = name;
}

void clear_alloc_trace_primitive() {
  current_alloc_trace_primitive = nullptr;
}

VulkanAllocator::VulkanAllocator()
    : buffer_cache_(
          query_page_size(),
          [](VulkanBuffer* buf) { return buf->allocation_size; },
          [this](VulkanBuffer* buf) { free_vulkan_buffer(buf); }) {
  const auto& info = gpu::device_info(0);
  auto memory_size = std::get<size_t>(info.at("memory_size"));
  auto max_rec_size =
      std::get<size_t>(info.at("max_recommended_working_set_size"));

  if (memory_size == 0) {
    memory_size = 1UL << 33;
  }
  if (max_rec_size == 0) {
    max_rec_size = memory_size;
  }

  resource_limit_ = std::get<size_t>(info.at("resource_limit"));
  if (resource_limit_ == 0) {
    resource_limit_ = std::numeric_limits<size_t>::max();
  }

  block_limit_ = std::min(
      static_cast<size_t>(1.5 * static_cast<double>(max_rec_size)),
      static_cast<size_t>(0.95 * static_cast<double>(memory_size)));
  gc_limit_ = std::min(
      static_cast<size_t>(0.95 * static_cast<double>(max_rec_size)),
      block_limit_);
  max_pool_size_ = std::max(
      min_cache_pool_size,
      std::min({default_cache_pool_size, gc_limit_ / 4, block_limit_}));
  max_cacheable_size_ = default_max_cacheable_size(max_pool_size_);
}

size_t VulkanAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, max_pool_size_);
  max_cacheable_size_ = default_max_cacheable_size(max_pool_size_);
  if (get_cache_memory() > max_pool_size_) {
    num_resources_ -= buffer_cache_.release_cached_buffers(
        get_cache_memory() - max_pool_size_);
  }
  return limit;
}

size_t VulkanAllocator::set_memory_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, block_limit_);
  gc_limit_ = std::min(gc_limit_, block_limit_);
  max_pool_size_ =
      std::max(min_cache_pool_size, std::min(max_pool_size_, block_limit_));
  max_cacheable_size_ = default_max_cacheable_size(max_pool_size_);
  return limit;
}

size_t VulkanAllocator::get_memory_limit() const {
  return block_limit_;
}

size_t VulkanAllocator::set_wired_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(limit, wired_limit_);
  return limit;
}

void VulkanAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  num_resources_ -= buffer_cache_.clear();
}

bool VulkanAllocator::owns(Buffer buffer) const {
  auto* vk_buffer = static_cast<VulkanBuffer*>(buffer.ptr());
  if (vk_buffer == nullptr) {
    return false;
  }

  std::unique_lock lk(mutex_);
  return live_buffers_.contains(vk_buffer);
}

Buffer VulkanAllocator::malloc(size_t size) {
  if (size == 0) {
    return Buffer{nullptr};
  }

  // Try to reuse a buffer from the cache.
  {
    std::unique_lock lk(mutex_);
    auto* cached = buffer_cache_.reuse_from_cache(size);
    if (cached) {
      if (cached->allocation_size > size + buffer_cache_.page_size()) {
        buffer_cache_.recycle_to_cache(cached);
      } else {
        cached->size = size;
        active_memory_ += cached->size;
        peak_memory_ = std::max(peak_memory_, active_memory_);
        live_buffers_.insert(cached);
        trace_alloc(size, cached->allocation_size, true);
        return Buffer{static_cast<void*>(cached)};
      }
    }

    // If we have memory pressure, try to reclaim from the cache.
    int64_t mem_to_free =
        get_active_memory() + get_cache_memory() + size - gc_limit_;
    if (mem_to_free > 0) {
      num_resources_ -= buffer_cache_.release_cached_buffers(mem_to_free);
    }

    if (num_resources_ >= resource_limit_) {
      num_resources_ -=
          buffer_cache_.release_cached_buffers(get_cache_memory());
      if (num_resources_ >= resource_limit_) {
        std::ostringstream msg;
        msg << "[vulkan::malloc] Resource limit (" << resource_limit_
            << ") exceeded.";
        throw std::runtime_error(msg.str());
      }
    }
  }

  auto& ctx = VulkanContext::get();
  auto vk_device = ctx.device();

  // Use C++ Vulkan API for buffer creation
  vk::BufferCreateInfo buffer_info(vk::BufferCreateFlags(), size);
  buffer_info.usage = vk::BufferUsageFlagBits::eStorageBuffer |
      vk::BufferUsageFlagBits::eTransferSrc |
      vk::BufferUsageFlagBits::eTransferDst;
  std::array<uint32_t, 2> queue_family_indices = {
      ctx.compute_queue_family_index(), ctx.transfer_queue_family_index()};
  if (ctx.has_separate_transfer_queue() &&
      ctx.compute_queue_family_index() != ctx.transfer_queue_family_index()) {
    buffer_info.sharingMode = vk::SharingMode::eConcurrent;
    buffer_info.queueFamilyIndexCount = 2;
    buffer_info.pQueueFamilyIndices = queue_family_indices.data();
  } else {
    buffer_info.sharingMode = vk::SharingMode::eExclusive;
  }

  vk::Buffer vk_buffer = vk_device.createBuffer(buffer_info);

  vk::MemoryRequirements mem_requirements =
      vk_device.getBufferMemoryRequirements(vk_buffer);
  const size_t allocation_size = static_cast<size_t>(mem_requirements.size);

  {
    std::unique_lock lk(mutex_);
    if (num_resources_ >= resource_limit_) {
      vk_device.destroyBuffer(vk_buffer);
      std::ostringstream msg;
      msg << "[vulkan::malloc] Resource limit (" << resource_limit_
          << ") exceeded.";
      throw std::runtime_error(msg.str());
    }
  }

  std::vector<vk::MemoryPropertyFlags> preferred_memory_types;
  if (ctx.is_unified_memory()) {
    preferred_memory_types = {
        vk::MemoryPropertyFlagBits::eDeviceLocal |
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent};
  } else {
    preferred_memory_types = {
        vk::MemoryPropertyFlagBits::eDeviceLocal |
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eHostCached};
  }

  uint32_t memory_type_index = 0;
  try {
    memory_type_index = find_memory_type_index(
        ctx, mem_requirements.memoryTypeBits, preferred_memory_types);
  } catch (...) {
    vk_device.destroyBuffer(vk_buffer);
    throw;
  }

  auto mem_props = ctx.memory_properties();
  const auto memory_flags =
      mem_props.memoryTypes[memory_type_index].propertyFlags;

  // Use C++ Vulkan API for memory allocation
  vk::MemoryAllocateInfo alloc_info(mem_requirements.size, memory_type_index);
  vk::DeviceMemory vk_memory = vk_device.allocateMemory(alloc_info);

  // Bind memory - C++ API doesn't have this method, so use C function
  if (vkBindBufferMemory(vk_device, vk_buffer, vk_memory, 0) != VK_SUCCESS) {
    vk_device.freeMemory(vk_memory);
    vk_device.destroyBuffer(vk_buffer);
    throw std::runtime_error("[vulkan::malloc] Failed to bind buffer memory.");
  }

  void* mapped_ptr = nullptr;
  if (memory_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
    // Use C++ API to map memory
    mapped_ptr = vk_device.mapMemory(vk_memory, 0, VK_WHOLE_SIZE);
  }

  auto* buf = new VulkanBuffer{
      mapped_ptr, vk_buffer, vk_memory, size, allocation_size, memory_flags};

  {
    std::unique_lock lk(mutex_);
    active_memory_ += buf->size;
    peak_memory_ = std::max(peak_memory_, active_memory_);
    num_resources_++;
    live_buffers_.insert(buf);
    trace_alloc(size, buf->allocation_size, false);

    // Maintain the cache below the requested limit.
    const auto pool_limit = cache_pool_limit(active_memory_, max_pool_size_);
    if (get_cache_memory() > pool_limit) {
      num_resources_ -= buffer_cache_.release_cached_buffers(
          get_cache_memory() - pool_limit);
    }
  }

  return Buffer{static_cast<void*>(buf)};
}

void VulkanAllocator::free(Buffer buffer) {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }

  {
    std::unique_lock lk(mutex_);
    if (live_buffers_.find(buf) == live_buffers_.end()) {
      return;
    }
    live_buffers_.erase(buf);
    active_memory_ -= std::min(active_memory_, buf->size);

    const auto cacheable_limit =
        cacheable_size_limit(active_memory_, max_cacheable_size_);
    const auto pool_limit = cache_pool_limit(active_memory_, max_pool_size_);
    if (buf->allocation_size <= cacheable_limit &&
        get_cache_memory() + buf->allocation_size <= pool_limit) {
      buffer_cache_.recycle_to_cache(buf);
      trace_free(buf->size, true);
      return;
    }

    num_resources_ -= std::min<size_t>(1, num_resources_);
    trace_free(buf->size, false);
  }

  free_vulkan_buffer(buf);
}

void VulkanAllocator::free_vulkan_buffer(VulkanBuffer* buf) {
  auto vk_device = VulkanContext::get().device();
  vk_device.destroyBuffer(buf->buffer);
  vk_device.freeMemory(buf->memory);
  delete buf;
}

size_t VulkanAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  return buf ? buf->size : 0;
}

Buffer VulkanAllocator::make_buffer(void*, size_t) {
  // Vulkan no-copy host-pointer import requires optional extensions and
  // additional device setup that is not enabled yet.
  return Buffer{nullptr};
}

void VulkanAllocator::release(Buffer) {
  // No-op because make_buffer currently returns nullptr.
}

VulkanAllocator& allocator() {
  static auto* allocator_ = new VulkanAllocator();
  return *allocator_;
}

bool is_vulkan_buffer(Buffer buffer) {
  return allocator().owns(buffer);
}

} // namespace vulkan

size_t set_cache_limit(size_t limit) {
  return vulkan::allocator().set_cache_limit(limit);
}

size_t set_memory_limit(size_t limit) {
  return vulkan::allocator().set_memory_limit(limit);
}

size_t get_memory_limit() {
  return vulkan::allocator().get_memory_limit();
}

size_t set_wired_limit(size_t limit) {
  return vulkan::allocator().set_wired_limit(limit);
}

size_t get_active_memory() {
  return vulkan::allocator().get_active_memory();
}

size_t get_peak_memory() {
  return vulkan::allocator().get_peak_memory();
}

void reset_peak_memory() {
  vulkan::allocator().reset_peak_memory();
}

size_t get_cache_memory() {
  return vulkan::allocator().get_cache_memory();
}

void clear_cache() {
  vulkan::allocator().clear_cache();
}

} // namespace mlx::core
