// Copyright © 2024 Apple Inc.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

// Use C++ Vulkan API instead of C API
#include <vulkan/vulkan.hpp>

#include "mlx/api.h"

namespace mlx::core::vulkan {

MLX_API bool is_available();
MLX_API bool is_unified_memory();
MLX_API int device_count();

enum class GpuArchitecture : uint8_t {
  Unknown,
  AmdRdna,
  AmdCdna,
  Nvidia,
  Intel,
  Apple,
  Qualcomm,
};

class VulkanContext {
 public:
  static VulkanContext& get();

  // C++ Vulkan API accessors
  const vk::Instance& instance() const {
    return instance_;
  }
  const vk::PhysicalDevice& physical_device() const {
    return physical_device_;
  }
  const vk::Device& device() const {
    return device_;
  }
  const vk::Queue& compute_queue() const {
    return compute_queue_;
  }
  const vk::Queue& transfer_queue() const {
    return transfer_queue_;
  }
  uint32_t compute_queue_family_index() const {
    return compute_queue_family_index_;
  }
  uint32_t transfer_queue_family_index() const {
    return transfer_queue_family_index_;
  }
  bool has_separate_transfer_queue() const {
    return has_separate_transfer_queue_;
  }

  // Timeline semaphore accessors
  vk::Semaphore timeline_semaphore() const {
    return timeline_semaphore_;
  }
  uint64_t current_timeline_value() const {
    return timeline_value_.load();
  }
  uint64_t increment_timeline();

  // Memory properties
  bool is_unified_memory() const {
    return is_unified_memory_;
  }
  const vk::PhysicalDeviceMemoryProperties& memory_properties() const {
    return mem_properties_;
  }
  bool shader_float16_supported() const {
    return shader_float16_supported_;
  }
  bool shader_bfloat16_supported() const;
  bool subgroup_size_control_supported() const {
    return subgroup_size_control_supported_;
  }
  bool subgroup_require_full_support() const {
    return subgroup_require_full_support_;
  }
  uint32_t subgroup_min_size() const {
    return subgroup_min_size_;
  }
  uint32_t subgroup_max_size() const {
    return subgroup_max_size_;
  }
  uint32_t subgroup_size() const {
    return subgroup_size_;
  }
  bool pipeline_robustness_supported() const {
    return pipeline_robustness_supported_;
  }
  bool cooperative_matrix_supported() const {
    return cooperative_matrix_supported_;
  }
  bool coopmat_flash_attention_f32acc_supported() const {
    return coopmat_flash_attention_f32acc_supported_;
  }
  bool coopmat2_conv2d_supported() const {
    return coopmat2_conv2d_supported_;
  }
  bool integer_dot_product_supported() const {
    return integer_dot_product_supported_;
  }
  bool push_descriptor_supported() const {
    return push_descriptor_supported_;
  }
  // Get the vkCmdPushDescriptorSetKHR function pointer (nullptr if not supported)
  PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn() const {
    return push_descriptor_fn_;
  }
  uint32_t vendor_id() const {
    return vendor_id_;
  }
  uint32_t device_id() const {
    return device_id_;
  }
  GpuArchitecture architecture() const {
    return architecture_;
  }
  uint32_t shader_core_count() const {
    return shader_core_count_;
  }

  // Find memory type that supports the given properties
  uint32_t find_memory_type(
      uint32_t typeFilter,
      VkMemoryPropertyFlags properties) const;

 private:
  VulkanContext();
  ~VulkanContext();

  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  void init();
  void cleanup();
  bool probe_shader_bfloat16_support() const;

  // C++ Vulkan API objects (RAII)
  vk::Instance instance_;
  vk::PhysicalDevice physical_device_;
  vk::Device device_;
  vk::Queue compute_queue_;
  vk::Queue transfer_queue_;
  uint32_t compute_queue_family_index_{0};
  uint32_t transfer_queue_family_index_{0};
  bool has_separate_transfer_queue_{false};

  // Timeline semaphore for cross-queue synchronization
  vk::Semaphore timeline_semaphore_;
  std::atomic<uint64_t> timeline_value_{0};

  bool initialized_{false};
  bool is_unified_memory_{false};
  bool shader_float16_supported_{false};
  bool shader_bfloat16_extension_present_{false};
  bool shader_bfloat16_reported_supported_{false};
  mutable std::once_flag shader_bfloat16_probe_once_;
  mutable bool shader_bfloat16_supported_{false};
  bool subgroup_size_control_supported_{false};
  bool subgroup_require_full_support_{false};
  uint32_t subgroup_min_size_{0};
  uint32_t subgroup_max_size_{0};
  uint32_t subgroup_size_{0};
  bool pipeline_robustness_supported_{false};
  bool cooperative_matrix_supported_{false};
  bool coopmat_flash_attention_f32acc_supported_{false};
  bool coopmat2_conv2d_supported_{false};
  bool integer_dot_product_supported_{false};
  bool push_descriptor_supported_{false};
  PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn_{nullptr};
  uint32_t vendor_id_{0};
  uint32_t device_id_{0};
  GpuArchitecture architecture_{GpuArchitecture::Unknown};
  uint32_t shader_core_count_{0};
  vk::PhysicalDeviceMemoryProperties mem_properties_{};
};

// Helper function to check Vulkan result and throw on error
inline void throw_if_vk_error(VkResult result, const std::string& context) {
  if (result != VK_SUCCESS) {
    throw std::runtime_error(
        context + " (VkResult=" + std::to_string(result) + ").");
  }
}

} // namespace mlx::core::vulkan
