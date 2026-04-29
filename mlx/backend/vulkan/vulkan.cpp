// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/vulkan.h"

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::vulkan {

namespace {

struct QueueFamilyIndices {
  uint32_t compute_family{0};
  uint32_t compute_queue_index{0};
  uint32_t transfer_family{0};
  uint32_t transfer_queue_index{0};
  bool has_separate_transfer{false};
};

std::optional<uint32_t> parse_env_uint32(const char* env_name) {
  const char* env = std::getenv(env_name);
  if (env == nullptr || *env == '\0') {
    return std::nullopt;
  }

  errno = 0;
  char* end = nullptr;
  unsigned long parsed = std::strtoul(env, &end, 0);
  if (errno != 0 || end == env || (end != nullptr && *end != '\0') ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("[vulkan::init] Invalid ") + env_name + "='" + env +
        "'. Expected unsigned integer.");
  }
  return static_cast<uint32_t>(parsed);
}

uint32_t device_type_rank(vk::PhysicalDeviceType type) {
  switch (type) {
    case vk::PhysicalDeviceType::eDiscreteGpu:
      return 5;
    case vk::PhysicalDeviceType::eIntegratedGpu:
      return 4;
    case vk::PhysicalDeviceType::eVirtualGpu:
      return 3;
    case vk::PhysicalDeviceType::eCpu:
      return 2;
    default:
      return 1;
  }
}

uint64_t total_device_local_memory(vk::PhysicalDevice physical_device) {
  const auto mem = physical_device.getMemoryProperties();
  uint64_t total = 0;
  for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
    if ((mem.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) !=
        vk::MemoryHeapFlagBits{}) {
      total += mem.memoryHeaps[i].size;
    }
  }
  return total;
}

uint64_t score_physical_device(
    vk::PhysicalDevice physical_device,
    const QueueFamilyIndices& indices,
    std::optional<uint32_t> preferred_vendor_id) {
  const auto properties = physical_device.getProperties();
  const uint64_t type_score =
      static_cast<uint64_t>(device_type_rank(properties.deviceType)) << 60;
  const uint64_t local_mem_score =
      std::min<uint64_t>(total_device_local_memory(physical_device), (1ull << 56) - 1)
      << 4;
  const uint64_t queue_topology_score =
      indices.has_separate_transfer ? (1ull << 3) : 0ull;
  const uint64_t vendor_score =
      (preferred_vendor_id.has_value() &&
       properties.vendorID == preferred_vendor_id.value())
      ? (1ull << 2)
      : 0ull;
  return type_score + local_mem_score + queue_topology_score + vendor_score;
}

uint32_t find_queue_family(
    const std::vector<vk::QueueFamilyProperties>& queue_families,
    const vk::QueueFlags& required,
    const vk::QueueFlags& avoid,
    int32_t compute_index,
    uint32_t min_num_queues) {
  const uint32_t qfsize = queue_families.size();

  for (uint32_t i = 0; i < qfsize; ++i) {
    if (queue_families[i].queueCount >= min_num_queues &&
        (compute_index < 0 || i != static_cast<uint32_t>(compute_index)) &&
        (queue_families[i].queueFlags & required) &&
        !(queue_families[i].queueFlags & avoid)) {
      return i;
    }
  }

  for (uint32_t i = 0; i < qfsize; ++i) {
    if (queue_families[i].queueCount >= min_num_queues &&
        (compute_index < 0 || i != static_cast<uint32_t>(compute_index)) &&
        (queue_families[i].queueFlags & required)) {
      return i;
    }
  }

  for (uint32_t i = 0; i < qfsize; ++i) {
    if (queue_families[i].queueCount >= min_num_queues &&
        (queue_families[i].queueFlags & required)) {
      return i;
    }
  }

  for (uint32_t i = 0; i < qfsize; ++i) {
    if (queue_families[i].queueFlags & required) {
      return i;
    }
  }

  return 0;
}

QueueFamilyIndices find_queue_families(vk::PhysicalDevice physical_device) {
  auto queue_families = physical_device.getQueueFamilyProperties();

  const uint32_t compute_family = find_queue_family(
      queue_families,
      vk::QueueFlagBits::eCompute,
      vk::QueueFlagBits::eGraphics,
      -1,
      1);

  const uint32_t transfer_family = find_queue_family(
      queue_families,
      vk::QueueFlagBits::eTransfer,
      vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics,
      compute_family,
      1);

  QueueFamilyIndices indices;
  indices.compute_family = compute_family;
  indices.transfer_family = compute_family;

  const bool distinct_transfer_family = (compute_family != transfer_family) &&
      (queue_families[transfer_family].queueFlags &
       vk::QueueFlagBits::eTransfer) != vk::QueueFlagBits{};
  if (distinct_transfer_family) {
    indices.transfer_family = transfer_family;
    indices.has_separate_transfer = true;
    return indices;
  }

  if ((queue_families[compute_family].queueFlags &
       vk::QueueFlagBits::eTransfer) != vk::QueueFlagBits{} &&
      queue_families[compute_family].queueCount > 1) {
    indices.transfer_family = compute_family;
    indices.transfer_queue_index = 1;
    indices.has_separate_transfer = true;
  }

  return indices;
}

bool has_device_extension(
    const std::vector<vk::ExtensionProperties>& extensions,
    const char* name) {
  return std::any_of(
      extensions.begin(),
      extensions.end(),
      [name](const vk::ExtensionProperties& ext) {
        return std::string(ext.extensionName) == name;
      });
}

bool device_name_contains(const std::string& name, const char* needle) {
  return name.find(needle) != std::string::npos;
}

GpuArchitecture classify_gpu_architecture(
    uint32_t vendor_id,
    const std::string& device_name) {
  switch (vendor_id) {
    case 0x1002u:
      if (device_name_contains(device_name, "Instinct") ||
          device_name_contains(device_name, "MI")) {
        return GpuArchitecture::AmdCdna;
      }
      return GpuArchitecture::AmdRdna;
    case 0x10DEu:
      return GpuArchitecture::Nvidia;
    case 0x8086u:
      return GpuArchitecture::Intel;
    case 0x106Bu:
      return GpuArchitecture::Apple;
    case 0x5143u:
      return GpuArchitecture::Qualcomm;
    default:
      return GpuArchitecture::Unknown;
  }
}

bool bf16_capability_debug_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("MLX_VULKAN_DEBUG_CAPS");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

std::optional<bool> forced_bf16_capability() {
  static const std::optional<bool> enabled = []() -> std::optional<bool> {
    const char* env = std::getenv("MLX_VULKAN_FORCE_BF16_CAPABILITY");
    if (env == nullptr) {
      return std::nullopt;
    }

    const std::string value(env);
    if (value == "0") {
      return false;
    }
    if (value == "1") {
      return true;
    }
    return std::nullopt;
  }();
  return enabled;
}

bool nearly_equal(float a, float b, float atol = 1e-3f) {
  return std::fabs(a - b) <= atol;
}

uint32_t intel_shader_core_count(uint32_t vendor_id, uint32_t device_id) {
  if (vendor_id != 0x8086u) {
    return 0;
  }

  // Vulkan does not expose a generic Intel Xe core-count property, so keep a
  // conservative device-ID lookup and fall back to heuristics when unknown.
  switch (device_id) {
    case 0x56A6u: // A310
      return 6;
    case 0x5693u: // A370M
    case 0x56A5u: // A380
    case 0x56B1u: // Pro A40/A50
      return 8;
    case 0x5697u: // A530M
      return 12;
    case 0x5692u: // A550M
    case 0x56B3u: // Pro A60
    case 0xE212u: // Pro B50
      return 16;
    case 0xE20Cu: // B570
      return 18;
    case 0xE20Bu: // B580
      return 20;
    case 0x56A2u: // A580
      return 24;
    case 0x5691u: // A730M
    case 0x56A1u: // A750
      return 28;
    case 0x56A0u: // A770
    case 0x5690u: // A770M
      return 32;
    default:
      return 0;
  }
}

} // namespace

bool VulkanContext::shader_bfloat16_supported() const {
  std::call_once(shader_bfloat16_probe_once_, [this]() {
    if (auto forced = forced_bf16_capability(); forced.has_value()) {
      shader_bfloat16_supported_ = *forced;
    } else {
      shader_bfloat16_supported_ = probe_shader_bfloat16_support();
    }

    if (bf16_capability_debug_enabled()) {
      std::cerr << "[vulkan::caps] shader_bfloat16 extension_present="
                << shader_bfloat16_extension_present_
                << " reported_supported=" << shader_bfloat16_reported_supported_
                << " forced_override="
                << (forced_bf16_capability().has_value()
                        ? (*forced_bf16_capability() ? 1 : 0)
                        : -1)
                << " runtime_probe_supported=" << shader_bfloat16_supported_
                << "\n";
    }
  });

  return shader_bfloat16_supported_;
}

bool is_available() {
  try {
    VulkanContext::get();
    return true;
  } catch (...) {
    return false;
  }
}

bool is_unified_memory() {
  return VulkanContext::get().is_unified_memory();
}

int device_count() {
  return is_available() ? 1 : 0;
}

VulkanContext& VulkanContext::get() {
  static VulkanContext context;
  static std::once_flag init_once;
  auto* context_ptr = &context;
  std::call_once(init_once, [context_ptr]() { context_ptr->init(); });
  return context;
}

VulkanContext::VulkanContext() = default;

VulkanContext::~VulkanContext() {
  cleanup();
}

bool VulkanContext::probe_shader_bfloat16_support() const {
  try {
    const Stream s{0, Device{Device::gpu, 0}};

    auto make_array = [](Shape shape, Dtype dtype) {
      array out(std::move(shape), dtype, nullptr, {});
      out.set_data(allocator::malloc(out.nbytes()));
      return out;
    };

    array a = make_array({2, 3}, bfloat16);
    array b_t = make_array({2, 3}, bfloat16);
    array out_t = make_array({2, 2}, float32);
    array out = make_array({2, 2}, float32);

    auto* a_ptr = a.data<bfloat16_t>();
    a_ptr[0] = 1.0f;
    a_ptr[1] = 2.0f;
    a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f;
    a_ptr[4] = 5.0f;
    a_ptr[5] = 6.0f;

    auto* b_ptr = b_t.data<bfloat16_t>();
    b_ptr[0] = 1.0f;
    b_ptr[1] = 0.0f;
    b_ptr[2] = 1.0f;
    b_ptr[3] = 0.0f;
    b_ptr[4] = 1.0f;
    b_ptr[5] = -1.0f;

    auto verify_fp32 = [](const array& out_arr) {
      const auto* out_ptr = out_arr.data<float>();
      return nearly_equal(out_ptr[0], 4.0f) &&
          nearly_equal(out_ptr[1], -1.0f) &&
          nearly_equal(out_ptr[2], 10.0f) &&
          nearly_equal(out_ptr[3], -1.0f);
    };

    auto verify_bf16 = [](const array& out_arr) {
      const auto* out_ptr = out_arr.data<bfloat16_t>();
      return nearly_equal(static_cast<float>(out_ptr[0]), 4.0f) &&
          nearly_equal(static_cast<float>(out_ptr[1]), -1.0f) &&
          nearly_equal(static_cast<float>(out_ptr[2]), 10.0f) &&
          nearly_equal(static_cast<float>(out_ptr[3]), -1.0f);
    };

    auto dispatch_probe = [&](StaticShaderId shader_id, array& dst) {
      MatmulPushConstants push_constants{};
      push_constants.M = 2;
      push_constants.N = 2;
      push_constants.K = 3;
      push_constants.stride_a = static_cast<uint32_t>(a.strides(-2));
      push_constants.stride_b = static_cast<uint32_t>(b_t.strides(-2));
      push_constants.stride_d = static_cast<uint32_t>(dst.strides(-2));
      push_constants.batch_stride_a = 6;
      push_constants.batch_stride_b = 6;
      push_constants.batch_stride_d =
          static_cast<uint32_t>(dst.shape(-2) * dst.shape(-1));
      push_constants.num_batches = 1;
      push_constants.k_split = 3;
      push_constants.ne02 = 1;
      push_constants.ne12 = 1;
      push_constants.broadcast2 = 1;
      push_constants.broadcast3 = 1;
      push_constants.padded_N = 2;
      push_constants.base_work_group_z = 0;

      auto command_buffer = begin_command_recording(s.index);
      dispatch_mul_mm_op(
          a,
          b_t,
          dst,
          shader_id,
          command_buffer,
          s,
          push_constants,
          {1u, 1u, 1u});
      set_force_immediate_submit(s);
      end_command_recording(s.index);
    };

    std::vector<std::pair<StaticShaderId, array>> direct_probe_outputs;
    for (auto shader_id : {
             StaticShaderId::matmul_direct_bf16,
             StaticShaderId::matmul_direct_bf16_f16acc,
         }) {
      try {
        array out_direct = make_array({2, 2}, bfloat16);
        dispatch_probe(shader_id, out_direct);
        direct_probe_outputs.emplace_back(shader_id, std::move(out_direct));
      } catch (const std::runtime_error&) {
      }
    }

    std::vector<std::pair<StaticShaderId, array>> legacy_probe_outputs;
    for (auto shader_id : {
             StaticShaderId::matmul_bf16_fp32,
             StaticShaderId::matmul_bf16,
         }) {
      try {
        array out = make_array({2, 2}, float32);
        dispatch_probe(shader_id, out);
        legacy_probe_outputs.emplace_back(shader_id, std::move(out));
      } catch (const std::runtime_error&) {
      }
    }

    for (const auto& output : direct_probe_outputs) {
      if (verify_bf16(output.second)) {
        return true;
      }
    }
    for (const auto& output : legacy_probe_outputs) {
      if (verify_fp32(output.second)) {
        return true;
      }
    }

    return false;
  } catch (const std::runtime_error&) {
    return false;
  }
}

void VulkanContext::init() {
  if (initialized_) {
    return;
  }

  // C++ Vulkan API objects (will be automatically destroyed)
  vk::Instance instance;
  vk::PhysicalDevice physical_device;
  vk::Device device;
  vk::Queue compute_queue;
  vk::Queue transfer_queue;
  uint32_t compute_queue_family_index = 0;
  uint32_t compute_queue_index = 0;
  uint32_t transfer_queue_family_index = 0;
  uint32_t transfer_queue_index = 0;
  bool has_separate_transfer_queue = false;
  bool is_unified_memory = false;
  bool shader_float16_supported = false;
  bool shader_int8_supported = false;
  bool storage_buffer_8bit_supported = false;
  bool scalar_block_layout_supported = false;
  bool shader_bfloat16_supported = false;
  bool shader_buffer_atomic_float32_supported = false;
  bool subgroup_size_control_supported = false;
  bool subgroup_require_full_support = false;
  uint32_t subgroup_min_size = 0;
  uint32_t subgroup_max_size = 0;
  uint32_t subgroup_size = 0;
  bool pipeline_robustness_supported = false;
  bool cooperative_matrix_supported = false;
  bool coopmat_flash_attention_f32acc_supported = false;
  bool coopmat2_conv2d_supported = false;
  bool integer_dot_product_supported = false;
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  GpuArchitecture architecture = GpuArchitecture::Unknown;
  uint32_t shader_core_count = 0;
  vk::PhysicalDeviceMemoryProperties mem_properties;
  vk::Semaphore timeline_semaphore;
  uint64_t timeline_value = 0;

  try {
    // 1. Create instance using C++ API
    vk::ApplicationInfo app_info(
        "MLX Vulkan Backend",
        vk::makeVersion(1, 0, 0),
        "MLX",
        vk::makeVersion(1, 0, 0),
        VK_API_VERSION_1_2);

    vk::InstanceCreateInfo create_info({}, &app_info);

    instance = vk::createInstance(create_info);

    // 2. Pick physical device with compute support and find queue families
    auto available_devices = instance.enumeratePhysicalDevices();
    if (available_devices.empty()) {
      throw std::runtime_error(
          "[vulkan::init] Failed to find GPUs with Vulkan support.");
    }

    const auto forced_device_index = parse_env_uint32("MLX_VULKAN_DEVICE_INDEX");
    const auto preferred_vendor_id =
        parse_env_uint32("MLX_VULKAN_PREFERRED_VENDOR_ID");

    std::optional<std::pair<uint64_t, uint32_t>> best_candidate;
    bool found_compute_device = false;
    for (uint32_t candidate_index = 0;
         candidate_index < available_devices.size();
         ++candidate_index) {
      auto candidate = available_devices[candidate_index];
      auto queue_families = candidate.getQueueFamilyProperties();
      bool has_compute = false;
      for (const auto& qf : queue_families) {
        if ((qf.queueFlags & vk::QueueFlagBits::eCompute) !=
            vk::QueueFlagBits{}) {
          has_compute = true;
          break;
        }
      }
      if (has_compute) {
        if (forced_device_index.has_value() &&
            forced_device_index.value() != candidate_index) {
          continue;
        }
        auto indices = find_queue_families(candidate);
        const uint64_t score =
            score_physical_device(candidate, indices, preferred_vendor_id);
        if (!best_candidate.has_value() || score > best_candidate->first) {
          best_candidate = std::make_pair(score, candidate_index);
        }
      }
    }

    if (best_candidate.has_value()) {
      const uint32_t selected_index = best_candidate->second;
      physical_device = available_devices[selected_index];
      auto indices = find_queue_families(physical_device);
      compute_queue_family_index = indices.compute_family;
      compute_queue_index = indices.compute_queue_index;
      transfer_queue_family_index = indices.transfer_family;
      transfer_queue_index = indices.transfer_queue_index;
      has_separate_transfer_queue = indices.has_separate_transfer;
      found_compute_device = true;
    }

    if (!found_compute_device) {
      if (forced_device_index.has_value()) {
        throw std::runtime_error(
            "[vulkan::init] Forced MLX_VULKAN_DEVICE_INDEX does not refer to a compute-capable physical device.");
      }
      throw std::runtime_error(
          "[vulkan::init] Failed to find a compute-capable physical device.");
    }

    // 3. Query memory properties and unified-memory characteristics
    mem_properties = physical_device.getMemoryProperties();

    auto extensions = physical_device.enumerateDeviceExtensionProperties();

    const bool has_subgroup_size_control_ext = has_device_extension(
        extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    const bool has_pipeline_robustness_ext = has_device_extension(
        extensions, VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    const bool has_cooperative_matrix_ext = has_device_extension(
        extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    const bool has_nv_cooperative_matrix2_ext = has_device_extension(
        extensions, "VK_NV_cooperative_matrix2");
    const bool has_shader_integer_dot_product_ext = has_device_extension(
        extensions, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    const bool has_shader_bfloat16_ext =
        has_device_extension(extensions, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
    const bool has_shader_atomic_float_ext = has_device_extension(
        extensions, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    const bool has_push_descriptor_ext = has_device_extension(
        extensions, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

    auto device_properties = physical_device.getProperties();
    vendor_id = device_properties.vendorID;
    device_id = device_properties.deviceID;
    architecture = classify_gpu_architecture(
        vendor_id, std::string(device_properties.deviceName));
    shader_core_count = intel_shader_core_count(vendor_id, device_id);

    is_unified_memory =
        (device_properties.deviceType ==
         vk::PhysicalDeviceType::eIntegratedGpu);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
      const auto flags = mem_properties.memoryTypes[i].propertyFlags;
      if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
          (flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
          (flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
        is_unified_memory = true;
        break;
      }
    }

    // 4. Create logical device using C++ API
    float queue_priority = 1.0f;
    float queue_priority_low = 0.5f;
    std::array<float, 2> shared_family_priorities = {
        queue_priority, queue_priority_low};
    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;

    if (has_separate_transfer_queue &&
        compute_queue_family_index == transfer_queue_family_index) {
      vk::DeviceQueueCreateInfo queue_info(
          vk::DeviceQueueCreateFlags(),
          compute_queue_family_index,
          static_cast<uint32_t>(shared_family_priorities.size()),
          shared_family_priorities.data());
      queue_create_infos.push_back(queue_info);
    } else if (has_separate_transfer_queue) {
      vk::DeviceQueueCreateInfo compute_queue_info(
          vk::DeviceQueueCreateFlags(),
          compute_queue_family_index,
          1,
          &queue_priority);
      vk::DeviceQueueCreateInfo transfer_queue_info(
          vk::DeviceQueueCreateFlags(),
          transfer_queue_family_index,
          1,
          &queue_priority_low);
      queue_create_infos.push_back(compute_queue_info);
      queue_create_infos.push_back(transfer_queue_info);
    } else {
      vk::DeviceQueueCreateInfo queue_info(
          vk::DeviceQueueCreateFlags(),
          compute_queue_family_index,
          1,
          &queue_priority);
      queue_create_infos.push_back(queue_info);
    }

    // Build feature chain
    vk::PhysicalDeviceFeatures2 supported_features;
    vk::PhysicalDeviceVulkan11Features supported_vulkan11_features;
    vk::PhysicalDeviceShaderFloat16Int8Features supported_shader_float16_int8;
    vk::PhysicalDevice8BitStorageFeatures supported_storage_8bit;
    vk::PhysicalDeviceScalarBlockLayoutFeatures supported_scalar_block_layout;
    supported_features.pNext = &supported_vulkan11_features;
    supported_vulkan11_features.pNext = &supported_shader_float16_int8;
    supported_shader_float16_int8.pNext = &supported_storage_8bit;
    supported_storage_8bit.pNext = &supported_scalar_block_layout;
    vk::PhysicalDeviceShaderIntegerDotProductFeatures
        supported_shader_integer_dot_product{};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT supported_shader_atomic_float{};
    supported_shader_atomic_float.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    supported_scalar_block_layout.pNext = &supported_shader_integer_dot_product;
    supported_shader_integer_dot_product.pNext = &supported_shader_atomic_float;

    vk::PhysicalDeviceSubgroupSizeControlFeatures
        supported_subgroup_size_control{};
    vk::PhysicalDevicePipelineRobustnessFeaturesEXT
        supported_pipeline_robustness{};
    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR
        supported_cooperative_matrix{};
    vk::PhysicalDeviceShaderBfloat16FeaturesKHR supported_shader_bfloat16{};

    if (has_subgroup_size_control_ext) {
      supported_shader_atomic_float.pNext =
          &supported_subgroup_size_control;
      if (has_pipeline_robustness_ext) {
        supported_subgroup_size_control.pNext = &supported_pipeline_robustness;
        if (has_cooperative_matrix_ext) {
          supported_pipeline_robustness.pNext = &supported_cooperative_matrix;
          if (has_shader_bfloat16_ext) {
            supported_cooperative_matrix.pNext = &supported_shader_bfloat16;
          }
        } else if (has_shader_bfloat16_ext) {
          supported_pipeline_robustness.pNext = &supported_shader_bfloat16;
        }
      } else if (has_cooperative_matrix_ext) {
        supported_subgroup_size_control.pNext = &supported_cooperative_matrix;
        if (has_shader_bfloat16_ext) {
          supported_cooperative_matrix.pNext = &supported_shader_bfloat16;
        }
      } else if (has_shader_bfloat16_ext) {
        supported_shader_atomic_float.pNext = &supported_shader_bfloat16;
      }
    } else if (has_pipeline_robustness_ext) {
      supported_shader_atomic_float.pNext =
          &supported_pipeline_robustness;
      if (has_cooperative_matrix_ext) {
        supported_pipeline_robustness.pNext = &supported_cooperative_matrix;
        if (has_shader_bfloat16_ext) {
          supported_cooperative_matrix.pNext = &supported_shader_bfloat16;
        }
      } else if (has_shader_bfloat16_ext) {
        supported_pipeline_robustness.pNext = &supported_shader_bfloat16;
      }
    } else if (has_cooperative_matrix_ext) {
      supported_shader_atomic_float.pNext =
          &supported_cooperative_matrix;
      if (has_shader_bfloat16_ext) {
        supported_cooperative_matrix.pNext = &supported_shader_bfloat16;
      }
    } else if (has_shader_bfloat16_ext) {
      supported_shader_atomic_float.pNext = &supported_shader_bfloat16;
    }

    physical_device.getFeatures2(&supported_features);

    // Query subgroup size control properties
    vk::PhysicalDeviceSubgroupProperties subgroup_props{};
    vk::PhysicalDeviceSubgroupSizeControlProperties subgroup_size_control_props;
    vk::PhysicalDeviceShaderIntegerDotProductProperties
        shader_integer_dot_product_props{};
    vk::PhysicalDeviceProperties2 props2;
    props2.pNext = &subgroup_props;
    subgroup_props.pNext = &subgroup_size_control_props;
    subgroup_size_control_props.pNext = &shader_integer_dot_product_props;
    physical_device.getProperties2(&props2);
    subgroup_size = subgroup_props.subgroupSize;

    // Build enabled features
    vk::PhysicalDeviceFeatures2 enabled_features;
    vk::PhysicalDeviceVulkan11Features enabled_vulkan11_features;
    vk::PhysicalDeviceShaderFloat16Int8Features enabled_shader_float16_int8;
    vk::PhysicalDevice8BitStorageFeatures enabled_storage_8bit;
    vk::PhysicalDeviceScalarBlockLayoutFeatures enabled_scalar_block_layout;
    enabled_features.pNext = &enabled_vulkan11_features;
    enabled_vulkan11_features.pNext = &enabled_shader_float16_int8;
    enabled_shader_float16_int8.pNext = &enabled_storage_8bit;
    enabled_storage_8bit.pNext = &enabled_scalar_block_layout;
    vk::PhysicalDeviceShaderIntegerDotProductFeatures
        enabled_shader_integer_dot_product{};
    enabled_scalar_block_layout.pNext = &enabled_shader_integer_dot_product;

    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT enabled_shader_atomic_float{};
    enabled_shader_atomic_float.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

    // Link enabled atomic float feature if extension is available
    if (has_shader_atomic_float_ext) {
      enabled_shader_integer_dot_product.pNext =
          &enabled_shader_atomic_float;
    }

    vk::PhysicalDeviceSubgroupSizeControlFeatures
        enabled_subgroup_size_control{};
    vk::PhysicalDevicePipelineRobustnessFeaturesEXT
        enabled_pipeline_robustness{};
    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR enabled_cooperative_matrix{};
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV enabled_cooperative_matrix2{};
    vk::PhysicalDeviceShaderBfloat16FeaturesKHR enabled_shader_bfloat16{};

    // Link enabled feature chain (same pattern as supported features)
    if (has_shader_atomic_float_ext && has_subgroup_size_control_ext) {
      enabled_shader_atomic_float.pNext = &enabled_subgroup_size_control;
      if (has_pipeline_robustness_ext) {
        enabled_subgroup_size_control.pNext = &enabled_pipeline_robustness;
        if (has_cooperative_matrix_ext) {
          enabled_pipeline_robustness.pNext = &enabled_cooperative_matrix;
          if (has_shader_bfloat16_ext) {
            enabled_cooperative_matrix.pNext = &enabled_shader_bfloat16;
          }
        } else if (has_shader_bfloat16_ext) {
          enabled_pipeline_robustness.pNext = &enabled_shader_bfloat16;
        }
      } else if (has_cooperative_matrix_ext) {
        enabled_subgroup_size_control.pNext = &enabled_cooperative_matrix;
        if (has_shader_bfloat16_ext) {
          enabled_cooperative_matrix.pNext = &enabled_shader_bfloat16;
        }
      } else if (has_shader_bfloat16_ext) {
        enabled_shader_atomic_float.pNext = &enabled_shader_bfloat16;
      }
    } else if (has_shader_atomic_float_ext && has_pipeline_robustness_ext) {
      enabled_shader_atomic_float.pNext = &enabled_pipeline_robustness;
      if (has_cooperative_matrix_ext) {
        enabled_pipeline_robustness.pNext = &enabled_cooperative_matrix;
        if (has_shader_bfloat16_ext) {
          enabled_cooperative_matrix.pNext = &enabled_shader_bfloat16;
        }
      } else if (has_shader_bfloat16_ext) {
        enabled_pipeline_robustness.pNext = &enabled_shader_bfloat16;
      }
    } else if (has_shader_atomic_float_ext && has_cooperative_matrix_ext) {
      enabled_shader_atomic_float.pNext = &enabled_cooperative_matrix;
      if (has_shader_bfloat16_ext) {
        enabled_cooperative_matrix.pNext = &enabled_shader_bfloat16;
      }
    } else if (has_shader_atomic_float_ext && has_shader_bfloat16_ext) {
      enabled_shader_atomic_float.pNext = &enabled_shader_bfloat16;
    } else if (!has_shader_atomic_float_ext && has_subgroup_size_control_ext) {
      enabled_shader_integer_dot_product.pNext = &enabled_subgroup_size_control;
      if (has_pipeline_robustness_ext) {
        enabled_subgroup_size_control.pNext = &enabled_pipeline_robustness;
        if (has_cooperative_matrix_ext) {
          enabled_pipeline_robustness.pNext = &enabled_cooperative_matrix;
          if (has_shader_bfloat16_ext) {
            enabled_cooperative_matrix.pNext = &enabled_shader_bfloat16;
          }
        } else if (has_shader_bfloat16_ext) {
          enabled_pipeline_robustness.pNext = &enabled_shader_bfloat16;
        }
      } else if (has_cooperative_matrix_ext) {
        enabled_subgroup_size_control.pNext = &enabled_cooperative_matrix;
        if (has_shader_bfloat16_ext) {
          enabled_cooperative_matrix.pNext = &enabled_shader_bfloat16;
        }
      } else if (has_shader_bfloat16_ext) {
        enabled_shader_integer_dot_product.pNext = &enabled_shader_bfloat16;
      }
    } else if (!has_shader_atomic_float_ext && has_pipeline_robustness_ext) {
      enabled_shader_integer_dot_product.pNext = &enabled_pipeline_robustness;
      if (has_cooperative_matrix_ext) {
        enabled_pipeline_robustness.pNext = &enabled_cooperative_matrix;
        if (has_shader_bfloat16_ext) {
          enabled_cooperative_matrix.pNext = &enabled_shader_bfloat16;
        }
      } else if (has_shader_bfloat16_ext) {
        enabled_pipeline_robustness.pNext = &enabled_shader_bfloat16;
      }
    } else if (!has_shader_atomic_float_ext && has_cooperative_matrix_ext) {
      enabled_shader_integer_dot_product.pNext = &enabled_cooperative_matrix;
      if (has_shader_bfloat16_ext) {
        enabled_cooperative_matrix.pNext = &enabled_shader_bfloat16;
      }
    } else if (!has_shader_atomic_float_ext && has_shader_bfloat16_ext) {
      enabled_shader_integer_dot_product.pNext = &enabled_shader_bfloat16;
    }

    if (supported_vulkan11_features.storageBuffer16BitAccess) {
      enabled_vulkan11_features.storageBuffer16BitAccess = VK_TRUE;
    }
    if (supported_storage_8bit.storageBuffer8BitAccess) {
      enabled_storage_8bit.storageBuffer8BitAccess = VK_TRUE;
      storage_buffer_8bit_supported = true;
    }
    if (supported_scalar_block_layout.scalarBlockLayout) {
      enabled_scalar_block_layout.scalarBlockLayout = VK_TRUE;
      scalar_block_layout_supported = true;
    }
    if (supported_features.features.shaderInt16) {
      enabled_features.features.shaderInt16 = VK_TRUE;
    }
    if (supported_shader_float16_int8.shaderInt8) {
      enabled_shader_float16_int8.shaderInt8 = VK_TRUE;
      shader_int8_supported = true;
    }
    if (supported_shader_float16_int8.shaderFloat16) {
      enabled_shader_float16_int8.shaderFloat16 = VK_TRUE;
      shader_float16_supported = true;
    }
    if (has_shader_bfloat16_ext &&
        supported_shader_bfloat16.shaderBFloat16Type) {
      enabled_shader_bfloat16.shaderBFloat16Type = VK_TRUE;
      shader_bfloat16_supported = true;
    }
    if (has_shader_atomic_float_ext &&
        supported_shader_atomic_float.shaderBufferFloat32AtomicAdd) {
      enabled_shader_atomic_float.shaderBufferFloat32AtomicAdd = VK_TRUE;
      shader_buffer_atomic_float32_supported = true;
    }
    if (has_subgroup_size_control_ext &&
        supported_subgroup_size_control.subgroupSizeControl &&
        (subgroup_size_control_props.requiredSubgroupSizeStages &
         vk::ShaderStageFlagBits::eCompute)) {
      enabled_subgroup_size_control.subgroupSizeControl = VK_TRUE;
      subgroup_size_control_supported = true;
      subgroup_min_size = subgroup_size_control_props.minSubgroupSize;
      subgroup_max_size = subgroup_size_control_props.maxSubgroupSize;
    }
    if (has_subgroup_size_control_ext &&
        supported_subgroup_size_control.computeFullSubgroups) {
      enabled_subgroup_size_control.computeFullSubgroups = VK_TRUE;
      subgroup_require_full_support = true;
    }
    if (has_pipeline_robustness_ext &&
        supported_pipeline_robustness.pipelineRobustness) {
      enabled_pipeline_robustness.pipelineRobustness = VK_TRUE;
      pipeline_robustness_supported = true;
    }
    if (has_shader_integer_dot_product_ext &&
        supported_shader_integer_dot_product.shaderIntegerDotProduct) {
      enabled_shader_integer_dot_product.shaderIntegerDotProduct = VK_TRUE;
      integer_dot_product_supported = true;
    }
    if (has_cooperative_matrix_ext &&
        supported_cooperative_matrix.cooperativeMatrix) {
      enabled_cooperative_matrix.cooperativeMatrix = VK_TRUE;
      cooperative_matrix_supported = true;
      coopmat_flash_attention_f32acc_supported = false;

      auto get_coopmat_props = reinterpret_cast<
          PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
      if (get_coopmat_props != nullptr) {
        uint32_t coopmat_prop_count = 0;
        if (get_coopmat_props(physical_device, &coopmat_prop_count, nullptr) ==
                VK_SUCCESS &&
            coopmat_prop_count > 0) {
          std::vector<VkCooperativeMatrixPropertiesKHR> coopmat_props(
              coopmat_prop_count);
          for (auto& prop : coopmat_props) {
            prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
          }
          if (get_coopmat_props(
                  physical_device, &coopmat_prop_count, coopmat_props.data()) ==
              VK_SUCCESS) {
            for (const auto& prop : coopmat_props) {
              if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16 &&
                  prop.scope == VK_SCOPE_SUBGROUP_KHR &&
                  prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                  prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                  prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                  prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                coopmat_flash_attention_f32acc_supported = true;
                break;
              }
            }
          }
        }
      }
    }

    if (has_nv_cooperative_matrix2_ext && cooperative_matrix_supported) {
      // Query NV_cooperative_matrix2 properties and features
      VkPhysicalDeviceCooperativeMatrix2PropertiesNV cm2_props{};
      cm2_props.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV;
      VkPhysicalDeviceCooperativeMatrix2FeaturesNV cm2_features{};
      cm2_features.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;

      // Append cm2 properties to the properties chain
      VkBaseOutStructure* prop_tail =
          reinterpret_cast<VkBaseOutStructure*>(&props2);
      while (prop_tail->pNext != nullptr) {
        prop_tail = prop_tail->pNext;
      }
      prop_tail->pNext =
          reinterpret_cast<VkBaseOutStructure*>(&cm2_props);

      // Append cm2 supported features to the features chain
      VkBaseOutStructure* feat_tail =
          reinterpret_cast<VkBaseOutStructure*>(&supported_features);
      while (feat_tail->pNext != nullptr) {
        feat_tail = feat_tail->pNext;
      }
      feat_tail->pNext =
          reinterpret_cast<VkBaseOutStructure*>(&cm2_features);

      // Re-query with the extended chains
      physical_device.getFeatures2(&supported_features);
      physical_device.getProperties2(&props2);

      if (cm2_features.cooperativeMatrixWorkgroupScope &&
          cm2_features.cooperativeMatrixFlexibleDimensions &&
          cm2_props.cooperativeMatrixFlexibleDimensionsMaxDimension >= 512) {
        coopmat2_conv2d_supported = true;
      }
    }

    if (coopmat2_conv2d_supported) {
      enabled_cooperative_matrix2.cooperativeMatrixWorkgroupScope = VK_TRUE;
      enabled_cooperative_matrix2.cooperativeMatrixFlexibleDimensions = VK_TRUE;
      // Append to enabled feature chain
      VkBaseOutStructure* enabled_feat_tail =
          reinterpret_cast<VkBaseOutStructure*>(&enabled_features);
      while (enabled_feat_tail->pNext != nullptr) {
        enabled_feat_tail = enabled_feat_tail->pNext;
      }
      enabled_feat_tail->pNext =
          reinterpret_cast<VkBaseOutStructure*>(&enabled_cooperative_matrix2);
    }

    std::vector<const char*> device_extensions;
    device_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    if (subgroup_size_control_supported || subgroup_require_full_support) {
      device_extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }
    if (pipeline_robustness_supported) {
      device_extensions.push_back(VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    }
    if (cooperative_matrix_supported) {
      device_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    if (integer_dot_product_supported) {
      device_extensions.push_back(
          VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    }
    if (shader_bfloat16_supported) {
      device_extensions.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
    }
    if (shader_buffer_atomic_float32_supported) {
      device_extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }
    if (has_device_extension(extensions, VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME)) {
      device_extensions.push_back(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
    }
    if (coopmat2_conv2d_supported) {
      device_extensions.push_back(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
    }
    if (has_push_descriptor_ext) {
      device_extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo device_create_info;
    device_create_info.flags = vk::DeviceCreateFlags();
    device_create_info.queueCreateInfoCount =
        static_cast<uint32_t>(queue_create_infos.size());
    device_create_info.pQueueCreateInfos = queue_create_infos.data();
    device_create_info.enabledExtensionCount =
        static_cast<uint32_t>(device_extensions.size());
    device_create_info.ppEnabledExtensionNames = device_extensions.data();
    device_create_info.pEnabledFeatures = nullptr;

    // Set the feature chain using pNext
    device_create_info.pNext = &enabled_features;

    device = physical_device.createDevice(device_create_info);

    // Load extension function pointers
    PFN_vkCmdPushDescriptorSetKHR push_descriptor_fn = nullptr;
    if (has_push_descriptor_ext) {
      push_descriptor_fn = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
          vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));
    }

    // 5. Get queues and create timeline semaphore
    compute_queue =
        device.getQueue(compute_queue_family_index, compute_queue_index);
    if (has_separate_transfer_queue) {
      transfer_queue =
          device.getQueue(transfer_queue_family_index, transfer_queue_index);
    } else {
      transfer_queue = compute_queue;
    }

    vk::SemaphoreTypeCreateInfo timeline_ci;
    timeline_ci.sType = vk::StructureType::eSemaphoreTypeCreateInfo;
    timeline_ci.semaphoreType = vk::SemaphoreType::eTimeline;
    timeline_ci.pNext = nullptr;
    vk::SemaphoreCreateInfo ci;
    ci.pNext = &timeline_ci;
    timeline_semaphore = device.createSemaphore(ci);

    // Store in member variables (C++ API objects manage their own cleanup)
    instance_ = instance;
    physical_device_ = physical_device;
    device_ = device;
    compute_queue_ = compute_queue;
    transfer_queue_ = transfer_queue;
    compute_queue_family_index_ = compute_queue_family_index;
    transfer_queue_family_index_ = transfer_queue_family_index;
    has_separate_transfer_queue_ = has_separate_transfer_queue;
    timeline_semaphore_ = timeline_semaphore;
    timeline_value_ = timeline_value;
    mem_properties_ = mem_properties;
    is_unified_memory_ = is_unified_memory;
    this->shader_float16_supported_ = shader_float16_supported;
    this->shader_int8_supported_ = shader_int8_supported;
    this->storage_buffer_8bit_supported_ = storage_buffer_8bit_supported;
    this->scalar_block_layout_supported_ = scalar_block_layout_supported;
    this->shader_buffer_atomic_float32_supported_ = shader_buffer_atomic_float32_supported;
    this->shader_bfloat16_extension_present_ = has_shader_bfloat16_ext;
    this->shader_bfloat16_reported_supported_ = shader_bfloat16_supported;
    this->shader_bfloat16_supported_ = false;
    this->subgroup_size_control_supported_ = subgroup_size_control_supported;
    this->subgroup_require_full_support_ = subgroup_require_full_support;
    this->subgroup_min_size_ = subgroup_min_size;
    this->subgroup_max_size_ = subgroup_max_size;
    this->subgroup_size_ = subgroup_size;
    this->pipeline_robustness_supported_ = pipeline_robustness_supported;
    this->cooperative_matrix_supported_ = cooperative_matrix_supported;
    this->coopmat_flash_attention_f32acc_supported_ =
        coopmat_flash_attention_f32acc_supported &&
        subgroup_require_full_support;
    this->coopmat2_conv2d_supported_ = coopmat2_conv2d_supported;
    this->integer_dot_product_supported_ = integer_dot_product_supported;
    // Only enable push descriptor support if both extension is present AND
    // function pointer was successfully resolved (avoid descriptor unbound issues)
    this->push_descriptor_supported_ =
        has_push_descriptor_ext && push_descriptor_fn != nullptr;
    this->push_descriptor_fn_ = push_descriptor_fn;
    this->vendor_id_ = vendor_id;
    this->device_id_ = device_id;
    this->architecture_ = architecture;
    this->shader_core_count_ = shader_core_count;
    initialized_ = true;
  } catch (...) {
    // Clean up partially initialized resources on failure
    // Note: We don't call waitIdle() here to avoid hangs during failed init
    if (device_) {
      device_.destroy();
      device_ = nullptr;
    }
    if (instance_) {
      instance_.destroy();
      instance_ = nullptr;
    }
    throw;
  }
}

void VulkanContext::cleanup() {
  // Note: vk::Instance and vk::Device are non-owning handles.
  // We must manually destroy them to avoid resource leaks.
  if (device_) {
    // Only wait for idle if we were fully initialized
    // to avoid hanging on partially initialized devices
    if (initialized_) {
      device_.waitIdle();
    }
    if (timeline_semaphore_) {
      device_.destroySemaphore(timeline_semaphore_);
      timeline_semaphore_ = nullptr;
    }
    device_.destroy();
    device_ = nullptr;
  }
  if (instance_) {
    instance_.destroy();
    instance_ = nullptr;
  }
  physical_device_ = nullptr;
  compute_queue_ = nullptr;
  transfer_queue_ = nullptr;
  compute_queue_family_index_ = 0;
  transfer_queue_family_index_ = 0;
  has_separate_transfer_queue_ = false;
  timeline_semaphore_ = nullptr;
  timeline_value_ = 0;
  is_unified_memory_ = false;
  shader_float16_supported_ = false;
  shader_int8_supported_ = false;
  storage_buffer_8bit_supported_ = false;
  scalar_block_layout_supported_ = false;
  shader_bfloat16_extension_present_ = false;
  shader_bfloat16_reported_supported_ = false;
  shader_bfloat16_supported_ = false;
  subgroup_size_control_supported_ = false;
  subgroup_require_full_support_ = false;
  subgroup_min_size_ = 0;
  subgroup_max_size_ = 0;
  subgroup_size_ = 0;
  pipeline_robustness_supported_ = false;
  cooperative_matrix_supported_ = false;
  coopmat_flash_attention_f32acc_supported_ = false;
  coopmat2_conv2d_supported_ = false;
  integer_dot_product_supported_ = false;
  vendor_id_ = 0;
  device_id_ = 0;
  architecture_ = GpuArchitecture::Unknown;
  shader_core_count_ = 0;

  // Clear memory properties by creating a default-constructed one
  vk::PhysicalDeviceMemoryProperties empty_props;
  mem_properties_ = empty_props;

  initialized_ = false;
}

uint64_t VulkanContext::increment_timeline() {
  return timeline_value_.fetch_add(1) + 1;
}

uint32_t VulkanContext::find_memory_type(
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties) const {
  // Convert VkMemoryPropertyFlags to vk::MemoryPropertyFlags for comparison
  auto vk_properties = static_cast<vk::MemoryPropertyFlags>(properties);

  for (uint32_t i = 0; i < mem_properties_.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (mem_properties_.memoryTypes[i].propertyFlags & vk_properties) ==
            vk_properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type");
}

} // namespace mlx::core::vulkan
