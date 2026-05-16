// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <limits>
#include <sstream>

#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"

namespace mlx::core {

namespace {

constexpr uint32_t kMaxWorkgroupsY = 512;
constexpr uint32_t kVendorIdAmd = 0x1002;
constexpr uint32_t kVendorIdIntel = 0x8086;

struct Conv2dPushConstants {
  uint32_t Cout;
  uint32_t Cin;
  uint32_t N;
  uint32_t W;
  uint32_t H;
  uint32_t OW;
  uint32_t OH;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t nb1;
  uint32_t nb2;
  uint32_t nb3;
  uint32_t OWmp;
  uint32_t OWL;
  uint32_t OWOHmp;
  uint32_t OWOHL;
};

struct DepthwiseConv2dPushConstants {
  uint32_t ne;
  uint32_t batches;
  uint32_t channels;
  uint32_t dst_w;
  uint32_t dst_h;
  uint32_t src_w;
  uint32_t src_h;
  uint32_t knl_w;
  uint32_t knl_h;
  int32_t stride_x;
  int32_t stride_y;
  int32_t pad_x;
  int32_t pad_y;
  int32_t dilation_x;
  int32_t dilation_y;
};

struct ConvBlockSize {
  uint32_t k;
  uint32_t npq;
  uint32_t crs;
};

struct ConvPipelineConfig {
  ConvBlockSize block_size;
  uint32_t workgroup_size;
  uint32_t thread_tile_k;
  uint32_t shmem_pad;
  uint32_t use_collectives;
  vulkan::StaticShaderId shader_id;
};

uint32_t div_up_u32(uint32_t n, uint32_t d) {
  return (n + d - 1) / d;
}

size_t num_elements(const Shape& shape) {
  size_t total = 1;
  for (auto dim : shape) {
    total *= static_cast<size_t>(dim);
  }
  return total;
}

void init_fastdiv_values(uint32_t d, uint32_t& mp, uint32_t& l) {
  if (d == 0) {
    throw std::runtime_error(
        "[vulkan::conv] fastdiv divisor must be non-zero.");
  }

  l = 0;
  while (l < 32 && (uint32_t{1} << l) < d) {
    l++;
  }
  mp = static_cast<uint32_t>(
      ((uint64_t{1} << 32) * ((uint64_t{1} << l) - d) / d) + 1);
}

std::vector<VkDescriptorSetLayoutBinding> conv_layout_bindings() {
  std::vector<VkDescriptorSetLayoutBinding> bindings(3);
  for (uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  return bindings;
}

VkDescriptorBufferInfo descriptor_buffer_info(
    const array& arr,
    const char* name) {
  auto* vulkan_buffer = static_cast<const vulkan::VulkanBuffer*>(
      static_cast<const void*>(arr.buffer().ptr()));
  if (vulkan_buffer == nullptr || vulkan_buffer->buffer == VK_NULL_HANDLE) {
    throw std::runtime_error(
        std::string("[vulkan::conv] Missing Vulkan buffer for ") + name + ".");
  }

  VkDescriptorBufferInfo info{};
  info.buffer = vulkan_buffer->buffer;
  info.offset = 0;
  info.range = VK_WHOLE_SIZE;
  return info;
}

array permute_view(
    const array& base,
    const Shape& shape,
    const Strides& strides,
    const std::array<int, 4>& perm) {
  Shape permuted_shape(4);
  Strides permuted_strides(4);
  for (size_t i = 0; i < 4; ++i) {
    permuted_shape[i] = shape[perm[i]];
    permuted_strides[i] = strides[perm[i]];
  }

  array view(permuted_shape, base.dtype(), nullptr, {});
  const auto [data_size, row_contiguous, col_contiguous] =
      check_contiguity(permuted_shape, permuted_strides);
  const bool contiguous = data_size == base.data_size();
  view.copy_shared_buffer(
      base,
      permuted_strides,
      {contiguous, row_contiguous, col_contiguous},
      base.data_size());
  return view;
}

array squeeze_unit_axis_view(const array& base, int axis) {
  if (base.shape(axis) != 1) {
    throw std::runtime_error(
        "[vulkan::conv] Cannot squeeze non-unit axis from view.");
  }

  Shape squeezed_shape;
  Strides squeezed_strides;
  squeezed_shape.reserve(base.ndim() - 1);
  squeezed_strides.reserve(base.ndim() - 1);
  for (int i = 0; i < base.ndim(); ++i) {
    if (i == axis) {
      continue;
    }
    squeezed_shape.push_back(base.shape(i));
    squeezed_strides.push_back(base.strides(i));
  }

  array view(squeezed_shape, base.dtype(), nullptr, {});
  const auto [data_size, row_contiguous, col_contiguous] =
      check_contiguity(squeezed_shape, squeezed_strides);
  view.copy_shared_buffer(
      base,
      squeezed_strides,
      {data_size == base.data_size(), row_contiguous, col_contiguous},
      base.data_size());
  return view;
}

array expand_unit_axis_view(const array& base, int axis) {
  Shape expanded_shape(base.shape().begin(), base.shape().end());
  Strides expanded_strides(base.strides().begin(), base.strides().end());
  expanded_shape.insert(expanded_shape.begin() + axis, 1);
  expanded_strides.insert(
      expanded_strides.begin() + axis,
      axis < base.ndim() ? base.strides(axis) : int64_t{1});

  array view(expanded_shape, base.dtype(), nullptr, {});
  const auto [data_size, row_contiguous, col_contiguous] =
      check_contiguity(expanded_shape, expanded_strides);
  view.copy_shared_buffer(
      base,
      expanded_strides,
      {data_size == base.data_size(), row_contiguous, col_contiguous},
      base.data_size());
  return view;
}

ConvPipelineConfig select_conv2d_pipeline(
    Dtype weight_dtype,
    uint32_t cout,
    uint32_t npq,
    uint32_t crs) {
  const auto& ctx = vulkan::VulkanContext::get();
  const uint32_t vendor_id = ctx.vendor_id();

  ConvPipelineConfig config{
      {64u, 32u, 32u},
      256u,
      4u,
      4u,
      0u,
      weight_dtype == float16 ? vulkan::StaticShaderId::conv2d_f16_f32_unroll
                              : vulkan::StaticShaderId::conv2d_f32_unroll};
  if (cout > 64 && npq >= 128) {
    config.block_size = {128u, 128u, 16u};
    config.thread_tile_k = 8u;
  } else if (cout <= 32 && npq >= 512) {
    config.block_size = {32u, 256u, 16u};
    config.thread_tile_k = 8u;
  }

  if (vendor_id == kVendorIdIntel) {
    config.shmem_pad = 0u;
    config.shader_id = weight_dtype == float16
        ? vulkan::StaticShaderId::conv2d_f16_f32
        : vulkan::StaticShaderId::conv2d_f32;
  }

  if (vendor_id == kVendorIdAmd) {
    config.shmem_pad =
        ctx.architecture() == vulkan::GpuArchitecture::AmdCdna ? 1u : 4u;

    if (ctx.coopmat_flash_attention_f32acc_supported() && crs >= 64 &&
        cout >= 64) {
      config.shmem_pad = 0u;
      config.shader_id = weight_dtype == float16
          ? vulkan::StaticShaderId::conv2d_f16_f32_cm1
          : vulkan::StaticShaderId::conv2d_f32_cm1;
    }
  }

  if (ctx.coopmat2_conv2d_supported()) {
    config.shader_id = weight_dtype == float16
        ? vulkan::StaticShaderId::conv2d_f16_f32_cm2
        : vulkan::StaticShaderId::conv2d_f32_cm2;
  }

  const uint32_t shmem_bytes =
      (config.block_size.k * (config.block_size.crs + config.shmem_pad) +
       config.block_size.crs * (config.block_size.npq + config.shmem_pad)) *
      sizeof(float);
  const uint32_t max_shmem =
      ctx.physical_device().getProperties().limits.maxComputeSharedMemorySize;
  if (shmem_bytes > max_shmem) {
    config.block_size.crs = std::min(crs, 8u);
  }

  return config;
}

array make_tracked_contiguous_copy(
    const array& src,
    Stream s,
    const char* name) {
  array out(src.shape(), src.dtype(), nullptr, {});
  out.set_data(vulkan::allocator().malloc(out.nbytes()));
  vulkan::record_primitive_for_stream(s, name);
  vulkan::begin_primitive_tracking(s, {src}, {out});
  copy_gpu(src, out, CopyType::General, s);
  vulkan::end_primitive_tracking(s, {src}, {out});
  return out;
}

bool try_eval_conv2d_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::vector<int>& kernel_strides,
    const std::vector<int>& padding_lo,
    const std::vector<int>& padding_hi,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s) {
  if (inputs.size() != 2 || inputs[0].ndim() != 4 || inputs[1].ndim() != 4 ||
      out.ndim() != 4) {
    return false;
  }
  if (kernel_strides.size() != 2 || padding_lo.size() != 2 ||
      padding_hi.size() != 2 || kernel_dilation.size() != 2 ||
      input_dilation.size() != 2) {
    return false;
  }
  if (groups <= 0) {
    return false;
  }

  const auto& in = inputs[0];
  const auto& wt = inputs[1];
  if (in.dtype() != float32 || out.dtype() != float32 ||
      (wt.dtype() != float32 && wt.dtype() != float16)) {
    return false;
  }

  const int batch_size = in.shape(0);
  const int in_height = in.shape(1);
  const int in_width = in.shape(2);
  const int in_channels = in.shape(3);
  const int out_channels = wt.shape(0);
  const int kernel_h = wt.shape(1);
  const int kernel_w = wt.shape(2);
  const int channels_per_group = wt.shape(3);
  const int out_height = out.shape(1);
  const int out_width = out.shape(2);

  if (in_channels != channels_per_group * groups ||
      out_channels % groups != 0) {
    return false;
  }

  const bool supports_direct_kernel = groups == 1 && !flip &&
      input_dilation[0] == 1 && input_dilation[1] == 1 &&
      padding_lo == padding_hi;

  if (!supports_direct_kernel) {
    auto in_work = in;
    if (input_dilation[0] != 1) {
      auto expanded = expand_dims(in_work, 2, s);
      auto padded =
          pad(expanded,
              std::vector<std::pair<int, int>>{
                  {0, 0}, {0, 0}, {0, input_dilation[0] - 1}, {0, 0}, {0, 0}},
              array(0, in.dtype()),
              "constant",
              s);
      auto reshaped = reshape(
          padded,
          {
              in_work.shape(0),
              in_work.shape(1) * input_dilation[0],
              in_work.shape(2),
              in_work.shape(3),
          },
          s);
      in_work = slice(
          reshaped,
          {0, 0, 0, 0},
          {in_work.shape(0),
           (in_work.shape(1) - 1) * input_dilation[0] + 1,
           in_work.shape(2),
           in_work.shape(3)},
          s);
    }
    if (input_dilation[1] != 1) {
      auto expanded = expand_dims(in_work, 3, s);
      auto padded =
          pad(expanded,
              std::vector<std::pair<int, int>>{
                  {0, 0}, {0, 0}, {0, 0}, {0, input_dilation[1] - 1}, {0, 0}},
              array(0, in.dtype()),
              "constant",
              s);
      auto reshaped = reshape(
          padded,
          {
              in_work.shape(0),
              in_work.shape(1),
              in_work.shape(2) * input_dilation[1],
              in_work.shape(3),
          },
          s);
      in_work = slice(
          reshaped,
          {0, 0, 0, 0},
          {in_work.shape(0),
           in_work.shape(1),
           (in_work.shape(2) - 1) * input_dilation[1] + 1,
           in_work.shape(3)},
          s);
    }

    auto wt_work = wt;
    if (flip) {
      const auto& wt_shape = wt.shape();
      const auto& wt_strides = wt.strides();
      wt_work = as_strided(
          wt,
          wt_shape,
          {wt_strides[0], -wt_strides[1], -wt_strides[2], wt_strides[3]},
          static_cast<size_t>(
              (wt_shape[1] - 1) * wt_strides[1] +
              (wt_shape[2] - 1) * wt_strides[2]),
          s);
    }

    auto padded =
        pad(in_work,
            std::vector<std::pair<int, int>>{
                {0, 0},
                {padding_lo[0], padding_hi[0]},
                {padding_lo[1], padding_hi[1]},
                {0, 0}},
            array(0, in.dtype()),
            "constant",
            s);

    const auto& padded_strides = padded.strides();
    auto patches = as_strided(
        padded,
        {batch_size, out_height, out_width, kernel_h, kernel_w, in_channels},
        {
            padded_strides[0],
            padded_strides[1] * kernel_strides[0],
            padded_strides[2] * kernel_strides[1],
            padded_strides[1] * kernel_dilation[0],
            padded_strides[2] * kernel_dilation[1],
            padded_strides[3],
        },
        0,
        s);

    const int out_channels_per_group = out_channels / groups;
    std::vector<array> group_outputs;
    group_outputs.reserve(groups);

    for (int g = 0; g < groups; ++g) {
      auto patches_g = slice(
          patches,
          {0, 0, 0, 0, 0, g * channels_per_group},
          {batch_size,
           out_height,
           out_width,
           kernel_h,
           kernel_w,
           (g + 1) * channels_per_group},
          s);
      patches_g = reshape(
          patches_g,
          {batch_size * out_height * out_width,
           kernel_h * kernel_w * channels_per_group},
          s);

      auto wt_g = slice(
          wt_work,
          {g * out_channels_per_group, 0, 0, 0},
          {(g + 1) * out_channels_per_group,
           kernel_h,
           kernel_w,
           channels_per_group},
          s);
      wt_g = reshape(
          wt_g,
          {out_channels_per_group, kernel_h * kernel_w * channels_per_group},
          s);
      wt_g = transpose(wt_g, {1, 0}, s);

      auto out_g = matmul(patches_g, wt_g, s);
      group_outputs.push_back(reshape(
          out_g,
          {batch_size, out_height, out_width, out_channels_per_group},
          s));
    }

    auto result = group_outputs.size() == 1
        ? group_outputs[0]
        : concatenate(std::move(group_outputs), 3, s);
    copy_gpu(result, out, CopyType::General, s);
    return true;
  }

  const uint32_t batch = checked_u32_size(in.shape(0), "batch");
  const uint32_t height = checked_u32_size(in.shape(1), "height");
  const uint32_t width = checked_u32_size(in.shape(2), "width");
  const uint32_t cin = checked_u32_size(in.shape(3), "channels");
  const uint32_t cout = checked_u32_size(wt.shape(0), "output channels");
  const uint32_t kh = checked_u32_size(wt.shape(1), "kernel height");
  const uint32_t kw = checked_u32_size(wt.shape(2), "kernel width");
  const uint32_t oh = checked_u32_size(out.shape(1), "output height");
  const uint32_t ow = checked_u32_size(out.shape(2), "output width");
  const uint32_t crs =
      checked_mul_u32(cin, checked_mul_u32(kh, kw, "kernel area"), "crs");
  const uint32_t npq =
      checked_mul_u32(batch, checked_mul_u32(oh, ow, "output pixels"), "npq");
  if (cout == 0 || cin == 0 || kh == 0 || kw == 0 || oh == 0 || ow == 0 ||
      npq == 0 || crs == 0) {
    out.set_data(vulkan::allocator().malloc(out.nbytes()));
    if (out.nbytes() > 0) {
      auto out_buf =
          static_cast<const vulkan::VulkanBuffer*>(out.buffer().ptr());
      vulkan::record_primitive_for_stream(s, "conv2d.zero_fill");
      auto cmd_buffer = vulkan::begin_command_recording(s.index);
      vkCmdFillBuffer(cmd_buffer, out_buf->buffer, 0, out.nbytes(), 0);
      vulkan::end_command_recording(s.index);
    }
    return true;
  }

  auto in_nchw = permute_view(
      in, in.shape(), in.strides(), std::array<int, 4>{0, 3, 1, 2});
  auto wt_ochw = permute_view(
      wt, wt.shape(), wt.strides(), std::array<int, 4>{0, 3, 1, 2});
  auto in_work = make_tracked_contiguous_copy(in_nchw, s, "conv2d.copy_input");
  auto wt_work = make_tracked_contiguous_copy(wt_ochw, s, "conv2d.copy_weight");

  Shape out_work_shape = {
      out.shape(0), out.shape(3), out.shape(1), out.shape(2)};
  array out_work(out_work_shape, out.dtype(), nullptr, {});
  out_work.set_data(vulkan::allocator().malloc(out_work.nbytes()));

  auto config = select_conv2d_pipeline(wt.dtype(), cout, npq, crs);
  std::vector<uint32_t> specialization_constants = {
      config.workgroup_size,
      config.block_size.k,
      config.block_size.crs,
      config.block_size.npq,
      config.thread_tile_k,
      config.use_collectives,
      config.shmem_pad,
      checked_u32_size(kernel_strides[1], "stride x"),
      checked_u32_size(kernel_strides[0], "stride y"),
      checked_u32_size(padding_lo[1], "pad x"),
      checked_u32_size(padding_lo[0], "pad y"),
      checked_u32_size(kernel_dilation[1], "dilation x"),
      checked_u32_size(kernel_dilation[0], "dilation y"),
      kw,
      kh,
  };

  auto bindings = conv_layout_bindings();
  auto& manager = vulkan::KernelManager::get();
  auto* pipeline = manager.get_pipeline(
      config.shader_id,
      bindings,
      static_cast<uint32_t>(sizeof(Conv2dPushConstants)),
      specialization_constants);
  if (pipeline == nullptr) {
    return false;
  }

  const uint64_t descriptor_epoch = vulkan::descriptor_epoch_for_stream(s);
  vk::DescriptorSet descriptor_set =
      manager.allocate_descriptor_set(pipeline->descriptor_layout);
  manager.defer_descriptor_set_free(s.index, descriptor_epoch, descriptor_set);

  std::array<VkDescriptorBufferInfo, 3> descriptor_infos = {{
      descriptor_buffer_info(wt_work, "weights"),
      descriptor_buffer_info(in_work, "input"),
      descriptor_buffer_info(out_work, "output"),
  }};
  std::array<VkWriteDescriptorSet, 3> descriptor_writes{};
  for (uint32_t i = 0; i < descriptor_writes.size(); ++i) {
    descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[i].dstSet = descriptor_set;
    descriptor_writes[i].dstBinding = i;
    descriptor_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[i].descriptorCount = 1;
    descriptor_writes[i].pBufferInfo = &descriptor_infos[i];
  }
  vkUpdateDescriptorSets(
      vulkan::VulkanContext::get().device(),
      static_cast<uint32_t>(descriptor_writes.size()),
      descriptor_writes.data(),
      0,
      nullptr);

  Conv2dPushConstants push_constants{};
  push_constants.Cout = cout;
  push_constants.Cin = cin;
  push_constants.N = batch;
  push_constants.W = width;
  push_constants.H = height;
  push_constants.OW = ow;
  push_constants.OH = oh;
  push_constants.nb01 = checked_u32_size(wt_work.strides(2), "weight stride 1");
  push_constants.nb02 = checked_u32_size(wt_work.strides(1), "weight stride 2");
  push_constants.nb03 = checked_u32_size(wt_work.strides(0), "weight stride 3");
  push_constants.nb11 = checked_u32_size(in_work.strides(2), "input stride 1");
  push_constants.nb12 = checked_u32_size(in_work.strides(1), "input stride 2");
  push_constants.nb13 = checked_u32_size(in_work.strides(0), "input stride 3");
  push_constants.nb1 = checked_u32_size(out_work.strides(2), "output stride 1");
  push_constants.nb2 = checked_u32_size(out_work.strides(1), "output stride 2");
  push_constants.nb3 = checked_u32_size(out_work.strides(0), "output stride 3");
  init_fastdiv_values(
      push_constants.OW, push_constants.OWmp, push_constants.OWL);
  init_fastdiv_values(
      checked_mul_u32(push_constants.OW, push_constants.OH, "ow*oh"),
      push_constants.OWOHmp,
      push_constants.OWOHL);

  const uint32_t grid_x = div_up_u32(cout, config.block_size.k);
  const uint32_t npq_blocks = div_up_u32(npq, config.block_size.npq);
  const uint32_t grid_y = std::min(npq_blocks, kMaxWorkgroupsY);
  const uint32_t grid_z = div_up_u32(npq_blocks, kMaxWorkgroupsY);

  vulkan::record_primitive_for_stream(s, "conv2d.direct");
  vulkan::begin_primitive_tracking(s, {wt_work, in_work}, {out_work});
  auto command_buffer = vulkan::begin_command_recording(s.index);
  vkCmdBindPipeline(
      command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
  VkDescriptorSet vk_descriptor_set =
      static_cast<VkDescriptorSet>(descriptor_set);
  vkCmdBindDescriptorSets(
      command_buffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline->layout,
      0,
      1,
      &vk_descriptor_set,
      0,
      nullptr);
  vkCmdPushConstants(
      command_buffer,
      pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(push_constants),
      &push_constants);
  vkCmdDispatch(command_buffer, grid_x, grid_y, grid_z);
  vulkan::end_command_recording(s.index);
  vulkan::end_primitive_tracking(s, {wt_work, in_work}, {out_work});

  out.set_data(vulkan::allocator().malloc(out.nbytes()));
  auto out_nhwc = permute_view(
      out_work,
      out_work.shape(),
      out_work.strides(),
      std::array<int, 4>{0, 2, 3, 1});
  vulkan::record_primitive_for_stream(s, "conv2d.copy_output");
  vulkan::begin_primitive_tracking(s, {out_nhwc}, {out});
  copy_gpu(out_nhwc, out, CopyType::General, s);
  vulkan::end_primitive_tracking(s, {out_nhwc}, {out});
  vulkan::retain_array_for_stream(s, in_work);
  vulkan::retain_array_for_stream(s, wt_work);
  vulkan::retain_array_for_stream(s, out_work);
  return true;
}

array reverse_kernel_1d(const array& wt, Stream s) {
  const auto& shape = wt.shape();
  const auto& strides = wt.strides();
  return as_strided(
      wt,
      shape,
      {strides[0], -strides[1], strides[2]},
      static_cast<size_t>((shape[1] - 1) * strides[1]),
      s);
}

array dilate_input_1d(const array& in, int dilation, Stream s) {
  if (dilation == 1) {
    return in;
  }

  auto expanded = expand_dims(in, 2, s);
  auto padded =
      pad(expanded,
          std::vector<std::pair<int, int>>{
              {0, 0}, {0, 0}, {0, dilation - 1}, {0, 0}},
          array(0, in.dtype()),
          "constant",
          s);
  auto reshaped =
      reshape(padded, {in.shape(0), in.shape(1) * dilation, in.shape(2)}, s);
  return slice(
      reshaped,
      {0, 0, 0},
      {in.shape(0), (in.shape(1) - 1) * dilation + 1, in.shape(2)},
      s);
}

bool is_row_contiguous_zero_offset(const array& arr) {
  return arr.flags().row_contiguous && arr.offset() == 0 &&
      (arr.ndim() == 0 || arr.strides(-1) == 1);
}

array ensure_conv1d_storage(array arr, Stream s, const char* name) {
  if (!vulkan::is_vulkan_storage_array(arr) ||
      !is_row_contiguous_zero_offset(arr)) {
    arr = make_tracked_contiguous_copy(arr, s, name);
  }
  return arr;
}

std::string conv1d_storage_type(Dtype dtype) {
  switch (dtype) {
    case float32:
      return "float";
    case float16:
      return "float16_t";
    case bfloat16:
      return "uint16_t";
    default:
      throw std::runtime_error("Unsupported Conv1d Vulkan dtype.");
  }
}

std::string conv1d_load_expr(const std::string& expr, Dtype dtype) {
  switch (dtype) {
    case float32:
      return expr;
    case float16:
      return "float(" + expr + ")";
    case bfloat16:
      return "bf16_to_fp32(uint(" + expr + "))";
    default:
      throw std::runtime_error("Unsupported Conv1d Vulkan dtype.");
  }
}

std::string build_conv1d_shader(Dtype in_dtype, Dtype wt_dtype) {
  std::ostringstream os;
  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
  if (in_dtype == float16 || wt_dtype == float16) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
  }
  if (in_dtype == float16 || wt_dtype == float16 || in_dtype == bfloat16 ||
      wt_dtype == bfloat16) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  os << "\nlayout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  if (in_dtype == bfloat16 || wt_dtype == bfloat16) {
    os << "float bf16_to_fp32(uint u) { return uintBitsToFloat(u << 16); }\n\n";
  }
  os << "layout(push_constant) uniform PushConstants {\n";
  os << "  uint total; uint batch; uint in_len; uint in_channels;\n";
  os << "  uint out_len; uint out_channels; uint kernel; uint channels_per_group;\n";
  os << "  uint out_channels_per_group; uint stride; uint padding; uint dilation;\n";
  os << "  uint input_dilation; uint flip;\n";
  os << "} pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer Input {"
     << conv1d_storage_type(in_dtype) << " data[];} in_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer Weight {"
     << conv1d_storage_type(wt_dtype) << " data[];} wt_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {float data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint idx = gl_GlobalInvocationID.x;\n";
  os << "  if (idx >= pc.total) return;\n";
  os << "  uint oc = idx % pc.out_channels;\n";
  os << "  uint ox = (idx / pc.out_channels) % pc.out_len;\n";
  os << "  uint b = idx / (pc.out_channels * pc.out_len);\n";
  os << "  uint group = oc / pc.out_channels_per_group;\n";
  os << "  uint ic_base = group * pc.channels_per_group;\n";
  os << "  float acc = 0.0;\n";
  os << "  for (uint k = 0; k < pc.kernel; ++k) {\n";
  os << "    int dilated_ix = int(ox * pc.stride + k * pc.dilation) - int(pc.padding);\n";
  os << "    if (dilated_ix < 0) continue;\n";
  os << "    if ((uint(dilated_ix) % pc.input_dilation) != 0) continue;\n";
  os << "    uint ix = uint(dilated_ix) / pc.input_dilation;\n";
  os << "    if (ix >= pc.in_len) continue;\n";
  os << "    uint wk = pc.flip != 0 ? (pc.kernel - 1 - k) : k;\n";
  os << "    for (uint c = 0; c < pc.channels_per_group; ++c) {\n";
  os << "      uint ic = ic_base + c;\n";
  os << "      uint in_index = (b * pc.in_len + ix) * pc.in_channels + ic;\n";
  os << "      uint wt_index = (oc * pc.kernel + wk) * pc.channels_per_group + c;\n";
  os << "      float x = "
     << conv1d_load_expr("in_buf.data[in_index]", in_dtype) << ";\n";
  os << "      float w = "
     << conv1d_load_expr("wt_buf.data[wt_index]", wt_dtype) << ";\n";
  os << "      acc += x * w;\n";
  os << "    }\n";
  os << "  }\n";
  os << "  out_buf.data[idx] = acc;\n";
  os << "}\n";
  return os.str();
}

struct Conv1dPushConstants {
  uint32_t total;
  uint32_t batch;
  uint32_t in_len;
  uint32_t in_channels;
  uint32_t out_len;
  uint32_t out_channels;
  uint32_t kernel;
  uint32_t channels_per_group;
  uint32_t out_channels_per_group;
  uint32_t stride;
  uint32_t padding;
  uint32_t dilation;
  uint32_t input_dilation;
  uint32_t flip;
};

bool try_eval_conv1d_direct_vulkan(
    array in,
    array wt,
    array& out,
    int stride,
    int padding,
    int dilation,
    int input_dilation,
    int groups,
    bool flip,
    Stream s) {
  if ((in.dtype() != float32 && in.dtype() != float16 &&
       in.dtype() != bfloat16) ||
      (wt.dtype() != float32 && wt.dtype() != float16 &&
       wt.dtype() != bfloat16)) {
    return false;
  }
  if (stride <= 0 || padding < 0 || dilation <= 0 || input_dilation <= 0 ||
      groups <= 0) {
    return false;
  }

  in = ensure_conv1d_storage(std::move(in), s, "conv1d.copy_input");
  wt = ensure_conv1d_storage(std::move(wt), s, "conv1d.copy_weight");

  array out_work(out.shape(), float32, nullptr, {});
  out_work.set_data(vulkan::allocator().malloc(out_work.nbytes()));
  if (out_work.size() == 0) {
    out.set_data(vulkan::allocator().malloc(out.nbytes()));
    copy_gpu(out_work, out, CopyType::General, s);
    return true;
  }

  Conv1dPushConstants pc{};
  pc.total = checked_u32_size(out_work.size(), "conv1d total");
  pc.batch = checked_u32_size(in.shape(0), "conv1d batch");
  pc.in_len = checked_u32_size(in.shape(1), "conv1d input length");
  pc.in_channels = checked_u32_size(in.shape(2), "conv1d input channels");
  pc.out_len = checked_u32_size(out.shape(1), "conv1d output length");
  pc.out_channels = checked_u32_size(out.shape(2), "conv1d output channels");
  pc.kernel = checked_u32_size(wt.shape(1), "conv1d kernel");
  pc.channels_per_group =
      checked_u32_size(wt.shape(2), "conv1d channels/group");
  pc.out_channels_per_group =
      checked_u32_size(out.shape(2) / groups, "conv1d output channels/group");
  pc.stride = checked_u32_size(stride, "conv1d stride");
  pc.padding = checked_u32_size(padding, "conv1d padding");
  pc.dilation = checked_u32_size(dilation, "conv1d dilation");
  pc.input_dilation = checked_u32_size(input_dilation, "conv1d input dilation");
  pc.flip = flip ? 1u : 0u;

  const std::string shader_name = "dynamic_conv1d_" +
      std::to_string(static_cast<int>(in.dtype().val())) + "_" +
      std::to_string(static_cast<int>(wt.dtype().val()));
  const std::string glsl_source = build_conv1d_shader(in.dtype(), wt.dtype());
  vulkan::DynamicArrayRef arrays[] = {{&in, 0}, {&wt, 1}, {&out_work, 2}};
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, glsl_source, 3, arrays, sizeof(Conv1dPushConstants), s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(Conv1dPushConstants),
      &pc);
  vkCmdDispatch(dispatch.command_buffer, (pc.total + 255u) / 256u, 1, 1);
  vulkan::end_command_recording(s.index);

  out.set_data(vulkan::allocator().malloc(out.nbytes()));
  copy_gpu(out_work, out, CopyType::General, s);
  vulkan::retain_array_for_stream(s, in);
  vulkan::retain_array_for_stream(s, wt);
  vulkan::retain_array_for_stream(s, out_work);
  return true;
}

bool try_eval_conv1d_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::vector<int>& kernel_strides,
    const std::vector<int>& padding_lo,
    const std::vector<int>& padding_hi,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s) {
  if (inputs.size() != 2 || inputs[0].ndim() != 3 || inputs[1].ndim() != 3 ||
      out.ndim() != 3) {
    return false;
  }

  if (kernel_strides.size() != 1 || padding_lo.size() != 1 ||
      padding_hi.size() != 1 || kernel_dilation.size() != 1 ||
      input_dilation.size() != 1) {
    return false;
  }

  if (groups <= 0) {
    return false;
  }

  const auto& in = inputs[0];
  const auto& wt = inputs[1];

  const int in_channels = in.shape(2);
  const int out_channels = wt.shape(0);
  const int channels_per_group = wt.shape(2);

  if (in_channels != channels_per_group * groups ||
      out_channels % groups != 0) {
    return false;
  }

  return try_eval_conv1d_direct_vulkan(
      in,
      wt,
      out,
      kernel_strides[0],
      padding_lo[0],
      kernel_dilation[0],
      input_dilation[0],
      groups,
      flip,
      s);
}

bool try_eval_conv2d_as_conv1d_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::vector<int>& kernel_strides,
    const std::vector<int>& padding_lo,
    const std::vector<int>& padding_hi,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s) {
  if (inputs.size() != 2 || inputs[0].ndim() != 4 || inputs[1].ndim() != 4 ||
      out.ndim() != 4) {
    return false;
  }
  if (kernel_strides.size() != 2 || padding_lo.size() != 2 ||
      padding_hi.size() != 2 || kernel_dilation.size() != 2 ||
      input_dilation.size() != 2) {
    return false;
  }

  const auto& in = inputs[0];
  const auto& wt = inputs[1];
  if (in.shape(2) != 1 || wt.shape(2) != 1 || out.shape(2) != 1 ||
      kernel_strides[1] != 1 || padding_lo[1] != 0 || padding_hi[1] != 0 ||
      kernel_dilation[1] != 1 || input_dilation[1] != 1) {
    return false;
  }

  array in_1d = squeeze_unit_axis_view(in, 2);
  array wt_1d = squeeze_unit_axis_view(wt, 2);
  array out_1d(
      {out.shape(0), out.shape(1), out.shape(3)}, out.dtype(), nullptr, {});
  if (!try_eval_conv1d_vulkan(
          {in_1d, wt_1d},
          out_1d,
          {kernel_strides[0]},
          {padding_lo[0]},
          {padding_hi[0]},
          {kernel_dilation[0]},
          {input_dilation[0]},
          groups,
          flip,
          s)) {
    return false;
  }

  out.copy_shared_buffer(expand_unit_axis_view(out_1d, 2));
  return true;
}

bool try_eval_conv3d_vulkan(
    const std::vector<array>& inputs,
    array& out,
    const std::vector<int>& kernel_strides,
    const std::vector<int>& padding_lo,
    const std::vector<int>& padding_hi,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    Stream s) {
  if (inputs.size() != 2 || inputs[0].ndim() != 5 || inputs[1].ndim() != 5 ||
      out.ndim() != 5) {
    return false;
  }
  if (kernel_strides.size() != 3 || padding_lo.size() != 3 ||
      padding_hi.size() != 3 || kernel_dilation.size() != 3 ||
      input_dilation.size() != 3 || groups <= 0) {
    return false;
  }

  const auto& in = inputs[0];
  const auto& wt = inputs[1];
  if (in.dtype() != float32 || out.dtype() != float32 ||
      (wt.dtype() != float32 && wt.dtype() != float16)) {
    return false;
  }

  const int batch_size = in.shape(0);
  const int in_depth = in.shape(1);
  const int in_height = in.shape(2);
  const int in_width = in.shape(3);
  const int in_channels = in.shape(4);
  const int out_channels = wt.shape(0);
  const int kernel_d = wt.shape(1);
  const int kernel_h = wt.shape(2);
  const int kernel_w = wt.shape(3);
  const int channels_per_group = wt.shape(4);
  const int out_depth = out.shape(1);
  const int out_height = out.shape(2);
  const int out_width = out.shape(3);

  if (in_channels != channels_per_group * groups ||
      out_channels % groups != 0) {
    return false;
  }

  auto in_work = in;
  if (input_dilation[0] != 1) {
    auto expanded = expand_dims(in_work, 2, s);
    auto padded = pad(
        expanded,
        std::vector<std::pair<int, int>>{
            {0, 0}, {0, 0}, {0, input_dilation[0] - 1}, {0, 0}, {0, 0}, {0, 0}},
        array(0, in.dtype()),
        "constant",
        s);
    auto reshaped = reshape(
        padded,
        {
            in_work.shape(0),
            in_work.shape(1) * input_dilation[0],
            in_work.shape(2),
            in_work.shape(3),
            in_work.shape(4),
        },
        s);
    in_work = slice(
        reshaped,
        {0, 0, 0, 0, 0},
        {
            in_work.shape(0),
            (in_work.shape(1) - 1) * input_dilation[0] + 1,
            in_work.shape(2),
            in_work.shape(3),
            in_work.shape(4),
        },
        s);
  }
  if (input_dilation[1] != 1) {
    auto expanded = expand_dims(in_work, 3, s);
    auto padded = pad(
        expanded,
        std::vector<std::pair<int, int>>{
            {0, 0}, {0, 0}, {0, 0}, {0, input_dilation[1] - 1}, {0, 0}, {0, 0}},
        array(0, in.dtype()),
        "constant",
        s);
    auto reshaped = reshape(
        padded,
        {
            in_work.shape(0),
            in_work.shape(1),
            in_work.shape(2) * input_dilation[1],
            in_work.shape(3),
            in_work.shape(4),
        },
        s);
    in_work = slice(
        reshaped,
        {0, 0, 0, 0, 0},
        {
            in_work.shape(0),
            in_work.shape(1),
            (in_work.shape(2) - 1) * input_dilation[1] + 1,
            in_work.shape(3),
            in_work.shape(4),
        },
        s);
  }
  if (input_dilation[2] != 1) {
    auto expanded = expand_dims(in_work, 4, s);
    auto padded = pad(
        expanded,
        std::vector<std::pair<int, int>>{
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, input_dilation[2] - 1}, {0, 0}},
        array(0, in.dtype()),
        "constant",
        s);
    auto reshaped = reshape(
        padded,
        {
            in_work.shape(0),
            in_work.shape(1),
            in_work.shape(2),
            in_work.shape(3) * input_dilation[2],
            in_work.shape(4),
        },
        s);
    in_work = slice(
        reshaped,
        {0, 0, 0, 0, 0},
        {
            in_work.shape(0),
            in_work.shape(1),
            in_work.shape(2),
            (in_work.shape(3) - 1) * input_dilation[2] + 1,
            in_work.shape(4),
        },
        s);
  }

  auto wt_work = wt;
  if (flip) {
    const auto& wt_shape = wt.shape();
    const auto& wt_strides = wt.strides();
    wt_work = as_strided(
        wt,
        wt_shape,
        {
            wt_strides[0],
            -wt_strides[1],
            -wt_strides[2],
            -wt_strides[3],
            wt_strides[4],
        },
        static_cast<size_t>(
            (wt_shape[1] - 1) * wt_strides[1] +
            (wt_shape[2] - 1) * wt_strides[2] +
            (wt_shape[3] - 1) * wt_strides[3]),
        s);
  }

  auto padded =
      pad(in_work,
          std::vector<std::pair<int, int>>{
              {0, 0},
              {padding_lo[0], padding_hi[0]},
              {padding_lo[1], padding_hi[1]},
              {padding_lo[2], padding_hi[2]},
              {0, 0}},
          array(0, in.dtype()),
          "constant",
          s);

  const auto& padded_strides = padded.strides();
  auto patches = as_strided(
      padded,
      {
          batch_size,
          out_depth,
          out_height,
          out_width,
          kernel_d,
          kernel_h,
          kernel_w,
          in_channels,
      },
      {
          padded_strides[0],
          padded_strides[1] * kernel_strides[0],
          padded_strides[2] * kernel_strides[1],
          padded_strides[3] * kernel_strides[2],
          padded_strides[1] * kernel_dilation[0],
          padded_strides[2] * kernel_dilation[1],
          padded_strides[3] * kernel_dilation[2],
          padded_strides[4],
      },
      0,
      s);

  const int out_channels_per_group = out_channels / groups;
  std::vector<array> group_outputs;
  group_outputs.reserve(groups);

  for (int g = 0; g < groups; ++g) {
    auto patches_g = slice(
        patches,
        {0, 0, 0, 0, 0, 0, 0, g * channels_per_group},
        {
            batch_size,
            out_depth,
            out_height,
            out_width,
            kernel_d,
            kernel_h,
            kernel_w,
            (g + 1) * channels_per_group,
        },
        s);
    patches_g = reshape(
        patches_g,
        {
            batch_size * out_depth * out_height * out_width,
            kernel_d * kernel_h * kernel_w * channels_per_group,
        },
        s);

    auto wt_g = slice(
        wt_work,
        {g * out_channels_per_group, 0, 0, 0, 0},
        {
            (g + 1) * out_channels_per_group,
            kernel_d,
            kernel_h,
            kernel_w,
            channels_per_group,
        },
        s);
    wt_g = reshape(
        wt_g,
        {
            out_channels_per_group,
            kernel_d * kernel_h * kernel_w * channels_per_group,
        },
        s);
    wt_g = transpose(wt_g, {1, 0}, s);

    auto out_g = matmul(patches_g, wt_g, s);
    group_outputs.push_back(reshape(
        out_g,
        {batch_size, out_depth, out_height, out_width, out_channels_per_group},
        s));
  }

  auto result = group_outputs.size() == 1
      ? group_outputs[0]
      : concatenate(std::move(group_outputs), 4, s);
  copy_gpu(result, out, CopyType::General, s);
  return true;
}

} // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (try_eval_conv2d_as_conv1d_vulkan(
          inputs,
          out,
          kernel_strides_,
          padding_lo_,
          padding_hi_,
          kernel_dilation_,
          input_dilation_,
          groups_,
          flip_,
          stream())) {
    return;
  }

  if (try_eval_conv2d_vulkan(
          inputs,
          out,
          kernel_strides_,
          padding_lo_,
          padding_hi_,
          kernel_dilation_,
          input_dilation_,
          groups_,
          flip_,
          stream())) {
    return;
  }

  if (try_eval_conv3d_vulkan(
          inputs,
          out,
          kernel_strides_,
          padding_lo_,
          padding_hi_,
          kernel_dilation_,
          input_dilation_,
          groups_,
          flip_,
          stream())) {
    return;
  }

  if (try_eval_conv1d_vulkan(
          inputs,
          out,
          kernel_strides_,
          padding_lo_,
          padding_hi_,
          kernel_dilation_,
          input_dilation_,
          groups_,
          flip_,
          stream())) {
    return;
  }

  throw std::runtime_error(
      "Convolution has no Vulkan implementation for this input.");
}

} // namespace mlx::core
