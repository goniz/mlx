// Copyright © 2024 Apple Inc.

#include <limits>

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  if (num_keys == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }

  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;

  if (keys.dtype() != uint32) {
    throw std::runtime_error(
        "RandomBits failed on Vulkan (only uint32 keys supported).");
  }

  if (!keys.flags().contiguous || keys.offset() != 0 ||
      keys.strides().back() != 1) {
    keys = contiguous_copy_gpu(keys, stream());
  }

  out.set_data(allocator::malloc(out.nbytes()));
  size_t out_skip = (bytes_per_key + 4 - 1) / 4;
  size_t half_size = out_skip / 2;
  bool odd = out_skip % 2 != 0;

  if (num_keys > std::numeric_limits<uint32_t>::max() ||
      bytes_per_key > std::numeric_limits<uint32_t>::max() ||
      out_skip > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("RandomBits failed on Vulkan (shape too large).");
  }

  vulkan::RandomBitsPushConstants push_constants{
      static_cast<uint32_t>(num_keys),
      static_cast<uint32_t>(bytes_per_key),
      odd ? 1u : 0u,
      static_cast<uint32_t>(out_skip)};

  auto command_buffer = vulkan::begin_command_recording(stream().index);
  vulkan::dispatch_random_bits_op(
      keys,
      out,
      vulkan::StaticShaderId::random_bits_f32,
      command_buffer,
      stream(),
      push_constants,
      {static_cast<uint32_t>(num_keys),
       static_cast<uint32_t>(half_size + odd),
       1});
  vulkan::end_command_recording(stream().index);
}

} // namespace mlx::core
