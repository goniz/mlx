// Copyright © 2024 Apple Inc.

#include <algorithm>

#include "mlx/backend/cpu/threefry.h"
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

  if (width_ == 4 && bytes_per_key % 4 == 0) {
    if (!keys.flags().contiguous || keys.offset() != 0 || keys.strides().back() != 1) {
      keys = contiguous_copy_gpu(keys, stream());
    }
    keys.wait();

    auto* kptr = keys.data<uint32_t>();
    std::vector<uint32_t> host_out(out.size());
    size_t out_skip = bytes_per_key / 4;
    size_t half_size = out_skip / 2;
    bool even = out_skip % 2 == 0;
    for (size_t i = 0; i < num_keys; ++i) {
      auto key = std::make_pair(kptr[2 * i], kptr[2 * i + 1]);
      auto* dst = host_out.data() + i * out_skip;

      std::pair<uintptr_t, uintptr_t> count{0, half_size + !even};
      for (; count.first + 1 < half_size; count.first++, count.second++) {
        std::tie(dst[count.first], dst[count.second]) =
            random::threefry2x32_hash(key, count);
      }
      if (count.first < half_size) {
        auto rb = random::threefry2x32_hash(key, count);
        dst[count.first++] = rb.first;
        dst[count.second] = rb.second;
      }
      if (!even) {
        count.second = 0;
        dst[half_size] = random::threefry2x32_hash(key, count).first;
      }
    }
    copy_gpu(
        array(host_out.begin(), out.shape(), uint32),
        out,
        CopyType::GeneralGeneral,
        stream());
    return;
  }

  if (!keys.flags().contiguous || keys.offset() != 0 || keys.strides().back() != 1) {
    keys = contiguous_copy_gpu(keys, stream());
  }
  keys.wait();

  auto* kptr = keys.data<uint32_t>();
  auto host_out = std::make_shared<std::vector<char>>(out.nbytes());
  auto* cptr = host_out->data();
  auto copy_word = [&](char* dst, size_t word_idx, uint32_t v) {
    const size_t byte_offset = 4 * word_idx;
    if (byte_offset + 4 <= bytes_per_key) {
      std::copy(
          reinterpret_cast<const char*>(&v),
          reinterpret_cast<const char*>(&v) + 4,
          dst + byte_offset);
    } else {
      std::copy(
          reinterpret_cast<const char*>(&v),
          reinterpret_cast<const char*>(&v) + (bytes_per_key - byte_offset),
          dst + byte_offset);
    }
  };

  size_t out_skip = (bytes_per_key + 4 - 1) / 4;
  size_t half_size = out_skip / 2;
  bool even = out_skip % 2 == 0;
  for (size_t i = 0; i < num_keys; ++i, cptr += bytes_per_key) {
    auto key = std::make_pair(kptr[2 * i], kptr[2 * i + 1]);

    std::pair<uintptr_t, uintptr_t> count{0, half_size + !even};
    for (; count.first + 1 < half_size; count.first++, count.second++) {
      auto rb = random::threefry2x32_hash(key, count);
      copy_word(cptr, count.first, rb.first);
      copy_word(cptr, count.second, rb.second);
    }
    if (count.first < half_size) {
      auto rb = random::threefry2x32_hash(key, count);
      copy_word(cptr, count.first++, rb.first);
      copy_word(cptr, count.second, rb.second);
    }
    if (!even) {
      count.second = 0;
      copy_word(cptr, half_size, random::threefry2x32_hash(key, count).first);
    }
  }
  copy_gpu(
      array(
          static_cast<void*>(host_out->data()),
          out.shape(),
          out.dtype(),
          [host_out](void*) {}),
      out,
      CopyType::GeneralGeneral,
      stream());
}

} // namespace mlx::core
