// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/utils.h"

namespace mlx::core {

void broadcast(const array& in, array& out) {
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  Strides strides(out.ndim(), 0);
  int diff = out.ndim() - in.ndim();
  for (int i = in.ndim() - 1; i >= 0; --i) {
    strides[i + diff] = (in.shape()[i] == 1) ? 0 : in.strides()[i];
  }
  auto flags = in.flags();
  if (out.size() > in.size()) {
    flags.row_contiguous = flags.col_contiguous = false;
  }
  auto [data_size, row_contiguous, col_contiguous] =
      check_contiguity(out.shape(), strides);
  flags.contiguous = data_size == out.size();
  flags.row_contiguous = row_contiguous;
  flags.col_contiguous = col_contiguous;
  out.copy_shared_buffer(in, strides, flags, data_size);
}

} // namespace mlx::core
