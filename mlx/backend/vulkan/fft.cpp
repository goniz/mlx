// Copyright © 2024 Apple Inc.

#include <limits>
#include <vector>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

constexpr int MAX_STOCKHAM_FFT_SIZE = 4096;

inline const std::vector<int>& supported_radices() {
  static const std::vector<int> kRadices = {13, 11, 8, 7, 6, 5, 4, 3, 2};
  return kRadices;
}

bool try_plan_stockham_fft(int n, std::vector<int>& plan) {
  plan.assign(supported_radices().size(), 0);
  if (n > MAX_STOCKHAM_FFT_SIZE) {
    return false;
  }

  int orig_n = n;
  if (n == 1) {
    return true;
  }

  for (int i = 0; i < supported_radices().size(); ++i) {
    const int radix = supported_radices()[i];
    if (is_power_of_2(orig_n) && orig_n < 512 && radix > 4) {
      continue;
    }
    while (n % radix == 0) {
      plan[i] += 1;
      n /= radix;
      if (n == 1) {
        return true;
      }
    }
  }
  return false;
}

array make_axis_contiguous_copy(const array& x, int axis, Stream s) {
  bool has_vulkan_storage =
      x.data_shared_ptr() != nullptr && vulkan::is_vulkan_buffer(x.buffer());
  bool no_copy = has_vulkan_storage && x.strides()[axis] == 1 &&
      (x.flags().row_contiguous || x.flags().col_contiguous);
  if (no_copy) {
    return x;
  }

  return contiguous_copy_gpu(x, s);
}

bool try_eval_fft_stockham_c2c_vulkan(
    const array& in,
    array& out,
    int axis,
    bool inverse,
    Stream s) {
  if (in.dtype() != complex64 || out.dtype() != complex64) {
    return false;
  }

  const int normalized_axis = normalize_axis(axis, in.ndim());
  const int n = static_cast<int>(in.shape(normalized_axis));
  std::vector<int> stockham_plan;
  if (!try_plan_stockham_fft(n, stockham_plan)) {
    return false;
  }

  // The current shader only safely covers stages where one 256-thread wave can
  // read every butterfly input before writes begin.
  const auto radices = supported_radices();
  for (int i = 0; i < stockham_plan.size(); ++i) {
    if (stockham_plan[i] > 0 && (n / radices[i]) > 256) {
      return false;
    }
  }

  array in_kernel = in;
  array out_kernel = out;
  if (normalized_axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in_kernel, normalized_axis, in.ndim() - 1);
    out_kernel = swapaxes_in_eval(out_kernel, normalized_axis, out.ndim() - 1);
  }

  in_kernel = make_axis_contiguous_copy(in_kernel, in_kernel.ndim() - 1, s);
  if (out_kernel.size() == 0) {
    out.set_data(allocator::malloc(0));
    if (normalized_axis != in.ndim() - 1) {
      array out_kernel_storage(
          out_kernel.shape(), out_kernel.dtype(), nullptr, {});
      out_kernel_storage.set_data(allocator::malloc(0));
      array restored =
          swapaxes_in_eval(out_kernel_storage, out.ndim() - 1, normalized_axis);
      copy_gpu(restored, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  array out_kernel_storage(out_kernel.shape(), out_kernel.dtype(), nullptr, {});
  out_kernel_storage.set_data(allocator::malloc(out_kernel_storage.nbytes()));

  const uint64_t in_offset = in_kernel.offset() / size_of(in_kernel.dtype());
  const uint64_t out_offset =
      out_kernel_storage.offset() / size_of(out_kernel_storage.dtype());
  const uint64_t total = out_kernel_storage.data_size();
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  vulkan::FFTPushConstants pc{};
  pc.in_offset = static_cast<uint32_t>(in_offset);
  pc.out_offset = static_cast<uint32_t>(out_offset);
  pc.batch_count = static_cast<uint32_t>(total / n);
  pc.n = static_cast<uint32_t>(n);
  pc.inverse = inverse ? 1u : 0u;

  std::vector<uint32_t> specialization_constants;
  specialization_constants.reserve(stockham_plan.size());
  for (int step : stockham_plan) {
    specialization_constants.push_back(static_cast<uint32_t>(step));
  }

  auto command_buffer = vulkan::begin_command_recording(s.index);
  vulkan::dispatch_fft_op(
      in_kernel,
      out_kernel_storage,
      vulkan::StaticShaderId::fft_c64,
      command_buffer,
      s,
      pc,
      specialization_constants);
  vulkan::end_command_recording(s.index);

  if (normalized_axis != in.ndim() - 1) {
    array restored =
        swapaxes_in_eval(out_kernel_storage, out.ndim() - 1, normalized_axis);
    copy_gpu(restored, out, CopyType::GeneralGeneral, s);
  } else {
    copy_gpu(out_kernel_storage, out, CopyType::GeneralGeneral, s);
  }
  return true;
}

void fft_op(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    const Stream& s) {
  const int normalized_axis = normalize_axis(axis, in.ndim());
  if (real) {
    if (inverse) {
      const int n = static_cast<int>(out.shape(normalized_axis));
      const int back_offset = (n % 2 == 0) ? 2 : 1;

      array full_spectrum(out.shape(), complex64, nullptr, {});
      if (in.shape(normalized_axis) > back_offset) {
        Shape slice_shape = in.shape();
        slice_shape[normalized_axis] -= back_offset;
        array reversed_tail(slice_shape, complex64, nullptr, {});
        array mirrored_tail(slice_shape, complex64, nullptr, {});
        Shape starts(in.ndim(), 0);
        Shape strides(in.ndim(), 1);
        starts[normalized_axis] = in.shape(normalized_axis) - back_offset;
        strides[normalized_axis] = -1;
        slice_gpu(in, reversed_tail, starts, strides, s);
        Conjugate(s).eval_gpu({reversed_tail}, mirrored_tail);

        concatenate_gpu({in, mirrored_tail}, full_spectrum, normalized_axis, s);
      } else {
        copy_gpu(in, full_spectrum, CopyType::General, s);
      }

      array full_inverse(out.shape(), complex64, nullptr, {});
      fft_op(
          full_spectrum,
          full_inverse,
          normalized_axis,
          /*inverse=*/true,
          /*real=*/false,
          s);
      AsType(s, float32).eval_gpu({full_inverse}, out);
      return;
    }

    array complex_in(in.shape(), complex64, nullptr, {});
    copy_gpu(in, complex_in, CopyType::General, s);

    array full_fft(in.shape(), complex64, nullptr, {});
    fft_op(
        complex_in,
        full_fft,
        normalized_axis,
        /*inverse=*/false,
        /*real=*/false,
        s);

    Shape starts(out.ndim(), 0);
    Shape strides(out.ndim(), 1);
    slice_gpu(full_fft, out, starts, strides, s);
    return;
  }
  if (!try_eval_fft_stockham_c2c_vulkan(in, out, normalized_axis, inverse, s)) {
    throw std::runtime_error("FFT has no Vulkan implementation.");
  }
}

void nd_fft_op(
    const array& in,
    array& out,
    const std::vector<size_t>& axes,
    bool inverse,
    bool real,
    const Stream& s) {
  auto temp_shape = inverse ? in.shape() : out.shape();
  std::vector<array> temp_arrs;
  temp_arrs.emplace_back(temp_shape, complex64, nullptr, std::vector<array>{});
  if (axes.size() > 2) {
    temp_arrs.emplace_back(
        temp_shape, complex64, nullptr, std::vector<array>{});
  }
  for (int i = static_cast<int>(axes.size()) - 1; i >= 0; --i) {
    int reverse_index = static_cast<int>(axes.size()) - i - 1;
    int index = inverse ? reverse_index : i;
    size_t axis = axes[index];
    bool step_real = (real && index == static_cast<int>(axes.size()) - 1);
    const array& in_arr =
        i == static_cast<int>(axes.size()) - 1 ? in : temp_arrs[i % 2];
    array& out_arr = i == 0 ? out : temp_arrs[1 - i % 2];
    fft_op(in_arr, out_arr, axis, inverse, step_real, s);
  }
}

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  if (axes_.size() > 1) {
    nd_fft_op(in, out, axes_, inverse_, real_, stream());
  } else {
    fft_op(in, out, axes_[0], inverse_, real_, stream());
  }
}

} // namespace mlx::core
