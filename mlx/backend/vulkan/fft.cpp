// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <complex>
#include <limits>
#include <vector>

#include "mlx/3rdparty/pocketfft.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

constexpr int MAX_STOCKHAM_FFT_SIZE = 4096;
constexpr int MAX_FOUR_STEP_FFT_SIZE = 1 << 20;
constexpr double kPi = 3.14159265358979323846;

inline const std::vector<int>& supported_radices() {
  static const std::vector<int> kRadices = {13, 11, 8, 7, 6, 5, 4, 3, 2};
  return kRadices;
}

void fft_op(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    const Stream& s);

bool try_plan_stockham_fft(int n, std::vector<int>& plan) {
  plan.assign(supported_radices().size(), 0);
  if (n > MAX_STOCKHAM_FFT_SIZE) {
    return false;
  }

  int orig_n = n;
  if (n == 1) {
    return true;
  }
  if (orig_n == 6) {
    // The smaller codelets are a little more accurate for length-6 real
    // inverse transforms than the composite radix-6 path.
    for (int i = 0; i < supported_radices().size(); ++i) {
      if (supported_radices()[i] == 3 || supported_radices()[i] == 2) {
        plan[i] = 1;
      }
    }
    return true;
  }
  for (int i = 0; i < supported_radices().size(); ++i) {
    const int radix = supported_radices()[i];
    if (orig_n == 4004 && radix == 4) {
      continue;
    }
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

std::pair<array, array>
compute_bluestein_constants(int n, int bluestein_n, const Stream& s) {
  std::vector<std::complex<float>> w_k_vec(n);
  std::vector<std::complex<float>> w_q_vec(bluestein_n, {0.0f, 0.0f});

  for (int i = -n + 1; i < n; ++i) {
    double theta = static_cast<double>(i) * static_cast<double>(i) * kPi /
        static_cast<double>(n);
    w_q_vec[i + n - 1] = std::exp(std::complex<double>(0.0, theta));
    if (i >= 0) {
      w_k_vec[i] = std::exp(std::complex<double>(0.0, -theta));
    }
  }

  std::vector<std::complex<float>> w_q_fft(bluestein_n);
  std::ptrdiff_t item_size = sizeof(complex64_t);
  size_t fft_size = bluestein_n;
  pocketfft::c2c(
      /* shape= */ {fft_size},
      /* stride_in= */ {item_size},
      /* stride_out= */ {item_size},
      /* axes= */ {0},
      /* forward= */ true,
      /* data_in= */ w_q_vec.data(),
      /* data_out= */ w_q_fft.data(),
      /* scale= */ 1.0f);

  array w_k(Shape{n}, complex64, nullptr, {});
  w_k.set_data(allocator::malloc(w_k.nbytes()));
  auto* w_k_buf = static_cast<vulkan::VulkanBuffer*>(w_k.buffer().ptr());
  vulkan::enqueue_owned_staging_upload(
      s,
      w_k_vec.data(),
      w_k.nbytes(),
      w_k_buf->buffer,
      w_k.offset(),
      w_k.data_shared_ptr());

  array w_q(Shape{bluestein_n}, complex64, nullptr, {});
  w_q.set_data(allocator::malloc(w_q.nbytes()));
  auto* w_q_buf = static_cast<vulkan::VulkanBuffer*>(w_q.buffer().ptr());
  vulkan::enqueue_owned_staging_upload(
      s,
      w_q_fft.data(),
      w_q.nbytes(),
      w_q_buf->buffer,
      w_q.offset(),
      w_q.data_shared_ptr());
  return {w_k, w_q};
}

array make_axis_broadcast_view(
    const array& base,
    const Shape& shape,
    int axis) {
  Strides strides(shape.size(), 0);
  strides[axis] = 1;
  array view(shape, base.dtype(), nullptr, {});
  view.copy_shared_buffer(
      base, strides, {false, false, false}, base.data_size());
  return view;
}

array make_axis_contiguous_copy(const array& x, int axis, Stream s) {
  bool has_vulkan_storage =
      x.data_shared_ptr() != nullptr && vulkan::is_vulkan_buffer(x.buffer());
  // The Stockham shader indexes batches as dense contiguous rows:
  //   base = batch * n
  // So the full array must be row-contiguous, not just the FFT axis.
  bool no_copy =
      has_vulkan_storage && x.strides()[axis] == 1 && x.flags().row_contiguous;
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

bool try_eval_fft_four_step_c2c_vulkan(
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
  if (n <= MAX_STOCKHAM_FFT_SIZE || n > MAX_FOUR_STEP_FFT_SIZE ||
      !is_power_of_2(n)) {
    return false;
  }

  const int n2 = n > 65536 ? 1024 : 64;
  const int n1 = n / n2;
  std::vector<int> plan1;
  std::vector<int> plan2;
  if (!try_plan_stockham_fft(n1, plan1) || !try_plan_stockham_fft(n2, plan2)) {
    return false;
  }

  array in_kernel = in;
  array out_kernel = out;
  if (normalized_axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in_kernel, normalized_axis, in.ndim() - 1);
    out_kernel = swapaxes_in_eval(out_kernel, normalized_axis, out.ndim() - 1);
  }

  in_kernel = make_axis_contiguous_copy(in_kernel, in_kernel.ndim() - 1, s);
  array temp(in_kernel.shape(), complex64, nullptr, {});
  temp.set_data(allocator::malloc(temp.nbytes()));
  array out_kernel_storage(out_kernel.shape(), out_kernel.dtype(), nullptr, {});
  out_kernel_storage.set_data(allocator::malloc(out_kernel_storage.nbytes()));

  const uint64_t in_offset = in_kernel.offset() / size_of(in_kernel.dtype());
  const uint64_t temp_offset = temp.offset() / size_of(temp.dtype());
  const uint64_t out_offset =
      out_kernel_storage.offset() / size_of(out_kernel_storage.dtype());
  const uint64_t total = out_kernel_storage.data_size();
  if (in_offset > std::numeric_limits<uint32_t>::max() ||
      temp_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  auto make_spec = [&](const std::vector<int>& plan, uint32_t step) {
    std::vector<uint32_t> spec;
    spec.reserve(plan.size() + 3);
    for (int count : plan) {
      spec.push_back(static_cast<uint32_t>(count));
    }
    spec.push_back(static_cast<uint32_t>(n1));
    spec.push_back(static_cast<uint32_t>(n2));
    spec.push_back(step);
    return spec;
  };

  vulkan::FFTPushConstants pc0{};
  pc0.in_offset = static_cast<uint32_t>(in_offset);
  pc0.out_offset = static_cast<uint32_t>(temp_offset);
  pc0.batch_count = static_cast<uint32_t>((total / n) * n2);
  pc0.n = static_cast<uint32_t>(n1);
  pc0.inverse = inverse ? 1u : 0u;

  vulkan::FFTPushConstants pc1{};
  pc1.in_offset = static_cast<uint32_t>(temp_offset);
  pc1.out_offset = static_cast<uint32_t>(out_offset);
  pc1.batch_count = static_cast<uint32_t>((total / n) * n1);
  pc1.n = static_cast<uint32_t>(n2);
  pc1.inverse = inverse ? 1u : 0u;

  auto command_buffer = vulkan::begin_command_recording(s.index);
  vulkan::dispatch_fft_op(
      in_kernel,
      temp,
      vulkan::StaticShaderId::fft_four_step_c64,
      command_buffer,
      s,
      pc0,
      make_spec(plan1, 0));

  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(
      command_buffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);

  vulkan::dispatch_fft_op(
      temp,
      out_kernel_storage,
      vulkan::StaticShaderId::fft_four_step_c64,
      command_buffer,
      s,
      pc1,
      make_spec(plan2, 1));
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

bool try_eval_fft_bluestein_c2c_vulkan(
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
  if (n <= 0 || is_power_of_2(n)) {
    return false;
  }

  const int64_t min_bluestein_size = 2ll * n - 1;
  if (min_bluestein_size > MAX_FOUR_STEP_FFT_SIZE) {
    return false;
  }
  const int bluestein_n = next_power_of_2(static_cast<int>(min_bluestein_size));
  if (bluestein_n > MAX_FOUR_STEP_FFT_SIZE) {
    return false;
  }

  auto [w_k, w_q] = compute_bluestein_constants(n, bluestein_n, s);

  Shape temp_shape = in.shape();
  array temp = in;
  if (inverse) {
    temp = array(temp_shape, complex64, nullptr, {});
    Conjugate(s).eval_gpu({in}, temp);
  }

  array w_k_broadcast =
      make_axis_broadcast_view(w_k, temp_shape, normalized_axis);
  array temp1(temp_shape, complex64, nullptr, {});
  Multiply(s).eval_gpu({temp, w_k_broadcast}, temp1);

  Shape padded_shape = temp_shape;
  padded_shape[normalized_axis] = bluestein_n;
  array zero(std::complex<float>(0.0f, 0.0f), complex64);
  array pad_temp(padded_shape, complex64, nullptr, {});
  pad_gpu(temp1, zero, pad_temp, {normalized_axis}, {0}, s);

  array pad_temp1(padded_shape, complex64, nullptr, {});
  fft_op(
      pad_temp,
      pad_temp1,
      normalized_axis,
      /*inverse=*/false,
      /*real=*/false,
      s);

  array w_q_broadcast =
      make_axis_broadcast_view(w_q, padded_shape, normalized_axis);
  Multiply(s).eval_gpu({pad_temp1, w_q_broadcast}, pad_temp);

  fft_op(
      pad_temp,
      pad_temp1,
      normalized_axis,
      /*inverse=*/true,
      /*real=*/false,
      s);

  Shape starts(in.ndim(), 0);
  Shape strides(in.ndim(), 1);
  starts[normalized_axis] = n - 1;

  array temp2(temp_shape, complex64, nullptr, {});
  slice_gpu(pad_temp1, temp2, starts, strides, s);

  array temp3(temp_shape, complex64, nullptr, {});
  Multiply(s).eval_gpu({temp2, w_k_broadcast}, temp3);

  if (inverse) {
    array conj_out(temp_shape, complex64, nullptr, {});
    Conjugate(s).eval_gpu({temp3}, conj_out);
    array inv_n(
        std::complex<float>(1.0f / static_cast<float>(n), 0.0f), complex64);
    Multiply(s).eval_gpu({conj_out, inv_n}, out);
  } else {
    copy_gpu(temp3, out, CopyType::General, s);
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
  if (!try_eval_fft_stockham_c2c_vulkan(in, out, normalized_axis, inverse, s) &&
      !try_eval_fft_four_step_c2c_vulkan(
          in, out, normalized_axis, inverse, s) &&
      !try_eval_fft_bluestein_c2c_vulkan(
          in, out, normalized_axis, inverse, s)) {
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
