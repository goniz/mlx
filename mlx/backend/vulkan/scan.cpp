// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"

namespace mlx::core {

namespace {

bool has_vulkan_buffer(const array& arr) {
  auto data = arr.data_shared_ptr();
  return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
}

bool try_eval_scan_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    Stream s) {
  if (inputs.size() != 1 ||
      (reduce_type != Scan::Sum && reduce_type != Scan::Prod &&
       reduce_type != Scan::Min && reduce_type != Scan::Max &&
       reduce_type != Scan::LogAddExp)) {
    return false;
  }

  array in = inputs[0];
  const bool cumsum_i32 =
      reduce_type == Scan::Sum && in.dtype() == int32 && out.dtype() == int32;
  const bool cumprod_i32 =
      reduce_type == Scan::Prod && in.dtype() == int32 && out.dtype() == int32;
  const bool scan_f32 = in.dtype() == float32 && out.dtype() == float32;
  if (in.ndim() == 0 || (!scan_f32 && !cumsum_i32 && !cumprod_i32)) {
    return false;
  }

  int normalized_axis = normalize_axis(axis, in.ndim());
  if (normalized_axis < 0 || normalized_axis >= in.ndim()) {
    return false;
  }

  array in_kernel = in;
  if (normalized_axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, normalized_axis, in.ndim() - 1);
  }

  array scan_input = in_kernel;

  if (!scan_input.flags().contiguous || scan_input.offset() != 0 ||
      scan_input.strides().back() != 1 ||
      !is_supported_unary_layout(scan_input)) {
    scan_input = contiguous_copy_gpu(scan_input, s);
  }
  if (!has_vulkan_buffer(scan_input)) {
    scan_input = contiguous_copy_gpu(scan_input, s);
  }

  Shape scan_shape = out.shape();
  if (normalized_axis != in.ndim() - 1) {
    std::swap(scan_shape[normalized_axis], scan_shape[in.ndim() - 1]);
  }
  if (scan_input.shape() != scan_shape) {
    return false;
  }

  if (scan_input.size() > std::numeric_limits<uint32_t>::max() ||
      scan_input.shape(scan_input.ndim() - 1) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  if (cumsum_i32 &&
      (reverse || scan_input.shape(scan_input.ndim() - 1) > 128)) {
    return false;
  }

  if ((reduce_type == Scan::Prod || reduce_type == Scan::Min ||
       reduce_type == Scan::Max || reduce_type == Scan::LogAddExp) &&
      scan_input.shape(scan_input.ndim() - 1) > 128) {
    return false;
  }

  array inclusive_out(scan_input.shape(), scan_input.dtype(), nullptr, {});
  inclusive_out.set_data(allocator::malloc(inclusive_out.nbytes()));
  if (inclusive_out.size() == 0) {
    copy_gpu(inclusive_out, out, CopyType::GeneralGeneral, s);
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);

    switch (reduce_type) {
      case Scan::Sum:
        vulkan::dispatch_cumsum_op(
            scan_input,
            inclusive_out,
            cumsum_i32 ? vulkan::StaticShaderId::cumsum_i32
                       : vulkan::StaticShaderId::cumsum_f32,
            command_buffer,
            s,
            reverse,
            inclusive);
        break;
      case Scan::Prod:
        vulkan::dispatch_scan_op(
            scan_input,
            inclusive_out,
            cumprod_i32 ? vulkan::StaticShaderId::cumprod_i32
                        : vulkan::StaticShaderId::cumprod_f32,
            command_buffer,
            s,
            reverse,
            inclusive);
        break;
      case Scan::Min:
        vulkan::dispatch_scan_op(
            scan_input,
            inclusive_out,
            vulkan::StaticShaderId::cummin_f32,
            command_buffer,
            s,
            reverse,
            inclusive);
        break;
      case Scan::Max:
        vulkan::dispatch_scan_op(
            scan_input,
            inclusive_out,
            vulkan::StaticShaderId::cummax_f32,
            command_buffer,
            s,
            reverse,
            inclusive);
        break;
      case Scan::LogAddExp:
        vulkan::dispatch_scan_op(
            scan_input,
            inclusive_out,
            vulkan::StaticShaderId::cumlogaddexp_f32,
            command_buffer,
            s,
            reverse,
            inclusive);
        break;
      default:
        throw std::runtime_error("Unsupported Vulkan scan reduce type.");
    }

    array scan_result = inclusive_out;

    vulkan::end_command_recording(s.index);

    array restored = scan_result;
    if (normalized_axis != in.ndim() - 1) {
      restored = swapaxes_in_eval(restored, normalized_axis, in.ndim() - 1);
    }
    if (!is_supported_unary_layout(restored)) {
      restored = contiguous_copy_gpu(restored, s);
    }

    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(restored, out, CopyType::GeneralGeneral, s);
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "scan_dispatch_failed reduce_type="
          << static_cast<int>(reduce_type) << " reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis, reverse, inclusive] = state();
  if (!try_eval_scan_vulkan(
          inputs, out, reduce_type, axis, reverse, inclusive, stream())) {
    throw std::runtime_error(
        "Scan operation failed on Vulkan (unsupported dtype or layout).");
  }
}

} // namespace mlx::core
