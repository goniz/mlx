// Copyright © 2024 Apple Inc.

#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"
#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/dtype_utils.h"

#include <cstring>
#include <optional>
#include <sstream>
#include <vector>

namespace mlx::core {

namespace {

std::string build_sum_rows_i64_shader() {
  std::ostringstream os;
  os << "#version 450\n";
  os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  os << "#extension GL_KHR_memory_scope_semantics : enable\n";
  os << "#pragma use_vulkan_memory_model\n\n";
  os << R"(
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform PushConstants {
  uint n_cols;
  uint n_rows;
} pc;
layout(set = 0, binding = 0) readonly buffer Input { int64_t data[]; } in_buf;
layout(set = 0, binding = 1) buffer Output { int64_t data[]; } out_buf;
shared int64_t vals[256];

void main() {
  uint tid = gl_LocalInvocationID.x;
  uint row = gl_WorkGroupID.x;
  if (row >= pc.n_rows) return;

  int64_t acc = int64_t(0);
  uint base = row * pc.n_cols;
  for (uint col = tid; col < pc.n_cols; col += 256u) {
    acc += in_buf.data[base + col];
  }
  vals[tid] = acc;
  barrier();
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      vals[tid] += vals[tid + s];
    }
    barrier();
  }
  if (tid == 0u) {
    out_buf.data[row] = vals[0];
  }
}
)";
  return os.str();
}

std::string build_simple_reduce_rows_shader(
    Dtype dtype,
    Reduce::ReduceType reduce_type) {
  std::ostringstream os;
  os << "#version 450\n";
  if (dtype == int64 || dtype == uint64) {
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  os << "#extension GL_KHR_memory_scope_semantics : enable\n";
  os << "#pragma use_vulkan_memory_model\n\n";
  os << R"(
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(push_constant) uniform PushConstants {
  uint n_cols;
  uint n_rows;
} pc;
)";

  std::string glsl_type;
  std::string init_value;
  if (dtype == int64) {
    glsl_type = "int64_t";
    if (reduce_type == Reduce::Prod) {
      init_value = "int64_t(1)";
    } else if (reduce_type == Reduce::Min) {
      init_value = "int64_t(9223372036854775807ul)";
    } else if (reduce_type == Reduce::Max) {
      init_value = "-int64_t(9223372036854775807ul) - int64_t(1)";
    } else {
      init_value = "int64_t(0)";
    }
  } else if (dtype == uint64) {
    glsl_type = "uint64_t";
    if (reduce_type == Reduce::Prod) {
      init_value = "uint64_t(1)";
    } else if (reduce_type == Reduce::Min) {
      init_value = "uint64_t(0xfffffffffffffffful)";
    } else {
      init_value = "uint64_t(0)";
    }
  } else {
    glsl_type = "vec2";
    if (reduce_type == Reduce::Prod) {
      init_value = "vec2(1.0, 0.0)";
    } else if (reduce_type == Reduce::Min) {
      init_value = "vec2(3.402823466e+38, 3.402823466e+38)";
    } else if (reduce_type == Reduce::Max) {
      init_value = "vec2(-3.402823466e+38, -3.402823466e+38)";
    } else {
      init_value = "vec2(0.0, 0.0)";
    }
  }

  os << "layout(set = 0, binding = 0) readonly buffer Input { " << glsl_type
     << " data[]; } in_buf;\n";
  os << "layout(set = 0, binding = 1) buffer Output { " << glsl_type
     << " data[]; } out_buf;\n";
  os << "shared " << glsl_type << " vals[256];\n\n";
  if (dtype == complex64) {
    os << R"(
vec2 cmul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
bool cless(vec2 a, vec2 b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}
)";
  }
  os << "void main() {\n";
  os << "  uint tid = gl_LocalInvocationID.x;\n";
  os << "  uint row = gl_WorkGroupID.x;\n";
  os << "  if (row >= pc.n_rows) return;\n";
  os << "  " << glsl_type << " acc = " << init_value << ";\n";
  os << "  uint base = row * pc.n_cols;\n";
  os << "  for (uint col = tid; col < pc.n_cols; col += 256u) {\n";
  os << "    " << glsl_type << " value = in_buf.data[base + col];\n";
  if (reduce_type == Reduce::Prod) {
    if (dtype == complex64) {
      os << "    acc = cmul(acc, value);\n";
    } else {
      os << "    acc *= value;\n";
    }
  } else if (reduce_type == Reduce::Min) {
    if (dtype == complex64) {
      os << "    acc = cless(value, acc) ? value : acc;\n";
    } else {
      os << "    acc = value < acc ? value : acc;\n";
    }
  } else if (reduce_type == Reduce::Max) {
    if (dtype == complex64) {
      os << "    acc = cless(acc, value) ? value : acc;\n";
    } else {
      os << "    acc = value > acc ? value : acc;\n";
    }
  } else {
    os << "    acc += value;\n";
  }
  os << "  }\n";
  os << "  vals[tid] = acc;\n";
  os << "  barrier();\n";
  os << "  for (uint s = 128u; s > 0u; s >>= 1u) {\n";
  os << "    if (tid < s) {\n";
  if (reduce_type == Reduce::Prod) {
    if (dtype == complex64) {
      os << "      vals[tid] = cmul(vals[tid], vals[tid + s]);\n";
    } else {
      os << "      vals[tid] *= vals[tid + s];\n";
    }
  } else if (reduce_type == Reduce::Min) {
    if (dtype == complex64) {
      os << "      vals[tid] = cless(vals[tid + s], vals[tid]) ? vals[tid + s] : vals[tid];\n";
    } else {
      os << "      vals[tid] = vals[tid + s] < vals[tid] ? vals[tid + s] : vals[tid];\n";
    }
  } else if (reduce_type == Reduce::Max) {
    if (dtype == complex64) {
      os << "      vals[tid] = cless(vals[tid], vals[tid + s]) ? vals[tid + s] : vals[tid];\n";
    } else {
      os << "      vals[tid] = vals[tid + s] > vals[tid] ? vals[tid + s] : vals[tid];\n";
    }
  } else {
    os << "      vals[tid] += vals[tid + s];\n";
  }
  os << "    }\n";
  os << "    barrier();\n";
  os << "  }\n";
  os << "  if (tid == 0u) {\n";
  os << "    out_buf.data[row] = vals[0];\n";
  os << "  }\n";
  os << "}\n";
  return os.str();
}

bool try_dispatch_sum_rows_i64(const array& in, array& out, Stream s) {
  if (in.dtype() != int64 || out.dtype() != int64 || in.ndim() == 0 ||
      in.shape(-1) == 0 || in.size() != out.size() * in.shape(-1)) {
    return false;
  }
  array in_kernel = in;
  if (!in_kernel.flags().row_contiguous || in_kernel.offset() != 0 ||
      in_kernel.strides(-1) != 1 || !is_supported_unary_layout(in_kernel)) {
    in_kernel = contiguous_copy_gpu(in_kernel, s);
  }
  const uint32_t n_cols =
      checked_u32_size(in_kernel.shape(-1), "sum_rows_i64 n_cols");
  const uint32_t n_rows = checked_u32_size(out.size(), "sum_rows_i64 n_rows");
  if (n_rows == 0) {
    return true;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  vulkan::DynamicArrayRef arrays[] = {{&in_kernel, 0}, {&out, 1}};
  const std::array<uint32_t, 2> pc = {n_cols, n_rows};
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      "dynamic_sum_rows_i64",
      build_sum_rows_i64_shader(),
      2,
      arrays,
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      pc.data());
  vkCmdDispatch(dispatch.command_buffer, n_rows, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

bool try_dispatch_simple_reduce_rows(
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    Stream s) {
  if (in.dtype() != out.dtype() || in.ndim() == 0 || in.shape(-1) == 0 ||
      in.size() != out.size() * in.shape(-1) ||
      (in.dtype() != int64 && in.dtype() != uint64 &&
       in.dtype() != complex64)) {
    return false;
  }
  if (reduce_type != Reduce::Sum && reduce_type != Reduce::Prod &&
      reduce_type != Reduce::Min && reduce_type != Reduce::Max) {
    return false;
  }
  array in_kernel = in;
  if (!in_kernel.flags().row_contiguous || in_kernel.offset() != 0 ||
      in_kernel.strides(-1) != 1 || !is_supported_unary_layout(in_kernel)) {
    in_kernel = contiguous_copy_gpu(in_kernel, s);
  }
  const uint32_t n_cols =
      checked_u32_size(in_kernel.shape(-1), "simple_reduce_rows n_cols");
  const uint32_t n_rows =
      checked_u32_size(out.size(), "simple_reduce_rows n_rows");
  if (n_rows == 0) {
    return true;
  }

  out.set_data(allocator::malloc(out.nbytes()));
  vulkan::DynamicArrayRef arrays[] = {{&in_kernel, 0}, {&out, 1}};
  const std::array<uint32_t, 2> pc = {n_cols, n_rows};
  std::string shader_name = "dynamic_reduce_rows_";
  shader_name += dtype_to_string(in.dtype());
  shader_name += "_";
  shader_name += reduce_type == Reduce::Sum ? "sum"
      : reduce_type == Reduce::Prod         ? "prod"
      : reduce_type == Reduce::Min          ? "min"
                                            : "max";
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name,
      build_simple_reduce_rows_shader(in.dtype(), reduce_type),
      2,
      arrays,
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      s);
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(pc.size() * sizeof(uint32_t)),
      pc.data());
  vkCmdDispatch(dispatch.command_buffer, n_rows, 1, 1);
  vulkan::end_command_recording(s.index);
  return true;
}

array stage_zero_offset_row_contiguous(const array& in, Stream s) {
  array staged(in.shape(), in.dtype(), nullptr, {});
  staged.set_data(allocator::malloc(staged.nbytes()));
  copy_gpu_inplace(in, staged, CopyType::Vector, s);
  return staged;
}

std::optional<vulkan::StaticShaderId> integer_reduce_rows_shader(
    Reduce::ReduceType reduce_type,
    Dtype in_dtype,
    Dtype out_dtype) {
  if (reduce_type == Reduce::Sum) {
    if (in_dtype == uint8 && out_dtype == uint32) {
      return vulkan::StaticShaderId::sum_rows_u8_u32;
    }
    if (in_dtype == uint16 && out_dtype == uint32) {
      return vulkan::StaticShaderId::sum_rows_u16_u32;
    }
    if (in_dtype == uint32 && out_dtype == uint32) {
      return vulkan::StaticShaderId::sum_rows_u32;
    }
    if (in_dtype == int32 && out_dtype == int32) {
      return vulkan::StaticShaderId::sum_rows_i32;
    }
  }
  if (reduce_type == Reduce::Prod) {
    if (in_dtype == uint8 && out_dtype == uint32) {
      return vulkan::StaticShaderId::prod_rows_u8_u32;
    }
    if (in_dtype == uint16 && out_dtype == uint32) {
      return vulkan::StaticShaderId::prod_rows_u16_u32;
    }
    if (in_dtype == uint32 && out_dtype == uint32) {
      return vulkan::StaticShaderId::prod_rows_u32;
    }
    if (in_dtype == int32 && out_dtype == int32) {
      return vulkan::StaticShaderId::prod_rows_i32;
    }
  }
  if (reduce_type == Reduce::Max) {
    if (in_dtype == uint8 && out_dtype == uint8) {
      return vulkan::StaticShaderId::max_rows_u8;
    }
    if (in_dtype == uint16 && out_dtype == uint16) {
      return vulkan::StaticShaderId::max_rows_u16;
    }
    if (in_dtype == uint32 && out_dtype == uint32) {
      return vulkan::StaticShaderId::max_rows_u32;
    }
    if (in_dtype == int8 && out_dtype == int8) {
      return vulkan::StaticShaderId::max_rows_i8;
    }
    if (in_dtype == int16 && out_dtype == int16) {
      return vulkan::StaticShaderId::max_rows_i16;
    }
    if (in_dtype == int32 && out_dtype == int32) {
      return vulkan::StaticShaderId::max_rows_i32;
    }
  }
  if (reduce_type == Reduce::Min) {
    if (in_dtype == uint8 && out_dtype == uint8) {
      return vulkan::StaticShaderId::min_rows_u8;
    }
    if (in_dtype == uint16 && out_dtype == uint16) {
      return vulkan::StaticShaderId::min_rows_u16;
    }
    if (in_dtype == uint32 && out_dtype == uint32) {
      return vulkan::StaticShaderId::min_rows_u32;
    }
    if (in_dtype == int8 && out_dtype == int8) {
      return vulkan::StaticShaderId::min_rows_i8;
    }
    if (in_dtype == int16 && out_dtype == int16) {
      return vulkan::StaticShaderId::min_rows_i16;
    }
    if (in_dtype == int32 && out_dtype == int32) {
      return vulkan::StaticShaderId::min_rows_i32;
    }
  }
  return std::nullopt;
}

bool try_eval_reduce_sum_rows_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    Stream s) {
  if (inputs.size() != 1 || axes.empty()) {
    return false;
  }

  array in = inputs[0];
  const bool sum_reduce = reduce_type == Reduce::Sum;
  const bool prod_reduce = reduce_type == Reduce::Prod;
  const bool bool_prod = reduce_type == Reduce::Prod && in.dtype() == bool_ &&
      out.dtype() == int32;
  const bool max_reduce = reduce_type == Reduce::Max;
  const bool min_reduce = reduce_type == Reduce::Min;
  const bool logic_reduce =
      reduce_type == Reduce::And || reduce_type == Reduce::Or;
  if (!sum_reduce && !prod_reduce && !max_reduce && !min_reduce &&
      !logic_reduce) {
    return false;
  }

  const bool f32_io = in.dtype() == float32 && out.dtype() == float32;
  const bool f16_io = in.dtype() == float16 && out.dtype() == float16;
  const bool bf16_io = in.dtype() == bfloat16 && out.dtype() == bfloat16;
  const bool bool_sum =
      sum_reduce && in.dtype() == bool_ && out.dtype() == int32;
  const bool bool_io = in.dtype() == bool_ && out.dtype() == bool_;
  const bool bool_min_max = (max_reduce || min_reduce) && bool_io;
  const bool integer_sum_prod = (sum_reduce || prod_reduce) &&
      ((in.dtype() == uint8 && out.dtype() == uint32) ||
       (in.dtype() == uint16 && out.dtype() == uint32) ||
       (in.dtype() == uint32 && out.dtype() == uint32) ||
       (in.dtype() == uint64 && out.dtype() == uint64) ||
       (in.dtype() == int8 && out.dtype() == int32) ||
       (in.dtype() == int16 && out.dtype() == int32) ||
       (in.dtype() == int32 && out.dtype() == int32) ||
       (in.dtype() == int64 && out.dtype() == int64) ||
       (in.dtype() == complex64 && out.dtype() == complex64));
  const bool integer_min_max = (max_reduce || min_reduce) &&
      ((in.dtype() == uint32 && out.dtype() == uint32) ||
       (in.dtype() == uint64 && out.dtype() == uint64) ||
       (in.dtype() == int32 && out.dtype() == int32) ||
       (in.dtype() == int64 && out.dtype() == int64) ||
       (in.dtype() == int8 && out.dtype() == int8) ||
       (in.dtype() == int16 && out.dtype() == int16) ||
       (in.dtype() == uint8 && out.dtype() == uint8) ||
       (in.dtype() == uint16 && out.dtype() == uint16) ||
       (in.dtype() == complex64 && out.dtype() == complex64));
  if ((sum_reduce || prod_reduce || max_reduce || min_reduce) && !f32_io &&
      !f16_io && !bf16_io && !bool_sum && !bool_prod && !bool_min_max &&
      !integer_sum_prod && !integer_min_max) {
    return false;
  }
  if (logic_reduce && !bool_prod && out.dtype() != bool_) {
    return false;
  }

  const bool use_f32_staging_io =
      (sum_reduce || prod_reduce || max_reduce || min_reduce) &&
      (f16_io || bf16_io || bool_sum || bool_prod);
  const bool use_i32_staging_io = (sum_reduce || prod_reduce) &&
      ((in.dtype() == int8 || in.dtype() == int16) && out.dtype() == int32);
  const bool use_u8_staging_io = logic_reduce || bool_min_max;
  const bool use_integer_staging_io = use_i32_staging_io;
  if (use_f32_staging_io) {
    if (in.offset() > 0xFFFF && in.flags().row_contiguous) {
      in = stage_zero_offset_row_contiguous(in, s);
    }
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }
  if (use_i32_staging_io) {
    array in_i32(in.shape(), int32, nullptr, {});
    copy_gpu(in, in_i32, CopyType::General, s);
    in = in_i32;
  }
  if (use_u8_staging_io) {
    if (in.dtype() != bool_) {
      array in_bool(in.shape(), bool_, nullptr, {});
      copy_gpu(in, in_bool, CopyType::General, s);
      in = in_bool;
    }
    array in_u8(in.shape(), uint8, nullptr, {});
    in_u8.copy_shared_buffer(
        in, in.strides(), in.flags(), in.data_size(), in.offset());
    in = in_u8;
  }

  array reduce_out_target = use_f32_staging_io
      ? array(out.shape(), float32, nullptr, {})
      : use_integer_staging_io ? array(out.shape(), in.dtype(), nullptr, {})
      : use_u8_staging_io
      ? array(out.shape(), bool_prod ? int32 : bool_, nullptr, {})
      : out;
  if (use_u8_staging_io) {
    reduce_out_target.set_data(allocator::malloc(reduce_out_target.nbytes()));
  }

  if ((logic_reduce || bool_min_max) && in.ndim() == 0) {
    copy_gpu(in, out, CopyType::General, s);
    return true;
  }

  if (in.size() == 0 && (sum_reduce || prod_reduce)) {
    array fill_value(sum_reduce ? 0.0f : 1.0f, reduce_out_target.dtype());
    fill_gpu(fill_value, reduce_out_target, s);
    if (use_f32_staging_io) {
      copy_gpu(reduce_out_target, out, CopyType::General, s);
    } else if (use_integer_staging_io) {
      copy_gpu(reduce_out_target, out, CopyType::General, s);
    }
    return true;
  }

  if (in.ndim() == 0) {
    return false;
  }

  std::vector<int> normalized_axes;
  if (!normalize_unique_axes(axes, in.ndim(), normalized_axes)) {
    return false;
  }
  const bool full_reduce = normalized_axes.size() == in.ndim();

  if (in.ndim() > 4 && full_reduce) {
    Shape flattened = {
        static_cast<ShapeElem>(in.shape(0) * in.shape(1)),
        in.ndim() > 2 ? in.shape(2) : 1,
        in.ndim() > 3 ? in.shape(3) : 1,
        in.ndim() > 4 ? in.shape(4) : 1,
    };
    for (int i = 5; i < in.ndim(); ++i) {
      flattened[0] *= in.shape(i);
    }
    in = reshape_in_eval(in, flattened, s);
    normalized_axes = {0, 1, 2, 3};
  }

  const bool staged_full_reduce_output =
      reduce_out_target.ndim() > 4 && full_reduce;
  if (staged_full_reduce_output) {
    reduce_out_target =
        array(Shape{1, 1, 1, 1}, reduce_out_target.dtype(), nullptr, {});
    if (use_u8_staging_io) {
      reduce_out_target.set_data(allocator::malloc(reduce_out_target.nbytes()));
    }
  }

  bool out_is_keepdims = false;
  bool out_is_squeezed = false;
  bool reshape_final_to_output = false;

  if (in.ndim() > 4 && !full_reduce && normalized_axes.size() == 2 &&
      normalized_axes[0] == in.ndim() - 3 &&
      normalized_axes[1] == in.ndim() - 1) {
    const bool high_rank_keepdims =
        has_keepdims_axes_shape(in, reduce_out_target, normalized_axes);
    const bool high_rank_squeezed =
        has_squeezed_axes_shape(in, reduce_out_target, normalized_axes);
    if (high_rank_keepdims || high_rank_squeezed) {
      Shape reshaped = {1, in.shape(-3), in.shape(-2), in.shape(-1)};
      for (int i = 0; i < in.ndim() - 3; ++i) {
        reshaped[0] *= in.shape(i);
      }
      in = reshape_in_eval(in, reshaped, s);
      normalized_axes = {1, 3};
      out_is_keepdims = true;
      out_is_squeezed = high_rank_squeezed;
      reshape_final_to_output = true;
    }
  }

  if (in.ndim() > 4 && !full_reduce && !reshape_final_to_output &&
      normalized_axes.size() == 1) {
    const bool high_rank_keepdims =
        has_keepdims_axes_shape(in, reduce_out_target, normalized_axes);
    const bool high_rank_squeezed =
        has_squeezed_axes_shape(in, reduce_out_target, normalized_axes);
    if (high_rank_keepdims || high_rank_squeezed) {
      const int axis = normalized_axes[0];
      ShapeElem before = 1;
      for (int i = 0; i < axis; ++i) {
        before *= in.shape(i);
      }
      ShapeElem after = 1;
      for (int i = axis + 1; i < in.ndim(); ++i) {
        after *= in.shape(i);
      }
      in = reshape_in_eval(in, {before, in.shape(axis), after}, s);
      normalized_axes = {1};
      out_is_keepdims = true;
      out_is_squeezed = high_rank_squeezed;
      reshape_final_to_output = true;
    }
  }

  if (f32_io && sum_reduce && in.ndim() > 4 && !full_reduce &&
      !reshape_final_to_output && normalized_axes.size() > 2) {
    const int block_row_axis = in.ndim() - 3;
    const int block_col_axis = in.ndim() - 1;
    bool has_block_row_axis = false;
    bool has_block_col_axis = false;
    std::vector<int> remaining_axes;
    remaining_axes.reserve(normalized_axes.size() - 2);
    for (int axis : normalized_axes) {
      if (axis == block_row_axis) {
        has_block_row_axis = true;
      } else if (axis == block_col_axis) {
        has_block_col_axis = true;
      } else {
        remaining_axes.push_back(axis);
      }
    }
    if (has_block_row_axis && has_block_col_axis && !remaining_axes.empty()) {
      const std::vector<int> block_axes = {block_row_axis, block_col_axis};
      array block_reduced(
          keepdims_shape_for_axes(in, block_axes), in.dtype(), nullptr, {});
      if (!try_eval_reduce_sum_rows_vulkan(
              {in}, block_reduced, reduce_type, block_axes, s)) {
        return false;
      }
      return try_eval_reduce_sum_rows_vulkan(
          {block_reduced}, out, reduce_type, remaining_axes, s);
    }
  }

  // For remaining >4D multi-axis reductions, reuse the supported keepdims
  // single-axis path one axis at a time.
  if (in.ndim() > 4 && !full_reduce && normalized_axes.size() > 1 &&
      has_keepdims_axes_shape(in, out, normalized_axes)) {
    array reduced = in;
    for (int axis : normalized_axes) {
      array axis_out(
          keepdims_shape_for_axis(reduced, axis), out.dtype(), nullptr, {});
      if (!try_eval_reduce_sum_rows_vulkan(
              {reduced}, axis_out, reduce_type, {axis}, s)) {
        return false;
      }
      reduced = std::move(axis_out);
    }

    if (reduced.ndim() == 0 && out.ndim() == 0) {
      copy_gpu(reduced, out, CopyType::Scalar, s);
    } else {
      copy_gpu(reduced, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  if (in.ndim() > 4 ||
      (reduce_out_target.ndim() > 4 && !reshape_final_to_output)) {
    return false;
  }

  if (!reshape_final_to_output && !out_is_squeezed) {
    out_is_keepdims = staged_full_reduce_output ||
        has_keepdims_axes_shape(in, reduce_out_target, normalized_axes);
    out_is_squeezed =
        has_squeezed_axes_shape(in, reduce_out_target, normalized_axes);
  }
  if (!out_is_keepdims && !out_is_squeezed) {
    return false;
  }

  if (logic_reduce && in.size() == 0) {
    array fill_value(reduce_type == Reduce::And ? true : false, bool_);
    fill_gpu(fill_value, out, s);
    return true;
  }

  array reduced = in;
  for (int axis : normalized_axes) {
    array in_kernel = reduced;
    if (axis != reduced.ndim() - 1) {
      in_kernel = swapaxes_in_eval(reduced, axis, reduced.ndim() - 1);
    }

    if (in_kernel.ndim() == 0 || in_kernel.strides(-1) != 1 ||
        !is_supported_unary_layout(in_kernel)) {
      in_kernel = contiguous_copy_gpu(in_kernel, s);
    }

    const auto integer_shader = integer_reduce_rows_shader(
        reduce_type, in_kernel.dtype(), reduce_out_target.dtype());
    const auto kernel_out_dtype =
        integer_shader ? out.dtype() : in_kernel.dtype();
    array kernel_out(
        keepdims_shape_for_axis(in_kernel, in_kernel.ndim() - 1),
        kernel_out_dtype,
        nullptr,
        {});

    const bool staged_output = !kernel_out.flags().row_contiguous ||
        !is_supported_unary_layout(kernel_out);
    array out_work = staged_output
        ? array(kernel_out.shape(), kernel_out.dtype(), nullptr, {})
        : kernel_out;

    out_work.set_data(allocator::malloc(out_work.nbytes()));
    if (out_work.size() == 0) {
      if (staged_output) {
        copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
      }
    } else {
      try {
        if (sum_reduce && in_kernel.dtype() == int64 &&
            out_work.dtype() == int64) {
          if (!try_dispatch_sum_rows_i64(in_kernel, out_work, s)) {
            return false;
          }
          if (staged_output) {
            copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
          }
          reduced = (axis == reduced.ndim() - 1)
              ? kernel_out
              : swapaxes_in_eval(kernel_out, axis, reduced.ndim() - 1);
          continue;
        }
        if ((in_kernel.dtype() == int64 || in_kernel.dtype() == uint64 ||
             in_kernel.dtype() == complex64) &&
            out_work.dtype() == in_kernel.dtype()) {
          if (!try_dispatch_simple_reduce_rows(
                  in_kernel, out_work, reduce_type, s)) {
            return false;
          }
          if (staged_output) {
            copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
          }
          reduced = (axis == reduced.ndim() - 1)
              ? kernel_out
              : swapaxes_in_eval(kernel_out, axis, reduced.ndim() - 1);
          continue;
        }
        auto command_buffer = vulkan::begin_command_recording(s.index);
        const auto shader_id = integer_shader ? *integer_shader
            : sum_reduce   ? vulkan::StaticShaderId::sum_rows_f32
            : prod_reduce  ? vulkan::StaticShaderId::prod_rows_f32
            : bool_min_max ? (max_reduce ? vulkan::StaticShaderId::any_rows_u8
                                         : vulkan::StaticShaderId::all_rows_u8)
            : max_reduce   ? vulkan::StaticShaderId::max_rows_f32
            : min_reduce   ? vulkan::StaticShaderId::min_rows_f32
            : reduce_type == Reduce::And ? vulkan::StaticShaderId::all_rows_u8
                                         : vulkan::StaticShaderId::any_rows_u8;
        vulkan::dispatch_sum_rows_op(
            in_kernel, out_work, shader_id, command_buffer, s, 1.0f);
        vulkan::end_command_recording(s.index);
      } catch (const std::runtime_error&) {
        return false;
      }
      if (staged_output) {
        copy_gpu(out_work, kernel_out, CopyType::GeneralGeneral, s);
      }
    }

    reduced = (axis == reduced.ndim() - 1)
        ? kernel_out
        : swapaxes_in_eval(kernel_out, axis, reduced.ndim() - 1);
  }

  array final_result =
      (reshape_final_to_output || out_is_squeezed || staged_full_reduce_output)
      ? reshape_in_eval(reduced, out.shape(), s)
      : reduced;

  if (use_f32_staging_io) {
    if (reshape_final_to_output) {
      copy_gpu(final_result, out, CopyType::General, s);
    } else {
      copy_gpu(final_result, reduce_out_target, CopyType::General, s);
      copy_gpu(reduce_out_target, out, CopyType::General, s);
    }
  } else if (use_integer_staging_io) {
    copy_gpu(final_result, out, CopyType::General, s);
  } else if (use_u8_staging_io) {
    if (reshape_final_to_output) {
      copy_gpu(final_result, out, CopyType::General, s);
    } else {
      out.copy_shared_buffer(
          final_result,
          final_result.strides(),
          final_result.flags(),
          final_result.data_size(),
          final_result.offset());
    }
  } else if (final_result.ndim() == 0 && out.ndim() == 0) {
    copy_gpu(final_result, out, CopyType::Scalar, s);
  } else {
    copy_gpu(final_result, out, CopyType::GeneralGeneral, s);
  }

  return true;
}

bool try_eval_arg_reduce_vulkan(
    const std::vector<array>& inputs,
    array& out,
    ArgReduce::ReduceType reduce_type,
    int axis,
    Stream s) {
  if (inputs.size() != 1 ||
      (reduce_type != ArgReduce::ArgMax && reduce_type != ArgReduce::ArgMin)) {
    return false;
  }

  array in = inputs[0];
  const bool f32_input = in.dtype() == float32;
  if (in.ndim() == 0 || in.dtype() == float64 || out.dtype() != uint32) {
    return false;
  }

  if (!f32_input) {
    array in_f32(in.shape(), float32, nullptr, {});
    copy_gpu(in, in_f32, CopyType::General, s);
    in = in_f32;
  }

  axis = normalize_axis(axis, in.ndim());

  const bool out_is_keepdims = has_keepdims_axis_shape(in, out, axis);
  const bool out_is_squeezed = has_squeezed_axis_shape(in, out, axis);
  if (!out_is_keepdims && !out_is_squeezed) {
    return false;
  }

  array in_kernel = in;
  if (axis != in.ndim() - 1) {
    in_kernel = swapaxes_in_eval(in, axis, in.ndim() - 1);
  }

  if (in_kernel.size() > std::numeric_limits<uint32_t>::max() ||
      out.size() > std::numeric_limits<uint32_t>::max() ||
      in_kernel.shape(in_kernel.ndim() - 1) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  if (in_kernel.ndim() == 0 || in_kernel.strides(-1) != 1 ||
      in_kernel.offset() != 0 || !is_supported_unary_layout(in_kernel)) {
    in_kernel = contiguous_copy_gpu(in_kernel, s);
  }

  const bool direct_to_out = (axis == in.ndim() - 1) && out_is_keepdims &&
      out.flags().row_contiguous && out.offset() == 0;

  array kernel_out(
      keepdims_shape_for_axis(in_kernel, in_kernel.ndim() - 1),
      out.dtype(),
      nullptr,
      {});
  array out_work = kernel_out;
  if (direct_to_out) {
    out_work = out;
  }

  out_work.set_data(allocator::malloc(out_work.nbytes()));
  if (out_work.size() == 0) {
    if (!direct_to_out) {
      if (out_is_squeezed) {
        auto squeezed = reshape_in_eval(kernel_out, out.shape(), s);
        copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
      } else {
        copy_gpu(kernel_out, out, CopyType::GeneralGeneral, s);
      }
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    const auto shader_id = reduce_type == ArgReduce::ArgMin
        ? vulkan::StaticShaderId::argmin_f32
        : vulkan::StaticShaderId::argmax_f32;
    vulkan::dispatch_argmax_op(
        in_kernel, out_work, shader_id, command_buffer, s);
    vulkan::end_command_recording(s.index);

    if (direct_to_out) {
      return true;
    }

    array restored_keepdims = kernel_out;
    if (axis != in.ndim() - 1) {
      restored_keepdims = swapaxes_in_eval(kernel_out, axis, in.ndim() - 1);
    }

    if (out_is_squeezed) {
      auto squeezed = reshape_in_eval(restored_keepdims, out.shape(), s);
      copy_gpu(squeezed, out, CopyType::GeneralGeneral, s);
    } else {
      copy_gpu(restored_keepdims, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

} // namespace

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  if (!try_eval_arg_reduce_vulkan(inputs, out, reduce_type, axis, stream())) {
    throw std::runtime_error(
        "ArgReduce has no Vulkan implementation for this input.");
  }
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axes] = state();
  if (!try_eval_reduce_sum_rows_vulkan(
          inputs, out, reduce_type, axes, stream())) {
    if (trace_fallback_enabled() && !inputs.empty()) {
      std::ostringstream oss;
      oss << "reduce_vulkan_unsupported in_shape=" << inputs[0].shape()
          << " in_dtype=" << inputs[0].dtype() << " out_shape=" << out.shape()
          << " out_dtype=" << out.dtype() << " axes=";
      for (auto axis : axes) {
        oss << axis << ",";
      }
      trace_fallback(oss.str());
    }
    throw std::runtime_error(
        "Reduce has no Vulkan implementation for this input.");
  }
}

} // namespace mlx::core
