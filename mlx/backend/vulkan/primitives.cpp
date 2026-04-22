// Copyright © 2024 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/primitives_utils.h"
#include "mlx/backend/vulkan/shader_compiler.h"

namespace mlx::core {

#define NYI_OP(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NYI_OP_STATE(func)                                            \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NYI_OP_MULTI(func)                                              \
  void func::eval_gpu(                                                  \
      const std::vector<array>& inputs, std::vector<array>& outputs) {  \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NYI_OP_MULTI_STATE(func)                                       \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NO_GPU_MULTI_STATE(func)                                       \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no Vulkan implementation.");  \
  }

#define NO_GPU_STATE(func)                                            \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no Vulkan implementation."); \
  }

namespace {

array collapse_power_leading_dims(const array& arr, Stream s) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  return flatten_in_eval(arr, 0, arr.ndim() - 4, s);
}

bool ensure_vulkan_buffer_power(array& arr, Stream s) {
  auto data = arr.data_shared_ptr();
  if (data != nullptr && vulkan::is_vulkan_buffer(data->buffer)) {
    return true;
  }
  if (arr.has_primitive()) {
    arr = contiguous_copy_gpu(arr, s);
    data = arr.data_shared_ptr();
    return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
  }
  arr.wait();
  data = arr.data_shared_ptr();
  if (data != nullptr && vulkan::is_vulkan_buffer(data->buffer)) {
    return true;
  }
  arr = contiguous_copy_gpu(arr, s);
  data = arr.data_shared_ptr();
  return data != nullptr && vulkan::is_vulkan_buffer(data->buffer);
}

std::string emit_bf16_power_helpers() {
  return R"(
uint fp32_to_bf16(float f) {
  uint u = floatBitsToUint(f);
  u = (u + (0x7fffu + ((u >> 16) & 1u))) >> 16;
  return u;
}

float bf16_to_fp32(uint u) {
  return uintBitsToFloat(u << 16);
}

)";
}

std::string power_input_expr(
    Dtype dtype,
    const char* buffer_name,
    const char* index_name) {
  if (dtype == bfloat16) {
    return std::string("bf16_to_fp32(uint(") + buffer_name + ".data[" +
        index_name + "]))";
  }
  if (dtype == float16) {
    return std::string("float(") + buffer_name + ".data[" + index_name + "])";
  }
  return std::string(buffer_name) + ".data[" + index_name + "]";
}

std::string power_output_expr(Dtype out_dtype, const std::string& expr) {
  if (out_dtype == bfloat16) {
    return "uint16_t(fp32_to_bf16(" + expr + "))";
  }
  if (out_dtype == float16) {
    return "float16_t(" + expr + ")";
  }
  return expr;
}

std::string build_power_shader(Dtype a_dtype, Dtype b_dtype, Dtype out_dtype) {
  std::ostringstream os;
  const bool uses_bfloat16 =
      a_dtype == bfloat16 || b_dtype == bfloat16 || out_dtype == bfloat16;
  const bool uses_float16 =
      a_dtype == float16 || b_dtype == float16 || out_dtype == float16;
  if (uses_bfloat16 || uses_float16) {
    os << "#version 450\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    os << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n";
    os << "#extension GL_EXT_shader_16bit_storage : require\n\n";
    os << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
  } else {
    os << vulkan::emit_dynamic_shader_preamble(a_dtype, out_dtype, false);
  }
  if (a_dtype == bfloat16 || b_dtype == bfloat16 || out_dtype == bfloat16) {
    os << emit_bf16_power_helpers();
  }
  os << "layout(push_constant) uniform PushConstants { uint a_offset; uint b_offset; uint out_offset; uint total_elements; } pc;\n";
  os << "layout(set = 0, binding = 0) readonly buffer InputA {"
     << vulkan::dtype_to_glsl_storage_type(a_dtype) << " data[];} a_buf;\n";
  os << "layout(set = 0, binding = 1) readonly buffer InputB {"
     << vulkan::dtype_to_glsl_storage_type(b_dtype) << " data[];} b_buf;\n";
  os << "layout(set = 0, binding = 2) buffer Output {"
     << vulkan::dtype_to_glsl_storage_type(out_dtype) << " data[];} out_buf;\n\n";
  os << "void main() {\n";
  os << "  uint linear_idx = gl_GlobalInvocationID.x;\n";
  os << "  if (linear_idx >= pc.total_elements) return;\n";
  os << "  uint idx = linear_idx;\n";
  os << "  uint a_idx = idx + pc.a_offset;\n";
  os << "  uint b_idx = idx + pc.b_offset;\n";
  os << "  float lhs = " << power_input_expr(a_dtype, "a_buf", "a_idx") << ";\n";
  os << "  float rhs = " << power_input_expr(b_dtype, "b_buf", "b_idx") << ";\n";
  os << "  out_buf.data[idx + pc.out_offset] = "
      << power_output_expr(out_dtype, "pow(lhs, rhs)") << ";\n";
  os << "}\n";
  return os.str();
}

bool try_eval_power_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 2) {
    return false;
  }
  array a = inputs[0];
  array b = inputs[1];

  auto is_supported_dtype = [](Dtype dtype) {
    return dtype == float16 || dtype == float32 || dtype == bfloat16;
  };
  if (!is_supported_dtype(a.dtype()) || !is_supported_dtype(b.dtype()) ||
      !is_supported_dtype(out.dtype())) {
    return false;
  }

  auto materialize_broadcast_input = [&](array& in) {
    if (in.shape() == out.shape()) {
      return true;
    }
    if (broadcast_shapes(in.shape(), out.shape()) != out.shape()) {
      return false;
    }
    if (!ensure_vulkan_buffer_power(in, s)) {
      return false;
    }
    array view(out.shape(), in.dtype(), nullptr, {});
    broadcast(in, view);
    in = view;
    return true;
  };

  if (!materialize_broadcast_input(a) || !materialize_broadcast_input(b)) {
    return false;
  }

  if (!is_supported_elementwise_layout(a)) {
    a = contiguous_copy_gpu(a, s);
  }
  if (!is_supported_elementwise_layout(b)) {
    b = contiguous_copy_gpu(b, s);
  }

  a = collapse_power_leading_dims(a, s);
  b = collapse_power_leading_dims(b, s);

  const bool staged_output = !is_supported_elementwise_layout(out);
  array out_work = staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_work.set_data(allocator::malloc(out_work.nbytes()));
  out_work = collapse_power_leading_dims(out_work, s);

  if (!is_supported_elementwise_layout(a) || !is_supported_elementwise_layout(b) ||
      !is_supported_elementwise_layout(out_work)) {
    return false;
  }
  if (!ensure_vulkan_buffer_power(a, s) || !ensure_vulkan_buffer_power(b, s) ||
      !ensure_vulkan_buffer_power(out_work, s)) {
    return false;
  }
  if (out_work.size() == 0) {
    if (staged_output) {
      copy_gpu(out_work, out, CopyType::General, s);
    }
    return true;
  }

  const auto a_offset = static_cast<uint64_t>(a.offset() / size_of(a.dtype()));
  const auto b_offset = static_cast<uint64_t>(b.offset() / size_of(b.dtype()));
  const auto out_offset = static_cast<uint64_t>(out_work.offset() / size_of(out_work.dtype()));
  const auto total = static_cast<uint64_t>(out_work.data_size());
  if (a_offset > std::numeric_limits<uint32_t>::max() ||
      b_offset > std::numeric_limits<uint32_t>::max() ||
      out_offset > std::numeric_limits<uint32_t>::max() ||
      total > std::numeric_limits<uint32_t>::max()) {
    return false;
  }

  const std::string shader_name = "dynamic_power_" +
      std::to_string(static_cast<int>(a.dtype().val())) + "_" +
      std::to_string(static_cast<int>(b.dtype().val())) + "_" +
      std::to_string(static_cast<int>(out_work.dtype().val()));
  const std::string glsl_source =
      build_power_shader(a.dtype(), b.dtype(), out_work.dtype());
  vulkan::DynamicArrayRef arrays[] = {{&a, 0}, {&b, 1}, {&out_work, 2}};
  constexpr uint32_t kPushConstantSize = sizeof(uint32_t) * 4;
  auto dispatch = vulkan::dispatch_dynamic_compute_begin(
      shader_name, glsl_source, 3, arrays, kPushConstantSize, s);

  struct PushConstants {
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t out_offset;
    uint32_t total_elements;
  } pc{
      static_cast<uint32_t>(a_offset),
      static_cast<uint32_t>(b_offset),
      static_cast<uint32_t>(out_offset),
      static_cast<uint32_t>(total),
  };
  vkCmdPushConstants(
      dispatch.command_buffer,
      dispatch.pipeline->layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      kPushConstantSize,
      &pc);
  const uint32_t workgroups =
      std::max<uint32_t>((static_cast<uint32_t>(total) + 255u) / 256u, 1u);
  vkCmdDispatch(dispatch.command_buffer, workgroups, 1, 1);
  vulkan::end_command_recording(s.index);

  if (staged_output) {
    copy_gpu(out_work, out, CopyType::General, s);
  }
  return true;
}

bool is_supported_select_layout(const array& arr) {
  return arr.flags().contiguous && arr.offset() == 0 && arr.ndim() > 0 &&
      arr.strides().back() == 1;
}

array collapse_select_leading_dims(const array& arr, Stream s) {
  if (arr.ndim() <= 4) {
    return arr;
  }
  return flatten_in_eval(arr, 0, arr.ndim() - 4, s);
}

std::optional<vulkan::StaticShaderId> select_shader_id(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return vulkan::StaticShaderId::select_bool;
    case float16:
      return vulkan::StaticShaderId::select_f16;
    case float32:
      return vulkan::StaticShaderId::select_f32;
    case bfloat16:
      return vulkan::StaticShaderId::select_bf16;
    default:
      return std::nullopt;
  }
}

bool try_eval_select_vulkan(
    const std::vector<array>& inputs,
    array& out,
    Stream s) {
  if (inputs.size() != 3) {
    return false;
  }

  array condition = inputs[0];
  array x = inputs[1];
  array y = inputs[2];
  if (condition.dtype() != bool_ || x.dtype() != out.dtype() ||
      y.dtype() != out.dtype() || condition.shape() != out.shape() ||
      x.shape() != out.shape() || y.shape() != out.shape()) {
    return false;
  }

  auto shader_id = select_shader_id(out.dtype());
  if (!shader_id.has_value()) {
    return false;
  }

  auto materialize = [&](array arr) {
    if (!is_supported_select_layout(arr)) {
      arr = contiguous_copy_gpu(arr, s);
    }
    return arr;
  };

  condition = materialize(condition);
  x = materialize(x);
  y = materialize(y);

  const bool staged_output = !is_supported_select_layout(out);
  array out_storage =
      staged_output ? array(out.shape(), out.dtype(), nullptr, {}) : out;
  out_storage.set_data(allocator::malloc(out_storage.nbytes()));

  array cond_kernel = collapse_select_leading_dims(condition, s);
  array x_kernel = collapse_select_leading_dims(x, s);
  array y_kernel = collapse_select_leading_dims(y, s);
  array out_kernel = collapse_select_leading_dims(out_storage, s);

  if (!is_supported_elementwise_layout(cond_kernel) ||
      !is_supported_elementwise_layout(x_kernel) ||
      !is_supported_elementwise_layout(y_kernel) ||
      !is_supported_elementwise_layout(out_kernel)) {
    return false;
  }

  if (out_kernel.size() == 0) {
    if (staged_output) {
      copy_gpu(out_storage, out, CopyType::GeneralGeneral, s);
    }
    return true;
  }

  try {
    auto command_buffer = vulkan::begin_command_recording(s.index);
    vulkan::dispatch_ternary_op(
        cond_kernel,
        x_kernel,
        y_kernel,
        out_kernel,
        *shader_id,
        command_buffer,
        s);
    vulkan::end_command_recording(s.index);
    if (staged_output) {
      copy_gpu(out_storage, out, CopyType::GeneralGeneral, s);
    }
    return true;
  } catch (const std::runtime_error& e) {
    if (trace_fallback_enabled()) {
      std::ostringstream oss;
      oss << "select_dispatch_failed reason=" << e.what();
      trace_fallback(oss.str());
    }
    return false;
  }
}

} // namespace

// Primitives with state that have CPU fallbacks
NYI_OP_STATE(Equal)

// Primitives implemented in other files:
// - binary.cpp: Add, Minimum, Maximum, Divide, Subtract, Multiply
// - unary.cpp: Abs, Ceil, Cos, Exp, Erf, ErfInv, Floor, Log, Sin, etc.
// - reduce.cpp: Reduce, ArgReduce
// - softmax.cpp: Softmax, LogSumExp
// - gather.cpp: Gather, GatherAxis
// - scan.cpp: Scan
// - arange.cpp: Arange
// - rope.cpp: RoPE (no-op fallback)
// - fast.cpp: LayerNorm, RMSNorm, Quantize, ConvertFP8, CustomKernel, SDPA
// - random.cpp: RandomBits

// Load primitive - throw NYI like Metal backend
void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("[Load::eval_gpu] Not implemented.");
}

// CPU fallbacks for primitives not implemented on Vulkan
NYI_OP(ArcCos)
NYI_OP(ArcCosh)
NYI_OP(ArcSin)
NYI_OP(ArcSinh)
NYI_OP(ArcTan)
NYI_OP(ArcTan2)
NYI_OP(ArcTanh)
NYI_OP_STATE(ArgPartition)
NYI_OP_STATE(ArgSort)
NYI_OP_STATE(BitwiseBinary)
NYI_OP(BitwiseInvert)
NYI_OP(Conjugate)
NYI_OP(Cosh)
NYI_OP_MULTI(DivMod)
NYI_OP(Remainder)
NYI_OP(Expm1)
NYI_OP_STATE(FFT)
NYI_OP_STATE(GatherMM)
NYI_OP_STATE(GatherQMM)
NYI_OP(Greater)
NYI_OP_STATE(Hadamard)
NYI_OP(Imag)
NYI_OP(Less)
NYI_OP(LessEqual)
NYI_OP(Log1p)
NYI_OP(LogicalNot)
NYI_OP(LogicalAnd)
NYI_OP(LogicalOr)
NYI_OP(LogAddExp)
// Linear algebra operations - throw NYI like Metal backend
NO_GPU_MULTI(LUF)
NO_GPU_MULTI(QRF)
NO_GPU_STATE(Inverse)
NO_GPU_STATE(Cholesky)
NO_GPU_MULTI_STATE(Eigh)
NO_GPU_MULTI_STATE(Eig)
NO_GPU_MULTI_STATE(SVD)

NYI_OP(NotEqual)
NYI_OP_STATE(Partition)
void Power::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_power_vulkan(inputs, out, stream())) {
    throw std::runtime_error("Power has no Vulkan implementation for this input.");
  }
}
// QuantizedMatmul and QQMatmul are implemented in quantized.cpp.

NYI_OP(Real)
NYI_OP(Sign)
NYI_OP(Sinh)
NYI_OP_STATE(Sort)
NYI_OP(Tan)
NO_GPU(MaskedScatter)
// Scatter and ScatterAxis are implemented in scatter.cpp

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  if (reduce_type_ != SliceUpdate::None) {
    throw std::runtime_error(
        "[SliceUpdate::eval_gpu] Vulkan only supports SliceUpdate::None. "
        "Reduce operations (Sum, Prod, Min, Max) are not yet implemented.");
  }

  if (is_donatable(in, out) && in.dtype() == out.dtype()) {
    out.copy_shared_buffer(in);
  } else {
    auto ctype = in.flags().contiguous && in.size() == in.data_size()
        ? CopyType::Vector
        : CopyType::General;
    copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());
  }

  auto [data_offset, out_strides] =
      prepare_slice(out, start_indices_, strides_);
  copy_gpu_inplace(
      upd,
      out,
      upd.shape(),
      upd.strides(),
      out_strides,
      0,
      data_offset,
      CopyType::GeneralGeneral,
      stream());
}

NYI_OP(SegmentedMM)

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (!try_eval_select_vulkan(inputs, out, stream())) {
    throw std::runtime_error(
        "Select has no Vulkan implementation for this input.");
  }
}

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
NO_GPU_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core
