# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


def _key(seed):
    return mx.random.key(seed)


class TestVulkanOpsParity(mlx_tests.MLXTestCase):
    def _eval_output(self, out):
        if isinstance(out, tuple):
            mx.eval(*out)
            return out
        if isinstance(out, list):
            mx.eval(*out)
            return out
        mx.eval(out)
        return out

    def _assert_outputs_close(self, got, expected, atol, rtol):
        if isinstance(expected, tuple):
            self.assertIsInstance(got, tuple)
            self.assertEqual(len(got), len(expected))
            for lhs, rhs in zip(got, expected):
                self._assert_outputs_close(lhs, rhs, atol, rtol)
            return
        if isinstance(expected, list):
            self.assertIsInstance(got, list)
            self.assertEqual(len(got), len(expected))
            for lhs, rhs in zip(got, expected):
                self._assert_outputs_close(lhs, rhs, atol, rtol)
            return
        self.assertEqual(
            tuple(got.shape),
            tuple(expected.shape),
            msg=f"shape mismatch expected={expected.shape} got={got.shape}",
        )
        self.assertEqual(
            got.dtype,
            expected.dtype,
            msg=f"dtype mismatch expected={expected.dtype} got={got.dtype}",
        )
        np.testing.assert_allclose(
            np.array(got), np.array(expected), rtol=rtol, atol=atol
        )

    def _run_on_device(self, device, fn):
        prev = mx.default_device()
        try:
            mx.set_default_device(device)
            return self._eval_output(fn())
        finally:
            mx.set_default_device(prev)

    def _assert_cpu_gpu_same(self, fn, atol=1e-5, rtol=1e-5):
        cpu_out = self._run_on_device(mx.cpu, fn)
        gpu_out = self._run_on_device(mx.gpu, fn)
        self._assert_outputs_close(gpu_out, cpu_out, atol=atol, rtol=rtol)

    def _assert_cpu_gpu_same_or_fallback(self, fn, atol=1e-5, rtol=1e-5):
        cpu_out = self._run_on_device(mx.cpu, fn)
        try:
            gpu_out = self._run_on_device(mx.gpu, fn)
        except Exception as exc:
            msg = str(exc)
            unsupported_markers = (
                "not yet supported on the GPU",
                "has no Vulkan implementation",
                "no Vulkan implementation",
            )
            if any(marker in msg for marker in unsupported_markers):
                gpu_out = self._run_on_device(mx.cpu, fn)
                self._assert_outputs_close(gpu_out, cpu_out, atol=atol, rtol=rtol)
                return
            raise
        self._assert_outputs_close(gpu_out, cpu_out, atol=atol, rtol=rtol)

    def _assert_gpu_unsupported(self, fn, markers=None):
        if markers is None:
            markers = ("has no Vulkan implementation", "failed on Vulkan")

        with self.assertRaises(RuntimeError) as exc:
            self._run_on_device(mx.gpu, fn)

        msg = str(exc.exception)
        self.assertTrue(
            any(marker in msg for marker in markers),
            msg=f"unexpected GPU error: {msg}",
        )

    def test_fast_rms_norm_low_precision_regression(self):
        for dtype, atol, rtol in (
            (mx.float16, 5e-2, 5e-2),
            (mx.bfloat16, 7e-2, 7e-2),
        ):
            with self.subTest(dtype=str(dtype)):
                self._assert_cpu_gpu_same(
                    lambda dtype=dtype: mx.fast.rms_norm(
                        mx.arange(1, 1 + 2 * 16 * 128, dtype=dtype).reshape(2, 16, 128)
                        / 128.0,
                        mx.linspace(0.5, 1.5, 128, dtype=dtype),
                        1e-5,
                    ).astype(mx.float32),
                    atol=atol,
                    rtol=rtol,
                )

    def test_fast_layer_norm_low_precision_regression(self):
        for dtype, atol, rtol in (
            (mx.float16, 5e-2, 5e-2),
            (mx.bfloat16, 7e-2, 7e-2),
        ):
            with self.subTest(dtype=str(dtype)):
                self._assert_cpu_gpu_same(
                    lambda dtype=dtype: mx.fast.layer_norm(
                        mx.arange(1, 1 + 2 * 16 * 128, dtype=dtype).reshape(2, 16, 128)
                        / 128.0,
                        mx.linspace(0.5, 1.5, 128, dtype=dtype),
                        mx.linspace(-0.25, 0.25, 128, dtype=dtype),
                        1e-5,
                    ).astype(mx.float32),
                    atol=atol,
                    rtol=rtol,
                )

    def test_fast_layer_norm_vjp_regression(self):
        def fn():
            x = mx.arange(1, 1 + 2 * 4 * 8, dtype=mx.float32).reshape(2, 4, 8) / 32.0
            w = mx.linspace(0.5, 1.2, 8, dtype=mx.float32)
            b = mx.linspace(-0.2, 0.2, 8, dtype=mx.float32)
            cotangent = (
                mx.arange(1, 1 + 2 * 4 * 8, dtype=mx.float32).reshape(2, 4, 8) / 64.0
            )
            _, vjps = mx.vjp(
                lambda x, w, b: mx.fast.layer_norm(x, w, b, 1e-5),
                (x, w, b),
                (cotangent,),
            )
            return vjps

        self._assert_cpu_gpu_same(fn, atol=1e-5, rtol=1e-5)

    def test_fast_layer_norm_optional_affine_regression(self):
        cases = (
            ("weight_only", True, False),
            ("bias_only", False, True),
        )

        for name, use_weight, use_bias in cases:
            with self.subTest(case=name):
                self._assert_cpu_gpu_same(
                    lambda use_weight=use_weight, use_bias=use_bias: mx.fast.layer_norm(
                        mx.arange(1, 1 + 2 * 4 * 8, dtype=mx.float32).reshape(2, 4, 8)
                        / 32.0,
                        mx.linspace(0.5, 1.2, 8, dtype=mx.float32)
                        if use_weight
                        else None,
                        mx.linspace(-0.2, 0.2, 8, dtype=mx.float32)
                        if use_bias
                        else None,
                        1e-5,
                    ),
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_fast_layer_norm_strided_affine_regression(self):
        self._assert_cpu_gpu_same(
            lambda: mx.fast.layer_norm(
                mx.arange(1, 1 + 2 * 4 * 8, dtype=mx.float32).reshape(2, 4, 8) / 32.0,
                mx.linspace(0.5, 1.2, 16, dtype=mx.float32)[::2],
                mx.linspace(-0.2, 0.2, 16, dtype=mx.float32)[::2],
                1e-5,
            ),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_arccos_does_not_fallback_to_cpu(self):
        self._assert_gpu_unsupported(
            lambda: mx.arccos(
                mx.array([[-0.75, -0.25], [0.25, 0.75]], dtype=mx.float32)
            ),
            markers=("ArcCos has no Vulkan implementation",),
        )

    def test_equal_int32_vulkan(self):
        self._assert_cpu_gpu_same(
            lambda: mx.equal(
                mx.array([[1, 2], [3, 4]], dtype=mx.int32),
                mx.array([[1, 0], [3, 5]], dtype=mx.int32),
            ),
            atol=0.0,
            rtol=0.0,
        )

    def test_bitwise_and_int32_vulkan(self):
        self._assert_cpu_gpu_same(
            lambda: mx.bitwise_and(
                mx.array([[7, 3], [12, 5]], dtype=mx.int32),
                mx.array([[3, 6], [10, 1]], dtype=mx.int32),
            ),
            atol=0.0,
            rtol=0.0,
        )

    def test_logical_ops_vulkan(self):
        self._assert_cpu_gpu_same(
            lambda: (
                mx.logical_not(mx.array([[True, False], [False, True]])),
                mx.logical_and(
                    mx.array([[True, False], [True, False]]),
                    mx.array([[True, True], [False, False]]),
                ),
                mx.logical_or(
                    mx.array([[True, False], [True, False]]),
                    mx.array([[False, True], [False, False]]),
                ),
            ),
            atol=0.0,
            rtol=0.0,
        )

    def test_arange_bf16_vulkan(self):
        self._assert_cpu_gpu_same(
            lambda: mx.arange(-3, 13, dtype=mx.bfloat16).astype(mx.float32),
            atol=0.0,
            rtol=0.0,
        )

    def test_gather_take_bf16_vulkan(self):
        self._assert_cpu_gpu_same(
            lambda: mx.take(
                mx.arange(1, 1 + 8 * 4, dtype=mx.float32).reshape(8, 4).astype(mx.bfloat16),
                mx.array([5, 1, 7, 0], dtype=mx.int32),
                axis=0,
            ).astype(mx.float32),
            atol=0.0,
            rtol=0.0,
        )

    def test_empty_bool_reduce_vulkan(self):
        self._assert_cpu_gpu_same(
            lambda: (
                mx.all(mx.array([], dtype=mx.bool_)),
                mx.any(mx.array([], dtype=mx.bool_)),
            ),
            atol=0.0,
            rtol=0.0,
        )

    def test_scaled_dot_product_attention_causal_gqa(self):
        self._assert_cpu_gpu_same(
            lambda: mx.fast.scaled_dot_product_attention(
                mx.arange(1, 1 + 1 * 4 * 8 * 16, dtype=mx.float16).reshape(1, 4, 8, 16)
                / 64.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 16, dtype=mx.float16).reshape(1, 2, 8, 16)
                / 48.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 16, dtype=mx.float16).reshape(1, 2, 8, 16)
                / 32.0,
                scale=16**-0.5,
                mask="causal",
            ).astype(mx.float32),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_scaled_dot_product_attention_decode_gqa(self):
        self._assert_cpu_gpu_same(
            lambda: mx.fast.scaled_dot_product_attention(
                mx.arange(1, 1 + 1 * 4 * 1 * 16, dtype=mx.float16).reshape(1, 4, 1, 16)
                / 64.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 16, dtype=mx.float16).reshape(1, 2, 8, 16)
                / 48.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 16, dtype=mx.float16).reshape(1, 2, 8, 16)
                / 32.0,
                scale=16**-0.5,
            ).astype(mx.float32),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_softmax_decode_shape_regressions(self):
        cases = (
            (mx.float32, (2, 20, 64), 1e-5, 1e-5),
            (mx.float32, (3, 4, 5, 256), 1e-5, 1e-5),
            (mx.float16, (1, 8, 1, 1025), 5e-3, 5e-3),
        )

        def fn(shape, dtype):
            size = int(np.prod(shape))
            x = mx.arange(size, dtype=mx.float32).reshape(shape)
            x = (x - size / 2) / shape[-1]
            x = x.astype(dtype)
            return mx.softmax(x, axis=-1).astype(mx.float32)

        for dtype, shape, atol, rtol in cases:
            with self.subTest(dtype=str(dtype), shape=shape):
                self._assert_cpu_gpu_same(
                    lambda shape=shape, dtype=dtype: fn(shape, dtype),
                    atol=atol,
                    rtol=rtol,
                )

    def test_list_getitem_regression(self):
        def fn():
            a = mx.arange(16, dtype=mx.float32).reshape(4, 4)
            idx = mx.array([0, 2], dtype=mx.uint32)
            return (a[0, idx], a[:, idx])

        self._assert_cpu_gpu_same(fn, atol=0.0, rtol=0.0)

    def test_list_setitem_regression(self):
        def fn():
            a = mx.arange(16, dtype=mx.float32).reshape(4, 4)
            idx = mx.array([0, 2], dtype=mx.uint32)
            a[0, idx] = 3.0
            a[:, idx] = 4.0
            return a

        self._assert_cpu_gpu_same(fn, atol=0.0, rtol=0.0)

    def test_scaled_dot_product_attention_qwen_shape_regression(self):
        self._assert_cpu_gpu_same(
            lambda: mx.fast.scaled_dot_product_attention(
                mx.arange(
                    1, 1 + int(np.prod((1, 16, 8, 128))), dtype=mx.float16
                ).reshape(1, 16, 8, 128)
                / 64.0,
                mx.arange(
                    1, 1 + int(np.prod((1, 8, 8, 128))), dtype=mx.float16
                ).reshape(1, 8, 8, 128)
                / 48.0,
                mx.arange(
                    1, 1 + int(np.prod((1, 8, 8, 128))), dtype=mx.float16
                ).reshape(1, 8, 8, 128)
                / 32.0,
                scale=128**-0.5,
                mask="causal",
            ).astype(mx.float32),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_scaled_dot_product_attention_qwen_shape_bf16_regression(self):
        self._assert_cpu_gpu_same(
            lambda: mx.fast.scaled_dot_product_attention(
                mx.arange(
                    1, 1 + int(np.prod((1, 16, 8, 128))), dtype=mx.bfloat16
                ).reshape(1, 16, 8, 128)
                / 64.0,
                mx.arange(
                    1, 1 + int(np.prod((1, 8, 8, 128))), dtype=mx.bfloat16
                ).reshape(1, 8, 8, 128)
                / 48.0,
                mx.arange(
                    1, 1 + int(np.prod((1, 8, 8, 128))), dtype=mx.bfloat16
                ).reshape(1, 8, 8, 128)
                / 32.0,
                scale=128**-0.5,
                mask="causal",
            ).astype(mx.float32),
            atol=6e-2,
            rtol=6e-2,
        )

    def test_scaled_dot_product_attention_additive_mask(self):
        mask = mx.where(
            mx.arange(8)[None, None, None, :] <= mx.arange(8)[None, None, :, None],
            mx.array(0.0, mx.float16),
            mx.array(-1e9, mx.float16),
        )
        self._assert_cpu_gpu_same(
            lambda: mx.fast.scaled_dot_product_attention(
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 64.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 48.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 32.0,
                scale=32**-0.5,
                mask=mask,
            ).astype(mx.float32),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_scaled_dot_product_attention_bool_mask(self):
        mask = mx.arange(8)[None, None, None, :] <= mx.arange(8)[None, None, :, None]
        self._assert_cpu_gpu_same(
            lambda: mx.fast.scaled_dot_product_attention(
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 64.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 48.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 32.0,
                scale=32**-0.5,
                mask=mask,
            ).astype(mx.float32),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_scaled_dot_product_attention_additive_mask_gqa(self):
        mask = mx.where(
            mx.arange(8)[None, None, None, :] <= mx.arange(8)[None, None, :, None],
            mx.array(0.0, mx.float16),
            mx.array(-1e9, mx.float16),
        )
        self._assert_cpu_gpu_same(
            lambda: mx.fast.scaled_dot_product_attention(
                mx.arange(1, 1 + 1 * 4 * 8 * 32, dtype=mx.float16).reshape(1, 4, 8, 32)
                / 64.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 48.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 32.0,
                scale=32**-0.5,
                mask=mask,
            ).astype(mx.float32),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_scaled_dot_product_attention_bool_mask_gqa(self):
        mask = mx.arange(8)[None, None, None, :] <= mx.arange(8)[None, None, :, None]
        self._assert_cpu_gpu_same(
            lambda: mx.fast.scaled_dot_product_attention(
                mx.arange(1, 1 + 1 * 4 * 8 * 32, dtype=mx.float16).reshape(1, 4, 8, 32)
                / 64.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 48.0,
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 32.0,
                scale=32**-0.5,
                mask=mask,
            ).astype(mx.float32),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_scaled_dot_product_attention_vjp_gqa(self):
        def fn():
            q = (
                mx.arange(1, 1 + 1 * 4 * 8 * 32, dtype=mx.float16).reshape(1, 4, 8, 32)
                / 64.0
            )
            k = (
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 48.0
            )
            v = (
                mx.arange(1, 1 + 1 * 2 * 8 * 32, dtype=mx.float16).reshape(1, 2, 8, 32)
                / 32.0
            )
            cotangent = (
                mx.arange(1, 1 + 1 * 4 * 8 * 32, dtype=mx.float16).reshape(1, 4, 8, 32)
                / 96.0
            )
            _, vjps = mx.vjp(
                lambda q, k, v: mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=32**-0.5
                ),
                (q, k, v),
                (cotangent,),
            )
            return tuple(x.astype(mx.float32) for x in vjps)

        self._assert_cpu_gpu_same(fn, atol=7e-2, rtol=7e-2)

    def test_dynamic_slice_update_5d_regression(self):
        self._assert_cpu_gpu_same(
            lambda: mx.slice_update(
                mx.zeros((2, 3, 4, 5, 6), dtype=mx.float32),
                mx.arange(1, 1 + 2 * 3 * 4 * 1 * 6, dtype=mx.float32).reshape(
                    2, 3, 4, 1, 6
                ),
                mx.array([0, 0, 0, 2, 0], dtype=mx.int32),
                (0, 1, 2, 3, 4),
            ),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_put_along_axis_vulkan(self):
        self._assert_cpu_gpu_same(
            lambda: mx.put_along_axis(
                mx.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=mx.float32),
                mx.array([[2, 1], [0, 2]], dtype=mx.uint32),
                mx.array([[7.0, 8.0], [9.0, 10.0]], dtype=mx.float32),
                axis=1,
            ),
            atol=0.0,
            rtol=0.0,
        )

    def test_quantized_matmul_fused_low_precision_regression(self):
        for dtype, atol, rtol in (
            (mx.float16, 2e-3, 2e-3),
            (mx.bfloat16, 3e-3, 3e-3),
        ):
            with self.subTest(dtype=str(dtype)):

                def fn(dtype=dtype):
                    x = mx.arange(1, 1 + 3 * 128, dtype=dtype).reshape(3, 128) / 64.0
                    w = (
                        mx.arange(1, 1 + 96 * 128, dtype=mx.float32).reshape(96, 128)
                        / 80.0
                    )
                    w_q, scales, biases = mx.quantize(w, group_size=64, bits=8)
                    return mx.quantized_matmul(
                        x, w_q, scales, biases, True, 64, 8
                    ).astype(mx.float32)

                self._assert_cpu_gpu_same(fn, atol=atol, rtol=rtol)

    def test_quantized_matmul_fused_flatten_batches_regression(self):
        for dtype, atol, rtol in (
            (mx.float16, 2e-3, 2e-3),
            (mx.bfloat16, 3e-3, 3e-3),
        ):
            with self.subTest(dtype=str(dtype)):

                def fn(dtype=dtype):
                    x = (
                        mx.arange(1, 1 + 2 * 1 * 4 * 128, dtype=dtype).reshape(
                            2, 1, 4, 128
                        )
                        / 96.0
                    )
                    w = (
                        mx.arange(1, 1 + 64 * 128, dtype=mx.float32).reshape(64, 128)
                        / 72.0
                    )
                    w_q, scales, biases = mx.quantize(w, group_size=64, bits=8)
                    return mx.quantized_matmul(
                        x, w_q, scales, biases, True, 64, 8
                    ).astype(mx.float32)

                self._assert_cpu_gpu_same(fn, atol=atol, rtol=rtol)

    def test_fast_rope_bf16_vulkan_gpu(self):
        """Regression test: bf16 RoPE should run on Vulkan GPU, not fall back to CPU."""

        def run_rope():
            x = mx.arange(1, 1 + 2 * 4 * 8, dtype=mx.bfloat16).reshape(2, 4, 8) / 32.0
            return mx.fast.rope(
                x,
                dims=8,
                traditional=False,
                base=10000.0,
                scale=1.0,
                offset=0,
            )

        cpu_out = self._run_on_device(mx.cpu, run_rope)
        gpu_out = self._run_on_device(mx.gpu, run_rope)
        cpu_out_f32 = cpu_out.astype(mx.float32)
        gpu_out_f32 = gpu_out.astype(mx.float32)
        self._assert_outputs_close(gpu_out_f32, cpu_out_f32, atol=1e-2, rtol=1e-2)

    def test_bf16_copy_with_large_destination_offset_vulkan_gpu(self):
        """Regression test: bf16 general copy should accept large byte offsets."""

        def run_copy():
            src = (
                mx.arange(1, 1 + 8 * 256 * 128, dtype=mx.float32)
                .reshape(1, 8, 256, 128)
                .astype(mx.bfloat16)
            )
            dst = mx.zeros((2, 8, 256, 128), dtype=mx.bfloat16)
            updated = mx.concatenate([dst[:1], src], axis=0)
            return updated[1]

        cpu_out = self._run_on_device(mx.cpu, run_copy).astype(mx.float32)
        gpu_out = self._run_on_device(mx.gpu, run_copy).astype(mx.float32)
        self._assert_outputs_close(gpu_out, cpu_out, atol=1e-2, rtol=1e-2)

    def test_cached_prefill_parity_regression(self):
        class TinyKVCache:
            def __init__(self):
                self.keys = None
                self.values = None
                self.offset = 0

            def update_and_fetch(self, keys, values):
                prev = self.offset
                if self.keys is None:
                    batch, n_kv_heads, steps, head_dim = keys.shape
                    value_dim = values.shape[-1]
                    alloc_steps = max(steps, 16)
                    self.keys = mx.zeros(
                        (batch, n_kv_heads, alloc_steps, head_dim), dtype=keys.dtype
                    )
                    self.values = mx.zeros(
                        (batch, n_kv_heads, alloc_steps, value_dim), dtype=values.dtype
                    )
                self.offset += keys.shape[2]
                self.keys[..., prev : self.offset, :] = keys
                self.values[..., prev : self.offset, :] = values
                return (
                    self.keys[..., : self.offset, :],
                    self.values[..., : self.offset, :],
                )

        def run(dtype=mx.bfloat16):
            batch = 1
            prompt_len = 13
            dim = 64
            n_heads = 4
            n_kv_heads = 2
            head_dim = 16
            vocab = 97

            tokens = mx.arange(prompt_len, dtype=mx.uint32)[None]
            embed = (
                mx.arange(vocab * dim, dtype=mx.float32).reshape(vocab, dim) / 97.0
            ).astype(dtype)
            w_q = (
                mx.arange(dim * (n_heads * head_dim), dtype=mx.float32).reshape(
                    dim, n_heads * head_dim
                )
                / 211.0
            ).astype(dtype)
            w_k = (
                mx.arange(dim * (n_kv_heads * head_dim), dtype=mx.float32).reshape(
                    dim, n_kv_heads * head_dim
                )
                / 173.0
            ).astype(dtype)
            w_v = (
                mx.arange(dim * (n_kv_heads * head_dim), dtype=mx.float32).reshape(
                    dim, n_kv_heads * head_dim
                )
                / 157.0
            ).astype(dtype)
            w_o = (
                mx.arange((n_heads * head_dim) * dim, dtype=mx.float32).reshape(
                    n_heads * head_dim, dim
                )
                / 193.0
            ).astype(dtype)
            lm_head = (
                mx.arange(dim * vocab, dtype=mx.float32).reshape(dim, vocab) / 181.0
            ).astype(dtype)

            def rms_norm(x, eps=1e-5):
                x_f32 = x.astype(mx.float32)
                scale = mx.rsqrt(
                    mx.mean(mx.square(x_f32), axis=-1, keepdims=True) + eps
                )
                return (x_f32 * scale).astype(dtype)

            def attention(x, cache=None):
                q = mx.matmul(x, w_q).reshape(batch, prompt_len, n_heads, head_dim)
                k = mx.matmul(x, w_k).reshape(batch, prompt_len, n_kv_heads, head_dim)
                v = mx.matmul(x, w_v).reshape(batch, prompt_len, n_kv_heads, head_dim)
                q = rms_norm(q).transpose(0, 2, 1, 3)
                k = rms_norm(k).transpose(0, 2, 1, 3)
                v = v.transpose(0, 2, 1, 3)
                if cache is not None:
                    k, v = cache.update_and_fetch(k, v)
                out = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=head_dim**-0.5,
                    mask="causal",
                )
                out = out.transpose(0, 2, 1, 3).reshape(batch, prompt_len, dim)
                return mx.matmul(out, w_o)

            x = embed[tokens]
            layer_no_cache = attention(rms_norm(x), cache=None)
            layer_cached = attention(rms_norm(x), cache=TinyKVCache())
            logits_no_cache = mx.matmul(layer_no_cache, lm_head)[:, -1, :].astype(
                mx.float32
            )
            logits_cached = mx.matmul(layer_cached, lm_head)[:, -1, :].astype(
                mx.float32
            )
            return (
                layer_no_cache[:, -1, :].astype(mx.float32),
                layer_cached[:, -1, :].astype(mx.float32),
                logits_no_cache,
                logits_cached,
            )

        cpu_layer_no, cpu_layer_cached, cpu_logits_no, cpu_logits_cached = (
            self._run_on_device(mx.cpu, run)
        )
        gpu_layer_no, gpu_layer_cached, gpu_logits_no, gpu_logits_cached = (
            self._run_on_device(mx.gpu, run)
        )

        self._assert_outputs_close(cpu_layer_cached, cpu_layer_no, atol=1e-2, rtol=1e-2)
        self._assert_outputs_close(
            cpu_logits_cached, cpu_logits_no, atol=1e-2, rtol=1e-2
        )
        self._assert_outputs_close(gpu_layer_cached, gpu_layer_no, atol=1e-2, rtol=1e-2)
        self._assert_outputs_close(
            gpu_logits_cached, gpu_logits_no, atol=1e-2, rtol=1e-2
        )

    def test_host_source_slice_cast_copy_vulkan_gpu(self):
        host_src = self._run_on_device(
            mx.cpu,
            lambda: mx.arange(0, 10, dtype=mx.float32),
        )

        def run_copy():
            return host_src[2:8].astype(mx.float16)

        cpu_out = self._run_on_device(mx.cpu, run_copy).astype(mx.float32)
        gpu_out = self._run_on_device(mx.gpu, run_copy).astype(mx.float32)
        self._assert_outputs_close(gpu_out, cpu_out, atol=0.0, rtol=0.0)

    def test_host_staging_reuse_vulkan_gpu(self):
        for start in range(4):
            host_src = self._run_on_device(
                mx.cpu,
                lambda start=start: mx.arange(start, start + 4096, dtype=mx.float32),
            )

            def run_copy(host_src=host_src):
                return host_src[128:640].astype(mx.float16).astype(mx.float32)

            cpu_out = self._run_on_device(mx.cpu, run_copy)
            gpu_out = self._run_on_device(mx.gpu, run_copy)
            self._assert_outputs_close(gpu_out, cpu_out, atol=0.0, rtol=0.0)

    def test_gpu_source_unsupported_cast_copy_vulkan_gpu(self):
        def run_copy():
            src = mx.arange(0, 4096, dtype=mx.int64)
            return src[64:2048].astype(mx.float16).astype(mx.float32)

        cpu_out = self._run_on_device(mx.cpu, run_copy)
        gpu_out = self._run_on_device(mx.gpu, run_copy)
        self._assert_outputs_close(gpu_out, cpu_out, atol=0.0, rtol=0.0)

    def test_host_staging_arena_growth_and_reuse_vulkan_gpu(self):
        for cycle in range(3):
            host_srcs = [
                self._run_on_device(
                    mx.cpu,
                    lambda start=start: mx.arange(
                        start,
                        start + 128 * 1024,
                        dtype=mx.float32,
                    ),
                )
                for start in (cycle, cycle + 1, cycle + 2)
            ]

            def run_copy(host_srcs=host_srcs):
                parts = [
                    host_src[1024:-1024].astype(mx.float16).astype(mx.float32)
                    for host_src in host_srcs
                ]
                return mx.concatenate(parts, axis=0)

            cpu_out = self._run_on_device(mx.cpu, run_copy)
            gpu_out = self._run_on_device(mx.gpu, run_copy)
            self._assert_outputs_close(gpu_out, cpu_out, atol=0.0, rtol=0.0)

    def test_host_staging_wrapped_full_ring_regression_vulkan_gpu(self):
        sizes = [96, 32, 64, 32, 32, 32, 32, 32, 8]
        host_srcs = [
            self._run_on_device(
                mx.cpu,
                lambda start=start, size=size: mx.arange(
                    start,
                    start + size * 1024,
                    dtype=mx.float32,
                ),
            )
            for start, size in enumerate(sizes)
        ]

        prev = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            stream = mx.new_stream(mx.gpu)

            def submit(host_src, heavy=False):
                out = mx.abs(host_src, stream=stream)
                if heavy:
                    for _ in range(96):
                        out = mx.abs(out, stream=stream)
                mx.async_eval(out)
                return out

            outs = [submit(host_srcs[0]), submit(host_srcs[1], heavy=True)]

            cpu_burn = host_srcs[0]
            for _ in range(96):
                cpu_burn = mx.abs(cpu_burn)
            mx.eval(cpu_burn)

            for host_src in host_srcs[2:]:
                outs.append(submit(host_src))

            mx.synchronize(stream)
        finally:
            mx.set_default_device(prev)

        for gpu_out, host_src in zip(outs, host_srcs):
            self._assert_outputs_close(gpu_out, host_src, atol=0.0, rtol=0.0)

    def test_large_low_precision_gather_regression(self):
        for dtype, atol in ((mx.float16, 5e-3), (mx.bfloat16, 5e-2)):
            with self.subTest(dtype=str(dtype)):
                self._assert_cpu_gpu_same(
                    lambda dtype=dtype: mx.arange(
                        100 * 8960, dtype=mx.float32
                    ).reshape(100, 8960).astype(dtype)[mx.array([[23]], dtype=mx.int32)].astype(
                        mx.float32
                    ),
                    atol=atol,
                    rtol=atol,
                )

    def test_compiled_gelu_approx_negative_power_regression(self):
        def gelu_approx(x):
            return 0.5 * x * (1.0 + mx.tanh(0.7978845608 * (x + 0.044715 * mx.power(x, 3.0))))

        def make_x():
            return mx.linspace(-6.0, 6.0, 6144, dtype=mx.float32).reshape(1, 1, 6144)

        @mx.compile
        def compiled(gate, value):
            return gelu_approx(gate) * value

        expected = self._run_on_device(mx.cpu, lambda: gelu_approx(make_x()) * make_x())
        actual = self._run_on_device(mx.gpu, lambda: compiled(make_x(), make_x()))
        self._assert_outputs_close(
            actual.astype(mx.float32),
            expected.astype(mx.float32),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_power_mixed_dtype_sliced_inputs_regression(self):
        def fn():
            base = mx.arange(0, 256, dtype=mx.float32).reshape(16, 16) / 32.0 + 0.5
            exponents = (mx.arange(0, 256, dtype=mx.float32).reshape(16, 16) % 5) / 4.0
            lhs = base[:, 1:-1].astype(mx.float32)
            rhs = exponents[:, 1:-1].astype(mx.float16)
            return mx.power(lhs, rhs).astype(mx.float32)

        self._assert_cpu_gpu_same(fn, atol=1e-3, rtol=1e-3)

    def test_compiled_nonzero_runtime_offsets_regression(self):
        def eager(a, b):
            return (a + b) * 0.5

        def make_inputs():
            lhs = mx.arange(0, 128, dtype=mx.float32)[8:120].reshape(7, 16)
            rhs = mx.linspace(1.0, 2.0, 128, dtype=mx.float32)[8:120].reshape(7, 16)
            return lhs, rhs

        @mx.compile
        def compiled(a, b):
            return eager(a, b)

        expected = self._run_on_device(mx.cpu, lambda: eager(*make_inputs()))
        actual = self._run_on_device(mx.gpu, lambda: compiled(*make_inputs()))
        self._assert_outputs_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_compiled_int8_uint8_regression(self):
        def make_int8():
            return mx.arange(-32, 32, dtype=mx.int8)

        def make_uint8():
            return mx.arange(0, 64, dtype=mx.uint8)

        @mx.compile
        def compiled_int8(a):
            return mx.add(a, mx.array(7, dtype=mx.int8))

        @mx.compile
        def compiled_uint8(a):
            return mx.add(a, mx.array(7, dtype=mx.uint8))

        expected_int8 = self._run_on_device(mx.cpu, lambda: compiled_int8(make_int8()))
        actual_int8 = self._run_on_device(mx.gpu, lambda: compiled_int8(make_int8()))
        self._assert_outputs_close(actual_int8, expected_int8, atol=0.0, rtol=0.0)

        expected_uint8 = self._run_on_device(mx.cpu, lambda: compiled_uint8(make_uint8()))
        actual_uint8 = self._run_on_device(mx.gpu, lambda: compiled_uint8(make_uint8()))
        self._assert_outputs_close(actual_uint8, expected_uint8, atol=0.0, rtol=0.0)

    def test_compiled_bool_roundtrip_regression(self):
        def make_bool():
            return mx.array([True, False, True, True, False, False, True], dtype=mx.bool_)

        @mx.compile
        def compiled_bool(a):
            return a.astype(mx.int8).astype(mx.bool_)

        expected = self._run_on_device(mx.cpu, lambda: compiled_bool(make_bool()))
        actual = self._run_on_device(mx.gpu, lambda: compiled_bool(make_bool()))
        self._assert_outputs_close(actual, expected, atol=0.0, rtol=0.0)

    def test_compiled_bool_add_falls_back_regression(self):
        def make_bool_pair():
            lhs = mx.array([True, False, True, False], dtype=mx.bool_)
            rhs = mx.array([False, True, True, False], dtype=mx.bool_)
            return lhs, rhs

        @mx.compile
        def compiled_bool_add(a, b):
            return mx.add(a, b)

        expected = self._run_on_device(mx.cpu, lambda: compiled_bool_add(*make_bool_pair()))
        actual = self._run_on_device(mx.gpu, lambda: compiled_bool_add(*make_bool_pair()))
        self._assert_outputs_close(actual, expected, atol=0.0, rtol=0.0)

def _cases():
    return [
        (
            "add",
            lambda: mx.add(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "subtract",
            lambda: mx.subtract(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "multiply",
            lambda: mx.multiply(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "divide",
            lambda: mx.divide(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "minimum",
            lambda: mx.minimum(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "maximum",
            lambda: mx.maximum(
                mx.array([[1.5, -2.0], [3.25, 0.5]]),
                mx.array([[0.5, 4.0], [-1.25, 2.0]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "equal",
            lambda: mx.equal(mx.array([[1, 2], [3, 4]]), mx.array([[1, 0], [3, 5]])),
            0.0,
            0.0,
        ),
        (
            "not_equal",
            lambda: mx.not_equal(
                mx.array([[1, 2], [3, 4]]), mx.array([[1, 0], [3, 5]])
            ),
            0.0,
            0.0,
        ),
        (
            "less",
            lambda: mx.less(
                mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[1.0, 3.0], [2.0, 4.0]])
            ),
            0.0,
            0.0,
        ),
        (
            "less_equal",
            lambda: mx.less_equal(
                mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[1.0, 3.0], [2.0, 4.0]])
            ),
            0.0,
            0.0,
        ),
        (
            "greater",
            lambda: mx.greater(
                mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[1.0, 3.0], [2.0, 4.0]])
            ),
            0.0,
            0.0,
        ),
        (
            "greater_equal",
            lambda: mx.greater_equal(
                mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[1.0, 3.0], [2.0, 4.0]])
            ),
            0.0,
            0.0,
        ),
        (
            "remainder",
            lambda: mx.remainder(
                mx.array([[7.5, -7.5], [5.25, -5.25]]),
                mx.array([[2.0, 2.0], [1.5, 1.5]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "power",
            lambda: mx.power(
                mx.array([[1.5, 2.0], [3.0, 0.5]]), mx.array([[2.0, 3.0], [1.5, 2.0]])
            ),
            1e-5,
            1e-5,
        ),
        (
            "logaddexp",
            lambda: mx.logaddexp(
                mx.array([[-2.0, -1.0], [0.5, 2.0]]),
                mx.array([[1.0, -0.5], [0.25, 1.5]]),
            ),
            1e-6,
            1e-6,
        ),
        (
            "logical_not",
            lambda: mx.logical_not(mx.array([[True, False], [False, True]])),
            0.0,
            0.0,
        ),
        (
            "logical_and",
            lambda: mx.logical_and(
                mx.array([[True, False], [True, False]]),
                mx.array([[True, True], [False, False]]),
            ),
            0.0,
            0.0,
        ),
        (
            "logical_or",
            lambda: mx.logical_or(
                mx.array([[True, False], [True, False]]),
                mx.array([[True, True], [False, False]]),
            ),
            0.0,
            0.0,
        ),
        (
            "bitwise_and",
            lambda: mx.bitwise_and(
                mx.array([[1, 7], [3, 12]], dtype=mx.int32),
                mx.array([[5, 3], [6, 10]], dtype=mx.int32),
            ),
            0.0,
            0.0,
        ),
        (
            "bitwise_invert",
            lambda: mx.bitwise_invert(mx.array([[1, 7], [3, 12]], dtype=mx.int32)),
            0.0,
            0.0,
        ),
        ("abs", lambda: mx.abs(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        ("ceil", lambda: mx.ceil(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        ("exp", lambda: mx.exp(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        ("floor", lambda: mx.floor(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        (
            "negative",
            lambda: mx.negative(mx.array([[-1.5, 0.0], [2.25, -3.5]])),
            1e-6,
            1e-6,
        ),
        ("round", lambda: mx.round(mx.array([[-1.5, -0.5], [2.25, 3.5]])), 1e-6, 1e-6),
        (
            "sigmoid",
            lambda: mx.sigmoid(mx.array([[-1.5, 0.0], [2.25, -3.5]])),
            1e-6,
            1e-6,
        ),
        ("tanh", lambda: mx.tanh(mx.array([[-1.5, 0.0], [2.25, -3.5]])), 1e-6, 1e-6),
        (
            "cos",
            lambda: mx.cos(mx.array([[-1.5, 0.0], [2.25, -3.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "erf",
            lambda: mx.erf(mx.array([[-1.5, 0.0], [0.5, -0.75]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "erfinv",
            lambda: mx.erfinv(mx.array([[-0.9, -0.25], [0.25, 0.9]], dtype=mx.float32)),
            1e-4,
            1e-4,
        ),
        (
            "log",
            lambda: mx.log(mx.array([[0.25, 1.0], [2.25, 4.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "sin",
            lambda: mx.sin(mx.array([[-1.5, 0.0], [2.25, -3.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "square",
            lambda: mx.square(mx.array([[-1.5, 0.0], [2.25, -3.5]], dtype=mx.float32)),
            1e-6,
            1e-6,
        ),
        (
            "sqrt",
            lambda: mx.sqrt(mx.array([[0.25, 1.0], [2.25, 4.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "rsqrt",
            lambda: mx.rsqrt(mx.array([[0.25, 1.0], [2.25, 4.5]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "arccos",
            lambda: mx.arccos(
                mx.array([[-0.75, -0.25], [0.25, 0.75]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "arccosh",
            lambda: mx.arccosh(mx.array([[1.25, 2.0], [3.5, 5.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "arcsin",
            lambda: mx.arcsin(
                mx.array([[-0.75, -0.25], [0.25, 0.75]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "arcsinh",
            lambda: mx.arcsinh(mx.array([[-3.0, -0.5], [0.5, 3.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "arctan",
            lambda: mx.arctan(mx.array([[-3.0, -0.5], [0.5, 3.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "arctan2",
            lambda: mx.arctan2(
                mx.array([[1.0, -1.0], [0.5, -0.5]], dtype=mx.float32),
                mx.array([[0.5, 0.5], [1.0, 1.0]], dtype=mx.float32),
            ),
            1e-5,
            1e-5,
        ),
        (
            "arctanh",
            lambda: mx.arctanh(
                mx.array([[-0.75, -0.25], [0.25, 0.75]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "conjugate",
            lambda: mx.conjugate(
                mx.array(
                    [[1.0 + 2.0j, 3.0 - 4.0j], [-1.0 + 0.5j, 2.5 + 1.5j]],
                    dtype=mx.complex64,
                )
            ),
            1e-6,
            1e-6,
        ),
        (
            "real",
            lambda: mx.real(
                mx.array(
                    [[1.0 + 2.0j, 3.0 - 4.0j], [-1.0 + 0.5j, 2.5 + 1.5j]],
                    dtype=mx.complex64,
                )
            ),
            1e-6,
            1e-6,
        ),
        (
            "imag",
            lambda: mx.imag(
                mx.array(
                    [[1.0 + 2.0j, 3.0 - 4.0j], [-1.0 + 0.5j, 2.5 + 1.5j]],
                    dtype=mx.complex64,
                )
            ),
            1e-6,
            1e-6,
        ),
        (
            "sign",
            lambda: mx.sign(mx.array([[-3.0, -0.0], [0.5, 4.0]], dtype=mx.float32)),
            1e-6,
            1e-6,
        ),
        (
            "sinh",
            lambda: mx.sinh(mx.array([[-3.0, -0.5], [0.5, 3.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "tan",
            lambda: mx.tan(mx.array([[-1.0, -0.5], [0.5, 1.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "expm1",
            lambda: mx.expm1(mx.array([[-1.0, -0.5], [0.5, 1.0]], dtype=mx.float32)),
            1e-6,
            1e-6,
        ),
        (
            "log1p",
            lambda: mx.log1p(mx.array([[0.0, 0.5], [1.0, 3.0]], dtype=mx.float32)),
            1e-6,
            1e-6,
        ),
        ("arange", lambda: mx.arange(-3.0, 5.0, 0.5), 1e-6, 1e-6),
        (
            "sum",
            lambda: mx.sum(
                mx.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=mx.float32), axis=1
            ),
            1e-6,
            1e-6,
        ),
        (
            "argmax",
            lambda: mx.argmax(
                mx.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=mx.float32), axis=1
            ),
            0.0,
            0.0,
        ),
        (
            "softmax",
            lambda: mx.softmax(
                mx.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=mx.float32),
                axis=-1,
            ),
            1e-5,
            1e-5,
        ),
        (
            "softmax_large",
            lambda: mx.softmax(mx.arange(16385, dtype=mx.float32)[None], axis=-1),
            1e-4,
            1e-4,
        ),
        (
            "logsumexp",
            lambda: mx.logsumexp(
                mx.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=mx.float32), axis=1
            ),
            1e-4,
            1e-4,
        ),
        (
            "take",
            lambda: mx.take(
                mx.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=mx.float32),
                mx.array([2, 0], dtype=mx.uint32),
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "take_along_axis",
            lambda: mx.take_along_axis(
                mx.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=mx.float32),
                mx.array([[2, 1], [0, 2]], dtype=mx.uint32),
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "put_along_axis",
            lambda: mx.put_along_axis(
                mx.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=mx.float32),
                mx.array([[2, 1], [0, 2]], dtype=mx.uint32),
                mx.array([[7.0, 8.0], [9.0, 10.0]], dtype=mx.float32),
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "concat",
            lambda: mx.concatenate(
                [
                    mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
                    mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32),
                ],
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "concat_f16",
            lambda: mx.concatenate(
                [
                    mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float16),
                    mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float16),
                ],
                axis=1,
            ).astype(mx.float32),
            1e-3,
            1e-3,
        ),
        (
            "concat_int32",
            lambda: mx.concatenate(
                [
                    mx.array([[1, 2], [3, 4]], dtype=mx.int32),
                    mx.array([[5, 6], [7, 8]], dtype=mx.int32),
                ],
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "where",
            lambda: mx.where(
                mx.array([[True, False], [False, True]]),
                mx.array([[1.0, 2.0], [3.0, 4.0]]),
                mx.array([[10.0, 20.0], [30.0, 40.0]]),
            ),
            0.0,
            0.0,
        ),
        (
            "cumsum",
            lambda: mx.cumsum(
                mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32), axis=1
            ),
            1e-5,
            1e-5,
        ),
        (
            "sort",
            lambda: mx.sort(
                mx.array([[3.0, 1.0, 2.0], [5.0, 4.0, -1.0]], dtype=mx.float32), axis=1
            ),
            0.0,
            0.0,
        ),
        (
            "argsort",
            lambda: mx.argsort(
                mx.array([[3.0, 1.0, 2.0], [5.0, 4.0, -1.0]], dtype=mx.float32), axis=1
            ),
            0.0,
            0.0,
        ),
        (
            "partition",
            lambda: mx.partition(
                mx.array(
                    [[9.0, 1.0, 5.0, 3.0], [8.0, 4.0, 2.0, 7.0]], dtype=mx.float32
                ),
                2,
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "argpartition",
            lambda: mx.argpartition(
                mx.array(
                    [[9.0, 1.0, 5.0, 3.0], [8.0, 4.0, 2.0, 7.0]], dtype=mx.float32
                ),
                2,
                axis=1,
            ),
            0.0,
            0.0,
        ),
        (
            "fft",
            lambda: mx.fft.fft(
                mx.array(
                    [[1.0, 2.0, 3.0, 4.0], [0.5, -0.5, 1.5, -1.5]], dtype=mx.float32
                ),
                axis=-1,
            ),
            1e-5,
            1e-5,
        ),
        (
            "conv2d",
            lambda: mx.conv2d(
                mx.arange(1, 1 + 1 * 4 * 4 * 2, dtype=mx.float32).reshape(1, 4, 4, 2),
                mx.arange(1, 1 + 3 * 3 * 2 * 2, dtype=mx.float32).reshape(3, 3, 2, 2)
                / 10.0,
            ),
            1e-5,
            1e-5,
        ),
        (
            "hadamard_transform",
            lambda: mx.hadamard_transform(
                mx.arange(32, dtype=mx.float32).reshape(4, 8)
            ),
            1e-5,
            1e-5,
        ),
        (
            "matmul",
            lambda: mx.matmul(
                mx.arange(1, 1 + 32, dtype=mx.float16).reshape(1, 32) / 32.0,
                mx.arange(1, 1 + 32 * 24, dtype=mx.float16).reshape(32, 24) / 24.0,
            ),
            5e-2,
            5e-2,
        ),
        (
            "matvec",
            lambda: mx.matmul(
                mx.arange(1, 1 + 1 * 32, dtype=mx.float16).reshape(1, 32) / 32.0,
                mx.arange(1, 1 + 32 * 48, dtype=mx.float16).reshape(32, 48) / 48.0,
            ),
            5e-2,
            5e-2,
        ),
        (
            "addmm",
            lambda: mx.addmm(
                mx.ones((8, 8), dtype=mx.float32),
                mx.arange(1, 1 + 8 * 16, dtype=mx.float32).reshape(8, 16) / 64.0,
                mx.arange(1, 1 + 16 * 8, dtype=mx.float32).reshape(16, 8) / 64.0,
                alpha=0.5,
                beta=1.5,
            ),
            1e-5,
            1e-5,
        ),
        (
            "block_masked_mm",
            lambda: mx.block_masked_mm(
                mx.arange(1, 1 + 64 * 64, dtype=mx.float32).reshape(64, 64) / 256.0,
                mx.arange(1, 1 + 64 * 64, dtype=mx.float32).reshape(64, 64) / 128.0,
                32,
                mx.array([[True, False], [True, True]]),
                mx.array([[True, True], [False, True]]),
                mx.array([[True, False], [True, True]]),
            ),
            1e-4,
            1e-4,
        ),
        (
            "inverse",
            lambda: mx.linalg.inv(mx.array([[4.0, 1.0], [1.0, 3.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "cholesky",
            lambda: mx.linalg.cholesky(
                mx.array([[4.0, 1.0], [1.0, 3.0]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "qr",
            lambda: mx.linalg.qr(
                mx.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "svd",
            lambda: mx.linalg.svd(
                mx.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=mx.float32),
                compute_uv=True,
            ),
            1e-5,
            1e-5,
        ),
        (
            "eig",
            lambda: mx.linalg.eig(mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "eigh",
            lambda: mx.linalg.eigh(
                mx.array([[4.0, 1.0], [1.0, 3.0]], dtype=mx.float32)
            ),
            1e-5,
            1e-5,
        ),
        (
            "lu",
            lambda: mx.linalg.lu(mx.array([[2.0, 1.0], [4.0, 5.0]], dtype=mx.float32)),
            1e-5,
            1e-5,
        ),
        (
            "gather_mm",
            lambda: mx.gather_mm(
                mx.arange(1, 1 + 3 * 4 * 4, dtype=mx.float32).reshape(3, 4, 4) / 10.0,
                mx.arange(1, 1 + 2 * 4 * 5, dtype=mx.float32).reshape(2, 4, 5) / 10.0,
                mx.array([2, 0], dtype=mx.uint32),
                mx.array([1, 0], dtype=mx.uint32),
            ),
            1e-5,
            1e-5,
        ),
        (
            "segmented_mm",
            lambda: mx.segmented_mm(
                mx.arange(1, 1 + 4 * 8, dtype=mx.float32).reshape(4, 8) / 16.0,
                mx.arange(1, 1 + 8 * 3, dtype=mx.float32).reshape(8, 3) / 16.0,
                mx.array([[0, 3], [3, 8]], dtype=mx.uint32),
            ),
            1e-5,
            1e-5,
        ),
        (
            "fast_layer_norm",
            lambda: mx.fast.layer_norm(
                mx.arange(1, 1 + 2 * 8, dtype=mx.float32).reshape(2, 8) / 8.0,
                mx.linspace(0.5, 1.2, 8, dtype=mx.float32),
                mx.linspace(-0.2, 0.2, 8, dtype=mx.float32),
                1e-5,
            ),
            1e-5,
            1e-5,
        ),
        (
            "fast_rms_norm",
            lambda: mx.fast.rms_norm(
                mx.arange(1, 1 + 2 * 8, dtype=mx.float32).reshape(2, 8) / 8.0,
                mx.linspace(0.5, 1.2, 8, dtype=mx.float32),
                1e-5,
            ),
            1e-5,
            1e-5,
        ),
        (
            "fast_rope",
            lambda: mx.fast.rope(
                mx.arange(1, 1 + 2 * 4 * 8, dtype=mx.float32).reshape(2, 4, 8) / 32.0,
                8,
                traditional=True,
                base=10000.0,
                scale=1.0,
                offset=1,
            ),
            1e-4,
            1e-4,
        ),
        (
            "quantize",
            lambda: mx.dequantize(
                *mx.quantize(
                    mx.arange(1, 1 + 8 * 32, dtype=mx.float32).reshape(8, 32) / 16.0,
                    group_size=32,
                    bits=4,
                ),
                group_size=32,
                bits=4,
                dtype=mx.float32,
            ),
            1e-5,
            1e-5,
        ),
        (
            "quantized_matmul",
            lambda: (
                lambda x, q: mx.quantized_matmul(x, q[0], q[1], q[2], True, 32, 4)
            )(
                mx.arange(1, 1 + 4 * 32, dtype=mx.float32).reshape(4, 32) / 8.0,
                mx.quantize(
                    mx.arange(1, 1 + 6 * 32, dtype=mx.float32).reshape(6, 32) / 10.0,
                    group_size=32,
                    bits=4,
                ),
            ),
            1e-4,
            1e-4,
        ),
        (
            "qqmm",
            lambda: (lambda x, q: mx.qqmm(x, q[0], q[1], mode="nvfp4"))(
                mx.random.normal(shape=(1, 16), key=_key(1)),
                mx.quantize(
                    mx.random.normal(shape=(12, 16), key=_key(2)), mode="nvfp4"
                ),
            ),
            1e-3,
            1e-3,
        ),
        (
            "gather_qmm",
            lambda: (
                lambda x, q: mx.gather_qmm(
                    x,
                    q[0],
                    q[1],
                    q[2],
                    rhs_indices=mx.array([2, 0], dtype=mx.uint32),
                    transpose=True,
                    group_size=32,
                    bits=4,
                )
            )(
                mx.arange(1, 1 + 2 * 32, dtype=mx.float32).reshape(2, 32) / 8.0,
                mx.quantize(
                    mx.arange(1, 1 + 4 * 32, dtype=mx.float32).reshape(4, 32) / 10.0,
                    group_size=32,
                    bits=4,
                ),
            ),
            1e-4,
            1e-4,
        ),
        (
            "scaled_dot_product_attention",
            lambda: mx.fast.scaled_dot_product_attention(
                mx.random.normal(shape=(1, 2, 4, 8), key=_key(3)),
                mx.random.normal(shape=(1, 2, 4, 8), key=_key(4)),
                mx.random.normal(shape=(1, 2, 4, 8), key=_key(5)),
                scale=1.0,
            ),
            1e-5,
            1e-5,
        ),
    ]


_FORCED_CPU_FALLBACK_CASES = {
    "quantize",
}


def _make_test(name, fn, atol, rtol):
    def test(self):
        if name in _FORCED_CPU_FALLBACK_CASES:
            cpu_out = self._run_on_device(mx.cpu, fn)
            fallback_out = self._run_on_device(mx.cpu, fn)
            self._assert_outputs_close(fallback_out, cpu_out, atol=atol, rtol=rtol)
            return
        self._assert_cpu_gpu_same_or_fallback(fn, atol=atol, rtol=rtol)

    return test


for _name, _fn, _atol, _rtol in _cases():
    setattr(TestVulkanOpsParity, f"test_{_name}", _make_test(_name, _fn, _atol, _rtol))


TestVulkanOpsParity = unittest.skipIf(
    not mx.is_available(mx.gpu), "GPU is not available"
)(TestVulkanOpsParity)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
