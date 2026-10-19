"""Compare fused Gluon norm/quant with an independent vLLM-style Triton norm."""

import os
import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.common.gated_rmsnorm_fp8 import gated_rmsnorm_fp8


# Norm expression and launch schedule adapted from vLLM's Apache-2.0
# third_party/flash_linear_attention/ops/layernorm_guard.py.
# Copyright contributors to vLLM and Tri Dao.
@triton.jit
def _reference_norm(
    X,
    Z,
    W,
    B,
    Y,
    Rstd,
    M,
    SX: tl.constexpr,
    SZ: tl.constexpr,
    SY: tl.constexpr,
    BIAS: tl.constexpr,
    SILU: tl.constexpr,
    EPS,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    cols = tl.arange(0, 128)
    off = cols[None, :] + group * 128
    mask = rows[:, None] < M
    x = tl.load(X + rows[:, None] * SX + off, mask, 0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / 128
    rstd = tl.rsqrt(var + EPS)
    tl.store(Rstd + group * M + rows, rstd, rows < M)
    w = tl.load(W + cols + group * 128).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    if BIAS:
        y += tl.load(B + cols + group * 128).to(tl.float32)[None, :]
    z = tl.load(Z + rows[:, None] * SZ + off, mask, 0).to(tl.float32)
    gate = z * tl.sigmoid(z) if SILU else tl.sigmoid(z)
    y *= gate
    tl.store(Y + rows[:, None] * SY + off, y, mask)


def reference(x, z, w, b=None, activation="silu"):
    shared = w.numel() == 128
    xx = x.reshape(-1, 128) if shared else x
    zz = z.reshape_as(xx)
    out = torch.empty(xx.shape, device=x.device, dtype=x.dtype)
    m, n = xx.shape
    if m:
        sm = torch.cuda.get_device_properties(x.device).multi_processor_count
        bt = min(triton.next_power_of_2(triton.cdiv(m, 2 * sm)), 4)
        rstd = torch.empty((m * (n // 128),), device=x.device)
        _reference_norm[(triton.cdiv(m, bt), n // 128)](
            xx,
            zz,
            w,
            b,
            out,
            rstd,
            m,
            xx.stride(0),
            zz.stride(0),
            out.stride(0),
            b is not None,
            activation == "silu",
            1e-6,
            bt,
            num_warps=1,
        )
    return out.reshape_as(x)


def assert_bits(test, a, b):
    test.assertEqual(a.dtype, b.dtype)
    test.assertEqual(a.shape, b.shape)
    aa = a.flatten().clone(memory_format=torch.contiguous_format).view(torch.uint8)
    bb = b.flatten().clone(memory_format=torch.contiguous_format).view(torch.uint8)
    test.assertTrue(torch.equal(aa, bb))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GatedRmsNormFp8VllmTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(247)

    def inputs(self, m=257, n=8192, shared=True, strided=False):
        x = torch.randn((m, n + 32), device="cuda", dtype=torch.bfloat16)[:, :n]
        z = torch.randn((m, n + 128), device="cuda", dtype=torch.bfloat16)[:, 128:]
        if not strided:
            x, z = x.contiguous(), z.contiguous()
        w = torch.randn((128 if shared else n,), device="cuda", dtype=torch.bfloat16)
        return x, z, w

    def compare(self, x, z, w, bias=None, activation="silu", ue=True):
        expected_y = reference(x, z, w, bias, activation)
        expected_q, expected_s = sgl_per_token_group_quant_fp8(
            expected_y,
            128,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=ue,
        )
        observed = torch.empty_like(expected_y)
        q, s = gated_rmsnorm_fp8(
            x,
            z,
            w,
            bias,
            activation=activation,
            vllm_math=True,
            scale_ue8m0=ue,
            out_y=observed,
        )
        assert_bits(self, observed, expected_y)
        assert_bits(self, q, expected_q)
        assert_bits(self, s, expected_s)
        q0, s0 = gated_rmsnorm_fp8(
            x,
            z,
            w,
            bias,
            activation=activation,
            vllm_math=True,
            scale_ue8m0=ue,
        )
        assert_bits(self, q0, q)
        assert_bits(self, s0, s)

    def test_shared_rows(self):
        for m in (1, 5, 257, 8192):
            with self.subTest(m=m):
                self.compare(*self.inputs(m))

    def test_full_weight_bias_sigmoid_strides(self):
        x, z, w = self.inputs(shared=False, strided=True)
        for ue in (False, True):
            self.compare(x, z, w, torch.randn_like(w), "sigmoid", ue)

    def test_empty_and_singleton_scales(self):
        for m in (0, 1, 3):
            self.compare(*self.inputs(m, 512))

    def test_zero_small_and_large_gates(self):
        x, z, w = self.inputs(17)
        x[0] = 0
        x[1] *= 1e-6
        z[2, ::2] = 80
        z[2, 1::2] = -80
        self.compare(x, z, w)

    def test_dynamic_graph_operands(self):
        x, z, w = self.inputs(17)
        for _ in range(3):
            gated_rmsnorm_fp8(x, z, w, vllm_math=True, scale_ue8m0=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gated_rmsnorm_fp8(x, z, w, vllm_math=True, scale_ue8m0=True)
        for _ in range(3):
            x.normal_()
            z.normal_()
            w.normal_()
            y = reference(x, z, w)
            expected = sgl_per_token_group_quant_fp8(
                y,
                128,
                eps=1e-10,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            graph.replay()
            assert_bits(self, actual[0], expected[0])
            assert_bits(self, actual[1], expected[1])

    def test_decode_dispatch_without_bf16_allocation(self):
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.rmsnorm_gated_fp8_quant import (
            maybe_rmsnorm_gated_fp8_quant,
        )

        x, z, w = self.inputs(3)
        with patch.dict(
            os.environ,
            {"RTP_LENGTH_ALIGN_GATED_NORM": "1", "RTP_QWEN35_DECODE_FUSION": "1"},
        ):
            # Optional diagnostic output is caller-owned; production returns None.
            with patch.object(
                torch, "empty_like", side_effect=AssertionError("BF16 allocation")
            ):
                y, q, s = maybe_rmsnorm_gated_fp8_quant(x, z, w)
            self.assertIsNone(y)
            expected = gated_rmsnorm_fp8(x, z, w, vllm_math=True, scale_ue8m0=True)
            assert_bits(self, q, expected[0])
            assert_bits(self, s, expected[1])

    def test_floor_independent_of_norm_switch(self):
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.rmsnorm_gated_fp8_quant import (
            maybe_rmsnorm_gated_fp8_quant,
        )

        x, z, w = self.inputs(3, 512)
        x.zero_()
        for enabled in ("0", "1"):
            with patch.dict(
                os.environ,
                {
                    "RTP_LENGTH_ALIGN_GATED_NORM": enabled,
                    "RTP_QWEN35_DECODE_FUSION": "1",
                },
            ):
                for floor in (1e-10, 1e-4):
                    expected = sgl_per_token_group_quant_fp8(
                        reference(x, z, w),
                        128,
                        eps=floor,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    # The omitted argument must select the new 1e-10 default.
                    kwargs = {} if floor == 1e-10 else {"amax_floor": floor}
                    actual = gated_rmsnorm_fp8(x, z, w, scale_ue8m0=True, **kwargs)
                    assert_bits(self, actual[0], expected[0])
                    assert_bits(self, actual[1], expected[1])
                    _, q, s = maybe_rmsnorm_gated_fp8_quant(x, z, w, quant_eps=floor)
                    assert_bits(self, q, expected[0])
                    assert_bits(self, s, expected[1])

    def test_bad_observation_buffer(self):
        x, z, w = self.inputs(3)
        with self.assertRaises(ValueError):
            gated_rmsnorm_fp8(x, z, w, vllm_math=True, out_y=x.float())


if __name__ == "__main__":
    unittest.main()
