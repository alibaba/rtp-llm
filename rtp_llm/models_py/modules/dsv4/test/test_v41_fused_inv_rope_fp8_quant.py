"""V4.1 group32 output projection: eager numeric boundary and graph replay."""

import os
import unittest
from unittest.mock import patch

import deep_gemm
import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_output_projection import (
    grouped_output_projection,
    is_supported,
    quantization_scale_min,
)
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb


def eager_quant(o, freqs, group_size=32):
    """Actual eager V4.1 RoPE and eight independently dispatched quantizers."""
    source = o.reshape(-1, 64, 512).clone()
    rows = source.shape[0]
    repeated_freqs = freqs.repeat_interleave(rows // freqs.shape[0], dim=0)
    apply_rotary_emb(source[..., -64:].unsqueeze(0), repeated_freqs, inverse=True)
    grouped = source.reshape(rows, 8, 4096)
    quantized, scales = [], []
    for group in range(8):
        q, s = sgl_per_token_group_quant_fp8(
            grouped[:, group].contiguous(),
            group_size=group_size,
            eps=torch.finfo(torch.float32).tiny,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        quantized.append(q)
        scales.append(s)
    return torch.stack(quantized, dim=1), torch.stack(scales, dim=1)


def fused_quant(o, freqs, *, impl="optimized", group_size=32):
    rows = o.numel() // (64 * 512)
    return fused_inv_rope_fp8_quant(
        o,
        freqs,
        n_groups=8,
        heads_per_group=8,
        nope_dim=448,
        rope_head_dim=64,
        quant_group_size=group_size,
        eps=torch.finfo(torch.float32).tiny,
        round_rope_to_input_dtype=True,
        scale_min=quantization_scale_min(rows, 4096),
        impl=impl,
    )


def make_inputs(rows, seed=0):
    torch.manual_seed(seed)
    o = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.3
    angle = torch.rand(rows, 32, device="cuda") * 6.28
    freqs = torch.polar(torch.ones_like(angle), angle)
    return o, freqs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41OutputProjectionTest(unittest.TestCase):
    def assert_quant_equal(self, expected, actual):
        for ref, got in zip(expected, actual):
            self.assertEqual(ref.shape, got.shape)
            torch.testing.assert_close(
                got.contiguous().view(torch.uint8),
                ref.contiguous().view(torch.uint8),
                rtol=0,
                atol=0,
            )

    def assert_quant_boundary_close(self, expected, actual):
        # The eager complex-multiply and fused real arithmetic can straddle a
        # BF16 midpoint after cancellation. Both materialize BF16 before FP8;
        # admit only one FP8 code at <1 ppm, with identical scale bytes.
        ref, got = expected[0], actual[0]
        delta = (
            ref.contiguous().view(torch.uint8).to(torch.int16)
            - got.contiguous().view(torch.uint8).to(torch.int16)
        ).abs()
        self.assertLessEqual(delta.max().item(), 1)
        self.assertLessEqual((delta != 0).sum().item(), max(1, ref.numel() // 1000000))
        torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)

    def test_group32_actual_quantizer_and_layout(self):
        for backend in ("legacy", "v2", "auto"):
            for rows in (1, 6, 24, 128, 1024):
                with self.subTest(backend=backend, rows=rows), patch.dict(
                    os.environ, {"DSV4_FP8_QUANT_KERNEL": backend}
                ):
                    o, freqs = make_inputs(rows, rows)
                    result = fused_quant(o, freqs)
                    self.assert_quant_boundary_close(eager_quant(o, freqs), result)
                    self.assertEqual(result[0].shape, (rows, 8, 4096))
                    self.assertEqual(result[1].shape, (rows, 8, 32))
                    self.assertEqual(result[0].stride(-1), 1)
                    self.assertEqual(result[1].stride(0), 1)
                    self.assertEqual(result[1].stride(-1), (rows + 3) // 4 * 4)

    def test_zero_tiny_scale_boundaries(self):
        o, freqs = make_inputs(6, 19)
        # Include zero, subnormal/tiny values, and exact UE8M0 scale transitions.
        values = torch.tensor(
            [0, 1e-35, 1e-10, 1e-8, 4.48e-8, 1e-4, 0.875, 1.75, 448, 896],
            device="cuda",
            dtype=torch.bfloat16,
        )
        o.flatten().copy_(values.repeat((o.numel() + 9) // 10)[: o.numel()])
        o[0].zero_()
        o[1].fill_(1e-35)
        for backend in ("legacy", "v2"):
            with self.subTest(backend=backend), patch.dict(
                os.environ, {"DSV4_FP8_QUANT_KERNEL": backend}
            ):
                self.assert_quant_equal(eager_quant(o, freqs), fused_quant(o, freqs))

    def test_deterministic_cancellation_preserves_eager_rounding(self):
        # Explicit inputs: separate FP32 products land exactly on BF16
        # midpoints, then on FP8 midpoints. No device RNG is involved.
        o = torch.full((2, 64, 512), 0.5, device="cuda", dtype=torch.bfloat16)
        o[1].fill_(1.0)
        o[:, 0, 448:450] = torch.tensor(
            [[-0.326171875, -0.51953125], [-0.302734375, -0.828125]],
            device="cuda",
            dtype=torch.bfloat16,
        )
        freqs = torch.ones((2, 32), device="cuda", dtype=torch.complex64)
        freqs[:, 0] = torch.complex(
            torch.tensor([0.5317438840866089, -0.34328246116638184], device="cuda"),
            torch.tensor([0.8469052314758301, -0.9392322301864624], device="cuda"),
        )
        for backend in ("legacy", "v2", "auto"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": backend}):
                expected = eager_quant(o, freqs)
                self.assertEqual(
                    expected[0].view(torch.uint8)[:, 0, 449].tolist(), [133, 135]
                )
                for impl in ("legacy", "optimized"):
                    with self.subTest(backend=backend, impl=impl):
                        self.assert_quant_equal(
                            expected, fused_quant(o, freqs, impl=impl)
                        )

    def test_signed_zero_all_input_signs_and_frequency_quadrants(self):
        o = torch.zeros((4, 64, 512), device="cuda", dtype=torch.bfloat16)
        # Four head patterns cover (++), (+-), (-+), (--) input zero signs.
        o[:, 1::4, 449::2] = -0.0
        o[:, 2::4, 448::2] = -0.0
        o[:, 3::4, 448:] = -0.0
        freqs = torch.complex(
            torch.tensor([-0.6, -0.6, 0.6, 0.6], device="cuda")[:, None].expand(4, 32),
            torch.tensor([0.8, -0.8, 0.8, -0.8], device="cuda")[:, None].expand(4, 32),
        )
        for backend in ("legacy", "v2", "auto"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": backend}):
                expected = eager_quant(o, freqs)
                self.assertEqual(
                    set(expected[0].view(torch.uint8).unique().tolist()), {0, 128}
                )
                for impl in ("legacy", "optimized"):
                    with self.subTest(backend=backend, impl=impl):
                        self.assert_quant_equal(
                            expected, fused_quant(o, freqs, impl=impl)
                        )

    def test_noncontiguous_and_batched_freqs(self):
        o, freqs = make_inputs(24, 2)
        storage = torch.zeros(24, 72, 520, device="cuda", dtype=o.dtype)
        sliced = storage[:, :64, :512]
        sliced.copy_(o)
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}):
            self.assert_quant_equal(
                eager_quant(sliced, freqs), fused_quant(sliced, freqs)
            )
            batched = o.view(4, 6, 64, 512)
            self.assert_quant_equal(
                eager_quant(batched, freqs[:4]), fused_quant(batched, freqs[:4])
            )
            self.assert_quant_equal(
                eager_quant(batched, freqs), fused_quant(batched, freqs)
            )

    def test_legacy_triton_group32_and_group128(self):
        o, freqs = make_inputs(6, 6)
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}):
            for group_size in (32, 128):
                self.assert_quant_equal(
                    eager_quant(o, freqs, group_size),
                    fused_quant(o, freqs, impl="legacy", group_size=group_size),
                )

    def test_graph_replay_changes_inputs_and_freqs(self):
        for rows in (1, 6, 24):
            with self.subTest(rows=rows), patch.dict(
                os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}
            ):
                o, freqs = make_inputs(rows, 3)
                fused_quant(o, freqs)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fused_quant(o, freqs)
                for seed in (9, 22):
                    next_o, next_freqs = make_inputs(rows, seed)
                    o.copy_(next_o)
                    freqs.copy_(next_freqs)
                    graph.replay()
                    self.assert_quant_equal(eager_quant(o, freqs), output)

    def test_grouped_gemm_and_full_projection_graph(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("MXFP8 grouped DeepGEMM requires SM100")
        torch.manual_seed(11)
        weight = (
            torch.randn(8, 1024, 4096, device="cuda", dtype=torch.bfloat16) * 0.3
        ).to(torch.float8_e4m3fn)
        raw_scale = torch.full((8, 1024, 128), 0.0625, device="cuda")
        scale = get_mn_major_tma_aligned_packed_ue8m0_tensor(raw_scale)
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}):
            for rows in (1, 6, 24, 128):
                o, freqs = make_inputs(rows, 33)
                self.assertTrue(is_supported(o, freqs, weight, scale))
                q, s = eager_quant(o, freqs)
                ref_groups = []
                for group in range(8):
                    ref = torch.empty(rows, 1024, device="cuda", dtype=o.dtype)
                    # Repack the reference scales because stack changed strides.
                    ref_scale = get_mn_major_tma_aligned_packed_ue8m0_tensor(
                        raw_scale[group]
                    )
                    q_group, s_group = q[:, group].contiguous(), s[:, group]
                    tma = (rows + 3) // 4 * 4
                    packed = torch.empty(32 * tma, device="cuda", dtype=torch.int32)
                    s_input = packed.as_strided((rows, 32), (1, tma))
                    s_input.copy_(s_group)
                    deep_gemm.fp8_fp4_gemm_nt(
                        (q_group, s_input),
                        (weight[group], ref_scale),
                        ref,
                        recipe=(1, 1, 32),
                    )
                    ref_groups.append(ref)
                expected = torch.stack(ref_groups, dim=1).flatten(1)
                actual = grouped_output_projection(o, freqs, weight, scale)
                torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.001)
                # CP4/TP4 prefill has two local output groups and 16 heads.
                tp4 = grouped_output_projection(o[:, :16], freqs, weight[:2], scale[:2])
                torch.testing.assert_close(
                    tp4, expected[:, :2048], rtol=0.016, atol=0.001
                )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = grouped_output_projection(o, freqs, weight, scale)
                next_o, next_freqs = make_inputs(rows, 44)
                o.copy_(next_o)
                freqs.copy_(next_freqs)
                graph.replay()
                expected_replay = grouped_output_projection(o, freqs, weight, scale)
                torch.testing.assert_close(captured, expected_replay, rtol=0, atol=0)
                self.assertFalse(is_supported(o[:0], freqs[:0], weight, scale))
                self.assertFalse(is_supported(o.float(), freqs, weight, scale))
            invalid = torch.empty(4, 6, 64, 512, device="cuda", dtype=torch.bfloat16)
            self.assertFalse(is_supported(invalid[::2], freqs[:2], weight, scale))
            self.assertFalse(is_supported(invalid, freqs[:3], weight, scale))

    def test_quant_backend_dispatch_boundary(self):
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "auto"}):
            self.assertEqual(quantization_scale_min(1023, 4096), 1e-10)
            self.assertEqual(quantization_scale_min(1024, 4096), 0)


if __name__ == "__main__":
    unittest.main()
