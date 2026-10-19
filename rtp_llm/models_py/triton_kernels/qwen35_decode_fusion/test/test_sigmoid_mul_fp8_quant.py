"""Equivalence test for fused ``attn * sigmoid(gate)`` + UE8M0 FP8 quant.

Compares ``sigmoid_mul_fp8_quant`` against an FP32 gate/product reference::

    y = (attn.float() * torch.sigmoid(gate.float())).to(attn.dtype)
    fp8, scale = sgl_per_token_group_quant_fp8(
        y, 128, eps=1e-10,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )

FP8 must stay within 1 ULP; packed UE8M0 scale layout must match.
"""

from __future__ import annotations

import os
import unittest

import torch


def _fp8_ulp_diff(ref: torch.Tensor, got: torch.Tensor) -> int:
    ref_u = ref.view(torch.uint8).to(torch.int32)
    got_u = got.view(torch.uint8).to(torch.int32)
    if ref_u.numel() == 0:
        return 0
    return int((ref_u - got_u).abs().max().item())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SigmoidMulFp8QuantTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"

    @classmethod
    def setUpClass(cls):
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.sigmoid_mul_fp8_quant import (
            is_supported,
            maybe_sigmoid_mul_fp8_quant,
            sigmoid_mul_fp8_quant,
        )

        cls._fused = staticmethod(sigmoid_mul_fp8_quant)
        cls._maybe = staticmethod(maybe_sigmoid_mul_fp8_quant)
        cls._is_supported = staticmethod(is_supported)
        cls._sgl = None
        try:
            from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
                sgl_per_token_group_quant_fp8,
            )

            cls._sgl = staticmethod(sgl_per_token_group_quant_fp8)
        except Exception:
            cls._sgl = None

    @staticmethod
    def _torch_ue8m0_quant(y: torch.Tensor):
        """sgl_per_token_group_quant_fp8 UE8M0 layout, torch-only fallback."""
        from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fp8_scale import (
            make_ue8m0_scale_like,
        )

        group_size = 128
        m, n = y.shape
        y_f = y.to(torch.float32)
        num_groups = n // group_size
        finfo = torch.finfo(torch.float8_e4m3fn)
        if m == 0:
            out_q = torch.empty((m, n), device=y.device, dtype=torch.float8_e4m3fn)
            return out_q, make_ue8m0_scale_like(y.shape, device=y.device)
        grouped = y_f.view(m, num_groups, group_size)
        absmax = grouped.abs().amax(dim=-1).clamp_min(1e-10)
        raw_scale = (absmax / float(finfo.max)).clamp_min(1e-10)
        exponent = torch.ceil(torch.log2(raw_scale))
        scale = torch.exp2(exponent)
        out_q = (
            torch.clamp(
                grouped / scale.unsqueeze(-1), float(finfo.min), float(finfo.max)
            )
            .to(torch.float8_e4m3fn)
            .view(m, n)
        )
        exp_biased = (exponent + 127.0).clamp(0, 255).to(torch.int32)
        num_packed = (num_groups + 3) // 4
        if num_groups % 4 != 0:
            exp_biased = torch.nn.functional.pad(exp_biased, (0, 4 - num_groups % 4))
        packed = exp_biased.view(m, num_packed, 4)
        rowmajor = (
            packed[:, :, 0]
            | (packed[:, :, 1] << 8)
            | (packed[:, :, 2] << 16)
            | (packed[:, :, 3] << 24)
        )
        out_s = make_ue8m0_scale_like(y.shape, device=y.device)
        out_s.copy_(rowmajor)
        return out_q, out_s

    @staticmethod
    def _ref(attn: torch.Tensor, gate: torch.Tensor):
        y = (attn.float() * torch.sigmoid(gate.float())).to(attn.dtype)
        if SigmoidMulFp8QuantTest._sgl is not None:
            return SigmoidMulFp8QuantTest._sgl(
                y,
                128,
                eps=1e-10,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        return SigmoidMulFp8QuantTest._torch_ue8m0_quant(y)

    def _check(
        self,
        attn: torch.Tensor,
        gate: torch.Tensor,
        *,
        msg: str,
    ):
        ref_q, ref_s = self._ref(attn, gate)
        got_q, got_s = self._fused(attn, gate)

        self.assertEqual(got_q.shape, attn.shape, msg)
        self.assertEqual(got_q.dtype, torch.float8_e4m3fn, msg)
        self.assertEqual(got_s.dtype, torch.int32, msg)
        self.assertEqual(tuple(got_s.shape), tuple(ref_s.shape), f"{msg}: scale shape")
        self.assertEqual(got_s.stride(), ref_s.stride(), f"{msg}: scale stride")

        ulp = _fp8_ulp_diff(ref_q, got_q)
        self.assertLessEqual(ulp, 1, f"{msg}: FP8 ULP={ulp}")

        if got_s.numel() == 0:
            return
        # N=128 has 1 group / 4 packed bytes; sgl may leave the unused
        # bytes untouched (empty alloc). Compare only live groups.
        groups = attn.shape[-1] // 128
        live_bytes = (groups + 3) // 4 * 4
        if groups % 4 == 0:
            scale_mismatch = int((got_s != ref_s).sum().item())
            self.assertEqual(
                scale_mismatch,
                0,
                f"{msg}: UE8M0 scale mismatch {scale_mismatch}/{got_s.numel()}",
            )
        else:
            mask = (1 << (8 * (groups % 4))) - 1
            ref_live = ref_s.to(torch.int32) & mask
            got_live = got_s.to(torch.int32) & mask
            scale_mismatch = int((got_live != ref_live).sum().item())
            self.assertEqual(
                scale_mismatch,
                0,
                f"{msg}: live UE8M0 scale mismatch {scale_mismatch}/{got_s.numel()} "
                f"(compared low {groups % 4} bytes; {live_bytes=})",
            )

    def _rand(
        self, m: int, n: int, scale: float = 2.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn = torch.randn(m, n, dtype=torch.bfloat16, device="cuda") * scale
        gate = torch.randn(m, n, dtype=torch.bfloat16, device="cuda") * scale
        return attn.contiguous(), gate.contiguous()

    def test_typical_m(self):
        for m in (1, 8, 32, 128, 256):
            attn, gate = self._rand(m, 8192)
            self._check(attn, gate, msg=f"M={m} N=8192")

    def test_m0(self):
        attn = torch.empty(0, 8192, dtype=torch.bfloat16, device="cuda")
        gate = torch.empty(0, 8192, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(self._is_supported(attn, gate))
        self._check(attn, gate, msg="M=0 N=8192")

    def test_m3_tma_padding(self):
        attn, gate = self._rand(3, 8192)
        self._check(attn, gate, msg="M=3 N=8192")

    def test_n128(self):
        attn, gate = self._rand(16, 128)
        self._check(attn, gate, msg="M=16 N=128")

    def test_gate_zero(self):
        m, n = 32, 8192
        attn = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        gate = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
        self._check(attn, gate, msg="gate=0")

    def test_gate_large(self):
        m, n = 8, 8192
        attn = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        for value in (20.0, -20.0):
            gate = torch.full((m, n), value, dtype=torch.bfloat16, device="cuda")
            self._check(attn, gate, msg=f"gate={value}")

    def test_fp32_gate_rounding_boundary(self):
        # The anchor fixes every group's scale at 1. Rounding sigmoid to BF16
        # first gives product 0.453125 / FP8 byte 46; rounding the FP32 product
        # only gives 0.455078125 / FP8 byte 47.
        attn = torch.full((3, 512), 0.515625, dtype=torch.bfloat16, device="cuda")
        gate = torch.full_like(attn, 2.0)
        attn[:, ::128] = 512.0
        gate[:, ::128] = 0.0
        ref_q, ref_s = self._ref(attn, gate)
        old_q, old_s = self._torch_ue8m0_quant(attn * torch.sigmoid(gate))
        got_q, got_s = self._fused(attn, gate)
        self.assertEqual(int(ref_q.view(torch.uint8)[0, 1]), 47)
        self.assertEqual(int(old_q.view(torch.uint8)[0, 1]), 46)
        torch.testing.assert_close(
            got_q.view(torch.uint8), ref_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(got_s, ref_s, rtol=0, atol=0)
        torch.testing.assert_close(old_s, ref_s, rtol=0, atol=0)

    def test_cuda_graph_replay_with_preallocated_outputs(self):
        attn, gate = self._rand(3, 512)
        out_q, out_s = self._fused(attn, gate)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._fused(attn, gate, out_q=out_q, out_scale=out_s)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self._fused(attn, gate, out_q=out_q, out_scale=out_s)
        for core_value, gate_value in (
            (0.515625, 0.0),
            (0.515625, 2.0),
            (0.515625, -2.0),
            (0.0, 0.0),
            (1e-12, 0.0),
            (1e-6, 0.0),
        ):
            attn.fill_(core_value)
            gate.fill_(gate_value)
            if core_value >= 0.5:
                attn[:, ::128] = 512.0
                gate[:, ::128] = 0.0
            graph.replay()
            ref_q, ref_s = self._ref(attn, gate)
            torch.testing.assert_close(
                out_q.view(torch.uint8), ref_q.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(out_s, ref_s, rtol=0, atol=0)

    def test_zero_and_small_inputs_match_native_scale_floor(self):
        for value in (0.0, 1e-12, 1e-8, 1e-6):
            with self.subTest(value=value):
                attn = torch.full((3, 512), value, dtype=torch.bfloat16, device="cuda")
                gate = torch.zeros_like(attn)
                ref_q, ref_s = self._ref(attn, gate)
                got_q, got_s = self._fused(attn, gate)
                torch.testing.assert_close(
                    got_q.view(torch.uint8), ref_q.view(torch.uint8), rtol=0, atol=0
                )
                torch.testing.assert_close(got_s, ref_s, rtol=0, atol=0)
                if value <= 1e-12:
                    self.assertTrue(torch.all(got_s == 0x5E5E5E5E))

    def test_o_proj_floor_is_scoped_to_existing_deepgemm_instance(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from rtp_llm.models_py.modules.factory import LinearFactory
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )
        from rtp_llm.models_py.modules.hybrid.causal_attention import CausalAttention

        def quantizer():
            linear = CudaFp8DeepGEMMLinear.__new__(CudaFp8DeepGEMMLinear)
            torch.nn.Module.__init__(linear)
            linear.K = 512
            linear.scale_ue8m0 = True
            linear.cached_scales = None
            linear.cached_scales_max_len = 0
            return linear

        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                qkv, output, unrelated = quantizer(), quantizer(), quantizer()
                receiver = output
                if wrapped:
                    receiver = CudaFp8GEMMLinear.__new__(CudaFp8GEMMLinear)
                    torch.nn.Module.__init__(receiver)
                    receiver._deepgemm_linear = output
                with patch.object(
                    LinearFactory,
                    "create_linear_from_weights",
                    side_effect=[qkv, receiver],
                ), patch.object(CudaFp8DeepGEMMLinear, "maybe_cache_quant_scale"):
                    attention = CausalAttention(
                        SimpleNamespace(head_num=8, kv_head_num=2, size_per_head=64),
                        SimpleNamespace(get_attn_tp_size=lambda: 1),
                        {},
                        1e-6,
                    )
                self.assertIs(attention.o_proj, receiver)
                self.assertEqual(output.input_quant_eps, 1e-10)
                self.assertEqual(qkv.input_quant_eps, 1e-4)
                self.assertEqual(unrelated.input_quant_eps, 1e-4)
                x = torch.full((3, 512), 1e-6, dtype=torch.bfloat16, device="cuda")
                selected_q, selected_s = output.quantize_input(x)
                ref_q, ref_s = self._torch_ue8m0_quant(x)
                torch.testing.assert_close(
                    selected_q.view(torch.uint8),
                    ref_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(selected_s, ref_s, rtol=0, atol=0)
                _, old_s = qkv.quantize_input(x)
                self.assertFalse(torch.equal(selected_s, old_s))

    def test_unsupported_returns_none(self):
        attn, gate = self._rand(4, 8192)
        self.assertIsNotNone(self._maybe(attn, gate))

        cpu = attn.cpu()
        self.assertIsNone(self._maybe(cpu, gate.cpu()))

        mismatched = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(self._maybe(attn, mismatched))

        fp32 = attn.float()
        self.assertIsNone(self._maybe(fp32, gate.float()))

        n127 = torch.randn(4, 127, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(self._maybe(n127, n127))

        non_contig = attn[:, :4096]
        self.assertFalse(non_contig.is_contiguous())
        self.assertIsNone(self._maybe(non_contig, non_contig))

        os.environ["RTP_QWEN35_DECODE_FUSION"] = "0"
        try:
            self.assertFalse(self._is_supported(attn, gate))
            self.assertIsNone(self._maybe(attn, gate))
        finally:
            os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"


if __name__ == "__main__":
    unittest.main()
