import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from flashinfer import fp4_quantize

import rtp_llm.models_py.kernels.cuda.modelopt_fp4_quant as quant
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
    CudaFp4GEMMLinear,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class ModelOptFp4ProducerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1548)
        self.enabled = patch.object(quant, "_FUSED_PRODUCER_ENABLED", True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)
        self.mask = torch.full(
            (1,), torch.iinfo(torch.int32).max, device="cuda", dtype=torch.int32
        )
        self.act = FusedSiluAndMul()

    def assert_quant_equal(self, x, scale):
        expected_q, expected_s = fp4_quantize(self.act(x), scale)
        q, s = quant.silu_fp4_quant(x, scale, self.mask)
        self.assertTrue(torch.equal(q, expected_q))
        m, two_k = x.shape
        k = two_k // 2
        i = torch.arange(m, device=x.device)[:, None]
        j = torch.arange(k // 16, device=x.device)[None, :]
        # Padding rows have no defined values and are never GEMM operands.
        offsets = (
            (i // 128) * (k // 64 * 512)
            + (j // 4) * 512
            + (i % 32) * 16
            + ((i % 128) // 32) * 4
            + j % 4
        )
        self.assertTrue(
            torch.equal(s.flatten()[offsets], expected_s.flatten()[offsets])
        )

    def test_quant_bytes_and_scale_layout(self):
        for k in (8704, 17408):
            for m in (1, 3, 17, 64, 257):
                for value in (0.125, 3.7, 32.0):
                    with self.subTest(k=k, m=m, scale=value):
                        x = (
                            torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
                            * 2
                        )
                        scale = torch.tensor([value], device="cuda")
                        self.assert_quant_equal(x, scale)

    def test_zero_and_small_values(self):
        x = torch.zeros(3, 17408, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([32.0], device="cuda")
        self.assert_quant_equal(x, scale)
        x[:, :8704] = torch.linspace(-80, 80, 8704, device="cuda")
        x[:, 8704:] = 0.0001
        self.assert_quant_equal(x, scale)

    def test_graph_replay_reads_current_input_and_scale(self):
        x = torch.randn(17, 17408, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([32.0], device="cuda")
        quant.silu_fp4_quant(x, scale, self.mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, s = quant.silu_fp4_quant(x, scale, self.mask)
        x.normal_()
        scale.fill_(3.7)
        graph.replay()
        expected_q, expected_s = quant.silu_fp4_quant(x, scale, self.mask)
        self.assertTrue(torch.equal(q, expected_q))
        # All scales in the first 16-row swizzle group belong to row 0.
        self.assertTrue(torch.equal(s.flatten()[:4], expected_s.flatten()[:4]))
        graph.reset()

    def test_projection_matches_old_silu_quant_gemm(self):
        k, n = 8704, 5120
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        wscale = torch.tensor([32.0], device="cuda")
        wq, ws = fp4_quantize(weight, wscale)
        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "1"}):
            layer = CudaFp4GEMMLinear(
                wq,
                ws,
                torch.tensor([1 / 32], device="cuda"),
                bias=torch.randn(n, device="cuda", dtype=torch.bfloat16),
                weight_scale_2=1 / wscale,
            )
        for m in (1, 3, 5, 10, 17, 20, 40):
            x = torch.randn(m, k * 2, device="cuda", dtype=torch.bfloat16)
            layer.input_scale_inv = torch.tensor([1 / 0.01713], device="cuda")
            layer.alpha = torch.tensor([0.01713 / 32], device="cuda")
            layer._decode_auto_backend = False
            reference = layer(self.act(x))
            layer._decode_auto_backend = True
            torch.testing.assert_close(
                layer.forward_with_silu(x), reference, rtol=0, atol=0
            )

    def test_unsupported_and_disabled_fallback(self):
        x = torch.randn(1, 17408, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([32.0], device="cuda")
        for unsupported in (x.float(), x[:, :8704], x.expand(2, -1), x[:0]):
            self.assertIsNone(quant.silu_fp4_quant(unsupported, scale, self.mask))
        with patch.object(quant, "_FUSED_PRODUCER_ENABLED", False):
            self.assertIsNone(quant.silu_fp4_quant(x, scale, self.mask))
        self.assertIsNone(quant.silu_fp4_quant(x, scale, None))

    def test_large_dense_input_retains_original_producer(self):
        width = 17408
        rows = (2**31 + width - 1) // width
        x = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
        scale = torch.ones(1, device="cuda", dtype=torch.float32)
        self.assertIsNone(quant.silu_fp4_quant(x, scale, self.mask))

    def test_ar_norm_unsupported_inputs_preserve_tensors(self):
        x = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        gamma = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
        scale = torch.ones(1, device="cuda")
        proxy = SimpleNamespace(is_supported=lambda tensor: True)
        before_x, before_r = x.clone(), r.clone()
        self.assertIsNone(quant.residual_norm_fp4_quant(x, r, gamma, scale, 1e-6))
        for bad in (x.float(), x[:0], x[:1], x[:3], x[:, ::2]):
            self.assertIsNone(
                quant.residual_norm_fp4_quant(bad, r, gamma, scale, 1e-6, proxy)
            )
        self.assertIsNone(
            quant.residual_norm_fp4_quant(x, x, gamma, scale, 1e-6, proxy)
        )
        with patch.object(quant, "_FUSED_PRODUCER_ENABLED", False):
            self.assertIsNone(
                quant.residual_norm_fp4_quant(x, r, gamma, scale, 1e-6, proxy)
            )
        self.assertTrue(torch.equal(x, before_x))
        self.assertTrue(torch.equal(r, before_r))

    def test_up_projection_matches_cutlass(self):
        k, n = 5120, 17408
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        wscale = torch.tensor([31.7], device="cuda")
        wq, ws = fp4_quantize(weight, wscale)
        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "1"}):
            layer = CudaFp4GEMMLinear(
                wq,
                ws,
                torch.tensor([0.01713], device="cuda"),
                weight_scale_2=1 / wscale,
            )
        if not layer._decode_cudnn_backend:
            self.skipTest("requires cuDNN 9.15 or newer")
        for m in (5, 10, 17, 20, 32):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            layer._decode_cudnn_backend = False
            reference = layer(x)
            layer._decode_cudnn_backend = True
            torch.testing.assert_close(layer(x), reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
