import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant import static_fp8_quant_rows
from rtp_llm.models_py.kernels.cuda.sm120_fp8_skinny import skinny_fp8_mm
from rtp_llm.models_py.modules.factory.linear.impl.cuda.modelopt_fp8_linear import (
    CudaModelOptFp8Linear,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class Sm120Fp8SkinnyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1548)

    def operands(self, m, n=3072, k=5120):
        x = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).T
        a = torch.linspace(0.017, 0.033, m, device="cuda").reshape(m, 1)
        b = torch.linspace(0.011, 0.029, n, device="cuda").reshape(1, n)
        return x, w, a, b

    def reference(self, x, w, a, b):
        return torch._scaled_mm(
            x,
            w,
            scale_a=a,
            scale_b=b.expand(1, w.shape[1]).contiguous(),
            out_dtype=torch.bfloat16,
        )

    def test_row_scales_and_dispatch_boundaries(self):
        for n, k, limit in [
            (3072, 5120, 64),
            (4096, 5120, 64),
            (5120, 3072, 64),
            (8192, 5120, 32),
        ]:
            for m in [1, 5, 16, 17, 31, 32, 33, 40, 61, 64, 65]:
                with self.subTest(m=m, n=n, k=k):
                    args = self.operands(m, n, k)
                    actual = skinny_fp8_mm(*args)
                    if m > limit:
                        self.assertIsNone(actual)
                    else:
                        self.assertTrue(torch.equal(actual, self.reference(*args)))

    def test_graph_reads_current_quantized_operands_and_scales(self):
        for m in (1, 5, 16, 20, 40, 64):
            args = self.operands(m)
            x, w, a, b = args
            skinny_fp8_mm(*args)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = skinny_fp8_mm(*args)
            x.view(torch.uint8).bitwise_xor_(1)
            w.view(torch.uint8).bitwise_xor_(1)
            a.mul_(0.91)
            b.mul_(0.83)
            graph.replay()
            self.assertTrue(torch.equal(actual, self.reference(*args)))
            graph.reset()

    def test_linear_entry_points_match_torch(self):
        n, k, m = 5120, 3072, 5
        w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.empty(2, n, device="cuda")
        scales[0].copy_(torch.linspace(0.011, 0.029, n, device="cuda"))
        scales[1].fill_(0.0273)
        with patch.dict(os.environ, RTP_LLM_SM120_FUSION="1"):
            layer = CudaModelOptFp8Linear(w, scales)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        q, a = static_fp8_quant_rows(x, scales[1, 0])
        a.mul_(torch.linspace(0.5, 1.0, m, device="cuda").reshape(m, 1))
        calls = [
            lambda: layer(x),
            lambda: layer.forward_with_gate(x, gate),
            lambda: layer.forward_with_prequantized(x, (q, a)),
        ]
        for call in calls:
            actual = call()
            with patch.object(layer, "_use_skinny", False):
                expected = call()
            self.assertTrue(torch.equal(actual, expected))

    def test_disabled_and_unqualified_shapes_fall_back(self):
        w = torch.randn(512, 5120, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(2, 512, device="cuda") * 0.0273
        scales[0].copy_(torch.linspace(0.01, 0.03, 512, device="cuda"))
        with patch.dict(os.environ, RTP_LLM_SM120_FUSION="1"):
            unqualified = CudaModelOptFp8Linear(w, scales)
        with patch.dict(os.environ, RTP_LLM_SM120_FUSION="0"):
            disabled = CudaModelOptFp8Linear(w, scales)
        self.assertTrue(unqualified._use_skinny)
        self.assertFalse(disabled._use_skinny)
        x = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16)
        with patch(
            "rtp_llm.models_py.kernels.cuda.sm120_fp8_skinny._module",
            side_effect=AssertionError("fallback invoked the new kernel"),
        ):
            self.assertTrue(torch.equal(unqualified(x), disabled(x)))

    def test_device_guard_restores_current_device(self):
        initial = torch.cuda.current_device()
        device = 1 if torch.cuda.device_count() > 1 else initial
        with torch.cuda.device(device):
            args = self.operands(5)
        actual = skinny_fp8_mm(*args)
        self.assertEqual(torch.cuda.current_device(), initial)
        self.assertTrue(torch.equal(actual, self.reference(*args)))

    def test_unsupported_dtype_does_not_launch(self):
        x, w, a, b = self.operands(5)
        with patch(
            "rtp_llm.models_py.kernels.cuda.sm120_fp8_skinny._module",
            side_effect=AssertionError("unsupported dtype launched"),
        ):
            self.assertIsNone(skinny_fp8_mm(x.float(), w, a, b))


if __name__ == "__main__":
    unittest.main()
