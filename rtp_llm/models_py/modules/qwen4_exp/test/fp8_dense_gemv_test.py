import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.fp8_dense import Qwen4Fp8Linear
from rtp_llm.models_py.modules.qwen4_exp.fp8_dense_gemv_triton import (
    fp8_hc_gemv,
    is_supported,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120 CUDA",
)
class Fp8HcGemvTest(unittest.TestCase):
    def _inputs(self, m, n=10240, k=320):
        torch.manual_seed(418)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
        return x, weight, scale

    def _check(self, output, x, weight, scale):
        reference = (x.float() @ weight.float().t()) * scale
        torch.testing.assert_close(output.float(), reference, rtol=0.01, atol=0.002)

    def test_adopted_shapes_and_bias(self):
        for m in (1, 3, 4):
            with self.subTest(m=m):
                x, weight, scale = self._inputs(m)
                self.assertTrue(is_supported(x, weight, scale))
                self._check(Qwen4Fp8Linear(weight, scale)(x), x, weight, scale)
                bias = torch.randn(10240, device="cuda", dtype=torch.float32)
                expected = (fp8_hc_gemv(x, weight, scale) + bias).to(x.dtype)
                actual = Qwen4Fp8Linear(weight, scale, bias)(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_side_stream_and_graph_live_input(self):
        x, weight, scale = self._inputs(3)
        module = Qwen4Fp8Linear(weight, scale)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                output = module(x)
        torch.cuda.current_stream().wait_stream(side)
        self._check(output, x, weight, scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module(x)
        for factor in (0.37, -0.5):
            x.mul_(factor).add_(1)
            graph.replay()
            self._check(output, x, weight, scale)

    def test_prefill_retains_scaled_mm_at_small_batch(self):
        x, weight, scale = self._inputs(3)
        kernel = "rtp_llm.models_py.modules.qwen4_exp.fp8_dense_gemv_triton.fp8_hc_gemv"
        with patch(kernel, wraps=fp8_hc_gemv) as gemv:
            prefill = Qwen4Fp8Linear(weight, scale, role="PREFILL")(x)
            gemv.assert_not_called()
            self.assertEqual(prefill.shape, (3, 10240))
            self.assertTrue(torch.isfinite(prefill).all())
            for role in ("DECODE", "PDFUSION"):
                self._check(
                    Qwen4Fp8Linear(weight, scale, role=role)(x), x, weight, scale
                )
            self.assertEqual(gemv.call_count, 2)

    def test_slower_shapes_excluded_from_support_domain(self):
        for m, n, k in ((5, 10240, 320), (8, 10240, 320), (3, 320, 10240)):
            with self.subTest(m=m, n=n, k=k):
                x, weight, scale = self._inputs(m, n, k)
                self.assertFalse(is_supported(x, weight, scale))
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    fp8_hc_gemv(x, weight, scale)
                self.assertEqual(Qwen4Fp8Linear(weight, scale)(x).shape, (m, n))


if __name__ == "__main__":
    unittest.main()
