import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.fp8_dense import Qwen4Fp8Linear
from rtp_llm.models_py.modules.qwen4_exp.fp8_dense_gemv_triton import (
    fp8_hc_gemv,
    is_supported,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12,
    "requires SM120 CUDA",
)
class Fp8HcGemvTest(unittest.TestCase):
    def _inputs(self, m, n, k):
        torch.manual_seed(418)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([0.01], device="cuda", dtype=torch.float32)
        return x, weight, scale

    def _check(self, output, x, weight, scale):
        reference = (x.float() @ weight.float().t()) * scale
        torch.testing.assert_close(output.float(), reference, rtol=0.01, atol=0.002)

    def test_real_hc_shapes_and_prime_batch(self):
        for n, k in ((320, 10240), (10240, 320)):
            for m in (1, 3, 4, 8):
                with self.subTest(m=m, n=n, k=k):
                    x, weight, scale = self._inputs(m, n, k)
                    self.assertTrue(is_supported(x, weight, scale))
                    self._check(fp8_hc_gemv(x, weight, scale), x, weight, scale)

    def test_side_stream_and_graph_live_input(self):
        x, weight, scale = self._inputs(3, 10240, 320)
        module = Qwen4Fp8Linear(weight, scale)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), patch.dict(
            os.environ, RTP_LLM_QWEN4_FP8_HC_GEMV_EXPERIMENTAL="1"
        ):
            for _ in range(3):
                output = module(x)
        torch.cuda.current_stream().wait_stream(side)
        self._check(output, x, weight, scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), patch.dict(
            os.environ, RTP_LLM_QWEN4_FP8_HC_GEMV_EXPERIMENTAL="1"
        ):
            output = module(x)
        x.mul_(0.37).add_(1)
        graph.replay()
        self._check(output, x, weight, scale)

    def test_module_gate_bias_and_unsupported_fallback(self):
        x, weight, scale = self._inputs(3, 10240, 320)
        bias = torch.randn(10240, device="cuda", dtype=torch.float32)
        module = Qwen4Fp8Linear(weight, scale, bias)
        with patch.dict(os.environ, RTP_LLM_QWEN4_FP8_HC_GEMV_EXPERIMENTAL="1"):
            actual = module(x)
        expected = (fp8_hc_gemv(x, weight, scale) + bias).to(x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for m, n, k in ((9, 320, 10240), (8, 10240, 320), (3, 320, 10240)):
            x, weight, scale = self._inputs(m, n, k)
            module = Qwen4Fp8Linear(weight, scale)
            with patch.dict(os.environ, RTP_LLM_QWEN4_FP8_HC_GEMV_EXPERIMENTAL="0"):
                baseline = module(x)
            with patch.dict(os.environ, RTP_LLM_QWEN4_FP8_HC_GEMV_EXPERIMENTAL="1"):
                torch.testing.assert_close(module(x), baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
