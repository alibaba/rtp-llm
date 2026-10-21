import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant import static_fp8_quant_rows
from rtp_llm.ops.compute_ops import per_tensor_quant_fp8


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class ModelOptFp8QuantTest(unittest.TestCase):
    def reference(self, x, scale):
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        per_tensor_quant_fp8(x.contiguous(), q, scale, True)
        return q, scale.reshape(1, 1).expand(x.shape[0], 1).contiguous()

    def assert_same(self, got, expected):
        self.assertTrue(
            torch.equal(got[0].view(torch.uint8), expected[0].view(torch.uint8))
        )
        self.assertTrue(torch.equal(got[1], expected[1]))

    def test_codes_and_scales(self):
        torch.manual_seed(417)
        with patch.dict(os.environ, {"RTP_LLM_MODELOPT_FP8_FUSED_QUANT": "1"}):
            for dtype in (torch.bfloat16, torch.float16):
                for m in (1, 3, 17, 127):
                    for k in (3072, 5120, 6144):
                        with self.subTest(dtype=dtype, m=m, k=k):
                            x = torch.randn(m, k, device="cuda", dtype=dtype)
                            x.reshape(-1)[:8] = torch.tensor(
                                [
                                    0.0,
                                    -0.0,
                                    float("inf"),
                                    -float("inf"),
                                    float("nan"),
                                    448,
                                    -448,
                                    1e-5,
                                ],
                                device="cuda",
                                dtype=dtype,
                            )
                            scale = torch.tensor(0.1121651828289032, device="cuda")
                            self.assert_same(
                                static_fp8_quant_rows(x, scale),
                                self.reference(x, scale),
                            )

    def test_graph_reads_updated_input_and_scale(self):
        x = torch.randn(3, 5120, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(0.2064732164144516, device="cuda")
        with patch.dict(os.environ, {"RTP_LLM_MODELOPT_FP8_FUSED_QUANT": "1"}):
            for _ in range(3):
                static_fp8_quant_rows(x, scale)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = static_fp8_quant_rows(x, scale)
            for _ in range(4):
                x.normal_()
                scale.mul_(0.75)
                graph.replay()
                self.assert_same(got, self.reference(x, scale))

    def test_nondefault_stream_and_row_stride(self):
        storage = torch.randn(6, 6144, device="cuda", dtype=torch.bfloat16)
        x = storage[::2]
        scale = torch.tensor(0.25, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with patch.dict(os.environ, {"RTP_LLM_MODELOPT_FP8_FUSED_QUANT": "1"}):
            with torch.cuda.stream(stream):
                got = static_fp8_quant_rows(x, scale)
            torch.cuda.current_stream().wait_stream(stream)
            self.assert_same(got, self.reference(x, scale))

    def test_fallback_and_empty(self):
        x = torch.empty(0, 5120, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(0.25, device="cuda")
        with patch.dict(os.environ, {"RTP_LLM_MODELOPT_FP8_FUSED_QUANT": "0"}):
            self.assertIsNone(static_fp8_quant_rows(x, scale))
        with patch.dict(os.environ, {"RTP_LLM_MODELOPT_FP8_FUSED_QUANT": "1"}):
            q, s = static_fp8_quant_rows(x, scale)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(s.shape, (0, 1))
            self.assertIsNone(static_fp8_quant_rows(x[:, :128], scale))

    def test_default_dispatch(self):
        x = torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(0.25, device="cuda")
        with patch.dict(os.environ):
            os.environ.pop("RTP_LLM_MODELOPT_FP8_FUSED_QUANT", None)
            self.assert_same(static_fp8_quant_rows(x, scale), self.reference(x, scale))


if __name__ == "__main__":
    unittest.main()
