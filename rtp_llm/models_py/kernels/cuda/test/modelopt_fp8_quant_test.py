import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.modelopt_mixed import ModelOptMixedPrecisionConfig
from rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant import static_fp8_quant_rows
from rtp_llm.models_py.modules.factory.linear.impl.cuda.modelopt_fp8_linear import (
    CudaModelOptFp8Linear,
)
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
        with patch(
            "rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant._FUSED_QUANT_ENABLED",
            True,
        ):
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
        with patch(
            "rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant._FUSED_QUANT_ENABLED",
            True,
        ):
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
        with patch(
            "rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant._FUSED_QUANT_ENABLED",
            True,
        ):
            with torch.cuda.stream(stream):
                got = static_fp8_quant_rows(x, scale)
            torch.cuda.current_stream().wait_stream(stream)
            self.assert_same(got, self.reference(x, scale))

    def test_fallback_and_empty(self):
        x = torch.empty(0, 5120, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(0.25, device="cuda")
        with patch(
            "rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant._FUSED_QUANT_ENABLED",
            False,
        ):
            self.assertIsNone(static_fp8_quant_rows(x, scale))
        with patch(
            "rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant._FUSED_QUANT_ENABLED",
            True,
        ):
            q, s = static_fp8_quant_rows(x, scale)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(s.shape, (0, 1))
            self.assertIsNone(static_fp8_quant_rows(x[:, :128], scale))

    def test_dispatch_does_not_query_environment_or_gpu_capability_each_call(self):
        x = torch.randn(3, 5120, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(0.25, device="cuda")
        static_fp8_quant_rows(x, scale)  # Resolve the existing per-device SM cache.
        with patch(
            "rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant.os"
        ) as module_os, patch(
            "torch.cuda.get_device_capability",
            side_effect=AssertionError("hot-path device query"),
        ):
            module_os.getenv.side_effect = AssertionError("hot-path environment lookup")
            self.assert_same(static_fp8_quant_rows(x, scale), self.reference(x, scale))

    def test_default_dispatch(self):
        x = torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(0.25, device="cuda")
        with patch.dict(os.environ):
            os.environ.pop("RTP_LLM_MODELOPT_FP8_FUSED_QUANT", None)
            self.assert_same(static_fp8_quant_rows(x, scale), self.reference(x, scale))

    def test_large_row_offset_uses_64_bit_addressing(self):
        # The last row starts beyond INT32_MAX elements for K=6144.
        rows, width = (1 << 31) // 6144 + 1, 6144
        x = torch.zeros(rows, width, device="cuda", dtype=torch.bfloat16)
        x[-1, :16] = 3.0
        scale = torch.ones((), device="cuda", dtype=torch.float32)
        with patch(
            "rtp_llm.models_py.kernels.cuda.modelopt_fp8_quant._FUSED_QUANT_ENABLED",
            True,
        ):
            q, row_scales = static_fp8_quant_rows(x, scale)
        self.assertTrue(torch.equal(q[0, :16].float(), torch.zeros(16, device="cuda")))
        self.assertTrue(
            torch.equal(q[-1, :16].float(), torch.full((16,), 3.0, device="cuda"))
        )
        self.assertEqual(row_scales[-1].item(), 1.0)

    def test_rowwise_fp8_linear_forward(self):
        torch.manual_seed(38)
        x = torch.randn(3, 3072, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(64, 3072, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        scales = torch.full((2, 64), 0.5, device="cuda", dtype=torch.float32)
        scales[1, :32] = 0.1
        scales[1, 32:] = 0.2
        linear = CudaModelOptFp8Linear(
            weight, scales, quant_config=ModelOptMixedPrecisionConfig({})
        )
        actual = linear(x)
        expected_parts = []
        for start, end, activation_scale in ((0, 32, 0.1), (32, 64, 0.2)):
            quantized = (
                (x.float() / activation_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            )
            expected_parts.append(
                (quantized.float() * activation_scale)
                @ (weight[start:end].float() * scales[0, start:end, None]).T
            )
        expected = torch.cat(expected_parts, dim=-1)
        torch.testing.assert_close(actual.float(), expected, rtol=0.05, atol=0.5)

    def test_multiscale_prefill_chunks_preserve_values_and_bound_peak_memory(self):
        torch.manual_seed(4096)
        x = torch.randn(8193, 3072, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(64, 3072, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.full((2, 64), 0.5, device="cuda")
        scales[1, :32] = 0.125
        scales[1, 32:] = 0.25
        bias = torch.randn(64, device="cuda", dtype=torch.float32)
        linear = CudaModelOptFp8Linear(weight, scales, bias=bias)
        # Warm both row shapes before comparing live tensor peaks.
        linear(x)
        linear._forward_rows(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = linear._forward_rows(x)
        torch.cuda.synchronize()
        baseline_peak = torch.cuda.max_memory_allocated()
        del baseline
        torch.cuda.reset_peak_memory_stats()
        actual = linear(x)
        torch.cuda.synchronize()
        chunked_peak = torch.cuda.max_memory_allocated()
        expected = linear._forward_rows(x)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.5)
        self.assertLess(chunked_peak, baseline_peak)

    def test_mixed_fp8_forward_rejects_fp16_before_gemm(self):
        weight = torch.zeros(4, 8).to(torch.float8_e4m3fn)
        scales = torch.ones(2, 4)
        linear = CudaModelOptFp8Linear(
            weight, scales, quant_config=ModelOptMixedPrecisionConfig({})
        )
        with self.assertRaisesRegex(ValueError, "requires BF16"):
            linear(torch.zeros(1, 8, dtype=torch.float16))

    def test_multiscale_chunked_prefill_graph_replays_updated_input(self):
        x = torch.randn(9, 3072, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(64, 3072, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(2, 64, device="cuda")
        scales[1, :32] = 0.125
        scales[1, 32:] = 0.25
        linear = CudaModelOptFp8Linear(weight, scales)
        with patch.object(linear, "_PREFILL_CHUNK_ROWS", 4):
            for _ in range(3):
                linear(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = linear(x)
            for _ in range(3):
                x.normal_()
                graph.replay()
                expected = linear._forward_rows(x)
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.5)


if __name__ == "__main__":
    unittest.main()
