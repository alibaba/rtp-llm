"""Native-byte, consumer and Graph contracts for the H20 quantization dispatch."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.base.common.fp8_linear import (
    Fp8Linear,
    _quantize_projection_input,
)
from rtp_llm.models_py.triton_kernels.common import fp8_small_quant as quant

SHAPES = (
    (64, 320),
    (32, 640),
    (64, 640),
    (32, 2560),
    (64, 2560),
    (256, 2560),
    (1024, 2560),
    (16, 10240),
    (32, 10240),
    (64, 10240),
)


class Fp8SM90QuantTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (9, 0))
        torch.manual_seed(20261116)

    def exact(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def compare(self, graph, actual, x):
        graph.replay()
        expected = scaled_fp8_per_tensor_quant(x)
        self.exact(actual[0], expected[0])
        self.exact(actual[1], expected[1])

    def test_native_bytes(self):
        for shape in SHAPES:
            with self.subTest(shape=shape):
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = quant.try_small_fp8_quant(x)
                self.assertIsNotNone(actual)
                for magnitude in (0.0, 1e-38, 1e-12, 0.01, 1.0, 1e10):
                    x.normal_().mul_(magnitude)
                    self.compare(graph, actual, x)
                for value in (float("nan"), float("inf"), -float("inf"), -0.0):
                    x.fill_(value)
                    self.compare(graph, actual, x)
                values = (
                    torch.arange(65536, device="cuda", dtype=torch.int32)
                    .to(torch.int16)
                    .view(torch.bfloat16)
                )
                for start in range(0, values.numel(), x.numel()):
                    x.zero_()
                    part = values[start : start + x.numel()]
                    x.flatten()[: part.numel()].copy_(part)
                    self.compare(graph, actual, x)
                graph.reset()

    def test_guards_and_sm120_unchanged(self):
        inputs = [torch.empty(s, device="cuda", dtype=torch.bfloat16) for s in SHAPES]
        for x in inputs:
            self.assertFalse(quant.is_supported(x))
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            for capability in ((12, 0), (8, 0)):
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ):
                    for x in inputs:
                        self.assertFalse(quant.is_supported(x))
            for rows in (513, 1025, 8192, 32768):
                x = torch.empty((rows, 2560), device="cuda", dtype=torch.bfloat16)
                self.assertFalse(quant.is_supported(x))
            x = torch.randn(32, 5120, device="cuda", dtype=torch.bfloat16)[:, ::2]
            self.assertFalse(quant.is_supported(x))
            self.assertFalse(quant.is_supported(x.contiguous().float()))
            x = x.contiguous()
            alias = x.view(torch.float8_e4m3fn).flatten()[: x.numel()].view_as(x)
            self.assertFalse(quant.is_supported(x, alias))
        for shape in ((1, 2560), (25, 2560), (4, 640), (6, 10240)):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            q = torch.empty(
                ((shape[0] + 15) // 16 * 16, shape[1]),
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            actual = quant.try_small_fp8_quant(x, output=q)
            expected = scaled_fp8_per_tensor_quant(x, output=torch.empty_like(q))
            self.exact(actual[0], expected[0])
            self.exact(actual[1], expected[1])

    def test_explicit_scale_and_prefill_fallback(self):
        x = torch.randn(64, 2560, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([0.025], device="cuda")
        with patch.object(
            quant,
            "try_small_fp8_quant",
            side_effect=AssertionError("unexpected dispatch"),
        ):
            actual = _quantize_projection_input(x, role="PDFUSION", scale=scale)
            expected = scaled_fp8_per_tensor_quant(x, scale=scale)
            self.exact(actual[0], expected[0])
            actual = _quantize_projection_input(x, role="PREFILL")
            expected = scaled_fp8_per_tensor_quant(x)
            self.exact(actual[0], expected[0])
            self.exact(actual[1], expected[1])

    def test_graph_refresh_and_independent_scales(self):
        for shape in SHAPES:
            with self.subTest(shape=shape):
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                y = torch.randn_like(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.assertTrue(quant.is_supported(x))
                    a = _quantize_projection_input(x, role="PDFUSION")
                    b = _quantize_projection_input(y, role="PDFUSION")
                self.assertNotEqual(a[1].data_ptr(), b[1].data_ptr())
                addresses = [t.data_ptr() for t in (*a, *b)]
                for magnitude in (0.0, 1e-12, 1.0, 10.0):
                    x.normal_().mul_(magnitude)
                    y.normal_().mul_(0.125)
                    a[0].fill_(float("nan"))
                    b[0].fill_(float("nan"))
                    graph.replay()
                    for actual, inp in ((a, x), (b, y)):
                        expected = scaled_fp8_per_tensor_quant(inp)
                        self.exact(actual[0], expected[0])
                        self.exact(actual[1], expected[1])
                    self.assertEqual(addresses, [t.data_ptr() for t in (*a, *b)])
                graph.reset()

    def test_linear_consumer(self):
        for rows, width in SHAPES:
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            weight = (
                torch.randn(128, width, device="cuda")
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            layer = Fp8Linear(weight, torch.tensor([0.1], device="cuda"))
            expected = layer(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = layer(x)
            graph.replay()
            self.exact(actual, expected)
            graph.reset()

    def test_nondefault_stream(self):
        stream = torch.cuda.Stream()
        x = torch.randn(64, 2560, device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = quant.try_small_fp8_quant(x)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.compare(graph, actual, x)
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    unittest.main()
