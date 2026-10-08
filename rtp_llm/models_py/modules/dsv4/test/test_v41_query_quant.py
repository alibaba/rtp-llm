"""Check decode Q fusion against the actual norm and CUDA quant producers."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4._v41_fused_qkv import strided_q_rmsnorm
from rtp_llm.models_py.modules.dsv4._v41_query_quant import (
    query_norm_quant,
    try_project_quantized_qkv,
)
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


def reference(x, weight):
    normalized = strided_q_rmsnorm(x, weight, 1e-6)
    q, sf = sgl_per_token_group_quant_fp8(
        normalized.reshape(-1, x.shape[-1]),
        32,
        eps=torch.finfo(torch.float32).tiny,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return normalized, (q.view(x.shape), sf)


class QueryQuantGateTest(unittest.TestCase):
    def test_disabled_does_not_access_attention(self):
        with patch.dict(os.environ, {"DSV41_FUSED_QUERY_QUANT": "0"}):
            self.assertIsNone(try_project_quantized_qkv(None, None))

    def test_non_native_consumer_keeps_original_path(self):
        attn = SimpleNamespace(wq_b=torch.nn.Identity())
        self.assertIsNone(try_project_quantized_qkv(attn, None))

    def test_quant_backend_selection_falls_back_before_projection(self):
        attn = SimpleNamespace(
            wq_b=Mock(spec=V41MXFP8Linear),
            wq_a_wkv=Mock(),
            q_norm=None,
            q_lora_rank=1280,
        )
        with patch(
            "rtp_llm.models_py.modules.dsv4._v41_query_quant.is_supported",
            return_value=True,
        ):
            for setting in ("v2", " V2 ", "invalid"):
                with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": setting}):
                    self.assertIsNone(
                        try_project_quantized_qkv(attn, torch.empty(4, 128))
                    )
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "auto"}):
                self.assertIsNone(
                    try_project_quantized_qkv(attn, torch.empty(4096, 128))
                )
        attn.wq_a_wkv.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class QueryQuantCudaTest(unittest.TestCase):
    def assert_same(self, got, ref):
        torch.testing.assert_close(got[0], ref[0], rtol=0, atol=0)
        torch.testing.assert_close(
            got[1][0].view(torch.uint8), ref[1][0].view(torch.uint8)
        )
        rows = got[0].numel() // got[0].shape[-1]
        groups = got[0].shape[-1] // 32
        for left, right in zip(
            got[1][1].T.contiguous().view(torch.uint8),
            ref[1][1].T.contiguous().view(torch.uint8),
        ):
            torch.testing.assert_close(
                left[: rows * 4], right[: rows * 4], rtol=0, atol=0
            )
        self.assertEqual(got[1][1].shape, ref[1][1].shape)
        self.assertEqual(got[1][1].stride(), ref[1][1].stride())
        self.assertEqual(groups % 4, 0)

    def test_shapes_strides_and_dynamic_range(self):
        torch.manual_seed(13)
        for rows in (0, 1, 4, 7, 20, 24, 40, 48, 97):
            for width in (128, 1280, 2560):
                with self.subTest(rows=rows, width=width):
                    packed = torch.randn(
                        rows, width + 512, device="cuda", dtype=torch.bfloat16
                    )
                    x = packed[:, :width]
                    w = torch.randn(width, device="cuda", dtype=torch.bfloat16)
                    self.assert_same(query_norm_quant(x, w, 1e-6), reference(x, w))
        for scale in (0.0, 1e-25, 1e-10, 1e-4, 1.0, 1000.0):
            x = torch.randn(7, 1280, device="cuda", dtype=torch.bfloat16) * scale
            w = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
            self.assert_same(query_norm_quant(x, w, 1e-6), reference(x, w))

    def test_quantization_thresholds(self):
        x = torch.ones(8, 1280, device="cuda", dtype=torch.bfloat16)
        values = torch.tensor(
            [0, 1e-10, 0.4375, 0.439453125, 448, 450, 896, 900], device="cuda"
        )
        w = values.repeat_interleave(160).bfloat16()
        self.assert_same(query_norm_quant(x, w, 1e-6), reference(x, w))

    def test_non_default_stream_graph_replay_updates_input(self):
        x = torch.randn(24, 1792, device="cuda", dtype=torch.bfloat16)[:, :1280]
        w = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                query_norm_quant(x, w, 1e-6)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = query_norm_quant(x, w, 1e-6)
        for scale in (0.0, 1.0, 100.0):
            x.copy_(torch.randn_like(x) * scale)
            graph.replay()
            self.assert_same(result, reference(x, w))

    def test_native_consumer_and_output_buffer(self):
        x = torch.randn(4, 6, 1280, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
        norm, quant = query_norm_quant(x, w, 1e-6)
        weight = (torch.randn(256, 1280, device="cuda") * 0.02).to(torch.float8_e4m3fn)
        scales = torch.ones(8, 40, device="cuda").to(torch.float8_e8m0fnu)
        linear = V41MXFP8Linear(weight, scales)
        expected = linear(norm)
        output = torch.empty_like(expected)
        actual = linear.forward_quantized(*quant, out=output)
        self.assertEqual(actual.data_ptr(), output.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("V41_QUERY_QUANT_BENCH"), "opt-in benchmark")
    def test_benchmark(self):
        def elapsed(fn):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(50):
                    fn()
            for _ in range(10):
                graph.replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end)

        results = []
        for rows in (1, 4, 7, 20, 24, 40, 48, 97, 257):
            x = torch.randn(rows, 1792, device="cuda", dtype=torch.bfloat16)[:, :1280]
            w = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
            ref_us = elapsed(lambda: reference(x, w))
            fused_us = elapsed(lambda: query_norm_quant(x, w, 1e-6))
            results.append(
                dict(
                    rows=rows,
                    baseline_us=ref_us,
                    fused_us=fused_us,
                    speedup=ref_us / fused_us,
                )
            )
        result = json.dumps(results, indent=2)
        print(result)
        destination = os.environ.get("V41_QUERY_QUANT_BENCH_JSON")
        if destination:
            Path(destination).write_text(result + "\n")


if __name__ == "__main__":
    unittest.main()
