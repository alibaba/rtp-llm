"""Exhaustive BF16 inputs and submission timing before enabling the fusion."""

import json
import os
import statistics
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import rtp_llm.models_py.triton_kernels.common.clamped_swiglu as module
import torch
import torch.nn.functional as F
from rtp_llm.models_py.triton_kernels.common.clamped_swiglu import clamped_swiglu


def reference(x, limit):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


class TestClampedSwiglu(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        dispatcher = patch.object(module, "_COMPILED_DISPATCH", True)
        dispatcher.start()
        self.addCleanup(dispatcher.stop)

    def check(self, x, limit):
        expected, actual = reference(x, limit), clamped_swiglu(x, limit)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual.isnan(), expected.isnan()))
        valid = ~expected.isnan()
        differing = actual.view(torch.int16)[valid] != expected.view(torch.int16)[valid]
        if differing.any():
            raise AssertionError(
                f"{differing.sum().item()} different BF16 outputs at shape={x.shape}, limit={limit}"
            )

    @torch.inference_mode()
    def test_all_bf16_patterns_and_shapes(self):
        values = (
            torch.arange(65536, dtype=torch.int32, device="cuda")
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        for limit in (5.3, 10.0, 16.0):
            for up in (torch.ones_like(values), values.roll(19733), -values):
                x = torch.cat(
                    (values.reshape(256, 256), up.reshape(256, 256)), dim=1
                ).contiguous()
                self.check(x, limit)
        torch.manual_seed(1353)
        for tokens in (0, 1, 13, 81, 509, 4093):
            for width in (127, 768, 1024, 4096):
                x = (
                    torch.randn(tokens, 2 * width, device="cuda", dtype=torch.bfloat16)
                    * 7
                )
                self.check(x, 10.0)
        self.assertIsNone(clamped_swiglu(torch.zeros(2, 4), 10.0))
        self.assertIsNone(clamped_swiglu(torch.zeros(2, 4, device="cuda"), 10.0))

    @torch.inference_mode()
    def test_compiled_dispatch_device_stream_and_fallback(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        for device in (0, 1):
            with torch.cuda.device(device):
                x = torch.randn(81, 2048, device="cuda", dtype=torch.bfloat16)
                self.check(x, 10.0)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    clamped_swiglu(x, 10.0)
                    stream.synchronize()
                    torch.cuda._sleep(100_000_000)
                    pending = torch.cuda.Event()
                    pending.record()
                    actual = clamped_swiglu(x, 10.0)
                    self.assertFalse(pending.query())
                stream.synchronize()
                torch.testing.assert_close(actual, reference(x, 10.0), rtol=0, atol=0)
                unaligned = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[
                    1:
                ].view_as(x)
                unaligned.copy_(x)
                with patch.object(
                    module,
                    "_runner",
                    side_effect=AssertionError("unsupported fast dispatch"),
                ):
                    self.check(unaligned, 10.0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = clamped_swiglu(x, 10.0)
                graph.replay()
                torch.testing.assert_close(captured, reference(x, 10.0), rtol=0, atol=0)
        self.assertLessEqual(module._runner.cache_info().currsize, 128)

    @torch.inference_mode()
    def test_compiled_dispatch_wrapper_benchmark(self):
        rows = []
        for tokens in (1, 81, 419, 5103, 9889):
            x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
            outputs = []
            for enabled in (False, True):
                module._COMPILED_DISPATCH = enabled
                for _ in range(30):
                    actual = clamped_swiglu(x, 10.0)
                outputs.append(actual)
            self.assertTrue(
                torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
            )
            samples = {False: [], True: []}
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for repeat in range(60):
                for enabled in ((False, True) if repeat % 2 else (True, False)):
                    module._COMPILED_DISPATCH = enabled
                    torch.cuda.synchronize()
                    start.record()
                    begin = time.perf_counter_ns()
                    clamped_swiglu(x, 10.0)
                    cpu_us = (time.perf_counter_ns() - begin) / 1000
                    end.record()
                    end.synchronize()
                    samples[enabled].append(
                        dict(cpu_us=cpu_us, event_us=start.elapsed_time(end) * 1000)
                    )
            rows.append(dict(tokens=tokens, width=1024, samples=samples))
            print(
                "CLAMPED_DISPATCH",
                tokens,
                {
                    enabled: {
                        metric: statistics.median(r[metric] for r in sample)
                        for metric in ("cpu_us", "event_us")
                    }
                    for enabled, sample in samples.items()
                },
                flush=True,
            )
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "clamped_dispatch.json"
        ).write_text(json.dumps(rows, indent=2))

    @torch.inference_mode()
    def test_model_prefill_scope_guard(self):
        from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA
        from rtp_llm.models_py.modules.hybrid.dense_mlp import ClampedSiluAndMul

        with patch.dict(os.environ, {"GLM53_PREFILL_FUSED_CLAMPED_SWIGLU": "1"}):
            layer = ClampedSiluAndMul(10.0)
        self.assertIsNotNone(layer._prefill_fusion)
        x = torch.randn(81, 2048, device="cuda", dtype=torch.bfloat16)
        expected = reference(x, 10.0)
        with patch.object(
            layer, "_prefill_fusion", wraps=layer._prefill_fusion
        ) as fused:
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
            fused.assert_not_called()
            token = _FORWARD_METADATA.set({})
            try:
                torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
                fused.assert_called_once()
            finally:
                _FORWARD_METADATA.reset(token)
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
            fused.assert_called_once()

    @torch.inference_mode()
    def test_submission_benchmark(self):
        rows = []
        for tokens in (1, 81, 419, 1021, 4093):
            x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
            self.check(x, 10.0)
            row = {"tokens": tokens, "width": 1024}
            for name, function in (("eager", reference), ("fused", clamped_swiglu)):
                for _ in range(20):
                    function(x, 10.0)
                cpu, gpu = [], []
                for _ in range(5):
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    before = time.perf_counter()
                    for _ in range(100):
                        function(x, 10.0)
                    cpu.append((time.perf_counter() - before) * 1e6 / 100)
                    end.record()
                    end.synchronize()
                    gpu.append(start.elapsed_time(end) * 1000 / 100)
                row[name] = {
                    "cpu_us_median": statistics.median(cpu),
                    "stream_us_median": statistics.median(gpu),
                }
            rows.append(row)
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "clamped_swiglu.json"
        ).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    unittest.main()
