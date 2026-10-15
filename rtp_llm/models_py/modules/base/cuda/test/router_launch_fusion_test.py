import json
import os
import time
from pathlib import Path
from typing import TypedDict
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.linear.fixed_m_linear import fixed_m_linear
from rtp_llm.models_py.modules.factory.linear.fixed_m_pad_triton import try_pad_rows


class Measurement(TypedDict):
    enabled: str
    cuda_event_us: float
    wall_us: float
    gpu_events: list[str]


class RouterPaddingFusionTest(TestCase):
    def setUp(self) -> None:
        self.assertTrue(torch.cuda.is_available(), "This target requires an NVIDIA GPU")
        self.env = patch.dict(
            os.environ,
            {"RTP_LLM_FUSED_ROUTER_PAD": "1"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(3871)

    def test_pad_preserves_bits_and_zero_tail(self) -> None:
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            source = torch.randn(17, 257, device="cuda", dtype=dtype)
            source[0, :4] = torch.tensor(
                [-0.0, float("inf"), -float("inf"), float("nan")], device="cuda"
            )
            expected = torch.zeros(23, 257, device="cuda", dtype=dtype)
            expected[:17].copy_(source)

            padded = try_pad_rows(source, 23)

            self.assertIsNotNone(padded)
            assert padded is not None
            self.assertTrue(
                torch.equal(padded.view(torch.uint8), expected.view(torch.uint8))
            )

    def test_padding_preserves_every_bfloat16_bit_pattern(self) -> None:
        source = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.uint16)
            .view(torch.bfloat16)
            .reshape(256, 256)
        )
        expected = torch.zeros((257, 256), device="cuda", dtype=torch.bfloat16)
        expected[:256].copy_(source)

        actual = try_pad_rows(source, 257)

        self.assertIsNotNone(actual)
        assert actual is not None
        self.assertTrue(
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        )

    def test_cuda_fallbacks_and_multiple_chunks_match_reference(self) -> None:
        for rows in (1, 8, 9, 17):
            for strided, gradients in ((False, False), (True, False), (False, True)):
                with self.subTest(rows=rows, strided=strided, gradients=gradients):
                    source = torch.randn(rows, 34, device="cuda")
                    source = source[:, ::2] if strided else source[:, :17].contiguous()
                    source.requires_grad_(gradients)
                    weight = torch.randn(11, 17, device="cuda")

                    def linear(value: torch.Tensor) -> torch.Tensor:
                        return torch.nn.functional.linear(value, weight)

                    with patch.dict(os.environ, {"RTP_LLM_FUSED_ROUTER_PAD": "0"}):
                        expected = fixed_m_linear(linear, source, 8)
                    actual = fixed_m_linear(linear, source, 8)

                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    if gradients:
                        expected_grad = torch.autograd.grad(expected.sum(), source)[0]
                        actual_grad = torch.autograd.grad(actual.sum(), source)[0]
                        torch.testing.assert_close(
                            actual_grad, expected_grad, atol=0, rtol=0
                        )

    def test_fixed_m_outputs_preserve_router_gemm(self) -> None:
        source = torch.randn(1252, 6144, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(256, 6144, device="cuda", dtype=torch.bfloat16)
        linear = lambda value: torch.nn.functional.linear(value, weight)
        with patch.dict(os.environ, {"RTP_LLM_FUSED_ROUTER_PAD": "0"}):
            expected = fixed_m_linear(linear, source, 8192)

        actual = fixed_m_linear(linear, source, 8192)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_graph_replay_uses_new_input(self) -> None:
        logits = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                try_pad_rows(logits, 32)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            padded = try_pad_rows(logits, 32)
        logits.add_(0.25)

        graph.replay()

        assert padded is not None
        torch.testing.assert_close(padded[:17], logits, atol=0, rtol=0)
        self.assertEqual(torch.count_nonzero(padded[17:]).item(), 0)

    def test_measure_padding_launch_reduction(self) -> None:
        source = torch.randn(1252, 6144, device="cuda", dtype=torch.bfloat16)
        results: list[Measurement] = []
        for enabled in ("0", "1", "unset"):
            with patch.dict(os.environ, {"RTP_LLM_FUSED_ROUTER_PAD": enabled}):
                if enabled == "unset":
                    os.environ.pop("RTP_LLM_FUSED_ROUTER_PAD")
                for _ in range(10):
                    fixed_m_linear(lambda value: value, source, 8192)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for _ in range(100):
                    fixed_m_linear(lambda value: value, source, 8192)
                end.record()
                end.synchronize()
                wall_us = (time.perf_counter() - wall_start) * 1e6 / 100
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    fixed_m_linear(lambda value: value, source, 8192)
                    torch.cuda.synchronize()
                gpu_events = [
                    event.name
                    for event in profile.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                ]
                results.append(
                    {
                        "enabled": enabled,
                        "cuda_event_us": start.elapsed_time(end) * 10,
                        "wall_us": wall_us,
                        "gpu_events": gpu_events,
                    }
                )
        artifact = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
            / "router_padding_measurement.json"
        )
        artifact.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "warmup": 10,
                    "iterations": 100,
                    "results": results,
                },
                indent=2,
            )
        )
        self.assertEqual(len(results[0]["gpu_events"]), 2)
        self.assertEqual(len(results[1]["gpu_events"]), 1)
        self.assertEqual(len(results[2]["gpu_events"]), 1)


if __name__ == "__main__":
    main()
