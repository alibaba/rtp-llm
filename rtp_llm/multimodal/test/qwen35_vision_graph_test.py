import copy
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_graph import (
    VisionGraphCache,
)


class TinyVision(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(64, device=device, dtype=torch.float32).reshape(8, 8) / 64
        )

    def prepare_graph_metadata(self, grid, pixels):
        return {"attention_backend": "fa4", "offset": int(grid[0, 1])}

    def forward(self, pixels, grid_thw, _graph_metadata=None, **kwargs):
        metadata = _graph_metadata or self.prepare_graph_metadata(grid_thw, pixels)
        return SimpleNamespace(pooler_output=pixels @ self.weight + metadata["offset"])


class VisionGraphTest(unittest.TestCase):
    @mock.patch(
        "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_graph.kmonitor.report"
    )
    def test_fallback_reports_m3_and_legacy_metrics(self, report):
        vision = TinyVision()
        cache = VisionGraphCache(vision, enabled=True)
        pixels, grid = torch.ones(16, 8), torch.tensor([[1, 4, 4]])
        torch.testing.assert_close(
            cache.run(pixels, grid), vision(pixels, grid).pooler_output
        )
        report.assert_any_call(AccMetrics.VIT_CUDA_GRAPH_FALLBACK_QPS_METRIC, 1)
        report.assert_any_call(
            AccMetrics.VIT_GRAPH_EVENT_QPS_METRIC,
            1,
            {"event": "fallback", "model": "qwen35"},
        )
        self.assertEqual(cache.stats()["fallback"], 1)

    @mock.patch(
        "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_graph.kmonitor.report"
    )
    def test_explicit_bypasses_report_fallback(self, report):
        vision = TinyVision()
        pixels = torch.ones(16, 8)
        for options, grid in (
            ({"enabled": False}, torch.tensor([[1, 4, 4]])),
            ({"enabled": True, "max_entries": 0}, torch.tensor([[1, 4, 4]])),
            ({"enabled": True}, torch.tensor([[1, 2, 4], [1, 2, 4]])),
        ):
            with self.subTest(options=options, grid=grid.tolist()):
                report.reset_mock()
                cache = VisionGraphCache(vision, **options)
                torch.testing.assert_close(
                    cache.run(pixels, grid), vision(pixels, grid).pooler_output
                )
                report.assert_any_call(AccMetrics.VIT_CUDA_GRAPH_FALLBACK_QPS_METRIC, 1)
                self.assertEqual(cache.stats()["fallback"], 1)
                self.assertEqual(cache.stats()["miss"], 0)
                self.assertEqual(cache.stats()["entries"], 0)

    @mock.patch(
        "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_graph.kmonitor.report"
    )
    def test_exact_grid_metrics_have_zero_padding(self, report):
        cache = VisionGraphCache(TinyVision())
        for event, metric in (
            ("hit", AccMetrics.VIT_CUDA_GRAPH_HIT_QPS_METRIC),
            ("capture", AccMetrics.VIT_CUDA_GRAPH_CAPTURE_QPS_METRIC),
        ):
            with self.subTest(event=event):
                report.reset_mock()
                cache._report(event)
                report.assert_any_call(metric, 1)
                report.assert_any_call(
                    GaugeMetrics.VIT_CUDA_GRAPH_PADDING_RATIO_METRIC, 0
                )

    @mock.patch(
        "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_graph.kmonitor.report",
        side_effect=RuntimeError("metrics unavailable"),
    )
    def test_report_failure_does_not_fail_inference(self, report):
        vision = TinyVision()
        cache = VisionGraphCache(vision, enabled=False)
        pixels, grid = torch.ones(16, 8), torch.tensor([[1, 4, 4]])
        torch.testing.assert_close(
            cache.run(pixels, grid), vision(pixels, grid).pooler_output
        )
        self.assertEqual(cache.stats()["fallback"], 1)

    def test_cpu_fallback_and_invalid_limits(self):
        vision = TinyVision()
        cache = VisionGraphCache(vision)
        pixels, grid = torch.ones(16, 8), torch.tensor([[1, 4, 4]])
        torch.testing.assert_close(
            cache.run(pixels, grid), vision(pixels, grid).pooler_output
        )
        self.assertEqual(cache.stats()["entries"], 0)
        with self.assertRaises(ValueError):
            VisionGraphCache(vision, max_entries=-1)

    def test_capture_error_checks_are_scoped_to_vit_thread(self):
        # Exercise the production capture branch without requiring a GPU.
        vision = TinyVision()
        cache = VisionGraphCache(vision, enabled=True)
        pixels, grid = torch.ones(16, 8), torch.tensor([[1, 4, 4]])
        with (
            mock.patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=False),
            mock.patch("torch.cuda.current_stream"),
            mock.patch("torch.cuda.Stream") as stream,
            mock.patch("torch.cuda.Event"),
            mock.patch("torch.cuda.CUDAGraph") as graph,
            mock.patch("torch.cuda.stream", side_effect=lambda _: nullcontext()),
            mock.patch("torch.cuda.graph") as capture,
        ):
            cache.run(pixels, grid)
            result = cache.run(pixels, grid)
            capture.assert_called_once_with(
                graph.return_value,
                stream=stream.return_value,
                capture_error_mode="thread_local",
            )
        self.assertTrue(torch.equal(result, vision(pixels, grid).pooler_output))
        self.assertEqual(cache.stats()["capture"], 1)
        self.assertEqual(cache.stats()["fallback"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_capture_overlaps_other_thread_h2d_d2h(self):
        capture_started = threading.Event()
        copies_done = threading.Event()
        cancelled = threading.Event()
        errors, copied = [], []

        class GatedVision(TinyVision):
            def forward(self, *args, **kwargs):
                if torch.cuda.is_current_stream_capturing():
                    capture_started.set()
                    if not copies_done.wait(timeout=30):
                        raise RuntimeError(
                            "LLM copy thread did not finish during capture"
                        )
                return super().forward(*args, **kwargs)

        vision = GatedVision("cuda")
        pixels, grid = torch.ones(16, 8, device="cuda"), torch.tensor([[1, 4, 4]])
        cache = VisionGraphCache(vision, enabled=True)
        expected = vision(pixels, grid).pooler_output
        cache.run(pixels, grid)
        host_input = torch.arange(128, dtype=torch.float32).pin_memory()
        device_buffer = torch.empty_like(host_input, device=pixels.device)
        copy_stream = torch.cuda.Stream(device=pixels.device)
        copy_stream.wait_stream(torch.cuda.current_stream(pixels.device))

        def llm_copies():
            try:
                if not capture_started.wait(timeout=30):
                    raise RuntimeError("ViT capture did not start")
                if cancelled.is_set():
                    return
                with torch.inference_mode(), torch.cuda.stream(copy_stream):
                    # Model the pinned allocator and async transfers used by
                    # prefill inputs / NormalOutputDispatcher in another thread.
                    host_output = torch.empty_like(host_input, pin_memory=True)
                    device_buffer.copy_(host_input, non_blocking=True)
                    host_output.copy_(device_buffer, non_blocking=True)
                    copy_stream.synchronize()
                    # Cover the blocking D2H path as well, still on our stream.
                    blocking_output = device_buffer.to("cpu")
                    copied.extend((host_output, blocking_output))
            except Exception as error:
                errors.append(error)
            finally:
                copies_done.set()

        worker = threading.Thread(target=llm_copies, daemon=True)
        worker.start()
        try:
            first = cache.run(pixels, grid)
        finally:
            cancelled.set()
            capture_started.set()
            worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "LLM copy thread did not exit")
        self.assertEqual(errors, [])
        self.assertEqual(len(copied), 2)
        for output in copied:
            self.assertTrue(torch.equal(output, host_input))
        self.assertEqual(cache.stats()["capture"], 1)
        self.assertEqual(cache.stats()["fallback"], 0)
        self.assertTrue(torch.equal(first, expected))
        # Replaying new inputs must not mutate the previous output.
        changed = pixels * 2
        replayed = cache.run(changed, grid)
        self.assertTrue(torch.equal(replayed, vision(changed, grid).pooler_output))
        self.assertTrue(torch.equal(first, expected))
        self.assertEqual(cache.stats()["hit"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_replay_isolation_exact_grid_and_eviction(self):
        vision = TinyVision("cuda")
        cache = VisionGraphCache(vision, max_entries=1)
        pixels = torch.randn(16, 8, device="cuda")
        grid = torch.tensor([[1, 4, 4]])
        expected = vision(pixels, grid).pooler_output
        cache.run(pixels, grid)
        original = cache.run(pixels, grid)
        other = cache.run(pixels * 2, grid)
        torch.testing.assert_close(original, expected)
        torch.testing.assert_close(other, vision(pixels * 2, grid).pooler_output)
        self.assertNotEqual(original.data_ptr(), other.data_ptr())
        self.assertEqual(cache.stats()["capture"], 1)
        self.assertEqual(cache.stats()["hit"], 1)
        # Equal pixel shape with a different grid must use a different graph.
        second_grid = torch.tensor([[1, 2, 8]])
        cache.run(pixels, second_grid)
        different = cache.run(pixels, second_grid)
        torch.testing.assert_close(different, vision(pixels, second_grid).pooler_output)
        self.assertEqual(cache.stats()["capture"], 2)
        self.assertEqual(cache.stats()["entries"], 1)
        torch.testing.assert_close(original, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_real_qwen_vision_graph_matches_eager(self):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
            Qwen3_5MoeVisionConfig,
            Qwen3_5MoeVisionModel,
        )

        config = Qwen3_5MoeVisionConfig(
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=2,
            out_hidden_size=128,
            num_position_embeddings=64,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
        )
        config.vit_attention_backend = "fa4"
        torch.manual_seed(12)
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeVisionModel as HFVisionModel,
        )

        vision = Qwen3_5MoeVisionModel(config).eval()
        vision.load_weights(HFVisionModel(copy.deepcopy(config)).state_dict().items())
        vision = vision.to(device="cuda", dtype=torch.bfloat16)
        cache = VisionGraphCache(vision, max_entries=1)
        pixels = torch.randn(16, 3 * 2 * 16 * 16, device="cuda", dtype=torch.bfloat16)
        grid = torch.tensor([[1, 4, 4]])
        with torch.inference_mode():
            expected = vision(pixels, grid).pooler_output
            cache.run(pixels, grid)
            actual = cache.run(pixels, grid)
            replay = cache.run(pixels + 0.1, grid)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(
                replay, vision(pixels + 0.1, grid).pooler_output, atol=1e-2, rtol=1e-2
            )
        self.assertEqual(
            cache.stats()["capture"],
            1,
            "must capture, not silently exercise eager fallback",
        )
        self.assertEqual(cache.stats()["hit"], 1)


if __name__ == "__main__":
    unittest.main()
