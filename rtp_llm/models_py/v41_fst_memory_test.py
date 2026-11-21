import logging
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py import v41_fst_memory
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.v41_fst_memory import V41FstMemoryTrace


class TraceTest(unittest.TestCase):
    def test_enable_only_for_explicit_target_or_draft_cuda_fst(self):
        def trace(name, method, device, forced=False):
            cfg = NewLoaderConfig(device=device, force_cpu_load_weights=forced)
            return V41FstMemoryTrace.for_load(name, method, cfg)

        with patch.object(v41_fst_memory.logger, "isEnabledFor", return_value=True):
            for name in ("deepseek_v41", "deepseek_v41_dspark"):
                self.assertIsNotNone(trace(name, "fastsafetensors", "cuda:1"))
            self.assertIsNone(trace("unrelated", "fastsafetensors", "cuda"))
            self.assertIsNone(trace("deepseek_v41", "scratch", "cuda"))
            self.assertIsNone(trace("deepseek_v41", "fastsafetensors", "cpu"))
            self.assertIsNone(trace("deepseek_v41", "fastsafetensors", "cuda", True))

    def test_observes_only_current_device_and_preserves_errors(self):
        trace = V41FstMemoryTrace("deepseek_v41_dspark", torch.device("cuda:1"), 7)
        with ExitStack() as stack:
            log = stack.enter_context(
                self.assertLogs(v41_fst_memory.logger, logging.INFO)
            )
            current = stack.enter_context(
                patch.object(torch.cuda, "current_device", return_value=0)
            )
            init = stack.enter_context(
                patch.object(torch.cuda, "is_initialized", return_value=False)
            )
            stack.enter_context(
                patch.object(v41_fst_memory, "is_cuda", return_value=True)
            )
            free = stack.enter_context(
                patch.object(torch.cuda, "mem_get_info", return_value=(17, 29))
            )
            backend = stack.enter_context(
                patch.object(torch.cuda, "get_allocator_backend", return_value="native")
            )
            stats = stack.enter_context(
                patch.object(
                    torch.cuda,
                    "memory_stats",
                    return_value={
                        "allocated_bytes.all.current": 3,
                        "reserved_bytes.all.current": 8,
                    },
                )
            )
            trace.record("before_model")
            current.assert_not_called()
            init.return_value = True
            trace.record("before_model")
            free.assert_not_called()
            current.return_value = 1
            trace.record("before_model")
            free.assert_called_once_with(1)
            stats.assert_called_once_with(1)
            self.assertIn(
                "rank=7 device=cuda:1 stage=before_model driver_free_bytes=17",
                log.output[-1],
            )
            self.assertIn(
                "torch_allocated_bytes=3 torch_reserved_bytes=8", log.output[-1]
            )
            backend.return_value = "cudaMallocAsync"
            trace.record("after_reader_close")
            stats.assert_called_once()
            self.assertIn("torch_allocated_bytes=unavailable", log.output[-1])
            backend.return_value = "native"
            stats.side_effect = RuntimeError("allocator API unavailable")
            trace.record("after_reader_close")
            self.assertIn("driver_free_bytes=17", log.output[-1])
            self.assertIn("torch_allocated_bytes=unavailable", log.output[-1])
            free.side_effect = RuntimeError("no sample")
            trace.record("after_post_load")
            self.assertIn("unavailable=RuntimeError", log.output[-1])

    def test_real_checkpoint_stage_order_and_reader_failure(self):
        # Only the observer is forced on for the CPU scratch fixture; dispatch,
        # reader closing, validation and post-load are NewModelLoader's.
        from rtp_llm.models_py import weight_mapper

        class Owner(RtpModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2), requires_grad=False)

            def process_weights_after_loading(self):
                events.append("post_load_hook")
                if fail == "post_load":
                    raise RuntimeError("fixture post_load failure")

        with tempfile.TemporaryDirectory() as root:
            save_file(
                {"weight": torch.tensor([2.0, 3.0])},
                str(Path(root) / "model.safetensors"),
            )
            for fail in (None, "close", "post_load"):
                events = []
                original = weight_mapper.get_all_weights

                def reader(*args, **kwargs):
                    try:
                        yield from original(*args, **kwargs)
                    finally:
                        events.append("reader_close")
                        if fail == "close":
                            raise RuntimeError("fixture close failure")

                loader = NewModelLoader(
                    {"model_type": "deepseek_v41"},
                    NewLoaderConfig(device="cpu", load_method="scratch"),
                    model_path=root,
                )
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            V41FstMemoryTrace,
                            "for_load",
                            return_value=SimpleNamespace(record=events.append),
                        )
                    )
                    stack.enter_context(
                        patch.object(loader, "_create_model", side_effect=Owner)
                    )
                    stack.enter_context(
                        patch.object(
                            weight_mapper, "get_all_weights", side_effect=reader
                        )
                    )
                    if fail is not None:
                        with self.assertRaisesRegex(
                            RuntimeError, f"fixture {fail} failure"
                        ):
                            loader.load()
                        expected = ["before_model", "reader_close"]
                        if fail == "post_load":
                            expected += ["after_reader_close", "post_load_hook"]
                        self.assertEqual(events, expected)
                        self.assertIsNone(loader.loaded_method)
                    else:
                        loaded = loader.load()
                        self.assertEqual(loaded.weight.tolist(), [2.0, 3.0])
                        self.assertEqual(
                            events,
                            [
                                "before_model",
                                "reader_close",
                                "after_reader_close",
                                "post_load_hook",
                                "after_post_load",
                            ],
                        )


if __name__ == "__main__":
    unittest.main()
