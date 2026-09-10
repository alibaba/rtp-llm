"""Snapshot ownership, durable failure reporting, and Graph replay contracts."""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

TRACE_PATH = Path(__file__).resolve().parents[1] / "k3_tensor_trace.py"
spec = importlib.util.spec_from_file_location("k3_trace_under_test", TRACE_PATH)
trace_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace_module
spec.loader.exec_module(trace_module)
TensorTrace = trace_module.TensorTrace


class TensorTraceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "trace"

    def trace(self, **kwargs):
        return TensorTrace(self.root, identity={"rank": 0, "role": "decode"}, **kwargs)

    def read(self, frame=0):
        return torch.load(self.root / f"frame-{frame:08d}.pt", weights_only=True)

    def test_file_fragment_limit_is_independent_of_total_snapshot_budget(self):
        trace = self.trace(max_pending_bytes=1024, max_fragment_bytes=16)
        trace.begin({"case": "separate-file-budget"})
        value = torch.zeros(4)
        for layer in range(4):
            value.fill_(layer)
            trace.record(f"layer.{layer}", value)
        value.fill_(-1)
        trace.end()
        trace.close()
        for layer in range(4):
            frame = self.read(layer)
            self.assertEqual(len(frame["tensors"]), 1)
            torch.testing.assert_close(
                frame["tensors"][0]["value"], torch.full((4,), float(layer))
            )
            self.assertEqual(
                frame["metadata"]["trace_fragment"],
                {"index": layer, "final": layer == 3},
            )

    def test_single_large_tensor_stays_whole_in_dedicated_fragment(self):
        trace = self.trace(max_pending_bytes=128, max_fragment_bytes=16)
        trace.begin({"case": "whole-tensor"})
        trace.record("before", torch.zeros(2))
        trace.record("large", torch.arange(12))
        trace.record("after", torch.ones(2))
        trace.end()
        trace.close()
        self.assertEqual(
            [self.read(i)["tensors"][0]["name"] for i in range(3)],
            ["before", "large", "after"],
        )
        torch.testing.assert_close(
            self.read(1)["tensors"][0]["value"], torch.arange(12)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA Graph")
    def test_capture_larger_than_file_budget_replays_in_lossless_fragments(self):
        trace = self.trace(max_pending_bytes=1024, max_fragment_bytes=64)
        value = torch.ones(16, device="cuda")
        graph = torch.cuda.CUDAGraph()
        trace.begin_capture("bs1")
        with torch.cuda.graph(graph):
            for layer in range(3):
                trace.record(f"layer.{layer}", value * (layer + 1))
        trace.end_capture()
        for step in range(2):
            value.fill_(step + 2)
            graph.replay()
            trace.replay("bs1", {"step": step})
        trace.close()
        for step in range(2):
            for layer in range(3):
                frame = self.read(step * 3 + layer)
                self.assertEqual(frame["metadata"]["step"], step)
                self.assertEqual(frame["tensors"][0]["name"], f"layer.{layer}")
                torch.testing.assert_close(
                    frame["tensors"][0]["value"],
                    torch.full((16,), float((step + 2) * (layer + 1))),
                )

    def test_in_place_overwrite_cannot_change_noncontiguous_bfloat16_snapshot(self):
        trace = self.trace()
        tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
        expected = tensor.clone()
        trace.begin({"case": "a", "position": 3})
        trace.record("layer.0.output", tensor, valid_tokens=4)
        tensor.fill_(-100)
        trace.record("layer.0.output", tensor, valid_tokens=4)
        trace.end()
        trace.close()
        frame = self.read()
        self.assertEqual(len(frame["tensors"]), 2)
        torch.testing.assert_close(
            frame["tensors"][0]["value"], expected, rtol=0, atol=0
        )
        self.assertEqual(frame["tensors"][0]["metadata"]["source_stride"], [1, 4])
        entries = [
            json.loads(line)
            for line in (self.root / "index.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            entries[0]["sha256"],
            hashlib.sha256((self.root / entries[0]["path"]).read_bytes()).hexdigest(),
        )
        self.assertFalse(
            json.loads((self.root / "recorder_closed.json").read_text())[
                "coverage_verified"
            ]
        )

    def test_unfinished_frame_cannot_report_successful_flush(self):
        trace = self.trace()
        trace.begin({"case": "unfinished"})
        trace.record("output", torch.ones(3))
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            trace.close()
        self.assertTrue((self.root / "incomplete.json").exists())
        self.assertFalse((self.root / "recorder_closed.json").exists())

    def test_nonzero_native_overflow_flag_prevents_successful_trace_close(self):
        trace = self.trace()
        trace.begin({"case": "native-overflow"})
        flag = torch.ones(1, dtype=torch.int32)
        trace.record("experts.native.overflow", flag, assert_zero=True)
        flag.zero_()
        trace.end()
        with self.assertRaisesRegex(RuntimeError, "nonzero diagnostic failure flag"):
            trace.close()
        self.assertTrue((self.root / "incomplete.json").exists())
        self.assertFalse((self.root / "recorder_closed.json").exists())

    def test_pending_limit_fails_instead_of_truncating_large_tensor(self):
        trace = self.trace(max_pending_bytes=8)
        trace.begin({})
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            trace.record("too_large", torch.ones(3, dtype=torch.float32))
        with self.assertRaises(RuntimeError):
            trace.close()
        self.assertFalse((self.root / "recorder_closed.json").exists())

    def test_long_frame_spills_without_losing_order_or_snapshot_ownership(self):
        trace = self.trace(max_pending_bytes=16)
        value = torch.zeros(4)
        trace.begin({"case": "long-prefill", "step": 0})
        for layer in range(7):
            value.fill_(layer)
            trace.record(f"layer.{layer}", value)
        value.fill_(-1)
        trace.end()
        trace.close()
        fragments = [self.read(i) for i in range(7)]
        observations = {f["metadata"]["observation_id"] for f in fragments}
        self.assertEqual(len(observations), 1)
        for layer, frame in enumerate(fragments):
            self.assertEqual(
                frame["metadata"]["trace_fragment"],
                {
                    "index": layer,
                    "final": layer == 6,
                },
            )
            self.assertEqual(frame["tensors"][0]["name"], f"layer.{layer}")
            torch.testing.assert_close(
                frame["tensors"][0]["value"], torch.full((4,), float(layer))
            )
        self.assertEqual(trace._pending, 0)

    def test_writer_failure_during_backpressure_aborts_incomplete_observation(self):
        trace = self.trace(max_pending_bytes=8)
        trace.begin({"case": "writer-failure"})
        trace.record("layer.0", torch.ones(2))
        with patch.object(torch, "save", side_effect=OSError("test disk full")):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                trace.record("layer.1", torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            trace.close()
        self.assertFalse((self.root / "recorder_closed.json").exists())

    def test_compression_failure_cannot_publish_a_successful_frame(self):
        trace = self.trace(compression="deflate")
        trace.begin({"case": "compression-failure"})
        trace.record("output", torch.ones(2))
        with patch.object(
            trace_module.zipfile,
            "ZipFile",
            side_effect=OSError("compression disk full"),
        ):
            trace.end()
            with self.assertRaisesRegex(RuntimeError, "compression disk full"):
                trace.close()
        self.assertTrue((self.root / "incomplete.json").exists())
        self.assertFalse((self.root / "recorder_closed.json").exists())
        self.assertFalse((self.root / "frame-00000000.pt").exists())
        self.assertEqual((self.root / "index.jsonl").read_text(), "")

    @unittest.skipUnless(torch.cuda.is_available(), "requires pinned D2H")
    def test_cuda_long_frame_spills_before_device_and_host_budget_is_exhausted(self):
        trace = self.trace(max_pending_bytes=128)
        trace.begin({"case": "cuda-long-prefill"})
        value = torch.zeros(16, device="cuda")
        for layer in range(5):
            value.fill_(layer)
            trace.record(f"layer.{layer}", value)
        value.fill_(-1)
        trace.end()
        trace.close()
        for layer in range(5):
            torch.testing.assert_close(
                self.read(layer)["tensors"][0]["value"], torch.full((16,), float(layer))
            )
        self.assertEqual(trace._pending, 0)

    def test_writer_error_is_returned_to_runner(self):
        trace = self.trace()
        trace.begin({})
        trace.record("value", torch.zeros(2))
        with patch.object(torch, "save", side_effect=OSError("test disk full")):
            trace.end()
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                trace.close()
        self.assertTrue((self.root / "incomplete.json").exists())

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires CUDA Graph and pinned D2H"
    )
    def test_every_graph_replay_gets_distinct_snapshot_and_dynamic_metadata(self):
        trace = self.trace()
        value = torch.ones(16, device="cuda")
        graph = torch.cuda.CUDAGraph()
        trace.begin_capture("decode.bs1")
        with torch.cuda.graph(graph):
            output = value * 3
            trace.record("layer.0.output", output)
        trace.end_capture()
        for step in range(3):
            value.fill_(step + 1)
            graph.replay()
            trace.replay("decode.bs1", {"step": step})
        trace.close()
        for step in range(3):
            frame = self.read(step)
            self.assertEqual(frame["metadata"]["step"], step)
            torch.testing.assert_close(
                frame["tensors"][0]["value"], torch.full((16,), 3.0 * (step + 1))
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA streams")
    def test_auxiliary_stream_snapshot_waits_for_its_own_producer(self):
        trace = self.trace()
        stream = torch.cuda.Stream()
        trace.begin({"phase": "prefill"})
        with torch.cuda.stream(stream):
            value = torch.arange(1024, device="cuda", dtype=torch.float32)
            trace.record("aux.output", value)
            value.fill_(-1)
        trace.end()
        trace.close()
        torch.testing.assert_close(
            self.read()["tensors"][0]["value"], torch.arange(1024, dtype=torch.float32)
        )


class EagerRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = patch.dict(
            os.environ,
            {
                "K3_TRACE_ROOT": str(self.root),
                "K3_TRACE_ENGINE": "rtp",
                "K3_TRACE_RUN_ID": "unit-test",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.states = patch.object(trace_module, "_runtimes", {})
        self.states.start()
        self.addCleanup(self.states.stop)

    def frames(self):
        return [
            torch.load(path, weights_only=True)
            for path in sorted(self.root.glob("*/frame-*.pt"))
        ]

    def test_nested_sampler_events_preserve_history_before_overwrite(self):
        history = torch.tensor([[1, 2, 3]])
        parent = trace_module.enter_scope("sampler.forward")
        child = trace_module.enter_scope("sampling.greedy")
        trace_module.event("penalty.inputs", {"history": history}, {"valid": 2})
        history.fill_(7)
        trace_module.event("sampling.output", {"history": history})
        trace_module.exit_scope(child)
        trace_module.exit_scope(parent)
        trace_module.close_process()
        frames = self.frames()
        before = next(f for f in frames if f["metadata"]["event"] == "penalty.inputs")
        self.assertEqual(before["tensors"][0]["value"].tolist(), [[1, 2, 3]])
        self.assertEqual(
            [s["id"] for s in before["metadata"]["scopes"]], [parent, child]
        )
        self.assertEqual(before["metadata"]["details"]["valid"], 2)

    def test_disabled_wrapper_does_not_create_runtime_or_require_identity(self):
        with patch.dict(os.environ, {"K3_TRACE_ROOT": ""}):

            @trace_module.traced_scope("disabled")
            def operation(x):
                trace_module.event("disabled", {"x": x})
                return x + 1

            self.assertEqual(operation(3), 4)
            self.assertEqual(list(self.root.iterdir()), [])
            self.assertEqual(trace_module._runtimes, {})

    def test_original_exception_survives_and_trace_is_incomplete(self):
        @trace_module.traced_scope("broken")
        def operation():
            raise ValueError("original inference failure")

        with self.assertRaisesRegex(ValueError, "original inference failure"):
            operation()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            trace_module.close_process()
        self.assertEqual(len(list(self.root.glob("*/incomplete.json"))), 1)
        self.assertEqual(list(self.root.glob("*/recorder_closed.json")), [])

    def test_shutdown_flushes_a_serving_thread_without_losing_frames(self):
        errors = []

        def serve():
            try:
                trace_module.event("worker.output", {"value": torch.arange(10)})
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=serve)
        thread.start()
        thread.join()
        self.assertEqual(errors, [])
        trace_module.close_process()
        self.assertEqual(
            self.frames()[0]["tensors"][0]["value"].tolist(), list(range(10))
        )
        self.assertEqual(len(list(self.root.glob("*/recorder_closed.json"))), 1)


if __name__ == "__main__":
    unittest.main()
