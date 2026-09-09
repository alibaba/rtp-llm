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

    def test_pending_limit_fails_instead_of_truncating_large_tensor(self):
        trace = self.trace(max_pending_bytes=8)
        trace.begin({})
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            trace.record("too_large", torch.ones(3, dtype=torch.float32))
        with self.assertRaises(RuntimeError):
            trace.close()
        self.assertFalse((self.root / "recorder_closed.json").exists())

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
