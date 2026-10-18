import gc
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock


def load_helper():
    # The JSON/feature-switch contract must be testable without importing the
    # serving package or loading Torch/CUDA on a standard-library-only worker.
    name = "_smoke_evidence_under_test"
    path = Path(__file__).resolve().parents[1] / "smoke_evidence.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EventCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("[RTP_SMOKE_EVENT] "):
            self.events.append(json.loads(message[len("[RTP_SMOKE_EVENT] ") :]))


class SmokeEventTest(unittest.TestCase):
    def setUp(self):
        self.helper = load_helper()
        self.collector = EventCollector()
        self.helper._LOGGER.addHandler(self.collector)
        self.previous_level = self.helper._LOGGER.level
        self.helper._LOGGER.setLevel(logging.INFO)
        self.addCleanup(self.helper._LOGGER.setLevel, self.previous_level)
        self.addCleanup(self.helper._LOGGER.removeHandler, self.collector)
        self.env = mock.patch.dict(os.environ, {"RTP_LLM_SMOKE_EVIDENCE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_only_exact_one_enables_evidence(self):
        for value in (None, "", "0", "true", "01", " 1", "1 ", "1"):
            with self.subTest(value=value), mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop("RTP_LLM_SMOKE_EVIDENCE", None)
                else:
                    os.environ["RTP_LLM_SMOKE_EVIDENCE"] = value
                self.assertEqual(self.helper.smoke_evidence_enabled(), value == "1")

    def test_disabled_calls_do_not_import_torch_or_inspect_tensors(self):
        with mock.patch.dict(os.environ, {"RTP_LLM_SMOKE_EVIDENCE": "0"}):
            with mock.patch.dict(sys.modules, {"torch": None}):
                self.helper.emit_smoke_event("disabled", value=object())
                self.helper.emit_smoke_tensor_event("disabled", {"value": object()})
        self.assertEqual(self.collector.events, [])
        self.assertIsNone(self.helper._worker)

    def test_json_preserves_kind_and_reserves_event_timestamp(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            with mock.patch.object(self.helper.time, "time_ns", return_value=123456):
                self.helper.emit_smoke_event(
                    "attention", kind="page_rr", count=7, ready=True, time_ns=-1
                )
        self.assertEqual(
            self.collector.events,
            [
                {
                    "event": "attention",
                    "time_ns": 123456,
                    "kind": "page_rr",
                    "count": 7,
                    "ready": True,
                }
            ],
        )

    def test_non_json_scalar_is_not_logged_as_valid_evidence(self):
        with self.assertRaises(ValueError):
            self.helper.emit_smoke_event("invalid", count=float("nan"))
        self.assertEqual(self.collector.events, [])

    def test_async_worker_waits_for_completion_and_keeps_sample_timestamp(self):
        release = threading.Event()
        waiting = threading.Event()
        producer_thread = threading.get_ident()

        class CompletedEvent:
            def synchronize(self):
                self_test.assertNotEqual(threading.get_ident(), producer_thread)
                waiting.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release event")

        self_test = self
        worker = self.helper._get_worker()
        self.assertTrue(worker.reserve())
        worker.submit(
            self.helper._Observation(
                "completed", 37, {"kind": "scalar"}, events=[CompletedEvent()]
            )
        )
        try:
            self.assertTrue(waiting.wait(5))
            self.assertEqual(self.collector.events, [])
            release.set()
            self.assertTrue(self.helper.flush_smoke_evidence(5))
            self.assertEqual(
                self.collector.events,
                [{"event": "completed", "time_ns": 37, "kind": "scalar"}],
            )
        finally:
            release.set()
            self.helper.flush_smoke_evidence(5)

    def test_full_queue_reports_error_instead_of_valid_or_dropped_evidence(self):
        release = threading.Event()

        class PendingEvent:
            def synchronize(self):
                if not release.wait(5):
                    raise RuntimeError("test did not release event")

        with mock.patch.object(self.helper, "_MAX_PENDING_OBSERVATIONS", 1):
            worker = self.helper._get_worker()
            self.assertTrue(worker.reserve())
            worker.submit(
                self.helper._Observation("first", 11, {}, events=[PendingEvent()])
            )
            try:
                with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
                    self.helper.emit_smoke_tensor_event("second", {}, kind="count")
                errors = self.collector.events
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]["event"], "evidence_error")
                self.assertEqual(errors[0]["observation_event"], "second")
                self.assertEqual(errors[0]["reason"], "queue_full")
            finally:
                release.set()
                self.assertTrue(self.helper.flush_smoke_evidence(5))
        self.assertEqual(self.collector.events[-1]["event"], "first")

    def test_flush_without_observer_has_no_cuda_dependency(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertTrue(self.helper.flush_smoke_evidence())

    def test_failed_event_waits_for_retained_stream_before_releasing_source(self):
        release = threading.Event()
        waiting = threading.Event()
        producer_thread = threading.get_ident()
        self_test = self

        class Source:
            pass

        class ProducerStream:
            def synchronize(self):
                self_test.assertNotEqual(threading.get_ident(), producer_thread)
                waiting.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release stream")

        source = Source()
        source_ref = weakref.ref(source)
        worker = self.helper._get_worker()
        self.assertTrue(worker.reserve())
        worker.submit(
            self.helper._Observation(
                "failed_copy",
                41,
                {},
                sources=[source],
                fallback_streams=[ProducerStream()],
                error="event-record-failure",
            )
        )
        del source
        try:
            self.assertTrue(waiting.wait(5))
            self.assertIsNotNone(source_ref())
            self.assertEqual(self.collector.events, [])
            release.set()
            self.assertTrue(self.helper.flush_smoke_evidence(5))
            self.assertIsNone(source_ref())
            self.assertEqual(len(self.collector.events), 1)
            failure = self.collector.events[0]
            self.assertEqual(failure["event"], "evidence_error")
            self.assertEqual(failure["observation_event"], "failed_copy")
            self.assertIn("event-record-failure", failure["detail"])
        finally:
            release.set()
            self.helper.flush_smoke_evidence(5)


class SmokeTensorCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available() or torch.version.hip:
            raise RuntimeError("The CUDA evidence test requires a real CUDA GPU")
        cls.torch = torch
        # Initialize the context and pinned allocator before measuring whether
        # observation returns while producer work is still pending.
        torch.empty((), dtype=torch.int64, pin_memory=True)
        torch.empty((), dtype=torch.int64, device="cuda")

    def setUp(self):
        self.helper = load_helper()
        self.collector = EventCollector()
        self.helper._LOGGER.addHandler(self.collector)
        self.previous_level = self.helper._LOGGER.level
        self.helper._LOGGER.setLevel(logging.INFO)
        self.addCleanup(self.helper._LOGGER.setLevel, self.previous_level)
        self.addCleanup(self.helper._LOGGER.removeHandler, self.collector)
        self.env = mock.patch.dict(os.environ, {"RTP_LLM_SMOKE_EVIDENCE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.helper.flush_smoke_evidence, 30)

    def test_producer_copy_precedes_later_mutation_and_keeps_sources_alive(self):
        torch = self.torch
        stream = torch.cuda.Stream()
        finished = torch.cuda.Event()
        producer_thread = threading.get_ident()
        original_item = torch.Tensor.item
        original_event_wait = torch.cuda.Event.synchronize

        def checked_item(tensor, *args, **kwargs):
            self.assertNotEqual(threading.get_ident(), producer_thread)
            return original_item(tensor, *args, **kwargs)

        def checked_wait(event, *args, **kwargs):
            self.assertNotEqual(threading.get_ident(), producer_thread)
            return original_event_wait(event, *args, **kwargs)

        # Ensure the background observer exists before the producer fixture.
        self.helper._get_worker()
        with mock.patch.object(torch.Tensor, "item", checked_item):
            with mock.patch.object(torch.cuda.Event, "synchronize", checked_wait):
                with mock.patch.object(
                    torch.cuda,
                    "synchronize",
                    side_effect=AssertionError("global CUDA synchronization"),
                ):
                    with torch.cuda.stream(stream):
                        source = torch.empty((), device="cuda", dtype=torch.int64)
                        source_ref = weakref.ref(source)
                        torch.cuda._sleep(500_000_000)
                        source.fill_(17)
                        self.helper.emit_smoke_tensor_event(
                            "snapshot", {"owned_token_rows": source}, kind="page_rr"
                        )
                        source.fill_(99)
                        mutated = source.detach()
                        finished.record(stream)
                    self.assertFalse(
                        finished.query(),
                        "fixture must still have pending producer work",
                    )
                    del source
                    gc.collect()
                    self.assertIsNotNone(source_ref())
                    self.assertTrue(self.helper.flush_smoke_evidence(30))
        finished.synchronize()
        self.assertEqual(mutated.item(), 99)
        self.assertEqual(len(self.collector.events), 1)
        record = self.collector.events[0]
        self.assertEqual(record["event"], "snapshot")
        self.assertEqual(record["owned_token_rows"], 17)
        self.assertEqual(record["kind"], "page_rr")
        self.assertIsInstance(record["time_ns"], int)

    def test_two_int64_counters_are_sampled_without_copying_the_table(self):
        torch = self.torch
        stream = torch.cuda.Stream()
        before = time.time_ns()
        with torch.cuda.stream(stream):
            slots = torch.tensor([3, -1, 5, -1, -1, -1], device="cuda")
            self.helper.emit_smoke_tensor_event(
                "rows",
                {
                    "owned_token_rows": (slots[:3] >= 0).sum(),
                    "padding_owned_slots": (slots[3:] >= 0).sum(),
                },
                kind="page_rr",
            )
            slots.fill_(7)
        after = time.time_ns()
        self.assertTrue(self.helper.flush_smoke_evidence(30))
        self.assertEqual(len(self.collector.events), 1)
        record = self.collector.events[0]
        self.assertEqual(record["owned_token_rows"], 2)
        self.assertEqual(record["padding_owned_slots"], 0)
        self.assertGreaterEqual(record["time_ns"], before)
        self.assertLessEqual(record["time_ns"], after)

    def test_event_record_failure_keeps_copy_alive_until_background_stream_wait(self):
        torch = self.torch
        stream = torch.cuda.Stream()
        finished = torch.cuda.Event()
        producer_thread = threading.get_ident()
        original_wait = torch.cuda.Stream.synchronize
        original_empty = torch.empty_like
        waiting = threading.Event()
        snapshots = []

        def checked_wait(producer, *args, **kwargs):
            self.assertNotEqual(threading.get_ident(), producer_thread)
            waiting.set()
            return original_wait(producer, *args, **kwargs)

        def keep_snapshot(*args, **kwargs):
            result = original_empty(*args, **kwargs)
            snapshots.append(result)
            return result

        self.helper._get_worker()
        with mock.patch.object(torch.cuda.Stream, "synchronize", checked_wait):
            with mock.patch.object(
                torch.cuda,
                "synchronize",
                side_effect=AssertionError("global CUDA synchronization"),
            ):
                with torch.cuda.stream(stream):
                    source = torch.empty((), dtype=torch.int64, device="cuda")
                    source_ref = weakref.ref(source)
                    torch.cuda._sleep(500_000_000)
                    source.fill_(23)
                    with mock.patch.object(torch, "empty_like", keep_snapshot):
                        with mock.patch.object(
                            torch.cuda.Event,
                            "record",
                            side_effect=RuntimeError("event-record-failure"),
                        ):
                            self.helper.emit_smoke_tensor_event(
                                "failed_snapshot", {"count": source}
                            )
                    source.fill_(99)
                    mutated = source.detach()
                    finished.record(stream)
                self.assertTrue(waiting.wait(5))
                self.assertFalse(finished.query())
                del source
                gc.collect()
                self.assertIsNotNone(source_ref())
                self.assertTrue(self.helper.flush_smoke_evidence(30))
        finished.synchronize()
        self.assertEqual(mutated.item(), 99)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].item(), 23)
        self.assertEqual(len(self.collector.events), 1)
        failure = self.collector.events[0]
        self.assertEqual(failure["event"], "evidence_error")
        self.assertEqual(failure["observation_event"], "failed_snapshot")
        self.assertEqual(failure["reason"], "observation_failed")
        self.assertIn("event-record-failure", failure["detail"])

    def test_non_scalar_table_is_rejected_before_pinned_allocation(self):
        torch = self.torch
        table = torch.zeros(8, dtype=torch.int64, device="cuda")
        with mock.patch.object(torch, "empty_like") as allocate:
            self.helper.emit_smoke_tensor_event("table", {"slots": table})
            allocate.assert_not_called()
        self.assertEqual(self.collector.events[0]["event"], "evidence_error")
        self.assertEqual(self.collector.events[0]["reason"], "invalid_observation")

    def test_cuda_graph_capture_does_not_enqueue_observations(self):
        torch = self.torch
        scalar = torch.ones((), dtype=torch.int64, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scalar.add_(1)
            self.helper.emit_smoke_tensor_event("captured", {"count": scalar})
        graph.replay()
        self.assertIsNone(self.helper._worker)
        self.assertEqual(self.collector.events, [])


if __name__ == "__main__":
    unittest.main()
