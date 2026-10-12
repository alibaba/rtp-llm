"""CPU tests for actual prefetch deadline/lifetime code, with CUDA boundaries mocked."""

import importlib.util
import sys
import threading
import unittest
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

_spec = importlib.util.spec_from_file_location(
    "engram_deadline_test", Path(__file__).resolve().parents[1] / "engram.py"
)
engram = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engram)


class TensorMetadata:
    def __init__(self, rows=4, columns=1):
        self.shape = (rows, columns)
        self.dtype = torch.int64
        self.device = torch.device("cuda:0")
        self.is_cuda = True
        self._version = 0

    def stride(self):
        return (self.shape[1], 1)

    def storage_offset(self):
        return 0

    def is_inference(self):
        return False


class DeferredExecutor:
    """Deterministic FIFO ownership; real futures, no sleeping or GPU context."""

    def __init__(self):
        self.jobs = []

    def submit(self, function, *args):
        future = Future()
        self.jobs.append((future, function, args))
        return future

    def run_all(self):
        while self.jobs:
            future, function, args = self.jobs.pop(0)
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)


class TimeoutCompletionRace(Future):
    def __init__(self, rows, error=None):
        super().__init__()
        self.rows, self.error = rows, error

    def result(self, timeout=None):
        if timeout is not None and not self.done():
            if self.error is None:
                self.set_result(self.rows)
            else:
                self.set_exception(self.error)
            raise TimeoutError("expired just before completion")
        return super().result(timeout)


class EngramCpuDeadlineTest(unittest.TestCase):
    def setUp(self):
        self.current = Mock()
        self.current.query.return_value = False
        self.slot = SimpleNamespace(
            device=torch.device("cuda:0"),
            copied=Mock(),
            deadline_enabled=True,
            collect_timing=True,
            last_timing={},
            heads=1,
        )
        self.runtime = object.__new__(engram.EngramCPULookupRuntime)
        self.runtime.retired = []
        self.runtime.executor = DeferredExecutor()
        self.runtime.capacity = 4
        self.runtime.slots = {1: self.slot}
        self.runtime.num_threads = 16
        for name, value in (
            ("current_stream", Mock(return_value=self.current)),
            ("is_current_stream_capturing", Mock(return_value=False)),
            ("device", lambda device: nullcontext()),
            ("stream", lambda stream: nullcontext()),
        ):
            context = patch.object(torch.cuda, name, value)
            context.start()
            self.addCleanup(context.stop)

    def work(self, *, future=None, ids=None):
        ids = ids or TensorMetadata()
        work = engram._EngramCPULookupWork(self.slot, ids, self.runtime)
        work.allocation_stream = self.current
        work.future = Future() if future is None else future
        work.timing = self.slot.last_timing = {}
        return work, ids

    def test_ready_future_has_no_stream_query_or_host_synchronization(self):
        work, ids = self.work()
        rows = object()
        work.future.set_result(rows)
        self.assertIs(work.consume(ids, TensorMetadata()), rows)
        self.current.query.assert_not_called()
        self.current.wait_event.assert_called_once_with(self.slot.copied)
        self.slot.copied.synchronize.assert_not_called()
        self.assertEqual(self.slot.last_timing["consumer_stream_queries"], 0)
        self.assertEqual(self.slot.last_timing["outcome"], "consumed")
        self.assertIsNone(work.future)
        self.assertIsNone(work.ids)
        work.clear()
        self.assertEqual(self.runtime.retired, [])

    def test_gpu_lead_allows_future_to_finish_without_fallback(self):
        work, ids = self.work()
        rows = object()

        def gpu_still_busy():
            work.future.set_result(rows)
            return False

        self.current.query.side_effect = gpu_still_busy
        self.assertIs(work.consume(ids, TensorMetadata()), rows)
        self.assertFalse(work.abandoned)
        self.assertEqual(self.slot.last_timing["consumer_stream_queries"], 1)

    def test_timeout_completion_race_uses_result_not_stale_timeout(self):
        rows = object()
        work, ids = self.work(future=TimeoutCompletionRace(rows))
        self.assertIs(work.consume(ids, TensorMetadata()), rows)
        self.assertEqual(self.current.query.call_count, 1)

    def test_timeout_completion_race_preserves_worker_exception(self):
        work, ids = self.work(
            future=TimeoutCompletionRace(None, RuntimeError("native failed"))
        )
        with self.assertRaisesRegex(RuntimeError, "native failed"):
            work.consume(ids, TensorMetadata())
        self.current.wait_event.assert_not_called()

    def test_running_lookup_falls_back_without_wait_and_retires_once(self):
        future = Future()
        future.set_running_or_notify_cancel()
        work, ids = self.work(future=future)
        self.current.query.return_value = True
        self.assertIsNone(work.consume(ids, TensorMetadata()))
        self.assertTrue(work.abandoned)
        self.assertTrue(future.running())
        self.slot.copied.synchronize.assert_not_called()
        self.current.wait_event.assert_not_called()
        self.assertEqual(work.timing["outcome"], "gpu_deadline_fallback")
        work.clear()
        work.clear()
        self.assertEqual(len(self.runtime.retired), 1)
        self.assertFalse(self.runtime.reap())
        future.set_result(None)
        self.runtime.executor.run_all()
        self.assertTrue(self.runtime.reap())
        self.assertIsNone(work.future)
        self.slot.copied.synchronize.assert_not_called()

    def test_queued_lookup_cancel_and_same_ids_next_forward_get_fresh_rows(self):
        work, ids = self.work()
        cancelled = work.future
        self.current.query.return_value = True
        self.assertIsNone(work.consume(ids, TensorMetadata()))
        self.assertTrue(cancelled.cancelled())
        self.runtime.executor.run_all()
        self.assertTrue(self.runtime.reap())
        fresh, _ = self.work(ids=ids)
        new_rows = object()
        fresh.future.set_result(new_rows)
        self.assertIs(fresh.consume(ids, TensorMetadata()), new_rows)
        self.assertIsNone(work.future)
        self.assertIsNone(work.ids)

    def test_copy_already_committed_keeps_cpu_result(self):
        rows = object()
        work, ids = self.work(future=TimeoutCompletionRace(rows))
        work.copy_submitted = True
        self.current.query.return_value = True
        self.assertIs(work.consume(ids, TensorMetadata()), rows)
        self.assertFalse(work.abandoned)
        self.assertEqual(self.runtime.retired, [])
        self.current.wait_event.assert_called_once_with(self.slot.copied)

    def test_abandoned_copy_is_kept_until_background_drain_finishes(self):
        work, _ = self.work()
        work.copy_submitted = True
        work.future.set_result(object())
        work.clear()
        self.assertIsNotNone(work.future)
        self.slot.copied.synchronize.assert_not_called()

        def synchronize_with_owner():
            self.assertIsNotNone(work.future)

        self.slot.copied.synchronize.side_effect = synchronize_with_owner
        self.runtime.executor.run_all()
        self.assertTrue(self.runtime.reap())
        self.slot.copied.synchronize.assert_called_once()
        self.assertIsNone(work.future)

    def test_pending_retirement_prevents_new_hashes_and_queue_growth(self):
        pending = Future()
        self.runtime.retired = [pending]
        state = SimpleNamespace(layout=SimpleNamespace(max_ngram_size=4))
        model = SimpleNamespace(
            _engram_cpu_runtime=self.runtime, _engram_hash_state=state
        )
        windows = torch.zeros((4, 4), dtype=torch.int32)
        for _ in range(32):
            plan = engram.EngramCPULookupPrefetch(model, windows, -2)
            self.assertIsNone(plan.hashes)
            self.assertEqual(plan.work, [])
            plan.clear()
        self.assertEqual(len(self.runtime.retired), 1)
        self.assertEqual(self.runtime.executor.jobs, [])
        pending.set_result(None)
        self.assertTrue(self.runtime.reap())

    def test_pending_hash_is_retained_without_blocking_request_clear(self):
        pending = Future()
        pending.set_running_or_notify_cancel()
        plan = object.__new__(engram.EngramCPULookupPrefetch)
        plan.runtime, plan.work, plan.hashes = self.runtime, [], pending
        plan.clear()
        self.assertTrue(pending.running())
        self.assertIsNone(plan.hashes)
        self.assertEqual(self.runtime.retired, [pending])
        plan.clear()
        self.assertEqual(self.runtime.retired, [pending])
        pending.set_result(torch.zeros((4, 1, 1), dtype=torch.int64))
        self.assertTrue(self.runtime.reap())

    def test_identity_or_version_mismatch_abandons_without_host_wait(self):
        for mismatch in ("identity", "version", "stream"):
            with self.subTest(mismatch=mismatch):
                work, ids = self.work()
                future = work.future
                future.set_running_or_notify_cancel()
                consumer_ids = ids
                if mismatch == "identity":
                    consumer_ids = TensorMetadata()
                elif mismatch == "version":
                    ids._version += 1
                else:
                    work.allocation_stream = object()
                self.assertIsNone(work.consume(consumer_ids, TensorMetadata()))
                self.assertTrue(future.running())
                self.assertTrue(work.abandoned)
                future.set_result(None)
                self.runtime.executor.run_all()
                self.assertTrue(self.runtime.reap())
        self.current.query.assert_not_called()
        self.slot.copied.synchronize.assert_not_called()

    def test_reap_surfaces_background_error_once_and_preserves_pending(self):
        failed, done, pending = Future(), Future(), Future()
        failed.set_exception(RuntimeError("retired native failed"))
        done.set_result(None)
        self.runtime.retired = [failed, done, pending]
        with self.assertRaisesRegex(RuntimeError, "retired native failed"):
            self.runtime.reap()
        self.assertEqual(self.runtime.retired, [pending])
        self.assertFalse(self.runtime.reap())
        pending.set_result(None)
        self.assertTrue(self.runtime.reap())

    def test_close_drains_then_surfaces_background_error(self):
        future = Future()
        self.runtime.retired = [future]
        self.runtime._finalizer = lambda: future.set_exception(
            RuntimeError("shutdown failed")
        )
        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            self.runtime.close()
        self.assertEqual(self.runtime.retired, [])

    def test_queued_worker_does_not_become_runtime_final_owner(self):
        entered = threading.Event()

        class RunningLookup(Future):
            def result(self, timeout=None):
                entered.set()
                return super().result(timeout)

        lookup = RunningLookup()
        lookup.set_running_or_notify_cancel()
        executor = ThreadPoolExecutor(max_workers=1)
        original_shutdown = executor.shutdown
        shutdown_threads, closed_threads = [], []
        runtime = object.__new__(engram.EngramCPULookupRuntime)
        runtime.executor, runtime.retired = executor, []
        slot = SimpleNamespace(
            close=lambda: closed_threads.append(threading.get_ident())
        )
        runtime._finalizer = weakref.finalize(
            runtime, engram._close_cpu_lookup, executor, (slot,)
        )
        work = engram._EngramCPULookupWork(self.slot, TensorMetadata(), runtime)
        work.future, work.abandoned = lookup, True
        runtime.retire(work)
        self.assertTrue(entered.wait(5))
        runtime_reference = weakref.ref(runtime)

        def shutdown(*, wait=True):
            shutdown_threads.append(threading.get_ident())
            # Model release starts shutdown on this test thread. Let the
            # blocked lookup finish so the real drain/join can complete.
            if not lookup.done():
                lookup.set_result(None)
            original_shutdown(wait=wait)

        try:
            with patch.object(executor, "shutdown", side_effect=shutdown):
                del work
                del runtime
                self.assertIsNone(runtime_reference())
                self.assertEqual(shutdown_threads, [threading.get_ident()])
                self.assertEqual(closed_threads, [threading.get_ident()])
        finally:
            if not lookup.done():
                lookup.set_result(None)
            original_shutdown(wait=True)

    def test_capture_never_submits_cpu_hash_or_deadline_work(self):
        model = SimpleNamespace(
            _engram_cpu_runtime=self.runtime,
            _engram_hash_state=SimpleNamespace(
                layout=SimpleNamespace(max_ngram_size=4)
            ),
        )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            plan = engram.EngramCPULookupPrefetch(
                model, torch.zeros((4, 4), dtype=torch.int32), -2
            )
        self.assertEqual(plan.work, [])
        self.assertEqual(self.runtime.executor.jobs, [])
        self.current.query.assert_not_called()

    def test_running_native_abandonment_skips_h2d_and_keeps_timing(self):
        entered, finish = threading.Event(), threading.Event()
        slot = object.__new__(engram._EngramCPULookupSlot)
        slot.__dict__.update(self.slot.__dict__)
        slot.stream = Mock()
        slot.embedding = SimpleNamespace(weight=object(), scales=object())
        slot.host = torch.empty((4, 32), dtype=torch.bfloat16)
        slot.hash_index, slot.num_threads = 0, 16
        slot.packed = False
        slot._copy = Mock()
        self.slot = slot
        work, ids = self.work()
        hashes = Future()
        hashes.set_result(torch.zeros((4, 1, 1), dtype=torch.int64))
        output = SimpleNamespace(shape=(4, 32))

        def native(*args, **kwargs):
            entered.set()
            if not finish.wait(5):
                raise RuntimeError("test native release missing")

        package = "rtp_llm.models_py.modules.dsv4.engram_cpu"
        with patch.dict(
            sys.modules, {package: SimpleNamespace(lookup_host_rows_cpu=native)}
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.runtime.executor = executor
                work.future = executor.submit(slot.run, hashes, output, 1, work)
                try:
                    self.assertTrue(entered.wait(5))
                    self.current.query.return_value = True
                    self.assertIsNone(work.consume(ids, TensorMetadata()))
                    self.assertTrue(work.abandoned)
                    slot._copy.assert_not_called()
                finally:
                    finish.set()
            self.assertTrue(self.runtime.reap())
        slot._copy.assert_not_called()
        self.assertEqual(work.timing["outcome"], "gpu_deadline_fallback")
        self.assertIn("lookup_finished_ns", work.timing)
        self.assertIsNone(work.future)

    def test_h2d_commit_race_is_atomic_against_deadline_abandonment(self):
        copying, release, queried = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        slot = object.__new__(engram._EngramCPULookupSlot)
        slot.__dict__.update(self.slot.__dict__)
        slot.stream = Mock()
        slot.embedding = SimpleNamespace(weight=object(), scales=object())
        slot.host = torch.empty((4, 32), dtype=torch.bfloat16)
        slot.hash_index, slot.num_threads, slot.packed = 0, 16, False
        self.slot = slot
        work, ids = self.work()
        hashes = Future()
        hashes.set_result(torch.zeros((4, 1, 1), dtype=torch.int64))
        output = SimpleNamespace(shape=(4, 32))

        def copy(*args):
            copying.set()
            if not release.wait(5):
                raise RuntimeError("test copy release missing")

        def gpu_at_consumer():
            queried.set()
            return True

        slot._copy = Mock(side_effect=copy)
        self.current.query.side_effect = gpu_at_consumer
        package = "rtp_llm.models_py.modules.dsv4.engram_cpu"
        native = SimpleNamespace(lookup_host_rows_cpu=Mock())
        with patch.dict(sys.modules, {package: native}):
            with (
                ThreadPoolExecutor(max_workers=1) as worker,
                ThreadPoolExecutor(max_workers=1) as consumer,
            ):
                self.runtime.executor = worker
                work.future = worker.submit(slot.run, hashes, output, 1, work)
                try:
                    self.assertTrue(copying.wait(5))
                    consumption = consumer.submit(work.consume, ids, TensorMetadata())
                    self.assertTrue(queried.wait(5))
                finally:
                    release.set()
                self.assertIs(consumption.result(timeout=5), output)
        self.assertTrue(work.copy_submitted)
        self.assertFalse(work.abandoned)
        self.assertEqual(self.runtime.retired, [])
        slot._copy.assert_called_once()
        self.current.wait_event.assert_called_once_with(slot.copied)


if __name__ == "__main__":
    unittest.main()
