import gc
import json
import os
import sys
import tempfile
import threading
import time
import weakref
from concurrent.futures import Future
from typing import Any, List, Optional
from unittest import TestCase, main, mock

import torch

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.mm_profiler import MMProfiler
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import MMWorkEstimate
from rtp_llm.multimodal.multimodal_util import (
    build_multimodal_output_pb,
    maybe_tensor_to_list,
)
from rtp_llm.multimodal.mm_scheduler import (
    MMScheduler,
    MMSchedulerExecutionError,
    MMSchedulerOverloadError,
    OutputCountMismatchError,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class _FakeMMPart:
    """CPU-only stand-in for a MultiModalEmbeddingInterface.

    Records the size of every batched_embedding call so tests can assert how
    requests were combined, and can optionally block to drive timeout cases.
    """

    def __init__(
        self,
        delay: float = 0.0,
        oom_over: Optional[int] = None,
        short_over: Optional[int] = None,
        block_until: Optional[threading.Event] = None,
        work_budget: Optional[MMWorkEstimate] = None,
    ):
        self.delay = delay
        # Raise a CUDA OOM when a single forward carries more than this many items.
        self.oom_over = oom_over
        # Return one fewer output when a forward carries more than this many items.
        self.short_over = short_over
        # When set, the forward blocks until this event is set (bounded), so a
        # test can hold a forward "in flight" while it drives close().
        self.block_until = block_until
        # Set the first time a forward is entered, so a test can wait until a
        # request is actually inside the (blocked) forward.
        self.forward_entered = threading.Event()
        self.work_budget = work_budget
        self.calls: List[int] = []
        self.call_values: List[List[float]] = []
        self.call_started = threading.Event()
        self._lock = threading.Lock()

    def get_batch_work_budget(self, max_batch_media: int) -> Optional[MMWorkEstimate]:
        return self.work_budget

    @staticmethod
    def _is_poison(data: Any) -> bool:
        # A work item whose preprocess tensor holds a negative value is "poison".
        return isinstance(data, torch.Tensor) and bool((data < 0).any())

    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ) -> List[torch.Tensor]:
        with self._lock:
            self.calls.append(len(data_list))
            self.call_values.append(
                [
                    (
                        float(data.reshape(-1)[0])
                        if isinstance(data, torch.Tensor)
                        else 0.0
                    )
                    for data in data_list
                ]
            )
            self.call_started.set()
        self.forward_entered.set()
        if self.block_until is not None:
            # Bounded so a broken test fails fast instead of hanging forever.
            self.block_until.wait(timeout=30.0)
        if self.delay:
            time.sleep(self.delay)
        if self.oom_over is not None and len(data_list) > self.oom_over:
            raise torch.cuda.OutOfMemoryError("fake CUDA OOM")
        if any(self._is_poison(d) for d in data_list):
            raise RuntimeError("poison item in batch")
        n = len(data_list)
        if self.short_over is not None and len(data_list) > self.short_over:
            n = len(data_list) - 1
        return [torch.zeros(1) for _ in range(n)]


class _OOMOnceMMPart(_FakeMMPart):
    """First forward OOMs with a tensor reachable only from its traceback."""

    def __init__(self):
        super().__init__()
        self._fail_next = True
        self.failed_intermediate_ref = None

    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ) -> List[torch.Tensor]:
        with self._lock:
            self.calls.append(len(data_list))
        self.forward_entered.set()
        if self._fail_next:
            self._fail_next = False
            device = "cuda" if torch.cuda.is_available() else "cpu"
            failed_intermediate = torch.zeros(1, device=device)
            self.failed_intermediate_ref = weakref.ref(failed_intermediate)
            raise torch.cuda.OutOfMemoryError("fake first-batch CUDA OOM")
        return [torch.zeros(1) for _ in data_list]


class _FakeWorkItem:
    """Minimal work item exposing only the fields MMScheduler touches."""

    def __init__(
        self,
        images: int = 1,
        timeout_ms: int = 5000,
        mm_type: MMUrlType = MMUrlType.IMAGE,
        preprocess_result: Any = None,
        input_patches: Optional[int] = None,
    ):
        # mm_inputs is the raw media list; its length is what the scheduler
        # bounds batches by (sum(len(wi.mm_inputs)) across a request).
        self.mm_inputs = [None] * images
        if preprocess_result is None:
            preprocess_result = torch.zeros(1)
        self.preprocess_result: Any = preprocess_result
        self.mm_type = mm_type
        self.mm_timeout_ms = timeout_ms
        self.embedding_result: Optional[Any] = None
        self.need_check_cache = False
        self.cache_key = None
        self.work_estimate = (
            MMWorkEstimate(input_patches=input_patches)
            if input_patches is not None
            else None
        )


class _BlockingClaimFuture(Future):
    """Future that pauses at the scheduler's PENDING-to-RUNNING transition."""

    def __init__(self):
        super().__init__()
        self.claim_entered = threading.Event()
        self.release_claim = threading.Event()

    def set_running_or_notify_cancel(self) -> bool:
        self.claim_entered.set()
        if not self.release_claim.wait(5.0):
            raise TimeoutError("test did not release Future claim")
        return super().set_running_or_notify_cancel()


class _ObservedLock:
    """Lock exposing when a second thread blocks behind its current owner."""

    def __init__(self):
        self._lock = threading.Lock()
        self.blocked_enter = threading.Event()

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.blocked_enter.set()
            self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


def _submit_concurrently(
    sched: MMScheduler,
    requests: List[List[_FakeWorkItem]],
    barrier: Optional[threading.Barrier] = None,
):
    """Submit each request from its own thread; return per-request exceptions.

    If a barrier is given, every thread waits on it immediately before
    submitting, so all requests enter the scheduler together regardless of
    thread-start jitter — useful for asserting on exact batch merging.
    """
    errors: List[Optional[Exception]] = [None] * len(requests)

    def run(i: int, work_items: List[_FakeWorkItem]):
        try:
            if barrier is not None:
                barrier.wait()
            sched.submit_and_wait(work_items)
        except Exception as e:  # noqa: BLE001 - recorded for assertions
            errors[i] = e

    threads = [
        threading.Thread(target=run, args=(i, wis)) for i, wis in enumerate(requests)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


class MMSchedulerTest(TestCase):
    def test_queue_metrics_report_depth_and_wait(self):
        """Queue gauges expose backlog depth and time before a forward starts."""
        fake = _FakeMMPart(delay=0.2)
        sched = MMScheduler(fake, batch_wait_ms=0, max_batch_size=1)
        errors: List[Optional[Exception]] = [None, None]

        def submit(index: int):
            try:
                sched.submit_and_wait([_FakeWorkItem(timeout_ms=5000)])
            except Exception as error:  # noqa: BLE001 - asserted below
                errors[index] = error

        with mock.patch("rtp_llm.multimodal.mm_scheduler.kmonitor.report") as report:
            first = threading.Thread(target=submit, args=(0,))
            second = threading.Thread(target=submit, args=(1,))
            try:
                first.start()
                self.assertTrue(fake.call_started.wait(timeout=1.0))
                second.start()
                # Keep the second request behind the first forward so its queue
                # wait is observable rather than a scheduler race.
                time.sleep(0.03)
                second.join(timeout=2.0)
                first.join(timeout=2.0)
            finally:
                sched.close()

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [None, None])

            depth_values = [
                call.args[1]
                for call in report.call_args_list
                if call.args
                and call.args[0] == GaugeMetrics.VIT_EMBEDDING_QUEUE_SIZE_METRIC
            ]
            wait_values = [
                call.args[1]
                for call in report.call_args_list
                if call.args
                and call.args[0] == GaugeMetrics.VIT_EMBEDDING_QUEUE_WAIT_RT_METRIC
            ]
            self.assertIn(1, depth_values)
            self.assertEqual(depth_values[-1], 0)
            self.assertGreater(max(wait_values), 50.0)

    def test_multi_request_batching(self):
        """Several concurrent submissions are merged into one forward."""
        fake = _FakeMMPart()
        sched = MMScheduler(
            fake, batch_wait_ms=300, max_batch_size=8, max_batch_images=10**9
        )
        # Barrier so all 4 threads submit together rather than racing the
        # 300ms window — keeps the single-forward assertion CI-stable.
        barrier = threading.Barrier(4)
        try:
            errors = _submit_concurrently(
                sched, [[_FakeWorkItem()] for _ in range(4)], barrier=barrier
            )
        finally:
            sched.close()

        self.assertTrue(all(e is None for e in errors), errors)
        self.assertEqual(fake.calls, [4])

    def test_max_batch_size_splits_batches(self):
        """No batch exceeds max_batch_size, and every request is served."""
        fake = _FakeMMPart()
        sched = MMScheduler(
            fake, batch_wait_ms=300, max_batch_size=2, max_batch_images=10**9
        )
        try:
            errors = _submit_concurrently(sched, [[_FakeWorkItem()] for _ in range(5)])
        finally:
            sched.close()

        self.assertTrue(all(e is None for e in errors), errors)
        self.assertTrue(all(c <= 2 for c in fake.calls), fake.calls)
        self.assertEqual(sum(fake.calls), 5)

    def test_max_batch_images_splits_and_rejects(self):
        """Image budget caps batch size; an over-budget request is rejected."""
        fake = _FakeMMPart()
        sched = MMScheduler(
            fake, batch_wait_ms=300, max_batch_size=100, max_batch_images=10
        )
        try:
            # 4 images each -> at most 2 requests per batch (8 <= 10, 12 > 10).
            errors = _submit_concurrently(
                sched, [[_FakeWorkItem(images=4)] for _ in range(5)]
            )
            self.assertTrue(all(e is None for e in errors), errors)
            self.assertTrue(all(c <= 2 for c in fake.calls), fake.calls)
            self.assertEqual(sum(fake.calls), 5)

            # A single request above the whole budget is rejected up front.
            with self.assertRaisesRegex(ValueError, "exceeds gpu_max_batch_images"):
                sched.submit_and_wait([_FakeWorkItem(images=20)])
        finally:
            sched.close()

    def test_timeout_cancel(self):
        """A request that outlasts its mm_timeout_ms raises TimeoutError."""
        fake = _FakeMMPart(delay=1.0)
        sched = MMScheduler(fake, batch_wait_ms=10)
        try:
            with self.assertRaisesRegex(TimeoutError, "timeout"):
                sched.submit_and_wait([_FakeWorkItem(timeout_ms=100)])
        finally:
            sched.close()

    def test_multiple_work_items_use_largest_timeout(self):
        fake = _FakeMMPart(delay=0.15)
        sched = MMScheduler(fake, batch_wait_ms=0)
        try:
            sched.submit_and_wait(
                [
                    _FakeWorkItem(timeout_ms=50),
                    _FakeWorkItem(timeout_ms=1000),
                ]
            )
            self.assertEqual(fake.calls, [2])
        finally:
            sched.close()

    def test_none_timeout_falls_back(self):
        """mm_timeout_ms=None must not crash submit_and_wait (falls back)."""
        fake = _FakeMMPart()
        sched = MMScheduler(fake, batch_wait_ms=10)
        try:
            # Without the None guard the timeout calc raises TypeError; with it,
            # the request runs to completion under the default fallback timeout.
            sched.submit_and_wait([_FakeWorkItem(timeout_ms=None)])
        finally:
            sched.close()

    def test_images_summed_across_work_items_for_reject(self):
        """Per-request reject sums mm_inputs across all the request's work items."""
        fake = _FakeMMPart()
        sched = MMScheduler(fake, max_batch_images=5)
        try:
            # 3 + 3 = 6 images > 5 -> rejected, even though no single item exceeds.
            with self.assertRaisesRegex(ValueError, "exceeds gpu_max_batch_images"):
                sched.submit_and_wait(
                    [_FakeWorkItem(images=3), _FakeWorkItem(images=3)]
                )
        finally:
            sched.close()

    def test_cost_budget_splits_cross_request_batch(self):
        """Model work, not just media count, limits cross-request packing."""
        fake = _FakeMMPart(work_budget=MMWorkEstimate(input_patches=10))
        sched = MMScheduler(
            fake, batch_wait_ms=300, max_batch_size=8, max_batch_images=100
        )
        barrier = threading.Barrier(4)
        try:
            errors = _submit_concurrently(
                sched,
                [[_FakeWorkItem(input_patches=6)] for _ in range(4)],
                barrier=barrier,
            )
        finally:
            sched.close()

        self.assertTrue(all(e is None for e in errors), errors)
        self.assertEqual(fake.calls, [1, 1, 1, 1])

    def test_cost_aware_model_splits_large_request(self):
        """An opted-in model advances an oversized request in bounded chunks."""
        fake = _FakeMMPart(work_budget=MMWorkEstimate(input_patches=10))
        sched = MMScheduler(fake, batch_wait_ms=0, max_batch_size=8, max_batch_images=5)
        items = [
            _FakeWorkItem(images=3, input_patches=6),
            _FakeWorkItem(images=3, input_patches=6),
        ]
        try:
            sched.submit_and_wait(items)
        finally:
            sched.close()

        self.assertEqual(fake.calls, [1, 1])
        self.assertTrue(all(item.embedding_result is not None for item in items))

    def test_split_request_yields_to_waiting_request(self):
        """A large request queues each next chunk at the tail for fairness."""
        fake = _FakeMMPart(
            delay=0.05,
            work_budget=MMWorkEstimate(input_patches=10),
        )
        sched = MMScheduler(fake, batch_wait_ms=0, max_batch_size=1, max_batch_images=5)
        large_items = [
            _FakeWorkItem(
                preprocess_result=torch.tensor([1.0]),
                input_patches=6,
            ),
            _FakeWorkItem(
                preprocess_result=torch.tensor([2.0]),
                input_patches=6,
            ),
        ]
        small_item = _FakeWorkItem(
            preprocess_result=torch.tensor([3.0]),
            input_patches=1,
        )
        large_error: List[Exception] = []

        def submit_large_request():
            try:
                sched.submit_and_wait(large_items)
            except Exception as error:
                large_error.append(error)

        large_thread = threading.Thread(target=submit_large_request)
        try:
            large_thread.start()
            self.assertTrue(fake.call_started.wait(timeout=1.0))
            sched.submit_and_wait([small_item])
            large_thread.join(timeout=1.0)
        finally:
            sched.close()

        self.assertFalse(large_thread.is_alive())
        self.assertEqual(large_error, [])
        self.assertEqual(fake.call_values, [[1.0], [3.0], [2.0]])

    def test_cost_aware_model_requires_work_estimate(self):
        """Opting into cost admission requires every preprocessed item to estimate."""
        fake = _FakeMMPart(work_budget=MMWorkEstimate(input_patches=10))
        sched = MMScheduler(fake, max_batch_images=5)
        try:
            with self.assertRaisesRegex(RuntimeError, "has no work estimate"):
                sched.submit_and_wait([_FakeWorkItem()])
        finally:
            sched.close()

    def test_cost_aware_model_requires_typed_work_estimate(self):
        """A non-null estimate still has to satisfy the generic cost contract."""
        fake = _FakeMMPart(work_budget=MMWorkEstimate(input_patches=10))
        sched = MMScheduler(fake, max_batch_images=5)
        item = _FakeWorkItem()
        item.work_estimate = object()
        try:
            with self.assertRaisesRegex(TypeError, "must be MMWorkEstimate"):
                sched.submit_and_wait([item])
        finally:
            sched.close()

    def test_cost_aware_model_runs_oversized_single_item_alone(self):
        """One indivisible item may exceed the soft model budget."""
        fake = _FakeMMPart(work_budget=MMWorkEstimate(input_patches=10))
        sched = MMScheduler(fake, batch_wait_ms=0, max_batch_size=8, max_batch_images=5)
        item = _FakeWorkItem(images=1, input_patches=15)
        try:
            sched.submit_and_wait([item])
        finally:
            sched.close()

        self.assertEqual(fake.calls, [1])
        self.assertIsNotNone(item.embedding_result)

    def test_failure_isolated_across_batches(self):
        """A failing forward only fails its own batch; other batches still succeed."""
        fake = _FakeMMPart()
        # max_batch_size=1 -> every request is its own batch, fully isolated.
        sched = MMScheduler(fake, batch_wait_ms=10, max_batch_size=1)
        try:
            requests = [
                [_FakeWorkItem()],  # good
                [_FakeWorkItem(preprocess_result=torch.tensor([-1.0]))],  # poison
                [_FakeWorkItem()],  # good
            ]
            errors = _submit_concurrently(sched, requests)
        finally:
            sched.close()

        self.assertIsNone(errors[0])
        self.assertIsInstance(errors[1], RuntimeError)
        self.assertIsNone(errors[2])

    def test_oom_fails_whole_batch(self):
        """A batch that OOMs fails every request in it; there is no retry."""
        fake = _FakeMMPart(oom_over=1)  # any forward with >1 item OOMs
        sched = MMScheduler(
            fake, batch_wait_ms=300, max_batch_size=8, max_batch_images=10**9
        )
        try:
            errors = _submit_concurrently(sched, [[_FakeWorkItem()] for _ in range(3)])
        finally:
            sched.close()

        # All-or-nothing: the combined forward OOMs and the whole batch is
        # discarded — no per-request retry runs. Futures retain only lightweight
        # failure metadata, never the original OOM or its traceback.
        self.assertTrue(
            all(isinstance(e, MMSchedulerExecutionError) for e in errors), errors
        )
        self.assertTrue(
            all(
                e.source_type == torch.cuda.OutOfMemoryError.__name__
                for e in errors
            ),
            errors,
        )
        self.assertTrue(all(e.is_oom for e in errors), errors)
        self.assertTrue(all(e.__cause__ is None for e in errors), errors)
        self.assertEqual(fake.calls, [3])  # the failed combined forward only

    def test_oom_releases_failed_forward_before_smaller_batch(self):
        """An OOM traceback must not retain its tensor into the next forward."""
        fake = _OOMOnceMMPart()
        sched = MMScheduler(
            fake, batch_wait_ms=300, max_batch_size=8, max_batch_images=10**9
        )
        cleanup_order = []
        real_gc_collect = gc.collect

        def collect():
            cleanup_order.append("gc")
            return real_gc_collect()

        def empty_cache():
            cleanup_order.append("empty_cache")

        try:
            with mock.patch(
                "rtp_llm.multimodal.mm_scheduler.gc.collect", side_effect=collect
            ), mock.patch(
                "rtp_llm.multimodal.mm_scheduler.torch.cuda.empty_cache",
                side_effect=empty_cache,
            ):
                barrier = threading.Barrier(3)
                errors = _submit_concurrently(
                    sched,
                    [[_FakeWorkItem()] for _ in range(3)],
                    barrier=barrier,
                )

                self.assertTrue(
                    all(isinstance(e, MMSchedulerExecutionError) for e in errors),
                    errors,
                )
                self.assertTrue(all(e.is_oom for e in errors), errors)
                self.assertIsNotNone(fake.failed_intermediate_ref)
                self.assertIsNone(
                    fake.failed_intermediate_ref(),
                    "failed forward tensor is still retained by an exception",
                )
                self.assertEqual(cleanup_order, ["gc", "empty_cache"])

                # The same worker must remain usable after recovery. A smaller
                # second batch succeeds instead of inheriting the first OOM.
                sched.submit_and_wait([_FakeWorkItem()])
        finally:
            sched.close()

        self.assertEqual(fake.calls, [3, 1])

    def test_count_mismatch_fails_whole_batch(self):
        """A short combined return fails the whole batch; there is no retry."""
        fake = _FakeMMPart(short_over=1)  # any forward with >1 item returns short
        sched = MMScheduler(
            fake, batch_wait_ms=300, max_batch_size=8, max_batch_images=10**9
        )
        try:
            errors = _submit_concurrently(sched, [[_FakeWorkItem()] for _ in range(3)])
        finally:
            sched.close()

        # The original mismatch traceback is logged on the worker and discarded;
        # callers receive only its type/message metadata.
        self.assertTrue(
            all(isinstance(e, MMSchedulerExecutionError) for e in errors), errors
        )
        self.assertTrue(
            all(e.source_type == OutputCountMismatchError.__name__ for e in errors),
            errors,
        )
        self.assertTrue(all(not e.is_oom for e in errors), errors)
        self.assertTrue(all(e.__cause__ is None for e in errors), errors)
        self.assertEqual(fake.calls, [3])

    def test_executor_loop_error_unblocks_caller(self):
        """An unexpected error escaping _execute_batch must unblock the caller
        with that error, not leave it hanging until its submit timeout."""
        fake = _FakeMMPart()
        sched = MMScheduler(fake, batch_wait_ms=10)

        def boom(batch):
            raise RuntimeError("unexpected executor error")

        # Simulate a bug that escapes _execute_batch before it sets done.
        sched._execute_batch = boom
        try:
            # timeout_ms is generous: if the loop's fallback did not fire, the
            # caller would block on it; instead it returns promptly with the error.
            with self.assertRaises(RuntimeError) as ctx:
                sched.submit_and_wait([_FakeWorkItem(timeout_ms=10000)])
        finally:
            sched.close()

        self.assertEqual(ctx.exception.source_type, "RuntimeError")
        self.assertIn("unexpected executor error", ctx.exception.source_message)
        self.assertIsNone(ctx.exception.__cause__)

    def test_close_during_collection_window_rejects_promptly(self):
        """close() while a request sits in the collection window (before any
        forward starts) must reject it immediately, not leave it blocked until
        its mm_timeout_ms elapses."""
        fake = _FakeMMPart()
        # A long wait window keeps the lone request parked in _collect_batch
        # (waiting for batch-mates that never arrive), so it has not entered a
        # forward yet when close() fires.
        sched = MMScheduler(fake, batch_wait_ms=5000, max_batch_size=8)

        errors: List[Optional[Exception]] = [None]
        submitted = threading.Event()

        def submit():
            submitted.set()
            try:
                # Generous timeout: if the request were left hanging it would
                # block here for 30s; the close() path must resolve it far sooner.
                sched.submit_and_wait([_FakeWorkItem(timeout_ms=30000)])
            except Exception as e:  # noqa: BLE001 - recorded for assertions
                errors[0] = e

        t = threading.Thread(target=submit)
        t.start()
        # Ensure the request is enqueued and the executor has pulled it into the
        # collection window before we close.
        submitted.wait()
        time.sleep(0.2)

        start = time.monotonic()
        sched.close()
        t.join(timeout=5.0)
        elapsed = time.monotonic() - start

        self.assertFalse(t.is_alive(), "submit_and_wait did not unblock on close()")
        # Resolved promptly by close() (stop noticed within the poll interval),
        # NOT by waiting out the 5s collection window or the 30s submit timeout.
        self.assertLess(elapsed, 2.0)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("closed before request completed", str(errors[0]))
        # No forward ever ran: close() rejected before entering batched_embedding.
        self.assertEqual(fake.calls, [])

    def test_close_timeout_still_completes_queued(self):
        """A forward that outlasts close()'s join must not strand queued
        requests: once the forward returns the executor drains and fails them,
        so their callers unblock with a close error instead of waiting out their
        submit timeout."""
        release = threading.Event()
        fake = _FakeMMPart(block_until=release)
        # max_batch_size=1 so request A is its own forward and B stays queued.
        sched = MMScheduler(fake, batch_wait_ms=0, max_batch_size=1)

        errors: List[Optional[Exception]] = [None, None]

        def submit(i: int):
            try:
                sched.submit_and_wait([_FakeWorkItem(timeout_ms=30000)])
            except Exception as e:  # noqa: BLE001 - recorded for assertions
                errors[i] = e

        ta = threading.Thread(target=submit, args=(0,))
        ta.start()
        # Wait until A is actually inside the (blocked) forward.
        self.assertTrue(fake.forward_entered.wait(2.0))

        tb = threading.Thread(target=submit, args=(1,))
        tb.start()
        # Let B settle into the waiting queue behind the in-flight forward.
        time.sleep(0.2)

        start = time.monotonic()
        # join times out: A's forward is blocked, so close() returns early.
        sched.close(timeout=0.5)
        self.assertLess(time.monotonic() - start, 3.0)
        self.assertTrue(ta.is_alive(), "in-flight forward should still be running")

        # Releasing the forward lets A finish; the executor then exits and its
        # finally drains B, unblocking B's caller with the close error.
        release.set()
        ta.join(5.0)
        tb.join(5.0)
        total = time.monotonic() - start

        self.assertFalse(ta.is_alive())
        self.assertFalse(tb.is_alive(), "queued request was not unblocked on close")
        # A was already in flight -> allowed to finish successfully.
        self.assertIsNone(errors[0])
        # B never started -> failed with the close cause, well before its 30s timeout.
        self.assertIsInstance(errors[1], RuntimeError)
        self.assertIn("closed before request completed", str(errors[1]))
        self.assertLess(total, 10.0)

    def test_queue_full_rejects_then_drains(self):
        """A stalled forward must not grow the queue unbounded: once the bounded
        queue is full, submit fails fast with MMSchedulerOverloadError; after the
        forward unsticks the backlog drains and those requests still succeed."""
        release = threading.Event()
        fake = _FakeMMPart(block_until=release)
        # max_batch_size=1 so A is a single forward; capacity 2 behind it.
        sched = MMScheduler(fake, batch_wait_ms=0, max_batch_size=1, max_queue_size=2)

        errors: dict = {}
        threads: List[threading.Thread] = []

        def submit(name: str):
            try:
                sched.submit_and_wait([_FakeWorkItem(timeout_ms=30000)])
            except Exception as e:  # noqa: BLE001 - recorded for assertions
                errors[name] = e

        # A occupies the executor in a blocked forward.
        ta = threading.Thread(target=submit, args=("A",))
        ta.start()
        threads.append(ta)
        self.assertTrue(fake.forward_entered.wait(2.0))

        # Fill the bounded queue (capacity 2) behind the stalled forward.
        for name in ("B", "C"):
            t = threading.Thread(target=submit, args=(name,))
            t.start()
            threads.append(t)
        deadline = time.monotonic() + 2.0
        while sched._waiting.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(sched._waiting.qsize(), 2)

        # Queue full -> the next submit fails fast, before blocking on its future.
        with self.assertRaises(MMSchedulerOverloadError):
            sched.submit_and_wait([_FakeWorkItem(timeout_ms=30000)])

        # Unstick: A finishes, then B and C drain and succeed; the queue empties.
        release.set()
        for t in threads:
            t.join(5.0)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(errors, {})
        self.assertEqual(sched._waiting.qsize(), 0)
        sched.close()

    def test_close_between_collect_and_execute_starts_no_forward(self):
        """close() landing after a batch is collected but before it is claimed
        must reject it under the shared lock — no NEW forward starts (TOCTOU)."""
        collected = threading.Event()
        release = threading.Event()
        orig_collect = MMScheduler._collect_batch

        def slow_collect(scheduler):
            batch = orig_collect(scheduler)
            if batch is not None:
                # Park AFTER collecting, BEFORE the loop's stopped-check, so the
                # test can force close() into exactly that window.
                collected.set()
                release.wait()
            return batch

        fake = _FakeMMPart()
        errors: List[Optional[Exception]] = [None]

        with mock.patch.object(MMScheduler, "_collect_batch", slow_collect):
            # Patch the class before construction so the worker cannot enter the
            # original method before the test installs its collection barrier.
            sched = MMScheduler(fake, batch_wait_ms=0, max_batch_size=1)

            def submit():
                try:
                    sched.submit_and_wait([_FakeWorkItem(timeout_ms=30000)])
                except Exception as e:  # noqa: BLE001 - recorded for assertions
                    errors[0] = e

            t = threading.Thread(target=submit)
            close_thread = threading.Thread(target=sched.close)
            try:
                t.start()
                self.assertTrue(collected.wait(2.0))

                # close() wins the scheduler lock while the executor is parked
                # after collection but before _claim_batch.
                close_thread.start()
                self.assertTrue(sched._stopped.wait(2.0))
                release.set()
            finally:
                release.set()
                sched.close(timeout=5.0)
                if close_thread.ident is not None:
                    close_thread.join(3.0)
                t.join(3.0)

        self.assertFalse(t.is_alive())
        self.assertFalse(close_thread.is_alive())
        # The forward never ran and was never registered in-flight: the under-lock
        # check saw stopped and rejected.
        self.assertEqual(fake.calls, [])
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("closed before request completed", str(errors[0]))

    def test_close_after_stop_check_waits_for_claimed_batch(self):
        """If claim wins the scheduler lock, close treats that batch as in flight."""
        release_forward = threading.Event()
        fake = _FakeMMPart(block_until=release_forward)
        sched = MMScheduler(fake, batch_wait_ms=0, max_batch_size=1)
        observed_lock = _ObservedLock()
        sched._lock = observed_lock

        future = _BlockingClaimFuture()
        submit_error: List[Optional[Exception]] = [None]
        close_result: List[Optional[bool]] = [None]

        def submit():
            try:
                sched.submit_and_wait([_FakeWorkItem(timeout_ms=30000)])
            except Exception as e:  # noqa: BLE001 - recorded for assertions
                submit_error[0] = e

        def close():
            close_result[0] = sched.close(timeout=5.0)

        submit_thread = threading.Thread(target=submit)
        close_thread = threading.Thread(target=close)
        try:
            with mock.patch(
                "rtp_llm.multimodal.mm_scheduler.Future", return_value=future
            ):
                submit_thread.start()
                self.assertTrue(future.claim_entered.wait(2.0))

                # _claim_batch holds the scheduler lock after checking _stopped but
                # before marking this Future RUNNING. Prove close() is waiting for
                # that same lock, then let the claim complete.
                close_thread.start()
                self.assertTrue(observed_lock.blocked_enter.wait(2.0))
                future.release_claim.set()

                # Claim won, so close may set _stopped now but cannot reject the
                # batch. It must join until the claimed forward returns.
                self.assertTrue(fake.forward_entered.wait(2.0))
                close_thread.join(timeout=0.1)
                self.assertTrue(close_thread.is_alive())
        finally:
            future.release_claim.set()
            release_forward.set()
            sched.close(timeout=5.0)
            if close_thread.ident is not None:
                close_thread.join(3.0)
            submit_thread.join(3.0)

        self.assertFalse(close_thread.is_alive())
        self.assertFalse(submit_thread.is_alive())
        self.assertTrue(close_result[0])
        self.assertIsNone(submit_error[0])
        self.assertEqual(fake.calls, [1])

    def test_device_bind_failure_fails_construction(self):
        """A set_device failure in the executor thread must fail construction
        fast (fail-fast handshake), not leave requests to time out."""
        fake = _FakeMMPart()
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.set_device", side_effect=RuntimeError("bad device")
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to bind device"):
                MMScheduler(fake, device="cuda:7")

    def test_device_cuda_no_index_skips_bind(self):
        """device='cuda' (no index, exactly what the standalone VIT service
        passes) must NOT call set_device — binding a bare cuda device raises — so
        startup succeeds and uses the current device."""
        fake = _FakeMMPart()
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.set_device"
        ) as set_device:
            sched = MMScheduler(fake, device="cuda")
            sched.close()
        set_device.assert_not_called()

    def test_device_cuda_index_binds(self):
        """device='cuda:N' pins that exact index (non-zero local rank coverage)."""
        fake = _FakeMMPart()
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.set_device"
        ) as set_device:
            sched = MMScheduler(fake, device="cuda:3")
            sched.close()
        set_device.assert_called_once_with(3)

    def test_forward_profiler_captures_batched_embedding(self):
        """The forward_profiler hook runs on the executor thread around the
        forward, so the exported trace contains the batched_embedding event —
        i.e. the core GPU compute is captured, not lost to the worker thread."""
        profiler = MMProfiler()
        with tempfile.TemporaryDirectory() as tmp:
            # Inject the output dir (rank omitted keeps this path), then arm 1.
            profiler._output_path = tmp
            profiler.start_profile(count=1)

            fake = _FakeMMPart()
            sched = MMScheduler(
                fake, batch_wait_ms=0, forward_profiler=profiler.profile_forward
            )
            try:
                sched.submit_and_wait([_FakeWorkItem()])
            finally:
                sched.close()

            # profile_forward exports the trace before the future resolves, so it
            # exists by the time submit_and_wait returns.
            trace_path = os.path.join(tmp, "timeline_0.json")
            self.assertTrue(os.path.exists(trace_path), "forward trace not exported")
            with open(trace_path) as f:
                trace = json.load(f)
            names = {e.get("name", "") for e in trace.get("traceEvents", [])}
            self.assertIn("batched_embedding", names)

    def test_invalid_params_rejected(self):
        """Non-positive limits / negative wait are rejected at construction."""
        fake = _FakeMMPart()
        with self.assertRaisesRegex(ValueError, "batch_wait_ms must be >= 0"):
            MMScheduler(fake, batch_wait_ms=-1)
        with self.assertRaisesRegex(ValueError, "max_batch_size must be > 0"):
            MMScheduler(fake, max_batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_batch_images must be > 0"):
            MMScheduler(fake, max_batch_images=0)
        with self.assertRaisesRegex(ValueError, "max_queue_size must be > 0"):
            MMScheduler(fake, max_queue_size=0)


class VitEmbeddingSchedulerArgsTest(TestCase):
    def test_default_serial_policy(self):
        cfg = VitConfig()
        args = cfg.embedding_scheduler_args()

        self.assertEqual(args["max_batch_size"], 1)
        self.assertEqual(args["batch_wait_ms"], 0)
        self.assertEqual(args["max_batch_images"], sys.maxsize)

    def test_batch_policy_preserves_config(self):
        cfg = VitConfig()
        cfg.gpu_max_batch_size = 8
        cfg.gpu_batch_wait_ms = 20
        cfg.gpu_max_batch_images = 200

        args = cfg.embedding_scheduler_args()
        self.assertEqual(args["max_batch_size"], 8)
        self.assertEqual(args["batch_wait_ms"], 20)
        self.assertEqual(args["max_batch_images"], 200)

    def test_invalid_policy_values_rejected(self):
        cfg = VitConfig()
        cfg.gpu_max_batch_size = 0
        with self.assertRaises(ValueError):
            cfg.embedding_scheduler_args()

        cfg.gpu_max_batch_size = 8
        cfg.gpu_batch_wait_ms = -1
        with self.assertRaises(ValueError):
            cfg.embedding_scheduler_args()


class MMWorkEstimateTest(TestCase):
    def test_add_scale_and_budget(self):
        first = MMWorkEstimate(
            input_patches=4,
            output_tokens=2,
            estimated_workspace_bytes=40,
            max_attention_segment=3,
            attention_work=9,
        )
        second = MMWorkEstimate(
            input_patches=5,
            output_tokens=3,
            estimated_workspace_bytes=50,
            max_attention_segment=4,
            attention_work=16,
        )

        total = first + second
        self.assertEqual(total.input_patches, 9)
        self.assertEqual(total.output_tokens, 5)
        self.assertEqual(total.estimated_workspace_bytes, 90)
        self.assertEqual(total.max_attention_segment, 4)
        self.assertEqual(total.attention_work, 25)
        self.assertTrue(total.fits_within(total))
        self.assertFalse(total.fits_within(MMWorkEstimate(input_patches=8)))

        scaled = first.scaled(3)
        self.assertEqual(scaled.input_patches, 12)
        self.assertEqual(scaled.max_attention_segment, 3)
        self.assertEqual(scaled.attention_work, 27)

    def test_negative_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "input_patches"):
            MMWorkEstimate(input_patches=-1)


class MaybeTensorToListTest(TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(maybe_tensor_to_list(None), [])

    def test_non_tensor_returned_as_is(self):
        obj = ["a", "b"]
        self.assertIs(maybe_tensor_to_list(obj), obj)

    def test_at_or_below_dim_wraps_single(self):
        t = torch.zeros(3, 4)  # 2-D, ndim_threshold=2 -> single-element list
        out = maybe_tensor_to_list(t, ndim_threshold=2)
        self.assertEqual(len(out), 1)
        self.assertIs(out[0], t)

    def test_above_dim_splits_leading(self):
        t = torch.zeros(5, 3, 4)  # 3-D, ndim_threshold=2 -> split into 5 of (3, 4)
        out = maybe_tensor_to_list(t, ndim_threshold=2)
        self.assertEqual(len(out), 5)
        self.assertEqual(tuple(out[0].shape), (3, 4))

    def test_extra_input_dim1(self):
        t = torch.zeros(2, 6)  # 2-D, ndim_threshold=1 -> split into 2 of shape (6,)
        out = maybe_tensor_to_list(t, ndim_threshold=1)
        self.assertEqual(len(out), 2)
        self.assertEqual(tuple(out[0].shape), (6,))


class BuildMultimodalOutputPbTest(TestCase):
    def test_empty_embeddings_returns_empty_pb(self):
        pb = build_multimodal_output_pb([], [], [])
        self.assertEqual(list(pb.split_size), [])
        self.assertFalse(pb.HasField("multimodal_pos_id"))
        self.assertEqual(len(pb.multimodal_extra_input), 0)

    def test_split_size_and_fields(self):
        embeddings = [torch.randn(2, 4), torch.randn(3, 4)]
        position_ids = [torch.zeros(2, 3), torch.zeros(3, 3)]
        extra_input = [torch.zeros(7), torch.zeros(9)]
        pb = build_multimodal_output_pb(embeddings, position_ids, extra_input)

        # split_size records the per-image leading dim so the receiver can re-split.
        self.assertEqual(list(pb.split_size), [2, 3])
        self.assertTrue(pb.HasField("multimodal_embedding"))
        self.assertTrue(pb.HasField("multimodal_pos_id"))
        self.assertEqual(len(pb.multimodal_extra_input), 2)

    def test_without_position_or_extra(self):
        pb = build_multimodal_output_pb([torch.randn(1, 4)], [], [])
        self.assertEqual(list(pb.split_size), [1])
        self.assertTrue(pb.HasField("multimodal_embedding"))
        self.assertFalse(pb.HasField("multimodal_pos_id"))
        self.assertEqual(len(pb.multimodal_extra_input), 0)


if __name__ == "__main__":
    main()
