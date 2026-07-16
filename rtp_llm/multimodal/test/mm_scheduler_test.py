import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import Future
from typing import Any, List, Optional
from unittest import TestCase, main, mock

import torch

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.mm_profiler import MMProfiler
from rtp_llm.multimodal.mm_scheduler import (
    MMScheduler,
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
        self.calls: List[int] = []
        self._lock = threading.Lock()

    @staticmethod
    def _is_poison(data: Any) -> bool:
        # A work item whose preprocess tensor holds a negative value is "poison".
        return isinstance(data, torch.Tensor) and bool((data < 0).any())

    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ) -> List[torch.Tensor]:
        with self._lock:
            self.calls.append(len(data_list))
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


class _FakeWorkItem:
    """Minimal work item exposing only the fields MMScheduler touches."""

    def __init__(
        self,
        images: int = 1,
        timeout_ms: int = 5000,
        mm_type: MMUrlType = MMUrlType.IMAGE,
        preprocess_result: Any = None,
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
        # discarded — no per-request retry runs. submit_and_wait wraps the cause
        # in a RuntimeError, with the typed OOM preserved as __cause__.
        self.assertTrue(all(isinstance(e, RuntimeError) for e in errors), errors)
        self.assertTrue(
            all(isinstance(e.__cause__, torch.cuda.OutOfMemoryError) for e in errors),
            errors,
        )
        self.assertEqual(fake.calls, [3])  # the failed combined forward only

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

        # Every request gets a wrapped RuntimeError whose __cause__ is the typed
        # OutputCountMismatchError.
        self.assertTrue(all(isinstance(e, RuntimeError) for e in errors), errors)
        self.assertTrue(
            all(isinstance(e.__cause__, OutputCountMismatchError) for e in errors),
            errors,
        )
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

        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertIn("unexpected executor error", str(ctx.exception.__cause__))

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


if __name__ == "__main__":
    main()
