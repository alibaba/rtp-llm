import threading
import time
from typing import Any, List, Optional
from unittest import TestCase, main

import torch

from rtp_llm.multimodal.mm_scheduler import MMScheduler, OutputCountMismatchError
from rtp_llm.multimodal.multimodal_util import (
    build_multimodal_output_pb,
    maybe_tensor_to_list,
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
    ):
        self.delay = delay
        # Raise a CUDA OOM when a single forward carries more than this many items.
        self.oom_over = oom_over
        # Return one fewer output when a forward carries more than this many items.
        self.short_over = short_over
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

    def test_invalid_params_rejected(self):
        """Non-positive limits / negative wait are rejected at construction."""
        fake = _FakeMMPart()
        with self.assertRaisesRegex(ValueError, "batch_wait_ms must be >= 0"):
            MMScheduler(fake, batch_wait_ms=-1)
        with self.assertRaisesRegex(ValueError, "max_batch_size must be > 0"):
            MMScheduler(fake, max_batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_batch_images must be > 0"):
            MMScheduler(fake, max_batch_images=0)


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
