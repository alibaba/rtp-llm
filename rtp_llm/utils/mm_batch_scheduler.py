"""Bounded, single-executor batching for independent prepared images."""

import logging
import threading
import time
from collections import deque
from concurrent.futures import CancelledError, Future, InvalidStateError
from dataclasses import dataclass
from typing import Any, Callable, List, Optional


@dataclass(frozen=True)
class BatchResult:
    output: Any
    batch_id: int
    batch_size: int


@dataclass
class _ImageWork:
    image: Any
    future: Future
    deadline: Optional[float]
    cancelled: threading.Event

    def error(self):
        if self.cancelled.is_set() or self.future.cancelled():
            return CancelledError("ViT request cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return TimeoutError("ViT request deadline exceeded")
        return None


class MMBatchScheduler:
    def __init__(
        self,
        batch_embedding: Callable[[List[Any]], List[Any]],
        batch_wait_ms: int,
        max_batch_images: int,
        max_batch_patches: int,
    ):
        if batch_wait_ms < 0 or max_batch_images <= 0 or max_batch_patches <= 0:
            raise ValueError("ViT batch wait must be >= 0 and batch limits > 0")
        self._batch_embedding = batch_embedding
        self._wait_seconds = batch_wait_ms / 1000.0
        self._max_images = max_batch_images
        self._max_patches = max_batch_patches
        # At most two batches wait behind the running batch. RPC admission also
        # bounds the number of callers retaining one just-preprocessed image.
        self._queue = deque()
        self._queued_patches = 0
        self._condition = threading.Condition()
        self._closed = False
        self._next_batch_id = 0
        self._thread = threading.Thread(target=self._run, name="vit-batch", daemon=True)
        self._thread.start()

    @staticmethod
    def _fail(work, error):
        try:
            work.future.set_exception(error)
        except InvalidStateError:
            # Future.cancel() can race the request's cancellation Event.
            pass

    def _prune_locked(self):
        live = deque()
        for work in self._queue:
            error = work.error()
            if error is None:
                live.append(work)
            else:
                self._queued_patches -= work.image.num_patches
                self._fail(work, error)
        self._queue = live

    def submit(self, image, deadline=None, cancelled=None) -> Future:
        if not 0 < image.num_patches <= self._max_patches:
            raise ValueError(
                f"image has {image.num_patches} patches; ViT batch limit is {self._max_patches}"
            )
        work = _ImageWork(image, Future(), deadline, cancelled or threading.Event())
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("ViT scheduler is closed")
                error = work.error()
                if error is not None:
                    raise error
                self._prune_locked()
                if (
                    len(self._queue) < 2 * self._max_images
                    and self._queued_patches + image.num_patches
                    <= 2 * self._max_patches
                ):
                    self._queue.append(work)
                    self._queued_patches += image.num_patches
                    self._condition.notify_all()
                    return work.future
                remaining = (
                    0.05 if deadline is None else min(0.05, deadline - time.monotonic())
                )
                self._condition.wait(max(0, remaining))

    def _collect(self):
        with self._condition:
            while not self._closed:
                self._prune_locked()
                if self._queue:
                    break
                self._condition.wait(0.05)
            if self._closed:
                return []
            first = self._queue.popleft()
            patches = first.image.num_patches
            self._queued_patches -= patches
            batch = [first]
            collect_until = time.monotonic() + self._wait_seconds
            if first.deadline is not None:
                collect_until = min(collect_until, first.deadline)
            self._condition.notify_all()
            while len(batch) < self._max_images and not self._closed:
                self._prune_locked()
                if self._queue:
                    candidate = self._queue[0]
                    if patches + candidate.image.num_patches > self._max_patches:
                        break
                    batch.append(self._queue.popleft())
                    patches += candidate.image.num_patches
                    self._queued_patches -= candidate.image.num_patches
                    if candidate.deadline is not None:
                        collect_until = min(collect_until, candidate.deadline)
                    self._condition.notify_all()
                else:
                    remaining = collect_until - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(min(remaining, 0.05))
            if self._closed:
                for work in batch:
                    self._fail(work, RuntimeError("ViT scheduler is closed"))
                return []
            return batch

    def _run(self):
        while True:
            batch = self._collect()
            if not batch:
                return
            self._execute(batch)
            batch.clear()

    def _execute(self, batch):
        live = []
        for work in batch:
            error = work.error()
            if error is not None:
                self._fail(work, error)
            elif work.future.set_running_or_notify_cancel():
                live.append(work)
        if not live:
            return
        self._next_batch_id += 1
        batch_id = self._next_batch_id
        try:
            outputs = self._batch_embedding([work.image for work in live])
            if len(outputs) != len(live):
                raise RuntimeError(
                    f"ViT batch returned {len(outputs)} outputs for {len(live)} images"
                )
            logging.info(
                "ViT batch id=%d images=%d patches=%d",
                batch_id,
                len(live),
                sum(work.image.num_patches for work in live),
            )
            for work, output in zip(live, outputs):
                error = work.error()
                if error is None:
                    work.future.set_result(BatchResult(output, batch_id, len(live)))
                else:
                    self._fail(work, error)
        except Exception as error:
            logging.exception("ViT batch %d failed", batch_id)
            for work in live:
                self._fail(work, error)

    def close(self):
        with self._condition:
            self._closed = True
            for work in self._queue:
                self._fail(work, RuntimeError("ViT scheduler is closed"))
            self._queue.clear()
            self._queued_patches = 0
            self._condition.notify_all()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            logging.warning("ViT batch executor is still finishing its running forward")
