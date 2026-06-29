from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.profiler

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import vit_emb_cache_
from rtp_llm.utils.time_util import Timer, current_time_ms

if TYPE_CHECKING:
    from rtp_llm.multimodal.mm_process_engine import MMWorkItem


class OutputCountMismatchError(RuntimeError):
    """batched_embedding returned a number of outputs != number of inputs.

    Raised by _run_embedding to fail loudly instead of letting zip silently
    truncate. A distinct type so the caller can tell a count mismatch apart from
    a forward (device) error in logs and error messages.
    """


def _run_embedding(
    mm_part: MultiModalEmbeddingInterface,
    items: List[MMWorkItem],
) -> None:
    """Run one GPU forward over `items` and write results back.

    Performs the batched forward, guards the output count, writes
    embedding_result onto each work item, and inserts cacheable results into
    vit_emb_cache_. Any forward error or count mismatch propagates unchanged.
    """
    data_list = [wi.preprocess_result for wi in items]
    type_list = [wi.mm_type for wi in items]

    with Timer() as route_timer:
        with torch.profiler.record_function("batched_embedding"):
            batch_outputs = mm_part.batched_embedding(data_list, type_list)
    # VIT_EMBEDDING_RT_METRIC times ONLY the batched_embedding forward (one batch
    # may carry multiple requests). The end-to-end per-request latency (queue wait
    # + batch-collect wait + forward) is the separate VIT_EMBEDDING_BATCH_RT_METRIC
    # reported in submit_and_wait.
    kmonitor.report(GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms())

    # Guard the zip below: a short return would silently leave some work items
    # with embedding_result=None, and a long one would drop outputs.
    if len(batch_outputs) != len(items):
        raise OutputCountMismatchError(
            f"batched_embedding returned {len(batch_outputs)} outputs "
            f"for {len(items)} work items"
        )

    for wi, result in zip(items, batch_outputs):
        wi.embedding_result = result
        if wi.need_check_cache:
            vit_emb_cache_.insert_cache(wi.cache_key, result)


class _EmbeddingRequest:
    """A single caller's submission to the GPU batch scheduler.

    Lifecycle: created by submit_and_wait -> put on _waiting -> pulled into a
    batch and run by the executor thread -> done.set() (with exception set on
    failure) -> submit_and_wait wakes and returns or raises.

    Fields:
      work_items: the caller's work items, run together in one forward.
      n_images:   raw media count (sum of len(wi.mm_inputs)); the unit batches
                  are bounded by. Computed once here since mm_inputs never
                  changes, so the batch-budget checks don't re-sum it.
      exception:  failure cause set by the executor before done.set(); None on
                  success.
      done:       signaled once the request is finished (success or failure).
      cancelled:  set by submit_and_wait on timeout so the executor skips it.
                  A plain bool (not an Event) is fine: one writer (the caller),
                  read-only for the executor, and the attribute read/write is
                  atomic under the GIL — a rare stale read only skips, or fails to
                  skip, one already-doomed request, never corrupts state.
    """

    __slots__ = ("work_items", "n_images", "exception", "done", "cancelled")

    def __init__(self, work_items: List[MMWorkItem]):
        self.work_items = work_items
        self.n_images = sum(len(wi.mm_inputs) for wi in work_items)
        self.exception: Optional[Exception] = None
        self.done = threading.Event()
        self.cancelled = False


# Fallback wait when a work item carries no positive mm_timeout_ms (e.g. a
# VitConfig built without arg parsing). Sourced from VitConfig's single default
# so the server arg, VitConfig, and this fallback never drift.
_DEFAULT_MM_TIMEOUT_MS = VitConfig.DEFAULT_MM_TIMEOUT_MS

# How often the idle executor re-checks _stopped while waiting for the first
# request of a batch. A new submission wakes the blocked get() immediately
# (queue.Queue.get returns on put, not on timeout), so this interval adds NO
# request latency — it only bounds how quickly the executor notices close().
# One scheduler per process, so the periodic idle wake-ups are negligible.
_STOP_POLL_INTERVAL_S = 0.01


class MMScheduler:
    """A background thread turns submitted work items into embeddings.

    Within a wait window it merges concurrent submissions into a single GPU
    forward, bounded by max_batch_size (requests) and max_batch_images (media).
    Set max_batch_size=1 (with batch_wait_ms=0) for serial, one-request-per-
    forward behavior — no cross-request batching."""

    def __init__(
        self,
        mm_part: MultiModalEmbeddingInterface,
        batch_wait_ms: int = 10,
        max_batch_size: int = 8,
        max_batch_images: int = 32,
    ):
        if batch_wait_ms < 0:
            raise ValueError(f"batch_wait_ms must be >= 0, got {batch_wait_ms}")
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}")
        if max_batch_images <= 0:
            raise ValueError(f"max_batch_images must be > 0, got {max_batch_images}")

        self._mm_part = mm_part
        self._batch_wait_ms = batch_wait_ms
        self._max_batch_size = max_batch_size
        self._max_batch_images = max_batch_images

        self._waiting: queue.Queue[_EmbeddingRequest] = queue.Queue()
        # A request popped from _waiting that would have overflowed the current
        # batch's image budget — carried over as the first request of the next
        # round so it is neither lost nor re-ordered behind newer arrivals.
        self._pending: Optional[_EmbeddingRequest] = None
        # Set by close(); the executor polls it to exit and submit rejects on it.
        self._stopped = threading.Event()
        # Orders submit's (stopped-check + enqueue) against close's set-stopped
        # so a submission can't slip in after close has drained the queue.
        self._lock = threading.Lock()

        self._executor = threading.Thread(
            target=self._executor_loop, daemon=True, name="mm-scheduler"
        )
        self._executor.start()

    def submit_and_wait(self, work_items: List[MMWorkItem]) -> None:
        req = _EmbeddingRequest(work_items)
        # max_batch_images is also the SINGLE-request image cap, not just the
        # cross-request batch cap: a request is never split across batches, so one
        # carrying more images than the cap can never fit and is rejected up front
        # (no graceful degradation / auto-splitting). This only bites when GPU
        # batching is on — serial mode passes sys.maxsize, preserving the old
        # inline path's "no single-request image limit". Operators enabling GPU
        # batching must set gpu_max_batch_images >= the largest single-request
        # image count their traffic can produce.
        if req.n_images > self._max_batch_images:
            raise ValueError(
                f"request image count {req.n_images} exceeds "
                f"gpu_max_batch_images {self._max_batch_images}, "
                f"request rejected"
            )

        # mm_timeout_ms is normally a positive int (server default 120000), but
        # guard against an unset/None/non-positive value so max()/division can't
        # raise — fall back to the default rather than wait unbounded.
        positive_timeouts = [
            wi.mm_timeout_ms
            for wi in work_items
            if wi.mm_timeout_ms is not None and wi.mm_timeout_ms > 0
        ]
        timeout_s = max(positive_timeouts, default=_DEFAULT_MM_TIMEOUT_MS) / 1000.0

        submit_ms = current_time_ms()

        # Hold the lock only across the stopped-check and the enqueue so it is
        # atomic w.r.t. close(); the blocking wait below stays outside the lock.
        with self._lock:
            if self._stopped.is_set():
                raise RuntimeError("MMScheduler is closed, request rejected")
            self._waiting.put(req)

        if not req.done.wait(timeout=timeout_s):
            req.cancelled = True
            waited_ms = current_time_ms() - submit_ms
            logging.warning(
                "MMScheduler: embedding wait timeout after %.0fms "
                "(queue_depth=%d, batch_wait_ms=%d)",
                waited_ms,
                self._waiting.qsize(),
                self._batch_wait_ms,
            )
            raise TimeoutError(
                f"MMScheduler: embedding wait timeout after {timeout_s * 1000:.0f}ms"
            )

        if req.exception:
            # Wrap so the caller gets a stable type plus the original cause chain
            # (__cause__) for debugging; the typed original is still inspectable
            # via the chain and was already logged with exc_info in _execute_batch.
            raise RuntimeError(
                f"batch embedding failed: {req.exception}"
            ) from req.exception

        # Success only: keep the latency gauge clean (failures are tracked by the
        # engine's VIT_ERROR_QPS_METRIC). Mirrors VIT_EMBEDDING_RT_METRIC, which is
        # likewise reported only when the forward completes.
        kmonitor.report(
            GaugeMetrics.VIT_EMBEDDING_BATCH_RT_METRIC, current_time_ms() - submit_ms
        )

    @staticmethod
    def _drain(
        q: "queue.Queue[_EmbeddingRequest]",
    ) -> List[_EmbeddingRequest]:
        """Pop and return every request currently queued."""
        drained: List[_EmbeddingRequest] = []
        while True:
            try:
                drained.append(q.get_nowait())
            except queue.Empty:
                break
        return drained

    def _executor_loop(self) -> None:
        while not self._stopped.is_set():
            batch = None
            try:
                batch = self._collect_batch()
                if batch is None:
                    break
                self._execute_batch(batch)
            except Exception as e:
                logging.error(f"MMScheduler: executor loop error: {e}", exc_info=True)
                # _execute_batch normally fails its own requests; reaching here
                # means something unexpected escaped it (or _collect_batch) before
                # done was set. Fail any still-pending request so its caller gets
                # the error now instead of blocking until its own submit timeout.
                # The loop keeps running so the consumer thread survives.
                if batch:
                    for req in batch:
                        if not req.done.is_set():
                            req.exception = e
                            req.done.set()

    def _collect_batch(self) -> Optional[List[_EmbeddingRequest]]:
        # Pick the first request, skipping any whose caller already timed out
        # (cancelled) so a dead request neither anchors a batch nor spends its
        # image budget. _pending (carried over from last round) goes first.
        #
        # The idle wait polls _stopped (see _STOP_POLL_INTERVAL_S) rather than
        # blocking forever, so close() needs no wake-up sentinel: a new request
        # still wakes get() immediately, and close() is noticed within one poll.
        # Once stopped, start NO new batch — return None so the executor exits
        # and close() fails whatever is still queued. Only the batch already
        # being collected/run is allowed to finish; the rest of the queue is
        # dropped (not processed).
        while True:
            if self._stopped.is_set():
                return None
            if self._pending is not None:
                first = self._pending
                self._pending = None
            else:
                try:
                    first = self._waiting.get(timeout=_STOP_POLL_INTERVAL_S)
                except queue.Empty:
                    continue
            if not first.cancelled:
                break
        batch = [first]
        n_images = first.n_images

        deadline = time.monotonic() + self._batch_wait_ms / 1000.0

        while len(batch) < self._max_batch_size and n_images < self._max_batch_images:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self._waiting.get(timeout=remaining)
            except queue.Empty:
                break
            if req.cancelled:
                continue  # caller already timed out; don't spend budget on it

            # The while guard already caps the request count, so here we only
            # need to stop when the next request would overflow the image budget.
            if n_images + req.n_images > self._max_batch_images:
                self._pending = req
                break
            batch.append(req)
            n_images += req.n_images

        return batch

    def _execute_batch(self, batch: List[_EmbeddingRequest]) -> None:
        """Run the batched GPU forward and write results back.

        All-or-nothing: if the forward raises, the whole batch is discarded and
        every request in it receives that same exception. There is no per-request
        retry — a co-batched request can fail as collateral, the deliberate trade
        for a single, simple failure path. Cross-batch isolation still holds: only
        this batch is affected; earlier/later batches are untouched.
        """
        # Single cancellation checkpoint: drop every request whose caller has
        # already timed out, so the forward never runs for work nobody awaits.
        batch = [req for req in batch if not req.cancelled]
        if not batch:
            return

        items = [wi for req in batch for wi in req.work_items]
        try:
            _run_embedding(self._mm_part, items)
        except Exception as e:
            # On OOM, reset the allocator so the *next* batch starts from a clean
            # state; other errors (device assert, count mismatch) need no reset.
            # The original typed exception (OutOfMemoryError /
            # OutputCountMismatchError / RuntimeError) propagates unchanged so the
            # caller sees a clear cause.
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            logging.error(
                f"MMScheduler: batch forward failed, discarding {len(batch)} "
                f"request(s): {type(e).__name__}: {e}",
                exc_info=True,
            )
            for req in batch:
                req.exception = e
                req.done.set()
            return

        for req in batch:
            req.done.set()

    def close(self, timeout: float = 10.0) -> None:
        # Set stopped atomically w.r.t. submit_and_wait so that after this point
        # no new request can be enqueued (all are rejected) and the drain below
        # sees every request that did make it in. The executor polls _stopped
        # and exits on its own — no wake-up sentinel needed.
        with self._lock:
            if self._stopped.is_set():
                return
            self._stopped.set()
        self._executor.join(timeout=timeout)
        if self._executor.is_alive():
            # Executor still running (a forward is stuck). Don't touch _pending /
            # _waiting here — that would race the live executor's _collect_batch.
            # Any not-yet-started request is bounded by its own submit_and_wait
            # timeout, so it still fails, just later.
            logging.warning("MMScheduler: executor join exceeded %.0fs", timeout)
            return

        # Executor has exited, so _pending / _waiting have no concurrent accessor:
        # fail only requests that never started (those still queued and the one
        # carried over in _pending). In-flight requests were finished by the
        # executor before it exited (result or error).
        exc = RuntimeError("MMScheduler closed before request completed")
        queued = [self._pending] if self._pending else []
        self._pending = None
        queued.extend(self._drain(self._waiting))
        for req in queued:
            req.exception = exc
            req.done.set()
