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
    MMWorkEstimate,
    MultiModalEmbeddingInterface,
)
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
    embedding_result onto each work item, and completes its cache claim. Any
    forward error or count mismatch propagates unchanged.
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
        complete_cache = getattr(wi, "complete_cache", None)
        if complete_cache is not None:
            complete_cache(result)


class _EmbeddingRequest:
    """A single caller's submission to the GPU batch scheduler.

    A request may be split into bounded chunks. Only its next chunk is queued;
    after that chunk completes, the executor appends the following one at the
    queue tail. This prevents one large request from monopolizing the scheduler
    while preserving the caller's original work-item/result order.
    """

    __slots__ = (
        "work_items",
        "chunks",
        "next_chunk_index",
        "remaining_chunks",
        "exception",
        "done",
        "cancelled",
    )

    def __init__(self, work_items: List[MMWorkItem]):
        self.work_items = work_items
        self.chunks: List[_EmbeddingChunk] = []
        self.next_chunk_index = 0
        self.remaining_chunks = 0
        self.exception: Optional[Exception] = None
        self.done = threading.Event()
        self.cancelled = False


class _EmbeddingChunk:
    """An indivisible scheduler unit belonging to one caller request."""

    __slots__ = ("request", "work_items", "n_images", "work_estimate")

    def __init__(
        self,
        request: _EmbeddingRequest,
        work_items: List[MMWorkItem],
        n_images: int,
        work_estimate: Optional[MMWorkEstimate],
    ):
        self.request = request
        self.work_items = work_items
        self.n_images = n_images
        self.work_estimate = work_estimate


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
    forward, bounded by max_batch_size, max_batch_images, and an optional
    model-derived work budget.
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
        self._work_budget = mm_part.get_batch_work_budget(max_batch_images)
        if self._work_budget is not None and not isinstance(
            self._work_budget, MMWorkEstimate
        ):
            raise TypeError(
                "get_batch_work_budget must return MMWorkEstimate or None, got "
                f"{type(self._work_budget).__name__}"
            )
        if self._work_budget is not None:
            logging.info("MMScheduler: model work budget=%s", self._work_budget)

        self._waiting: queue.Queue[_EmbeddingChunk] = queue.Queue()
        # A chunk popped from _waiting that would have overflowed the current
        # batch's media/work budget is carried over as the first chunk of the next
        # round so it is neither lost nor re-ordered behind newer arrivals.
        self._pending: Optional[_EmbeddingChunk] = None
        # Set by close(); the executor polls it to exit and submit rejects on it.
        self._stopped = threading.Event()
        # Orders submit's (stopped-check + enqueue) against close's set-stopped
        # so a submission can't slip in after close has drained the queue.
        self._lock = threading.Lock()

        self._executor = threading.Thread(
            target=self._executor_loop, daemon=True, name="mm-scheduler"
        )
        self._executor.start()

    @staticmethod
    def _sum_work_estimates(
        work_items: List[MMWorkItem],
    ) -> Optional[MMWorkEstimate]:
        total = MMWorkEstimate()
        for work_item in work_items:
            estimate = getattr(work_item, "work_estimate", None)
            if estimate is None:
                return None
            total = total + estimate
        return total

    def _would_exceed_work_budget(
        self,
        current: Optional[MMWorkEstimate],
        candidate: Optional[MMWorkEstimate],
    ) -> bool:
        if self._work_budget is None or current is None or candidate is None:
            return False
        budget = self._work_budget
        additive_fields = (
            "input_patches",
            "output_tokens",
            "estimated_workspace_bytes",
            "attention_work",
        )
        for field_name in additive_fields:
            limit = getattr(budget, field_name)
            if (
                limit > 0
                and getattr(current, field_name) + getattr(candidate, field_name)
                > limit
            ):
                return True
        return (
            budget.max_attention_segment > 0
            and max(
                current.max_attention_segment,
                candidate.max_attention_segment,
            )
            > budget.max_attention_segment
        )

    def _build_chunks(self, request: _EmbeddingRequest) -> None:
        if not request.work_items:
            raise ValueError("MMScheduler requires at least one work item")

        if len(request.work_items) == 1:
            work_item = request.work_items[0]
            n_images = len(work_item.mm_inputs)
            if n_images > self._max_batch_images:
                raise ValueError(
                    f"single work item image count {n_images} exceeds "
                    f"gpu_max_batch_images {self._max_batch_images}; "
                    "the model preprocess batch is not splittable"
                )
            work_estimate = getattr(work_item, "work_estimate", None)
            if self._work_budget is not None:
                if work_estimate is None:
                    raise RuntimeError(
                        "model enabled cost-aware multimodal scheduling, but a "
                        "preprocessed work item has no work estimate"
                    )
                if not isinstance(work_estimate, MMWorkEstimate):
                    raise TypeError(
                        "cost-aware multimodal work estimate must be "
                        f"MMWorkEstimate, got {type(work_estimate).__name__}"
                    )
                if not work_estimate.fits_within(self._work_budget):
                    logging.warning(
                        "MMScheduler: one work item exceeds the model work "
                        "budget; running it alone (estimate=%s, budget=%s)",
                        work_estimate,
                        self._work_budget,
                    )
            request.chunks = [
                _EmbeddingChunk(
                    request=request,
                    work_items=request.work_items,
                    n_images=n_images,
                    work_estimate=work_estimate,
                )
            ]
            request.remaining_chunks = 1
            request.next_chunk_index = 1
            return

        if self._work_budget is None:
            n_images = sum(len(work_item.mm_inputs) for work_item in request.work_items)
            if n_images > self._max_batch_images:
                raise ValueError(
                    f"request image count {n_images} exceeds "
                    f"gpu_max_batch_images {self._max_batch_images}, "
                    "request rejected"
                )
            request.chunks = [
                _EmbeddingChunk(
                    request=request,
                    work_items=request.work_items,
                    n_images=n_images,
                    work_estimate=self._sum_work_estimates(request.work_items),
                )
            ]
            request.remaining_chunks = 1
            request.next_chunk_index = 1
            return

        chunks: List[_EmbeddingChunk] = []
        chunk_items: List[MMWorkItem] = []
        chunk_images = 0
        chunk_work = MMWorkEstimate()

        def finish_chunk() -> None:
            nonlocal chunk_items, chunk_images, chunk_work
            if not chunk_items:
                return
            chunks.append(
                _EmbeddingChunk(
                    request=request,
                    work_items=chunk_items,
                    n_images=chunk_images,
                    work_estimate=chunk_work,
                )
            )
            chunk_items = []
            chunk_images = 0
            chunk_work = MMWorkEstimate()

        for work_item in request.work_items:
            item_images = len(work_item.mm_inputs)
            if item_images > self._max_batch_images:
                raise ValueError(
                    f"single work item image count {item_images} exceeds "
                    f"gpu_max_batch_images {self._max_batch_images}; "
                    "the model preprocess batch is not splittable"
                )

            item_work = getattr(work_item, "work_estimate", None)
            if item_work is None:
                raise RuntimeError(
                    "model enabled cost-aware multimodal scheduling, but a "
                    "preprocessed work item has no work estimate"
                )
            if not isinstance(item_work, MMWorkEstimate):
                raise TypeError(
                    "cost-aware multimodal work estimate must be "
                    f"MMWorkEstimate, got {type(item_work).__name__}"
                )
            image_overflow = (
                bool(chunk_items)
                and chunk_images + item_images > self._max_batch_images
            )
            work_overflow = bool(chunk_items) and self._would_exceed_work_budget(
                chunk_work, item_work
            )
            if image_overflow or work_overflow:
                logging.info(
                    "MMScheduler: split request before work item "
                    "(reason=%s, chunk_images=%d, item_images=%d, "
                    "chunk_work=%s, item_work=%s, budget=%s)",
                    "media" if image_overflow else "work",
                    chunk_images,
                    item_images,
                    chunk_work,
                    item_work,
                    self._work_budget,
                )
                finish_chunk()

            chunk_items.append(work_item)
            chunk_images += item_images
            chunk_work = chunk_work + item_work

            if len(chunk_items) == 1 and not item_work.fits_within(self._work_budget):
                # A model work item is not generically splittable (for example,
                # one long video). Run it alone rather than reintroduce the old
                # whole-request rejection; a true OOM still reaches the caller.
                logging.warning(
                    "MMScheduler: one work item exceeds the model work budget; "
                    "running it alone (estimate=%s, budget=%s)",
                    item_work,
                    self._work_budget,
                )

        finish_chunk()
        request.chunks = chunks
        request.remaining_chunks = len(chunks)
        request.next_chunk_index = 1

    def submit_and_wait(self, work_items: List[MMWorkItem]) -> None:
        req = _EmbeddingRequest(work_items)
        self._build_chunks(req)

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
            self._waiting.put(req.chunks[0])

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
        q: "queue.Queue[_EmbeddingChunk]",
    ) -> List[_EmbeddingChunk]:
        """Pop and return every chunk currently queued."""
        drained: List[_EmbeddingChunk] = []
        while True:
            try:
                drained.append(q.get_nowait())
            except queue.Empty:
                break
        return drained

    @staticmethod
    def _fail_chunks(chunks: List[_EmbeddingChunk], error: Exception) -> None:
        seen_requests = set()
        for chunk in chunks:
            request = chunk.request
            request_id = id(request)
            if request_id in seen_requests or request.done.is_set():
                continue
            seen_requests.add(request_id)
            request.cancelled = True
            request.exception = error
            request.done.set()

    def _complete_chunk(self, chunk: _EmbeddingChunk) -> None:
        request = chunk.request
        if request.cancelled or request.done.is_set():
            return

        request.remaining_chunks -= 1
        if request.remaining_chunks == 0:
            request.done.set()
            return

        with self._lock:
            if self._stopped.is_set():
                request.cancelled = True
                request.exception = RuntimeError(
                    "MMScheduler closed before request completed"
                )
                request.done.set()
                return
            next_chunk = request.chunks[request.next_chunk_index]
            request.next_chunk_index += 1
            self._waiting.put(next_chunk)

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
                    self._fail_chunks(batch, e)

    def _collect_batch(self) -> Optional[List[_EmbeddingChunk]]:
        # Pick the first chunk, skipping any whose caller already timed out
        # (cancelled) so dead work neither anchors a batch nor spends its budget.
        # _pending (carried over from last round) goes first.
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
            if not first.request.cancelled:
                break
        batch = [first]
        n_images = first.n_images
        batch_work = first.work_estimate

        deadline = time.monotonic() + self._batch_wait_ms / 1000.0

        while len(batch) < self._max_batch_size and n_images < self._max_batch_images:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                chunk = self._waiting.get(timeout=remaining)
            except queue.Empty:
                break
            if chunk.request.cancelled:
                continue  # caller already timed out; don't spend budget on it

            image_overflow = n_images + chunk.n_images > self._max_batch_images
            work_overflow = self._would_exceed_work_budget(
                batch_work, chunk.work_estimate
            )
            if image_overflow or work_overflow:
                self._pending = chunk
                break
            batch.append(chunk)
            n_images += chunk.n_images
            if batch_work is None or chunk.work_estimate is None:
                batch_work = None
            else:
                batch_work = batch_work + chunk.work_estimate

        return batch

    def _run_items_with_oom_split(self, items: List[MMWorkItem]) -> None:
        """Run one chunk, recursively halving its items after a CUDA OOM."""
        try:
            _run_embedding(self._mm_part, items)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(items) <= 1:
                raise
            midpoint = len(items) // 2
            logging.warning(
                "MMScheduler: OOM retry splitting one request chunk "
                "from %d items into %d + %d",
                len(items),
                midpoint,
                len(items) - midpoint,
            )
            self._run_items_with_oom_split(items[:midpoint])
            self._run_items_with_oom_split(items[midpoint:])

    def _execute_batch(self, batch: List[_EmbeddingChunk]) -> None:
        """Run a batch, isolating CUDA OOMs by binary split and retry."""
        # Drop chunks whose callers already timed out, so the forward never runs
        # for work nobody awaits.
        batch = [chunk for chunk in batch if not chunk.request.cancelled]
        if not batch:
            return

        items = [wi for chunk in batch for wi in chunk.work_items]
        log_composition = logging.getLogger().isEnabledFor(logging.INFO)
        if log_composition:
            n_images = sum(chunk.n_images for chunk in batch)
            work_estimate = self._sum_work_estimates(items)
            t0 = time.time()
        try:
            _run_embedding(self._mm_part, items)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            if self._work_budget is None:
                logging.error(
                    "MMScheduler: batch OOM with cost-aware scheduling disabled: %s",
                    error,
                    exc_info=True,
                )
                self._fail_chunks(batch, error)
                return
            if len(batch) > 1:
                midpoint = len(batch) // 2
                logging.warning(
                    "MMScheduler: OOM retry splitting batch from %d chunks "
                    "into %d + %d",
                    len(batch),
                    midpoint,
                    len(batch) - midpoint,
                )
                self._execute_batch(batch[:midpoint])
                self._execute_batch(batch[midpoint:])
                return

            chunk = batch[0]
            if len(chunk.work_items) > 1:
                midpoint = len(chunk.work_items) // 2
                try:
                    self._run_items_with_oom_split(chunk.work_items[:midpoint])
                    self._run_items_with_oom_split(chunk.work_items[midpoint:])
                except Exception as retry_error:
                    logging.error(
                        "MMScheduler: single request OOM retry failed: " "%s: %s",
                        type(retry_error).__name__,
                        retry_error,
                        exc_info=True,
                    )
                    self._fail_chunks(batch, retry_error)
                    return
                self._complete_chunk(chunk)
                return

            logging.error(
                "MMScheduler: one work item still OOM after isolation: %s",
                error,
                exc_info=True,
            )
            self._fail_chunks(batch, error)
            return
        except Exception as error:
            logging.error(
                "MMScheduler: batch forward failed, discarding %d chunk(s): " "%s: %s",
                len(batch),
                type(error).__name__,
                error,
                exc_info=True,
            )
            self._fail_chunks(batch, error)
            return

        if log_composition:
            dt = (time.time() - t0) * 1000
            if work_estimate is None:
                logging.info(
                    "[SCHEDULER] requests=%d items=%d imgs=%d forward=%.0fms",
                    len(batch),
                    len(items),
                    n_images,
                    dt,
                )
            else:
                logging.info(
                    "[SCHEDULER] requests=%d items=%d imgs=%d patches=%d "
                    "tokens=%d workspace=%.1fMiB forward=%.0fms",
                    len(batch),
                    len(items),
                    n_images,
                    work_estimate.input_patches,
                    work_estimate.output_tokens,
                    work_estimate.estimated_workspace_bytes / (1024 * 1024),
                    dt,
                )

        for chunk in batch:
            self._complete_chunk(chunk)

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

        # Executor has exited, so _pending / _waiting have no concurrent accessor.
        # Fail requests with a chunk that never started.
        exc = RuntimeError("MMScheduler closed before request completed")
        queued = [self._pending] if self._pending else []
        self._pending = None
        queued.extend(self._drain(self._waiting))
        self._fail_chunks(queued, exc)
