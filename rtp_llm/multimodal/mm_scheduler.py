from __future__ import annotations

import gc
import logging
import queue
import threading
import time
import traceback
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import torch.profiler

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MMWorkEstimate,
    MultiModalEmbeddingInterface,
)
from rtp_llm.utils.time_util import Timer, current_time_ms

if TYPE_CHECKING:
    from rtp_llm.multimodal.mm_process_engine import MMWorkItem


class MMSchedulerError(RuntimeError):
    """Base for errors raised by the multimodal embedding scheduler.

    Boundaries (gRPC / local) can catch this to handle any scheduler error, or a
    specific subclass to map a precise status (e.g. overload -> RESOURCE_EXHAUSTED).
    """


class OutputCountMismatchError(MMSchedulerError):
    """batched_embedding returned a number of outputs != number of inputs.

    A distinct worker-side type so the lightweight caller error can retain an
    unambiguous source_type without retaining this exception's traceback.
    """


class MMSchedulerExecutionError(MMSchedulerError):
    """A claimed batch failed while executing on the scheduler worker."""

    def __init__(
        self, source_type: str, source_message: str, is_oom: bool = False
    ) -> None:
        self.source_type = source_type
        self.source_message = source_message
        self.is_oom = is_oom
        super().__init__(
            f"batch embedding failed: {source_type}: {source_message}"
        )


class MMSchedulerOverloadError(MMSchedulerError):
    """Submission rejected because the waiting queue is full.

    A distinct type so callers can map overload to a 503-style response instead
    of a generic error.
    """


class MMSchedulerTimeoutError(MMSchedulerError, TimeoutError):
    """Timeout while waiting for the scheduler queue/forward result.

    Also a TimeoutError for backward-compatible callers; the RPC boundary maps it
    to gRPC DEADLINE_EXCEEDED.
    """


class MMSchedulerRequestTooLargeError(MMSchedulerError, ValueError):
    """A single request exceeds a per-request limit (e.g. image count) and can
    never fit a batch. Also a ValueError for backward-compatible callers; the RPC
    boundary maps it to gRPC INVALID_ARGUMENT.
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
    # Forward-only latency, sampled per merged batch -> its own metric. The
    # historical per-request (wait + forward) latency stays on
    # VIT_EMBEDDING_RT_METRIC, reported in submit_and_wait.
    kmonitor.report(GaugeMetrics.VIT_EMBEDDING_FORWARD_RT_METRIC, route_timer.cost_ms())

    # A short/long return would silently mis-pair work items with outputs.
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

    Lifecycle: submit_and_wait creates it -> _waiting -> the executor pulls it
    into a batch -> future resolved (set_result / set_exception) -> the caller's
    future.result() returns or re-raises. On timeout the caller cancels the
    future and the executor skips it.

    Fields:
      work_items: run together in one forward.
      n_images:   media count (sum of len(wi.mm_inputs)); the unit batches are
                  bounded by. Cached since mm_inputs never changes.
      future:     result/exception AND cancellation channel. The executor
                  resolves it once, from the executor thread; the caller blocks
                  on future.result(timeout). The Future gives the happens-before
                  between set_* and result() for free. On failure each request
                  gets its OWN lightweight wrapper (see _fail), containing only
                  type/message metadata rather than the worker exception and its
                  GPU-tensor-retaining traceback. Cancellation uses future.cancel()
                  + set_running_or_notify_cancel() (see _claim_batch), whose
                  shared lock resolves the cancel-vs-start race.

    A request whose model reports a work budget is split into bounded chunks.
    Only the next chunk is queued; the executor appends the following one at the
    queue tail once it completes. This keeps one large request from monopolizing
    the scheduler while preserving the caller's work-item/result order. The
    future still resolves exactly once, after the last chunk.
    """

    __slots__ = (
        "work_items",
        "n_images",
        "future",
        "chunks",
        "next_chunk_index",
        "remaining_chunks",
        "running_claimed",
        "abandoned",
    )

    def __init__(self, work_items: List[MMWorkItem]):
        self.work_items = work_items
        self.n_images = sum(len(wi.mm_inputs) for wi in work_items)
        self.future: Future[None] = Future()
        self.chunks: List[_EmbeddingChunk] = []
        self.next_chunk_index = 0
        self.remaining_chunks = 0
        # set_running_or_notify_cancel() may be called at most once per Future,
        # so only the request's first claimed chunk transitions it to RUNNING.
        self.running_claimed = False
        # Set when the caller's wait times out. future.cancel() is a no-op once
        # the request is RUNNING, so a split request needs this to stop queueing
        # its remaining chunks.
        self.abandoned = False


class _EmbeddingChunk:
    """An indivisible scheduler unit belonging to one caller request."""

    __slots__ = (
        "request",
        "work_items",
        "n_images",
        "work_estimate",
        "enqueued_at",
        "queue_wait_reported",
    )

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
        self.enqueued_at: Optional[float] = None
        self.queue_wait_reported = False


# Fallback for hand-built work items without a positive request timeout.
_DEFAULT_MM_TIMEOUT_MS = VitConfig.DEFAULT_MM_TIMEOUT_MS

# How often the idle executor re-checks _stopped. A new submission wakes the
# blocked get() immediately, so this adds no request latency — it only bounds
# how fast the executor notices close().
_STOP_POLL_INTERVAL_S = 0.01


def _cuda_device_index(device: Optional[str]) -> Optional[int]:
    """CUDA index to bind for `device`, or None to leave the current device.

    Only an explicit ``cuda:N`` pins a specific device. A bare ``cuda`` (no index,
    as the standalone VIT service passes) has index None, and binding it via
    ``set_device(torch.device("cuda"))`` raises — so we skip and use the thread's
    current device. Non-cuda / None also skip.
    """
    if not device:
        return None
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return dev.index  # None for bare "cuda"


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
        max_batch_images: int = 200,
        max_queue_size: int = 1024,
        device: Optional[str] = None,
        forward_profiler: Optional[Callable[[], AbstractContextManager]] = None,
    ):
        if batch_wait_ms < 0:
            raise ValueError(f"batch_wait_ms must be >= 0, got {batch_wait_ms}")
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}")
        if max_batch_images <= 0:
            raise ValueError(f"max_batch_images must be > 0, got {max_batch_images}")
        if max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be > 0, got {max_queue_size}")

        self._mm_part = mm_part
        self._batch_wait_ms = batch_wait_ms
        self._max_batch_size = max_batch_size
        self._max_batch_images = max_batch_images
        # Device the forward must run on. The executor is a fresh thread, which
        # defaults to cuda:0; without pinning, a non-zero local rank would run the
        # forward on the wrong device. None (tests / CPU) skips pinning.
        self._device = device
        # Optional per-forward profiling hook: a factory returning a context
        # manager, entered on THIS executor thread around the forward so the GPU
        # compute is actually captured (torch.profiler doesn't span threads). None
        # -> no profiling. Attribution is per forward/batch.
        self._forward_profiler = forward_profiler

        # Bounded so a stalled forward can't let cancelled/waiting requests (and
        # the preprocessed tensors they pin) grow without limit; over capacity,
        # submit fails fast with MMSchedulerOverloadError.
        self._waiting: queue.Queue[_EmbeddingChunk] = queue.Queue(
            maxsize=max_queue_size
        )
        # A chunk that would have overflowed the batch's image/work budget,
        # carried to the next round so it is neither lost nor re-ordered behind
        # newer arrivals.
        self._pending: Optional[_EmbeddingChunk] = None
        # Model-provided cost budget for one forward. None (the interface
        # default) keeps the legacy media-count-only admission, in which a
        # request is never split.
        self._work_budget: Optional[MMWorkEstimate] = mm_part.get_batch_work_budget(
            max_batch_images
        )
        # Set by close(); the executor polls it to exit and submit rejects on it.
        self._stopped = threading.Event()
        # Orders submit's (stopped-check + enqueue) against close's set-stopped
        # so a submission can't slip in after close has drained the queue.
        self._lock = threading.Lock()

        # Startup handshake: the executor binds the device before signaling ready,
        # so a bad device fails construction fast instead of stranding every
        # request on its submit timeout.
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None

        self._executor = threading.Thread(
            target=self._executor_loop, daemon=True, name="mm-scheduler"
        )
        self._executor.start()
        self._ready.wait()
        if self._init_error is not None:
            raise RuntimeError(
                f"MMScheduler failed to bind device {self._device!r}"
            ) from self._init_error

    @property
    def max_request_images(self) -> int:
        """Per-forward media cap. Without a model work budget a request never
        splits, so one exceeding this can never fit; callers can pre-check before
        preprocessing. With a budget it caps each chunk instead."""
        return self._max_batch_images

    def _queue_depth(self) -> int:
        """Return queued chunks, including a budget-overflow pending chunk.

        ``Queue.qsize()`` is intentionally used as a point-in-time gauge; it is
        approximate under concurrent producers, which is appropriate for
        monitoring and avoids adding a lock to the submission hot path.
        """
        return self._waiting.qsize() + int(self._pending is not None)

    def _report_queue_depth(self) -> None:
        kmonitor.report(
            GaugeMetrics.VIT_EMBEDDING_QUEUE_SIZE_METRIC, self._queue_depth()
        )

    def _enqueue_chunk(self, chunk: _EmbeddingChunk, *, block: bool = True) -> None:
        """Put a chunk into the waiting queue and stamp its queue-entry time."""
        chunk.enqueued_at = time.monotonic()
        chunk.queue_wait_reported = False
        self._waiting.put(chunk, block=block)
        self._report_queue_depth()

    def _report_queue_wait(self, batch: List[_EmbeddingChunk]) -> List[float]:
        """Report each active chunk's wait before its first forward attempt."""
        now = time.monotonic()
        wait_times = []
        for chunk in batch:
            if chunk.queue_wait_reported:
                continue
            chunk.queue_wait_reported = True
            if chunk.enqueued_at is None:
                continue
            wait_ms = max(0.0, (now - chunk.enqueued_at) * 1000.0)
            kmonitor.report(GaugeMetrics.VIT_EMBEDDING_QUEUE_WAIT_RT_METRIC, wait_ms)
            wait_times.append(wait_ms)
        return wait_times

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
        for field_name in (
            "input_patches",
            "output_tokens",
            "estimated_workspace_bytes",
            "attention_work",
        ):
            limit = getattr(budget, field_name)
            if (
                limit > 0
                and getattr(current, field_name) + getattr(candidate, field_name)
                > limit
            ):
                return True
        # max_attention_segment bounds the single largest segment, so it is a
        # max rather than a sum.
        return (
            budget.max_attention_segment > 0
            and max(current.max_attention_segment, candidate.max_attention_segment)
            > budget.max_attention_segment
        )

    def _require_work_estimate(self, work_item: MMWorkItem) -> MMWorkEstimate:
        estimate = getattr(work_item, "work_estimate", None)
        if estimate is None:
            raise MMSchedulerError(
                "model enabled cost-aware multimodal scheduling, but a "
                "preprocessed work item has no work estimate"
            )
        if not isinstance(estimate, MMWorkEstimate):
            raise MMSchedulerError(
                "cost-aware multimodal work estimate must be MMWorkEstimate, "
                f"got {type(estimate).__name__}"
            )
        return estimate

    def _build_chunks(self, request: _EmbeddingRequest) -> None:
        """Split a request into chunks that each fit one forward.

        Without a model work budget (or for a single work item, which the model
        preprocess batch makes indivisible) the request stays whole and the
        media-count cap still rejects it up front, preserving legacy behavior.
        """
        if not request.work_items:
            raise ValueError("MMScheduler requires at least one work item")

        splittable = self._work_budget is not None and len(request.work_items) > 1
        if not splittable:
            # max_batch_images is also the SINGLE-request cap here: an unsplit
            # request that exceeds it can never fit. Serial mode passes
            # sys.maxsize (no single-request limit).
            if request.n_images > self._max_batch_images:
                raise MMSchedulerRequestTooLargeError(
                    f"request image count {request.n_images} exceeds "
                    f"gpu_max_batch_images {self._max_batch_images}, "
                    f"request rejected"
                )
            request.chunks = [
                _EmbeddingChunk(
                    request=request,
                    work_items=request.work_items,
                    n_images=request.n_images,
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
                raise MMSchedulerRequestTooLargeError(
                    f"single work item image count {item_images} exceeds "
                    f"gpu_max_batch_images {self._max_batch_images}; "
                    "the model preprocess batch is not splittable"
                )
            item_work = self._require_work_estimate(work_item)
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
                # A single model work item is not generically splittable (one
                # long video, say). Run it alone rather than reintroduce the old
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

        # The scheduler owns only the embedding-stage timeout. Preprocessing keeps
        # its existing timeout semantics and does not consume this budget. Use the
        # largest positive timeout, matching MMWorkItem and the remote VIT proxy.
        timeout_values_ms = [
            wi.mm_timeout_ms
            for wi in work_items
            if wi.mm_timeout_ms is not None and wi.mm_timeout_ms > 0
        ]
        timeout_ms = (
            max(timeout_values_ms) if timeout_values_ms else _DEFAULT_MM_TIMEOUT_MS
        )
        timeout_s = timeout_ms / 1000.0

        submit_ms = current_time_ms()

        # Lock only the stopped-check + enqueue so it is atomic w.r.t. close();
        # the blocking wait below stays outside the lock.
        with self._lock:
            if self._stopped.is_set():
                raise RuntimeError("MMScheduler is closed, request rejected")
            # Non-blocking: if the queue is full (e.g. a stalled forward backing up
            # requests) fail fast with an overload signal instead of blocking the
            # caller and letting the backlog grow unbounded.
            try:
                # Only the first chunk is queued; _complete_chunk appends the
                # next one at the tail so a split request cannot monopolize the
                # scheduler.
                self._enqueue_chunk(req.chunks[0], block=False)
            except queue.Full:
                kmonitor.report(AccMetrics.VIT_EMBEDDING_OVERLOAD_QPS_METRIC, 1)
                raise MMSchedulerOverloadError(
                    f"MMScheduler queue full (max_queue_size={self._waiting.maxsize}), "
                    f"request rejected"
                ) from None

        try:
            # Blocks until the executor resolves the future; re-raises the
            # per-request lightweight wrapper on failure.
            req.future.result(timeout=timeout_s)
        except MMSchedulerExecutionError:
            # Future.result() adds this frame to the stored exception traceback.
            # Detach request inputs so retaining the public exception cannot retain
            # the failed request's preprocessed tensors through this frame/Future.
            req.work_items = []
            work_items = []
            raise
        except FutureTimeoutError:
            # PENDING -> CANCELLED so the executor skips it; if it is already
            # RUNNING, cancel() is a no-op and the forward's result is discarded.
            req.future.cancel()
            req.abandoned = True
            waited_ms = current_time_ms() - submit_ms
            logging.warning(
                "MMScheduler: embedding wait timeout after %.0fms "
                "(queue_depth=%d, batch_wait_ms=%d)",
                waited_ms,
                self._queue_depth(),
                self._batch_wait_ms,
            )
            # from None: the internal wait timeout isn't a cause worth surfacing.
            raise MMSchedulerTimeoutError(
                f"MMScheduler: embedding wait timeout after {timeout_s * 1000:.0f}ms"
            ) from None

        # Per-request latency = wait + forward (queue wait + batch-collect wait +
        # forward), preserving VIT_EMBEDDING_RT_METRIC's historical per-request
        # meaning and sampling. Reported only on success; failures are tracked by
        # VIT_ERROR_QPS_METRIC.
        kmonitor.report(
            GaugeMetrics.VIT_EMBEDDING_RT_METRIC, current_time_ms() - submit_ms
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
    def _fail(
        req: _EmbeddingRequest,
        source_type: str,
        source_message: str,
        is_oom: bool = False,
    ) -> None:
        """Fail one request without storing the worker exception in its Future."""
        req.future.set_exception(
            MMSchedulerExecutionError(source_type, source_message, is_oom)
        )

    @staticmethod
    def _failure_details(cause: BaseException) -> tuple[str, str, bool]:
        return (
            type(cause).__name__,
            str(cause),
            isinstance(cause, torch.cuda.OutOfMemoryError),
        )

    @staticmethod
    def _clear_exception_references(cause: BaseException) -> None:
        """Drop traceback chains after they have been logged.

        Model-forward frames can retain device tensors through their locals. They
        must not survive in a Future or remain live when empty_cache() runs.
        """
        tb = cause.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)
        cause.__traceback__ = None
        cause.__context__ = None
        cause.__cause__ = None

    def _reject_batch(self, batch: List[_EmbeddingChunk]) -> None:
        """Fail every not-yet-started request in `batch` because close() fired.

        These requests were already pulled out of _waiting into the executor's
        local batch, so close()'s post-join drain cannot see them — the executor
        must resolve them itself or their callers block until their submit
        timeout. Uses the same cause as close()'s drain for a consistent error.
        """
        source_type = "RuntimeError"
        source_message = "MMScheduler closed before request completed"
        for req in self._requests_of(batch):
            # done() skips already-resolved/cancelled requests; the try guards the
            # TOCTOU where a caller cancels a still-PENDING request concurrently.
            if not req.future.done():
                try:
                    self._fail(req, source_type, source_message)
                except InvalidStateError:
                    pass

    def _executor_loop(self) -> None:
        # Pin this thread to the engine's device once, up front: a new thread
        # inherits cuda:0, so a non-zero local rank would otherwise forward on the
        # wrong device. Only an explicit cuda:N is bound; a bare "cuda" (standalone
        # VIT) keeps the current device (binding it would raise). Guarded for CPU /
        # no-device setups. On failure, record the error, mark stopped (reject any
        # racing submit), signal ready to unblock __init__, and exit — __init__ then
        # raises. No requests are queued yet, so there is nothing to fail here.
        try:
            if torch.cuda.is_available():
                cuda_index = _cuda_device_index(self._device)
                if cuda_index is not None:
                    torch.cuda.set_device(cuda_index)
        except Exception as e:  # noqa: BLE001 - surfaced via __init__
            self._init_error = e
            self._stopped.set()
            self._ready.set()
            return
        self._ready.set()

        try:
            while not self._stopped.is_set():
                batch = None
                try:
                    batch = self._collect_batch()
                    if batch is None:
                        break
                    # Stop-check and Future claim are one atomic transition against
                    # close(). Once claimed, this batch is in flight and close()
                    # waits for it through the executor join. The forward itself
                    # still runs outside the lock.
                    claimed_batch = self._claim_batch(batch)
                    if claimed_batch is None:
                        # close() fired first: reject the not-yet-started batch so
                        # callers unblock immediately instead of waiting their timeout.
                        self._reject_batch(batch)
                        break
                    batch = claimed_batch
                    self._execute_batch(batch)
                except Exception as e:
                    source_type, source_message, is_oom = self._failure_details(e)
                    logging.error(
                        f"MMScheduler: executor loop error: {e}", exc_info=True
                    )
                    self._clear_exception_references(e)
                    e = None
                    if is_oom:
                        gc.collect()
                        torch.cuda.empty_cache()
                    # Something unexpected escaped _execute_batch/_collect_batch
                    # before the future was resolved. Fail any unresolved request so
                    # its caller gets the error now; the loop keeps running.
                    if batch:
                        for req in self._requests_of(batch):
                            # done() skips resolved/cancelled; the try guards the
                            # TOCTOU where a caller cancels a still-PENDING request.
                            if not req.future.done():
                                try:
                                    self._fail(
                                        req, source_type, source_message, is_oom
                                    )
                                except InvalidStateError:
                                    pass
        finally:
            # The loop only exits once _stopped is set (submit then rejects new
            # work under _lock), so on ANY exit path the executor is the sole
            # remaining accessor of _pending/_waiting. Fail every not-yet-started
            # request here, so queued callers unblock even when close()'s join
            # timed out on a long forward and close() already returned.
            self._fail_all_queued()

    def _fail_all_queued(self) -> None:
        """Fail every request still queued (not yet started) with the close cause.

        Runs on the executor thread once the loop has stopped, so _pending /
        _waiting have no concurrent producer or consumer and need no lock.
        """
        source_type = "RuntimeError"
        source_message = "MMScheduler closed before request completed"
        queued = [self._pending] if self._pending else []
        self._pending = None
        queued.extend(self._drain(self._waiting))
        self._report_queue_depth()
        for req in self._requests_of(queued):
            # Guard set_exception: a caller may cancel concurrently (its submit
            # timing out); a dropped delivery is then fine.
            try:
                self._fail(req, source_type, source_message)
            except InvalidStateError:
                pass

    @staticmethod
    def _requests_of(chunks: List[_EmbeddingChunk]) -> List[_EmbeddingRequest]:
        """Distinct requests behind `chunks`, preserving first-seen order.

        A batch may hold several chunks of one request only across rounds, but
        deduplicating keeps every request-level resolution exactly-once.
        """
        seen: set[int] = set()
        requests: List[_EmbeddingRequest] = []
        for chunk in chunks:
            key = id(chunk.request)
            if key not in seen:
                seen.add(key)
                requests.append(chunk.request)
        return requests

    def _collect_batch(self) -> Optional[List[_EmbeddingChunk]]:
        # Pick the first chunk, skipping ones whose caller already timed out
        # (read-only pre-check; the authoritative claim is
        # set_running_or_notify_cancel() in _claim_batch, so a chunk carried in
        # _pending is not prematurely marked RUNNING). _pending goes first.
        #
        # The idle wait polls _stopped instead of blocking forever, so close()
        # needs no wake-up sentinel. Once stopped, start no new batch (return
        # None); only the in-flight batch finishes, the rest of the queue drops.
        while True:
            if self._stopped.is_set():
                return None
            if self._pending is not None:
                first = self._pending
                self._pending = None
                self._report_queue_depth()
            else:
                try:
                    first = self._waiting.get(timeout=_STOP_POLL_INTERVAL_S)
                except queue.Empty:
                    continue
                self._report_queue_depth()
            if not (first.request.future.cancelled() or first.request.abandoned):
                break
        batch = [first]
        n_images = first.n_images
        batch_work = first.work_estimate

        deadline = time.monotonic() + self._batch_wait_ms / 1000.0

        while len(batch) < self._max_batch_size and n_images < self._max_batch_images:
            # close() during the wait window: stop collecting immediately rather
            # than waiting out the deadline, so waiters are released promptly. The
            # loop's pre-execute stop-check then rejects the not-yet-run batch.
            if self._stopped.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Cap the block at the stop-poll interval so close() is noticed within
            # _STOP_POLL_INTERVAL_S instead of waiting out the whole window. A real
            # arrival still wakes get() immediately; on the poll timeout we loop to
            # re-check _stopped and the deadline rather than ending the window.
            try:
                chunk = self._waiting.get(
                    timeout=min(remaining, _STOP_POLL_INTERVAL_S)
                )
            except queue.Empty:
                continue
            self._report_queue_depth()
            if chunk.request.future.cancelled() or chunk.request.abandoned:
                continue  # caller already timed out; don't spend budget on it

            # The while guard caps the count; here stop on image or work overflow.
            if n_images + chunk.n_images > self._max_batch_images or (
                self._would_exceed_work_budget(batch_work, chunk.work_estimate)
            ):
                self._pending = chunk
                self._report_queue_depth()
                break
            batch.append(chunk)
            n_images += chunk.n_images
            if batch_work is not None and chunk.work_estimate is not None:
                batch_work = batch_work + chunk.work_estimate
            else:
                batch_work = None

        return batch

    def _claim_batch(
        self, batch: List[_EmbeddingChunk]
    ) -> Optional[List[_EmbeddingChunk]]:
        """Atomically order close() against claiming a collected batch.

        None means close() won the scheduler lock and no chunk was claimed.
        Otherwise every returned chunk's request Future is RUNNING, so the batch
        is in flight and must be resolved even if close() sets _stopped
        immediately after this method releases the lock. Chunks of cancelled
        requests are omitted.

        set_running_or_notify_cancel() may be called at most once per Future, so
        a request transitions to RUNNING on its first claimed chunk; later chunks
        only re-check cancellation.
        """
        with self._lock:
            if self._stopped.is_set():
                return None
            claimed: List[_EmbeddingChunk] = []
            for chunk in batch:
                request = chunk.request
                if request.abandoned:
                    continue
                if request.running_claimed:
                    if not request.future.cancelled():
                        claimed.append(chunk)
                elif request.future.set_running_or_notify_cancel():
                    request.running_claimed = True
                    claimed.append(chunk)
            return claimed

    def _complete_chunk(self, chunk: _EmbeddingChunk) -> None:
        """Resolve one finished chunk, queueing its request's next chunk if any."""
        request = chunk.request
        request.remaining_chunks -= 1
        if request.abandoned and request.remaining_chunks > 0:
            # Caller已超时返回：不再为它排后续 chunk。Future 仍是 RUNNING，
            # 保持未决即可，调用方已经拿到 timeout 错误。
            return
        if request.remaining_chunks == 0:
            request.future.set_result(None)
            return

        # Append at the tail so a split request interleaves with other callers
        # instead of holding the executor for all of its chunks.
        with self._lock:
            if self._stopped.is_set():
                self._fail(
                    request,
                    "RuntimeError",
                    "MMScheduler closed before request completed",
                )
                return
            next_chunk = request.chunks[request.next_chunk_index]
            request.next_chunk_index += 1
            self._enqueue_chunk(next_chunk)

    def _execute_batch(self, batch: List[_EmbeddingChunk]) -> None:
        """Run the batched forward and write results back.

        All-or-nothing: if the forward raises, the whole batch is discarded and
        every request behind it fails with its own wrapper (see _fail) — no
        per-request retry. Other batches are unaffected. A request that still
        has chunks left is resolved only after its last one (see
        _complete_chunk); a failure resolves it immediately and its remaining
        chunks are never queued.
        """
        # _claim_batch already performed the authoritative cancellation checkpoint;
        # every request here is RUNNING and MUST be resolved below.
        if not batch:
            return

        # Actual composition of this forward (after cancellations). The batch-size
        # metric is reported every forward for continuous monitoring via kmonitor
        # (VIT_EMBEDDING_BATCH_SIZE_METRIC); no per-merge logging on the hot path.
        batch_size = len(batch)
        kmonitor.report(GaugeMetrics.VIT_EMBEDDING_BATCH_SIZE_METRIC, batch_size)
        self._report_queue_wait(batch)

        items = [wi for chunk in batch for wi in chunk.work_items]
        # Profile the forward on this executor thread (per forward/batch) when a
        # hook is set; nullcontext otherwise. The hook is a no-op unless profiling
        # is armed, so this stays cheap on the hot path.
        profiler_cm = (
            self._forward_profiler()
            if self._forward_profiler is not None
            else nullcontext()
        )
        try:
            # Annotate the trace with the batch size so a merged forward is
            # attributable (the profiler unit is per-forward/batch, not per-request).
            with profiler_cm:
                with torch.profiler.record_function(
                    f"mm_forward(batch_size={batch_size})"
                ):
                    _run_embedding(self._mm_part, items)
        except Exception as e:
            source_type, source_message, is_oom = self._failure_details(e)
            logging.error(
                f"MMScheduler: batch forward failed, discarding {len(batch)} "
                f"request(s): {type(e).__name__}: {e}",
                exc_info=True,
            )
            # Keep the complete traceback only for the log above. Clear model
            # frames and forward locals before allocator recovery; otherwise the
            # traceback can keep failed-batch device tensors alive and make
            # empty_cache() ineffective.
            self._clear_exception_references(e)
            e = None
            items.clear()
            profiler_cm = None
            if is_oom:
                gc.collect()
                torch.cuda.empty_cache()
            for req in self._requests_of(batch):
                # A request may already be resolved when two of its chunks share
                # this batch; keep failure delivery exactly-once.
                if not req.future.done():
                    try:
                        self._fail(req, source_type, source_message, is_oom)
                    except InvalidStateError:
                        pass
            batch.clear()
            return

        for chunk in batch:
            self._complete_chunk(chunk)

    def close(self, timeout: float = 10.0) -> bool:
        """Stop the scheduler. Returns True if it stopped cleanly (executor exited),
        False if an in-flight forward is still running past the join timeout.

        Closing cannot cancel a forward that is already running. The daemon thread
        exits and releases its references after that forward returns.
        """
        # Set stopped atomically w.r.t. submit_and_wait and _claim_batch. If close
        # wins this lock, the collected batch is rejected; if _claim_batch wins,
        # its Futures are already RUNNING and the executor join below treats that
        # batch as in flight. The executor drains queued requests in its finally,
        # so close must not touch _pending / _waiting while it is alive.
        with self._lock:
            self._stopped.set()
        self._executor.join(timeout=timeout)
        if self._executor.is_alive():
            # A forward is still stuck past the join timeout. The executor's finally
            # will drain queued requests once that forward returns, but the forward
            # itself is still running, so report that the join did not complete.
            logging.warning("MMScheduler: executor join exceeded %.0fs", timeout)
            return False
        return True
