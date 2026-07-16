from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import torch.profiler

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import vit_emb_cache_
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

    A distinct type so callers can tell a count mismatch apart from a forward
    (device) error.
    """


class MMSchedulerExecutionError(MMSchedulerError):
    """A claimed batch failed while executing on the scheduler worker."""


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

    Does the batched forward, guards the output count, writes embedding_result
    onto each work item, and caches cacheable results. Errors propagate.
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
        if wi.need_check_cache:
            vit_emb_cache_.insert_cache(wi.cache_key, result)


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
                  gets its OWN wrapper (see _fail) so concurrent callers never
                  share one exception instance. Cancellation uses future.cancel()
                  + set_running_or_notify_cancel() (see _claim_batch), whose
                  shared lock resolves the cancel-vs-start race.
    """

    __slots__ = ("work_items", "n_images", "future")

    def __init__(self, work_items: List[MMWorkItem]):
        self.work_items = work_items
        self.n_images = sum(len(wi.mm_inputs) for wi in work_items)
        self.future: Future[None] = Future()


# Fallback for hand-built work items without a positive request timeout.
_DEFAULT_MM_TIMEOUT_MS = 120000

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
        # One-shot guard: log only the FIRST cross-request merge to keep the hot
        # path quiet. Only ever written from the single executor thread.
        self._logged_merge = False
        # Optional per-forward profiling hook: a factory returning a context
        # manager, entered on THIS executor thread around the forward so the GPU
        # compute is actually captured (torch.profiler doesn't span threads). None
        # -> no profiling. Attribution is per forward/batch.
        self._forward_profiler = forward_profiler

        # Bounded so a stalled forward can't let cancelled/waiting requests (and
        # the preprocessed tensors they pin) grow without limit; over capacity,
        # submit fails fast with MMSchedulerOverloadError.
        self._waiting: queue.Queue[_EmbeddingRequest] = queue.Queue(
            maxsize=max_queue_size
        )
        # A request that would have overflowed the batch's image budget, carried
        # to the next round so it is neither lost nor re-ordered behind newer
        # arrivals.
        self._pending: Optional[_EmbeddingRequest] = None
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
        """Per-request media cap (a request never splits across batches, so one
        exceeding this can never fit). Callers can pre-check before preprocessing."""
        return self._max_batch_images

    def submit_and_wait(self, work_items: List[MMWorkItem]) -> None:
        req = _EmbeddingRequest(work_items)
        # max_batch_images is also the SINGLE-request cap: a request is never
        # split across batches, so one exceeding it can never fit — reject up
        # front. Serial mode passes sys.maxsize (no single-request limit).
        if req.n_images > self._max_batch_images:
            raise MMSchedulerRequestTooLargeError(
                f"request image count {req.n_images} exceeds "
                f"gpu_max_batch_images {self._max_batch_images}, "
                f"request rejected"
            )

        # The scheduler owns only the embedding-stage timeout. Preprocessing keeps
        # its existing timeout semantics and does not consume this budget. Use the
        # shortest positive timeout when a request contains multiple work items.
        timeout_values_ms = [
            wi.mm_timeout_ms
            for wi in work_items
            if wi.mm_timeout_ms is not None and wi.mm_timeout_ms > 0
        ]
        timeout_ms = (
            min(timeout_values_ms) if timeout_values_ms else _DEFAULT_MM_TIMEOUT_MS
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
                self._waiting.put_nowait(req)
            except queue.Full:
                kmonitor.report(AccMetrics.VIT_EMBEDDING_OVERLOAD_QPS_METRIC, 1)
                raise MMSchedulerOverloadError(
                    f"MMScheduler queue full (max_queue_size={self._waiting.maxsize}), "
                    f"request rejected"
                ) from None

        try:
            # Blocks until the executor resolves the future; re-raises the
            # per-request wrapper (with its cause chain) on failure.
            req.future.result(timeout=timeout_s)
        except FutureTimeoutError:
            # PENDING -> CANCELLED so the executor skips it; if it is already
            # RUNNING, cancel() is a no-op and the forward's result is discarded.
            req.future.cancel()
            waited_ms = current_time_ms() - submit_ms
            logging.warning(
                "MMScheduler: embedding wait timeout after %.0fms "
                "(queue_depth=%d, batch_wait_ms=%d)",
                waited_ms,
                self._waiting.qsize(),
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

    @staticmethod
    def _fail(req: _EmbeddingRequest, cause: BaseException) -> None:
        """Fail req's future with its OWN wrapper around `cause`.

        future.result() re-raises the stored instance in the caller's thread, so
        a shared instance would let N callers concurrently mutate one exception's
        traceback. A fresh wrapper per request avoids that; `cause` is chained
        (read-only) for debugging.
        """
        wrapped = MMSchedulerExecutionError(f"batch embedding failed: {cause}")
        # Mimic `raise ... from cause` for the later re-raise in result().
        wrapped.__cause__ = cause
        wrapped.__suppress_context__ = True
        req.future.set_exception(wrapped)

    def _reject_batch(self, batch: List[_EmbeddingRequest]) -> None:
        """Fail every not-yet-started request in `batch` because close() fired.

        These requests were already pulled out of _waiting into the executor's
        local batch, so close()'s post-join drain cannot see them — the executor
        must resolve them itself or their callers block until their submit
        timeout. Uses the same cause as close()'s drain for a consistent error.
        """
        exc = RuntimeError("MMScheduler closed before request completed")
        for req in batch:
            # done() skips already-resolved/cancelled requests; the try guards the
            # TOCTOU where a caller cancels a still-PENDING request concurrently.
            if not req.future.done():
                try:
                    self._fail(req, exc)
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
                    logging.error(
                        f"MMScheduler: executor loop error: {e}", exc_info=True
                    )
                    # Something unexpected escaped _execute_batch/_collect_batch
                    # before the future was resolved. Fail any unresolved request so
                    # its caller gets the error now; the loop keeps running.
                    if batch:
                        for req in batch:
                            # done() skips resolved/cancelled; the try guards the
                            # TOCTOU where a caller cancels a still-PENDING request.
                            if not req.future.done():
                                try:
                                    self._fail(req, e)
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
        exc = RuntimeError("MMScheduler closed before request completed")
        queued = [self._pending] if self._pending else []
        self._pending = None
        queued.extend(self._drain(self._waiting))
        for req in queued:
            # Guard set_exception: a caller may cancel concurrently (its submit
            # timing out); a dropped delivery is then fine.
            try:
                self._fail(req, exc)
            except InvalidStateError:
                pass

    def _collect_batch(self) -> Optional[List[_EmbeddingRequest]]:
        # Pick the first request, skipping cancelled ones (read-only pre-check;
        # the authoritative claim is set_running_or_notify_cancel() in
        # _claim_batch, so a request carried in _pending is not prematurely
        # marked RUNNING). _pending goes first.
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
            else:
                try:
                    first = self._waiting.get(timeout=_STOP_POLL_INTERVAL_S)
                except queue.Empty:
                    continue
            if not first.future.cancelled():
                break
        batch = [first]
        n_images = first.n_images

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
                req = self._waiting.get(timeout=min(remaining, _STOP_POLL_INTERVAL_S))
            except queue.Empty:
                continue
            if req.future.cancelled():
                continue  # caller already timed out; don't spend budget on it

            # The while guard caps the count; here only stop on image overflow.
            if n_images + req.n_images > self._max_batch_images:
                self._pending = req
                break
            batch.append(req)
            n_images += req.n_images

        return batch

    def _claim_batch(
        self, batch: List[_EmbeddingRequest]
    ) -> Optional[List[_EmbeddingRequest]]:
        """Atomically order close() against claiming a collected batch.

        None means close() won the scheduler lock and no request was claimed.
        Otherwise every returned Future is RUNNING, so the batch is in flight and
        must be resolved even if close() sets _stopped immediately after this
        method releases the lock. Cancelled Futures are omitted.
        """
        with self._lock:
            if self._stopped.is_set():
                return None
            return [
                req
                for req in batch
                if req.future.set_running_or_notify_cancel()
            ]

    def _execute_batch(self, batch: List[_EmbeddingRequest]) -> None:
        """Run the batched forward and write results back.

        All-or-nothing: if the forward raises, the whole batch is discarded and
        every request fails with its own wrapper (see _fail) — no per-request
        retry. Other batches are unaffected.
        """
        # _claim_batch already performed the authoritative cancellation checkpoint;
        # every request here is RUNNING and MUST be resolved below.
        if not batch:
            return

        # Actual composition of this forward (after cancellations). The batch-size
        # metric is reported every forward for continuous monitoring. The log is
        # a bounded one-shot: only the FIRST cross-request merge is logged, so the
        # hot path stays quiet while smoke still has a greppable signal that
        # batching happened at least once.
        batch_size = len(batch)
        kmonitor.report(GaugeMetrics.VIT_EMBEDDING_BATCH_SIZE_METRIC, batch_size)
        if batch_size > 1 and not self._logged_merge:
            self._logged_merge = True
            logging.info(
                "MMScheduler: forward batch=%d requests, images=%d "
                "(first cross-request merge)",
                batch_size,
                sum(req.n_images for req in batch),
            )

        items = [wi for req in batch for wi in req.work_items]
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
            # On OOM, reset the allocator so the next batch starts clean. The
            # typed exception is chained as each request's wrapper cause.
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            logging.error(
                f"MMScheduler: batch forward failed, discarding {len(batch)} "
                f"request(s): {type(e).__name__}: {e}",
                exc_info=True,
            )
            for req in batch:
                self._fail(req, e)
            return

        for req in batch:
            req.future.set_result(None)

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
