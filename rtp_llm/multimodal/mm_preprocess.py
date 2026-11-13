"""CPU preprocessing workers, IPC result collection and timing."""

from __future__ import annotations

import json
import logging
import multiprocessing.pool
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Set, Tuple

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.mm_preprocess_result import MMPreprocessResult
from rtp_llm.multimodal.mm_worker_inspection import (
    WorkerInspectionChannel,
    WorkerInspectionRunner,
)
from rtp_llm.multimodal.multimodal_util import (
    collect_download_timing,
    url_data_cache_,
    vit_emb_cache_,
)
from rtp_llm.multimodal.preprocess_ipc import register_preprocess_storage_reducer
from rtp_llm.multimodal.vit_metrics import (
    VitMetricSample,
    collect_vit_preprocess_metrics,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.telemetry import vit_media
from rtp_llm.utils.time_util import Timer

if TYPE_CHECKING:
    from rtp_llm.multimodal.mm_work_item import MMWorkItem


_worker_vit_config: Optional[VitConfig] = None
_worker_preprocess_params: Optional[dict] = None
_worker_preprocess_func: Optional[Callable] = None
_worker_inspection = None


def _set_download_region(region: Optional[str]):
    """Expose the passthrough workspace region to the downloader for one task.

    Returns a context token, or None when the internal download proxy is
    unavailable (open-source build) or the request carries no region.
    """
    if not region:
        return None
    try:
        from internal_source.rtp_llm.utils.download_proxy import current_download_region
    except ImportError:
        return None
    return current_download_region.set(region)


def _reset_download_region(token) -> None:
    if token is None:
        return
    from internal_source.rtp_llm.utils.download_proxy import current_download_region

    current_download_region.reset(token)


class _PreprocessTaskError(Exception):
    """Keep task I/O failures distinct from broken pool communication."""

    def __init__(
        self,
        error: Exception,
        samples: Optional[List[VitMetricSample]] = None,
        trace_timings=None,
    ):
        super().__init__(error)
        self.error = error
        self.samples = samples or []
        self.trace_timings = trace_timings


@dataclass
class PreprocessTaskResult:
    result: Any
    total_ms: float
    download_ms: float
    samples: List[VitMetricSample]
    started_at: float
    finished_at: float
    trace_timings: Any = None

    def __iter__(self):
        # Preserve the existing internal result-unpacking helper contract.
        return iter((self.result, self.total_ms, self.download_ms, self.samples))


def _run_preprocess_task(
    preprocess_func: Callable,
    mm_inputs: List[MultimodalInput],
    vit_config: VitConfig,
    preprocess_params: dict,
    region: Optional[str] = None,
    trace_spec=None,
) -> PreprocessTaskResult:
    """Run preprocessing and return total and media-load durations in ms."""
    started_at = time.monotonic()
    with vit_media.collect_worker(
        trace_spec, preprocess=True
    ) as timings, collect_vit_preprocess_metrics() as preprocess_metrics:
        with collect_download_timing() as download_timing:
            with Timer() as route_timer:
                # Expose the passthrough workspace region to the downloader
                # for this task only; concurrent tasks keep their own value.
                region_token = _set_download_region(region)
                try:
                    result = preprocess_func(mm_inputs, vit_config, **preprocess_params)
                except Exception as error:
                    if preprocess_metrics.samples or timings is not None:
                        raise _PreprocessTaskError(
                            error, preprocess_metrics.samples, timings
                        ) from error
                    raise
                finally:
                    _reset_download_region(region_token)
    return PreprocessTaskResult(
        result,
        route_timer.cost_ms(),
        download_timing.elapsed_ms,
        preprocess_metrics.samples,
        started_at,
        time.monotonic(),
        timings,
    )


def _report_result_path(work_item, payload):
    """Same-host monotonic timestamps include dispatch/IPC, not pure memcpy.

    The receive stamp is recorded by the pool callback, so waiting behind other
    request work in the caller does not inflate the return-path measurement.
    """
    if os.environ.get("MM_EMB_BENCHMARK_TIMING") == "1":
        try:
            identity = getattr(work_item, "source_identity", None)
            logging.info(
                "MM_EMB_BENCHMARK_JSON %s",
                json.dumps(
                    dict(
                        event="preprocess_timing",
                        trace_id=getattr(work_item, "benchmark_trace_id", None),
                        pid=os.getpid(),
                        timestamp_ns=time.time_ns(),
                        wall_ms=payload.total_ms,
                        download_ms=payload.download_ms,
                        scope="cpu_preprocess_including_source_download",
                        synchronized=False,
                        media_sha256=identity.sha256 if identity is not None else None,
                        media_nbytes=identity.nbytes if identity is not None else None,
                    ),
                    sort_keys=True,
                ),
            )
        except Exception:
            pass  # Optional diagnostic output must not fail preprocessing.
    for metric, seconds in (
        (
            GaugeMetrics.VIT_PREPROCESS_DISPATCH_RT_METRIC,
            payload.started_at - work_item.preprocess_submitted_at,
        ),
        (
            GaugeMetrics.VIT_PREPROCESS_RETURN_RT_METRIC,
            work_item.preprocess_received_at - payload.finished_at,
        ),
    ):
        try:
            kmonitor.report(metric, max(0.0, seconds * 1000.0))
        except Exception:
            logging.exception("Failed to report CPU preprocessing transport timing")


def _worker_initializer(
    vit_config: VitConfig,
    preprocess_params: dict,
    preprocess_func: Callable,
    inspection_args=None,
) -> None:
    """
    每个工作进程启动时调用的初始化函数。
    接收一次不变的参数，并将其存储在进程的全局变量中。
    """
    global _worker_vit_config, _worker_preprocess_params, _worker_preprocess_func
    global _worker_inspection
    # 让工作进程忽略 SIGINT 信号，这样主进程的 Ctrl+C 不会杀死它们
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    register_preprocess_storage_reducer()
    _worker_vit_config = vit_config
    _worker_preprocess_params = preprocess_params
    _worker_preprocess_func = preprocess_func
    _worker_inspection = (
        WorkerInspectionRunner(*inspection_args)
        if inspection_args is not None
        else None
    )
    # Spawned workers import fresh module-level caches; parent-only resizing
    # does not propagate. Apply the operator's capacities once per worker.
    url_data_cache_.resize_cache(vit_config.url_cache_item_num)
    vit_emb_cache_.resize_cache(vit_config.mm_cache_item_num)
    logging.info(f"Worker process {os.getpid()} initialized.")


def _worker_process_task(
    mm_inputs: List[MultimodalInput],
    capture_inspection_sources: bool = False,
    deadline: Optional[float] = None,
    inspection_task=None,
    region: Optional[str] = None,
    trace_spec=None,
) -> PreprocessTaskResult:
    """
    只接收变化的 `mm_inputs` 参数。
    """
    if _worker_preprocess_func is None:
        raise RuntimeError("Worker process has not been initialized correctly.")
    params = dict(_worker_preprocess_params)
    if capture_inspection_sources:
        params.update(capture_inspection_sources=True, deadline=deadline)
    try:
        if inspection_task is not None:
            if _worker_inspection is None:
                raise RuntimeError("CPU worker inspection is unavailable")
            _worker_inspection.started(inspection_task)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("ViT request expired before CPU preprocessing")
        try:
            payload = _run_preprocess_task(
                _worker_preprocess_func,
                mm_inputs,
                _worker_vit_config,
                params,
                region=region,
                trace_spec=trace_spec,
            )
        except (OSError, EOFError) as error:
            # For example, PIL's UnidentifiedImageError is an OSError. Letting it
            # escape as a pool error would terminate healthy workers and discard
            # unrelated requests that are already queued in the same pool.
            raise _PreprocessTaskError(error) from error
        if inspection_task is not None:
            result = payload.result
            if not isinstance(result, MMPreprocessResult) or len(
                result.inspection_sources
            ) != len(mm_inputs):
                raise ValueError(
                    "CPU preprocessing did not return matching inspection sources"
                )
            sources = result.inspection_sources
            # Transfer ownership inside this process. The Pool carries only RGB
            # and the ordinary model metadata; no original JPEG/base64 bytes.
            result.inspection_sources = []
            _worker_inspection.submit(inspection_task, mm_inputs, sources)
        return payload
    except BaseException as error:
        if inspection_task is not None and _worker_inspection is not None:
            _worker_inspection.failed(
                inspection_task,
                error.error if isinstance(error, _PreprocessTaskError) else error,
            )
        raise


def _unpack_preprocess_result(payload: Any) -> Tuple[Any, float, float]:
    """Read the four-field result returned by both preprocess executors."""
    result, total_ms, download_ms, samples = payload
    total_ms, download_ms = float(total_ms), float(download_ms)
    _report_vit_preprocess_samples(samples)
    # Clock/context overhead must never make the derived remainder negative.
    download_ms = max(0.0, min(download_ms, total_ms))
    return result, total_ms, download_ms


def _report_preprocess_timing(total_ms: float, download_ms: float) -> None:
    """Report the two non-overlapping preprocessing components in milliseconds."""
    for metric, value in (
        (GaugeMetrics.VIT_PREPROCESS_RT_METRIC, total_ms),
        (GaugeMetrics.VIT_DOWNLOAD_RT_METRIC, download_ms),
        (GaugeMetrics.VIT_PREPROCESS_OTHER_RT_METRIC, max(0.0, total_ms - download_ms)),
    ):
        try:
            kmonitor.report(metric, value)
        except Exception:
            logging.exception("Failed to report ViT preprocessing timing")


def _report_preprocess_queue_size(queue_size: int) -> None:
    """Report the number of preprocessing work items not yet completed."""
    try:
        kmonitor.report(
            GaugeMetrics.VIT_PREPROCESS_QUEUE_SIZE_METRIC, max(0, int(queue_size))
        )
    except Exception:
        # Telemetry must never change the preprocessing result.
        logging.exception("Failed to report ViT preprocess queue size")


def _report_vit_preprocess_samples(samples: List[VitMetricSample]) -> None:
    for sample in samples:
        try:
            kmonitor.report(sample.metric, sample.value, sample.tags)
        except Exception:
            # Parent-side telemetry must not discard a successful worker result.
            logging.exception("Failed to report ViT preprocessing sample")


class PreprocessExecutor:
    """预处理执行器抽象基类，封装预处理逻辑"""

    def submit(self, work_item: "MMWorkItem") -> None:
        raise NotImplementedError

    def get_result(self, work_item: "MMWorkItem") -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    def iter_ready(self, work_items):
        """Yield completed work without head-of-line blocking by input order."""
        pending = list(work_items)
        if not pending:
            return
        event = pending[0].preprocess_ready_event
        while pending:
            event.clear()
            ready = [
                item for item in pending if item.future is None or item.preprocess_ready
            ]
            for item in ready:
                pending.remove(item)
                yield item
            if pending and not ready:
                try:
                    if not event.wait(
                        timeout=min(item.remaining_timeout_s() for item in pending)
                    ):
                        self._on_wait_timeout(pending)
                        raise TimeoutError("CPU preprocessing result wait timed out")
                except Exception as error:
                    vit_media.failure(
                        "preprocess.worker", error, vit_media.item_media(pending)
                    )
                    raise

    def _on_wait_timeout(self, pending):
        pass

    @staticmethod
    def _set_result(work_item, result):
        if isinstance(result, MMPreprocessResult):
            if work_item.inspection_future is not None:
                if result.inspection_sources:
                    raise ValueError("Worker inspection leaked source bytes to parent")
            elif (
                work_item.capture_inspection_sources
                or result.inspection_sources
                or result.source_identity is None
            ):
                if len(result.inspection_sources) != len(work_item.mm_inputs):
                    raise ValueError("CPU inspection sources changed input count")
                work_item.inspection_sources = result.inspection_sources
            work_item.source_identity = result.source_identity
            work_item.preprocess_result = result.data
        elif work_item.capture_inspection_sources:
            raise ValueError("CPU preprocessing did not return inspection sources")
        else:
            work_item.preprocess_result = result


class LocalPreprocessExecutor(PreprocessExecutor):
    """本地预处理执行器（同步执行）"""

    def __init__(
        self,
        preprocess_func: Callable,
        vit_config: VitConfig,
        preprocess_params: dict,
    ):
        self.preprocess_func = preprocess_func
        self.vit_config = vit_config
        self.preprocess_params = preprocess_params
        _report_preprocess_queue_size(0)

    def submit(self, work_item: "MMWorkItem") -> None:
        if not work_item.should_preprocess:
            return

        try:
            params = dict(self.preprocess_params)
            if work_item.capture_inspection_sources:
                params.update(
                    capture_inspection_sources=True, deadline=work_item.deadline
                )
            work_item.preprocess_submitted_at = time.monotonic()
            payload = _run_preprocess_task(
                self.preprocess_func,
                work_item.mm_inputs,
                self.vit_config,
                params,
                region=work_item.region,
                trace_spec=vit_media.worker_spec(
                    getattr(work_item, "media_traces", ())
                ),
            )
            result, preprocess_time, download_time, samples = payload
            self._set_result(work_item, result)
            # 使用简单的对象模拟 future 行为
            work_item.future = _LocalResult(payload)
            work_item.preprocess_received_at = time.monotonic()
            work_item.preprocess_ready = True
            work_item.preprocess_ready_event.set()
        except _PreprocessTaskError as error:
            vit_media.consume(
                getattr(work_item, "media_traces", ()), error.trace_timings
            )
            _report_vit_preprocess_samples(error.samples)
            raise error.error from error
        except Exception as e:
            logging.error(f"Error in local preprocessing: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            payload = work_item.future.get()
            vit_media.consume(
                getattr(work_item, "media_traces", ()),
                getattr(payload, "trace_timings", None),
            )
            _, preprocess_time, download_time = _unpack_preprocess_result(payload)
            _report_result_path(work_item, payload)
            _report_preprocess_timing(preprocess_time, download_time)
            work_item.future = None
        except Exception as e:
            logging.error(f"Error getting local preprocess result: {e}", exc_info=True)
            raise


class _PreprocessPool(multiprocessing.pool.Pool):
    @staticmethod
    def _help_stuff_finish(inqueue, task_handler, size):
        """Do not wait on an orphaned read lock after the writer has exited.

        Python 3.10's Pool unconditionally acquires this lock on terminate().
        A worker dying in its audit thread while its main thread waits in get()
        leaves that lock held. With no writer left there is nothing to drain.
        Keep the upstream draining behavior while the task writer is alive.
        """
        while task_handler.is_alive():
            if inqueue._rlock.acquire(timeout=0.1):
                while task_handler.is_alive() and inqueue._reader.poll():
                    inqueue._reader.recv()
                    time.sleep(0)
                return


class MultiprocessPreprocessExecutor(PreprocessExecutor):
    """多进程预处理执行器

    Crash recovery: when a worker process dies or becomes unresponsive, the pool
    is automatically torn down and recreated via ``_rebuild_pool()``.  This is
    triggered in two paths:
      1. submit() — catches BrokenPipeError/OSError/EOFError, rebuilds, retries once.
      2. get_result() — catches the same errors or consecutive timeouts exceeding
         ``_max_consecutive_timeouts``, then rebuilds for subsequent requests.
    """

    def __init__(
        self,
        mp_context: multiprocessing.context.BaseContext,
        vit_config: VitConfig,
        preprocess_params: dict,
        preprocess_func: Callable,
        inspection_factory=None,
    ):
        self.mp_context = mp_context
        self.vit_config = vit_config
        self.preprocess_params = preprocess_params
        self.preprocess_func = preprocess_func
        self.pool: Optional[multiprocessing.pool.Pool] = None
        self._consecutive_timeouts = 0
        self._max_consecutive_timeouts = vit_config.mm_preprocess_max_workers
        # Counts completed tasks so a timeout can tell a stuck pool (no
        # progress) from an overloaded one (slow but draining).
        self._completed_tasks = 0
        self._last_timeout_completed = 0
        # Track accepted work items independently of multiprocessing.Pool's
        # private task queue. The gauge includes both running and waiting tasks.
        self._preprocess_queue_lock = threading.Lock()
        self._pending_preprocess_tasks: Set[int] = set()
        self._next_preprocess_task_id = 0
        # Serializes timeout-counter updates and pool rebuilds — without it
        # concurrent get_result/submit callers can race to _rebuild_pool, double
        # tear down the pool, or miscount consecutive timeouts.
        self._pool_lock = threading.Lock()
        self._pool_broken = threading.Event()
        limits = vit_config.async_compute_executor_args()
        self._inspection_channel = (
            WorkerInspectionChannel(
                mp_context,
                inspection_factory,
                max(
                    vit_config.mm_preprocess_max_workers,
                    limits["max_workers"] + limits["max_queue_size"],
                ),
                lambda: self._get_child_pids_from_pool(self.pool),
                self._pool_broken.set,
            )
            if inspection_factory is not None
            else None
        )
        _report_preprocess_queue_size(0)
        self._create_pool()

    def _create_pool(self) -> None:
        """创建进程池"""
        logging.info(
            f"Creating multiprocessing pool for preprocessing with {self.vit_config.mm_preprocess_max_workers} workers"
        )
        self.pool = _PreprocessPool(
            context=self.mp_context,
            processes=self.vit_config.mm_preprocess_max_workers,
            initializer=_worker_initializer,
            initargs=(
                self.vit_config,
                self.preprocess_params,
                self.preprocess_func,
                (
                    self._inspection_channel.worker_args
                    if self._inspection_channel is not None
                    else None
                ),
            ),
        )

    def _on_wait_timeout(self, pending):
        with self._pool_lock:
            self._record_timeout_locked()

    def _record_timeout_locked(self):
        # Distinguish a stuck pool (no task completed since the last timeout)
        # from an overloaded one (tasks are still draining). Rebuilding an
        # overloaded pool would discard queued work and make the backlog worse.
        completed = self._completed_tasks
        if completed != self._last_timeout_completed:
            self._consecutive_timeouts = 0
            self._last_timeout_completed = completed
            return
        self._consecutive_timeouts += 1
        if self._consecutive_timeouts >= self._max_consecutive_timeouts:
            logging.warning(
                f"Hit {self._consecutive_timeouts} consecutive timeouts "
                f"with no completed tasks, rebuilding pool "
                f"(workers may be stuck)"
            )
            self._rebuild_pool()
            self._consecutive_timeouts = 0
            self._last_timeout_completed = self._completed_tasks

    def _track_preprocess_task(self) -> int:
        with self._preprocess_queue_lock:
            self._next_preprocess_task_id += 1
            task_id = self._next_preprocess_task_id
            self._pending_preprocess_tasks.add(task_id)
            queue_size = len(self._pending_preprocess_tasks)
        _report_preprocess_queue_size(queue_size)
        return task_id

    def _finish_preprocess_task(self, task_id: int) -> None:
        with self._preprocess_queue_lock:
            if task_id not in self._pending_preprocess_tasks:
                return
            self._pending_preprocess_tasks.remove(task_id)
            self._completed_tasks = getattr(self, "_completed_tasks", 0) + 1
        # multiprocessing.Pool invokes callbacks on its single result-handler
        # thread, before waking ApplyResult.get(). Never run telemetry or I/O
        # here; the consuming request thread reports the latest count.

    def _clear_preprocess_tasks(self) -> None:
        with self._preprocess_queue_lock:
            if not self._pending_preprocess_tasks:
                return
            self._pending_preprocess_tasks.clear()
        _report_preprocess_queue_size(0)

    def _apply_async(self, work_item: "MMWorkItem", task_id: int) -> Any:
        inspection_task = None
        channel = getattr(self, "_inspection_channel", None)
        if channel is not None and work_item.inspection_request is not None:
            deadline = work_item.deadline or (
                time.monotonic() + work_item.remaining_timeout_s()
            )
            inspection_task, work_item.inspection_future = channel.register(
                work_item.inspection_request,
                deadline,
                media_traces=getattr(work_item, "media_traces", ()),
            )

        def completed(_result):
            if inspection_task is not None and isinstance(_result, BaseException):
                channel.finish(
                    inspection_task.task_id,
                    error=(
                        _result.error
                        if isinstance(_result, _PreprocessTaskError)
                        else _result
                    ),
                )
            work_item.preprocess_received_at = time.monotonic()
            self._finish_preprocess_task(task_id)
            work_item.preprocess_ready = True
            work_item.preprocess_ready_event.set()

        work_item.preprocess_submitted_at = time.monotonic()
        try:
            return self.pool.apply_async(
                _worker_process_task,
                args=(
                    work_item.mm_inputs,
                    work_item.capture_inspection_sources,
                    work_item.deadline,
                    inspection_task,
                    work_item.region,
                    vit_media.worker_spec(getattr(work_item, "media_traces", ())),
                ),
                callback=completed,
                error_callback=completed,
            )
        except BaseException as error:
            if inspection_task is not None:
                channel.finish(inspection_task.task_id, error=error)
            raise

    def _rebuild_pool(self) -> None:
        """Tear down the current pool and create a fresh one.

        Called when a worker dies / the pool's manager pipes are broken — without this
        the pool stays in a permanently-unusable state and every subsequent submit fails.
        """
        old = self.pool
        self.pool = None
        if self._inspection_channel is not None:
            self._inspection_channel.fail_all(
                RuntimeError("CPU preprocessing pool rebuilt")
            )
        self._clear_preprocess_tasks()
        try:
            if old is not None:
                old.terminate()
                old.join()
        except Exception as e:
            logging.warning(f"terminate broken pool failed: {e}")
        self._create_pool()

    def submit(self, work_item: "MMWorkItem") -> None:
        if not work_item.should_preprocess:
            return

        task_id: Optional[int] = None
        try:
            # Serialize submission with pool rebuilds. This keeps a task from
            # being submitted to an old pool while its queue accounting resets.
            with self._pool_lock:
                if (
                    getattr(self, "_pool_broken", None) is not None
                    and self._pool_broken.is_set()
                ):
                    self._rebuild_pool()
                    self._pool_broken.clear()
                # Track only after taking the rebuild lock. Otherwise a pool
                # rebuild can clear the task between accounting and submit.
                task_id = self._track_preprocess_task()
                try:
                    work_item.future = self._apply_async(work_item, task_id)
                except (BrokenPipeError, OSError, EOFError) as e:
                    # multiprocessing.Pool surfaces broken state via these —
                    # rebuild and retry once.
                    logging.error(
                        f"Pool broken on submit, rebuilding: {e}", exc_info=True
                    )
                    self._finish_preprocess_task(task_id)
                    self._rebuild_pool()
                    task_id = self._track_preprocess_task()
                    work_item.future = self._apply_async(work_item, task_id)
            return
        except (BrokenPipeError, OSError, EOFError) as e:
            if task_id is not None:
                self._finish_preprocess_task(task_id)
            vit_media.failure(
                "preprocess.submit", e, work_item.media_traces, "worker_lost"
            )
            raise
        except Exception as e:
            if task_id is not None:
                self._finish_preprocess_task(task_id)
            vit_media.failure("preprocess.submit", e, work_item.media_traces)
            logging.error(f"Unexpected error during submission: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            payload = work_item.future.get(timeout=work_item.remaining_timeout_s())
            vit_media.consume(
                getattr(work_item, "media_traces", ()),
                getattr(payload, "trace_timings", None),
            )
            (
                result,
                preprocess_time,
                download_time,
            ) = _unpack_preprocess_result(payload)
            self._set_result(work_item, result)
            _report_result_path(work_item, payload)
            work_item.future = None
            with self._pool_lock:
                self._consecutive_timeouts = 0
            _report_preprocess_timing(preprocess_time, download_time)
        except _PreprocessTaskError as e:
            vit_media.consume(getattr(work_item, "media_traces", ()), e.trace_timings)
            _report_vit_preprocess_samples(e.samples)
            logging.error("Preprocessing task failed: %s", e.error, exc_info=True)
            raise e.error from e
        except multiprocessing.pool.TimeoutError as e:
            vit_media.failure("preprocess.worker", e, work_item.media_traces, "timeout")
            with self._pool_lock:
                self._record_timeout_locked()
            raise TimeoutError(
                f"Preprocessing timeout after {work_item.mm_timeout_ms}ms"
            )
        except (BrokenPipeError, OSError, EOFError) as e:
            # worker died mid-task → pool is broken; rebuild so subsequent submits work
            vit_media.failure(
                "preprocess.worker", e, work_item.media_traces, "worker_lost"
            )
            logging.error(f"Pool broken on get_result, rebuilding: {e}", exc_info=True)
            with self._pool_lock:
                try:
                    self._rebuild_pool()
                except Exception as rb:
                    logging.error(f"pool rebuild failed: {rb}", exc_info=True)
            raise
        except Exception as e:
            vit_media.failure("preprocess.worker", e, work_item.media_traces)
            logging.error(f"Error getting preprocess result: {e}", exc_info=True)
            raise
        finally:
            with self._preprocess_queue_lock:
                queue_size = len(self._pending_preprocess_tasks)
            _report_preprocess_queue_size(queue_size)

    @staticmethod
    def _get_child_pids_from_pool(pool: multiprocessing.pool.Pool) -> List[int]:
        try:
            return [p.pid for p in pool._pool if p.is_alive()]
        except Exception:
            return []

    def shutdown(self) -> None:
        if self.pool is None:
            return
        if self._inspection_channel is not None:
            self._inspection_channel.fail_all(
                RuntimeError("CPU preprocessing pool stopped")
            )
        logging.info("Shutting down the preprocessing pool...")
        pool = self.pool
        pool.close()
        # Bounded join: if any worker is stuck running a long task, fall back
        # to terminate() so shutdown can't hang indefinitely.
        join_thread = threading.Thread(target=pool.join, daemon=True)
        join_thread.start()
        join_thread.join(timeout=10)
        if join_thread.is_alive():
            logging.warning("Preprocessing pool join exceeded 10s, terminating workers")
            pool.terminate()
            pool.join()
        self._clear_preprocess_tasks()
        if self._inspection_channel is not None:
            self._inspection_channel.close()
        self.pool = None
        logging.info("Preprocessing pool shut down.")


class _LocalResult:
    """本地预处理结果的简单包装类"""

    def __init__(self, payload: PreprocessTaskResult):
        self.payload = payload

    def get(self, timeout: Optional[float] = None) -> PreprocessTaskResult:
        return self.payload
