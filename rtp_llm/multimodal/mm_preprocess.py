"""CPU preprocessing workers, IPC result collection and timing."""

from __future__ import annotations

import logging
import multiprocessing.pool
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set, Tuple, TYPE_CHECKING

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.multimodal_util import (
    collect_download_timing,
    url_data_cache_,
    vit_emb_cache_,
)
from rtp_llm.multimodal.vit_metrics import (
    VitMetricSample,
    collect_vit_preprocess_metrics,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.multimodal.mm_preprocess_result import MMPreprocessResult
from rtp_llm.utils.time_util import Timer

if TYPE_CHECKING:
    from rtp_llm.multimodal.mm_work_item import MMWorkItem


_worker_vit_config: Optional[VitConfig] = None
_worker_preprocess_params: Optional[dict] = None
_worker_preprocess_func: Optional[Callable] = None


@dataclass
class PreprocessTaskResult:
    result: Any
    total_ms: float
    download_ms: float
    samples: List[VitMetricSample]
    started_at: float
    finished_at: float

    def __iter__(self):
        # Preserve the existing internal result-unpacking helper contract.
        return iter((self.result, self.total_ms, self.download_ms, self.samples))


def _run_preprocess_task(
    preprocess_func: Callable,
    mm_inputs: List[MultimodalInput],
    vit_config: VitConfig,
    preprocess_params: dict,
) -> PreprocessTaskResult:
    """Run preprocessing and return total and media-load durations in ms."""
    started_at = time.monotonic()
    with collect_vit_preprocess_metrics() as preprocess_metrics:
        with collect_download_timing() as download_timing:
            with Timer() as route_timer:
                result = preprocess_func(mm_inputs, vit_config, **preprocess_params)
    return PreprocessTaskResult(
        result,
        route_timer.cost_ms(),
        download_timing.elapsed_ms,
        preprocess_metrics.samples,
        started_at,
        time.monotonic(),
    )


def _report_result_path(work_item, payload):
    """Same-host monotonic timestamps include dispatch/IPC, not pure memcpy.

    The receive stamp is recorded by the pool callback, so waiting behind other
    request work in the caller does not inflate the return-path measurement.
    """
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
) -> None:
    """
    每个工作进程启动时调用的初始化函数。
    接收一次不变的参数，并将其存储在进程的全局变量中。
    """
    global _worker_vit_config, _worker_preprocess_params, _worker_preprocess_func
    # 让工作进程忽略 SIGINT 信号，这样主进程的 Ctrl+C 不会杀死它们
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker_vit_config = vit_config
    _worker_preprocess_params = preprocess_params
    _worker_preprocess_func = preprocess_func
    # Spawned workers import fresh module-level caches; parent-only resizing
    # does not propagate. Apply the operator's capacities once per worker.
    url_data_cache_.resize_cache(vit_config.url_cache_item_num)
    vit_emb_cache_.resize_cache(vit_config.mm_cache_item_num)
    logging.info(f"Worker process {os.getpid()} initialized.")


def _worker_process_task(
    mm_inputs: List[MultimodalInput],
    capture_inspection_sources: bool = False,
    deadline: Optional[float] = None,
) -> PreprocessTaskResult:
    """
    只接收变化的 `mm_inputs` 参数。
    """
    if _worker_preprocess_func is None:
        raise RuntimeError("Worker process has not been initialized correctly.")

    params = dict(_worker_preprocess_params)
    if capture_inspection_sources:
        params.update(capture_inspection_sources=True, deadline=deadline)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("ViT request expired before CPU preprocessing")
    return _run_preprocess_task(
        _worker_preprocess_func,
        mm_inputs,
        _worker_vit_config,
        params,
    )


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
                if not event.wait(
                    timeout=min(item.remaining_timeout_s() for item in pending)
                ):
                    self._on_wait_timeout()
                    raise TimeoutError("CPU preprocessing result wait timed out")

    def _on_wait_timeout(self):
        pass

    @staticmethod
    def _set_result(work_item, result):
        if isinstance(result, MMPreprocessResult):
            if len(result.inspection_sources) != len(work_item.mm_inputs):
                raise ValueError("CPU inspection sources changed input count")
            work_item.inspection_sources = result.inspection_sources
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
            )
            result, preprocess_time, download_time, samples = payload
            self._set_result(work_item, result)
            # 使用简单的对象模拟 future 行为
            work_item.future = _LocalResult(payload)
            work_item.preprocess_received_at = time.monotonic()
            work_item.preprocess_ready = True
            work_item.preprocess_ready_event.set()
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
            _, preprocess_time, download_time = _unpack_preprocess_result(payload)
            _report_result_path(work_item, payload)
            _report_preprocess_timing(preprocess_time, download_time)
            work_item.future = None
        except Exception as e:
            logging.error(f"Error getting local preprocess result: {e}", exc_info=True)
            raise


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
    ):
        self.mp_context = mp_context
        self.vit_config = vit_config
        self.preprocess_params = preprocess_params
        self.preprocess_func = preprocess_func
        self.pool: Optional[multiprocessing.pool.Pool] = None
        self._consecutive_timeouts = 0
        self._max_consecutive_timeouts = vit_config.mm_preprocess_max_workers
        # Track accepted work items independently of multiprocessing.Pool's
        # private task queue. The gauge includes both running and waiting tasks.
        self._preprocess_queue_lock = threading.Lock()
        self._pending_preprocess_tasks: Set[int] = set()
        self._next_preprocess_task_id = 0
        # Serializes timeout-counter updates and pool rebuilds — without it
        # concurrent get_result/submit callers can race to _rebuild_pool, double
        # tear down the pool, or miscount consecutive timeouts.
        self._pool_lock = threading.Lock()
        _report_preprocess_queue_size(0)
        self._create_pool()

    def _create_pool(self) -> None:
        """创建进程池"""
        logging.info(
            f"Creating multiprocessing pool for preprocessing with {self.vit_config.mm_preprocess_max_workers} workers"
        )
        self.pool = self.mp_context.Pool(
            processes=self.vit_config.mm_preprocess_max_workers,
            initializer=_worker_initializer,
            initargs=(
                self.vit_config,
                self.preprocess_params,
                self.preprocess_func,
            ),
        )

    def _on_wait_timeout(self):
        with self._pool_lock:
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                self._rebuild_pool()
                self._consecutive_timeouts = 0

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
        def completed(_result):
            work_item.preprocess_received_at = time.monotonic()
            self._finish_preprocess_task(task_id)
            work_item.preprocess_ready = True
            work_item.preprocess_ready_event.set()

        work_item.preprocess_submitted_at = time.monotonic()
        return self.pool.apply_async(
            _worker_process_task,
            args=(
                work_item.mm_inputs,
                work_item.capture_inspection_sources,
                work_item.deadline,
            ),
            callback=completed,
            error_callback=completed,
        )

    def _rebuild_pool(self) -> None:
        """Tear down the current pool and create a fresh one.

        Called when a worker dies / the pool's manager pipes are broken — without this
        the pool stays in a permanently-unusable state and every subsequent submit fails.
        """
        old = self.pool
        self.pool = None
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
            raise
        except Exception as e:
            if task_id is not None:
                self._finish_preprocess_task(task_id)
            logging.error(f"Unexpected error during submission: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            payload = work_item.future.get(timeout=work_item.remaining_timeout_s())
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
        except multiprocessing.pool.TimeoutError:
            with self._pool_lock:
                self._consecutive_timeouts += 1
                if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                    logging.warning(
                        f"Hit {self._consecutive_timeouts} consecutive timeouts, "
                        f"rebuilding pool (workers may be stuck)"
                    )
                    self._rebuild_pool()
                    self._consecutive_timeouts = 0
            raise TimeoutError(
                f"Preprocessing timeout after {work_item.mm_timeout_ms}ms"
            )
        except (BrokenPipeError, OSError, EOFError) as e:
            # worker died mid-task → pool is broken; rebuild so subsequent submits work
            logging.error(f"Pool broken on get_result, rebuilding: {e}", exc_info=True)
            with self._pool_lock:
                try:
                    self._rebuild_pool()
                except Exception as rb:
                    logging.error(f"pool rebuild failed: {rb}", exc_info=True)
            raise
        except Exception as e:
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
        logging.info("Preprocessing pool shut down.")


class _LocalResult:
    """本地预处理结果的简单包装类"""

    def __init__(self, payload: PreprocessTaskResult):
        self.payload = payload

    def get(self, timeout: Optional[float] = None) -> PreprocessTaskResult:
        return self.payload
