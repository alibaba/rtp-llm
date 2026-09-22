import concurrent.futures
import gc
import logging
import multiprocessing.pool
import os
import signal
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.profiler

from rtp_llm.access_logger.access_logger import MMAccessLogger
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.mm_embedding_cache import (
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
    MMHashKeyCache,
)
from rtp_llm.multimodal.mm_profiler import MMProfiler
from rtp_llm.multimodal.mm_scheduler import MMScheduler, MMSchedulerRequestTooLargeError
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MMWorkEstimate,
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import (
    collect_download_timing,
    trans_mm_input,
    url_data_cache_,
    vit_emb_cache_,
)
from rtp_llm.multimodal.vit_metrics import (
    VitMetricSample,
    collect_vit_preprocess_metrics,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.time_util import Timer

_worker_vit_config: Optional[VitConfig] = None
_worker_preprocess_params: Optional[dict] = None
_worker_preprocess_func: Optional[Callable] = None


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
    logging.info(
        "Worker process %s initialized: mm_cache_item_num=%s, url_cache_item_num=%s",
        os.getpid(),
        vit_config.mm_cache_item_num,
        vit_config.url_cache_item_num,
    )


def _worker_process_task(
    mm_inputs: List[MultimodalInput],
) -> Tuple[Any, float, List[VitMetricSample]]:
    """
    只接收变化的 `mm_inputs` 参数。
    """
    if _worker_preprocess_func is None:
        raise RuntimeError("Worker process has not been initialized correctly.")

    return _run_preprocess_task(
        _worker_preprocess_func,
        mm_inputs,
        _worker_vit_config,
        _worker_preprocess_params,
    )


def _run_preprocess_task(
    preprocess_func: Callable,
    mm_inputs: List[MultimodalInput],
    vit_config: VitConfig,
    preprocess_params: dict,
) -> Tuple[Any, float, List[VitMetricSample]]:
    """Keep worker samples and add M3's non-overlapping millisecond timings."""
    with collect_vit_preprocess_metrics() as preprocess_metrics:
        with collect_download_timing() as download_timing:
            with Timer() as route_timer:
                result = preprocess_func(mm_inputs, vit_config, **preprocess_params)
    total_ms = max(0.0, route_timer.cost_ms())
    download_ms = max(0.0, min(download_timing.elapsed_ms, total_ms))
    preprocess_metrics.report(GaugeMetrics.VIT_DOWNLOAD_RT_METRIC, download_ms)
    preprocess_metrics.report(
        GaugeMetrics.VIT_PREPROCESS_OTHER_RT_METRIC, total_ms - download_ms
    )
    return result, total_ms, preprocess_metrics.samples


def _report_vit_preprocess_samples(samples: List[VitMetricSample]) -> None:
    for sample in samples:
        try:
            kmonitor.report(sample.metric, sample.value, sample.tags)
        except Exception:
            logging.exception(
                "Failed to report ViT preprocess metric %s", sample.metric
            )


def _report_preprocess_queue_size(queue_size: int) -> None:
    """Report the number of preprocessing work items not yet completed."""
    try:
        kmonitor.report(
            GaugeMetrics.VIT_PREPROCESS_QUEUE_SIZE_METRIC, max(0, int(queue_size))
        )
    except Exception:
        # Telemetry must never change the preprocessing result.
        logging.exception("Failed to report ViT preprocess queue size")


def _count_images(mm_inputs: List[MultimodalInput]) -> int:
    """Count image-like inputs without treating videos or audio as images."""
    return sum(
        mm_input.mm_type in (MMUrlType.DEFAULT, MMUrlType.IMAGE)
        for mm_input in mm_inputs
    )


def _report_image_count(mm_inputs: List[MultimodalInput]) -> None:
    """Report the image count once for the current logical request."""
    try:
        kmonitor.report(
            GaugeMetrics.VIT_IMAGE_COUNT_METRIC,
            _count_images(mm_inputs),
        )
    except Exception:
        # Telemetry must never change the multimodal request result.
        logging.exception("Failed to report ViT image count")


class PreprocessExecutor:
    """预处理执行器抽象基类，封装预处理逻辑"""

    def submit(self, work_item: "MMWorkItem") -> None:
        raise NotImplementedError

    def get_result(self, work_item: "MMWorkItem") -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


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
            result, preprocess_time, samples = _run_preprocess_task(
                self.preprocess_func,
                work_item.mm_inputs,
                self.vit_config,
                self.preprocess_params,
            )
            work_item.preprocess_result = result
            # 使用简单的对象模拟 future 行为
            work_item.future = _LocalResult(result, preprocess_time, samples)
        except Exception as e:
            logging.error(f"Error in local preprocessing: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            _, preprocess_time, samples = work_item.future.get()
            _report_vit_preprocess_samples(
                [
                    VitMetricSample(
                        GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time
                    )
                ]
                + samples
            )
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
        # Accepted, unfinished work; includes both running and waiting tasks (M3).
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
            queue_size = len(self._pending_preprocess_tasks)
        _report_preprocess_queue_size(queue_size)

    def _clear_preprocess_tasks(self) -> None:
        with self._preprocess_queue_lock:
            if not self._pending_preprocess_tasks:
                return
            self._pending_preprocess_tasks.clear()
        _report_preprocess_queue_size(0)

    def _apply_async(self, work_item: "MMWorkItem", task_id: int) -> Any:
        return self.pool.apply_async(
            _worker_process_task,
            args=(work_item.mm_inputs,),
            callback=lambda _result: self._finish_preprocess_task(task_id),
            error_callback=lambda _error: self._finish_preprocess_task(task_id),
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
        try:
            kmonitor.report(AccMetrics.VIT_PROCESS_POOL_RESTART_QPS_METRIC, 1)
        except Exception:
            logging.exception("Failed to report ViT process pool restart")

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
            work_item.preprocess_result, preprocess_time, samples = (
                work_item.future.get(timeout=work_item.mm_timeout_ms / 1000.0)
            )
            with self._pool_lock:
                self._consecutive_timeouts = 0
            _report_vit_preprocess_samples(
                [
                    VitMetricSample(
                        GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time
                    )
                ]
                + samples
            )
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

    def __init__(
        self, result: Any, time: float, samples: Optional[List[VitMetricSample]] = None
    ):
        self.result = result
        self.time = time
        self.samples = samples or []

    def get(
        self, timeout: Optional[float] = None
    ) -> Tuple[Any, float, List[VitMetricSample]]:
        return (self.result, self.time, self.samples)


class MMEmbeddingRes:
    """Result container for multimodal embedding operations."""

    def __init__(
        self,
        embeddings: List[torch.Tensor],
        position_ids: Optional[List[torch.Tensor]] = None,
        extra_input: Optional[List[torch.Tensor]] = None,
        feature_hashes: Optional[List[torch.Tensor]] = None,
    ):
        self.feature_hashes = feature_hashes if feature_hashes is not None else []
        self.embeddings = embeddings
        self.position_ids = position_ids if position_ids is not None else []
        # Model-specific extra input, one opaque flat 1-D tensor per image (e.g. deepstack).
        self.extra_input = extra_input if extra_input is not None else []

    def __str__(self) -> str:
        return f"MMEmbeddingRes(length={len(self.embeddings)}, embeddings_shape={[e.shape for e in self.embeddings]}, position_ids_shape={[p.shape for p in self.position_ids] if self.position_ids is not None else []}, extra_input_shape={[d.shape for d in self.extra_input] if self.extra_input is not None else []})"


def _embedding_token_length(embeddings: List[Any]) -> int:
    """Count output tokens, treating a non-empty 1-D vector as one token."""
    total = 0
    for embedding in embeddings:
        if isinstance(embedding, torch.Tensor):
            if embedding.numel() > 0:
                total += int(embedding.shape[0]) if embedding.ndim >= 2 else 1
        else:
            try:
                total += len(embedding)
            except TypeError:
                logging.warning(
                    "Cannot derive embedding length from %s", type(embedding).__name__
                )
    return total


def _report_embedding_length(results: List[MMEmbeddingRes], hashes_only=False) -> None:
    """Report once per logical result retrieval, including cache/hash-only hits."""
    try:
        length = sum(
            (
                sum(h.numel() for h in result.feature_hashes)
                if hashes_only
                else _embedding_token_length(result.embeddings)
            )
            for result in results
        )
        kmonitor.report(GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC, length)
    except Exception:
        logging.exception("Failed to report ViT embedding length")


def _feature_hashes_from_result(result: Any) -> List[torch.Tensor]:
    """Build sidecar hashes while tolerating empty test/compatibility results."""
    from rtp_llm.ops import get_multimodal_feature_hash

    embeddings = MMProcessEngine._maybe_tensor_to_list(result[0], dim=2)
    return [
        get_multimodal_feature_hash(embedding)
        for embedding in embeddings
        if not (
            isinstance(embedding, torch.Tensor)
            and (embedding.numel() == 0 or embedding.ndim == 0)
        )
    ]


class _AsyncComputeTask:
    def __init__(self, cache_key: str, entry: MMEmbeddingCacheEntry):
        self.cache_key = cache_key
        self.entry = entry
        self.request_ids: Set[int] = set()
        self.future: Optional[concurrent.futures.Future] = None


class MMWorkItem:
    """Represents a work item for processing multimodal inputs."""

    def __init__(
        self,
        mm_inputs: List[MultimodalInput],
        mm_timeout_ms: Optional[int] = 120000,
        embedding_cache: Optional[MMEmbeddingCache] = None,
        cache_claim: Optional[Tuple[str, MMEmbeddingCacheEntry, str]] = None,
        defer_cache_complete: bool = False,
        hash_key_cache: Optional[MMHashKeyCache] = None,
    ):
        if not mm_inputs:
            raise ValueError("No mm_input for work item")

        self.mm_inputs = mm_inputs
        # Resolve each input independently, then use the largest budget for the
        # shared batch so the worker matches the proxy deadline calculation.
        default_timeout_ms = (
            mm_timeout_ms
            if mm_timeout_ms is not None and mm_timeout_ms > 0
            else VitConfig.DEFAULT_MM_TIMEOUT_MS
        )
        self.mm_timeout_ms = max(
            (
                input.mm_preprocess_config.mm_timeout_ms
                if input.mm_preprocess_config.mm_timeout_ms > 0
                else default_timeout_ms
            )
            for input in self.mm_inputs
        )
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result: Optional[Any] = None
        self.embedding_result: Optional[Any] = None
        self.feature_hashes: Optional[List[torch.Tensor]] = None
        self.work_estimate: Optional[MMWorkEstimate] = None

        self.need_check_cache = len(mm_inputs) == 1 and mm_inputs[0].url != ""
        self.embedding_cache = embedding_cache
        self.hash_key_cache = hash_key_cache
        self.cache_key: Optional[str] = None
        self.cache_entry: Optional[MMEmbeddingCacheEntry] = None
        self.cache_state: Optional[str] = None
        self.defer_cache_complete = defer_cache_complete

        if cache_claim is not None:
            if not self.need_check_cache:
                raise ValueError("cache_claim requires one non-empty multimodal input")
            self.cache_key, self.cache_entry, self.cache_state = cache_claim
            if self.cache_state == "complete":
                self.embedding_result = self.cache_entry.wait()
                if (
                    self.hash_key_cache is not None
                    and self.embedding_cache is not None
                    and self.embedding_cache.peek(self.cache_key) is self.cache_entry
                ):
                    self.feature_hashes = self.hash_key_cache.get(
                        self.cache_key, self.cache_entry.generation
                    )
        elif self.need_check_cache and self.embedding_cache is not None:
            self.cache_key = self.mm_inputs[0].cache_key()
            self.cache_state, self.cache_entry = self.embedding_cache.try_acquire(
                self.cache_key
            )
            if self.cache_state == "complete":
                self.embedding_result = self.cache_entry.wait()
                if (
                    self.hash_key_cache is not None
                    and self.embedding_cache.peek(self.cache_key) is self.cache_entry
                ):
                    self.feature_hashes = self.hash_key_cache.get(
                        self.cache_key, self.cache_entry.generation
                    )

        # future 可以是 ApplyResult (multiprocess) 或 _LocalResult (local)
        self.future: Optional[Any] = None

    @property
    def waiting_for_cache(self) -> bool:
        return self.cache_state == "in_progress" and self.embedding_result is None

    @property
    def should_preprocess(self) -> bool:
        return self.embedding_result is None and not self.waiting_for_cache

    def complete_cache(self, result: Any, force: bool = False) -> None:
        if self.feature_hashes is None:
            self.feature_hashes = _feature_hashes_from_result(result)
        if (
            (self.defer_cache_complete and not force)
            or self.embedding_cache is None
            or self.cache_key is None
            or self.cache_entry is None
        ):
            return
        self.embedding_cache.complete(
            self.cache_key, self.cache_entry, result, self.feature_hashes
        )
        if (
            self.hash_key_cache is not None
            and self.embedding_cache.peek(self.cache_key) is self.cache_entry
        ):
            self.hash_key_cache.put(
                self.cache_key, self.feature_hashes or [], self.cache_entry.generation
            )

    def fail_cache(self, error: Exception) -> None:
        if (
            self.embedding_cache is None
            or self.cache_key is None
            or self.cache_entry is None
            or self.cache_state != "miss"
        ):
            return
        self.embedding_cache.fail(self.cache_key, self.cache_entry, error)


class MMProcessEngine:
    """Engine for processing multimodal inputs with preprocessing and embedding."""

    def __init__(
        self,
        mm_part: MultiModalEmbeddingInterface,
        model_config: ModelConfig,
        vit_config: VitConfig,
        profiling_debug_logging_config: ProfilingDebugLoggingConfig,
        server_id: int = 0,
        is_proxy_mode: bool = False,
        device: str = "cuda:0",
    ):
        """
        Initialize the multimodal process engine.

        Args:
            model: 模型实例
            server_id: 服务器 ID
            vit_config: VIT 配置
            profiling_debug_logging_config: 性能调试日志配置
            is_proxy_mode: 是否在 proxy 模式下运行
                          - True: proxy 模式下的 worker 进程，QPS 由 proxy 层记录，此处不记录
                          - False: standalone 模式，需要在此处记录 QPS
        """
        self.server_id = server_id
        self.vit_config = vit_config
        self.is_proxy_mode = is_proxy_mode
        # The CUDA device this engine's weights live on; handed to the scheduler
        # so its background thread pins the right device before every forward.
        self.device = device
        self.contains_pos: bool = (
            model_config.mm_model_config.mm_position_ids_style != 0
        )
        self.mm_preprocess_batch_size: int = (
            model_config.mm_related_params.preprocess_batch_size
        )

        self.mp_context = multiprocessing.get_context("spawn")

        self.mm_part = mm_part

        # threading.Lock: protects gRPC-handler-thread access within this
        # process. multiprocessing.Lock would round-trip through an OS
        # semaphore on every acquire — wasteful since no cross-process sharing.
        self.query_num_lock = threading.Lock()

        # 根据 vit_config 创建预处理执行器
        preprocess_params = self.mm_part.get_preprocess_params()
        preprocess_func = self.mm_part.preprocess_input

        if vit_config.use_local_preprocess:
            self.preprocess_executor: PreprocessExecutor = LocalPreprocessExecutor(
                preprocess_func, vit_config, preprocess_params
            )
            logging.info(
                f"MMProcessEngine: Using LOCAL preprocessing mode (no subprocess pool)"
            )
        else:
            mp_context = multiprocessing.get_context("spawn")
            self.preprocess_executor = MultiprocessPreprocessExecutor(
                mp_context, vit_config, preprocess_params, preprocess_func
            )
            logging.info(
                f"MMProcessEngine: Using MULTIPROCESS preprocessing mode with {vit_config.mm_preprocess_max_workers} workers"
            )

        # All GPU embeddings use one scheduler path. max_batch_size=1 is the
        # serial mode; values greater than one enable cross-request batching.
        self.profiler = MMProfiler()

        try:
            scheduler_args = vit_config.embedding_scheduler_args()
            self._scheduler = MMScheduler(
                mm_part=mm_part,
                device=self.device,
                forward_profiler=self.profiler.profile_forward,
                **scheduler_args,
            )
        except Exception:
            self.preprocess_executor.shutdown()
            raise
        self._embedding_cache = MMEmbeddingCache(
            vit_config.mm_cache_gpu_max_bytes,
            vit_config.mm_cache_cpu_max_bytes,
            report_metrics=True,
        )
        self._hash_key_cache = MMHashKeyCache(vit_config.mm_hash_key_cache_max_bytes)
        self._async_compute_workers = max(1, int(vit_config.vit_concurrency))
        self._async_queue_size = max(0, int(vit_config.vit_max_queue_size))
        self._async_compute_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._async_compute_workers,
            thread_name_prefix="mm-async-compute",
        )
        self._async_admission_capacity = (
            self._async_compute_workers + self._async_queue_size
        )
        self._async_admission_lock = threading.Lock()
        self._async_admitted = 0
        # Future.cancel() invokes done callbacks synchronously.
        self._async_task_lock = threading.RLock()
        self._async_tasks: Dict[MMEmbeddingCacheEntry, _AsyncComputeTask] = {}
        self._async_request_tasks: Dict[int, Set[MMEmbeddingCacheEntry]] = {}
        self._async_cache = self._embedding_cache
        self._stopped = False
        logging.info(
            f"MMProcessEngine: MMScheduler "
            f"(configured_max_batch_size={vit_config.gpu_max_batch_size}, "
            f"configured_batch_wait_ms={vit_config.gpu_batch_wait_ms}, "
            f"{scheduler_args})"
        )

        self.query_num: int = 0
        self._access_logger = MMAccessLogger(
            get_log_path(),
            profiling_debug_logging_config.log_file_backup_count,
        )

        vit_emb_cache_.resize_cache(self.vit_config.mm_cache_item_num)
        url_data_cache_.resize_cache(self.vit_config.url_cache_item_num)

    def inc_query_num(self) -> None:
        """Increment the query counter."""
        with self.query_num_lock:
            self.query_num += 1

    def dec_query_num(self) -> None:
        """Decrement the query counter."""
        with self.query_num_lock:
            self.query_num -= 1

    def get_query_num(self) -> int:
        """Get the current number of active queries."""
        with self.query_num_lock:
            return self.query_num

    @staticmethod
    def _maybe_tensor_to_list(tensor: Any, dim: int = 2) -> List[Any]:
        """Convert tensor to list format if needed."""
        if tensor is None:
            return []
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if len(tensor.shape) > dim:
            return list(tensor)
        return [tensor]

    def mm_embedding_rpc(self, mm_inputs: MultimodalInputsPB) -> MMEmbeddingRes:
        """Process multimodal inputs from RPC protocol buffer."""
        converted_inputs = trans_mm_input(mm_inputs)
        return self.mm_embedding_impl(
            converted_inputs,
            request_id=getattr(mm_inputs, "request_id", 0),
        )

    def mm_embedding_cpp(
        self,
        urls: List[str],
        types: List[int],
        tensors: List[torch.Tensor],
        mm_preprocess_configs: List[Any],
        request_id: int = 0,
    ) -> MMEmbeddingRes:
        """Process multimodal inputs from C++ interface."""
        mm_inputs = [
            MultimodalInput(
                url, MMUrlType(url_type), tensor, MMPreprocessConfig(*config)
            )
            for url, url_type, tensor, config in zip(
                urls, types, tensors, mm_preprocess_configs
            )
        ]
        res = self.mm_embedding_impl(mm_inputs, request_id=request_id)
        res.position_ids = [pos.cpu() for pos in res.position_ids]
        return res

    def mm_embedding_impl(
        self,
        mm_inputs: List[MultimodalInput],
        request_id: int = 0,
    ) -> MMEmbeddingRes:
        result, _ = self._mm_embedding_impl(
            mm_inputs,
            request_id=request_id,
        )
        return result

    def _mm_embedding_impl(
        self,
        mm_inputs: List[MultimodalInput],
        request_id: int = 0,
        cache_claim=None,
        defer_cache_complete=False,
        report_image_count=True,
        report_embedding_length=True,
    ):
        work_items = []
        self.inc_query_num()
        try:
            with torch.profiler.record_function("mm_embedding_impl"):
                if self._stopped:
                    raise RuntimeError("MMProcessEngine is stopped")
                if report_image_count:
                    _report_image_count(mm_inputs)
                if not self.is_proxy_mode:
                    kmonitor.report(
                        AccMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"}
                    )
                if not self.vit_config.disable_access_log:
                    self._access_logger.log_query_access(
                        mm_inputs, request_id=request_id
                    )
                work_items = self._create_work_items(
                    mm_inputs,
                    cache_claim=cache_claim,
                    defer_cache_complete=defer_cache_complete,
                )
                self._wait_for_preprocessing(work_items)
                with torch.profiler.record_function("compute_embeddings"):
                    embeddings, positions, extras = self._compute_embeddings(work_items)
                result = MMEmbeddingRes(
                    embeddings,
                    positions,
                    extras,
                    [h for item in work_items for h in (item.feature_hashes or [])],
                )
                if report_embedding_length:
                    _report_embedding_length([result])
                if not self.vit_config.disable_access_log:
                    self._access_logger.log_success_access(
                        mm_inputs, str(result), request_id=request_id
                    )
                if not self.is_proxy_mode:
                    kmonitor.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)
                return result, work_items
        except Exception as error:
            for item in work_items:
                item.fail_cache(error)
            self.report_vit_error(error)
            if isinstance(error, torch.cuda.OutOfMemoryError) or isinstance(
                error.__cause__, torch.cuda.OutOfMemoryError
            ):
                torch.cuda.empty_cache()
                gc.collect()
            self._access_logger.log_exception_access(
                mm_inputs, error, request_id=request_id
            )
            raise
        finally:
            self.dec_query_num()

    def _create_work_items(
        self,
        mm_inputs: List[MultimodalInput],
        cache_claim=None,
        defer_cache_complete=False,
    ) -> List[MMWorkItem]:
        """Create work items and submit preprocessing tasks."""
        # Request-level image-count cap BEFORE any preprocessing: a request whose
        # media count exceeds the scheduler's per-request limit can never fit a
        # batch, so reject it up front instead of spending preprocess on it. (Serial
        # mode's cap is sys.maxsize, i.e. no limit — matches the old behavior.)
        max_images = self._scheduler.max_request_images
        if len(mm_inputs) > max_images:
            raise MMSchedulerRequestTooLargeError(
                f"request image count {len(mm_inputs)} exceeds per-request limit "
                f"{max_images}, request rejected"
            )

        if not mm_inputs:
            return []
        if cache_claim is not None and len(mm_inputs) != 1:
            raise ValueError("cache_claim is only valid for one multimodal input")
        batch_size = (
            self.mm_preprocess_batch_size
            if self.mm_preprocess_batch_size != -1
            else len(mm_inputs)
        )

        work_items = []
        try:
            for index in range(0, len(mm_inputs), batch_size):
                work_item = MMWorkItem(
                    mm_inputs[index : index + batch_size],
                    mm_timeout_ms=self.vit_config.mm_timeout_ms,
                    embedding_cache=self._embedding_cache,
                    hash_key_cache=self._hash_key_cache,
                    cache_claim=cache_claim if index == 0 else None,
                    defer_cache_complete=defer_cache_complete,
                )
                work_items.append(work_item)
                self.preprocess_executor.submit(work_item)
        except Exception as error:
            for item in work_items:
                item.fail_cache(error)
            raise
        return work_items

    def _wait_for_preprocessing(
        self,
        work_items: List[MMWorkItem],
    ) -> None:
        """Wait for all preprocessing tasks to complete."""
        for work_item in work_items:
            if work_item.waiting_for_cache:
                timeout_s = (
                    work_item.mm_timeout_ms / 1000.0
                    if work_item.mm_timeout_ms is not None
                    and work_item.mm_timeout_ms > 0
                    else 120.0
                )
                work_item.embedding_result = work_item.cache_entry.wait(
                    timeout=timeout_s
                )
            self.preprocess_executor.get_result(work_item)
            if work_item.embedding_result is None:
                estimate = self.mm_part.estimate_work(
                    work_item.preprocess_result, work_item.mm_type
                )
                if estimate is not None and not isinstance(estimate, MMWorkEstimate):
                    raise TypeError(
                        "estimate_work must return MMWorkEstimate or None, got "
                        f"{type(estimate).__name__}"
                    )
                work_item.work_estimate = estimate

    def _compute_embeddings(
        self, work_items: List[MMWorkItem]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Compute embeddings for all work items."""
        pending_items = [wi for wi in work_items if wi.embedding_result is None]

        if pending_items:
            self._scheduler.submit_and_wait(pending_items)

        emb_res, pos_res, tensor_res = [], [], []
        for wi in work_items:
            result = wi.embedding_result
            # Scheduler invariant: submit_and_wait either fills embedding_result
            # for every pending item or raises, so it is never None here.
            if result is None:
                raise RuntimeError(f"embedding_result not set for work item {wi}")
            if wi.feature_hashes is None and wi.cache_entry is not None:
                wi.feature_hashes = self._hash_key_cache.get(
                    wi.cache_key, wi.cache_entry.generation
                )
            if wi.feature_hashes is None:
                wi.complete_cache(result)
            emb_res.extend(self._maybe_tensor_to_list(result[0], dim=2))
            pos_res.extend(self._maybe_tensor_to_list(result[1], dim=2))
            if len(result) > 2:
                tensor_res.extend(self._maybe_tensor_to_list(result[2], dim=1))
        return emb_res, pos_res, tensor_res

    def report_vit_error(
        self,
        error: Optional[Any] = None,
        entry: Optional[MMEmbeddingCacheEntry] = None,
    ) -> None:
        """Report one ViT error, suppressing duplicate reports for one result.

        A failed async task can be observed by the task callback and by the
        caller waiting on its cache entry. The cache-entry claim handles
        different exception objects representing the same result; the
        exception marker covers RPC/wrapper layers that re-raise the same
        object. Worker RPC responses carry a marker so a proxy can suppress
        the corresponding duplicate report.
        """
        if entry is not None:
            try:
                if not entry.claim_error_report():
                    # Preserve the cross-layer marker even when the entry was
                    # already claimed, so a later wrapper cannot count it
                    # again without the entry.
                    if error is not None:
                        setattr(error, "_vit_error_qps_reported", True)
                    return
            except Exception:
                # Telemetry must never turn a request failure into a different
                # failure if a compatibility cache entry lacks this helper.
                pass
        if error is not None:
            try:
                if getattr(error, "_vit_error_qps_reported", False):
                    return
                setattr(error, "_vit_error_qps_reported", True)
            except Exception:
                # A few third-party exception types may not have a writable
                # __dict__. Reporting is still more useful than failing the
                # request because deduplication was unavailable.
                pass
        try:
            # Report at the process where the failure is first observed. In a
            # proxy deployment the RPC response carries a marker, allowing the
            # proxy to report transport-only failures without double counting.
            kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
        except Exception:
            # Metrics must never mask the original ViT request failure.
            logging.exception("Failed to report ViT error QPS")

    @staticmethod
    def _work_item_result_to_response(
        result: Any, feature_hashes: Optional[List[torch.Tensor]] = None
    ) -> MMEmbeddingRes:
        emb_res = MMProcessEngine._maybe_tensor_to_list(result[0], dim=2)
        pos_res = MMProcessEngine._maybe_tensor_to_list(result[1], dim=2)
        extra_res = (
            MMProcessEngine._maybe_tensor_to_list(result[2], dim=1)
            if len(result) > 2
            else []
        )
        return MMEmbeddingRes(emb_res, pos_res, extra_res, feature_hashes)

    def async_submit(
        self, mm_inputs: List[MultimodalInput], request_id: int = 0
    ) -> List[str]:
        """Asynchronously submit multimodal URLs for embedding computation.

        Each input is submitted independently, keyed by its own cache_key.
        Returns the list of cache keys. Inputs already in-progress or complete
        are not recomputed.
        """
        try:
            claims = self._claim_and_submit_async(mm_inputs, request_id=request_id)
            return [cache_key for cache_key, _ in claims]
        except Exception as error:
            self.report_vit_error(error)
            raise

    def get_embedding_result(
        self,
        mm_inputs: List[MultimodalInput],
        timeout_ms: int = 120000,
        request_id: int = 0,
        cancellation_event: Optional[threading.Event] = None,
        hashes_only: bool = False,
    ) -> List[MMEmbeddingRes]:
        """Retrieve embedding results, blocking until ready if necessary.

        Each input is looked up independently by its cache_key.
        If a key was never submitted, queues it on the shared async executor.
        If in-progress, blocks until the computing thread finishes.
        If complete, returns immediately. With hashes_only, return sidecar hashes
        without exporting embeddings or promoting a CPU entry with cached hashes.
        """
        current_entry: Optional[MMEmbeddingCacheEntry] = None
        try:
            _report_image_count(mm_inputs)
            timeout_ms = timeout_ms or self.vit_config.mm_timeout_ms
            claims = self._claim_and_submit_async(
                mm_inputs,
                request_id=request_id,
                queue_timeout_ms=timeout_ms,
                cancellation_event=cancellation_event,
            )
            deadline = time.monotonic() + timeout_ms / 1000.0
            results = []
            for cache_key, entry in claims:
                current_entry = entry
                remaining = max(0.0, deadline - time.monotonic())
                entry.wait_ready(timeout=remaining)
                feature_hashes = self._hash_key_cache.get(cache_key, entry.generation)
                raw_result = None
                if not hashes_only or feature_hashes is None:
                    raw_result = entry.wait(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                if feature_hashes is None:
                    feature_hashes = _feature_hashes_from_result(raw_result)
                    self._hash_key_cache.put(
                        cache_key, feature_hashes, entry.generation
                    )
                if hashes_only:
                    results.append(MMEmbeddingRes([], feature_hashes=feature_hashes))
                else:
                    results.append(
                        self._work_item_result_to_response(raw_result, feature_hashes)
                    )
                del raw_result

            _report_embedding_length(results, hashes_only=hashes_only)
            return results
        except Exception as error:
            if hashes_only:
                self.cancel_queued_request(request_id)
            self.report_vit_error(error, current_entry)
            raise

    def _claim_and_submit_async(
        self,
        mm_inputs: List[MultimodalInput],
        request_id: int = 0,
        queue_timeout_ms: Optional[int] = None,
        cancellation_event: Optional[threading.Event] = None,
    ) -> List[Tuple[str, MMEmbeddingCacheEntry]]:
        claims: List[Tuple[str, MMEmbeddingCacheEntry]] = []
        pending: List[Tuple[MultimodalInput, str, MMEmbeddingCacheEntry]] = []
        for mm_input in mm_inputs:
            if mm_input.url == "":
                raise ValueError(
                    "async embedding requires non-empty url for each input"
                )

            cache_key = mm_input.cache_key()
            with self._async_task_lock:
                state, entry = self._async_cache.try_acquire(cache_key)
                claims.append((cache_key, entry))
                if (
                    state == "complete"
                    and self._embedding_cache.peek(cache_key) is entry
                ):
                    self._hash_key_cache.get(cache_key, entry.generation)
                if state == "miss":
                    self._async_tasks[entry] = _AsyncComputeTask(cache_key, entry)
                    pending.append((mm_input, cache_key, entry))

                if state in ("miss", "in_progress"):
                    task = self._async_tasks.get(entry)
                    if task is not None:
                        task.request_ids.add(request_id)
                        self._async_request_tasks.setdefault(request_id, set()).add(
                            entry
                        )

        self._raise_if_async_request_cancelled(request_id, cancellation_event)

        self._submit_async_compute_batch(
            pending,
            request_id=request_id,
            queue_timeout_ms=queue_timeout_ms,
        )
        self._raise_if_async_request_cancelled(request_id, cancellation_event)
        return claims

    def _raise_if_async_request_cancelled(
        self,
        request_id: int,
        cancellation_event: Optional[threading.Event],
    ) -> None:
        if cancellation_event is None or not cancellation_event.is_set():
            return
        self.cancel_queued_request(request_id)
        raise FtRuntimeException(
            ExceptionType.CANCELLED_ERROR,
            f"ViT request {request_id} was cancelled",
        )

    def _forget_async_task_locked(self, entry: MMEmbeddingCacheEntry) -> None:
        task = self._async_tasks.pop(entry, None)
        if task is None:
            return
        for request_id in task.request_ids:
            request_tasks = self._async_request_tasks.get(request_id)
            if request_tasks is None:
                continue
            request_tasks.discard(entry)
            if not request_tasks:
                self._async_request_tasks.pop(request_id, None)

    def cancel_queued_request(self, request_id: int) -> int:
        """Cancel work that is still queued and exclusively owned by a request.

        Running futures are deliberately left alone. A deduplicated task remains
        queued while any other request still owns it.
        """
        cancelled = 0
        with self._async_task_lock:
            entries = list(self._async_request_tasks.pop(request_id, set()))
            for entry in entries:
                task = self._async_tasks.get(entry)
                if task is None:
                    continue
                task.request_ids.discard(request_id)
                if task.request_ids:
                    continue

                if task.future is None:
                    error = FtRuntimeException(
                        ExceptionType.CANCELLED_ERROR,
                        f"ViT request {request_id} was cancelled before submission",
                    )
                    self._forget_async_task_locked(entry)
                    self._fail_async_compute(task.cache_key, entry, error)
                    cancelled += 1
                else:
                    try:
                        future_cancelled = task.future.cancel()
                    except Exception as error:
                        # A custom Future can fail while cancelling. It is an
                        # exceptional result even though no worker started.
                        self._fail_async_compute(task.cache_key, entry, error)
                        logging.exception("Failed to cancel queued ViT work")
                        continue
                    if future_cancelled:
                        # Future.cancel() normally invokes the done callback
                        # synchronously, but fail explicitly as well so a
                        # custom Future cannot leave an unreported terminal
                        # result.
                        self._fail_async_compute(
                            task.cache_key,
                            entry,
                            FtRuntimeException(
                                ExceptionType.CANCELLED_ERROR,
                                "ViT async compute cancelled before execution",
                            ),
                        )
                        cancelled += 1

        if cancelled:
            logging.info(
                "Cancelled %d queued ViT task(s) for request %d",
                cancelled,
                request_id,
            )
        return cancelled

    def _try_reserve_async_slots(self, count: int) -> Tuple[bool, int]:
        if count <= 0:
            return True, self._async_admitted
        with self._async_admission_lock:
            if self._async_admitted + count > self._async_admission_capacity:
                return False, self._async_admitted
            self._async_admitted += count
            return True, self._async_admitted

    def _release_async_slots(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self._async_admission_lock:
            self._async_admitted -= count
            if self._async_admitted < 0:
                logging.error(
                    "MMProcessEngine: async admission count underflow: %d",
                    self._async_admitted,
                )
                self._async_admitted = 0

    def _resolve_async_timeout_ms(
        self,
        pending: List[Tuple[MultimodalInput, str, MMEmbeddingCacheEntry]],
        queue_timeout_ms: Optional[int],
    ) -> int:
        if queue_timeout_ms is not None and queue_timeout_ms > 0:
            return queue_timeout_ms
        per_input_timeouts = [
            item.mm_preprocess_config.mm_timeout_ms
            for item, _, _ in pending
            if item.mm_preprocess_config.mm_timeout_ms > 0
        ]
        return max(per_input_timeouts, default=self.vit_config.mm_timeout_ms)

    def _fail_async_compute(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        error: Exception,
    ) -> None:
        self.report_vit_error(error, entry)
        try:
            self._embedding_cache.fail(cache_key, entry, error)
        except Exception as cache_error:
            self.report_vit_error(cache_error, entry)
            logging.exception("Failed to publish ViT failure to embedding cache")
            raise

    def _on_async_compute_done(
        self,
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        future: concurrent.futures.Future,
    ) -> None:
        with self._async_task_lock:
            try:
                if future.cancelled():
                    error = FtRuntimeException(
                        ExceptionType.CANCELLED_ERROR,
                        "ViT async compute cancelled before execution",
                    )
                    self._fail_async_compute(cache_key, entry, error)
                    return

                error = future.exception()
                if error is not None:
                    self._fail_async_compute(cache_key, entry, error)
            except Exception as callback_error:
                # Future inspection itself can fail for unusual Future
                # implementations; it is still an exceptional ViT result.
                self._fail_async_compute(cache_key, entry, callback_error)
            finally:
                try:
                    self._forget_async_task_locked(entry)
                except Exception as cleanup_error:
                    self.report_vit_error(cleanup_error, entry)
                    logging.exception("Failed to remove completed ViT async task")
                try:
                    self._release_async_slots()
                except Exception as cleanup_error:
                    self.report_vit_error(cleanup_error, entry)
                    logging.exception("Failed to release ViT async admission slot")

    def _run_async_compute(
        self,
        mm_inputs: List[MultimodalInput],
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        request_id: int,
        deadline: float,
    ) -> None:
        if time.monotonic() >= deadline:
            error = FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT,
                "ViT queue wait timed out before execution",
            )
            self._fail_async_compute(cache_key, entry, error)
            return
        self._async_compute(mm_inputs, cache_key, entry, request_id)

    def _submit_async_compute_batch(
        self,
        pending: List[Tuple[MultimodalInput, str, MMEmbeddingCacheEntry]],
        request_id: int = 0,
        queue_timeout_ms: Optional[int] = None,
    ) -> None:
        if not pending:
            return

        # Serialize submission with cancellation so a request cannot create new
        # queued work after its cancellation callback has already run.
        with self._async_task_lock:
            active_pending = [
                item
                for item in pending
                if item[2] in self._async_tasks
                and self._async_tasks[item[2]].request_ids
            ]
            if not active_pending:
                return

            accepted, admitted = self._try_reserve_async_slots(len(active_pending))
            if not accepted:
                error = FtRuntimeException(
                    ExceptionType.CONCURRENCY_LIMIT_ERROR,
                    "ViT queue is full: "
                    f"admitted={admitted}, capacity={self._async_admission_capacity}, "
                    f"requested={len(active_pending)}",
                )
                logging.warning(error.message)
                for _, cache_key, entry in active_pending:
                    self._forget_async_task_locked(entry)
                    self._fail_async_compute(cache_key, entry, error)
                raise error

            timeout_ms = self._resolve_async_timeout_ms(
                active_pending, queue_timeout_ms
            )
            deadline = time.monotonic() + timeout_ms / 1000.0
            submitted = 0
            try:
                for mm_input, cache_key, entry in active_pending:
                    future = self._async_compute_executor.submit(
                        self._run_async_compute,
                        [mm_input],
                        cache_key,
                        entry,
                        request_id,
                        deadline,
                    )
                    self._async_tasks[entry].future = future
                    future.add_done_callback(
                        lambda completed, key=cache_key, cache_entry=entry: self._on_async_compute_done(
                            key, cache_entry, completed
                        )
                    )
                    submitted += 1
            except Exception as error:
                self.report_vit_error(error)
                unsubmitted = active_pending[submitted:]
                self._release_async_slots(len(unsubmitted))
                for _, cache_key, entry in unsubmitted:
                    self._forget_async_task_locked(entry)
                    self._fail_async_compute(cache_key, entry, error)
                raise

    def _async_compute(
        self,
        mm_inputs: List[MultimodalInput],
        cache_key: str,
        entry: MMEmbeddingCacheEntry,
        request_id: int = 0,
    ) -> None:
        context = (
            torch.cuda.device(self.device)
            if str(self.device).startswith("cuda")
            else nullcontext()
        )
        try:
            with context:
                _, work_items = self._mm_embedding_impl(
                    mm_inputs,
                    cache_claim=(cache_key, entry, "miss"),
                    defer_cache_complete=True,
                    request_id=request_id,
                    report_image_count=False,
                    report_embedding_length=False,
                )
                raw_result = work_items[0].embedding_result
                if raw_result is None:
                    raise RuntimeError("async embedding did not produce a cache value")
                work_items[0].complete_cache(raw_result, force=True)
        except Exception as error:
            self._fail_async_compute(cache_key, entry, error)

    def stop(self) -> None:
        """Shutdown the embedding scheduler and preprocessing executor."""
        if self._stopped:
            return
        self._stopped = True
        self._async_compute_executor.shutdown(wait=False, cancel_futures=True)
        self._scheduler.close()
        self.preprocess_executor.shutdown()
        self._embedding_cache.clear(RuntimeError("MMProcessEngine stopped"))
        self._hash_key_cache.clear()
