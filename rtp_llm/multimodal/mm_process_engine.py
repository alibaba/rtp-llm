import asyncio
import concurrent.futures
import gc
import logging
import multiprocessing.pool
import os
import signal
import threading
import time
from types import SimpleNamespace
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
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict, get_greennet_provider
from rtp_llm.multimodal.mm_embedding_cache import (
    MMEmbeddingAsyncCache,
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
)
from rtp_llm.multimodal.mm_profiler import MMProfiler
from rtp_llm.multimodal.mm_scheduler import MMScheduler
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MMWorkEstimate,
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import (
    collect_download_timing,
    maybe_tensor_to_list,
    trans_mm_input,
    url_data_cache_,
    vit_emb_cache_,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.time_util import Timer, timer_wrapper

_worker_vit_config: Optional[VitConfig] = None
_worker_preprocess_params: Optional[dict] = None
_worker_preprocess_func: Optional[Callable] = None


def _run_preprocess_task(
    preprocess_func: Callable,
    mm_inputs: List[MultimodalInput],
    vit_config: VitConfig,
    preprocess_params: dict,
) -> Tuple[Any, float, float]:
    """Run preprocessing and return total and media-load durations in ms."""
    with collect_download_timing() as download_timing:
        with Timer() as route_timer:
            result = preprocess_func(mm_inputs, vit_config, **preprocess_params)
    return result, route_timer.cost_ms(), download_timing.elapsed_ms


def _embedding_token_length(embeddings: List[Any]) -> int:
    """Return the total visual-token length represented by embedding tensors."""
    total = 0
    for embedding in embeddings:
        if isinstance(embedding, torch.Tensor):
            # Embeddings are normally [tokens, hidden]. A 1-D tensor is a
            # single token vector, so its hidden dimension must not be counted
            # as the token length.
            total += int(embedding.shape[0]) if embedding.ndim >= 2 else 1
            continue
        try:
            total += len(embedding)
        except TypeError:
            logging.warning(
                "Cannot derive embedding length from %s", type(embedding).__name__
            )
    return total


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
    logging.info(f"Worker process {os.getpid()} initialized.")


def _worker_process_task(
    mm_inputs: List[MultimodalInput],
) -> Tuple[Any, float, float]:
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


def _unpack_preprocess_result(payload: Any) -> Tuple[Any, float, float]:
    """Normalize preprocess results from current and legacy executors."""
    if not isinstance(payload, tuple) or len(payload) < 2:
        raise ValueError(
            "preprocess executor must return (result, total_ms[, download_ms])"
        )
    result, total_ms = payload[0], float(payload[1])
    download_ms = float(payload[2]) if len(payload) > 2 else 0.0
    # Clock/context overhead must never make the derived remainder negative.
    download_ms = max(0.0, min(download_ms, total_ms))
    return result, total_ms, download_ms


def _report_preprocess_timing(total_ms: float, download_ms: float) -> None:
    """Report the two non-overlapping preprocessing components in milliseconds."""
    kmonitor.report(GaugeMetrics.VIT_DOWNLOAD_RT_METRIC, download_ms)
    kmonitor.report(
        GaugeMetrics.VIT_PREPROCESS_OTHER_RT_METRIC, max(0.0, total_ms - download_ms)
    )


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

    def submit(self, work_item: "MMWorkItem") -> None:
        if not work_item.should_preprocess:
            return

        try:
            result, preprocess_time, download_time = _run_preprocess_task(
                self.preprocess_func,
                work_item.mm_inputs,
                self.vit_config,
                self.preprocess_params,
            )
            work_item.preprocess_result = result
            # 使用简单的对象模拟 future 行为
            work_item.future = _LocalResult(result, preprocess_time, download_time)
        except Exception as e:
            logging.error(f"Error in local preprocessing: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            _, preprocess_time, download_time = _unpack_preprocess_result(
                work_item.future.get()
            )
            _report_preprocess_timing(preprocess_time, download_time)
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
        # Serializes timeout-counter updates and pool rebuilds — without it
        # concurrent get_result/submit callers can race to _rebuild_pool, double
        # tear down the pool, or miscount consecutive timeouts.
        self._pool_lock = threading.Lock()
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

    def _rebuild_pool(self) -> None:
        """Tear down the current pool and create a fresh one.

        Called when a worker dies / the pool's manager pipes are broken — without this
        the pool stays in a permanently-unusable state and every subsequent submit fails.
        """
        old = self.pool
        self.pool = None
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

        try:
            work_item.future = self.pool.apply_async(
                _worker_process_task, args=(work_item.mm_inputs,)
            )
            return
        except (BrokenPipeError, OSError, EOFError) as e:
            # multiprocessing.Pool surfaces broken state via these — rebuild and retry once.
            # Keep both rebuild and the retry submission under _pool_lock so another thread
            # cannot tear self.pool down between our rebuild and the apply_async call.
            logging.error(f"Pool broken on submit, rebuilding: {e}", exc_info=True)
            with self._pool_lock:
                self._rebuild_pool()
                work_item.future = self.pool.apply_async(
                    _worker_process_task, args=(work_item.mm_inputs,)
                )
        except Exception as e:
            logging.error(f"Unexpected error during submission: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            payload = work_item.future.get(timeout=work_item.mm_timeout_ms / 1000.0)
            (
                work_item.preprocess_result,
                preprocess_time,
                download_time,
            ) = _unpack_preprocess_result(payload)
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
        logging.info("Preprocessing pool shut down.")


class _LocalResult:
    """本地预处理结果的简单包装类"""

    def __init__(self, result: Any, preprocess_time: float, download_time: float):
        self.result = result
        self.preprocess_time = preprocess_time
        self.download_time = download_time

    def get(self, timeout: Optional[float] = None) -> Tuple[Any, float, float]:
        return (self.result, self.preprocess_time, self.download_time)


class MMEmbeddingRes:
    """Result container for multimodal embedding operations."""

    def __init__(
        self,
        embeddings: List[torch.Tensor],
        position_ids: Optional[List[torch.Tensor]] = None,
        extra_input: Optional[List[torch.Tensor]] = None,
    ):
        self.embeddings = embeddings
        self.position_ids = position_ids if position_ids is not None else []
        # Model-specific extra input, one opaque flat 1-D tensor per image (e.g. deepstack).
        self.extra_input = extra_input if extra_input is not None else []

    def __str__(self) -> str:
        return f"MMEmbeddingRes(length={len(self.embeddings)}, embeddings_shape={[e.shape for e in self.embeddings]}, position_ids_shape={[p.shape for p in self.position_ids] if self.position_ids is not None else []}, extra_input_shape={[d.shape for d in self.extra_input] if self.extra_input is not None else []})"


class _AsyncComputeTask:
    def __init__(self, cache_key: str, entry: MMEmbeddingCacheEntry):
        self.cache_key = cache_key
        self.entry = entry
        self.request_ids: Set[int] = set()
        self.future: Optional[concurrent.futures.Future] = None


def _derive_embedding_cache_max_bytes(
    mm_part: MultiModalEmbeddingInterface,
    model_config: ModelConfig,
    max_items: int,
) -> Optional[int]:
    """Translate the existing item limit into a model-derived byte budget."""

    if max_items <= 0:
        return 0
    budget = mm_part.get_batch_work_budget(1)
    if budget is None:
        return None
    if not isinstance(budget, MMWorkEstimate):
        raise TypeError(
            "get_batch_work_budget must return MMWorkEstimate or None, got "
            f"{type(budget).__name__}"
        )
    if budget.output_tokens <= 0:
        return None

    hidden_size = int(getattr(model_config, "hidden_size", 0) or 0)
    if hidden_size <= 0:
        word_embeddings = getattr(mm_part, "word_embedding_weight", None)
        if isinstance(word_embeddings, torch.Tensor) and word_embeddings.ndim >= 2:
            hidden_size = int(word_embeddings.shape[-1])
    if hidden_size <= 0:
        visual = getattr(mm_part, "visual", None)
        hidden_size = int(getattr(visual, "out_hidden_size", 0) or 0)
    if hidden_size <= 0:
        return None

    try:
        dtype_bytes = torch.empty((), dtype=mm_part._data_type).element_size()
    except (AttributeError, NotImplementedError, TypeError):
        return None
    return max_items * budget.output_tokens * hidden_size * dtype_bytes


class MMWorkItem:
    """Represents a work item for processing multimodal inputs."""

    def __init__(
        self,
        mm_inputs: List[MultimodalInput],
        mm_timeout_ms: Optional[int] = 120000,
        embedding_cache: Optional[MMEmbeddingCache] = None,
        cache_claim: Optional[Tuple[str, MMEmbeddingCacheEntry, str]] = None,
        defer_cache_complete: bool = False,
    ):
        if not mm_inputs:
            raise ValueError("No mm_input for work item")

        self.mm_inputs = mm_inputs
        # proto3 default for unset int is 0; treat <= 0 as "not set" and fall back to the
        # caller-provided default (which comes from VitConfig.mm_timeout_ms, always initialized
        # at server startup via --mm_timeout_ms / MM_TIMEOUT_MS env, default 120000ms).
        per_request_timeout = self.mm_inputs[0].mm_preprocess_config.mm_timeout_ms
        self.mm_timeout_ms = (
            per_request_timeout if per_request_timeout > 0 else mm_timeout_ms
        )
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result: Optional[Any] = None
        self.embedding_result: Optional[Any] = None
        self.work_estimate: Optional[MMWorkEstimate] = None

        self.need_check_cache = len(mm_inputs) == 1 and mm_inputs[0].url != ""
        self.embedding_cache = embedding_cache
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
        elif self.need_check_cache and self.embedding_cache is not None:
            self.cache_key = self.mm_inputs[0].cache_key()
            self.cache_state, self.cache_entry = self.embedding_cache.try_acquire(
                self.cache_key
            )
            if self.cache_state == "complete":
                self.embedding_result = self.cache_entry.wait()

        # future 可以是 ApplyResult (multiprocess) 或 _LocalResult (local)
        self.future: Optional[Any] = None

    @property
    def waiting_for_cache(self) -> bool:
        return self.cache_state == "in_progress" and self.embedding_result is None

    @property
    def should_preprocess(self) -> bool:
        return self.embedding_result is None and not self.waiting_for_cache

    def complete_cache(self, result: Any, force: bool = False) -> None:
        if (
            (self.defer_cache_complete and not force)
            or self.embedding_cache is None
            or self.cache_key is None
            or self.cache_entry is None
        ):
            return
        self.embedding_cache.complete(self.cache_key, self.cache_entry, result)

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
        self.contains_pos: bool = (
            model_config.mm_model_config.mm_position_ids_style != 0
        )
        self.mm_preprocess_batch_size: int = (
            model_config.mm_related_params.preprocess_batch_size
        )

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

        async_compute_workers = max(1, int(vit_config.vit_concurrency))
        async_queue_size = max(0, int(vit_config.vit_max_queue_size))
        self._async_compute_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=async_compute_workers,
            thread_name_prefix="mm-async-compute",
        )
        self._async_compute_workers = async_compute_workers
        self._async_queue_size = async_queue_size
        self._async_admission_capacity = async_compute_workers + async_queue_size
        self._async_admission_lock = threading.Lock()
        self._async_admitted = 0
        # Protect task ownership from the gap between cache claim and executor
        # submission. An RLock is required because Future.cancel() invokes done
        # callbacks synchronously in the cancelling thread.
        self._async_task_lock = threading.RLock()
        self._async_tasks: Dict[MMEmbeddingCacheEntry, _AsyncComputeTask] = {}
        self._async_request_tasks: Dict[int, Set[MMEmbeddingCacheEntry]] = {}
        logging.info(
            "MMProcessEngine: async compute executor max_workers=%d "
            "max_queue_size=%d total_capacity=%d",
            async_compute_workers,
            async_queue_size,
            self._async_admission_capacity,
        )

        # Embedding scheduler: always an MMScheduler; gpu-batch vs serial is
        # resolved by VitConfig (serial = one request per forward, no batching).
        scheduler_args = vit_config.embedding_scheduler_args()
        self._scheduler = MMScheduler(mm_part=mm_part, **scheduler_args)
        logging.info(
            f"MMProcessEngine: MMScheduler "
            f"(use_gpu_batch={vit_config.use_gpu_batch}, {scheduler_args})"
        )

        self.profiler = MMProfiler()

        self.query_num: int = 0
        self._access_logger = MMAccessLogger(
            get_log_path(),
            profiling_debug_logging_config.log_file_backup_count,
        )

        url_data_cache_.resize_cache(self.vit_config.url_cache_item_num)
        # Some non-visual multimodal mixins use this as a model-internal,
        # per-item lookup cache. Scheduler-level embedding results no longer use
        # it, but its existing operator-controlled capacity must be preserved.
        vit_emb_cache_.resize_cache(self.vit_config.mm_cache_item_num)

        cache_max_bytes = _derive_embedding_cache_max_bytes(
            self.mm_part,
            model_config,
            self.vit_config.mm_cache_item_num,
        )
        self._embedding_cache = MMEmbeddingCache(
            max_size=self.vit_config.mm_cache_item_num,
            max_bytes=cache_max_bytes,
            report_metrics=True,
        )
        # Keep the old private name as an alias for callers/tests that inspect
        # async submission state. Both paths now use the same cache instance.
        self._async_cache = self._embedding_cache
        logging.info(
            "MMProcessEngine: unified embedding cache max_items=%d max_bytes=%s",
            self.vit_config.mm_cache_item_num,
            cache_max_bytes if cache_max_bytes is not None else "count-fallback",
        )

        # GreenNet (content safety) integration. The provider is a no-op when
        # internal_source is absent or ENABLE_SAFETY_INSPECTION is off, so this
        # is zero-cost for open-source / disabled deployments. The dedicated
        # asyncio loop (lazily started on first real use) hosts the background
        # inspect tasks, which must outlive any single preprocess_and_submit
        # call — so we cannot use a transient asyncio.run() per worker thread.
        self._greennet_provider = get_greennet_provider()
        self._greennet_loop: Optional[asyncio.AbstractEventLoop] = None
        self._greennet_loop_thread: Optional[threading.Thread] = None
        self._greennet_loop_lock = threading.Lock()
        self._greennet_timeout_s = (
            float(getattr(self.vit_config, "mm_timeout_ms", 120000) or 120000) / 1000.0
        )

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
        object. Proxy workers still leave accounting to the proxy layer.
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
        if not self.is_proxy_mode:
            try:
                kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            except Exception:
                # Metrics must never mask the original ViT request failure.
                logging.exception("Failed to report ViT error QPS")

    # ------------------------------------------------------------------
    # GreenNet (content safety) plumbing
    # ------------------------------------------------------------------

    def _greennet_enabled(self) -> bool:
        # Effective enablement (internal source present AND runtime flag on).
        # When off, every greennet path is skipped so behavior is identical to
        # the pre-greennet engine — no asyncio loop, no rewrite, no inspect.
        return self._greennet_provider.is_enabled()

    def _ensure_greennet_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily start the dedicated asyncio loop that hosts greennet inspect
        tasks. The loop runs in a daemon thread so background uploads / POSTs
        survive across worker-thread calls."""
        if self._greennet_loop is not None:
            return self._greennet_loop
        with self._greennet_loop_lock:
            if self._greennet_loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, daemon=True, name="greennet-loop"
                )
                thread.start()
                self._greennet_loop = loop
                self._greennet_loop_thread = thread
        return self._greennet_loop

    def _begin_greennet(
        self,
        mm_inputs: List[MultimodalInput],
        entry: Optional[MMEmbeddingCacheEntry] = None,
        request_id: int = 0,
    ) -> Tuple[
        List[MultimodalInput], Optional["concurrent.futures.Future"], Optional[Any]
    ]:
        """Run greennet preprocess (download + frame extraction + URL rewrite),
        kick the async inspect task, and return:
          (rewritten_inputs, verdict_future, handle)

        Preprocess is the hard dependency for ViT (it rewrites mm_input.url to a
        base64 / frames-pack form the mixin consumes). Inspect runs concurrently
        on the greennet loop; ``verdict_future`` resolves to a GreenNetVerdict.

        If ``entry`` is given, a done-callback stamps the verdict onto it the
        moment inspection finishes — so ``WaitGreenNetVerdict`` unblocks without
        waiting for ViT. Caller is responsible for cancelling ``handle``.

        On a no-op provider, returns the inputs unchanged with no future/handle
        (and stamps a passing verdict on ``entry`` if provided)."""
        if not self._greennet_enabled():
            if entry is not None:
                entry.set_greennet_verdict(GreenNetVerdict(passed=True))
            return mm_inputs, None, None

        loop = self._ensure_greennet_loop()
        req = SimpleNamespace(
            id=str(request_id),
            model_name=getattr(self.vit_config, "model_name", "") or "",
        )
        handle = asyncio.run_coroutine_threadsafe(
            self._greennet_provider.preprocess_and_submit(req, mm_inputs), loop
        ).result(timeout=self._greennet_timeout_s)
        rewritten = list(handle.rewritten_inputs)

        verdict_future = asyncio.run_coroutine_threadsafe(handle.wait_result(), loop)
        if entry is not None:

            def _stamp(fut: "concurrent.futures.Future") -> None:
                try:
                    verdict = fut.result()
                except Exception as error:  # noqa: BLE001 - convert to process error
                    inspect_error = RuntimeError(f"greennet inspect failed: {error}")
                    self.report_vit_error(inspect_error, entry)
                    verdict = GreenNetVerdict(
                        passed=False, code=11, message=str(inspect_error)
                    )
                    try:
                        setattr(verdict, "_vit_error_qps_reported", True)
                    except Exception:
                        pass
                try:
                    entry.set_greennet_verdict(verdict)
                except Exception as error:
                    self.report_vit_error(error, entry)

            verdict_future.add_done_callback(_stamp)
        return rewritten, verdict_future, handle

    def _cancel_greennet(self, handle: Optional[Any]) -> None:
        if handle is None or self._greennet_loop is None:
            return
        try:
            self._greennet_loop.call_soon_threadsafe(handle.cancel)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"greennet handle cancel failed: {e}")

    def _shutdown_greennet_loop(self) -> None:
        loop = self._greennet_loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"greennet loop stop failed: {e}")
        if self._greennet_loop_thread is not None:
            self._greennet_loop_thread.join(timeout=2.0)
        self._greennet_loop = None
        self._greennet_loop_thread = None

    def _embed_with_greennet_sync(
        self, mm_inputs: List[MultimodalInput], request_id: int = 0
    ) -> MMEmbeddingRes:
        """Run one synchronous request and account for every failure boundary."""
        try:
            return self._embed_with_greennet_sync_impl(mm_inputs, request_id=request_id)
        except Exception as error:
            self.report_vit_error(error)
            raise

    def _embed_with_greennet_sync_impl(
        self, mm_inputs: List[MultimodalInput], request_id: int = 0
    ) -> MMEmbeddingRes:
        """Synchronous embedding path with greennet (used by the in-process /
        cpp / rpc entrypoints). Preprocess + inspect run, ViT runs concurrently
        with inspect, then the verdict gates the result. Raises
        FtRuntimeException(UNSAFE_INPUT_CONTENT) on a non-passing verdict."""
        if len(mm_inputs) != 1 or mm_inputs[0].url == "":
            rewritten, verdict_future, handle = self._begin_greennet(
                mm_inputs, request_id=request_id
            )
            work_items: List[MMWorkItem] = []
            try:
                result, work_items = self._mm_embedding_impl(
                    rewritten,
                    defer_cache_complete=True,
                    request_id=request_id,
                    report_vit_error=False,
                )
                verdict = GreenNetVerdict(passed=True)
                if verdict_future is not None:
                    verdict = verdict_future.result(timeout=self._greennet_timeout_s)
                    if verdict is None:
                        raise RuntimeError("sync GreenNet returned no verdict")
                    if not verdict.passed:
                        raise FtRuntimeException(
                            ExceptionType.UNSAFE_INPUT_CONTENT,
                            verdict.message or "data inspection failed",
                        )
                for work_item in work_items:
                    if work_item.cache_entry is not None:
                        work_item.cache_entry.set_greennet_verdict(verdict)
                    if work_item.embedding_result is not None:
                        work_item.complete_cache(work_item.embedding_result, force=True)
                return result
            except Exception as error:
                self.report_vit_error(error)
                for work_item in work_items:
                    work_item.fail_cache(error)
                raise
            finally:
                self._cancel_greennet(handle)

        cache_key = mm_inputs[0].cache_key()
        state, entry = self._embedding_cache.try_acquire(cache_key)
        handle = None
        try:
            if state == "miss":
                rewritten, verdict_future, handle = self._begin_greennet(
                    mm_inputs, entry, request_id
                )
            else:
                # The owner already ran GreenNet and owns any URL rewrite. A
                # waiter consumes the canonical raw embedding from the entry.
                rewritten, verdict_future = mm_inputs, None

            result, work_items = self._mm_embedding_impl(
                rewritten,
                cache_claim=(cache_key, entry, state),
                defer_cache_complete=state == "miss",
                request_id=request_id,
                report_vit_error=False,
            )

            if state == "miss":
                verdict = (
                    verdict_future.result(timeout=self._greennet_timeout_s)
                    if verdict_future is not None
                    else GreenNetVerdict(passed=True)
                )
                if verdict is None:
                    raise RuntimeError("sync GreenNet returned no verdict")
                if not verdict.passed:
                    raise FtRuntimeException(
                        ExceptionType.UNSAFE_INPUT_CONTENT,
                        verdict.message or "data inspection failed",
                    )
                raw_result = work_items[0].embedding_result
                if raw_result is None:
                    raise RuntimeError("sync embedding did not produce a cache value")
                self._embedding_cache.complete(cache_key, entry, raw_result)
            elif self._greennet_enabled():
                verdict = entry.wait_greennet(timeout=self._greennet_timeout_s)
                if verdict is None:
                    raise RuntimeError("cached GreenNet returned no verdict")
                if not verdict.passed:
                    raise FtRuntimeException(
                        ExceptionType.UNSAFE_INPUT_CONTENT,
                        verdict.message or "data inspection failed",
                    )
            return result
        except Exception as error:
            self.report_vit_error(error, entry if state == "miss" else None)
            if state == "miss":
                self._embedding_cache.fail(cache_key, entry, error)
            raise
        finally:
            self._cancel_greennet(handle)

    def wait_greennet_verdict(
        self,
        mm_inputs: List[MultimodalInput],
        timeout_ms: int = 60000,
        request_id: int = 0,
        cancellation_event: Optional[threading.Event] = None,
    ) -> GreenNetVerdict:
        """Block until every input's greennet verdict is decided; return the
        first non-passing verdict (first-failure-wins), else a passing verdict.

        Called by the VIT RPC's ``WaitGreenNetVerdict`` handler before prefill.
        If an input was never async-submitted (cache miss), kick its compute
        now so the verdict gets produced."""
        current_entry: Optional[MMEmbeddingCacheEntry] = None
        try:
            if not self._greennet_enabled():
                return GreenNetVerdict(passed=True)

            valid_inputs = [mm_input for mm_input in mm_inputs if mm_input.url != ""]
            claims = self._claim_and_submit_async(
                valid_inputs,
                request_id=request_id,
                queue_timeout_ms=timeout_ms,
                cancellation_event=cancellation_event,
            )
            deadline = time.monotonic() + timeout_ms / 1000.0
            for _, entry in claims:
                current_entry = entry
                remaining = max(0.0, deadline - time.monotonic())
                verdict = entry.wait_greennet(timeout=remaining)
                if verdict is None:
                    raise RuntimeError("GreenNet returned no verdict")
                if not verdict.passed:
                    # A rejected verdict is an exceptional RPC result even
                    # though this API returns it instead of raising.
                    self.report_vit_error(verdict, current_entry)
                    return verdict
            return GreenNetVerdict(passed=True)
        except Exception as error:
            self.report_vit_error(error, current_entry)
            raise

    def mm_embedding_rpc(self, mm_inputs: MultimodalInputsPB) -> MMEmbeddingRes:
        """Process multimodal inputs from RPC protocol buffer."""
        try:
            converted_inputs = trans_mm_input(mm_inputs)
        except Exception as error:
            self.report_vit_error(error)
            raise
        return self._embed_with_greennet_sync(
            converted_inputs, request_id=mm_inputs.request_id
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
        try:
            mm_inputs = [
                MultimodalInput(
                    url, MMUrlType(url_type), tensor, MMPreprocessConfig(*config)
                )
                for url, url_type, tensor, config in zip(
                    urls, types, tensors, mm_preprocess_configs
                )
            ]
        except Exception as error:
            self.report_vit_error(error)
            raise
        res = self._embed_with_greennet_sync(mm_inputs, request_id=request_id)
        try:
            res.position_ids = [pos.cpu() for pos in res.position_ids]
        except Exception as error:
            self.report_vit_error(error)
            raise
        return res

    def mm_embedding_impl(
        self, mm_inputs: List[MultimodalInput], request_id: int = 0
    ) -> MMEmbeddingRes:
        """Core implementation for multimodal embedding processing."""
        result, _ = self._mm_embedding_impl(mm_inputs, request_id=request_id)
        return result

    def _mm_embedding_impl(
        self,
        mm_inputs: List[MultimodalInput],
        cache_claim: Optional[Tuple[str, MMEmbeddingCacheEntry, str]] = None,
        defer_cache_complete: bool = False,
        request_id: int = 0,
        report_embedding_length: bool = True,
        report_vit_error: bool = True,
    ) -> Tuple[MMEmbeddingRes, List[MMWorkItem]]:
        """Internal implementation that also exposes canonical work-item values.

        Async submit owns the cache PENDING entry before GreenNet rewrites the
        URL. It passes that claim here and defers completion until the verdict
        passes, so sync and async callers share one value and unsafe results are
        never made visible.
        """
        logging.debug(f"{self.server_id} request [{request_id}] received")
        work_items: List[MMWorkItem] = []
        query_started = False
        try:
            self.mm_part.validate_inputs(mm_inputs)
            with self.profiler.profile_request():
                with torch.profiler.record_function("mm_embedding_impl"):
                    if not self.is_proxy_mode:
                        kmonitor.report(
                            AccMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"}
                        )

                    self.inc_query_num()
                    query_started = True
                    if not self.vit_config.disable_access_log:
                        self._access_logger.log_query_access(mm_inputs, request_id)

                    with torch.profiler.record_function("preprocess"):
                        work_items = self._create_work_items(
                            mm_inputs,
                            cache_claim=cache_claim,
                            defer_cache_complete=defer_cache_complete,
                        )
                        self._wait_for_preprocessing(work_items)

                    with torch.profiler.record_function("compute_embeddings"):
                        emb_res, pos_res, extra_input_res = self._compute_embeddings(
                            work_items
                        )
                        if report_embedding_length:
                            kmonitor.report(
                                GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC,
                                _embedding_token_length(emb_res),
                            )

                    with torch.profiler.record_function("postprocess"):
                        result = MMEmbeddingRes(emb_res, pos_res, extra_input_res)

                    if not self.vit_config.disable_access_log:
                        self._access_logger.log_success_access(
                            mm_inputs, str(result), request_id
                        )

                    if not self.is_proxy_mode:
                        kmonitor.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)

            return result, work_items
        except Exception as e:
            if report_vit_error:
                # Report before cleanup: cache cleanup/access logging must not
                # hide the fact that this request returned an error.
                self.report_vit_error(e)
            for work_item in work_items:
                work_item.fail_cache(e)
            torch.cuda.empty_cache()
            gc.collect()
            self._access_logger.log_exception_access(mm_inputs, e, request_id)
            raise
        finally:
            if query_started:
                self.dec_query_num()

    def _create_work_items(
        self,
        mm_inputs: List[MultimodalInput],
        cache_claim: Optional[Tuple[str, MMEmbeddingCacheEntry, str]] = None,
        defer_cache_complete: bool = False,
    ) -> List[MMWorkItem]:
        """Create work items and submit preprocessing tasks."""
        if cache_claim is not None and len(mm_inputs) != 1:
            raise ValueError("cache_claim is only valid for one multimodal input")
        batch_size = (
            self.mm_preprocess_batch_size
            if self.mm_preprocess_batch_size != -1
            else len(mm_inputs)
        )

        work_items = []
        for index in range(0, len(mm_inputs), batch_size):
            batch = mm_inputs[index : index + batch_size]
            work_item = MMWorkItem(
                batch,
                mm_timeout_ms=self.vit_config.mm_timeout_ms,
                embedding_cache=self._embedding_cache,
                cache_claim=cache_claim if index == 0 else None,
                defer_cache_complete=defer_cache_complete,
            )
            work_items.append(work_item)
            self.preprocess_executor.submit(work_item)

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
            emb_res.extend(maybe_tensor_to_list(result[0], ndim_threshold=2))
            pos_res.extend(maybe_tensor_to_list(result[1], ndim_threshold=2))
            if len(result) > 2:
                tensor_res.extend(maybe_tensor_to_list(result[2], ndim_threshold=1))
        return emb_res, pos_res, tensor_res

    @staticmethod
    def _work_item_result_to_response(result: Any) -> MMEmbeddingRes:
        emb_res = maybe_tensor_to_list(result[0], ndim_threshold=2)
        pos_res = maybe_tensor_to_list(result[1], ndim_threshold=2)
        extra_res = (
            maybe_tensor_to_list(result[2], ndim_threshold=1) if len(result) > 2 else []
        )
        return MMEmbeddingRes(emb_res, pos_res, extra_res)

    def async_submit(
        self, mm_inputs: List[MultimodalInput], request_id: int = 0
    ) -> List[str]:
        """Asynchronously submit multimodal URLs for embedding computation.

        Each input is submitted independently, keyed by its own cache_key.
        Returns the list of cache keys. Inputs already in-progress or complete
        are not recomputed.
        """
        try:
            self.mm_part.validate_inputs(mm_inputs)
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
    ) -> List[MMEmbeddingRes]:
        """Retrieve embedding results, blocking until ready if necessary.

        Each input is looked up independently by its cache_key.
        If a key was never submitted, queues it on the shared async executor.
        If in-progress, blocks until the computing thread finishes.
        If complete, returns immediately.
        """
        current_entry: Optional[MMEmbeddingCacheEntry] = None
        try:
            self.mm_part.validate_inputs(mm_inputs)
            claims = self._claim_and_submit_async(
                mm_inputs,
                request_id=request_id,
                queue_timeout_ms=timeout_ms,
                cancellation_event=cancellation_event,
            )
            deadline = time.monotonic() + timeout_ms / 1000.0
            results = []
            for _, entry in claims:
                current_entry = entry
                remaining = max(0.0, deadline - time.monotonic())
                raw_result = entry.wait(timeout=remaining)
                results.append(self._work_item_result_to_response(raw_result))

            kmonitor.report(
                GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC,
                sum(_embedding_token_length(result.embeddings) for result in results),
            )
            return results
        except Exception as error:
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
            if not entry.is_greennet_decided:
                verdict = GreenNetVerdict(passed=False, code=11, message=str(error))
                try:
                    if getattr(error, "_vit_error_qps_reported", False):
                        setattr(verdict, "_vit_error_qps_reported", True)
                except Exception:
                    pass
                entry.set_greennet_verdict(verdict)
        except Exception as verdict_error:
            # Keep the cache terminal transition even if the optional safety
            # verdict bookkeeping fails. The request error was already counted.
            self.report_vit_error(verdict_error, entry)
            logging.exception("Failed to publish ViT failure verdict")
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
        handle = None
        try:
            # GreenNet preprocess (rewrites URL) + async inspect. The inspect
            # done-callback stamps entry.greennet_verdict the moment inspection
            # finishes, so WaitGreenNetVerdict unblocks independently of ViT.
            rewritten, verdict_future, handle = self._begin_greennet(
                mm_inputs, entry, request_id
            )
            # ViT embedding runs concurrently with inspection.
            _, work_items = self._mm_embedding_impl(
                rewritten,
                cache_claim=(cache_key, entry, "miss"),
                defer_cache_complete=True,
                request_id=request_id,
                report_embedding_length=False,
                report_vit_error=False,
            )
            if verdict_future is not None:
                verdict = verdict_future.result(timeout=self._greennet_timeout_s)
                if verdict is None:
                    raise RuntimeError("async GreenNet returned no verdict")
                if not verdict.passed:
                    raise FtRuntimeException(
                        ExceptionType.UNSAFE_INPUT_CONTENT,
                        verdict.message or "data inspection failed",
                    )
            raw_result = work_items[0].embedding_result
            if raw_result is None:
                raise RuntimeError("async embedding did not produce a cache value")
            self._embedding_cache.complete(cache_key, entry, raw_result)
        except Exception as e:
            # If greennet never decided (preprocess crash etc.), surface a
            # process-error verdict so WaitGreenNetVerdict doesn't hang.
            self._fail_async_compute(cache_key, entry, e)
        finally:
            self._cancel_greennet(handle)

    def stop(self) -> None:
        """Shutdown the preprocessing executor and embedding scheduler."""
        self._async_compute_executor.shutdown(wait=False, cancel_futures=True)
        self.preprocess_executor.shutdown()
        self._scheduler.close()
        self._shutdown_greennet_loop()
        self._embedding_cache.clear(RuntimeError("MMProcessEngine stopped"))
