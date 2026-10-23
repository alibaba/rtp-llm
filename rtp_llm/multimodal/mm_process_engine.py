import concurrent.futures
import gc
import logging
import multiprocessing.pool
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.profiler
from rtp_llm.multimodal import mm_work_item as work_item_ops
from rtp_llm.multimodal.mm_async_result import continue_after
from rtp_llm.multimodal.mm_inspection import MMInspectionCoordinator, SourceInspection
from rtp_llm.multimodal.mm_preprocess import (
    LocalPreprocessExecutor,
    MultiprocessPreprocessExecutor,
    PreprocessExecutor,
)
from rtp_llm.multimodal.mm_work_item import (
    MMEmbeddingRes,
    MMWorkItem,
    _acquire_input_cache,
    _input_has_url,
    _feature_hashes_from_result,
)

from rtp_llm.access_logger.access_logger import MMAccessLogger
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.greennet_hook import GreenNetVerdict, get_greennet_provider
from rtp_llm.multimodal.mm_embedding_cache import (
    MMEmbeddingAsyncCache,
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
    MMHashKeyCache,
)
from rtp_llm.multimodal.mm_profiler import MMProfiler
from rtp_llm.multimodal.mm_scheduler import (
    MMScheduler,
    MMSchedulerError,
    MMSchedulerRequestTooLargeError,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MMWorkEstimate,
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import (
    maybe_tensor_to_list,
    url_data_cache_,
    vit_emb_cache_,
)
from rtp_llm.ops import (
    MMPreprocessConfig,
    MultimodalInput,
    VitSeparation,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


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


def _report_vit_metric(metric, value, tags=None) -> None:
    try:
        kmonitor.report(metric, value, tags or {})
    except Exception:
        logging.exception("Failed to report ViT metric")


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


class _AsyncComputeTask:
    def __init__(self, cache_key: str, entry: MMEmbeddingCacheEntry):
        self.cache_key = cache_key
        self.entry = entry
        self.request_ids: Set[int] = set()
        self.future: Optional[concurrent.futures.Future] = None


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
        kv_cache_config: Optional[Any] = None,
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
        # Resolve the provider before allocating pools or starting worker threads.
        inspection_provider = get_greennet_provider()
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

        self.mm_part = mm_part

        self._remote_embedding_cache = None
        if vit_config.mm_remote_cache_enable:
            from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient
            from rtp_llm.multimodal.mm_remote_embedding_cache import (
                MMRemoteEmbeddingCache,
            )

            if vit_config.mm_cache_cpu_max_bytes <= 0:
                raise ValueError(
                    "MM_REMOTE_CACHE_ENABLE requires a nonzero CPU embedding cache"
                )
            limits = (
                vit_config.mm_remote_cache_max_object_bytes,
                vit_config.mm_remote_cache_max_inflight_bytes,
                vit_config.mm_remote_cache_max_pending,
                vit_config.mm_remote_cache_read_timeout_ms,
            )
            if min(limits) <= 0:
                raise ValueError("remote embedding cache limits must be positive")
            client = (
                RtpKvMetaObjectClient.from_kv_cache_config(
                    kv_cache_config, max_object_bytes=limits[0]
                )
                if kv_cache_config is not None
                else RtpKvMetaObjectClient.from_env(max_object_bytes=limits[0])
            )
            self._remote_embedding_cache = MMRemoteEmbeddingCache(
                client,
                max_object_bytes=limits[0],
                max_inflight_bytes=limits[1],
                max_pending=limits[2],
                read_timeout_ms=limits[3],
                cuda_device=(
                    torch.device("cuda", torch.cuda.current_device())
                    if torch.cuda.is_available()
                    else None
                ),
            )
            logging.info(
                "ViT shared remote cache enabled: instance=%s group=%s limits=%s; "
                "local eviction never removes shared objects",
                client.instance_id,
                client.instance_group,
                limits,
            )

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

        async_limits = vit_config.async_compute_executor_args()
        async_compute_workers = async_limits["max_workers"]
        async_queue_size = async_limits["max_queue_size"]
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

        self.profiler = MMProfiler()

        try:
            scheduler_args = vit_config.embedding_scheduler_args()
            self._scheduler = MMScheduler(
                mm_part=mm_part,
                device=self.device,
                forward_profiler=self.profiler.profile_forward,
                # LOCAL shares a process with uncoordinated LLM CUDA launches.
                # Runtime graph capture is only safe in the dedicated ViT role.
                allow_idle_work=(
                    vit_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE
                ),
                # Fill the existing bounded lookahead whenever it is provisioned.
                prefetch_fill_batch=bool(scheduler_args.get("prefetch_batches", 0)),
                **scheduler_args,
            )
        except Exception:
            self._async_compute_executor.shutdown(wait=True, cancel_futures=True)
            self.preprocess_executor.shutdown()
            if self._remote_embedding_cache is not None:
                self._remote_embedding_cache.close()
            raise
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
            disable_access_log=vit_config.disable_access_log,
        )

        url_data_cache_.resize_cache(self.vit_config.url_cache_item_num)
        # Some non-visual multimodal mixins use this as a model-internal,
        # per-item lookup cache. Scheduler-level embedding results no longer use
        # it, but its existing operator-controlled capacity must be preserved.
        vit_emb_cache_.resize_cache(self.vit_config.mm_cache_item_num)

        # Both caches have explicit byte budgets, independent of model shape
        # estimates and the legacy model-internal item-count cache above.
        self._hash_key_cache = MMHashKeyCache(
            max_bytes=self.vit_config.mm_hash_key_cache_max_bytes
        )
        self._embedding_cache = MMEmbeddingCache(
            gpu_max_bytes=self.vit_config.mm_cache_gpu_max_bytes,
            cpu_max_bytes=self.vit_config.mm_cache_cpu_max_bytes,
            report_metrics=True,
            on_cpu_evict=(
                self._remote_embedding_cache.submit_eviction
                if self._remote_embedding_cache is not None
                else None
            ),
        )
        # Keep the old private name as an alias for callers/tests that inspect
        # async submission state. Both paths now use the same cache instance.
        self._async_cache = self._embedding_cache
        logging.info(
            "MMProcessEngine: embedding cache gpu_max_bytes=%d cpu_max_bytes=%d; "
            "hash-key cache max_bytes=%d",
            self.vit_config.mm_cache_gpu_max_bytes,
            self.vit_config.mm_cache_cpu_max_bytes,
            self.vit_config.mm_hash_key_cache_max_bytes,
        )

        self._inspection = MMInspectionCoordinator(
            self.mm_part,
            self.vit_config,
            self._hash_key_cache,
            inspection_provider,
            self.report_vit_error,
        )

    @property
    def _greennet_timeout_s(self):
        return self._inspection._greennet_timeout_s

    @_greennet_timeout_s.setter
    def _greennet_timeout_s(self, timeout_s):
        self._inspection._greennet_timeout_s = timeout_s

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

    # ------------------------------------------------------------------
    # GreenNet (content safety) plumbing
    # ------------------------------------------------------------------

    def _greennet_enabled(self):
        return self._inspection._greennet_enabled()

    def _request_scoped_inspection(self):
        return self._inspection._request_scoped_inspection()

    def _begin_greennet(self, *args, **kwargs):
        return self._inspection._begin_greennet(*args, **kwargs)

    def _cancel_greennet(self, handle):
        self._inspection._cancel_greennet(handle)

    def _inspect_cached_inputs(
        self,
        mm_inputs,
        *,
        deadline,
        request_id,
        user_id,
        service_name,
        cancellation_event=None,
        external_request_id: str = "",
    ) -> GreenNetVerdict:
        """Gate shared embeddings with this caller's request-scoped inspection."""
        if not mm_inputs:
            return GreenNetVerdict(passed=True)
        self._raise_if_async_request_cancelled(request_id, cancellation_event)
        _, future, handle = self._begin_greennet(
            mm_inputs,
            request_id=request_id,
            user_id=user_id,
            service_name=service_name,
            external_request_id=external_request_id,
            deadline=deadline,
        )
        try:
            while future is not None:
                self._raise_if_async_request_cancelled(request_id, cancellation_event)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ViT request deadline expired during inspection")
                try:
                    verdict = future.result(timeout=min(0.05, remaining))
                    if verdict is None:
                        raise RuntimeError("GreenNet returned no verdict")
                    self._raise_if_async_request_cancelled(
                        request_id, cancellation_event
                    )
                    return verdict
                except concurrent.futures.TimeoutError:
                    if future.done():
                        raise
            return GreenNetVerdict(passed=True)
        finally:
            self._cancel_greennet(handle)

    def _shutdown_greennet_loop(self):
        self._inspection.close()

    def _embed_with_greennet_sync(
        self,
        mm_inputs: List[MultimodalInput],
        request_id: int = 0,
        user_id: str = "",
        service_name: str = "",
        external_request_id: str = "",
    ) -> MMEmbeddingRes:
        """Run one synchronous request and account for every failure boundary."""
        if not self.is_proxy_mode:
            _report_vit_metric(AccMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"})
        try:
            self.mm_part.validate_inputs(mm_inputs)
            result = self._embed_with_greennet_sync_impl(
                mm_inputs,
                request_id=request_id,
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
            )
            _report_vit_metric(
                GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC,
                _embedding_token_length(result.embeddings),
            )
            if not self.is_proxy_mode:
                _report_vit_metric(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)
            return result
        except Exception as error:
            self.report_vit_error(error)
            raise

    def _embed_with_greennet_sync_impl(
        self,
        mm_inputs: List[MultimodalInput],
        request_id: int = 0,
        user_id: str = "",
        service_name: str = "",
        external_request_id: str = "",
    ) -> MMEmbeddingRes:
        """Synchronous embedding path with greennet (used by the in-process /
        cpp / rpc entrypoints). Preprocess + inspect run, ViT runs concurrently
        with inspect, then the verdict gates the result. Raises
        FtRuntimeException(UNSAFE_INPUT_CONTENT) on a non-passing verdict."""
        deadline = time.monotonic() + self._greennet_timeout_s
        if len(mm_inputs) != 1 or not _input_has_url(mm_inputs[0]):
            rewritten, verdict_future, handle = self._begin_greennet(
                mm_inputs,
                request_id=request_id,
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
                deadline=deadline,
                worker_sources=True,
            )
            work_items: List[MMWorkItem] = []
            try:
                result, work_items = self._mm_embedding_impl(
                    rewritten,
                    inspection_handle=handle,
                    deadline=deadline,
                    defer_cache_complete=True,
                    request_id=request_id,
                    report_vit_error=False,
                    report_request_metrics=False,
                    report_embedding_length=False,
                )
                verdict = GreenNetVerdict(passed=True)
                if verdict_future is not None:
                    verdict = verdict_future.result(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                    if verdict is None:
                        raise RuntimeError("sync GreenNet returned no verdict")
                    if not verdict.passed:
                        raise FtRuntimeException(
                            (
                                ExceptionType.UNSAFE_INPUT_CONTENT
                                if verdict.code == 2
                                else ExceptionType.MM_PROCESS_ERROR
                            ),
                            verdict.message or "data inspection failed",
                        )
                for work_item in work_items:
                    if work_item.cache_entry is not None:
                        work_item.cache_entry.set_greennet_verdict(
                            verdict, checked=self._greennet_enabled()
                        )
                    if (
                        work_item.cache_state == "miss"
                        and work_item.embedding_result is not None
                    ):
                        work_item.complete_cache(work_item.embedding_result, force=True)
                return result
            except Exception as error:
                self.report_vit_error(error)
                for work_item in work_items:
                    work_item.fail_cache(error)
                raise
            finally:
                self._cancel_greennet(handle)

        cache_key, state, entry = _acquire_input_cache(
            self._embedding_cache,
            mm_inputs[0],
            require_key=self._remote_embedding_cache is not None,
        )
        handle = None
        try:
            if state == "miss":
                rewritten, verdict_future, handle = self._begin_greennet(
                    mm_inputs,
                    entry,
                    request_id,
                    user_id=user_id,
                    service_name=service_name,
                    external_request_id=external_request_id,
                    deadline=deadline,
                    worker_sources=True,
                )
            elif self._request_scoped_inspection():
                # Reuse the tensor, never another caller's content verdict.
                rewritten, verdict_future, handle = self._begin_greennet(
                    mm_inputs,
                    request_id=request_id,
                    user_id=user_id,
                    service_name=service_name,
                    external_request_id=external_request_id,
                    deadline=deadline,
                    worker_sources=True,
                )
            else:
                # The owner already ran GreenNet and owns any URL rewrite. A
                # waiter consumes the canonical raw embedding from the entry.
                rewritten, verdict_future = mm_inputs, None

            result, work_items = self._mm_embedding_impl(
                rewritten,
                inspection_handle=handle,
                deadline=deadline,
                cache_claim=(cache_key, entry, state),
                defer_cache_complete=state == "miss",
                request_id=request_id,
                report_vit_error=False,
                report_request_metrics=False,
                report_embedding_length=False,
            )

            if state == "miss":
                verdict = (
                    verdict_future.result(timeout=max(0.0, deadline - time.monotonic()))
                    if verdict_future is not None
                    else GreenNetVerdict(passed=True)
                )
                if verdict is None:
                    raise RuntimeError("sync GreenNet returned no verdict")
                if not verdict.passed:
                    raise FtRuntimeException(
                        (
                            ExceptionType.UNSAFE_INPUT_CONTENT
                            if verdict.code == 2
                            else ExceptionType.MM_PROCESS_ERROR
                        ),
                        verdict.message or "data inspection failed",
                    )
                entry.set_greennet_verdict(verdict, checked=self._greennet_enabled())
                raw_result = work_items[0].embedding_result
                if raw_result is None:
                    raise RuntimeError("sync embedding did not produce a cache value")
                work_items[0].complete_cache(raw_result, force=True)
            elif self._greennet_enabled():
                verdict = (
                    verdict_future.result(timeout=max(0.0, deadline - time.monotonic()))
                    if verdict_future is not None
                    else entry.wait_greennet(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                )
                if verdict is None:
                    raise RuntimeError("cached GreenNet returned no verdict")
                if not verdict.passed:
                    raise FtRuntimeException(
                        (
                            ExceptionType.UNSAFE_INPUT_CONTENT
                            if verdict.code == 2
                            else ExceptionType.MM_PROCESS_ERROR
                        ),
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
        timeout_ms: int = VitConfig.DEFAULT_MM_TIMEOUT_MS,
        request_id: int = 0,
        cancellation_event: Optional[threading.Event] = None,
        user_id: str = "",
        service_name: str = "",
        external_request_id: str = "",
    ) -> GreenNetVerdict:
        """Block until every input's greennet verdict is decided; return the
        first non-passing verdict (first-failure-wins), else a passing verdict.

        Called by the VIT RPC's ``WaitGreenNetVerdict`` handler before prefill.
        If an input was never async-submitted (cache miss), kick its compute
        now so the verdict gets produced."""
        claims: List[Tuple[str, MMEmbeddingCacheEntry]] = []
        cached_inspection_inputs: List[MultimodalInput] = []
        current_entry: Optional[MMEmbeddingCacheEntry] = None
        deadline = time.monotonic() + timeout_ms / 1000.0
        try:
            if not self._greennet_enabled():
                return GreenNetVerdict(passed=True)

            valid_inputs = [
                mm_input
                for mm_input in mm_inputs
                if _input_has_url(mm_input)
                and (
                    self._request_scoped_inspection()
                    or not self._hash_key_cache.greennet_passed(mm_input.cache_key())
                )
            ]
            self._raise_if_async_request_cancelled(request_id, cancellation_event)
            if time.monotonic() >= deadline:
                raise TimeoutError("ViT request deadline expired before submission")
            claims = self._claim_and_submit_async(
                valid_inputs,
                request_id=request_id,
                queue_timeout_ms=max(1, int((deadline - time.monotonic()) * 1000)),
                cancellation_event=cancellation_event,
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
                cached_inspection_inputs=(
                    cached_inspection_inputs
                    if self._request_scoped_inspection()
                    else None
                ),
            )
            verdict = self._inspect_cached_inputs(
                cached_inspection_inputs,
                deadline=deadline,
                request_id=request_id,
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
                cancellation_event=cancellation_event,
            )
            if not verdict.passed:
                self.report_vit_error(verdict)
                return verdict
            for _, entry in claims:
                current_entry = entry
                remaining = max(0.0, deadline - time.monotonic())
                verdict = entry.wait_greennet(
                    timeout=remaining, cancellation_event=cancellation_event
                )
                if verdict is None:
                    raise RuntimeError("GreenNet returned no verdict")
                if not verdict.passed:
                    # A rejected verdict is an exceptional RPC result even
                    # though this API returns it instead of raising.
                    self.report_vit_error(verdict, current_entry)
                    return verdict
            return GreenNetVerdict(passed=True)
        except Exception as error:
            self.cancel_queued_request(
                request_id, entries=[entry for _, entry in claims]
            )
            if isinstance(error, concurrent.futures.CancelledError):
                error = FtRuntimeException(
                    ExceptionType.CANCELLED_ERROR,
                    f"ViT request {request_id} was cancelled",
                )
            self.report_vit_error(error, current_entry)
            raise error

    def mm_embedding_cpp(
        self,
        urls: List[str],
        types: List[int],
        tensors: List[torch.Tensor],
        mm_preprocess_configs: List[Any],
        request_id: int = 0,
        user_id: str = "",
        service_name: str = "",
        external_request_id: str = "",
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
        res = self._embed_with_greennet_sync(
            mm_inputs,
            request_id=request_id,
            user_id=user_id,
            service_name=service_name,
            external_request_id=external_request_id,
        )
        try:
            with cuda_graph_gate.operation():
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
        report_image_count: bool = True,
        report_vit_error: bool = True,
        report_request_metrics: bool = True,
        defer_feature_hashes: bool = False,
        deadline: Optional[float] = None,
        inspection_handle: Optional[SourceInspection] = None,
    ) -> Tuple[MMEmbeddingRes, List[MMWorkItem]]:
        """Internal implementation that also exposes canonical work-item values.

        Async submit owns the cache PENDING entry before GreenNet rewrites the
        URL. It passes that claim here and defers completion until the verdict
        passes, so sync and async callers share one value and unsafe results are
        never made visible.
        """
        logging.debug(f"{self.server_id} request [{request_id}] received")
        work_items: List[MMWorkItem] = []
        scheduled = []
        query_started = False
        try:
            self.mm_part.validate_inputs(mm_inputs)
            if report_image_count:
                _report_image_count(mm_inputs)
            with torch.profiler.record_function("mm_embedding_impl"):
                if report_request_metrics and not self.is_proxy_mode:
                    _report_vit_metric(
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
                        defer_feature_hashes=defer_feature_hashes,
                        deadline=deadline,
                        capture_inspection_sources=isinstance(
                            inspection_handle, SourceInspection
                        ),
                    )
                    self._wait_for_preprocessing(work_items, scheduled, deadline)
                    if isinstance(inspection_handle, SourceInspection):
                        inspection_handle.supply_work_items(work_items)

                with torch.profiler.record_function("compute_embeddings"):
                    emb_res, pos_res, extra_input_res = self._compute_embeddings(
                        work_items, deadline=deadline, scheduled=scheduled
                    )
                    if report_embedding_length:
                        _report_vit_metric(
                            GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC,
                            _embedding_token_length(emb_res),
                        )

                with torch.profiler.record_function("postprocess"):
                    hashes = (
                        None
                        if defer_feature_hashes
                        else [h for wi in work_items for h in wi.feature_hashes or []]
                    )
                    result = MMEmbeddingRes(emb_res, pos_res, extra_input_res, hashes)

                if not self.vit_config.disable_access_log:
                    self._access_logger.log_success_access(
                        mm_inputs, str(result), request_id
                    )

            if report_request_metrics and not self.is_proxy_mode:
                _report_vit_metric(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)

            return result, work_items
        except Exception as e:
            for receipt in scheduled:
                self._scheduler.cancel(receipt)
            if report_vit_error:
                self.report_vit_error(e)
            for work_item in work_items:
                work_item.fail_cache(e)
            if not isinstance(e, MMSchedulerError) and (
                isinstance(e, torch.cuda.OutOfMemoryError)
                or isinstance(e.__cause__, torch.cuda.OutOfMemoryError)
            ):
                gc.collect()
                torch.cuda.empty_cache()
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
        defer_feature_hashes: bool = False,
        deadline: Optional[float] = None,
        capture_inspection_sources: bool = False,
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

        if cache_claim is not None and len(mm_inputs) != 1:
            raise ValueError("cache_claim is only valid for one multimodal input")
        batch_size = (
            self.mm_preprocess_batch_size
            if self.mm_preprocess_batch_size != -1
            else len(mm_inputs)
        )

        work_items = []
        ready_event = threading.Event()
        try:
            for index in range(0, len(mm_inputs), batch_size):
                batch = mm_inputs[index : index + batch_size]
                work_item = MMWorkItem(
                    batch,
                    mm_timeout_ms=self.vit_config.mm_timeout_ms,
                    embedding_cache=self._embedding_cache,
                    hash_key_cache=self._hash_key_cache,
                    cache_claim=cache_claim if index == 0 else None,
                    defer_cache_complete=defer_cache_complete,
                    defer_feature_hashes=defer_feature_hashes,
                    deadline=deadline,
                    require_cache_key=self._remote_embedding_cache is not None,
                )
                work_items.append(work_item)
                work_item.capture_inspection_sources = capture_inspection_sources
                work_item.preprocess_ready_event = ready_event
                remote_cache = self._remote_embedding_cache
                if (
                    remote_cache is not None
                    # Remote tensors do not carry matching source bytes/verdicts.
                    and not capture_inspection_sources
                    and work_item.cache_state == "miss"
                    and work_item.should_preprocess
                ):
                    work_item.embedding_result = remote_cache.load(
                        work_item.cache_key,
                        timeout_ms=max(1, int(work_item.remaining_timeout_s() * 1000)),
                    )
                work_item.remaining_timeout_s()
                self.preprocess_executor.submit(work_item)

        except Exception as error:
            # The caller has not received this list yet and cannot roll it back.
            for work_item in work_items:
                work_item.fail_cache(error)
            raise

        return work_items

    def _wait_for_preprocessing(self, work_items, scheduled=None, deadline=None):
        """Feed independent completed CPU work to the existing GPU owner."""
        for work_item in self.preprocess_executor.iter_ready(work_items):
            if work_item.waiting_for_cache and work_item.future is None:
                # Resolve followers in final assembly, after local producers
                # have been submitted. Never block ready CPU work on a cache.
                continue
            work_item.remaining_timeout_s()
            self.preprocess_executor.get_result(work_item)
            work_item.remaining_timeout_s()
            if work_item.embedding_result is None and not work_item.waiting_for_cache:
                estimate = self.mm_part.estimate_work(
                    work_item.preprocess_result, work_item.mm_type
                )
                if estimate is not None and not isinstance(estimate, MMWorkEstimate):
                    raise TypeError("estimate_work must return MMWorkEstimate or None")
                work_item.work_estimate = estimate
                if scheduled is not None:
                    scheduled.append(
                        self._scheduler.submit([work_item], deadline=deadline)
                    )

    def _compute_embeddings(
        self,
        work_items: List[MMWorkItem],
        deadline: Optional[float] = None,
        scheduled=None,
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Compute embeddings for all work items."""
        pending_items = [
            wi
            for wi in work_items
            if wi.embedding_result is None and not wi.waiting_for_cache
        ]

        if scheduled is not None:
            for receipt in scheduled:
                self._scheduler.wait(receipt)
        elif pending_items:
            self._scheduler.submit_and_wait(pending_items, deadline=deadline)

        emb_res, pos_res, tensor_res = [], [], []
        local_producers = {
            wi.cache_entry: wi
            for wi in work_items
            if wi.cache_state == "miss" and wi.cache_entry is not None
        }
        for wi in work_items:
            if wi.waiting_for_cache:
                producer = local_producers.get(wi.cache_entry)
                if producer is not None:
                    # The producer's cache publication may be deferred until
                    # GreenNet passes. Reuse its private result in this request.
                    wi.embedding_result = producer.embedding_result
                    wi.feature_hashes = producer.feature_hashes
                else:
                    wi.embedding_result = wi.cache_entry.wait(
                        timeout=wi.remaining_timeout_s()
                    )
            result = wi.embedding_result
            # Scheduler invariant: submit_and_wait either fills embedding_result
            # for every pending item or raises, so it is never None here.
            if result is None:
                raise RuntimeError(f"embedding_result not set for work item {wi}")
            if wi.feature_hashes is None and wi.cache_entry is not None:
                wi.feature_hashes = self._hash_key_cache.get(
                    wi.cache_key, wi.cache_entry.generation
                )
            if wi.feature_hashes is None and not wi.defer_feature_hashes:
                wi.complete_cache(result)
            emb_res.extend(maybe_tensor_to_list(result[0], ndim_threshold=2))
            pos_res.extend(maybe_tensor_to_list(result[1], ndim_threshold=2))
            if len(result) > 2:
                tensor_res.extend(maybe_tensor_to_list(result[2], ndim_threshold=1))
            # Forward completion owns the H2D/consumer lifetime. The request
            # awaiting its verdict only needs the resulting embedding now.
            wi.preprocess_result = None
        return emb_res, pos_res, tensor_res

    @staticmethod
    def _work_item_result_to_response(
        result: Any, feature_hashes: Optional[List[torch.Tensor]] = None
    ) -> MMEmbeddingRes:
        emb_res = maybe_tensor_to_list(result[0], ndim_threshold=2)
        pos_res = maybe_tensor_to_list(result[1], ndim_threshold=2)
        extra_res = (
            maybe_tensor_to_list(result[2], ndim_threshold=1) if len(result) > 2 else []
        )
        return MMEmbeddingRes(emb_res, pos_res, extra_res, feature_hashes)

    def async_submit(
        self,
        mm_inputs: List[MultimodalInput],
        request_id: int = 0,
        user_id: str = "",
        service_name: str = "",
        external_request_id: str = "",
    ) -> List[str]:
        """Asynchronously submit multimodal URLs for embedding computation.

        Each input is submitted independently, keyed by its own cache_key.
        Returns the list of cache keys. Inputs already in-progress or complete
        are not recomputed.
        """
        try:
            self.mm_part.validate_inputs(mm_inputs)
            claims = self._claim_and_submit_async(
                mm_inputs,
                request_id=request_id,
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
            )
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
        user_id: str = "",
        service_name: str = "",
        external_request_id: str = "",
    ) -> List[MMEmbeddingRes]:
        """Retrieve embedding results, blocking until ready if necessary.

        Each input is looked up independently by its cache_key.
        If a key was never submitted, queues it on the shared async executor.
        If in-progress, blocks until the computing thread finishes.
        If complete, returns immediately. With hashes_only, return sidecar hashes
        without exporting embeddings or promoting a CPU entry with cached hashes.
        """
        claims: List[Tuple[str, MMEmbeddingCacheEntry]] = []
        cached_inspection_inputs: List[MultimodalInput] = []
        current_entry: Optional[MMEmbeddingCacheEntry] = None
        deadline = time.monotonic() + timeout_ms / 1000.0
        try:
            self.mm_part.validate_inputs(mm_inputs)
            _report_image_count(mm_inputs)
            self._raise_if_async_request_cancelled(request_id, cancellation_event)
            if time.monotonic() >= deadline:
                raise TimeoutError("ViT request deadline expired before submission")
            claims = self._claim_and_submit_async(
                mm_inputs,
                request_id=request_id,
                queue_timeout_ms=max(1, int((deadline - time.monotonic()) * 1000)),
                cancellation_event=cancellation_event,
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
                cached_inspection_inputs=(
                    cached_inspection_inputs
                    if self._request_scoped_inspection()
                    else None
                ),
            )
            verdict = self._inspect_cached_inputs(
                cached_inspection_inputs,
                deadline=deadline,
                request_id=request_id,
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
                cancellation_event=cancellation_event,
            )
            if not verdict.passed:
                raise FtRuntimeException(
                    (
                        ExceptionType.UNSAFE_INPUT_CONTENT
                        if verdict.code == 2
                        else ExceptionType.MM_PROCESS_ERROR
                    ),
                    verdict.message or "data inspection failed",
                )
            results = []
            for cache_key, entry in claims:
                current_entry = entry
                remaining = max(0.0, deadline - time.monotonic())
                entry.wait_ready(
                    timeout=remaining, cancellation_event=cancellation_event
                )
                if self._greennet_enabled() and not entry.greennet_passed:
                    raise FtRuntimeException(
                        ExceptionType.MM_PROCESS_ERROR,
                        "ViT embedding has no completed GreenNet inspection",
                    )
                self._raise_if_async_request_cancelled(request_id, cancellation_event)
                feature_hashes = self._hash_key_cache.get(cache_key, entry.generation)
                raw_result = None
                if not hashes_only or feature_hashes is None:
                    raw_result = entry.wait(
                        timeout=max(0.0, deadline - time.monotonic()),
                        # The hash op synchronizes its stream before returning
                        # CPU hashes. Queue producer dependencies on that stream
                        # instead of blocking this CPU thread before the hash.
                        wait_on_current_stream=feature_hashes is None,
                        cancellation_event=cancellation_event,
                    )
                if feature_hashes is None:
                    feature_hashes = work_item_ops._feature_hashes_from_result(
                        raw_result
                    )
                    # Hashing waits for primary embeddings. Also finish any
                    # opaque outputs produced on other devices before export.
                    hashed_devices = {
                        embedding.device
                        for embedding in maybe_tensor_to_list(
                            raw_result[0], ndim_threshold=2
                        )
                        if isinstance(embedding, torch.Tensor)
                        and embedding.is_cuda
                        and embedding.ndim > 0
                        and embedding.numel() > 0
                    }
                    for producer in entry.producer_streams:
                        if producer.device not in hashed_devices:
                            with cuda_graph_gate.operation():
                                producer.synchronize()
                    self._hash_key_cache.put(
                        cache_key,
                        feature_hashes,
                        entry.generation,
                        greennet_passed=entry.greennet_passed,
                    )
                if hashes_only:
                    results.append(MMEmbeddingRes([], feature_hashes=feature_hashes))
                else:
                    results.append(
                        self._work_item_result_to_response(raw_result, feature_hashes)
                    )
                del raw_result

            _report_vit_metric(
                GaugeMetrics.VIT_EMBEDDING_LENGTH_METRIC,
                sum(
                    (
                        sum(h.numel() for h in result.feature_hashes)
                        if hashes_only
                        else _embedding_token_length(result.embeddings)
                    )
                    for result in results
                ),
            )
            return results
        except Exception as error:
            self.cancel_queued_request(
                request_id, entries=[entry for _, entry in claims]
            )
            if isinstance(error, concurrent.futures.CancelledError):
                error = FtRuntimeException(
                    ExceptionType.CANCELLED_ERROR,
                    f"ViT request {request_id} was cancelled",
                )
            self.report_vit_error(error, current_entry)
            raise error

    def _claim_and_submit_async(
        self,
        mm_inputs: List[MultimodalInput],
        request_id: int = 0,
        queue_timeout_ms: Optional[int] = None,
        cancellation_event: Optional[threading.Event] = None,
        user_id: str = "",
        service_name: str = "",
        cached_inspection_inputs: Optional[List[MultimodalInput]] = None,
        external_request_id: str = "",
    ) -> List[Tuple[str, MMEmbeddingCacheEntry]]:
        self._raise_if_async_request_cancelled(request_id, cancellation_event)
        # Validate every input before creating any shared PENDING entries.
        inputs = []
        for mm_input in mm_inputs:
            if not _input_has_url(mm_input):
                raise ValueError(
                    "async embedding requires non-empty url for each input"
                )
            inputs.append((mm_input, mm_input.cache_key()))

        claims: List[Tuple[str, MMEmbeddingCacheEntry]] = []
        pending: List[Tuple[MultimodalInput, str, MMEmbeddingCacheEntry]] = []
        try:
            for mm_input, cache_key in inputs:
                with self._async_task_lock:
                    if self._stopped:
                        raise RuntimeError("MMProcessEngine stopped")
                    state, entry = self._async_cache.try_acquire(cache_key)
                    claims.append((cache_key, entry))
                    if state != "miss" and cached_inspection_inputs is not None:
                        cached_inspection_inputs.append(mm_input)
                    if state == "miss":
                        self._async_tasks[entry] = _AsyncComputeTask(cache_key, entry)
                        pending.append((mm_input, cache_key, entry))
                    if (
                        state == "complete"
                        and self._embedding_cache.peek(cache_key) is entry
                    ):
                        self._hash_key_cache.get(cache_key, entry.generation)
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
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
            )
            self._raise_if_async_request_cancelled(request_id, cancellation_event)
            return claims
        except Exception as error:
            # Only this invocation can submit these new claims. Even if another
            # request joined one, an unsubmitted claim has no producer: fail it
            # and wake all waiters instead of leaving shared PENDING state.
            with self._async_task_lock:
                for _, cache_key, entry in pending:
                    task = self._async_tasks.get(entry)
                    if task is not None and task.future is None:
                        self._forget_async_task_locked(entry)
                        self._fail_async_compute(cache_key, entry, error)
            self.cancel_queued_request(
                request_id, entries=[entry for _, entry in claims]
            )
            raise

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

    def cancel_queued_request(
        self, request_id: int, *, entries: Optional[List[MMEmbeddingCacheEntry]] = None
    ) -> int:
        """Cancel work that is still queued and exclusively owned by a request.

        Running futures are deliberately left alone. A deduplicated task remains
        queued while any other request still owns it.
        """
        cancelled = 0
        with self._async_task_lock:
            if entries is None:
                entries = list(self._async_request_tasks.pop(request_id, set()))
            else:
                # A failed submission must not cancel earlier work using the
                # same request id (including callers that use the default 0).
                owned = self._async_request_tasks.get(request_id, set())
                entries = list(owned.intersection(entries))
                owned.difference_update(entries)
                if not owned:
                    self._async_request_tasks.pop(request_id, None)
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
        # CPU/GPU work may have handed off to a verdict continuation. Keep its
        # ownership and admission slot until publication, without keeping an
        # async-compute thread blocked on network inspection.
        try:
            continuation = None if future.cancelled() else future.result()
        except Exception:
            continuation = None  # normal error accounting below
        if isinstance(continuation, concurrent.futures.Future):
            with self._async_task_lock:
                task = self._async_tasks.get(entry)
                if task is not None:
                    task.future = continuation
                continuation.add_done_callback(
                    lambda completed: self._on_async_compute_done(
                        cache_key, entry, completed
                    )
                )
            return
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
        user_id: str = "",
        service_name: str = "",
        external_request_id: str = "",
    ) -> Optional[concurrent.futures.Future]:
        if time.monotonic() >= deadline:
            error = FtRuntimeException(
                ExceptionType.GENERATE_TIMEOUT,
                "ViT queue wait timed out before execution",
            )
            self._fail_async_compute(cache_key, entry, error)
            return
        return self._async_compute(
            mm_inputs,
            cache_key,
            entry,
            request_id,
            deadline=deadline,
            user_id=user_id,
            service_name=service_name,
            external_request_id=external_request_id,
        )

    def _submit_async_compute_batch(
        self,
        pending: List[Tuple[MultimodalInput, str, MMEmbeddingCacheEntry]],
        request_id: int = 0,
        queue_timeout_ms: Optional[int] = None,
        user_id: str = "",
        service_name: str = "",
        external_request_id: str = "",
    ) -> None:
        if not pending:
            return

        # Serialize submission with cancellation so a request cannot create new
        # queued work after its cancellation callback has already run.
        with self._async_task_lock:
            if self._stopped:
                raise RuntimeError("MMProcessEngine stopped")
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
                        user_id,
                        service_name,
                        external_request_id,
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
        user_id: str = "",
        service_name: str = "",
        *,
        deadline: float,
        external_request_id: str = "",
    ) -> Optional[concurrent.futures.Future]:
        handle = None
        if not self.is_proxy_mode:
            _report_vit_metric(AccMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"})
        try:
            # Worker-source adapters preserve inference URLs; legacy adapters
            # retain their rewrite contract. The verdict callback stamps the
            # entry independently of GPU completion.
            rewritten, verdict_future, handle = self._begin_greennet(
                mm_inputs,
                entry,
                request_id,
                user_id=user_id,
                service_name=service_name,
                external_request_id=external_request_id,
                deadline=deadline,
                worker_sources=True,
            )
            # ViT embedding runs concurrently with inspection.
            _, work_items = self._mm_embedding_impl(
                rewritten,
                inspection_handle=handle,
                cache_claim=(cache_key, entry, "miss"),
                defer_cache_complete=True,
                request_id=request_id,
                report_embedding_length=False,
                report_image_count=False,
                report_vit_error=False,
                report_request_metrics=False,
                deadline=deadline,
                # No routing metadata is retained when both caches are off.
                # get_embedding_result orders the hash after its producer and
                # computes the response hashes once, outside compute admission.
                defer_feature_hashes=(
                    not self._embedding_cache.enabled
                    and not self._hash_key_cache.enabled
                    # A cache entry claimed before a resize keeps its original
                    # completion/transfer callbacks and must use their path.
                    and not entry.has_read_callback
                ),
            )

            def publish(verdict):
                if verdict is None:
                    raise RuntimeError("async GreenNet returned no verdict")
                if not verdict.passed:
                    raise FtRuntimeException(
                        (
                            ExceptionType.UNSAFE_INPUT_CONTENT
                            if verdict.code == 2
                            else ExceptionType.MM_PROCESS_ERROR
                        ),
                        verdict.message or "data inspection failed",
                    )
                entry.set_greennet_verdict(verdict, checked=self._greennet_enabled())
                raw_result = work_items[0].embedding_result
                if raw_result is None:
                    raise RuntimeError("async embedding did not produce a cache value")
                work_items[0].complete_cache(raw_result, force=True)
                if not self.is_proxy_mode:
                    _report_vit_metric(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)

            if verdict_future is not None:
                owned_handle = handle
                completion = continue_after(
                    verdict_future,
                    self._async_compute_executor,
                    publish,
                    lambda: self._cancel_greennet(owned_handle),
                    lambda error: self._fail_async_compute(cache_key, entry, error),
                )
                handle = None  # continuation owns cancellation and publication
                return completion
            publish(GreenNetVerdict(passed=True))
        except Exception as e:
            # If greennet never decided (preprocess crash etc.), surface a
            # process-error verdict so WaitGreenNetVerdict doesn't hang.
            self._fail_async_compute(cache_key, entry, e)
        finally:
            self._cancel_greennet(handle)

    def stop(self) -> None:
        """Shutdown asynchronous work, the scheduler and preprocessing workers."""
        with self._async_task_lock:
            if self._stopped:
                return
            self._stopped = True
        self._async_compute_executor.shutdown(wait=True, cancel_futures=True)
        self.preprocess_executor.shutdown()
        self._scheduler.close()
        self._shutdown_greennet_loop()
        self._embedding_cache.clear(RuntimeError("MMProcessEngine stopped"))
        self._hash_key_cache.clear()
        if self._remote_embedding_cache is not None:
            self._remote_embedding_cache.close()
