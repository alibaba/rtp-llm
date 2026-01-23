import concurrent.futures
import gc
import logging
import multiprocessing.pool
import os
import signal
import time
from multiprocessing import Lock
from typing import Any, Callable, List, Optional, Tuple

import torch

from rtp_llm.access_logger.access_logger import MMAccessLogger
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import (
    trans_mm_input,
    url_data_cache_,
    vit_emb_cache_,
)
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)
from rtp_llm.utils.time_util import Timer, timer_wrapper

mm_embedding_lock = Lock()
pool_lock = Lock()
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
    logging.info(f"Worker process {os.getpid()} initialized.")


def _worker_process_task(
    mm_inputs: List[MultimodalInput],
) -> Tuple[Any, float]:
    """
    只接收变化的 `mm_inputs` 参数。
    """
    if _worker_preprocess_func is None:
        raise RuntimeError("Worker process has not been initialized correctly.")

    with Timer() as route_timer:
        # 3. 使用来自全局变量的不变参数
        result = _worker_preprocess_func(
            mm_inputs, _worker_vit_config, **_worker_preprocess_params
        )
    return result, route_timer.cost_ms()


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
        if work_item.embedding_result is not None:
            return

        try:
            with Timer() as route_timer:
                result = self.preprocess_func(
                    work_item.mm_inputs, self.vit_config, **self.preprocess_params
                )
            preprocess_time = route_timer.cost_ms()
            work_item.preprocess_result = result
            # 使用简单的对象模拟 future 行为
            work_item.future = _LocalResult(result, preprocess_time)
        except Exception as e:
            logging.error(f"Error in local preprocessing: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            _, preprocess_time = work_item.future.get()
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
        except Exception as e:
            logging.error(f"Error getting local preprocess result: {e}", exc_info=True)
            raise


class MultiprocessPreprocessExecutor(PreprocessExecutor):
    """多进程预处理执行器"""

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

    def submit(self, work_item: "MMWorkItem") -> None:
        if work_item.embedding_result is not None:
            return

        max_retries = 2
        for attempt in range(max_retries):
            try:
                work_item.future = self.pool.apply_async(
                    _worker_process_task, args=(work_item.mm_inputs,)
                )
                return
            except (BrokenPipeError, EOFError, OSError) as e:
                logging.warning(
                    f"Broken pool detected on submit (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    self._recover_pool()
                else:
                    logging.error(
                        f"Failed to recover from broken pool after {max_retries} attempts"
                    )
                    raise RuntimeError(
                        "Preprocessing pool is permanently broken."
                    ) from e
            except Exception as e:
                logging.error(f"Unexpected error during submission: {e}", exc_info=True)
                raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            work_item.preprocess_result, preprocess_time = work_item.future.get(
                timeout=work_item.mm_timeout_ms / 1000.0
            )
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
        except multiprocessing.pool.TimeoutError:
            raise TimeoutError(
                f"Preprocessing timeout after {work_item.mm_timeout_ms}ms"
            )
        except (BrokenPipeError, EOFError, OSError) as e:
            logging.error(f"Broken pool detected while waiting for result: {e}")
            self._recover_pool()
            raise RuntimeError(
                "Preprocessing failed due to a broken worker process."
            ) from e
        except Exception as e:
            logging.error(f"Error getting preprocess result: {e}", exc_info=True)
            raise

    def _recover_pool(self) -> None:
        old_pool = self.pool
        if old_pool is None:
            return

        with pool_lock:
            if self.pool is not old_pool:
                logging.debug("Pool already recovered by another thread")
                return

            kmonitor.report(AccMetrics.VIT_PROCESS_POOL_RESTART_QPS_METRIC, 1)
            child_pids = self._get_child_pids_from_pool(old_pool)

            logging.warning(
                f"Broken process pool detected. Terminating pool with PIDs: {child_pids}"
            )

            try:
                old_pool.terminate()
                old_pool.join()
            except Exception as e:
                logging.warning(f"Error during pool termination: {e}", exc_info=True)

            try:
                self._create_pool()
                logging.info("Recreated ProcessPool after it was broken.")
            except Exception as e:
                logging.error(f"Failed to create new ProcessPool: {e}", exc_info=True)
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
        self.pool.close()
        self.pool.join()
        logging.info("Preprocessing pool shut down.")


class _LocalResult:
    """本地预处理结果的简单包装类"""

    def __init__(self, result: Any, time: float):
        self.result = result
        self.time = time

    def get(self, timeout: Optional[float] = None) -> Tuple[Any, float]:
        return (self.result, self.time)


class MMEmbeddingRes:
    """Result container for multimodal embedding operations."""

    def __init__(
        self,
        embeddings: List[torch.Tensor],
        position_ids: Optional[List[torch.Tensor]] = None,
        deepstack_embeds: Optional[List[torch.Tensor]] = None,
    ):
        self.embeddings = embeddings
        self.position_ids = position_ids
        self.deepstack_embeds = deepstack_embeds

    def __str__(self) -> str:
        return f"MMEmbeddingRes(length={len(self.embeddings)})"


class MMWorkItem:
    """Represents a work item for processing multimodal inputs."""

    def __init__(
        self, mm_inputs: List[MultimodalInput], mm_timeout_ms: Optional[int] = 120000
    ):
        if not mm_inputs:
            raise ValueError("No mm_input for work item")

        self.mm_inputs = mm_inputs
        self.mm_timeout_ms = (
            self.mm_inputs[0].config.mm_timeout_ms
            if self.mm_inputs[0].config.mm_timeout_ms != -1
            else mm_timeout_ms
        )
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result: Optional[Any] = None
        self.embedding_result: Optional[Any] = None

        self.need_check_cache = len(mm_inputs) == 1 and mm_inputs[0].url is not None
        self.cache_key = (
            self.mm_inputs[0].to_string() if self.need_check_cache else None
        )
        self.embedding_result = vit_emb_cache_.check_cache(self.cache_key)

        # future 可以是 ApplyResult (multiprocess) 或 _LocalResult (local)
        self.future: Optional[Any] = None


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

        self.mp_context = multiprocessing.get_context("spawn")

        self.mm_part = mm_part

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

        self.query_num: int = 0
        self._access_logger = MMAccessLogger(
            get_log_path(),
            profiling_debug_logging_config.log_file_backup_count,
        )

        vit_emb_cache_.resize_cache(self.vit_config.mm_cache_item_num)
        url_data_cache_.resize_cache(self.vit_config.url_cache_item_num)

    def inc_query_num(self) -> None:
        """Increment the query counter."""
        self.query_num += 1

    def dec_query_num(self) -> None:
        """Decrement the query counter."""
        self.query_num -= 1

    def get_query_num(self) -> int:
        """Get the current number of active queries."""
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
        return self.mm_embedding_impl(converted_inputs)

    def mm_embedding_cpp(
        self,
        urls: List[str],
        types: List[int],
        tensors: List[torch.Tensor],
        mm_preprocess_configs: List[Any],
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
        res = self.mm_embedding_impl(mm_inputs)
        res.position_ids = [pos.cpu() for pos in res.position_ids]
        return res

    def mm_embedding_impl(self, mm_inputs: List[MultimodalInput]) -> MMEmbeddingRes:
        """Core implementation for multimodal embedding processing."""
        logging.debug(f"{self.server_id} request received")
        try:
            # 如果不是 proxy 模式（即 standalone 模式），记录 QPS
            if not self.is_proxy_mode:
                kmonitor.report(
                    AccMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"}
                )

            self.inc_query_num()
            if not self.vit_config.disable_access_log:
                self._access_logger.log_query_access(mm_inputs)

            work_items = self._create_work_items(mm_inputs)
            self._wait_for_preprocessing(work_items)
            emb_res, pos_res, deepstack_embeds_res = self._compute_embeddings(
                work_items
            )

            result = MMEmbeddingRes(emb_res, pos_res, deepstack_embeds_res)
            if not self.vit_config.disable_access_log:
                self._access_logger.log_success_access(mm_inputs, str(result))

            # 如果不是 proxy 模式（即 standalone 模式），记录成功 QPS
            if not self.is_proxy_mode:
                kmonitor.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)

            return result
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            # 如果不是 proxy 模式（即 standalone 模式），记录错误 QPS
            if not self.is_proxy_mode:
                kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            self._access_logger.log_exception_access(mm_inputs, e)
            raise
        finally:
            self.dec_query_num()

    def _create_work_items(self, mm_inputs: List[MultimodalInput]) -> List[MMWorkItem]:
        """Create work items and submit preprocessing tasks."""
        batch_size = (
            self.mm_preprocess_batch_size
            if self.mm_preprocess_batch_size != -1
            else len(mm_inputs)
        )

        work_items = []
        for index in range(0, len(mm_inputs), batch_size):
            batch = mm_inputs[index : index + batch_size]
            work_item = MMWorkItem(batch, mm_timeout_ms=self.vit_config.mm_timeout_ms)
            self.preprocess_executor.submit(work_item)
            work_items.append(work_item)

        return work_items

    def _wait_for_preprocessing(
        self,
        work_items: List[MMWorkItem],
    ) -> None:
        """Wait for all preprocessing tasks to complete."""
        for work_item in work_items:
            self.preprocess_executor.get_result(work_item)

    def _compute_embeddings(
        self, work_items: List[MMWorkItem]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Compute embeddings for all work items."""
        emb_res, pos_res, tensor_res = [], [], []

        ordered_emb: List[Optional[Any]] = [None] * len(work_items)
        ordered_pos: List[Optional[Any]] = [None] * len(work_items)
        ordered_tensor: List[Optional[Any]] = [None] * len(work_items)

        pending_items: List[Tuple[int, MMWorkItem]] = []
        for idx, work_item in enumerate(work_items):
            if work_item.embedding_result is not None:
                ordered_emb[idx] = work_item.embedding_result[0]
                ordered_pos[idx] = work_item.embedding_result[1]
                if len(work_item.embedding_result) > 2:
                    ordered_tensor[idx] = work_item.embedding_result[2]
            else:
                pending_items.append((idx, work_item))

        if pending_items:
            batch_outputs = None
            with Timer() as route_timer:
                with mm_embedding_lock:
                    batch_outputs = self.mm_part.batched_embedding(
                        [wi.preprocess_result for _, wi in pending_items],
                        [wi.mm_type for _, wi in pending_items],
                    )
            kmonitor.report(GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms())

            if batch_outputs is not None:
                for (idx, work_item), result in zip(pending_items, batch_outputs):
                    work_item.embedding_result = result
                    if work_item.need_check_cache:
                        vit_emb_cache_.insert_cache(work_item.cache_key, result)
                    ordered_emb[idx] = result[0]
                    ordered_pos[idx] = result[1]
                    if len(result) > 2:
                        ordered_tensor[idx] = result[2]

        for emb, pos, tensor in zip(ordered_emb, ordered_pos, ordered_tensor):
            emb_res.extend(self._maybe_tensor_to_list(emb, dim=2))
            pos_res.extend(self._maybe_tensor_to_list(pos, dim=2))
            tensor_res.extend(self._maybe_tensor_to_list(tensor, dim=3))
        return emb_res, pos_res, tensor_res

    def stop(self) -> None:
        """Shutdown the preprocessing executor."""
        self.preprocess_executor.shutdown()
