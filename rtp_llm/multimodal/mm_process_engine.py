import concurrent.futures
import gc
import logging
import multiprocessing
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
        return (
            f"MMEmbeddingRes(embeddings={self.embeddings}, "
            f"position_ids={self.position_ids}, "
            f"deepstack_embeds={self.deepstack_embeds})"
        )


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

        self.future: Optional[multiprocessing.pool.AsyncResult] = None

    def may_submit_preprocess(
        self,
        mm_preprocess_pool: multiprocessing.pool.Pool,
    ) -> None:
        """
        Submit preprocessing task if not cached.
        """
        if self.embedding_result is not None:
            return

        self.future = mm_preprocess_pool.apply_async(
            _worker_process_task,
            args=(self.mm_inputs,),
        )

    def may_get_preprocess_result(self) -> None:
        """
        Get preprocessing result from future.
        """
        if self.future is None:
            if self.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            self.preprocess_result, preprocess_time = self.future.get(
                timeout=self.mm_timeout_ms / 1000.0
            )
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
        except multiprocessing.pool.TimeoutError:
            raise TimeoutError(f"Preprocessing timeout after {self.mm_timeout_ms}ms")
        except Exception as e:
            logging.error(f"Error getting preprocess result: {e}", exc_info=True)
            raise

    def get_embedding_result(self, embedding_func: Callable) -> Any:
        """Compute embedding result from preprocessed data or return cached result."""
        if self.embedding_result is not None:
            return self.embedding_result

        if self.preprocess_result is None:
            raise ValueError(
                "Preprocess result and embedding result in work item both be None"
            )

        with Timer() as route_timer:
            with mm_embedding_lock:
                self.embedding_result = embedding_func(
                    self.preprocess_result, mm_type=self.mm_type
                )
        kmonitor.report(GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms())

        if self.need_check_cache:
            vit_emb_cache_.insert_cache(self.cache_key, self.embedding_result)

        return self.embedding_result


class MMProcessEngine:
    """Engine for processing multimodal inputs with preprocessing and embedding."""

    def __init__(
        self,
        mm_part: MultiModalEmbeddingInterface,
        model_config: ModelConfig,
        vit_config: VitConfig,
        profiling_debug_logging_config: ProfilingDebugLoggingConfig,
    ):
        """Initialize the multimodal process engine."""
        self.vit_config = vit_config
        self.contains_pos: bool = (
            model_config.mm_model_config.mm_position_ids_style != 0
        )
        self.mm_preprocess_batch_size: int = (
            model_config.mm_related_params.preprocess_batch_size
        )

        self.mp_context = multiprocessing.get_context("spawn")

        self.mm_part = mm_part

        self.mm_preprocess_pool = None

        self.query_num: int = 0
        self._access_logger = MMAccessLogger(
            get_log_path(),
            profiling_debug_logging_config.log_file_backup_count,
        )

        vit_emb_cache_.resize_cache(self.vit_config.mm_cache_item_num)
        url_data_cache_.resize_cache(self.vit_config.url_cache_item_num)

    # # Make the engine picklable for spawn: drop non-picklable fields and recreate lazily.
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # ProcessPoolExecutor and loggers hold locks; drop and recreate after unpickle.
    #     state["mm_preprocess_pool"] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     if self.mm_preprocess_pool is None:
    #         self.mm_preprocess_pool = self._create_pool()

    def _create_pool(self) -> multiprocessing.pool.Pool:
        """Helper function to create a new process pool."""
        logging.info("Creating a new multiprocessing pool for preprocessing...")
        return self.mp_context.Pool(
            processes=self.vit_config.mm_preprocess_max_workers,
            initializer=_worker_initializer,
            initargs=(
                self.vit_config,
                self.mm_part.get_preprocess_params(),
                self.mm_part.preprocess_input,
            ),
        )

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
        return self.mm_embedding_impl(mm_inputs)

    def mm_embedding_impl(self, mm_inputs: List[MultimodalInput]) -> MMEmbeddingRes:
        """Core implementation for multimodal embedding processing."""
        try:
            kmonitor.report(AccMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"})
            self.inc_query_num()
            self._access_logger.log_query_access(mm_inputs)

            work_items = self._create_work_items(mm_inputs)
            self._wait_for_preprocessing(work_items)
            emb_res, pos_res, deepstack_embeds_res = self._compute_embeddings(
                work_items
            )

            kmonitor.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)
            result = MMEmbeddingRes(emb_res, pos_res, deepstack_embeds_res)
            self._access_logger.log_success_access(mm_inputs, str(result))
            return result
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            self._access_logger.log_exception_access(mm_inputs, e)
            raise
        finally:
            self.dec_query_num()

    @staticmethod
    def _get_child_pids_from_pool(pool: multiprocessing.pool.Pool) -> List[int]:
        """Extract child process PIDs from a multiprocessing.Pool."""
        try:
            # `_pool` is an internal attribute but the most reliable way
            return [p.pid for p in pool._pool if p.is_alive()]
        except Exception:
            return []

    def _recover_from_broken_process_pool(self) -> None:
        """Recover from BrokenProcessPool by shutting down and recreating the pool."""
        old_pool = self.mm_preprocess_pool

        with pool_lock:
            # Double-check: executor already replaced by another thread
            if self.mm_preprocess_pool is not old_pool:
                logging.debug("Pool already recovered by another thread")
                return

            kmonitor.report(AccMetrics.VIT_PROCESS_POOL_RESTART_QPS_METRIC, 1)
            child_pids = self._get_child_pids_from_pool(old_pool)

            logging.warning(
                f"Broken process pool detected. Terminating pool with PIDs: {child_pids}"
            )

            try:
                # Forcefully terminate the old pool
                old_pool.terminate()
                old_pool.join()
            except Exception as e:
                logging.warning(f"Error during pool termination: {e}", exc_info=True)

            # Re-create the pool
            try:
                self.mm_preprocess_pool = self._create_pool()
                logging.info("Recreated ProcessPool after it was broken.")
            except Exception as e:
                logging.error(f"Failed to create new ProcessPool: {e}", exc_info=True)
                raise

    def _submit_with_recovery(self, work_item: MMWorkItem) -> None:
        """Submit preprocessing task with automatic recovery from a broken pool."""
        max_retries = 2
        with pool_lock:
            if self.mm_preprocess_pool is None:
                self.mm_preprocess_pool = self._create_pool()
        for attempt in range(max_retries):
            try:
                work_item.may_submit_preprocess(self.mm_preprocess_pool)
                return  # Success
            except (
                BrokenPipeError,
                EOFError,
                OSError,
            ) as e:  # More specific exceptions for broken pools
                logging.warning(
                    f"Broken pool detected on submit (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    self._recover_from_broken_process_pool()
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
            self._submit_with_recovery(work_item)
            work_items.append(work_item)

        return work_items

    def _wait_for_preprocessing(
        self,
        work_items: List[MMWorkItem],
    ) -> None:
        """Wait for all preprocessing tasks to complete."""
        for work_item in work_items:
            try:
                work_item.may_get_preprocess_result()
            except (
                BrokenPipeError,
                EOFError,
                OSError,
            ) as e:  # Catch broken pool exceptions
                logging.error(f"Broken pool detected while waiting for result: {e}")
                self._recover_from_broken_process_pool()
                # Re-raise as a standard exception for the caller
                raise RuntimeError(
                    "Preprocessing failed due to a broken worker process."
                ) from e
            except Exception:
                # Other exceptions (like TimeoutError) are re-raised from may_get_preprocess_result
                raise

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
            try:
                with Timer() as route_timer:
                    with mm_embedding_lock:
                        batch_outputs = self.mm_part.batched_embedding(
                            [wi.preprocess_result for _, wi in pending_items],
                            [wi.mm_type for _, wi in pending_items],
                        )
                kmonitor.report(
                    GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms()
                )
            except NotImplementedError:
                batch_outputs = None
            except Exception:
                batch_outputs = None

            if batch_outputs is not None:
                for (idx, work_item), result in zip(pending_items, batch_outputs):
                    work_item.embedding_result = result
                    if work_item.need_check_cache:
                        vit_emb_cache_.insert_cache(work_item.cache_key, result)
                    ordered_emb[idx] = result[0]
                    ordered_pos[idx] = result[1]
                    if len(result) > 2:
                        ordered_tensor[idx] = result[2]
            else:
                for idx, work_item in pending_items:
                    result = work_item.get_embedding_result(self.mm_part.embedding)
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
        if self.mm_preprocess_pool is None:
            return
        logging.info("Shutting down the preprocessing pool...")
        self.mm_preprocess_pool.close()
        self.mm_preprocess_pool.join()
        logging.info("Preprocessing pool shut down.")
