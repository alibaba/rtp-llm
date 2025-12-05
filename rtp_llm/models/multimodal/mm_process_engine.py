import concurrent.futures
import gc
from multiprocessing import Lock
from typing import Any, Callable, List, Optional, Tuple

import torch

from rtp_llm.access_logger.access_logger import MMAccessLogger
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalInputsPB
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.models.multimodal.multimodal_common import MultiModalEmbeddingInterface
from rtp_llm.models.multimodal.multimodal_util import trans_mm_input, vit_emb_cache_
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)
from rtp_llm.utils.time_util import Timer

# Global lock for embedding operations to ensure thread safety
mm_embedding_lock = Lock()


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

    def __init__(self, mm_inputs: List[MultimodalInput]):
        if not mm_inputs:
            raise ValueError("No mm_input for work item")

        self.mm_inputs = mm_inputs
        self.mm_timeout_ms = self.mm_inputs[0].config.mm_timeout_ms
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result: Optional[Any] = None
        self.embedding_result: Optional[Any] = None

        # Only cache single URL inputs
        self.need_check_cache = len(mm_inputs) == 1 and mm_inputs[0].url is not None

        self.check_cache()

    def check_cache(self) -> None:
        """Check if embedding result is available in cache."""
        if self.need_check_cache:
            mm_input = self.mm_inputs[0]
            cached_res = vit_emb_cache_.check_cache(mm_input.to_string())
            if cached_res is not None:
                self.embedding_result = cached_res

    @staticmethod
    def download_and_preprocess(
        mm_inputs: List[MultimodalInput],
        preprocess_params: dict,
        preprocess_func: Callable,
    ) -> Tuple[Any, float]:
        """Download and preprocess multimodal inputs in a separate process."""
        with Timer() as route_timer:
            result = preprocess_func(mm_inputs, **preprocess_params)
        return result, route_timer.cost_ms()

    def may_submit_preprocess(
        self,
        mm_part: Optional[MultiModalEmbeddingInterface] = None,
        mm_preprocess_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None,
    ) -> Optional[concurrent.futures.Future]:
        """
        Submit preprocessing task if not cached.

        Returns:
            Future object if task is submitted, None if result is cached.
        """
        if self.embedding_result is not None:
            return None

        if mm_part is None or mm_preprocess_executor is None:
            raise ValueError("mm_part and mm_preprocess_executor must be provided")

        return mm_preprocess_executor.submit(
            MMWorkItem.download_and_preprocess,
            self.mm_inputs,
            mm_part.get_preprocess_params(),
            mm_part.preprocess_input,
        )

    def may_get_preprocess_result(
        self, future: Optional[concurrent.futures.Future]
    ) -> None:
        """
        Get preprocessing result from future.

        Note: Future cannot be pickled, so it cannot be a member of MMWorkItem.
        """
        if future is None:
            if self.embedding_result is not None:
                return
            else:
                raise ValueError("Embedding result and future cannot both be None")

        try:
            self.preprocess_result, preprocess_time = future.result(
                timeout=self.mm_timeout_ms / 1000.0
            )
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"Preprocessing timeout after {self.mm_timeout_ms}ms")
        except Exception as e:
            future.cancel()
            raise

    def get_embedding_result(self, embedding_func: Callable) -> Any:
        """Compute embedding result from preprocessed data or return cached result."""
        if self.preprocess_result is not None:
            with Timer() as route_timer:
                with mm_embedding_lock:
                    self.embedding_result = embedding_func(
                        self.preprocess_result, mm_type=self.mm_type
                    )
            kmonitor.report(GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms())

            if self.need_check_cache:
                vit_emb_cache_.insert_cache(
                    self.mm_inputs[0].to_string(), self.embedding_result
                )
            return self.embedding_result
        elif self.embedding_result is not None:
            return self.embedding_result
        else:
            raise ValueError(
                "Preprocess result and embedding result in work item both be None"
            )


class MMProcessEngine:
    """Engine for processing multimodal inputs with preprocessing and embedding."""

    def __init__(self, model: Any):
        """
        Initialize the multimodal process engine.

        Args:
            model: The model containing configuration and multimodal components.
        """
        self.model = model
        self.contains_pos: bool = self.model.config.mm_position_ids_style != 0
        self.mm_preprocess_batch_size: int = self.model.config.mm_preprocess_batch_size

        self.mm_preprocess_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.model.config.py_env_configs.vit_config.mm_preprocess_max_workers
        )

        self.query_num: int = 0
        self.query_num_lock = Lock()
        self._access_logger = MMAccessLogger()

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

    def _maybe_tensor_to_list(self, tensor: Any, dim: int = 2) -> List[Any]:
        """
        Convert tensor to list format if needed.

        Args:
            tensor: Input tensor or list

        Returns:
            List representation of the tensor
        """
        if tensor is None:
            return []
        elif not isinstance(tensor, torch.Tensor):
            return tensor
        elif len(tensor.shape) > dim:
            return list(tensor)
        else:
            return [tensor]

    def mm_embedding_rpc(self, mm_inputs: MultimodalInputsPB) -> MMEmbeddingRes:
        """
        Process multimodal inputs from RPC protocol buffer.

        Args:
            mm_inputs: Protocol buffer containing multimodal inputs

        Returns:
            MMEmbeddingRes containing embeddings and position IDs
        """
        converted_inputs = trans_mm_input(mm_inputs)
        return self.mm_embedding_impl(converted_inputs)

    def mm_embedding_cpp(
        self,
        urls: List[str],
        types: List[int],
        tensors: List[torch.Tensor],
        mm_preprocess_configs: List[Any],
    ) -> MMEmbeddingRes:
        """
        Process multimodal inputs from C++ interface.

        Args:
            urls: List of input URLs
            types: List of URL types
            tensors: List of input tensors
            mm_preprocess_configs: List of preprocessing configurations

        Returns:
            MMEmbeddingRes containing embeddings and position IDs
        """
        mm_inputs = []
        for url, url_type, tensor, config in zip(
            urls, types, tensors, mm_preprocess_configs
        ):
            mm_inputs.append(
                MultimodalInput(
                    url, MMUrlType(url_type), tensor, MMPreprocessConfig(*config)
                )
            )
        return self.mm_embedding_impl(mm_inputs)

    def mm_embedding_impl(self, mm_inputs: List[MultimodalInput]) -> MMEmbeddingRes:
        """
        Core implementation for multimodal embedding processing.

        Args:
            mm_inputs: List of multimodal inputs to process

        Returns:
            MMEmbeddingRes containing embeddings and position IDs

        Raises:
            Exception: If processing fails
        """
        try:
            kmonitor.report(GaugeMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"})
            self.inc_query_num()
            self._access_logger.log_query_access(mm_inputs)

            work_items, futures = self._create_work_items(mm_inputs)
            self._wait_for_preprocessing(futures, work_items)
            emb_res, pos_res, tensor_res = self._compute_embeddings(work_items)

            kmonitor.report(GaugeMetrics.VIT_SUCCESS_QPS_METRIC, 1)
            result = MMEmbeddingRes(emb_res, pos_res, tensor_res)
            self._access_logger.log_success_access(mm_inputs, str(result))
            return result
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            kmonitor.report(GaugeMetrics.VIT_ERROR_QPS_METRIC, 1)
            self._access_logger.log_exception_access(mm_inputs, e)
            raise
        finally:
            self.dec_query_num()

    def _create_work_items(
        self, mm_inputs: List[MultimodalInput]
    ) -> Tuple[List[MMWorkItem], List[Optional[concurrent.futures.Future]]]:
        """Create work items and submit preprocessing tasks."""
        work_items = []
        futures = []

        batch_size = (
            self.mm_preprocess_batch_size
            if self.mm_preprocess_batch_size != -1
            else len(mm_inputs)
        )

        for index in range(0, len(mm_inputs), batch_size):
            batch = mm_inputs[index : index + batch_size]
            work_item = MMWorkItem(batch)
            future = work_item.may_submit_preprocess(
                self.model.mm_part, self.mm_preprocess_executor
            )
            futures.append(future)
            work_items.append(work_item)

        return work_items, futures

    def _wait_for_preprocessing(
        self,
        futures: List[Optional[concurrent.futures.Future]],
        work_items: List[MMWorkItem],
    ) -> None:
        """Wait for all preprocessing tasks to complete."""
        for future, work_item in zip(futures, work_items):
            work_item.may_get_preprocess_result(future)

    def _compute_embeddings(
        self, work_items: List[MMWorkItem]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Compute embeddings for all work items."""
        emb_res = []
        pos_res = []
        tensor_res = []

        for work_item in work_items:
            result = work_item.get_embedding_result(self.model.mm_part.embedding)
            emb_res.extend(self._maybe_tensor_to_list(result[0], dim=2))
            pos_res.extend(self._maybe_tensor_to_list(result[1], dim=2))
            if len(result) > 2:
                tensor_res.extend(self._maybe_tensor_to_list(result[2], dim=3))

        return emb_res, pos_res, tensor_res

    def stop(self) -> None:
        """Shutdown the preprocessing executor."""
        self.mm_preprocess_executor.shutdown(wait=True)
