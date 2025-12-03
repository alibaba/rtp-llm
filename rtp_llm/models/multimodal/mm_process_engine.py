import asyncio
import concurrent.futures
import enum
import gc
import logging
import time
import uuid
from abc import ABC, abstractmethod
from multiprocessing import Lock, Manager, Process
from queue import Queue
from typing import Any, List, Optional, Union

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


class MMEmbeddingRes:
    embeddings: List[torch.Tensor] = []
    position_ids: Optional[List[torch.Tensor]] = None
    deepstack_embeds: Optional[List[torch.Tensor]] = None

    def __init__(self, embeddings, position_ids=None, deepstack_embeds=None):
        self.embeddings = embeddings
        self.position_ids = position_ids
        self.deepstack_embeds = deepstack_embeds

    def __str__(self):
        return f"MMEmbeddingRes(embeddings={self.embeddings}, position_ids={self.position_ids}, deepstack_embeds={self.deepstack_embeds})"


class MMWorkItemStatus(enum.Enum):
    WAITING = 0
    PREPROCESSING = 1
    RUNNING = 2
    FINISHED = 3
    ERROR = 4


mm_embedding_lock = Lock()


class MMWorkItem:
    def __init__(
        self,
        mm_inputs: List[MultimodalInput],
    ):
        if len(mm_inputs) == 0:
            raise Exception("No mm_input for work item")
        self.mm_inputs = mm_inputs

        self.mm_timeout_ms = self.mm_inputs[0].config.mm_timeout_ms
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result = None
        self.embedding_result = None

        self.need_check_cache = len(mm_inputs) == 1 and mm_inputs[0].url is not None

        self.check_cache()

    def check_cache(self):
        # only cache url, type and config
        if self.need_check_cache:
            mm_input = self.mm_inputs[0]
            cached_res = vit_emb_cache_.check_cache(mm_input.to_string())
            if cached_res is not None:
                self.embedding_result = cached_res

    @staticmethod
    def download_and_preprocess(
        mm_inputs: List[MultimodalInput],
        preprocess_params,
        preprocess_func,
    ):
        with Timer() as route_timer:
            res = preprocess_func(mm_inputs, **preprocess_params)
        return res, route_timer.cost_ms()

    def may_submit_preprocess(
        self,
        mm_part: MultiModalEmbeddingInterface = None,
        mm_preprocess_executor: concurrent.futures.ProcessPoolExecutor = None,
    ):
        # cached
        if self.embedding_result is not None:
            return None

        return mm_preprocess_executor.submit(
            MMWorkItem.download_and_preprocess,
            self.mm_inputs,
            mm_part.get_preprocess_params(),
            mm_part.preprocess_input,
        )

    # future cannot be pickled, so it cannot be a member of MMWorkItem
    def may_get_preprocess_result(self, future):
        if future == None and self.embedding_result is not None:
            return
        elif future == None and self.embedding_result is None:
            raise Exception("Embedding result and future cannot both be None")

        try:
            self.preprocess_result, preprocess_time = future.result(
                timeout=self.mm_timeout_ms / 1000
            )
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
        except Exception as e:
            future.cancel()
            raise e

    def get_embedding_result(self, embedding_func):
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
            raise Exception(
                "Preprocess result and embedding result in work item both be None"
            )


class MMProcessEngine:
    def __init__(self, model):
        self.model = model
        self.contains_pos: bool = self.model.config.mm_position_ids_style != 0

        self.mm_preprocess_batch_size: int = self.model.config.mm_preprocess_batch_size

        self.mm_preprocess_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.model.config.py_env_configs.vit_config.mm_preprocess_max_workers
        )

        self.query_num = 0
        self.query_num_lock = Lock()
        self._access_logger = MMAccessLogger()

    def inc_query_num(self):
        with self.query_num_lock:
            self.query_num += 1

    def dec_query_num(self):
        with self.query_num_lock:
            self.query_num -= 1

    # for worker status
    def get_query_num(self):
        with self.query_num_lock:
            return self.query_num

    def _maybe_tensor_to_list(self, tensor):
        if tensor == None:
            return []
        elif not isinstance(tensor, torch.Tensor):
            return tensor
        elif len(tensor.shape) > 2:
            return list(tensor)
        else:
            return [tensor]

    def mm_embedding_rpc(
        self,
        mm_inputs: MultimodalInputsPB,
    ):
        mm_inputs = trans_mm_input(mm_inputs)
        return self.mm_embedding_impl(mm_inputs)

    # used for local multimodal processor
    def mm_embedding_cpp(
        self,
        urls: List[str],
        types: List[int],
        tensors: List[torch.Tensor],
        mm_preprocess_configs: List[Any],
    ):
        mm_inputs = []
        for url, type, tensor, config in zip(
            urls, types, tensors, mm_preprocess_configs
        ):
            mm_inputs.append(
                MultimodalInput(
                    url, MMUrlType(type), tensor, MMPreprocessConfig(*config)
                )
            )
        return self.mm_embedding_impl(mm_inputs)

    def mm_embedding_impl(self, mm_inputs: List[MultimodalInput]):
        try:
            kmonitor.report(GaugeMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"})
            self.inc_query_num()
            self._access_logger.log_query_access(mm_inputs)

            work_items = []
            futures = []
            emb_res = []
            pos_res = []
            tensor_res = []

            # embedding model; gather batches in advance
            mm_preprocess_batch_size = (
                self.mm_preprocess_batch_size
                if self.mm_preprocess_batch_size != -1
                else len(mm_inputs)
            )
            for index in range(0, len(mm_inputs), mm_preprocess_batch_size):
                work_item = MMWorkItem(
                    mm_inputs[index : index + mm_preprocess_batch_size]
                )
                future = work_item.may_submit_preprocess(
                    self.model.mm_part, self.mm_preprocess_executor
                )
                futures.append(future)
                work_items.append(work_item)
            for future, work_item in zip(futures, work_items):
                work_item.may_get_preprocess_result(future)
            for work_item in work_items:
                res = work_item.get_embedding_result(self.model.mm_part.embedding)
                emb_res.extend(self._maybe_tensor_to_list(res[0]))
                pos_res.extend(self._maybe_tensor_to_list(res[1]))

            kmonitor.report(GaugeMetrics.VIT_SUCCESS_QPS_METRIC, 1)
            res = MMEmbeddingRes(emb_res, pos_res, tensor_res)
            self._access_logger.log_success_access(mm_inputs, str(res))
            return res
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            kmonitor.report(GaugeMetrics.VIT_ERROR_QPS_METRIC, 1)
            self._access_logger.log_exception_access(mm_inputs, e)
            raise e
        finally:
            self.dec_query_num()

    def stop(self):
        self.mm_preprocess_executor.shutdown()
