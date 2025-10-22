import concurrent.futures
import enum
import gc
from multiprocessing import Lock
from queue import Queue
from typing import Any, List, Optional, Union

import torch

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
from rtp_llm.utils.util import check_with_info


class MMEmbeddingRes:
    embeddings: List[torch.Tensor] = []
    position_ids: Optional[List[torch.Tensor]] = None

    def __init__(self, embeddings, position_ids=None):
        self.embeddings = embeddings
        self.position_ids = position_ids


class MMWorkItemStatus(enum.Enum):
    WAITING = 0
    PREPROCESSING = 1
    PREPROCESSED = 2
    RUNNING = 3
    FINISHED = 4
    STOPPED = 5
    ERROR = 6


class MMWorkItem:
    def __init__(
        self,
        mm_inputs: List[MultimodalInput],
        lock: Lock = None,
    ):
        self.mm_inputs = mm_inputs
        self.lock = lock
        self.status = MMWorkItemStatus.WAITING

        self.mm_timeout_ms = self.mm_inputs[0].config.mm_timeout_ms

        self.res = None
        self.check_cache()

    def check_cache(self):
        # only cache url, type and config
        # consider multimodal embedding cases, they
        if len(self.mm_inputs) == 1 and self.mm_inputs[0].url is not None:
            mm_input = self.mm_inputs[0]
            cached_res = vit_emb_cache_.check_cache(mm_input.to_string())
            if cached_res is not None:
                self.status = MMWorkItemStatus.FINISHED
                self.res = cached_res

    @staticmethod
    def download_and_preprocess(
        mm_inputs: List[MultimodalInput],
        preprocess_params,
        preprocess_func,
    ):
        with Timer() as route_timer:
            res = preprocess_func(mm_inputs, **preprocess_params)
        kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, route_timer.cost_ms())
        return res

    def submit_preprocess(
        self,
        mm_part: MultiModalEmbeddingInterface = None,
        mm_preprocess_executor: concurrent.futures.ProcessPoolExecutor = None,
    ):
        if self.status == MMWorkItemStatus.WAITING:
            self.status = MMWorkItemStatus.PREPROCESSING
            self.future = mm_preprocess_executor.submit(
                MMWorkItem.download_and_preprocess,
                self.mm_inputs,
                mm_part.get_preprocess_params(),
                mm_part.preprocess_input,
            )
            return self.future

    # TODO: should be replaced by embedding engine
    def run_embedding(self, embedding_func: callable):
        if self.res != None:
            return self.res
        else:
            # result = self.res
            try:
                result = self.future.result(timeout=self.mm_timeout_ms / 1000)
            except Exception as e:
                self.future.cancel()
                raise e

            with Timer() as route_timer:
                with self.lock:
                    res = embedding_func(result, mm_type=self.mm_inputs[0].mm_type)
            kmonitor.report(GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms())
            return res


# class MMBatchProcessEngine:
#     def __init__(
#         self, mm_part: MultiModalEmbeddingInterface = None, max_batch_size: int = 1
#     ):
#         self.waiting_queue = Queue(max_batch_size)
#         self.running_list = []
#         self.finished_list = []
#         self.lock = Lock()

#     def submit(self, work_item: MMWorkItem):
#         with self.lock:
#             self.waiting_queue.put(work_item)

#     def get_batched_data(self):
#         while self.waiting_queue.qsize() > 0:
#             with self.lock:
#                 work_item = self.waiting_queue.get()
#             self.running_list.append(work_item)
#             work_item.submit_preprocess(self.mm_part, self.mm_preprocess_executor)


class MMProcessEngine:
    def __init__(self, model):
        self.model = model
        self.contains_pos: bool = self.model.config.mm_position_ids_style != 0

        self.mm_batch_size: int = self.model.config.mm_batch_size

        self.backend_engine_process = None
        self.backend_engine_batch_size = None

        self.mm_preprocess_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.model.config.py_env_configs.vit_config.mm_preprocess_max_workers
        )
        self.preprocess_lock = Lock()

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
        mm_preprocess_configs: List[List[int]],
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
            work_items = []
            emb_res = []
            pos_res = []

            # embedding model; gather batches in advance
            mm_batch_size = (
                self.mm_batch_size if self.mm_batch_size != -1 else len(mm_inputs)
            )
            for index in range(0, len(mm_inputs), mm_batch_size):
                work_item = MMWorkItem(
                    mm_inputs[index : index + mm_batch_size],
                    lock=self.preprocess_lock,
                )
                work_item.submit_preprocess(
                    self.model.mm_part, self.mm_preprocess_executor
                )
                work_items.append(work_item)
            for work_item in work_items:
                emb, pos = work_item.run_embedding(self.model.mm_part.embedding)
                emb_res.extend(self._maybe_tensor_to_list(emb))
                pos_res.extend(self._maybe_tensor_to_list(pos))
            return MMEmbeddingRes(emb_res, pos_res)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e
