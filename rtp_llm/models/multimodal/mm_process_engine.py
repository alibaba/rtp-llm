import asyncio
import concurrent.futures
import enum
import gc
import logging
import time
import uuid
from multiprocessing import Lock, Manager, Process
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
    ):
        if len(mm_inputs) == 0:
            raise Exception("No mm_input for work item")
        self.mm_inputs = mm_inputs
        self.status = MMWorkItemStatus.WAITING

        self.mm_timeout_ms = self.mm_inputs[0].config.mm_timeout_ms
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result = None
        self.embedding_result = None

        self.work_item_id: str = str(uuid.uuid4())
        # self.check_cache()

    @property
    def id(self):
        return self.id

    @property
    def finished(self):
        return self.status == MMWorkItemStatus.FINISHED

    def set_status(self, status: MMWorkItemStatus):
        self.status = status

    # def check_cache(self):
    #     # only cache url, type and config
    #     if len(self.mm_inputs) == 1 and self.mm_inputs[0].url is not None:
    #         mm_input = self.mm_inputs[0]
    #         cached_res = vit_emb_cache_.check_cache(mm_input.to_string())
    #         if cached_res is not None:
    #             self.status = MMWorkItemStatus.FINISHED
    #             self.embedding_result = cached_res

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
        if self.status == MMWorkItemStatus.WAITING:
            self.status = MMWorkItemStatus.PREPROCESSING
            return mm_preprocess_executor.submit(
                MMWorkItem.download_and_preprocess,
                self.mm_inputs,
                mm_part.get_preprocess_params(),
                mm_part.preprocess_input,
            )
        return None

    # future cannot be pickled, so it cannot be a member of MMWorkItem
    def may_get_preprocess_result(self, future):
        if self.status == MMWorkItemStatus.PREPROCESSING:
            if future == None:
                return
            try:
                self.preprocess_result, preprocess_time = future.result(
                    timeout=self.mm_timeout_ms / 1000
                )

                kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
                self.set_status(MMWorkItemStatus.PREPROCESSED)
            except Exception as e:
                future.cancel()
                self.set_status(MMWorkItemStatus.ERROR)
                raise e

    def update_embedding_result(self, res):
        self.set_status(MMWorkItemStatus.FINISHED)
        self.embedding_result = res
        # if len(self.mm_inputs) == 1 and self.mm_inputs[0].url is not None:
        #     vit_emb_cache_.insert_cache(self.mm_inputs[0].to_string(), res)


class MMProcessEngine:
    def __init__(self, model):
        self.model = model
        self.contains_pos: bool = self.model.config.mm_position_ids_style != 0

        self.mm_preprocess_batch_size: int = self.model.config.mm_preprocess_batch_size

        manager = Manager()
        self.waiting_queue = manager.Queue(maxsize=0)
        self.running_list = manager.list()
        self.backend_loop_thread = None
        # self.backend_loop_process = Process(
        #     target=MMProcessEngine.loop,
        #     args=(
        #         self.waiting_queue,
        #         self.running_list,
        #         self.model.config.py_env_configs.vit_config.mm_batch_size,
        #         self.model.mm_part.batched_embedding
        #     )
        # )
        # self.backend_loop_process.start()

        self.mm_preprocess_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.model.config.py_env_configs.vit_config.mm_preprocess_max_workers
        )

        self.query_num = 0
        self.query_num_lock = Lock()

    def inc_query_num(self):
        with self.query_num_lock:
            self.query_num += 1

    def dec_query_num(self):
        with self.query_num_lock:
            self.query_num -= 1

    def get_query_num(self):
        return self.query_num

    @staticmethod
    def loop(
        waiting_queue: Queue,
        running_list: List[MMWorkItem],
        max_batch_size: int,
        embedding_func,
    ):
        try:
            while True:
                running_list[:] = []
                while len(running_list) < max_batch_size and waiting_queue.qsize() > 0:
                    work_item = waiting_queue.get_nowait()
                    if work_item.status == MMWorkItemStatus.FINISHED:
                        continue
                    elif work_item.status == MMWorkItemStatus.PREPROCESSED:
                        work_item.set_status(MMWorkItemStatus.RUNNING)
                        running_list.append(work_item)
                    else:
                        logging.error(
                            f"Invalid work item status for loop: {work_item.status}"
                        )
                if len(running_list) == 0:
                    time.sleep(0.01)
                    continue
                gathered_inputs = []
                for work_item in running_list:
                    gathered_inputs.append(work_item.preprocess_result)
                with Timer() as route_timer:
                    gathered_results = embedding_func(
                        gathered_inputs,
                        mm_types=[work_item.mm_type for work_item in running_list],
                    )
                kmonitor.report(
                    GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms()
                )

                if len(gathered_results) != len(running_list):
                    raise Exception(
                        f"Invalid gathered results length: {len(gathered_results)} != {len(running_list)}"
                    )
                for index, work_item in enumerate(running_list):
                    work_item.update_embedding_result(gathered_results[index])
        except Exception as e:
            logging.error(f"Error in mm backend process loop: {e}")
            raise e

    def _maybe_tensor_to_list(self, tensor):
        if tensor == None:
            return []
        elif not isinstance(tensor, torch.Tensor):
            return tensor
        elif len(tensor.shape) > 2:
            return list(tensor)
        else:
            return [tensor]

    def submit(self, work_items: List[MMWorkItem]):
        for work_item in work_items:
            if work_item.status == MMWorkItemStatus.PREPROCESSED:
                self.waiting_queue.put_nowait(work_item)
            # cached result
            elif work_item.status == MMWorkItemStatus.FINISHED:
                continue
            else:
                raise Exception(
                    f"Invalid work item status for submit: {work_item.status}"
                )

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
            self.inc_query_num()
            work_items = []
            emb_res = []
            pos_res = []

            # embedding model; gather batches in advance
            mm_preprocess_batch_size = (
                self.mm_preprocess_batch_size
                if self.mm_preprocess_batch_size != -1
                else len(mm_inputs)
            )
            futures = []
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
                if work_item.status == MMWorkItemStatus.PREPROCESSED:
                    res = self.model.mm_part.embedding(
                        work_item.preprocess_result, mm_type=work_item.mm_type
                    )
                elif work_item.status == MMWorkItemStatus.FINISHED:
                    res = work_item.embedding_result
                else:
                    raise Exception(
                        f"Invalid work item status for embedding: {work_item.status}"
                    )
                emb_res.extend(self._maybe_tensor_to_list(res[0]))
                pos_res.extend(self._maybe_tensor_to_list(res[1]))

            # self.submit(work_items)
            # for work_item in work_items:
            #     while not work_item.finished:
            #         time.sleep(0.01)
            #     emb_res.extend(self._maybe_tensor_to_list(work_item.embedding_result[0]))
            #     pos_res.extend(self._maybe_tensor_to_list(work_item.embedding_result[1]))
            self.dec_query_num()
            return MMEmbeddingRes(emb_res, pos_res)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e

    def stop(self):
        # self.backend_loop_process.terminate()
        # self.backend_loop_process.join()
        self.mm_preprocess_executor.shutdown()
