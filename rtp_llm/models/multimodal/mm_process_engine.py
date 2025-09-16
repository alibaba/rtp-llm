import concurrent.futures
import enum
import gc
from multiprocessing import Lock
from queue import Queue
from typing import Any, List, Optional, Union

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.models.multimodal.multimodal_common import (
    MultiModalEmbeddingInterface,
    mm_lock,
)
from rtp_llm.utils.multimodal_util import MMPreprocessConfig, MMUrlType, vit_emb_cache_
from rtp_llm.utils.time_util import Timer
from rtp_llm.utils.util import check_with_info


class MMEmbeddingRes:
    embeddings: List[torch.Tensor] = []
    position_ids: Optional[List[torch.Tensor]] = None

    def __init__(self, embeddings, position_ids=None):
        self.embeddings = embeddings
        self.position_ids = position_ids


def download_and_preprocess(
    url,
    type,
    tensor,
    preprocess_config,
    preprocess_params,
    preprocess_func,
):
    with Timer() as route_timer:
        res = preprocess_func(url, type, tensor, preprocess_config, **preprocess_params)
    kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, route_timer.cost_ms())
    return res


class MMWorkItemStatus(enum.Enum):
    WAITING = 0
    PREPROCESSING = 1
    RUNNING = 2
    FINISHED = 3
    STOPPED = 4
    ERROR = 5


class MMWorkItem:
    def __init__(
        self,
        url: Optional[Union[str, List[str]]],
        type: Optional[Union[MMUrlType, List[MMUrlType]]],
        tensor: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        preprocess_config: Optional[
            Union[MMPreprocessConfig, List[MMPreprocessConfig]]
        ] = None,
        mm_part: MultiModalEmbeddingInterface = None,
        mm_preprocess_executor: concurrent.futures.ProcessPoolExecutor = None,
        is_batched: bool = False,
    ):
        self.url = url
        self.type = type
        self.tensor = tensor
        self.embedding_func = mm_part.embedding
        self.is_batched = is_batched

        self.mm_timeout_ms = (
            preprocess_config[0].mm_timeout_ms
            if isinstance(preprocess_config, list)
            else preprocess_config.mm_timeout_ms
        )

        self.res = None
        if not is_batched:
            cached_res = vit_emb_cache_.check_cache(url)
            if cached_res is not None:
                # self.status = MMWorkItemStatus.FINISHED
                self.res = cached_res

        if self.res == None:
            # self.status = MMWorkItemStatus.PREPROCESSING
            self.future = mm_preprocess_executor.submit(
                download_and_preprocess,
                url,
                type,
                tensor,
                preprocess_config,
                mm_part.get_preprocess_params(),
                mm_part.preprocess_input,
            )

    # TODO: should be replaced by embedding engine
    def run_embedding(self):
        if self.res != None:
            return self.res
        else:
            # result = self.res
            try:
                result = self.future.result(timeout=self.mm_timeout_ms / 1000)
            except Exception as e:
                self.future.cancel()
                raise TimeoutError(f"Timeout after {self.mm_timeout_ms / 1000} seconds")
            with Timer() as route_timer:
                with mm_lock:
                    res = self.embedding_func(result, mm_type=self.type)
            kmonitor.report(GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms())
            return res


# TODO: batch process embedding engine; backend process
# class MMBatchProcessEngine:
#     def __init__(
#         self, mm_part: MultiModalEmbeddingInterface = None, max_batch_size: int = 1
#     ):
#         self.mm_queue = Queue(max_batch_size)
#         self.lock = Lock()

#     def submit(self, work_item: MMWorkItem):
#         with self.lock:
#             self.mm_queue.put(work_item)

#     def gather_batch(self):
#         raise NotImplementedError("MMBatchProcessEngine.gather_batch is not implemented")


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

    def _maybe_tensor_to_list(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        elif len(tensor.shape) > 2:
            return list(tensor)
        else:
            return [tensor]

    def submit(
        self,
        urls: List[str],
        types: Optional[List[MMUrlType]] = None,
        tensors: Optional[List[torch.Tensor]] = None,
        preprocess_configs: Optional[List[List[Any]]] = None,
    ):
        if types is None or len(types) == 0:
            types = [MMUrlType.DEFAULT] * len(urls)
        if preprocess_configs is None or len(preprocess_configs) == 0:
            configs = [MMPreprocessConfig()] * len(urls)
        else:
            configs = [MMPreprocessConfig(*config) for config in preprocess_configs]
        try:
            work_items = []
            emb_res = []
            pos_res = [] if self.contains_pos else None

            # embedding model; gather batches in advance
            if self.model.config.mm_batch_size != 1:
                if self.model.config.mm_batch_size == -1:
                    work_items.append(
                        MMWorkItem(
                            urls,
                            types,
                            tensors,
                            configs,
                            self.model.mm_part,
                            self.mm_preprocess_executor,
                            is_batched=True,
                        )
                    )
                else:
                    for index in range(0, len(urls), self.mm_batch_size):
                        work_items.append(
                            MMWorkItem(
                                urls[index : index + self.mm_batch_size],
                                types[index : index + self.mm_batch_size],
                                tensors[index : index + self.mm_batch_size],
                                configs[index : index + self.mm_batch_size],
                                self.model.mm_part,
                                self.mm_preprocess_executor,
                                is_batched=True,
                            )
                        )
            else:
                for index in range(len(urls)):
                    work_items.append(
                        MMWorkItem(
                            urls[index],
                            types[index],
                            tensors[index],
                            configs[index],
                            self.model.mm_part,
                            self.mm_preprocess_executor,
                            is_batched=False,
                        )
                    )
            for work_item in work_items:
                emb, pos = work_item.run_embedding()
                emb_res.extend(self._maybe_tensor_to_list(emb))
                if self.contains_pos:
                    check_with_info(pos is not None, "pos should not be None")
                    pos_res.extend(self._maybe_tensor_to_list(pos))
            return MMEmbeddingRes(emb_res, pos_res)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e
