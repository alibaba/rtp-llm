import gc
import logging
import threading
import time
from concurrent.futures import CancelledError, Future, wait
from typing import List, Optional

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.ops import VitSeparation
from rtp_llm.utils.mm_batch_scheduler import BatchResult, MMBatchScheduler
from rtp_llm.utils.multimodal_util import MMDataCache, MMPreprocessConfig, MMUrlType
from rtp_llm.utils.time_util import Timer
from rtp_llm.utils.util import check_with_info


class MMEmbeddingRes:
    embeddings: List[torch.Tensor] = []
    position_ids: Optional[List[torch.Tensor]] = None

    def __init__(self, embeddings, position_ids=None, max_batch_size=0, gpu_forwards=0):
        self.embeddings = embeddings
        self.position_ids = position_ids
        self.max_batch_size = max_batch_size
        self.gpu_forwards = gpu_forwards


class MMProcessEngine:
    def __init__(self, model, vit_config):
        self.model = model
        self.vit_config = vit_config
        self.contains_pos: bool = (
            self.model.model_config.mm_model_config.mm_position_ids_style != 0
        )
        self.run_batch: bool = self.model.model_config.mm_related_params.support_batch
        self.download_headers = self.vit_config.download_headers
        self._scheduler = None
        self._embedding_cache = MMDataCache(getattr(vit_config, "mm_cache_item_num", 0))
        mm_part = getattr(model, "mm_part", None)
        if (
            getattr(vit_config, "vit_separation", VitSeparation.VIT_SEPARATION_LOCAL)
            == VitSeparation.VIT_SEPARATION_ROLE
            and callable(getattr(mm_part, "preprocess_embedding", None))
            and callable(getattr(mm_part, "batch_embedding", None))
        ):
            self._scheduler = MMBatchScheduler(
                self._batch_embedding,
                vit_config.vit_batch_wait_ms,
                vit_config.vit_max_batch_images,
                vit_config.vit_max_batch_patches,
            )

    @torch.inference_mode()
    def _batch_embedding(self, items):
        mm_part = self.model.mm_part
        device = torch.device(mm_part._device)
        if device.type == "cuda":
            # CUDA's current device is thread-local.
            with torch.cuda.device(device):
                return mm_part.batch_embedding(items)
        return mm_part.batch_embedding(items)

    @staticmethod
    def _check_request(deadline, cancelled):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("ViT request cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("ViT request deadline exceeded")

    def _submit_scheduled(self, urls, types, configs, deadline, cancelled):
        cancelled = cancelled or threading.Event()
        futures = []
        cache_keys = []
        output_tokens = 0
        try:
            for url, mm_type, config in zip(urls, types, configs):
                self._check_request(deadline, cancelled)
                key = (url, int(mm_type), tuple(vars(config).items()))
                cached = self._embedding_cache.check_cache(key)
                if cached is None:
                    image = self.model.mm_part.preprocess_embedding(
                        url,
                        mm_type,
                        download_headers=self.download_headers,
                        configs=config,
                    )
                    output_tokens += image.output_tokens
                else:
                    output_tokens += cached[0].size(0)
                if output_tokens >= self.model.model_config.max_seq_len:
                    raise ValueError("ViT output exceeds the model sequence length")
                output_bytes = (
                    output_tokens
                    * self.model.model_config.hidden_size
                    * torch.empty(
                        (), dtype=self.model.model_config.compute_dtype
                    ).element_size()
                )
                if output_bytes >= 1024 * 1024 * 1024 - 1024 * 1024:
                    raise ValueError("ViT output exceeds the unary RPC size limit")
                if cached is None:
                    future = self._scheduler.submit(image, deadline, cancelled)
                else:
                    future = Future()
                    future.set_result(BatchResult(cached, 0, 0))
                futures.append(future)
                cache_keys.append(key)
            embeddings, positions, batch_ids = [], [], set()
            max_batch_size = 0
            for key, future in zip(cache_keys, futures):
                # Poll cancellation without holding the GPU executor or a lock.
                while not future.done():
                    self._check_request(deadline, cancelled)
                    wait((future,), timeout=0.05)
                self._check_request(deadline, cancelled)
                result = future.result()
                embedding, pos = result.output
                if result.batch_id:
                    self._embedding_cache.insert_cache(key, result.output)
                embeddings.append(embedding)
                if self.contains_pos:
                    check_with_info(pos is not None, "pos_ids should not be None")
                    positions.append(pos)
                if result.batch_id:
                    batch_ids.add(result.batch_id)
                max_batch_size = max(max_batch_size, result.batch_size)
            return MMEmbeddingRes(
                embeddings,
                positions if self.contains_pos else None,
                max_batch_size,
                len(batch_ids),
            )
        except BaseException:
            cancelled.set()
            raise

    def stop(self):
        if self._scheduler is not None:
            self._scheduler.close()

    def _maybe_tensor_to_list(self, tensor: torch.Tensor):
        if len(tensor.shape) > 2:
            return list(tensor)
        else:
            return [tensor]

    def submit(
        self,
        urls: List[str],
        types: Optional[List[MMUrlType]] = None,
        tensors: Optional[List[torch.Tensor]] = None,
        preprocess_configs: Optional[List[List[int]]] = None,
        deadline: Optional[float] = None,
        cancelled: Optional[threading.Event] = None,
    ):
        if self._scheduler is not None:
            types = types or [MMUrlType.DEFAULT] * len(urls)
            configs = (
                [MMPreprocessConfig(*config) for config in preprocess_configs]
                if preprocess_configs
                else [MMPreprocessConfig() for _ in urls]
            )
            if len(types) != len(urls) or len(configs) != len(urls):
                raise ValueError("ViT URL/type/config counts do not match")
            return self._submit_scheduled(urls, types, configs, deadline, cancelled)
        self._check_request(deadline, cancelled)
        if self.run_batch:
            with Timer() as route_timer:
                res, pos = self.model.mm_part.mm_embedding(
                    urls=urls, mm_types=types, tensors=tensors
                )
            kmonitor.report(
                GaugeMetrics.VIT_PREPROCESS_RT_METRIC, route_timer.cost_ms()
            )
            return MMEmbeddingRes(res, pos)
        if types is None or len(types) == 0:
            types = [MMUrlType.DEFAULT] * len(urls)
        if preprocess_configs is None or len(preprocess_configs) == 0:
            configs = [MMPreprocessConfig()] * len(urls)
        else:
            configs = [MMPreprocessConfig(*config) for config in preprocess_configs]
        try:
            res: List[torch.Tensor] = []
            pos: Optional[List[torch.Tensor]] = [] if self.contains_pos else None
            for index in range(len(urls)):
                self._check_request(deadline, cancelled)
                embedding, pos_ids = self.model.mm_part.mm_embedding(
                    url=urls[index],
                    mm_type=types[index],
                    download_headers=self.download_headers,
                    configs=configs[index],
                )
                res.extend(self._maybe_tensor_to_list(embedding))
                if self.contains_pos:
                    check_with_info(pos_ids is not None, "pos_ids should not be None")
                    pos.extend(self._maybe_tensor_to_list(pos_ids))
            return MMEmbeddingRes(res, pos)
        except Exception as e:
            logging.exception("Exception in MMProcessEngine.submit:")
            torch.cuda.empty_cache()
            gc.collect()
            raise e
