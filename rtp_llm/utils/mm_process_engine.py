import gc
import logging
from typing import List, Optional

import torch

from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.utils.multimodal_util import MMPreprocessConfig, MMUrlType
from rtp_llm.utils.time_util import Timer
from rtp_llm.utils.util import check_with_info


class MMEmbeddingRes:
    embeddings: List[torch.Tensor] = []
    position_ids: Optional[List[torch.Tensor]] = None

    def __init__(self, embeddings, position_ids=None):
        self.embeddings = embeddings
        self.position_ids = position_ids


class MMProcessEngine:
    def __init__(self, model, vit_config):
        self.model = model
        self.vit_config = vit_config
        self.contains_pos: bool = self.model.model_config.mm_model_config.mm_position_ids_style != 0
        self.run_batch: bool = self.model.model_config.mm_related_params.support_batch
        self.download_headers = self.vit_config.download_headers

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
    ):
        if self.run_batch:
            with Timer() as route_timer:
                res, pos = self.model.mm_part.mm_embedding(urls=urls, mm_types=types, tensors=tensors)
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
                embedding, pos_ids = self.model.mm_part.mm_embedding(
                    url=urls[index], 
                    mm_type=types[index], 
                    download_headers=self.download_headers,
                    configs=configs[index]
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
