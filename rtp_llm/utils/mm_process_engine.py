import os
from torch.profiler import profile, record_function, ProfilerActivity

import gc
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
    def __init__(self, model):
        self.model = model
        self.contains_pos: bool = self.model.config.mm_position_ids_style != 0
        self.run_batch: bool = self.model.config.vit_run_batch

    def _maybe_tensor_to_list(self, tensor: torch.Tensor):
        if len(tensor.shape) > 2:
            return list(tensor)
        else:
            return [tensor]

    def submit(
        self,
        *args,
        **kwargs,
    ):
        if not os.environ.get("MM_TRACE"):
            return self._submit(*args, **kwargs)
        if not hasattr(self, '_forward_call_count'):
            self._forward_call_count = 0
        else:
            self._forward_call_count += 1

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            record_shapes=True
        ) as prof:
            with record_function("MMProcessEngine.submit"):
                ret = self._submit(*args, **kwargs)

        trace_file_name = f"{os.environ.get('MM_TRACE')}_fwd{self._forward_call_count}.json"
        prof.export_chrome_trace(trace_file_name)
        return ret
    def _submit(
        self,
        urls: List[str],
        types: Optional[List[MMUrlType]] = None,
        tensors: Optional[List[torch.Tensor]] = None,
        preprocess_configs: Optional[List[List[int]]] = None,
    ):
        if self.run_batch:
            with Timer() as route_timer:
                res, pos = self.model.mm_part.mm_embedding(urls, types, tensors)
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
                    urls[index], types[index], configs=configs[index]
                )
                res.extend(self._maybe_tensor_to_list(embedding))
                if self.contains_pos:
                    check_with_info(pos_ids is not None, "pos_ids should not be None")
                    pos.extend(self._maybe_tensor_to_list(pos_ids))
            return MMEmbeddingRes(res, pos)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e
