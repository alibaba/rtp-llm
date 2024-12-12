import os
import gc
import torch
import asyncio
from io import BytesIO
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future

from maga_transformer.utils.util import to_torch_dtype, check_with_info
from maga_transformer.utils.multimodal_util import (vit_emb_cache_,
                                                    get_bytes_io_from_url,
                                                    MMUrlType,
                                                    MMPreprocessConfig)

class MMEmbeddingRes:
    embeddings: List[torch.Tensor] = []
    position_ids: Optional[List[torch.Tensor]] = None

    def __init__(self, embeddings, position_ids = None):
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

    def submit(self, urls: List[str], types: Optional[List[MMUrlType]] = None, tensors: Optional[List[torch.Tensor]] = None, preprocess_configs: Optional[List[List[int]]] = None):
        if self.run_batch:
            res, pos = self.model.mm_part.mm_embedding(urls, types, tensors)
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
                embedding, pos_ids = self.model.mm_part.mm_embedding(urls[index], types[index], configs=configs[index])
                res.extend(self._maybe_tensor_to_list(embedding))
                if self.contains_pos:
                    check_with_info(pos_ids is not None, "pos_ids should not be None")
                    pos.extend(self._maybe_tensor_to_list(pos_ids))
            return MMEmbeddingRes(res, pos)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e
