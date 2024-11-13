import os
import torch
import asyncio
from io import BytesIO
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future

from maga_transformer.utils.util import to_torch_dtype
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

    def submit(self, urls: List[str], types: Optional[List[MMUrlType]] = None, preprocess_configs: Optional[List[List[int]]] = None):
        if types is None:
            types = [MMUrlType.DEFAULT] * len(urls)
        if preprocess_configs is None:
            configs = [MMPreprocessConfig()] * len(urls)
        else:
            configs = [MMPreprocessConfig(*config) for config in preprocess_configs]
        if self.model.config.mm_position_ids_style == 0:
            res = []
            for index in range(len(urls)):
                if os.environ.get('EXTRA_INPUT_IN_MM_EMBEDDING', '') == 'INDEX':
                    embedding = self.model.mm_part.mm_embedding(urls[index], types[index], configs=configs[index], index=index)
                else:
                    embedding = self.model.mm_part.mm_embedding(urls[index], types[index], configs=configs[index])
                if len(embedding.shape) > 2:
                    res.extend(list(embedding))
                else:
                    res.append(embedding)
            return MMEmbeddingRes(res)
        else:
            res = []
            pos_id = []
            for index in range(len(urls)):
                embedding_res = self.model.mm_part.mm_embedding(urls[index], types[index], configs=configs[index])
                if len(embedding_res[0].shape) > 2:
                    res.extend(list(embedding_res[0]))
                else:
                    res.append(embedding_res[0])
                # expect position id is [seq_len, id_width]
                if len(embedding_res[1].shape) > 2:
                    pos_id.extend(list(embedding_res[1]))
                else:
                    pos_id.append(embedding_res[1])
            return MMEmbeddingRes(res, pos_id)
                
