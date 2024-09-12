import os
import torch
import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, Future

from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.utils.multimodal_util import MMUrlType

class MMEmbeddingRes:
    embeddings: List[torch.Tensor] = []
    position_ids: Optional[List[torch.Tensor]] = None

class MMProcessEngine:
    def __init__(self, model):
        self.model = model

    def submit(self, urls: List[str], types: Optional[List[MMUrlType]] = None):
        if types is None:
            types = [MMUrlType.DEFAULT] * len(urls)
        mm_res = MMEmbeddingRes()
        if self.model.config.mm_position_ids_style == 0:
            res = []
            for index in range(len(urls)):
                if os.environ.get('EXTRA_INPUT_IN_MM_EMBEDDING', '') == 'INDEX':
                    embedding = self.model.mm_part.mm_embedding(urls[index], types[index], self.model.device, to_torch_dtype(self.model.config.data_type), index=index)
                else:
                    embedding = self.model.mm_part.mm_embedding(urls[index], types[index], self.model.device, to_torch_dtype(self.model.config.data_type))
                if len(embedding.shape) > 2:
                    res.extend(list(embedding))
                else:
                    res.append(embedding)
            mm_res.embeddings = res
            mm_res.position_ids = None
            return mm_res
        else:
            mm_res = MMEmbeddingRes()
            mm_res.position_ids = []
            for index in range(len(urls)):
                embedding_res = self.model.mm_part.mm_embedding(urls[index], types[index], self.model.device, to_torch_dtype(self.model.config.data_type))
                res = []
                pos_id = []
                if len(embedding_res[0].shape) > 2:
                    res.extend(list(embedding_res[0]))
                else:
                    res.append(embedding_res[0])
                if len(embedding_res[1].shape) > 3:
                    pos_id.extend(list(embedding_res[1]))
                else:
                    pos_id.append(embedding_res[1])
            mm_res.embeddings = res
            mm_res.position_ids = pos_id
            return mm_res
                
