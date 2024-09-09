import os
import torch
import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, Future

from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.utils.multimodal_util import MMUrlType

class MMProcessEngine:
    def __init__(self, model):
        self.model = model

    def submit(self, urls: List[str], types: Optional[List[MMUrlType]] = None):
        if types is None:
            types = [MMUrlType.DEFAULT] * len(urls)
        res = []
        for index in range(len(urls)):
            if os.environ.get('EXTRA_INPUT_IN_MM_EMBEDDING', '') == 'INDEX':
                embedding = self.model.mm_part.mm_embedding(urls[index], types[index], self.model.device, to_torch_dtype(self.model.config.data_type), index=index)
            else:
                embedding = self.model.mm_part.mm_embedding(urls[index], types[index], self.model.device, to_torch_dtype(self.model.config.data_type))
            if isinstance(embedding, list) or len(embedding.shape) > 2:
                res.extend(list(embedding))
            else:
                res.append(embedding)
        return res
