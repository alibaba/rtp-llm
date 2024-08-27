import os
import torch
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor, Future

from maga_transformer.utils.util import to_torch_dtype

class MMProcessEngine:
    def __init__(self, model):
        mm_concurrency = int(os.environ.get('MM_CONCURRENCY', '0'))
        self.model = model
        if mm_concurrency != 0:
            self.executor = ThreadPoolExecutor(max_workers = mm_concurrency)
        else:
            self.executor = ThreadPoolExecutor()

    def submit(self, urls: List[str]):
        if os.environ.get('EXTRA_INPUT_IN_MM_EMBEDDING', '') == 'INDEX':
            return [self.model.mm_part.mm_embedding(urls[index], self.model.device, to_torch_dtype(self.model.config.data_type), index=index) for index in range(len(urls))]
        else:
            return [self.model.mm_part.mm_embedding(url, self.model.device, to_torch_dtype(self.model.config.data_type)) for url in urls]