import os
import torch
import asyncio
import nest_asyncio

from concurrent.futures import ThreadPoolExecutor, Future
from typing import List

from maga_transformer.utils.util import to_torch_dtype

nest_asyncio.apply()

class MMProcessEngine:
    def __init__(self, model):
        mm_concurrency = int(os.environ.get('MM_CONCURRENCY', '0'))
        self.model = model
        if mm_concurrency != 0:
            self.executor = ThreadPoolExecutor(max_workers = mm_concurrency)
        else:
            self.executor = ThreadPoolExecutor()

    def submit(self, urls: List[str]):
        return [asyncio.wrap_future(self.executor.submit(self.model.mm_part.mm_embedding, url, self.model.device, to_torch_dtype(self.model.config.data_type))) for url in urls]

    @staticmethod
    async def get(futures: List[Future[torch.Tensor]], time_out: int = 100) -> List[torch.Tensor]:
        result = []

        for future in futures:
            try:
                embeddings = await asyncio.wait_for(future, timeout = time_out)
                result.append(embeddings)
            except Exception as e:
                raise e
        return result
