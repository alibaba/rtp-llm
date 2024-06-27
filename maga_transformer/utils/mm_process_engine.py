import os
import torch
import asyncio
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List
from maga_transformer.models.multimodal_mixin import MultiModalMixin

class MMProcessEngine:
    def __init__(self):
        mm_concurrency = int(os.environ.get('MM_CONCURRENCY', '0'))
        if mm_concurrency != 0:
            self.executor = ThreadPoolExecutor(max_workers = mm_concurrency)
        else:
            self.executor = ThreadPoolExecutor()

    def submit(self, urls: List[str], model):
        return [asyncio.wrap_future(self.executor.submit(model.mm_part.mm_embedding, url, model.device)) for url in urls]

    @staticmethod
    async def get(futures: List[Future[torch.Tensor]], time_out: int = 10) -> List[torch.Tensor]:
        result = []

        for future in futures:
            try:
                embeddings = await asyncio.wait_for(future, timeout = time_out)
                result.append(embeddings)
            except Exception as e:
                raise e
        return result