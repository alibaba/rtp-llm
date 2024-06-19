import os
import asyncio
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List
from maga_transformer.models.multimodal_mixin import MultiModalMixin

class VitEngine:
    def __init__(self):
        vit_concurrency = int(os.environ.get('VIT_CONCURRENCY', '0'))
        if vit_concurrency != 0:
            self.executor = ThreadPoolExecutor(max_workers = vit_concurrency)
        else:
            self.executor = ThreadPoolExecutor()

    def submit(self, urls: List[str], model: MultiModalMixin):
        return [asyncio.wrap_future(self.executor.submit(model.process_multimodel_input_func, url)) for url in urls]

    @staticmethod
    async def get(futures: List[Future[Image.Image]], time_out: int = 1000) -> List[Image.Image]:
        result = []

        for future in futures:
            try:
                image = await asyncio.wait_for(future, timeout = time_out)
                result.append(image)
            except Exception as e:
                raise e
        return result