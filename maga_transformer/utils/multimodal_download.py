from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import os
import requests
from PIL import Image
import asyncio
from typing import Any, List, Dict, Optional

def download_image(url: str):
    try:
        if url.startswith("http://") or url.startswith("https://"):
            if os.environ.get("IMAGE_RESIZE_SUFFIX", "") != "" and "picasso" in url:
                url += os.environ.get("IMAGE_RESIZE_SUFFIX", "")
            return Image.open(requests.get(url, stream=True).raw)
        else:
            return Image.open(url)
    except Exception as e:
        raise Exception(f"cannot download image from {url}, exception {e}")

class DownloadEngine:
    def __init__(self, thread_num: Optional[int] = None):
        self.executor = ThreadPoolExecutor(max_workers = thread_num)
    
    def submit(self, urls: List[str]) -> List[Future[Image.Image]]:
        return [self.executor.submit(download_image, url) for url in urls]

    @staticmethod
    async def get(futures: List[Future[Image.Image]]) -> List[Image.Image]:
        result = []

        asyncio_futures = [asyncio.wrap_future(future) for future in futures]

        for future in asyncio_futures:
            try:
                image = await future
                result.append(image.convert("RGB"))
            except Exception as e:
                raise e

        return result
