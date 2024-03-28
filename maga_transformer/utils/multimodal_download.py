from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import os
import requests
from PIL import Image
import asyncio
from typing import Any, List, Dict, Optional

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
}

def download_image(url: str):
    try:
        if url.startswith("http://") or url.startswith("https://"):
            if os.environ.get("IMAGE_RESIZE_SUFFIX", "") != "" and "picasso" in url:
                url += os.environ.get("IMAGE_RESIZE_SUFFIX", "")
            return Image.open(requests.get(url, stream=True, headers=headers).raw)
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
    async def get(futures: List[Future[Image.Image]], time_out: int = 10) -> List[Image.Image]:
        result = []

        asyncio_futures = [asyncio.wrap_future(future) for future in futures]

        for future in asyncio_futures:
            try:
                image = await asyncio.wait_for(future, timeout = time_out)
                result.append(image.convert("RGB"))
            except Exception as e:
                raise e

        return result
