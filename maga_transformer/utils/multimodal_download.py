from concurrent.futures import ThreadPoolExecutor, as_completed 
import requests
import asyncio
from enum import Enum
from PIL import Image
from typing import Any, List, Dict, Optional
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.utils.time_util import current_time_ms

def download_image(url: str):
    try:
        if url.startswith("http://") or url.startswith("https://"):
            if os.environ.get("IMAGE_RESIZE_SUFFIX", "") != "" and "picasso" in url:
                url += os.environ.get("IMAGE_RESIZE_SUFFIX", "")
            return Image.open(requests.get(url, stream=True).raw)
        else:
            return Image.open(url)
    except:
        raise Exception(f"cannot download image from {url}")

class DownloadEngine:
    def __init__(self, thread_num: Optional[int] = None):
        self.executor = ThreadPoolExecutor(max_workers = thread_num)
    
    def submit(self, urls: List[str]) -> List[Any]:
        return [self.executor.submit(download_image, url) for url in urls]
        
    @staticmethod
    def get(futures) -> List[Image.Image]:
        result = []

        for future in as_completed(futures):
            try:
                result.append(future.result())
            except Exception as e:
                raise e

        return result
