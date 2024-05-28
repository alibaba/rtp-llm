from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import os
import requests
from PIL import Image
import asyncio
import json
import torch
import threading
from typing import Any, List, Dict, Optional
from maga_transformer.utils.lru_dict import LruDict

image_cache = LruDict(int(os.environ.get('VIT_CACHE_ITEM_NUM', '10')))
cache_lock = threading.Lock()

if os.environ.get('DOWNLOAD_HEADERS', '') != '':
    headers = json.loads(os.environ['DOWNLOAD_HEADERS'])
else:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    }

def process_image(url: str, embedding_func):
    if url in image_cache:
        return image_cache[url]
    try:
        if url.startswith("http://") or url.startswith("https://"):
            if os.environ.get("IMAGE_RESIZE_SUFFIX", "") != "" and "picasso" in url:
                url += os.environ.get("IMAGE_RESIZE_SUFFIX", "")
            image = Image.open(requests.get(url, stream=True, headers=headers).raw)
        else:
            image = Image.open(url)
    except Exception as e:
        raise Exception(f"cannot download image from {url}, exception {e}")
    image_feature = embedding_func([image.convert("RGB")])[0]
    with cache_lock:
        image_cache[url] = image_feature
    return image_feature

class VitEngine:
    def __init__(self):
        vit_concurrency = int(os.environ.get('VIT_CONCURRENCY', '0'))
        if vit_concurrency != 0:
            self.executor = ThreadPoolExecutor(max_workers = vit_concurrency)
        else:
            self.executor = ThreadPoolExecutor()

    def submit(self, urls: List[str], model):
        return [asyncio.wrap_future(self.executor.submit(process_image, url, lambda x: model.visual.image_embedding(x, model.device))) for url in urls]

    @staticmethod
    async def get(futures: List[Future[Image.Image]], time_out: int = 10) -> List[Image.Image]:
        result = []

        for future in futures:
            try:
                image = await asyncio.wait_for(future, timeout = time_out)
                result.append(image)
            except Exception as e:
                raise e
        return result
