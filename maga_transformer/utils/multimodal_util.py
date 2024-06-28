import os
import torch
import json
import requests
import asyncio
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, List, Dict, Optional, Callable
from maga_transformer.utils.lru_dict import LruDict

mm_data_cache = LruDict(int(os.environ.get('MM_CACHE_ITEM_NUM', '10')))
cache_lock = threading.Lock()

if os.environ.get('DOWNLOAD_HEADERS', '') != '':
    HTTP_HEADS = json.loads(os.environ['DOWNLOAD_HEADERS'])
else:
    HTTP_HEADS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    }
    
def get_bytes_io_from_url(url: str):    
    if url.startswith("http") or url.startswith("https"):
        return BytesIO(requests.get(url, stream=True, headers=HTTP_HEADS).content)
    else:
        # treat url as local path
        with open(url, "rb") as fh:
            buf = BytesIO(fh.read())
        return buf
    
def check_cache(url: str):
    with cache_lock:
        if url in mm_data_cache:
            return mm_data_cache[url]
        else:
            return None

def insert_cache(url: str, features: torch.Tensor):
    with cache_lock:
        mm_data_cache[url] = features
