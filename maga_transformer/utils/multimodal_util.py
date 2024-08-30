import os
import torch
import json
import requests
import threading
from enum import IntEnum
from io import BytesIO
from typing import Any, Callable, Optional
from maga_transformer.utils.lru_dict import LruDict
from maga_transformer.utils.oss_util import get_bytes_io_from_oss_path

if os.environ.get('DOWNLOAD_HEADERS', '') != '':
    HTTP_HEADS = json.loads(os.environ['DOWNLOAD_HEADERS'])
else:
    HTTP_HEADS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    }

class MMUrlType(IntEnum):
    DEFAULT = 0
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3
    
def get_bytes_io_from_url(url: str):    
    if url.startswith("http") or url.startswith("https"):
        return BytesIO(requests.get(url, stream=True, headers=HTTP_HEADS).content)
    elif url.startswith("oss"):
        return get_bytes_io_from_oss_path(url)
    else:
        # treat url as local path
        with open(url, "rb") as fh:
            buf = BytesIO(fh.read())
        return buf
    
class MMDataCache(object):
    def __init__(self):
        self.mm_data_cache: Optional[LruDict] = None
        self.cache_lock = threading.Lock()
        self.cache_size = int(os.environ.get('MM_CACHE_ITEM_NUM', '10'))
        if self.cache_size > 0:
            self.mm_data_cache = LruDict(self.cache_size)        
    
    def check_cache(self, url: str):
        if self.mm_data_cache == None:
            return None
        with self.cache_lock:
            if url in self.mm_data_cache:
                return self.mm_data_cache[url]
            else:
                return None
        
    def insert_cache(self, url: str, features: torch.Tensor):
        if self.mm_data_cache == None:
            return
        with self.cache_lock:
            self.mm_data_cache[url] = features

data_cache_ = MMDataCache()