import os
import re
import torch
import json
import requests
import threading
from enum import IntEnum
from io import BytesIO
from typing import Any, Callable, Optional
from PIL import Image
from dataclasses import dataclass, field
import hashlib
import base64
import logging

from rtp_llm.utils.lru_dict import LruDict
from rtp_llm.utils.oss_util import get_bytes_io_from_oss_path

logger = logging.get_logger(__name__)

SSRF_CHECKER = None

def safe_check_ssrf(url):
    global SSRF_CHECKER
    if SSRF_CHECKER is None:
        try:
            from internal_source.rtp_llm.utils.ssrf_check import check_ssrf
            SSRF_CHECKER = check_ssrf
        except ImportError:
            logger.info("SSRF check module not available, skipping")
            SSRF_CHECKER = lambda _: True 
    
    return SSRF_CHECKER(url)

if os.environ.get('DOWNLOAD_HEADERS', '') != '':
    HTTP_HEADS = json.loads(os.environ['DOWNLOAD_HEADERS'])
else:
    HTTP_HEADS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    }

BASE64_PREFIX = 'data:image/jpeg;base64,'
URL_CACHE_SIZE = int(os.environ.get('URL_CACHE_ITEM_NUM', '100'))
MM_CACHE_SIZE = int(os.environ.get('MM_CACHE_ITEM_NUM', '10'))

def get_base64_prefix(s):
    match = re.match(r'^data:[^,]*;base64,', s)
    if not match:
        return 0
    return match.end()

class MMUrlType(IntEnum):
    DEFAULT = 0
    IMAGE = 1
    VIDEO = 2
    AUDIO = 3
    TENSOR = 4

@dataclass
class MMPreprocessConfig:
    width: int = -1
    height: int = -1
    min_pixels: int = -1
    max_pixels: int = -1
    fps: int = -1
    min_frames: int = -1
    max_frames: int = -1

class MultimodalInput:
    url: str
    mm_type: MMUrlType
    config: MMPreprocessConfig
    tensor: torch.Tensor

    def __init__(self, url: str, mm_type: MMUrlType=MMUrlType.DEFAULT,
                 config: MMPreprocessConfig=MMPreprocessConfig(), tensor: torch.Tensor=torch.empty(1)):
        self.url = url
        self.mm_type = mm_type
        self.config = config
        self.tensor = tensor

def get_vit_compute_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    else:
        return torch.half




def get_bytes_io_from_url(url: str):
    ssrf_success = safe_check_ssrf(url)
    if not ssrf_success:    
        raise Exception(f"url ssrf check failed {url}")
    cached_res = url_data_cache_.check_cache(url)
    if cached_res is None:
        try:
            if url.startswith("http") or url.startswith("https"):
                response = requests.get(url, stream=True, headers=HTTP_HEADS, timeout=10, allow_redirects=False)
                if response.status_code == 200:
                    res = BytesIO(response.content)
                else:
                    raise Exception(f'download failed, error code: {response.status_code}')
            elif url.startswith("oss"):
                res = get_bytes_io_from_oss_path(url)
            elif get_base64_prefix(url) > 0:
                url = maybe_unhash_url(url)[get_base64_prefix(url):]
                res = BytesIO(base64.b64decode(url))
            else:
                # treat url as local path
                with open(url, "rb") as fh:
                    buf = BytesIO(fh.read())
                res = buf
        except Exception as e:
            raise Exception(f"download and load {url} error, exception {e}")
        url_data_cache_.insert_cache(url, res)
        return res
    else:
        cached_res.seek(0)
        return cached_res

class MMDataCache(object):
    def __init__(self, cache_size: int = 10):
        self.mm_data_cache: Optional[LruDict] = None
        self.cache_lock = threading.Lock()
        if cache_size > 0:
            self.mm_data_cache = LruDict(cache_size)

    def check_cache(self, url: str):
        if self.mm_data_cache == None:
            return None
        with self.cache_lock:
            if url in self.mm_data_cache:
                return self.mm_data_cache[url]
            else:
                return None

    def insert_cache(self, url: str, data):
        if self.mm_data_cache == None:
            return
        with self.cache_lock:
            self.mm_data_cache[url] = data

hashed_url_cache_ = MMDataCache(URL_CACHE_SIZE)

def maybe_hash_url(url: str):
    prefix_len = get_base64_prefix(url)
    if prefix_len > 0:
        hashed_url = url[prefix_len:]
        hashed_url = url[:prefix_len] + hashlib.sha512(hashed_url.encode('utf-8')).hexdigest()
        hashed_url_cache_.insert_cache(hashed_url, url)
        return hashed_url
    return url

def maybe_unhash_url(hashed_url: str):
    prefix_len = get_base64_prefix(hashed_url)
    if prefix_len > 0:
        url = hashed_url_cache_.check_cache(hashed_url)
        if url is None:
            return hashed_url
        return url
    return hashed_url

vit_emb_cache_ = MMDataCache(MM_CACHE_SIZE)
url_data_cache_ = MMDataCache(URL_CACHE_SIZE)
