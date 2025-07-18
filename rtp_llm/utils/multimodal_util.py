import base64
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO
from typing import Any, Callable, Optional

import requests
import torch
from PIL import Image

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.utils.lru_dict import LruDict
from rtp_llm.utils.oss_util import get_bytes_io_from_oss_path

logger = logging.getLogger(__name__)

REQUEST_GET = None


def request_get(url, headers):
    global REQUEST_GET
    if REQUEST_GET is None:
        try:
            from internal_source.rtp_llm.utils.ssrf_check import safe_request_get

            REQUEST_GET = safe_request_get
        except ImportError:
            REQUEST_GET = lambda url, headers: requests.get(
                url, stream=True, headers=headers, timeout=10
            )
    return REQUEST_GET(url, headers)


if StaticConfig.vit_config.download_headers != "":
    HTTP_HEADS = json.loads(StaticConfig.vit_config.download_headers)
else:
    HTTP_HEADS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    }

BASE64_PREFIX = "data:image/jpeg;base64,"
URL_CACHE_SIZE = StaticConfig.vit_config.url_cache_item_num
MM_CACHE_SIZE = StaticConfig.vit_config.mm_cache_item_num


def get_base64_prefix(s):
    match = re.match(r"^data:[^,]*;base64,", s)
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

    def __init__(
        self,
        url: str,
        mm_type: MMUrlType = MMUrlType.DEFAULT,
        config: MMPreprocessConfig = MMPreprocessConfig(),
        tensor: torch.Tensor = torch.empty(1),
    ):
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
    cached_res = url_data_cache_.check_cache(url)
    if cached_res is None:
        try:
            if url.startswith("http") or url.startswith("https"):
                response = request_get(url, HTTP_HEADS)
                if response.status_code == 200:
                    res = BytesIO(response.content)
                else:
                    raise Exception(
                        f"download failed, error code: {response.status_code}"
                    )
            elif url.startswith("oss"):
                res = get_bytes_io_from_oss_path(url)
            elif get_base64_prefix(url) > 0:
                res = BytesIO(base64.b64decode(url[get_base64_prefix(url) :]))
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


vit_emb_cache_ = MMDataCache(MM_CACHE_SIZE)
url_data_cache_ = MMDataCache(URL_CACHE_SIZE)
