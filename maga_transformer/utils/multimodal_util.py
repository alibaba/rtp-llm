import os
import torch
import json
import requests
import threading
from enum import IntEnum
from io import BytesIO
from typing import Any, Callable, Optional
try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None
from PIL import Image

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

class MultimodalInput:
    url: str
    mm_type: MMUrlType

    def __init__(self, url: str, mm_type: MMUrlType=MMUrlType.DEFAULT):
        self.url = url
        self.mm_type = mm_type


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

    def insert_cache(self, url: str, features: torch.Tensor):
        if self.mm_data_cache == None:
            return
        with self.cache_lock:
            self.mm_data_cache[url] = features

vit_emb_cache_ = MMDataCache(int(os.environ.get('MM_CACHE_ITEM_NUM', '10')))
url_data_cache_ = MMDataCache(100)

def encode_video(video_path, max_num_frames: int = 32):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

def get_url_data_with_cache(url: str, type: MMUrlType):
    cached_res = url_data_cache_.check_cache(url)
    if cached_res is None:
        try:
            bytes_io = get_bytes_io_from_url(url)
            if type == MMUrlType.IMAGE:
                data = Image.open(bytes_io).convert("RGB")
            else:
                data = encode_video(bytes_io)
        except Exception as e:
            raise Exception(f"download and load {url} error, exception {e}")
        url_data_cache_.insert_cache(url, data)
        return data
    else:
        return cached_res
