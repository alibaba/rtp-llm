import os
import torch
import json
import requests
import asyncio
import threading
# import torchaudio
from PIL import Image
from decord import VideoReader, cpu
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
    
def common_image_process_func(url: str, handler_func: Callable[[Any], Any]) -> torch.Tensor:
    with cache_lock:
        if url in mm_data_cache:
            return mm_data_cache[url]
    try:
        bytes_io = get_bytes_io_from_url(url)
        image = Image.open(bytes_io)
    except Exception as e:
        raise Exception(f"cannot download image from {url}, exception {e}")
    image_feature = handler_func([image.convert("RGB")])[0]
    with cache_lock:
        mm_data_cache[url] = image_feature
    return image_feature

def common_audio_process_func(url: str, handler_func: Callable[[Any, Any], Any]) -> torch.Tensor:
    return None
    # todo
    with cache_lock:
        if url in mm_data_cache:
            return mm_data_cache[url]
    try:
        bytes_io = get_bytes_io_from_url(url)
        audio, sample_rate = torchaudio.open(bytes_io)
    except Exception as e:
        raise Exception(f"cannot download audio from {url}, exception {e}")
    audio_feature = handler_func(audio, sample_rate)
    with cache_lock:
        mm_data_cache[url] = audio_feature
    return audio_feature

def common_viedo_process_func(url: str, handler_func: Callable[[Any], Any]):
    with cache_lock:
        if url in mm_data_cache:
            return mm_data_cache[url]
    bytes_io = get_bytes_io_from_url(url)
    num_frm = 64
    vr = VideoReader(bytes_io, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = self._get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    h, w = 336, 336
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    video_feature = handler_func(clip_imgs)
    with cache_lock:
        mm_data_cache[url] = video_feature
    
    return video_feature