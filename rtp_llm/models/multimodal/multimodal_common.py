from typing import Any, List, Tuple, Union

import torch

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from pillow_heif import register_heif_opener

register_heif_opener()

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

from torchvision import transforms

from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.utils.multimodal_util import (
    MMUrlType,
    get_bytes_io_from_url,
    vit_emb_cache_,
)


def timeout_decorator(timeout_sec):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_sec)
                except TimeoutError:
                    raise TimeoutError(f"Function '{func.__name__}' timed out")

        return wrapper

    return decorator


mm_lock = threading.Lock()


class ImageTransform:

    def __init__(self, image_size: int):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def encode(
        self,
        images: List[Image.Image],
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor_images = (
            torch.stack([self.image_transform(image) for image in images], dim=0)
            .to(device=device)
            .to(dtype=dtype)
        )
        return tensor_images


class MultiModalEmbeddingInterface:
    @property
    def _data_type(self):
        return self.config.compute_dtype

    @property
    def _device(self):
        raise NotImplementedError

    @torch.inference_mode()
    def mm_embedding(
        self, 
        url: str, 
        mm_type: MMUrlType, 
        download_headers: str = "",
        **kwargs: Any
    ):
        dtype = self._data_type
        if g_parallel_info.tp_rank > 0:
            return torch.Tensor([])

        cached_res = vit_emb_cache_.check_cache(url)
        if cached_res is not None:
            return cached_res
        bytes_io = get_bytes_io_from_url(url, download_headers=download_headers)
        mm_input = self._mm_preprocess(bytes_io, mm_type=mm_type, **kwargs)
        with mm_lock:
            features = self.mm_process(mm_input, mm_type=mm_type, **kwargs)
        if isinstance(features, tuple):
            features = (features[0].to(dtype).contiguous(), features[1].contiguous())
        else:
            features = (features.to(dtype).contiguous(), None)
        vit_emb_cache_.insert_cache(url, features)
        return features

    @timeout_decorator(10)
    def _mm_preprocess(self, data, **kwargs):
        raise NotImplementedError

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        raise NotImplementedError


class ImageEmbeddingInterface(MultiModalEmbeddingInterface):
    @timeout_decorator(30)
    def _mm_preprocess(self, data, **kwargs):
        return Image.open(data).convert("RGB")

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        return self.image_embedding([mm_input])[0]

    @torch.inference_mode()
    def image_embedding(self, images: List[Image.Image]):
        raise NotImplementedError()


class AudioEmbeddingInterface(MultiModalEmbeddingInterface):
    @timeout_decorator(30)
    def _mm_preprocess(self, data, **kwargs):
        # temporary
        import torchaudio

        return torchaudio.load(data)

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        return self.audio_embedding(mm_input)

    @torch.inference_mode()
    def audio_embedding(self, audio: Tuple[torch.Tensor, int]):
        raise NotImplementedError()


class VideoEmbeddingInterface(MultiModalEmbeddingInterface):
    @timeout_decorator(30)
    def _mm_preprocess(self, data, **kwargs):
        return VideoReader(data, ctx=cpu(0))

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        return self.video_embedding(mm_input)

    @torch.inference_mode()
    def video_embedding(self, video: List[Image.Image]):
        raise NotImplementedError()
