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

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.worker_info import g_parallel_info
from rtp_llm.utils.multimodal_util import (
    MMPreprocessConfig,
    MMUrlType,
    get_bytes_io_from_url,
    get_vit_compute_dtype,
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
    # do not take GptInitModelParameters as class member, as it cannot be pickled
    def __init__(self, config: GptInitModelParameters):
        self.data_type = config.data_type

    @property
    def _data_type(self):
        return get_vit_compute_dtype(self.data_type)

    @property
    def _device(self):
        raise NotImplementedError

    @staticmethod
    def preprocess_input(
        url,
        mm_type: MMUrlType,
        tensor: torch.Tensor,
        config: MMPreprocessConfig,
        **kwargs,
    ):
        raise NotImplementedError

    def get_preprocess_params(self):
        return {}

    # embedding interface should be locked by mm_lock
    # todo: batch interface
    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        raise NotImplementedError


class ImageEmbeddingInterface(MultiModalEmbeddingInterface):
    @staticmethod
    def preprocess_input(
        url,
        mm_type: MMUrlType,
        tensor: torch.Tensor,
        config: MMPreprocessConfig,
        **kwargs,
    ):
        data = get_bytes_io_from_url(url)
        return Image.open(data).convert("RGB")


class AudioEmbeddingInterface(MultiModalEmbeddingInterface):
    @staticmethod
    def preprocess_input(
        url,
        mm_type: MMUrlType,
        tensor: torch.Tensor,
        config: MMPreprocessConfig,
        **kwargs,
    ):
        # temporary
        import torchaudio

        data = get_bytes_io_from_url(url)
        return torchaudio.load(data)


class VideoEmbeddingInterface(MultiModalEmbeddingInterface):
    @staticmethod
    def preprocess_input(
        url,
        mm_type: MMUrlType,
        tensor: torch.Tensor,
        config: MMPreprocessConfig,
        **kwargs,
    ):
        data = get_bytes_io_from_url(url)
        return VideoReader(data, ctx=cpu(0))
