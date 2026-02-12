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

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url, vit_emb_cache_
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
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
        raise NotImplementedError

    @property
    def _device(self):
        raise NotImplementedError

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        **kwargs,
    ):
        raise NotImplementedError

    def get_preprocess_params(self):
        return {}

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        raise NotImplementedError

    @torch.inference_mode()
    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ):
        res_list = []
        for data, mm_type in zip(data_list, mm_types):
            res_list.append(self.embedding(data, mm_type=mm_type, **kwargs))
        return res_list


class ImageEmbeddingInterface(MultiModalEmbeddingInterface):
    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        **kwargs,
    ):
        assert len(mm_inputs) == 1
        data = get_bytes_io_from_url(mm_inputs[0].url, vit_config.download_headers)
        return Image.open(data).convert("RGB")


class AudioEmbeddingInterface(MultiModalEmbeddingInterface):
    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        **kwargs,
    ):
        # temporary
        import torchaudio

        assert len(mm_inputs) == 1
        data = get_bytes_io_from_url(mm_inputs[0].url, vit_config.download_headers)
        return torchaudio.load(data)


class VideoEmbeddingInterface(MultiModalEmbeddingInterface):
    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        **kwargs,
    ):
        assert len(mm_inputs) == 1
        data = get_bytes_io_from_url(mm_inputs[0].url, vit_config.download_headers)
        return VideoReader(data, ctx=cpu(0))
