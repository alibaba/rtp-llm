import logging
from typing import Any, List, Optional, Tuple, Union

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

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

from torchvision import transforms

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.models.multimodal.multimodal_util import (
    get_bytes_io_from_url,
    vit_emb_cache_,
)
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
    data_type: torch.dtype = torch.float16

    @property
    def _data_type(self):
        return self.data_type

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


class CustomMultiModalEmbeddingInterface(MultiModalEmbeddingInterface):
    @torch.inference_mode()
    def mm_embedding(
        self,
        url: Optional[Union[str, List[str]]] = None,  # Ignored
        mm_type: Union[MMUrlType, List[MMUrlType]] = MMUrlType.DEFAULT,
        data: Optional[Union[bytes, List[bytes]]] = None,
        tensors: Optional[List[torch.Tensor]] = None,
        configs: Optional[List[Any]] = None,
        **kwargs: Any,
    ):
        dtype = self._data_type
        if g_parallel_info.tp_rank > 0:
            return (torch.Tensor([]), None)

        if isinstance(mm_type, list):
            input_types = mm_type
        else:
            input_types = [mm_type]

        if data is not None:
            input_datas = data if isinstance(data, list) else [data]
        else:
            input_datas = [None] * len(input_types)

        if configs is None:
            configs = [None] * len(input_types)

        assert len(input_types) == len(
            input_datas
        ), f"mm_type and data length mismatch: {len(input_types)} vs {len(input_datas)}"

        mm_inputs = []
        for t, d, cfg in zip(input_types, input_datas, configs):
            mm_inputs.append(
                self._mm_preprocess(mm_type=t, data=d, config=cfg, **kwargs)
            )

        with mm_lock:
            features_batch = self.mm_process(mm_inputs, mm_type=mm_type, **kwargs)

        processed_results = []
        for feat in features_batch:
            if isinstance(feat, list):
                tensor = torch.cat(feat, dim=0)
            else:
                tensor = feat

            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(dtype).contiguous()
            processed_results.append(tensor)

        if isinstance(mm_type, list):
            return (processed_results, None)
        else:
            if not processed_results:
                return (torch.tensor([]).to(dtype), None)
            return (processed_results[0], None)

    @timeout_decorator(30)
    def _mm_preprocess(
        self, mm_type: MMUrlType, data: Optional[bytes] = None, **kwargs: Any
    ):
        if data is not None:
            return data

        return b""

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        return self.custom_modal_embedding(mm_input)

    @torch.inference_mode()
    def mm_preprocess(self, mm_input, **kwargs):
        return self.custom_modal_preprocess(mm_input)

    @torch.inference_mode()
    def custom_modal_embedding(self, batch_data: Any):
        raise NotImplementedError()
