from typing import List, Tuple, Union

import torch
from decord import VideoReader, cpu
from PIL import Image
from torchvision import transforms

from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.multimodal_util import (data_cache_,
                                                    get_bytes_io_from_url)


class ImageTransform:

    def __init__(self, image_size: int):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def encode(self, images: List[Image.Image], device: Union[str, torch.device], dtype: torch.dtype) -> torch.Tensor:
        tensor_images = torch.stack(
            [self.image_transform(image) for image in images], dim=0
        ).to(device=device).to(dtype=dtype)
        return tensor_images


class MultiModalEmbeddingInterface:
    @torch.inference_mode()
    def mm_embedding(self, url: str, device, dtype, **kwargs):
        if g_parallel_info.tp_rank > 0:
            return torch.Tensor([])
        cached_res = data_cache_.check_cache(url)
        if cached_res is None:
            try:
                bytes_io = get_bytes_io_from_url(url)
                mm_input = self._mm_preprocess(bytes_io)
            except Exception as e:
                raise Exception(f"cannot download image from {url}, exception {e}")
            features = self.mm_process(mm_input, device, **kwargs).to(dtype).contiguous()
            data_cache_.insert_cache(url, features)
            return features
        else:
            return cached_res

    def _mm_preprocess(self, data):
        raise NotImplementedError

    @torch.inference_mode()
    def mm_process(self, mm_input, device, **kwargs):
        raise NotImplementedError


class ImageEmbeddingInterface(MultiModalEmbeddingInterface):
    def _mm_preprocess(self, data):
        return Image.open(data).convert("RGB")

    @torch.inference_mode()
    def mm_process(self, mm_input, device, **kwargs):
        return self.image_embedding([mm_input], device)[0]

    @torch.inference_mode()
    def image_embedding(self, images: List[Image.Image], device):
        raise NotImplementedError()


class AudioEmbeddingInterface(MultiModalEmbeddingInterface):
    def _mm_preprocess(self, data):
        # temporary
        import torchaudio
        return torchaudio.load(data)

    @torch.inference_mode()
    def mm_process(self, mm_input, device, **kwargs):
        return self.audio_embedding(mm_input, device)

    @torch.inference_mode()
    def audio_embedding(self, audio: Tuple[torch.Tensor, int], device):
        raise NotImplementedError()

class VideoEmbeddingInterface(MultiModalEmbeddingInterface):
    def _mm_preprocess(self, data):
        return VideoReader(data, ctx=cpu(0))

    @torch.inference_mode()
    def mm_process(self, mm_input, device, **kwargs):
        return self.video_embedding(mm_input, device)

    @torch.inference_mode()
    def video_embedding(self, video: List[Image.Image], device):
        raise NotImplementedError()