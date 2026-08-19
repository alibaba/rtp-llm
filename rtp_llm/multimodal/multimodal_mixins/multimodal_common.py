from dataclasses import dataclass
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

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

from torchvision import transforms

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url, vit_emb_cache_
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType


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


@dataclass(frozen=True)
class MMWorkEstimate:
    """Model-provided cost for one or more preprocessed media items.

    The scheduler treats zero-valued budget fields as unconstrained. Models can
    therefore start with the dimensions they can estimate reliably while the
    generic fallback remains compatible with existing multimodal models.
    """

    input_patches: int = 0
    output_tokens: int = 0
    estimated_workspace_bytes: int = 0
    max_attention_segment: int = 0
    attention_work: int = 0

    def __post_init__(self) -> None:
        for field_name, value in vars(self).items():
            if value < 0:
                raise ValueError(
                    f"MMWorkEstimate.{field_name} must be >= 0, got {value}"
                )

    def __add__(self, other: "MMWorkEstimate") -> "MMWorkEstimate":
        if not isinstance(other, MMWorkEstimate):
            return NotImplemented
        return MMWorkEstimate(
            input_patches=self.input_patches + other.input_patches,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated_workspace_bytes=(
                self.estimated_workspace_bytes + other.estimated_workspace_bytes
            ),
            max_attention_segment=max(
                self.max_attention_segment, other.max_attention_segment
            ),
            attention_work=self.attention_work + other.attention_work,
        )

    def scaled(self, count: int) -> "MMWorkEstimate":
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        return MMWorkEstimate(
            input_patches=self.input_patches * count,
            output_tokens=self.output_tokens * count,
            estimated_workspace_bytes=self.estimated_workspace_bytes * count,
            # This is a maximum, not an additive quantity.
            max_attention_segment=self.max_attention_segment,
            attention_work=self.attention_work * count,
        )

    def fits_within(self, budget: "MMWorkEstimate") -> bool:
        additive_fields = (
            "input_patches",
            "output_tokens",
            "estimated_workspace_bytes",
            "attention_work",
        )
        for field_name in additive_fields:
            limit = getattr(budget, field_name)
            if limit > 0 and getattr(self, field_name) > limit:
                return False
        return (
            budget.max_attention_segment <= 0
            or self.max_attention_segment <= budget.max_attention_segment
        )


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

    def validate_inputs(self, mm_inputs: List[MultimodalInput]) -> None:
        """Validate request-level constraints before preprocessing starts."""

    def estimate_work(
        self, data: Any, mm_type: Optional[MMUrlType] = None
    ) -> Optional[MMWorkEstimate]:
        """Return exact post-preprocess work when the model can provide it."""
        return None

    def get_batch_work_budget(self, max_batch_media: int) -> Optional[MMWorkEstimate]:
        """Derive an internal cost budget from the existing media-count cap."""
        return None

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
        if VideoReader is None:
            raise ImportError(
                "decord is required for video processing. "
                "Install it with `pip install decord`."
            )
        assert len(mm_inputs) == 1
        data = get_bytes_io_from_url(mm_inputs[0].url, vit_config.download_headers)
        return VideoReader(data, ctx=cpu(0))
