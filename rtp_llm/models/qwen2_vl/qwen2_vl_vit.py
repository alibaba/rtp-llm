import math
from typing import Any, List, Tuple

from PIL import Image

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from rtp_llm.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from rtp_llm.multimodal.multimodal_common import (
    MultiModalEmbeddingInterface,
    timeout_decorator,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class Qwen2VLImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, config: ModelConfig):
        self.data_type = config.compute_dtype
        self.mm_related_params = config.mm_related_params
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            config.mm_related_params.config["ckpt_path"]
        )

        self.visual = Qwen2VisionTransformerPretrainedModel(
            config.mm_related_params.config
        ).share_memory()
        self.spatial_merge_size = config.mm_related_params.config.get(
            "spatial_merge_size", 2
        )

    @property
    def _device(self):
        return self.visual.get_device()

    @staticmethod
    def load_image(data, configs, **kwargs):
        image = Image.open(data).convert("RGB")
        size_factor = IMAGE_FACTOR
        if configs.height != -1 and configs.width != -1:
            resized_height, resized_width = smart_resize(
                configs.height,
                configs.width,
                factor=size_factor,
            )
        else:
            width, height = image.size
            min_pixels = MIN_PIXELS if configs.min_pixels == -1 else configs.min_pixels
            max_pixels = MAX_PIXELS if configs.max_pixels == -1 else configs.max_pixels
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        image = image.resize((resized_width, resized_height))
        return image

    @staticmethod
    def load_video(data, configs, **kwargs):
        vr = VideoReader(data, ctx=cpu(0), num_threads=1)
        frames = len(vr)

        fps = FPS if configs.fps == -1 else configs.fps
        size_factor = FRAME_FACTOR
        nframes = frames / vr.get_avg_fps() * fps
        nframes = round_by_factor(nframes, size_factor)
        min_frames = FPS_MIN_FRAMES if configs.min_frames == -1 else configs.min_frames
        if nframes < min_frames:
            nframes = ceil_by_factor(min_frames, size_factor)
        max_frames = FPS_MAX_FRAMES if configs.max_frames == -1 else configs.max_frames
        if nframes > max_frames:
            nframes = floor_by_factor(max_frames, size_factor)

        idx = torch.linspace(0, frames - 1, nframes).round().long().tolist()
        height, width = vr[0].shape[:2]
        video = torch.tensor(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
        del vr

        if configs.height != -1 and configs.width != -1:
            resized_height, resized_width = smart_resize(
                configs.height,
                configs.width,
                factor=size_factor,
            )
        else:
            min_pixels = (
                VIDEO_MIN_PIXELS if configs.min_pixels == -1 else configs.min_pixels
            )
            total_pixels = (
                VIDEO_TOTAL_PIXELS if configs.max_pixels == -1 else configs.max_pixels
            )
            max_pixels = max(
                min(VIDEO_MAX_PIXELS, total_pixels / nframes * size_factor),
                min_pixels * 1.05,
            )
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video

    def get_position_ids(self, grid_thw: torch.Tensor = None) -> torch.Tensor:
        spatial_merge_size = self.spatial_merge_size

        t, h, w = (
            grid_thw[0][0].item(),
            grid_thw[0][1].item() // spatial_merge_size,
            grid_thw[0][2].item() // spatial_merge_size,
        )

        t_index = (
            torch.arange(t, dtype=torch.int32).view(-1, 1).expand(-1, h * w).flatten()
        )
        h_index = (
            torch.arange(h, dtype=torch.int32).view(1, -1, 1).expand(t, -1, w).flatten()
        )
        w_index = (
            torch.arange(w, dtype=torch.int32).view(1, 1, -1).expand(t, h, -1).flatten()
        )

        return torch.stack([t_index, h_index, w_index], dim=1)

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
    ):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type
        data = get_bytes_io_from_url(mm_input.url, vit_config.download_headers)
        if mm_type == MMUrlType.DEFAULT or mm_type == MMUrlType.IMAGE:
            data = Qwen2VLImageEmbedding.load_image(data, mm_input.config)
            res = processor(images=data, videos=None, return_tensors="pt")
            return res["pixel_values"], res["image_grid_thw"]
        elif mm_type == MMUrlType.VIDEO:
            data = Qwen2VLImageEmbedding.load_video(data, mm_input.config)
            res = processor(images=None, videos=data, return_tensors="pt")
            return res["pixel_values_videos"], res["video_grid_thw"]

    def get_preprocess_params(self):
        return {
            "processor": self.image_processor,
        }

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device).to(self._data_type)
        grid_thw = data[1].to(self._device)
        embeddings = self.visual(pixel_values, grid_thw=grid_thw).to(self._device)
        pos_id = self.get_position_ids(grid_thw)
        return embeddings, pos_id
