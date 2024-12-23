import copy
import math
from typing import List, Any, Tuple, Dict
from PIL import Image
try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import torch
import torch.nn as nn

from maga_transformer.models.multimodal.multimodal_common import MultiModalEmbeddingInterface
from maga_transformer.utils.multimodal_util import MMUrlType, MMPreprocessConfig
from maga_transformer.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from maga_transformer.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

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
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
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
    def __init__(self, config: GptInitModelParameters):
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(config.mm_related_params.config["ckpt_path"])
        self.visual = Qwen2VisionTransformerPretrainedModel(config.mm_related_params.config)
        self.config = config

    @property
    def _device(self):
        return self.visual.get_device()

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        mm_type = kwargs.get("mm_type")
        if mm_type == MMUrlType.DEFAULT:
            raise Exception("cannot infer multimodal input type")
        elif mm_type == MMUrlType.IMAGE:
            return self.image_embedding(mm_input)
        elif mm_type == MMUrlType.VIDEO:
            return self.video_embedding(mm_input)
        else:
            raise Exception("unknown mm url type")
        
    def _mm_preprocess(self, data, **kwargs):
        mm_type = kwargs.get("mm_type")
        if mm_type == MMUrlType.DEFAULT:
            raise Exception("cannot infer multimodal input type")
        elif mm_type == MMUrlType.IMAGE:
            return self.load_image(data, **kwargs)
        elif mm_type == MMUrlType.VIDEO:
            return self.load_video(data, **kwargs)
        else:
            raise Exception("unknown mm url type")
    
    def load_image(self, data, configs, **kwargs):
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
        
    def load_video(self, data, configs, **kwargs):
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
            min_pixels = VIDEO_MIN_PIXELS if configs.min_pixels == -1 else configs.min_pixels
            total_pixels = VIDEO_TOTAL_PIXELS if configs.max_pixels == -1 else configs.max_pixels
            max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * size_factor), min_pixels * 1.05)
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
    
    def image_embedding(self, images, **kwargs):
        device = self._device
        image_inputs = self.image_processor(images=images, videos=None, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device).to(self._data_type)
        image_grid_thw = image_inputs["image_grid_thw"].to(device)
        # raise Exception(self.visual.get_dtype())
        embeddings = self.visual(pixel_values, grid_thw=image_grid_thw).to(device)
        pos_id = self.get_position_ids(image_grid_thw)
        return embeddings, pos_id
    
    def video_embedding(self, video, **kwargs):
        device = self._device
        videos_inputs = self.image_processor(images=None, videos=video, return_tensors="pt")
        pixel_values = videos_inputs["pixel_values_videos"].to(device).to(self._data_type)
        video_grid_thw = videos_inputs["video_grid_thw"].to(device)
        embeddings = self.visual(pixel_values, grid_thw=video_grid_thw).to(device)
        pos_id = self.get_position_ids(video_grid_thw)
        return embeddings, pos_id

    def get_position_ids(
        self,
        grid_thw: torch.Tensor = None
    ) -> torch.Tensor:
        spatial_merge_size = self.config.mm_related_params.config.get("spatial_merge_size", 2)
        
        t, h, w = (
            grid_thw[0][0].item(),
            grid_thw[0][1].item() // spatial_merge_size,
            grid_thw[0][2].item() // spatial_merge_size,
        )

        t_index = torch.arange(t, dtype=torch.int32).view(-1, 1).expand(-1, h * w).flatten()
        h_index = torch.arange(h, dtype=torch.int32).view(1, -1, 1).expand(t, -1, w).flatten()
        w_index = torch.arange(w, dtype=torch.int32).view(1, 1, -1).expand(t, h, -1).flatten()

        return torch.stack([t_index, h_index, w_index], dim = 1)