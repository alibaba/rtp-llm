import copy
import math
from typing import List, Any, Tuple, Dict
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import tempfile
import torch
import torch.nn as nn

from maga_transformer.models.multimodal.multimodal_common import MultiModalEmbeddingInterface
from maga_transformer.utils.multimodal_util import MMUrlType
from maga_transformer.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from maga_transformer.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

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
    def __init__(self, config: Dict[str, Any]):
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(config["ckpt_path"])
        self.visual = Qwen2VisionTransformerPretrainedModel(config)
        self.config = config

    @torch.inference_mode()
    def mm_process(self, mm_input, device, **kwargs):
        mm_type = kwargs.get("mm_type")
        if mm_type == MMUrlType.DEFAULT:
            raise Exception("cannot infer multimodal input type")
        elif mm_type == MMUrlType.IMAGE:
            return self.image_embedding(mm_input, device)
        elif mm_type == MMUrlType.VIDEO:
            return self.video_embedding(mm_input, device)
        else:
            raise Exception("unknown mm url type")
        
    def _mm_preprocess(self, data, **kwargs):
        mm_type = kwargs.get("mm_type")
        if mm_type == MMUrlType.DEFAULT:
            origin_data = copy.copy(data)
            try:
                return self.load_image(data)
            except Exception as e:
                try:
                    return self.load_video(origin_data)
                except Exception as e:
                    raise Exception(str(e))
        elif mm_type == MMUrlType.IMAGE:
            return self.load_image(data)
        elif mm_type == MMUrlType.VIDEO:
            return self.load_video(data)
        else:
            raise Exception("unknown mm url type")
    
    def load_image(self, data, **kwargs):
        image = Image.open(data).convert("RGB")
        width, height = image.size
        min_pixels = MIN_PIXELS
        max_pixels = MAX_PIXELS
        size_factor = IMAGE_FACTOR
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = image.resize((resized_width, resized_height))
        return image
        
    def load_video(self, data, **kwargs):
        with tempfile.NamedTemporaryFile() as tmpfile:
            tmpfile.write(data.getbuffer())
            tmpfile_name = tmpfile.name
            video, audio, info = io.read_video(
                tmpfile_name,
                start_pts=0.0,
                end_pts=None,
                pts_unit="sec",
                output_format="TCHW",
            )

        fps = FPS
        size_factor = FRAME_FACTOR
        nframes = video.size(0) / info["video_fps"] * fps
        nframes = round_by_factor(nframes, size_factor)
        min_frames = FPS_MIN_FRAMES
        if nframes < min_frames:
            nframes = ceil_by_factor(min_frames, size_factor)
        max_frames = FPS_MAX_FRAMES
        if nframes > max_frames:
            nframes = floor_by_factor(max_frames, size_factor)

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        height, width = video.shape[2:]
        video = video[idx]

        min_pixels = VIDEO_MIN_PIXELS
        total_pixels = VIDEO_TOTAL_PIXELS
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
    
    def image_embedding(self, images, device, **kwargs):
        image_inputs = self.image_processor(images=images, videos=None, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].half().cuda()
        image_grid_thw = image_inputs["image_grid_thw"].cuda()
        embeddings = self.visual(pixel_values, grid_thw=image_grid_thw).to(device)
        pos_id = self.get_position_ids(image_grid_thw)
        return embeddings, pos_id
    
    def video_embedding(self, video, device, **kwargs):
        videos_inputs = self.image_processor(images=None, videos=video, return_tensors="pt")
        pixel_values = videos_inputs["pixel_values_videos"].half().cuda()
        video_grid_thw = videos_inputs["video_grid_thw"].cuda()
        embeddings = self.visual(pixel_values, grid_thw=video_grid_thw).to(device)
        pos_id = self.get_position_ids(video_grid_thw)
        return embeddings, pos_id

    def get_position_ids(
        self,
        grid_thw: torch.Tensor = None
    ) -> torch.Tensor:
        spatial_merge_size = self.config.get("spatial_merge_size", 2)
        
        t, h, w = (
            grid_thw[0][0].item(),
            grid_thw[0][1].item() // spatial_merge_size,
            grid_thw[0][2].item() // spatial_merge_size,
        )

        t_index = torch.arange(t, dtype=torch.int32).view(-1, 1).expand(-1, h * w).flatten()
        h_index = torch.arange(h, dtype=torch.int32).view(1, -1, 1).expand(t, -1, w).flatten()
        w_index = torch.arange(w, dtype=torch.int32).view(1, 1, -1).expand(t, h, -1).flatten()

        return torch.stack([t_index, h_index, w_index], dim = 1)