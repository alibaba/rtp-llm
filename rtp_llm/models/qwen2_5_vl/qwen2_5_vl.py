from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from rtp_llm.config.model_config import VitParameters
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen2_vl.qwen2_vl import QWen2_VL, QwenVL2VitWeight

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

from typing import List

import torch.library as tl

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)
from rtp_llm.models.qwen2_vl.qwen2_vl_vit import (
    FPS,
    FPS_MAX_FRAMES,
    FPS_MIN_FRAMES,
    FRAME_FACTOR,
    IMAGE_FACTOR,
    VIDEO_MAX_PIXELS,
    VIDEO_MIN_PIXELS,
    VIDEO_TOTAL_PIXELS,
    Qwen2VLImageEmbedding,
    Qwen2VLImageProcessor,
    ceil_by_factor,
    floor_by_factor,
    smart_resize,
    timeout_decorator,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)

if not hasattr(tl, "wrap_triton"):

    def wrap_triton(fn):
        return fn

    tl.wrap_triton = wrap_triton


def smart_nframes(configs, total_frames, video_fps) -> int:
    fps = configs.fps if configs.fps != -1 else FPS
    min_frames = ceil_by_factor(
        configs.min_frames if configs.min_frames != -1 else FPS_MIN_FRAMES, FRAME_FACTOR
    )
    max_frames = floor_by_factor(
        (
            configs.max_frames
            if configs.max_frames != -1
            else min(FPS_MAX_FRAMES, total_frames)
        ),
        FRAME_FACTOR,
    )
    nframes = total_frames / video_fps * fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


class Qwen2_5_VLImageEmbedding(Qwen2VLImageEmbedding):
    def __init__(self, config: ModelConfig):
        self.data_type = config.compute_dtype
        super().__init__(config)
        self.mm_related_params = config.mm_related_params
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            config.mm_related_params.config["ckpt_path"]
        )
        self.visual = Qwen2_5_VisionTransformerPretrainedModel(
            config.mm_related_params.config
        )
        self.spatial_merge_size = config.mm_related_params.config.get(
            "spatial_merge_size", 2
        )

    @staticmethod
    def load_video(data, configs, **kwargs):
        vr = VideoReader(data, ctx=cpu(0), num_threads=1)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        nframes = smart_nframes(configs, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        height, width = vr[0].shape[:2]

        video = torch.tensor(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
        del vr

        image_factor = IMAGE_FACTOR

        nframes, _, height, width = video.shape
        min_pixels = (
            configs.min_pixels if configs.min_pixels != -1 else VIDEO_MIN_PIXELS
        )
        total_pixels = VIDEO_TOTAL_PIXELS
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        max_pixels_supposed = (
            configs.max_pixels if configs.max_pixels != -1 else max_pixels
        )
        max_pixels = min(max_pixels_supposed, max_pixels)
        if configs.height != -1 and configs.width != -1:
            resized_height, resized_width = smart_resize(
                configs.height,
                configs.width,
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
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
            data = Qwen2_5_VLImageEmbedding.load_video(data, mm_input.config)
            res = processor(images=None, videos=data, return_tensors="pt")
            return res["pixel_values_videos"], res["video_grid_thw"]
        else:
            raise Exception("unknown mm url type")


class QWen2_5_VL(QWen2_VL):
    def _init_multimodal(
        self,
    ):
        # mm_related_params is in model_config, not mm_model_config
        self.mm_part = Qwen2_5_VLImageEmbedding(self.model_config)
        self.model_config.mm_related_params.vit_weights = QwenVL2VitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _get_mm_module(cls, config: ModelConfig):
        return Qwen2_5_VLImageEmbedding(config).visual


register_model("qwen2_5_vl", QWen2_5_VL, ["Qwen2_5_VLForConditionalGeneration"])
