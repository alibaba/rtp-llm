from typing import List

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
    get_bytes_io_from_url,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

import math

import torch
import torch.library as tl
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin import (
    FPS,
    FPS_MAX_FRAMES,
    FPS_MIN_FRAMES,
    FRAME_FACTOR,
    IMAGE_FACTOR,
    VIDEO_MAX_PIXELS,
    VIDEO_MIN_PIXELS,
    VIDEO_TOTAL_PIXELS,
    Qwen2_VLImageEmbedding,
    Qwen2_VLMixin,
    Qwen2_VLVitWeight,
    Qwen2VLImageProcessor,
    ceil_by_factor,
    floor_by_factor,
    smart_resize,
)
from rtp_llm.utils.model_weight import CkptWeightInfo, identity, sp_id

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


class Qwen2_5_VLImageEmbedding(Qwen2_VLImageEmbedding):
    def __init__(self, mm_related_params: VitParameters):
        self.mm_related_params = mm_related_params
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        self.visual = Qwen2_5_VisionTransformerPretrainedModel(mm_related_params.config)
        self.spatial_merge_size = mm_related_params.config.get("spatial_merge_size", 2)

    @property
    def _data_type(self):
        return self.visual.get_dtype()

    @property
    def _device(self):
        return self.visual.get_device()

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
        if mm_type == MMUrlType.DEFAULT:
            raise Exception("cannot infer multimodal input type")
        elif mm_type == MMUrlType.IMAGE:
            data = Qwen2_VLImageEmbedding.load_image(
                data, mm_input.mm_preprocess_config
            )
            res = processor(images=data, videos=None, return_tensors="pt")
            return res["pixel_values"], res["image_grid_thw"]
        elif mm_type == MMUrlType.VIDEO:
            data = Qwen2_5_VLImageEmbedding.load_video(
                data, mm_input.mm_preprocess_config
            )
            res = processor(images=None, videos=data, return_tensors="pt")
            return res["pixel_values_videos"], res["video_grid_thw"]
        else:
            raise Exception("unknown mm url type")


class Qwen2_5_VLWeightInfo(BaseMultiModalDeployWeightInfo):
    def get_weight_info(self):
        weights = []
        weight_names = self.vit_weights.weight_names
        ckpt_prefix = self.vit_weights.ckpt_prefix

        for w in weight_names:
            if ".up_gate_proj." in w:
                gate_proj_name = ckpt_prefix + w.replace(
                    ".up_gate_proj.", ".gate_proj."
                )
                up_proj_name = ckpt_prefix + w.replace(".up_gate_proj.", ".up_proj.")

                weights.append(
                    CustomAtomicWeight(
                        w,
                        [
                            CkptWeightInfo(gate_proj_name, identity),
                            CkptWeightInfo(up_proj_name, identity),
                        ],
                        lambda ts: torch.cat(ts, dim=0).contiguous(),
                        split_func=sp_id,
                    )
                )
            else:
                w_name = ckpt_prefix + w
                weights.append(
                    CustomAtomicWeight(
                        w,
                        [CkptWeightInfo(w_name, identity)],
                        identity,
                        split_func=sp_id,
                    )
                )
        return ModelWeightInfo(layer_weights=[], weights=weights)


class Qwen2_5_VLMixin(Qwen2_VLMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen2_5_VLImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen2_VLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @staticmethod
    def get_multimodal_mixin_weight_info():
        return Qwen2_5_VLWeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return Qwen2_5_VLImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["qwen2_5_vl"], Qwen2_5_VLMixin)
