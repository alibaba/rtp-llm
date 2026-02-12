import logging
from typing import List, Optional, Tuple, Any

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import VitParameters
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
    MultimodalInput,
    get_bytes_io_from_url,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.device import get_current_device
from rtp_llm.models_py.utils.arch import is_hip
from rtp_llm.utils.swizzle_utils import swizzle_tensor

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None

import math

import torch
import torch.library as tl
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

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
            data = Qwen2_VLImageEmbedding.load_image(data, mm_input.config)
            res = processor(images=data, videos=None, return_tensors="pt")
            return res["pixel_values"], res["image_grid_thw"]
        elif mm_type == MMUrlType.VIDEO:
            data = Qwen2_5_VLImageEmbedding.load_video(data, mm_input.config)
            res = processor(images=None, videos=data, return_tensors="pt")
            return res["pixel_values_videos"], res["video_grid_thw"]
        else:
            raise Exception("unknown mm url type")


class Qwen2_5_VLMixin(Qwen2_VLMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen2_5_VLImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen2_VLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return Qwen2_5_VLImageEmbedding(mm_related_params).visual

    def load_mm_weight(self, ctype: str, device: str):
        # 先走框架默认灌权重逻辑：保持 modeling 文件“纯模型/纯 forward”，不要在 load 阶段改写 layout
        super().load_mm_weight(ctype=ctype, device=device)

        # 再在权重加载完成之后，按设备/配置做后置 patch（仅 ROCm 生效）
        self._patch_vit_attention_linears()

    def _get_hw_kernel_config(self):
        try:
            return get_current_device().py_env_configs.py_hw_kernel_config
        except Exception:
            return None

    @staticmethod
    def _can_swizzle_kn(weight_kn: torch.Tensor) -> bool:
        # swizzle_tensor(col_maj=False) 约束：m(=N) % 16 == 0 且 k(=K) % 32 == 0（fp16/bf16）
        if weight_kn.dim() != 2:
            return False
        k, n = weight_kn.shape
        return (n % 16 == 0) and (k % 32 == 0)

    def _maybe_swizzle_kn(
        self, weight_kn: torch.Tensor, hw_kernel_config
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if hw_kernel_config is None or not getattr(hw_kernel_config, "use_swizzleA", False):
            return weight_kn.t(), hw_kernel_config
        if not self._can_swizzle_kn(weight_kn):
            # 不能满足 swizzle 约束时，降级为 no-swizzle（避免选到 bpreshuffle=True 的实现导致 silent wrong）
            logging.warning(
                "[qwen2_5_vl] weight shape %s cannot swizzle, fallback to no-swizzle linear",
                tuple(weight_kn.shape),
            )
            return weight_kn.t(), None
        # Follow aiter's approach: transpose to (n,k), shuffle, then transpose back to (k,n)
        return swizzle_tensor(weight_kn, False, MiM=16).t(), hw_kernel_config

    def _patch_vit_attention_linears(self):
        # 仅 ROCm：CUDA/CPU 下不做替换，避免破坏权重布局或引入无必要的差异
        if not is_hip():
            return

        hw_kernel_config = self._get_hw_kernel_config()
        # 这里先屏蔽不开swizzle的情况，因为nn.linear和hipb_mm在不开swizzle的情况下，调用的是同一个hipblas gemm算子，耗时没变化
        if hw_kernel_config is None or not getattr(hw_kernel_config, "use_swizzleA", False):
            return

        if not hasattr(self, "mm_part") or not hasattr(self.mm_part, "visual"):
            return
        visual = self.mm_part.visual

        # 允许的 vision 来源：qwen2.5  + qwen3（来自 transformers）
        allowed_modules = (
            "rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.modeling_qwen2_5_vl",
            "transformers.models.qwen3_vl.modeling_qwen3_vl",
        )
        if not any(m in type(visual).__module__ for m in allowed_modules):
            return

        blocks = getattr(visual, "blocks", None)
        if blocks is None:
            return

        from rtp_llm.models_py.modules.factory.linear import LinearFactory

        for block in blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            for attr in ("qkv", "proj"):
                layer = getattr(attn, attr, None)
                if layer is None or not isinstance(layer, nn.Linear):
                    continue

                # nn.Linear.weight: (out_features, in_features) -> (k,n) for hipb_mm
                weight_kn = layer.weight.detach()
                bias = layer.bias.detach() if layer.bias is not None else None

                weight_kn, kernel_cfg = self._maybe_swizzle_kn(weight_kn, hw_kernel_config)
                new_linear = LinearFactory.create_linear(
                    weight=weight_kn,
                    bias=bias,
                    weight_scales=None,
                    quant_config=None,
                    hw_kernel_config=kernel_cfg,
                )
                setattr(attn, attr, new_linear)


register_multimodal_mixin(["qwen2_5_vl"], Qwen2_5_VLMixin)