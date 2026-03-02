import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.library as tl
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLVisionModel,
)

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.qwen2_5_vl_mixin import (
    Qwen2_5_VLImageEmbedding,
    Qwen2_5_VLMixin,
    smart_resize,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.flash_attn_utils import can_use_flash_attn

default_attn_impl = "sdpa"
try:
    if can_use_flash_attn():
        default_attn_impl = "flash_attention_2"
except Exception as e:
    logging.info(
        f"initialize flash_attn failed, exception {e}, using sdpa attention in qwen2.5 vl vit"
    )

if not hasattr(tl, "wrap_triton"):

    def wrap_triton(fn):
        return fn

    tl.wrap_triton = wrap_triton


class Qwen3_VLImageEmbedding(Qwen2_5_VLImageEmbedding):
    def __init__(self, mm_related_params: VitParameters):
        self.mm_processor = AutoProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        self.mm_processor.image_processor = Qwen2VLImageProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        config_hf = Qwen3VLConfig.from_pretrained(mm_related_params.config["ckpt_path"])
        config_hf.vision_config._attn_implementation = default_attn_impl
        self.visual = Qwen3VLVisionModel._from_config(config_hf.vision_config)
        self.spatial_merge_size = self.visual.spatial_merge_size

    @property
    def _data_type(self):
        return self.visual.dtype

    @property
    def _device(self):
        return self.visual.device

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
        factor: int = 32,
    ):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type
        do_resize = True
        if mm_type == MMUrlType.DEFAULT or mm_type == MMUrlType.IMAGE:
            image = Image.open(get_bytes_io_from_url(mm_input.url))
            if (
                mm_input.mm_preprocess_config.height != -1
                and mm_input.mm_preprocess_config.width != -1
            ):
                resized_height, resized_width = smart_resize(
                    mm_input.mm_preprocess_config.height,
                    mm_input.mm_preprocess_config.width,
                    factor=factor,
                )
                image = image.resize((resized_width, resized_height))
                do_resize = False
            elif (
                mm_input.mm_preprocess_config.max_pixels != -1
                or mm_input.mm_preprocess_config.min_pixels != -1
            ):
                width, height = image.size
                min_pixels = (
                    0
                    if mm_input.mm_preprocess_config.min_pixels == -1
                    else mm_input.mm_preprocess_config.min_pixels
                )
                max_pixels = (
                    0x7FFFFFFF
                    if mm_input.mm_preprocess_config.max_pixels == -1
                    else mm_input.mm_preprocess_config.max_pixels
                )
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                image = image.resize((resized_width, resized_height))
                do_resize = False
            res = processor.image_processor(
                image, return_tensors="pt", do_resize=do_resize
            )
            return res["pixel_values"], res["image_grid_thw"]
        elif mm_type == MMUrlType.VIDEO:
            video = Qwen3_VLImageEmbedding.load_video(
                get_bytes_io_from_url(mm_input.url), mm_input.mm_preprocess_config
            )
            res = processor.video_processor(video, return_tensors="pt", do_resize=True)
            return res["pixel_values_videos"], res["video_grid_thw"]
        else:
            raise Exception("unknown mm url type")

    def get_preprocess_params(self):
        return {
            "processor": self.mm_processor,
            "factor": self.spatial_merge_size * self.visual.patch_size,
        }

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device).to(self._data_type)
        grid_thw = data[1].to(self._device)
        embeds, deepstack_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        embeds = torch.split(embeds, split_sizes)
        pos_id = self.get_position_ids(grid_thw)[0]
        deepstack_embeds = torch.stack(deepstack_embeds).to(self._data_type)
        return embeds[0].to(self._data_type), pos_id, deepstack_embeds

    @torch.inference_mode()
    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ):
        if not all(mm_type == MMUrlType.IMAGE for mm_type in mm_types):
            return super().batched_embedding(data_list, mm_types, **kwargs)
        res_list = []
        pixel_values_list = []
        grid_thw_list = []
        for data, mm_type in zip(data_list, mm_types):
            pixel_values_list.append(data[0])
            grid_thw_list.append(data[1])
        pixel_values = (
            torch.concat(pixel_values_list, dim=0).to(self._device).to(self._data_type)
        )
        grid_thw = torch.concat(grid_thw_list, dim=0).to(self._device)
        embeds, deepstack_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        embeds = torch.split(embeds, split_sizes)
        pos_id = self.get_position_ids(grid_thw)
        deepstack_embeds = (
            torch.stack(deepstack_embeds).to(self._data_type).split(split_sizes, dim=1)
        )
        for e, p, d in zip(embeds, pos_id, deepstack_embeds):
            res_list.append((e.to(self._data_type), p, d))
        return res_list


class Qwen3VLVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.visual."
        self._ft_prefix = "self.mm_part.visual."


class Qwen3_VLMixin(Qwen2_5_VLMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen3_VLImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen3VLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return Qwen3_VLImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["qwen3_vl"], Qwen3_VLMixin)
register_multimodal_mixin(["qwen3_vl_moe"], Qwen3_VLMixin)
