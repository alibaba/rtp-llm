import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.library as tl
from PIL import Image
from transformers import AutoProcessor, Qwen2VLImageProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
    Qwen3_VLImageEmbedding,
    Qwen3_VLMixin,
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
        f"initialize flash_attn failed, exception {e}, using sdpa attention in qwen3_5 vl vit"
    )


class Qwen3_5MoeImageEmbedding(Qwen3_VLImageEmbedding):
    def __init__(self, mm_related_params: VitParameters):
        self.mm_processor = AutoProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        self.mm_processor.image_processor = Qwen2VLImageProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        config_hf = Qwen3_5MoeVisionConfig.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        config_hf._attn_implementation = default_attn_impl
        self.visual = Qwen3_5MoeVisionModel._from_config(config_hf)

    @property
    def _data_type(self):
        return self.visual.dtype

    @property
    def _device(self):
        return self.visual.device

    def get_preprocess_params(self):
        return {
            "processor": self.mm_processor,
            "factor": self.visual.spatial_merge_size * self.visual.patch_size,
        }

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device).to(self._data_type)
        grid_thw = data[1].to(self._device)
        vision_output = self.visual(
            pixel_values, grid_thw=grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        pos_id = self.get_position_ids(grid_thw)[0]
        return image_embeds[0].to(self._data_type), pos_id

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
        vision_output = self.visual(
            pixel_values, grid_thw=grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        pos_id = self.get_position_ids(grid_thw)
        for e, p in zip(image_embeds, pos_id):
            res_list.append((e.to(self._data_type), p))
        return res_list

    def get_position_ids(self, grid_thw: torch.Tensor = None) -> torch.Tensor:
        spatial_merge_size = self.visual.spatial_merge_size
        device = grid_thw.device
        dtype = torch.int32

        t_all = grid_thw[:, 0].to(dtype)
        h_all = (grid_thw[:, 1] // spatial_merge_size).to(dtype)
        w_all = (grid_thw[:, 2] // spatial_merge_size).to(dtype)

        pos_list = []
        for t, h, w in zip(t_all.tolist(), h_all.tolist(), w_all.tolist()):
            if t == 0 or h == 0 or w == 0:
                pos_list.append(torch.empty((0, 3), device=device, dtype=dtype))
                continue

            t_grid = (
                torch.arange(t, device=device, dtype=dtype)
                .view(t, 1, 1)
                .expand(-1, h, w)
            )
            h_grid = (
                torch.arange(h, device=device, dtype=dtype)
                .view(1, h, 1)
                .expand(t, -1, w)
            )
            w_grid = (
                torch.arange(w, device=device, dtype=dtype)
                .view(1, 1, w)
                .expand(t, h, -1)
            )

            pos = torch.stack(
                (t_grid.reshape(-1), h_grid.reshape(-1), w_grid.reshape(-1)), dim=1
            )
            pos_list.append(pos)

        return pos_list


class Qwen3_5MoeVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.visual."
        self._ft_prefix = "self.mm_part.visual."


class Qwen3_5MoeMixin(Qwen3_VLMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen3_5MoeImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen3_5MoeVitWeight(
            {"vit": self.mm_part.visual}
        )


register_multimodal_mixin(["qwen35_moe"], Qwen3_5MoeMixin)
register_multimodal_mixin(["qwen35_dense"], Qwen3_5MoeMixin)
