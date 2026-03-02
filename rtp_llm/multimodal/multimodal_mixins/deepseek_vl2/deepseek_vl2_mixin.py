import json
import os
from typing import Any, Dict, List, Tuple, Union

import timm
import torch
import torch.nn as nn
from einops import rearrange, repeat
from PIL import Image, ImageOps

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_vl2.deepseek_vl2_vit import (
    ImageTransform,
    MlpProjector,
    MlpProjectorConfig,
    VisionEncoderConfig,
    select_best_resolution,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    ImageEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters


class DeepSeekVLV2ImageEmbedding(ImageEmbeddingInterface):
    def __init__(
        self,
        mm_related_params: "VitParameters",
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        ignore_id: int = -100,
    ):
        self.mm_related_params = mm_related_params
        vision_config_dict = mm_related_params.config.get("vision_config", {})
        vision_config = VisionEncoderConfig(**vision_config_dict)
        self.vision_config = vision_config
        self.patch_size = vision_config.patch_size
        self.image_size = vision_config.image_size

        self.image_mean = image_mean
        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )

        self.vision = timm.create_model(
            "vit_so400m_patch14_siglip_384.webli",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        self.vision = self.vision.to(dtype=torch.get_default_dtype())

        projector_config_dict = mm_related_params.config.get("projector_config", {})
        projector_config = MlpProjectorConfig(**projector_config_dict)
        self.projector_config = projector_config
        self.projector = MlpProjector(projector_config)
        self.downsample_ratio = projector_config.downsample_ratio

        self.ignore_id = ignore_id

        self.tile_tag = mm_related_params.config.get("tile_tag", "2D")
        self.global_view_pos = mm_related_params.config.get("global_view_pos", "head")
        self.candidate_resolutions = mm_related_params.config.get(
            "candidate_resolutions", {}
        )

        # 用于format image token sequence的特殊token
        embed_std = 1 / torch.sqrt(
            torch.tensor(projector_config.n_embed, dtype=torch.float32)
        )
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = self.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}"
                )
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, projector_config.n_embed))
                * embed_std
            )
        else:
            raise ValueError(
                f"tile tag should be either 1D or 2D, but got {self.tile_tag}"
            )

    @property
    def _device(self):
        return next(self.vision.parameters()).device

    @property
    def _data_type(self):
        return next(self.vision.parameters()).dtype

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        candidate_resolutions: List[Tuple[int, int]],
        image_size: int,
        image_mean: Tuple[float, float, float],
        image_transform: ImageTransform,
    ):
        images_list = []
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        assert (
            mm_input.mm_type == MMUrlType.IMAGE or mm_input.mm_type == MMUrlType.DEFAULT
        )
        data = get_bytes_io_from_url(mm_input.url, vit_config.download_headers)
        image = Image.open(data).convert("RGB")
        best_width, best_height = select_best_resolution(
            image.size, candidate_resolutions
        )

        _width, best_height = select_best_resolution(image.size, candidate_resolutions)

        """process the global view"""
        global_view = ImageOps.pad(
            image,
            (image_size, image_size),
            color=tuple(int(x * 255) for x in image_mean),
        )
        images_list.append(image_transform(global_view))
        """process the local views"""
        local_view = ImageOps.pad(
            image,
            (best_width, best_height),
            color=tuple(int(x * 255) for x in image_mean),
        )

        """record height / width crop num"""
        num_width_tiles, num_height_tiles = (
            best_width // image_size,
            best_height // image_size,
        )

        for i in range(0, best_height, image_size):
            for j in range(0, best_width, image_size):
                images_list.append(
                    image_transform(
                        local_view.crop((j, i, j + image_size, i + image_size))
                    )
                )

        tensor_images = torch.stack(images_list, dim=0)
        return [tensor_images, num_width_tiles, num_height_tiles]

    def get_preprocess_params(self):
        return {
            "candidate_resolutions": self.candidate_resolutions,
            "image_size": self.image_size,
            "image_mean": self.image_mean,
            "image_transform": self.image_transform,
        }

    @torch.inference_mode()
    def embedding(self, data, mm_type: MMUrlType, **kwargs):
        tensor_images, num_width_tiles, num_height_tiles = data

        tensor_images = tensor_images.to(self._device).to(self._data_type)
        images_feature = self.vision.forward_features(tensor_images)
        images_embeds = self.projector(images_feature)

        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        num_tiles_in_image = num_width_tiles * num_height_tiles
        tile_index = 0
        # [hw, D]
        global_features = images_embeds[tile_index]

        # [num_height_tiles * num_width_tiles, hw, D]
        local_features = images_embeds[
            tile_index + 1 : tile_index + 1 + num_tiles_in_image
        ]

        # ----------------- global view add newline -----------------
        # [hw, D] -> [h, w, D]
        global_features = global_features.view(h, w, n_dim)
        # [D]     -> [h, 1, D]
        new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
        # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
        global_features = torch.cat([global_features, new_lines_in_global], dim=1)
        # [h, w + 1, D] -> [h * (w + 1), D]
        global_features = global_features.view(-1, n_dim)

        # ----------------- local view add newline -----------------
        # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
        local_features = rearrange(
            local_features,
            "(th tw) (h w) d -> (th h) (tw w) d",
            th=num_height_tiles,
            tw=num_width_tiles,
            h=h,
            w=w,
        )

        # [D] -> [num_height_tiles * h, 1, D]
        new_lines_in_local = repeat(
            self.image_newline, "d -> (th h) 1 d", th=num_height_tiles, h=h
        )

        # [num_height_tiles * h, num_width_tiles * w + 1, D]
        local_features = torch.cat([local_features, new_lines_in_local], dim=1)

        # [num_height_tiles * h, num_width_tiles * w + 1, D]
        #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
        local_features = local_features.view(-1, n_dim)

        # ----------------- merge global and local tiles -----------------
        if self.global_view_pos == "head":
            global_local_features = torch.cat(
                [global_features, self.view_seperator[None, :], local_features], dim=0
            )
        else:
            global_local_features = torch.cat(
                [local_features, self.view_seperator[None, :], global_features], dim=0
            )

        # Return a list of tensors, one per image, to match the interface
        return global_local_features, None


class DeepSeekVLV2VitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part."


class DeepSeekVLV2Mixin(BaseMultiModalMixin):
    # override
    def _init_multimodal(self):
        self.mm_part = DeepSeekVLV2ImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = DeepSeekVLV2VitWeight(
            {
                "vision": self.mm_part.vision,
                "projector": self.mm_part.projector,
            },
            True,
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return torch.nn.ModuleList(
            [
                DeepSeekVLV2ImageEmbedding(mm_related_params, vit_config).vision,
                DeepSeekVLV2ImageEmbedding(mm_related_params, vit_config).projector,
            ]
        )


register_multimodal_mixin("deepseek_vl_v2", DeepSeekVLV2Mixin)
