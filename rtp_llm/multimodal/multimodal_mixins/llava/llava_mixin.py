import copy
import logging
import math
import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.llava.llava_utils import (
    expand2square,
    get_anyres_image_grid_shape,
    process_anyres_image,
    unpad_image,
)
from rtp_llm.multimodal.multimodal_mixins.llava.llava_vit import (
    CLIPVisionTower,
    IdentityMap,
    SigLipVisionTower,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import VitParameters


class LlavaImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters, vit_config: VitConfig):
        self.mm_config = mm_related_params.config
        self.extra_data_path = vit_config.extra_data_path
        self.local_extra_data_path = vit_config.local_extra_data_path
        if mm_related_params.config.get("vision_config", None) != None:
            raise Exception("llava-hf style config is not implemented yet")
        else:
            self.vision_tower = self.build_vision_tower(self.mm_config)
        self.mm_projector = self.build_vision_projector(self.mm_config)
        if "unpad" in mm_related_params.config.get("mm_patch_merge_type", "flat"):
            self.image_newline = nn.Parameter(
                torch.empty(mm_related_params.config["hidden_size"])
            )

    @staticmethod
    def load_image(data, config):
        image = Image.open(data).convert("RGB")
        if config.width > 0 and config.height > 0:
            image = image.resize((config.width, config.height))
        return [image]

    @staticmethod
    def load_video(data, config):
        fps = 1 if config.fps == -1 else config.fps
        vr = VideoReader(data, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        frame_num = round(video_time * fps)
        # set frame num between 1 and 100
        max_frame_num = config.max_frames if config.max_frames != -1 else 100
        min_frame_num = config.min_frames if config.min_frames != -1 else 1

        frame_num = max(min_frame_num, min(max_frame_num, frame_num))

        frame_idx = np.linspace(0, total_frame_num - 1, frame_num).tolist()
        frame_idx = [int(idx) for idx in frame_idx]

        video = vr.get_batch(frame_idx).asnumpy()

        vr.seek(0)
        return [Image.fromarray(frame) for frame in video]

    @property
    def _data_type(self):
        return self.vision_tower.dtype

    @property
    def _device(self):
        return self.vision_tower.device

    def encode_images(self, images):
        if images.shape[0] == 0:
            return images
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat(
            (
                image_feature,
                self.image_newline[:, None, None]
                .expand(*image_feature.shape[:-1], 1)
                .to(image_feature.device),
            ),
            dim=-1,
        )
        if self.mm_config["add_faster_video"]:
            image_feature = image_feature.view(feature_dim, num_frames, resize_h, -1)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            image_feature = image_feature.flatten(1, 2)
            return image_feature
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.vision_tower.num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        mm_spatial_pool_mode = self.mm_config["mm_spatial_pool_mode"]
        if mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(
                image_feature, size=scaled_shape, mode="bilinear"
            )

        else:
            raise ValueError(
                f"Unexpected mm_spatial_pool_mode: {self.mm_config['mm_spatial_pool_mode']}"
            )
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def build_vision_tower(self, vision_tower_cfg: Dict[str, Any], **kwargs: Any):
        vision_tower_name = self.extra_data_path
        vision_tower = self.local_extra_data_path
        if vision_tower is None or vision_tower == "":
            vision_tower_name = vision_tower_cfg["vit_tower_path"]
            vision_tower = vision_tower_cfg["vit_tower_path"]
        if "siglip" in vision_tower_name:
            return SigLipVisionTower(
                vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs
            )
        else:
            return CLIPVisionTower(
                vision_tower,
                select_layer=vision_tower_cfg.get("mm_vision_select_layer", -2),
                select_feature=vision_tower_cfg.get(
                    "mm_vision_select_feature", "patch"
                ),
                **kwargs,
            )

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature = torch.cat(
            (
                image_feature,
                self.image_newline[:, None, None]
                .expand(*image_feature.shape[:-1], 1)
                .to(image_feature.device),
            ),
            dim=-1,
        )
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def build_vision_projector(self, config, delay_load=False, **kwargs):
        projector_type = config.get("mm_projector_type", "linear")

        if projector_type == "linear":
            return torch.nn.Linear(config["mm_hidden_size"], config["hidden_size"])

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [torch.nn.Linear(config["mm_hidden_size"], config["hidden_size"])]
            for _ in range(1, mlp_depth):
                modules.append(torch.nn.GELU())
                modules.append(
                    torch.nn.Linear(config["hidden_size"], config["hidden_size"])
                )
            return torch.nn.Sequential(*modules)

        if projector_type == "identity":
            return IdentityMap()

        raise ValueError(f"Unknown projector type: {projector_type}")

    @staticmethod
    def load_from_bytes(data, mm_type, config):
        load_data = None
        if mm_type == MMUrlType.DEFAULT:
            origin_data = copy.copy(data)
            try:
                load_data = LlavaImageEmbedding.load_image(data, config)
            except Exception as e:
                try:
                    load_data = LlavaImageEmbedding.load_video(origin_data, config)
                except Exception as e:
                    raise Exception(str(e))
        elif mm_type == MMUrlType.IMAGE:
            load_data = LlavaImageEmbedding.load_image(data, config)
        elif mm_type == MMUrlType.VIDEO:
            load_data = LlavaImageEmbedding.load_video(data, config)
        else:
            raise Exception("unknown mm url type")
        return load_data

    @staticmethod
    def preprocess(
        load_data,
        mm_type: MMUrlType,
        tensor,
        config: MMPreprocessConfig,
        processor,
        image_aspect_ratio: str,
        image_grid_pinpoints: List[Any],
    ):
        image_sizes = [image.size for image in load_data]

        if mm_type == MMUrlType.VIDEO:
            return (
                processor.preprocess(load_data, return_tensors="pt")["pixel_values"],
                image_sizes,
            )

        new_images = []
        if image_aspect_ratio == "pad":
            for image in load_data:
                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                new_images.append(image)
        elif "anyres" in image_aspect_ratio:
            for image in load_data:
                image = process_anyres_image(image, processor, image_grid_pinpoints)
                new_images.append(image)
        else:
            return (
                processor.preprocess(load_data, return_tensors="pt")["pixel_values"],
                image_sizes,
            )
        return new_images, image_sizes

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
        image_aspect_ratio: str,
        image_grid_pinpoints: List[Any],
    ):
        assert len(mm_inputs) == 1
        data = get_bytes_io_from_url(mm_inputs[0].url, vit_config.download_headers)
        load_data = LlavaImageEmbedding.load_from_bytes(
            data, mm_inputs[0].mm_type, mm_inputs[0].config
        )
        return LlavaImageEmbedding.preprocess(
            load_data,
            mm_inputs[0].mm_type,
            mm_inputs[0].tensor,
            mm_inputs[0].config,
            processor,
            image_aspect_ratio,
            image_grid_pinpoints,
        )

    def get_preprocess_params(self):
        return {
            "processor": self.vision_tower.image_processor,
            "image_aspect_ratio": self.mm_config["image_aspect_ratio"],
            "image_grid_pinpoints": self.mm_config["image_grid_pinpoints"],
        }

    @torch.inference_mode()
    def embedding(self, data, mm_type: MMUrlType, **kwargs):
        processed_images = data[0]
        image_sizes = data[1]
        config = self.mm_config
        image_aspect_ratio = config["image_aspect_ratio"]
        mm_patch_merge_type = config.get("mm_patch_merge_type", "flat")
        mm_newline_position = config.get("mm_newline_position", "one_token")

        processed_images = [
            image.to(device=self._device, dtype=self._data_type)
            for image in processed_images
        ]
        processed_images = [
            image.unsqueeze(0) if image.ndim == 3 else image
            for image in processed_images
        ]
        split_sizes = [processed_image.shape[0] for processed_image in processed_images]
        processed_images = torch.cat(processed_images)
        image_features = self.encode_images(processed_images)
        image_features = list(torch.split(image_features, split_sizes, dim=0))

        if mm_type == MMUrlType.VIDEO:
            image_features = [self.get_2dPool(feature) for feature in image_features]

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if mm_type == MMUrlType.VIDEO:  # video operations
                    if mm_newline_position == "grid":
                        image_feature = self.add_token_per_grid(image_feature)
                        if self.mm_config["add_faster_video"]:
                            raise Exception("add_faster_video is not implemented")
                        new_image_features.append(image_feature)
                    elif mm_newline_position == "frame":
                        image_feature = self.add_token_per_frame(image_feature)
                        new_image_features.append(image_feature.flatten(0, 1))

                    elif mm_newline_position == "one_token":
                        # one-token
                        image_feature = image_feature.flatten(0, 1)
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.image_newline[None].to(image_feature.device),
                                ),
                                dim=0,
                            )
                        new_image_features.append(image_feature)
                    elif mm_newline_position == "no_token":
                        new_image_features.append(image_feature.flatten(0, 1))
                    else:
                        raise ValueError(
                            f"Unexpected mm_newline_position: {mm_newline_position}"
                        )

                elif image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.vision_tower.num_patches_per_side
                    assert height * width == base_image_feature.shape[0]

                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(
                            r"anyres_max_(\d+)", image_aspect_ratio
                        )
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(
                                matched_anyres_max_num_patches.group(1)
                            )

                    if (
                        image_aspect_ratio == "anyres"
                        or "anyres_max" in image_aspect_ratio
                    ):
                        try:
                            num_patch_width, num_patch_height = (
                                get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    config["image_grid_pinpoints"],
                                    self.vision_tower.config.image_size,
                                )
                            )
                        except Exception as e:
                            logging.error(
                                f"exception {str(e)}, set num_path_width and num_patch_height to 2"
                            )
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(
                            num_patch_height, num_patch_width, height, width, -1
                        )
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(
                            4, 0, 2, 1, 3
                        ).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif (
                        "unpad" in mm_patch_merge_type
                        and "anyres_max" in image_aspect_ratio
                        and matched_anyres_max_num_patches
                    ):
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(
                            4, 0, 2, 1, 3
                        ).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(
                            image_feature, image_sizes[image_idx]
                        )
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(
                                image_feature,
                                [int(h // times), int(w // times)],
                                mode="bilinear",
                            )[0]
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.image_newline[:, None, None]
                                .expand(*image_feature.shape[:-1], 1)
                                .to(image_feature.device),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(
                            4, 0, 2, 1, 3
                        ).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(
                            image_feature, image_sizes[image_idx]
                        )
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.image_newline[:, None, None]
                                .expand(*image_feature.shape[:-1], 1)
                                .to(image_feature.device),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(
                            0, 2, 1, 3, 4
                        ).contiguous()
                        image_feature = image_feature.flatten(0, 3)

                    if "nobase" in mm_patch_merge_type:
                        pass
                    else:
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0
                        )
                else:
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.image_newline[None].to(image_feature.device),
                            ),
                            dim=0,
                        )
                new_image_features.append(image_feature)
            image_features = new_image_features

        if mm_type == MMUrlType.VIDEO:
            return torch.cat(image_features), None
        return image_features, None


class LlavaMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        mm_related_params = self.mm_related_params

        self.mm_part = LlavaImageEmbedding(self.mm_related_params, self.vit_config)
        vit_weight_dict: Dict[str, Any] = {"mm_projector": self.mm_part.mm_projector}
        if mm_related_params.config.get(
            "unfreeze_mm_vision_tower", False
        ) or "mm_vision_tower" in mm_related_params.config.get("mm_tunable_parts", []):
            vit_weight_dict["vision_tower"] = self.mm_part.vision_tower
        if "unpad" in mm_related_params.config.get("mm_patch_merge_type", "flat"):
            vit_weight_dict["image_newline"] = self.mm_part.image_newline
        mm_related_params.vit_weights = BaseVitWeights(vit_weight_dict, True)

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return torch.nn.ModuleList(
            [
                LlavaImageEmbedding(mm_related_params, vit_config).vision_tower,
                LlavaImageEmbedding(mm_related_params, vit_config).mm_projector,
            ]
        )


register_multimodal_mixin(["llava"], LlavaMixin)
