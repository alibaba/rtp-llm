import copy
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from rtp_llm.config.model_config import VitParameters
import torch
import torch.nn as nn
import torch.utils.checkpoint
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from rtp_llm.models.llava_utils import (
    expand2square,
    get_anyres_image_grid_shape,
    process_anyres_image,
    unpad_image,
)
from rtp_llm.models.multimodal.multimodal_common import (
    MultiModalEmbeddingInterface,
    timeout_decorator,
)
from rtp_llm.utils.multimodal_util import MMUrlType

try:
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput


class LlavaImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, mm_related_params: "VitParameters", model_config=None):
        self.mm_related_params = mm_related_params
        self.model_config = model_config  # Store model_config for build_vision_tower
        if mm_related_params.config.get("vision_config", None) != None:
            raise Exception("llava-hf style config is not implemented yet")
        else:
            self.vision_tower = self.build_vision_tower(mm_related_params.config)
        self.mm_projector = self.build_vision_projector(mm_related_params.config)
        if "unpad" in mm_related_params.config.get("mm_patch_merge_type", "flat"):
            self.image_newline = nn.Parameter(
                torch.empty(mm_related_params.config["hidden_size"])
            )

    @torch.inference_mode()
    def mm_process(self, mm_input, **kwargs):
        mm_type = kwargs.get("mm_type")
        if mm_type == MMUrlType.DEFAULT:
            if isinstance(mm_input, list):
                return torch.cat(self.image_embedding(mm_input))
            else:
                return self.image_embedding([mm_input])[0]
        elif mm_type == MMUrlType.IMAGE:
            if isinstance(mm_input, list):
                raise Exception("expect single image input, but get a list")
            return self.image_embedding([mm_input])[0]
        elif mm_type == MMUrlType.VIDEO:
            if not isinstance(mm_input, list):
                raise Exception("expect video input, but get a single image")
            return torch.cat(self.image_embedding(mm_input, MMUrlType.VIDEO))
        else:
            raise Exception("unknown mm url type")

    @timeout_decorator(30)
    def _mm_preprocess(self, data, **kwargs):
        mm_type = kwargs.get("mm_type")
        if mm_type == MMUrlType.DEFAULT:
            origin_data = copy.copy(data)
            try:
                return self.load_image(data, **kwargs)
            except Exception as e:
                try:
                    return self.load_video(origin_data, **kwargs)
                except Exception as e:
                    raise Exception(str(e))
        elif mm_type == MMUrlType.IMAGE:
            return self.load_image(data, **kwargs)
        elif mm_type == MMUrlType.VIDEO:
            return self.load_video(data, **kwargs)
        else:
            raise Exception("unknown mm url type")

    def load_image(self, data, configs, **kwargs):
        image = Image.open(data).convert("RGB")
        if configs.width > 0 and configs.height > 0:
            image = image.resize((configs.width, configs.height))
        return image

    def load_video(self, data, configs, **kwargs):
        fps = 1 if configs.fps == -1 else configs.fps
        vr = VideoReader(data, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        frame_num = round(video_time * fps)
        # set frame num between 1 and 100
        max_frame_num = configs.max_frames if configs.max_frames != -1 else 100
        min_frame_num = configs.min_frames if configs.min_frames != -1 else 1

        frame_num = max(min_frame_num, min(max_frame_num, frame_num))

        frame_idx = np.linspace(0, total_frame_num - 1, frame_num).tolist()
        frame_idx = [int(idx) for idx in frame_idx]

        video = vr.get_batch(frame_idx).asnumpy()

        num_frames_to_sample = num_frames = len(frame_idx)
        vr.seek(0)
        return [Image.fromarray(frame) for frame in video]

    @property
    def _device(self):
        return self.vision_tower.device

    @torch.no_grad()
    def image_embedding(self, images: List[Image.Image], mm_type=MMUrlType.IMAGE):
        config = self.mm_related_params.config
        image_aspect_ratio = config["image_aspect_ratio"]
        mm_patch_merge_type = config.get("mm_patch_merge_type", "flat")
        mm_newline_position = config.get("mm_newline_position", "one_token")

        processed_images = process_images(
            images,
            image_aspect_ratio,
            self.vision_tower.image_processor,
            self._device,
            self._data_type,
            mm_type,
            image_grid_pinpoints=config.get("image_grid_pinpoints", []),
        )

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
            image_sizes = [image.size for image in images]
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if mm_type == MMUrlType.VIDEO:  # video operations
                    if mm_newline_position == "grid":
                        image_feature = self.add_token_per_grid(image_feature)
                        if self.mm_related_params.config.get("add_faster_video", False):
                            raise Exception("add_faster_video is not implemented")
                            # faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            # concat_slow_fater_token = []
                            # for _ in range(image_feature.shape[0]):
                            #     if _ % self.config.faster_token_stride == 0:
                            #         concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            #     else:
                            #         concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            # image_feature = torch.cat(concat_slow_fater_token)
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

        return image_features

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
        if self.mm_related_params.config.get("add_faster_video", False):
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
        mm_spatial_pool_mode = self.mm_related_params.config.get(
            "mm_spatial_pool_mode", "none"
        )
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
                f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}"
            )
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def build_vision_tower(self, vision_tower_cfg: Dict[str, Any], **kwargs: Any):
        vision_tower_name = self.model_config.extra_data_path
        vision_tower = self.model_config.local_extra_data_path
        if vision_tower is None:
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

        raise ValueError(f"Unknown vision tower: {vision_tower}")

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


# ViT
class CLIPVisionTower(nn.Module):
    def __init__(
        self, vision_tower, select_layer=-2, select_feature="patch", delay_load=False
    ):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = select_layer
        self.select_feature = select_feature

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


# Projector
class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


def process_images(
    images,
    image_aspect_ratio,
    image_processor,
    device,
    data_type,
    mm_type=MMUrlType.IMAGE,
    **kwargs,
):
    if mm_type == MMUrlType.VIDEO:
        return image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"
        ].to(device, dtype=data_type)

    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
            image = image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            new_images.append(image)
    elif "anyres" in image_aspect_ratio:
        for image in images:
            image = process_anyres_image(
                image, image_processor, kwargs.get("image_grid_pinpoints", [])
            )
            new_images.append(image)
    else:
        return image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"
        ].to(device, dtype=data_type)

    if type(new_images) is list:
        new_images = [image.to(device, dtype=data_type) for image in new_images]
    else:
        new_images = new_images.to(device, dtype=data_type)

    return new_images


class SigLipImageProcessor:
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        crop_size = (
            crop_size if crop_size is not None else {"height": 384, "width": 384}
        )
        crop_size = get_size_dict(
            crop_size, default_to_square=True, param_name="crop_size"
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(
                resize,
                size=self.size,
                resample=self.resample,
                data_format=self.data_format,
            ),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(
                normalize,
                mean=self.image_mean,
                std=self.image_std,
                data_format=self.data_format,
            ),
            partial(
                to_channel_dimension_format,
                channel_dim=self.data_format,
                input_channel_dim=self.data_format,
            ),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            self.load_model()
        elif (
            hasattr(vision_tower_cfg, "mm_tunable_parts")
            and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts
        ):
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )

        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
