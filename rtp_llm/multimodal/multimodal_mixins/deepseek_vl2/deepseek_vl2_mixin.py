import math
from typing import List, Optional, Tuple

import torch
from einops import rearrange, repeat
from PIL import Image, ImageOps

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.models_py.new_models.deepseek_vl2.vision import (
    DeepSeekVLV2VisionModel,
    ImageTransform,
    load_deepseek_vl2_vision,
    select_best_resolution,
)
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
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
        vision_model: Optional[DeepSeekVLV2VisionModel] = None,
    ):
        self.mm_related_params = mm_related_params
        if vision_model is None:
            vision_model = DeepSeekVLV2VisionModel(
                mm_related_params.config,
                torch.get_default_dtype(),
            )
        self.vision_model = vision_model
        self.vision_config = vision_model.vision_config
        self.patch_size = self.vision_config.patch_size
        self.image_size = self.vision_config.image_size

        self.image_mean = image_mean
        self.image_transform = ImageTransform(
            mean=image_mean, std=image_std, normalize=normalize
        )

        self.vision = vision_model.vision
        self.projector = vision_model.projector
        self.projector_config = vision_model.projector_config
        self.downsample_ratio = self.projector_config.downsample_ratio

        self.ignore_id = ignore_id

        self.tile_tag = vision_model.tile_tag
        self.global_view_pos = vision_model.global_view_pos
        self.candidate_resolutions = mm_related_params.config.get(
            "candidate_resolutions", []
        )
        if not self.candidate_resolutions:
            raise ValueError("DeepSeek-VL2 candidate_resolutions must not be empty")
        for index, resolution in enumerate(self.candidate_resolutions):
            if (
                not isinstance(resolution, (list, tuple))
                or len(resolution) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in resolution
                )
            ):
                raise ValueError(
                    f"candidate_resolutions[{index}] must contain two "
                    "positive integers"
                )
            if any(value % self.image_size for value in resolution):
                raise ValueError(
                    f"candidate_resolutions[{index}]={resolution} must be "
                    f"divisible by image_size={self.image_size}"
                )
        self.image_newline = vision_model.image_newline
        self.view_seperator = vision_model.view_seperator

    @property
    def _device(self):
        return self.vision_model.device

    @property
    def _data_type(self):
        return self.vision_model.dtype

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
        if len(mm_inputs) != 1:
            raise ValueError(
                "DeepSeek-VL2 preprocessing expects exactly one image, "
                f"got {len(mm_inputs)}"
            )
        mm_input = mm_inputs[0]
        if mm_input.mm_type not in {MMUrlType.IMAGE, MMUrlType.DEFAULT}:
            raise ValueError(
                f"DeepSeek-VL2 supports image inputs only, got {mm_input.mm_type}"
            )
        data = get_bytes_io_from_url(
            mm_input.url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
        )
        image = Image.open(data).convert("RGB")
        best_width, best_height = select_best_resolution(
            image.size, candidate_resolutions
        )

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
        if mm_type not in {MMUrlType.IMAGE, MMUrlType.DEFAULT}:
            raise ValueError(f"DeepSeek-VL2 supports image inputs only, got {mm_type}")
        if not isinstance(data, (list, tuple)) or len(data) != 3:
            raise ValueError(
                "DeepSeek-VL2 preprocessed data must contain images, "
                "num_width_tiles, and num_height_tiles"
            )
        tensor_images, num_width_tiles, num_height_tiles = data
        if (
            isinstance(num_width_tiles, bool)
            or not isinstance(num_width_tiles, int)
            or num_width_tiles <= 0
            or isinstance(num_height_tiles, bool)
            or not isinstance(num_height_tiles, int)
            or num_height_tiles <= 0
        ):
            raise ValueError(
                "DeepSeek-VL2 tile counts must be positive integers, got "
                f"{num_width_tiles}x{num_height_tiles}"
            )

        tensor_images = tensor_images.to(device=self._device, dtype=self._data_type)
        expected_images = 1 + num_width_tiles * num_height_tiles
        if tensor_images.dim() != 4 or tensor_images.size(0) != expected_images:
            raise ValueError(
                "DeepSeek-VL2 image tiles must have shape "
                f"[{expected_images}, C, H, W], got {tuple(tensor_images.shape)}"
            )
        images_embeds = self.vision_model(tensor_images)

        _, hw, n_dim = images_embeds.shape
        h = w = math.isqrt(hw)
        if h * w != hw:
            raise ValueError(
                f"DeepSeek-VL2 projected vision tokens must form a square, got {hw}"
            )

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
        if self.use_new_loader:
            vision_model = load_deepseek_vl2_vision(
                vision_config=self.mm_related_params.config,
                model_path=self.ckpt_path,
                compute_dtype=self.compute_dtype,
                device=self.device,
            )
            self.mm_part = DeepSeekVLV2ImageEmbedding(
                self.mm_related_params,
                vision_model=vision_model,
            )
            self.mm_related_params.vit_weights = None
            return

        self.mm_part = DeepSeekVLV2ImageEmbedding(self.mm_related_params)
        # Include the nn.Parameter siblings (image_newline / view_seperator)
        # so load_mm_weight picks them up from the checkpoint;
        # otherwise they stay at their random-init values, silently producing
        # wrong embeddings at inference time.
        vit_parts = {
            "vision": self.mm_part.vision,
            "projector": self.mm_part.projector,
        }
        vit_parts["image_newline"] = self.mm_part.image_newline
        vit_parts["view_seperator"] = self.mm_part.view_seperator
        self.mm_related_params.vit_weights = DeepSeekVLV2VitWeight(vit_parts, True)

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        del vit_config
        return DeepSeekVLV2VisionModel(
            mm_related_params.config,
            torch.get_default_dtype(),
        )


register_multimodal_mixin("deepseek_vl_v2", DeepSeekVLV2Mixin)
