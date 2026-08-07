"""Kimi-K3 image processor."""

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType

_DEFAULT_MEDIA_PROC_CFG: Dict[str, Any] = {
    "in_patch_limit": 65536,
    "patch_size": 14,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_kernel_size": 2,
    "fixed_output_tokens": None,
    "patch_limit_on_one_side": 512,
    "transparent_bg_config": {
        "pattern": "chessboard",
        "chessboard_square_size": 8,
        "chessboard_square_on_top_left": True,
        "chessboard_white_value": 255,
        "chessboard_gray_value": 180,
    },
    "transparent_bg_fill_stage": "after_resize",
}

_TRANSPARENT_BG_FILL_STAGES = ("before_resize", "after_resize")

# Per-image compressed byte limit, in KB.  Lives here so the renderer preflight
# and the vit url fallback, K3's two byte entry points, cannot disagree.
K3_MAX_IMAGE_FILE_SIZE_KB = 32 * 1024

# Largest square the tower's one-side patch limit can describe, 4x the patch
# budget it will actually use.  PIL's own 89 MP default is 256 MB decoded.
K3_MAX_IMAGE_PIXELS = (
    _DEFAULT_MEDIA_PROC_CFG["patch_limit_on_one_side"]
    * _DEFAULT_MEDIA_PROC_CFG["patch_size"]
) ** 2


def _navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: Optional[int],
) -> Dict[str, int]:
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = int(fixed_output_tokens)
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        assert token_height * merge_kernel_size <= patch_limit_on_one_side
        assert token_width * merge_kernel_size <= patch_limit_on_one_side
        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
    }


def _normalize(x: np.ndarray, mean: np.ndarray, std_inv: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    x -= mean
    x *= std_inv
    return x


def _navit_patchify(
    pixel_values: np.ndarray, patch_size: int
) -> Dict[str, np.ndarray]:
    T, H, W, C = pixel_values.shape
    assert C == 3
    patches = pixel_values.reshape(
        T, H // patch_size, patch_size, W // patch_size, patch_size, C
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = np.array([T, H // patch_size, W // patch_size])
    return {"pixel_values": patches, "grid_thw": grid_thw}


def _chessboard(
    height: int,
    width: int,
    square_size: int,
    square_on_top_left: bool,
    white_value: int,
    gray_value: int,
) -> np.ndarray:
    """Create one background without retaining image-sized arrays globally."""
    y = np.arange(height)[:, None] // square_size
    x = np.arange(width)[None, :] // square_size
    gray_mask = (x + y) % 2 == (1 if square_on_top_left else 0)
    background = np.full((height, width, 3), white_value, dtype=np.uint8)
    background[gray_mask] = gray_value
    return background


def _fill_transparent_background(
    image: Image.Image, config: Dict[str, Any]
) -> Image.Image:
    if image.mode == "RGB":
        return image
    if "A" not in image.getbands() and "transparency" not in image.info:
        return image.convert("RGB")

    rgba = np.asarray(image.convert("RGBA"))
    height, width = rgba.shape[:2]
    background = _chessboard(
        height,
        width,
        config["chessboard_square_size"],
        config["chessboard_square_on_top_left"],
        config["chessboard_white_value"],
        config["chessboard_gray_value"],
    )
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    result = alpha * rgba[:, :, :3] + (1.0 - alpha) * background
    return Image.fromarray(result.astype(np.uint8))


class KimiK3VisionProcessor(BaseImageProcessor):
    """Image-only NaViT processor for Kimi-K3."""

    model_type = "kimi_k3"
    model_input_names = ["pixel_values", "grid_thws"]

    def __init__(self, media_proc_cfg: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        cfg = dict(_DEFAULT_MEDIA_PROC_CFG)
        if media_proc_cfg:
            cfg.update(media_proc_cfg)
        fill_stage = cfg["transparent_bg_fill_stage"]
        if fill_stage not in _TRANSPARENT_BG_FILL_STAGES:
            raise ValueError(
                f"unsupported transparent_bg_fill_stage {fill_stage!r}, "
                f"expected one of {_TRANSPARENT_BG_FILL_STAGES}"
            )
        self.media_proc_cfg = cfg

    @staticmethod
    def _coerce_image(media: Dict[str, Any]) -> Dict[str, Any]:
        media_type = media.get("type", "image")
        if media_type != "image":
            raise ValueError(
                f"KimiK3VisionProcessor is image-only; got media type "
                f"{media_type!r}. Video / audio inputs are not supported."
            )
        image = media.get("image")
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"KimiK3VisionProcessor expects PIL.Image, got {type(image)}"
            )
        # Keep alpha/palette transparency until the resized image is composited
        # onto K3's configured background.
        return {"type": "image", "image": image}

    def _resize_config(self, image: Image.Image) -> Dict[str, int]:
        width, height = image.size
        return self.resize_config_for_size(width, height)

    def resize_config_for_size(self, width: int, height: int) -> Dict[str, int]:
        cfg = self.media_proc_cfg
        return _navit_resize_image(
            width,
            height,
            cfg["patch_size"],
            cfg["merge_kernel_size"],
            cfg["in_patch_limit"],
            cfg["patch_limit_on_one_side"],
            cfg["fixed_output_tokens"],
        )

    def _image_to_np(self, image: Image.Image, resize_to: tuple) -> np.ndarray:
        bg_config = self.media_proc_cfg["transparent_bg_config"]
        # Only differs on semi-transparent pixels: compositing before resize
        # blends the chessboard into resampled edges, after keeps it crisp.
        if self.media_proc_cfg["transparent_bg_fill_stage"] == "before_resize":
            image = _fill_transparent_background(image, bg_config)
            image = image.resize(resize_to, resample=Image.Resampling.BICUBIC)
        else:
            image = image.resize(resize_to, resample=Image.Resampling.BICUBIC)
            image = _fill_transparent_background(image, bg_config)
        return np.asarray(image)

    def media_tokens_calculator(self, media: Dict[str, Any]) -> int:
        media = self._coerce_image(media)
        return self._resize_config(media["image"])["num_tokens"]

    def preprocess(
        self,
        medias: Union[Dict[str, Any], List[Dict[str, Any]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if not isinstance(medias, list):
            medias = [medias]
        if not medias:
            return BatchFeature(data={}, tensor_type=return_tensors)

        cfg = self.media_proc_cfg
        image_mean = np.array(cfg["image_mean"])
        image_std_inv = 1.0 / np.array(cfg["image_std"])

        per_image_tensors: List[Dict[str, np.ndarray]] = []
        for item in medias:
            item = self._coerce_image(item)
            resize_config = self._resize_config(item["image"])
            new_width = resize_config["new_width"]
            new_height = resize_config["new_height"]
            pad_width = resize_config["pad_width"]
            pad_height = resize_config["pad_height"]

            array = self._image_to_np(item["image"], (new_width, new_height))
            if pad_height or pad_width:
                array = np.pad(
                    array,
                    ((0, pad_height), (0, pad_width), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            array = np.expand_dims(array, axis=0)
            array = _normalize(array, image_mean, image_std_inv)
            per_image_tensors.append(_navit_patchify(array, cfg["patch_size"]))

        pixel_value_tensors = [
            torch.from_numpy(item["pixel_values"]) for item in per_image_tensors
        ]
        grid_thw_tensors = [
            torch.from_numpy(item["grid_thw"]).to(torch.int64).unsqueeze(0)
            for item in per_image_tensors
        ]
        if len(per_image_tensors) == 1:
            pixel_values = pixel_value_tensors[0]
            grid_thws = grid_thw_tensors[0]
        else:
            pixel_values = torch.cat(pixel_value_tensors)
            grid_thws = torch.cat(grid_thw_tensors)
        return BatchFeature(
            data={"pixel_values": pixel_values, "grid_thws": grid_thws},
            tensor_type=return_tensors,
        )

    @staticmethod
    def make_image_prompt(width: int, height: int) -> str:
        return (
            f"<|media_begin|>image {width}x{height}"
            f"<|media_content|><|media_pad|><|media_end|>"
        )
