"""Image-only NaViT preprocessor for Kimi-K2.5.

This is a self-contained fallback used when the HF ckpt's
`KimiK25VisionProcessor` (which depends on `mecord` for video) cannot be
loaded via `AutoImageProcessor.from_pretrained(..., trust_remote_code=True)`.
It mirrors the resize / normalize / patchify logic from the ckpt's
`media_utils.py` but skips video paths entirely — feeding a `{type:
"video", ...}` media dict raises ``ValueError`` rather than silently
returning an empty / image-shaped batch.
"""

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType


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
        "sampled_nframes": 1,
    }


def _image_to_np(image: Image.Image, resize_to: tuple) -> np.ndarray:
    image = image.resize(resize_to, resample=Image.Resampling.BICUBIC)
    return np.asarray(image)


def _normalize(x: np.ndarray, mean: np.ndarray, std_inv: np.ndarray) -> np.ndarray:
    x = (x / 255.0).astype(np.float32)
    x -= mean
    x *= std_inv
    return x


def _navit_patchify(pixel_values: np.ndarray, patch_size: int) -> Dict[str, np.ndarray]:
    T, H, W, C = pixel_values.shape
    assert C == 3
    patches = pixel_values.reshape(
        T, H // patch_size, patch_size, W // patch_size, patch_size, C
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = np.array([T, H // patch_size, W // patch_size])
    return {"pixel_values": patches, "grid_thw": grid_thw}


_DEFAULT_MEDIA_PROC_CFG: Dict[str, Any] = {
    "in_patch_limit": 16384,
    "patch_size": 14,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_kernel_size": 2,
    "fixed_output_tokens": None,
    "patch_limit_on_one_side": 512,
    "in_patch_limit_each_frame": 4096,
    "in_patch_limit_video": None,
    "sample_fps": 2.0,
    "max_num_frames_each_video": None,
    "temporal_merge_kernel_size": 4,
    "timestamp_mode": "hh:mm:ss.fff",
}


class KimiK25VisionProcessor(BaseImageProcessor):
    """Image-only fallback preprocessor matching the ckpt-side outputs."""

    model_type = "kimi_k25"

    def __init__(self, media_proc_cfg: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        cfg = dict(_DEFAULT_MEDIA_PROC_CFG)
        if media_proc_cfg:
            cfg.update(media_proc_cfg)
        self.media_proc_cfg = cfg
        self.num_frames_per_chunk = cfg["temporal_merge_kernel_size"]

    @staticmethod
    def _coerce_image(media: Dict[str, Any]) -> Dict[str, Any]:
        media_type = media.get("type", "image")
        if media_type != "image":
            raise ValueError(
                f"KimiK25VisionProcessor is image-only; got media type "
                f"{media_type!r}. Video / audio inputs must be handled by the "
                f"ckpt-side processor."
            )
        img = media.get("image")
        if not isinstance(img, Image.Image):
            raise TypeError(
                f"KimiK25VisionProcessor (image-only) expects PIL.Image, got {type(img)}"
            )
        return {"type": "image", "image": img.convert("RGB")}

    def _resize_config(self, image: Image.Image) -> Dict[str, int]:
        w, h = image.size
        cfg = self.media_proc_cfg
        return _navit_resize_image(
            w,
            h,
            cfg["patch_size"],
            cfg["merge_kernel_size"],
            cfg["in_patch_limit"],
            cfg["patch_limit_on_one_side"],
            cfg["fixed_output_tokens"],
        )

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
            cfg_item = self._resize_config(item["image"])
            new_w, new_h = cfg_item["new_width"], cfg_item["new_height"]
            pad_w, pad_h = cfg_item["pad_width"], cfg_item["pad_height"]

            arr = _image_to_np(item["image"], (new_w, new_h))
            arr = np.pad(
                arr,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            arr = np.expand_dims(arr, axis=0)
            arr = _normalize(arr, image_mean, image_std_inv)
            per_image_tensors.append(_navit_patchify(arr, cfg["patch_size"]))

        pixel_values = torch.cat(
            [torch.from_numpy(item["pixel_values"]) for item in per_image_tensors]
        )
        grid_thws = torch.cat(
            [
                torch.from_numpy(item["grid_thw"]).to(torch.int64).unsqueeze(0)
                for item in per_image_tensors
            ]
        )
        data = {"pixel_values": pixel_values, "grid_thws": grid_thws}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out["media_proc_cfg"] = self.media_proc_cfg
        return out

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        cfg = config_dict.copy()
        media_proc_cfg = cfg.pop("media_proc_cfg", {})
        return cls(media_proc_cfg=media_proc_cfg, **cfg, **kwargs)
