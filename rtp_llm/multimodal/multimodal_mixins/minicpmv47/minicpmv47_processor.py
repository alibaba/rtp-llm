"""MiniCPM-V-4.7 image preprocessing, ported self-contained from transformers
MiniCPMV4_6ImageProcessorPil (the repo's bazel-pinned transformers 5.2.0 does
not ship the minicpmv4_6 module).

Per image: thumbnail (find_best_resize, divisible by patch*4) + optional slice
grid (get_sliced_grid), each slice resized from a refine image; every crop is
rescaled/normalized and patchified into the NaViT packed layout
``[C, patch, P]``.  Output contract matches the HF processor:
``pixel_values [1, C, patch, P]``, ``target_sizes [n_crops, 2]`` patch grids,
``grids`` per-image slice grids ([rows, cols]; [0, 0] when unsliced),
``num_patches_per_image``.  ``pixel_values`` is pre-cast to bf16 inside the
preprocess worker (round-to-nearest-even, identical values to the device-side
cast it replaces): this halves the H2D pixel transfer and removes the
device-side cast kernel from the vision forward.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def _find_best_resize(
    image_size: Tuple[int, int],
    scale_resolution: int,
    patch_size: int,
    allow_upscale: bool = False,
) -> Tuple[int, int]:
    height, width = image_size
    if (height * width > scale_resolution * scale_resolution) or allow_upscale:
        aspect_ratio = width / height
        height = int(scale_resolution / math.sqrt(aspect_ratio))
        width = int(height * aspect_ratio)
    # factor 4 = two successive 2x2 spatial merges (ViT insert merger + final merger).
    best_width = _ensure_divide(width, patch_size * 4)
    best_height = _ensure_divide(height, patch_size * 4)
    return best_height, best_width


def _get_refine_size(
    image_size: Tuple[int, int],
    grid: List[int],
    scale_resolution: int,
    patch_size: int,
    allow_upscale: bool = False,
) -> Tuple[int, int]:
    height, width = image_size
    grid_y, grid_x = grid
    refine_width = _ensure_divide(width, grid_x)
    refine_height = _ensure_divide(height, grid_y)
    best_height, best_width = _find_best_resize(
        image_size=(refine_height / grid_y, refine_width / grid_x),
        scale_resolution=scale_resolution,
        patch_size=patch_size,
        allow_upscale=allow_upscale,
    )
    return best_height * grid_y, best_width * grid_x


def _get_sliced_grid(
    image_size: Tuple[int, int],
    max_slice_nums: int,
    scale_resolution: int,
) -> Optional[List[int]]:
    original_height, original_width = image_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)
    if multiple <= 1:
        return None

    best_grid = [1, 1]
    min_error = float("inf")
    for num_slices in [multiple - 1, multiple, multiple + 1]:
        if num_slices == 1 or num_slices > max_slice_nums:
            continue
        for num_rows in range(1, num_slices + 1):
            if num_slices % num_rows == 0:
                num_cols = num_slices // num_rows
                error = abs(log_ratio - math.log(num_rows / num_cols))
                if error < min_error:
                    best_grid = [num_cols, num_rows]
                    min_error = error
    return best_grid


def _divide_to_patches(image: np.ndarray, patch_size: Tuple[int, int]) -> List[np.ndarray]:
    """Row-major [C, H, W] -> [C, patch_h, patch_w] crops."""
    patch_h, patch_w = patch_size
    height, width = image.shape[-2:]
    patches = []
    for i in range(0, height, patch_h):
        for j in range(0, width, patch_w):
            patches.append(image[..., i : i + patch_h, j : j + patch_w])
    return patches


def _reshape_by_patch(image: np.ndarray, patch_size: int) -> np.ndarray:
    """[C, H, W] -> NaViT patchified [C, patch_size, H*W/patch_size]."""
    num_channels, height, width = image.shape
    num_patches_h, num_patches_w = height // patch_size, width // patch_size
    patches = image.reshape(
        num_channels, num_patches_h, patch_size, num_patches_w, patch_size
    )
    patches = patches.transpose(0, 2, 1, 3, 4)
    return patches.reshape(num_channels, patch_size, -1)


class MiniCPMV47ImageProcessor:
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        max_slice_nums: int = 9,
        scale_resolution: int = 448,
        patch_size: int = 14,
        slice_mode: bool = True,
    ):
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.max_slice_nums = int(max_slice_nums)
        self.scale_resolution = int(scale_resolution)
        self.patch_size = int(patch_size)
        self.slice_mode = bool(slice_mode)

    def _to_normalized_array(self, image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
        resized = image.resize((size[1], size[0]), resample=Image.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.asarray(self.image_mean, dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(self.image_std, dtype=np.float32).reshape(-1, 1, 1)
        return (arr - mean) / std

    def __call__(self, image: Image.Image) -> Dict:
        image_size = image.size[::-1]  # (H, W)
        best_grid = None
        if self.slice_mode:
            best_grid = _get_sliced_grid(image_size, self.max_slice_nums, self.scale_resolution)

        patch_size = self.patch_size
        source_height, source_width = _find_best_resize(
            image_size, self.scale_resolution, patch_size, allow_upscale=(best_grid is None)
        )
        source_img = self._to_normalized_array(image, (source_height, source_width))
        image_pv = [_reshape_by_patch(source_img, patch_size)]
        image_ts = [[source_height // patch_size, source_width // patch_size]]

        if best_grid is not None:
            refine_h, refine_w = _get_refine_size(
                image_size, best_grid, self.scale_resolution, patch_size, allow_upscale=True
            )
            refine_img = self._to_normalized_array(image, (refine_h, refine_w))
            refine_height, refine_width = refine_img.shape[-2:]
            grid_y, grid_x = best_grid
            patch_height, patch_width = refine_height // grid_y, refine_width // grid_x
            slice_patches = _divide_to_patches(refine_img, (patch_height, patch_width))
            for patch_arr in slice_patches:
                image_pv.append(_reshape_by_patch(patch_arr, patch_size))
                image_ts.append([patch_height // patch_size, patch_width // patch_size])

        pixel_values = np.concatenate(image_pv, axis=-1)[np.newaxis, ...]
        return {
            "pixel_values": torch.from_numpy(np.ascontiguousarray(pixel_values)).to(torch.bfloat16),
            "target_sizes": torch.tensor(image_ts, dtype=torch.int32),
            "grids": [best_grid if best_grid is not None else [0, 0]],
            "num_patches_per_image": [len(image_pv)],
        }
