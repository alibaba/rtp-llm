"""Self-contained LFM2-VL image processor.

Faithful single-image port of ``transformers.models.lfm2_vl.
image_processing_lfm2_vl.Lfm2VlImageProcessor`` (slow path): smart_resize ->
optional tile split (2..10 tiles + full-image overview) -> rescale/normalize
-> 16x16 patches -> pad to max_num_patches with a validity mask.

Resizing happens in PIL (the runtime pip closure has no torchvision); PIL
bicubic matches the HF TorchvisionBackend reference to within 1 LSB on
~0.35% of pixels (validated on data/model/qwen_vl/1.jpeg). Batch
grouping/reordering from the HF class is dropped: the mixin preprocesses
one image at a time, and per-tile ops are independent.
"""

import math
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

# PILImageResampling values; the d1 processor_config pins resample=3 (BICUBIC)
_PIL_RESAMPLE = {
    0: Image.NEAREST,
    1: Image.LANCZOS,
    2: Image.BILINEAR,
    3: Image.BICUBIC,
}


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio
    return best_ratio


def convert_image_to_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    # (B, C, H, W) -> (B, nph*npw, ph*pw*C); channels last inside the patch
    batch_size, num_channels, image_height, image_width = images.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched = images.reshape(
        batch_size,
        num_channels,
        num_patches_height,
        patch_size,
        num_patches_width,
        patch_size,
    )
    patched = patched.permute(0, 2, 4, 3, 5, 1)
    return patched.reshape(batch_size, num_patches_height * num_patches_width, -1)


def pad_along_first_dim(images: torch.Tensor, target_length: int):
    current_length = images.shape[1]
    padding_length = target_length - current_length
    pixel_mask = torch.ones((target_length,), dtype=torch.int32)
    if padding_length > 0:
        paddings = (0, 0, 0, padding_length, 0, 0)
        images = torch.nn.functional.pad(images, paddings, mode="constant", value=0)
        pixel_mask[-padding_length:] = 0
    return images, pixel_mask


def split_to_tiles(images: torch.Tensor, num_tiles_height: int, num_tiles_width: int):
    # (B, C, H, W) -> (B, tiles, C, H/th, W/tw), row-major tile order
    batch_size, num_channels, height, width = images.size()
    images = images.view(
        batch_size,
        num_channels,
        num_tiles_height,
        height // num_tiles_height,
        num_tiles_width,
        width // num_tiles_width,
    )
    images = images.permute(0, 2, 4, 1, 3, 5).contiguous()
    return images.view(
        batch_size,
        num_tiles_width * num_tiles_height,
        num_channels,
        height // num_tiles_height,
        width // num_tiles_width,
    )


def _to_tensor(image: Image.Image) -> torch.Tensor:
    # PIL -> (1, C, H, W) uint8
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).unsqueeze(0)


class Lfm2VlVisionProcessor:
    def __init__(self, proc_cfg: dict):
        self.downsample_factor = int(proc_cfg.get("downsample_factor", 2))
        self.do_image_splitting = bool(proc_cfg.get("do_image_splitting", True))
        self.min_tiles = int(proc_cfg.get("min_tiles", 2))
        self.max_tiles = int(proc_cfg.get("max_tiles", 10))
        self.use_thumbnail = bool(proc_cfg.get("use_thumbnail", True))
        self.min_image_tokens = int(proc_cfg.get("min_image_tokens", 64))
        self.max_image_tokens = int(proc_cfg.get("max_image_tokens", 256))
        self.encoder_patch_size = int(proc_cfg.get("encoder_patch_size", 16))
        self.tile_size = int(proc_cfg.get("tile_size", 512))
        self.max_pixels_tolerance = float(proc_cfg.get("max_pixels_tolerance", 2.0))
        self.image_mean = proc_cfg.get("image_mean", [0.5, 0.5, 0.5])
        self.image_std = proc_cfg.get("image_std", [0.5, 0.5, 0.5])
        self.resample = int(proc_cfg.get("resample", 3))
        self.rescale_factor = float(proc_cfg.get("rescale_factor", 1.0 / 255.0))
        self.max_thumbnail_image_patches = (
            self.max_image_tokens * self.downsample_factor**2
        )
        self.tile_size_patches = (
            (self.tile_size // self.encoder_patch_size) ** 2
            if self.do_image_splitting
            else 0
        )
        self.max_num_patches = max(
            self.max_thumbnail_image_patches, self.tile_size_patches
        )

    @lru_cache(maxsize=8)
    def _target_ratios(self, min_tiles: int, max_tiles: int):
        ratios = [
            (w, h)
            for n in range(min_tiles, max_tiles + 1)
            for w in range(1, n + 1)
            for h in range(1, n + 1)
            if min_tiles <= w * h <= max_tiles
        ]
        return sorted(set(ratios), key=lambda x: x[0] * x[1])

    def _get_grid_layout(self, height, width, min_tiles, max_tiles, tile_size):
        aspect_ratio = width / height
        target_ratios = self._target_ratios(min_tiles, max_tiles)
        grid_width, grid_height = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, tile_size
        )
        return (
            grid_width,
            grid_height,
            tile_size * grid_width,
            tile_size * grid_height,
            grid_width * grid_height,
        )

    def smart_resize(self, height, width):
        total_factor = self.encoder_patch_size * self.downsample_factor
        min_pixels = (
            self.min_image_tokens
            * self.encoder_patch_size**2
            * self.downsample_factor**2
        )
        max_pixels = (
            self.max_image_tokens
            * self.encoder_patch_size**2
            * self.downsample_factor**2
        )

        h_bar = max(total_factor, round_by_factor(height, total_factor))
        w_bar = max(total_factor, round_by_factor(width, total_factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(
                total_factor, math.floor(height / beta / total_factor) * total_factor
            )
            w_bar = max(
                total_factor, math.floor(width / beta / total_factor) * total_factor
            )
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = max(
                total_factor, math.ceil(height * beta / total_factor) * total_factor
            )
            w_bar = max(
                total_factor, math.ceil(width * beta / total_factor) * total_factor
            )
        return w_bar, h_bar

    def _is_image_too_large(self, height, width):
        total_factor = self.encoder_patch_size * self.downsample_factor
        h_bar = max(self.encoder_patch_size, round_by_factor(height, total_factor))
        w_bar = max(self.encoder_patch_size, round_by_factor(width, total_factor))
        limit = (
            self.max_image_tokens
            * self.encoder_patch_size**2
            * self.downsample_factor**2
            * self.max_pixels_tolerance
        )
        return h_bar * w_bar > limit

    def _resize_pil(self, image: Image.Image, height: int, width: int) -> Image.Image:
        return image.resize((width, height), _PIL_RESAMPLE[self.resample])

    def _tile_split(self, image: Image.Image, new_height: int, new_width: int):
        # -> list of (1, C, th, tw) uint8 tiles + overview tile
        width, height = image.size
        grid_width, grid_height, target_width, target_height, _ = self._get_grid_layout(
            height, width, self.min_tiles, self.max_tiles, self.tile_size
        )
        resized = self._resize_pil(image, target_height, target_width)
        tiles = split_to_tiles(_to_tensor(resized), grid_height, grid_width)[0]
        tiles = [t.unsqueeze(0) for t in tiles]
        if self.use_thumbnail and grid_width * grid_height != 1:
            overview = self._resize_pil(image, new_height, new_width)
            tiles.append(_to_tensor(overview))
        return tiles

    def __call__(self, image: Image.Image, mm_preprocess_config=None) -> dict:
        image = image.convert("RGB")
        if mm_preprocess_config is not None:
            cfg_h = mm_preprocess_config.height
            cfg_w = mm_preprocess_config.width
            if cfg_h != -1 and cfg_w != -1:
                image = image.resize((cfg_w, cfg_h))

        do_image_splitting = self.do_image_splitting and not (
            self.min_tiles == self.max_tiles == 1
        )
        width, height = image.size
        is_large = self._is_image_too_large(height, width)
        new_width, new_height = self.smart_resize(height, width)

        if is_large and do_image_splitting:
            tiles = self._tile_split(image, new_height, new_width)
        else:
            tiles = [_to_tensor(self._resize_pil(image, new_height, new_width))]

        processed_tiles = []
        masks = []
        spatial_shapes = []
        mean = torch.tensor(self.image_mean).view(1, -1, 1, 1)
        std = torch.tensor(self.image_std).view(1, -1, 1, 1)
        for tile in tiles:
            tile = tile.float() * self.rescale_factor
            tile = (tile - mean) / std
            tile_height, tile_width = tile.shape[-2:]
            # keep the batch dim: pad_along_first_dim pads (1, n, dim) 3-D input
            patches = convert_image_to_patches(tile, self.encoder_patch_size)
            patches, mask = pad_along_first_dim(patches, self.max_num_patches)
            processed_tiles.append(patches[0])
            masks.append(mask)
            spatial_shapes.append(
                [
                    tile_height // self.encoder_patch_size,
                    tile_width // self.encoder_patch_size,
                ]
            )

        return {
            "pixel_values": torch.stack(processed_tiles),
            "pixel_attention_mask": torch.stack(masks),
            "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.int64),
        }
