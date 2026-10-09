"""Gemma4 image processor port (HF transformers 5.5.0 image_processing_gemma4).

Per-image: aspect-ratio-preserving resize (patch budget = max_soft_tokens *
pooling_kernel_size^2 patches, sides multiples of pooling_kernel_size *
patch_size) -> rescale 1/255 -> patchify to [P, 3*patch^2] -> (x, y) position
ids -> zero-pad patches and (-1, -1)-pad positions to max_patches.
"""

import math
from typing import List, Tuple

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F

PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
MAX_SOFT_TOKENS = 280
SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> Tuple[int, int]:
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be "
            f"divisible by pooling_kernel_size * patch_size={side_mult}."
        )
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            int(math.floor(height / width)) * side_mult, max_side_length
        )

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resized image has too many pixels: {target_height * target_width} > {target_px}"
        )
    return target_height, target_width


def convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched = image.reshape(
        num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched = patched.permute(1, 3, 2, 4, 0)
    return patched.reshape(num_patches_height * num_patches_width, -1)


def pad_along_first_dim(
    patches: torch.Tensor, positions: torch.Tensor, target_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    current_length = patches.shape[0]
    if current_length >= target_length:
        return patches[:target_length], positions[:target_length]
    pad_len = target_length - current_length
    patches = torch.cat(
        [patches, torch.zeros(pad_len, *patches.shape[1:], dtype=patches.dtype)], dim=0
    )
    positions = torch.cat(
        [positions, torch.full((pad_len, 2), -1, dtype=positions.dtype)], dim=0
    )
    return patches, positions


class Gemma4ImageProcessor:
    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        max_soft_tokens: int = MAX_SOFT_TOKENS,
        pooling_kernel_size: int = POOLING_KERNEL_SIZE,
    ):
        if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}"
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size

    @classmethod
    def from_pretrained(cls, _ckpt_path: str, **kwargs):
        return cls(**kwargs)

    def __call__(self, images: List[Image.Image]):
        return self.preprocess(images)

    def preprocess(self, images: List[Image.Image]):
        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []

        for pil_image in images:
            image = F.pil_to_tensor(pil_image.convert("RGB"))
            height, width = image.shape[-2], image.shape[-1]
            target_height, target_width = get_aspect_ratio_preserving_size(
                height, width, self.patch_size, max_patches, self.pooling_kernel_size
            )
            if target_height != height or target_width != width:
                image = F.resize(
                    image,
                    size=[target_height, target_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
            image = image.float() * (1.0 / 255.0)

            patch_height = image.shape[-2] // self.patch_size
            patch_width = image.shape[-1] // self.patch_size
            patches = convert_image_to_patches(image, self.patch_size)
            num_soft_tokens_per_image.append(
                patches.shape[0] // self.pooling_kernel_size**2
            )

            grid = torch.meshgrid(
                torch.arange(patch_width),
                torch.arange(patch_height),
                indexing="xy",
            )
            real_positions = torch.stack(grid, dim=-1).reshape(patches.shape[0], 2)
            patches, positions = pad_along_first_dim(
                patches, real_positions, max_patches
            )
            pixel_values.append(patches)
            position_ids.append(positions)

        return {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "image_position_ids": torch.stack(position_ids, dim=0),
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }


class Gemma4VideoProcessor:
    def __init__(
        self,
        patch_size: int = PATCH_SIZE,
        max_soft_tokens: int = 70,
        pooling_kernel_size: int = POOLING_KERNEL_SIZE,
        num_frames: int = 32,
    ):
        if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}"
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.num_frames = num_frames

    @classmethod
    def from_pretrained(cls, _ckpt_path: str, **kwargs):
        return cls(**kwargs)

    def __call__(self, videos: List[torch.Tensor]):
        return self.preprocess(videos)

    def preprocess(self, videos: List[torch.Tensor]):
        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_video = []

        for video in videos:
            if video.dim() != 4 or video.shape[1] != 3:
                raise ValueError(
                    f"video must have shape [frames, 3, height, width], got {tuple(video.shape)}"
                )
            height, width = video.shape[-2:]
            target_height, target_width = get_aspect_ratio_preserving_size(
                height,
                width,
                self.patch_size,
                max_patches,
                self.pooling_kernel_size,
            )
            if (target_height, target_width) != (height, width):
                video = F.resize(
                    video,
                    size=[target_height, target_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
            video = video.float() / 255.0
            frames, channels, resized_height, resized_width = video.shape
            patch_height = resized_height // self.patch_size
            patch_width = resized_width // self.patch_size
            patches = video.reshape(
                frames,
                channels,
                patch_height,
                self.patch_size,
                patch_width,
                self.patch_size,
            )
            patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(
                frames, patch_height * patch_width, -1
            )
            num_soft_tokens_per_video.append(
                patches.shape[1] // self.pooling_kernel_size**2
            )
            grid = torch.meshgrid(
                torch.arange(patch_width),
                torch.arange(patch_height),
                indexing="xy",
            )
            positions = torch.stack(grid, dim=-1).reshape(-1, 2)
            positions = positions.unsqueeze(0).expand(frames, -1, -1)
            pad_len = max_patches - patches.shape[1]
            if pad_len < 0:
                raise ValueError(
                    f"video frame has {patches.shape[1]} patches, max is {max_patches}"
                )
            if pad_len:
                patches = torch.cat(
                    [
                        patches,
                        torch.zeros(
                            frames,
                            pad_len,
                            patches.shape[-1],
                            dtype=patches.dtype,
                        ),
                    ],
                    dim=1,
                )
                positions = torch.cat(
                    [
                        positions,
                        torch.full((frames, pad_len, 2), -1, dtype=positions.dtype),
                    ],
                    dim=1,
                )
            pixel_values.append(patches)
            position_ids.append(positions)

        return {
            "pixel_values_videos": torch.stack(pixel_values, dim=0),
            "video_position_ids": torch.stack(position_ids, dim=0),
            "num_soft_tokens_per_video": num_soft_tokens_per_video,
        }
