"""Vendored DiffusionGemma image processor.

Reference: transformers 5.20 image_processing_gemma4.py
(Gemma4ImageProcessor). Pinned bazel transformers (5.2.0) predates the gemma4
family, so the resize/patchify/position pipeline is implemented here on
torchvision. Output contract matches the HF processor:

  pixel_values          [1, max_patches, 3 * patch_size^2] (rescaled [0,1])
  image_position_ids    [1, max_patches, 2]                ((x, y); -1 padding)
  num_soft_tokens_per_image [int]
"""

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as tvF
from torchvision.transforms import InterpolationMode

_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def _get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
):
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
            f"Attempting to resize to 0x0: height/width must stay divisible by {side_mult}."
        )
    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
    target_height = min(target_height, max_side_length)
    target_width = min(target_width, max_side_length)
    return target_height, target_width


def _convert_image_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(C, H, W) -> (H//p * W//p, p * p * C) row-major over the patch grid."""
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched = image.reshape(
        num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched = patched.permute(1, 3, 2, 4, 0)
    return patched.reshape(num_patches_height * num_patches_width, -1)


def _pad_along_first_dim(image, positions, target_length):
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        padding = [0, 0] * (image.ndim - 1) + [0, padding_length]
        pos_padding = (0, 0, 0, padding_length)
        image = F.pad(image, padding, mode="constant", value=0)
        positions = F.pad(positions, pos_padding, mode="constant", value=-1)
    return image, positions


class DiffusionGemmaImageProcessor:
    def __init__(
        self,
        patch_size: int = 16,
        max_soft_tokens: int = 280,
        pooling_kernel_size: int = 3,
        resample: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"max_soft_tokens must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}."
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.resample = resample

    @classmethod
    def from_vision_config(cls, vision_config: Dict) -> "DiffusionGemmaImageProcessor":
        return cls(
            patch_size=int(vision_config.get("patch_size", 16)),
            max_soft_tokens=int(
                vision_config.get("default_output_length", 280)
            ),
            pooling_kernel_size=int(vision_config.get("pooling_kernel_size", 3)),
        )

    def __call__(self, image, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        return self.preprocess([image], return_tensors=return_tensors)

    def preprocess(
        self, images: List, return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []
        for image in images:
            # Match the HF processor pipeline: resize happens on the uint8
            # tensor (BICUBIC uint8 rounding), then rescale to [0, 1].
            tensor = self._to_uint8_tensor(image)
            height, width = tensor.shape[-2], tensor.shape[-1]
            target_height, target_width = _get_aspect_ratio_preserving_size(
                height=height,
                width=width,
                patch_size=self.patch_size,
                max_patches=max_patches,
                pooling_kernel_size=self.pooling_kernel_size,
            )
            if (target_height, target_width) != (height, width):
                tensor = tvF.resize(
                    tensor,
                    size=[target_height, target_width],
                    interpolation=self.resample,
                    antialias=True,
                )
            tensor = tensor.float() / 255.0
            patches = _convert_image_to_patches(tensor, self.patch_size)
            num_soft_tokens_per_image.append(
                patches.shape[0] // self.pooling_kernel_size**2
            )
            patch_height = target_height // self.patch_size
            patch_width = target_width // self.patch_size
            device = tensor.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            stacked_grid = torch.stack(patch_grid, dim=-1)
            real_positions = stacked_grid.reshape(patches.shape[0], 2)
            patches, positions = _pad_along_first_dim(
                patches, real_positions, max_patches
            )
            pixel_values.append(patches)
            position_ids.append(positions)
        return {
            "pixel_values": torch.stack(pixel_values, dim=0).to(torch.float32),
            "image_position_ids": torch.stack(position_ids, dim=0).long(),
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }

    @staticmethod
    def _to_uint8_tensor(image) -> torch.Tensor:
        """PIL image (RGB) -> uint8 tensor (C, H, W), matching the HF
        TorchvisionBackend input dtype for the resize step."""
        import numpy

        if isinstance(image, torch.Tensor):
            return image
        return (
            torch.from_numpy(numpy.asarray(image.convert("RGB")))
            .permute(2, 0, 1)
            .contiguous()
        )
