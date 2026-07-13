# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
MiniMax VL family HuggingFace-compatible Processor, ImageProcessor, VideoProcessor.
"""
import math
from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import InterpolationMode
from transformers import BatchFeature
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import ImagesKwargs, Unpack
from transformers.utils import TensorType

from rtp_llm.multimodal.mm_error_messages import MMErr, raise_mm


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 451584,
    min_image_dimension: int = 10,
    max_image_aspect_ratio: float = 200.0,
) -> tuple[int, int]:
    if min_image_dimension > 0 and (
        height < min_image_dimension or width < min_image_dimension
    ):
        raise_mm(
            MMErr.IMG_HW.format(
                f"height:{height} or width:{width} must be larger than "
                f"{min_image_dimension}"
            )
        )
    if max_image_aspect_ratio > 0 and (
        max(height, width) / min(height, width) > max_image_aspect_ratio
    ):
        ratio_limit = f"{float(max_image_aspect_ratio):g}"
        raise_mm(
            MMErr.IMG_HW.format(
                f"absolute aspect ratio must be smaller than {ratio_limit}, "
                f"got {height} / {width}"
            )
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    if h_bar <= 0 or w_bar <= 0:
        raise_mm(MMErr.IMG_TOO_SMALL)
    return h_bar, w_bar


def get_hw_multiple_of(
    image_size: Tuple[int, int],
    multiple: int,
    max_size: Union[None, int, Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """Calculate target size aligned to *multiple*, optionally capped by *max_size*.

    Used for video frame resize (dimension-based ceil-align), as opposed to
    :func:`smart_resize` which is area-based.

    Args:
        image_size: ``(width, height)`` of the source frame.
        multiple: Alignment factor (``patch_size * spatial_merge_size``).
        max_size: ``None`` (no cap), ``int`` (cap longest edge), or
            ``(max_w, max_h)`` tuple.

    Returns:
        ``(new_width, new_height)``, both divisible by *multiple*.
    """
    w, h = image_size

    if isinstance(max_size, int):
        ratio = 1.0
        max_dim = max(w, h)
        if max_dim > max_size:
            ratio = max_size / max_dim
        new_w = round(w * ratio)
        new_h = round(h * ratio)
        new_w = (
            new_w if new_w % multiple == 0 else new_w + (multiple - new_w % multiple)
        )
        new_h = (
            new_h if new_h % multiple == 0 else new_h + (multiple - new_h % multiple)
        )
        return new_w, new_h

    new_w = w if w % multiple == 0 else w + (multiple - w % multiple)
    new_h = h if h % multiple == 0 else h + (multiple - h % multiple)

    if max_size is not None:
        assert isinstance(max_size, (list, tuple)) and len(max_size) == 2
        max_w, max_h = max_size
        assert max_w % multiple == 0 and max_h % multiple == 0
        if new_w > max_w or new_h > max_h:
            new_w_ = min((new_w * max_w) // new_w, (new_w * max_h) // new_h)
            new_h_ = min((new_h * max_w) // new_w, (new_h * max_h) // new_h)
            new_w = new_w_
            new_h = new_h_
            new_w = (
                new_w
                if new_w % multiple == 0
                else new_w + (multiple - new_w % multiple)
            )
            new_h = (
                new_h
                if new_h % multiple == 0
                else new_h + (multiple - new_h % multiple)
            )
        assert new_w <= max_w and new_h <= max_h

    return new_w, new_h


def compute_sampled_frame_indices(
    total_frames: int,
    video_fps: float,
    fps: float,
    max_frames: Optional[int] = None,
) -> List[int]:
    """Pick frame indices matching sglang's constant-mode behavior.

    Greedily selects frames whose timestamp is at least ``1/fps`` seconds
    after the previously kept frame; always appends the last frame if it
    adds temporal information.  When the result exceeds *max_frames*, it is
    uniformly sub-sampled while preserving the first and last frame.
    """
    if total_frames <= 0 or video_fps <= 0 or fps <= 0:
        return [0] if total_frames > 0 else []

    read_time_interval = 1.0 / fps
    eps = 1e-4

    indices: List[int] = []
    prev_kept_ts = -float("inf")
    while True:
        if not indices:
            target_frame = 0
        else:
            target_ts = prev_kept_ts + read_time_interval - eps
            target_frame = math.ceil(target_ts * video_fps)
            target_frame = max(target_frame, indices[-1] + 1)
        if target_frame >= total_frames:
            break
        indices.append(target_frame)
        prev_kept_ts = target_frame / video_fps

    last_frame_idx = total_frames - 1
    last_ts = last_frame_idx / video_fps
    if indices and indices[-1] != last_frame_idx and last_ts - prev_kept_ts > eps:
        indices.append(last_frame_idx)

    if not indices:
        indices = [0]
    if max_frames is not None and len(indices) > max_frames > 0:
        if max_frames == 1:
            return [indices[0]]
        last = indices[-1]
        step = len(indices) / (max_frames - 1)
        indices = [indices[int(i * step)] for i in range(max_frames - 1)]
        indices.append(last)
    return indices


# ==============================================================================
# MiniMax M3 VL Image Processor Fast (Fast Mode - Torch based)
# ==============================================================================


class MiniMaxM3VLImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int


class MiniMaxM3VLImageProcessor(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {
        "height": 672,
        "width": 672,
    }  # required by base class validation, not used as resize bound
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_pixels = 4 * 28 * 28  # 3136, matches smart_resize default lower bound
    max_pixels = 451584  # 672*672
    valid_kwargs = MiniMaxM3VLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiniMaxM3VLImageProcessorKwargs]):
        super().__init__(**kwargs)

    def preprocess(
        self, images, **kwargs: Unpack[MiniMaxM3VLImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: List[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        interpolation: PILImageResampling | InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | List[float] | None,
        image_std: float | List[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        max_pixels: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        min_pixels: int = 4 * 28 * 28,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping
        )
        resized_images_grouped = {}
        factor = patch_size * merge_size
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                stacked_images = self.resize(
                    stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images

        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]

            patches = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)

            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(
                    1,
                    temporal_patch_size - (patches.shape[1] % temporal_patch_size),
                    1,
                    1,
                    1,
                )
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(
            processed_images_grouped, grouped_images_index
        )
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids, dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            max_pixels=max_pixels,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w
