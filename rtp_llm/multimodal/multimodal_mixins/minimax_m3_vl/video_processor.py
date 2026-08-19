# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
MiniMax VL family HuggingFace-compatible VideoProcessor.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from transformers import BatchFeature
from transformers.image_utils import PILImageResampling, SizeDict
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import TensorType
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos

from rtp_llm.multimodal.mm_error_messages import MMErr, raise_mm

MAX_RATIO = 200
MIN_SHORT_SIDE_PIXEL = 112
VIDEO_MAX_TOTAL_PIXELS = 301_056_000


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize_by_long_side(
    height: int,
    width: int,
    factor: int,
    max_long_side_pixel: int,
    min_short_side_pixel: int,
) -> tuple[int, int]:
    if max_long_side_pixel <= 0:
        raise_mm(
            MMErr.IMG_HW.format(
                f"max_long_side_pixel must be positive, got {max_long_side_pixel}"
            )
        )

    long_side = max(height, width)
    short_side = min(height, width)
    scaled_height: float = height
    scaled_width: float = width
    if long_side > max_long_side_pixel:
        scale = max_long_side_pixel / long_side
        scaled_height = height * scale
        scaled_width = width * scale
    elif short_side < min_short_side_pixel:
        scale = min_short_side_pixel / short_side
        scaled_height = height * scale
        scaled_width = width * scale

    return (
        max(factor, round_by_factor(scaled_height, factor)),
        max(factor, round_by_factor(scaled_width, factor)),
    )


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 451584,
    max_long_side_pixel: Optional[int] = None,
    min_short_side_pixel: int = MIN_SHORT_SIDE_PIXEL,
) -> tuple[int, int]:
    if height < 10 or width < 10:
        raise_mm(
            MMErr.IMG_HW.format(
                f"height:{height} or width:{width} must be larger than 10"
            )
        )
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise_mm(
            MMErr.IMG_HW.format(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {height} / {width}"
            )
        )
    if max_long_side_pixel is not None:
        return _smart_resize_by_long_side(
            height,
            width,
            factor=factor,
            max_long_side_pixel=max_long_side_pixel,
            min_short_side_pixel=min_short_side_pixel,
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


class MiniMaxM3VLVideoProcessorKwargs(VideosKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    max_long_side_pixel: int
    total_pixels: int
    min_frames: int
    max_frames: int
    fps: float | int


class MiniMaxM3VLVideoProcessor(BaseVideoProcessor):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"height": 672, "width": 672}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True
    do_sample_frames = False
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_pixels = 4 * 28 * 28
    max_pixels = 768 * 28 * 28  # 602,112
    total_pixels = int(64000 * 28 * 28 * 0.9)  # ~45M, ~64k tokens budget
    max_long_side_pixel = None
    min_short_side_pixel = MIN_SHORT_SIDE_PIXEL
    max_total_pixels = VIDEO_MAX_TOTAL_PIXELS
    fps = 1.0
    min_frames = 4
    max_frames = 768
    valid_kwargs = MiniMaxM3VLVideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiniMaxM3VLVideoProcessorKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: List[torch.Tensor],
        do_convert_rgb: bool,
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
        min_pixels: int,
        max_pixels: int,
        max_long_side_pixel: Optional[int] = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        factor = patch_size * merge_size
        for shape, stacked_videos in grouped_videos.items():
            batch_size, num_frames, channels, height, width = stacked_videos.shape
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    max_long_side_pixel=max_long_side_pixel,
                    min_short_side_pixel=self.min_short_side_pixel,
                )
                if resized_height * resized_width * num_frames > self.max_total_pixels:
                    raise_mm(
                        MMErr.VIDEO_REQ.format(
                            f"video area {resized_height * resized_width * num_frames} "
                            f"(width * height * frames) exceeds max_total_pixels "
                            f"{self.max_total_pixels} after resizing"
                        )
                    )
                stacked_videos = stacked_videos.view(
                    batch_size * num_frames, channels, height, width
                )
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_videos = stacked_videos.view(
                    batch_size,
                    num_frames,
                    channels,
                    resized_height,
                    resized_width,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = stacked_videos.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )

            if pad := -patches.shape[1] % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channels = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channels,
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
                channels * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(
            processed_videos_grouped, grouped_videos_index
        )
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids, dtype=torch.long)

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            },
            tensor_type=return_tensors,
        )
