"""Metadata-only sizing for the current Qwen3.5 complete-tensor adapter.

Keep processor sampling/resizing identical to the pixel path. This module never
invokes a vision model, allocates image patches, or exposes video internals to FE.
A video is one complete tensor: visual, timestamp, and internal delimiter rows.
"""

import math

import torch
from PIL import Image

from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.qwen2_5_vl_mixin import (
    smart_resize,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.utils.base_model_datatypes import MMUrlType


def image_resize_plan(height, width, config, factor):
    """The first resize, shared by metadata and the actual Qwen3 pixel path."""
    if config.height != -1 and config.width != -1:
        height, width = smart_resize(config.height, config.width, factor=factor)
        return height, width, False
    if config.max_pixels != -1 or config.min_pixels != -1:
        height, width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=0 if config.min_pixels == -1 else config.min_pixels,
            max_pixels=0x7FFFFFFF if config.max_pixels == -1 else config.max_pixels,
        )
        return height, width, False
    return height, width, True


def video_output_length(grid, frame_indices, source_fps, processor):
    from rtp_llm.multimodal.qwen3_vl_video import video_timestamp_tokens

    timestamps = video_timestamp_tokens(grid, frame_indices, source_fps, processor)
    visual = int(grid.prod().item()) // processor.video_processor.merge_size**2
    return visual + sum(len(tokens) + 2 for tokens in timestamps)


def token_length(mm_input, vit_config, processor, factor, video_backend="nvdec"):
    config = mm_input.mm_preprocess_config
    if not mm_input.url:
        raise ValueError("metadata sizing requires a media URL")
    is_video = mm_input.mm_type == MMUrlType.VIDEO
    if mm_input.mm_type not in (MMUrlType.DEFAULT, MMUrlType.IMAGE, MMUrlType.VIDEO):
        raise ValueError("Qwen3.5 metadata sizing supports images and videos only")
    data = get_bytes_io_from_url(
        mm_input.url,
        vit_config.download_headers,
        max_file_size_kb=(
            vit_config.mm_video_max_file_size_kb
            if is_video
            else vit_config.mm_image_max_file_size_kb
        ),
    )
    if is_video:
        from rtp_llm.multimodal.qwen3_vl_video import (
            read_video_metadata,
            resolve_video_size,
            sample_frame_indices,
            video_resize_shape,
        )

        vp = processor.video_processor
        size = resolve_video_size(
            vp,
            vit_config.mm_video_total_min_pixels,
            vit_config.mm_video_total_max_pixels,
        )
        if video_backend == "nvdec":
            from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.gpu_video import (
                prepare_gpu_video,
            )

            # Existing production metadata planner: no CUDA context or decoded frames.
            plan, grid = prepare_gpu_video(
                data.getvalue(), config, vp, factor, size=size, vit_config=vit_config
            )
            return video_output_length(
                grid, plan.frame_indices, plan.source_fps, processor
            )
        if video_backend != "cpu":
            raise ValueError("unknown Qwen3.5 video backend")
        total, fps, height, width = read_video_metadata(data)
        indices = sample_frame_indices(total, fps, config, vit_config=vit_config)
        height, width = video_resize_shape(
            config, len(indices), height, width, vp, size
        )
        grid = torch.tensor(
            [
                [
                    math.ceil(len(indices) / vp.temporal_patch_size),
                    height // vp.patch_size,
                    width // vp.patch_size,
                ]
            ]
        )
        return video_output_length(grid, indices, fps, processor)

    with Image.open(data) as image:
        width, height = image.size
    height, width, do_resize = image_resize_plan(height, width, config, factor)
    image_processor = processor.image_processor
    if do_resize:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            smart_resize as processor_resize,
        )

        height, width = processor_resize(
            height,
            width,
            factor=image_processor.patch_size * image_processor.merge_size,
            min_pixels=image_processor.min_pixels,
            max_pixels=image_processor.max_pixels,
        )
    spatial_factor = image_processor.patch_size * image_processor.merge_size
    return (height // spatial_factor) * (width // spatial_factor)
