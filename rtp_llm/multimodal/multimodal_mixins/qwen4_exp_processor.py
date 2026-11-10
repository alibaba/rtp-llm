"""Qwen3-VL preprocessing and token layout for the released Qwen4 checkpoint."""

from dataclasses import dataclass

import torch
from PIL import Image, ImageOps

from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.multimodal.token_plan import MMTokenPlan
from rtp_llm.utils.base_model_datatypes import MMUrlType


@dataclass
class Qwen4VisualInput:
    pixels: torch.Tensor
    grid: torch.Tensor
    token_plan: MMTokenPlan
    positions: list
    semantic_ids: list


def visual_token_plan(grid, processor, is_video=False, timestamps=None):
    """Construct frame-local coordinates and the official timestamped prompt."""
    t, h, w = (int(x) for x in grid[0].tolist())
    merge = processor.image_processor.merge_size
    h, w = h // merge, w // merge
    if min(t, h, w) <= 0:
        raise ValueError("Qwen4 visual grid must have positive merged dimensions")
    if not is_video and t != 1:
        raise ValueError("an image must produce one temporal grid cell")
    if is_video and (timestamps is None or len(timestamps) != t):
        raise ValueError("video timestamps must match temporal patches")
    token_id = processor.video_token_id if is_video else processor.image_token_id
    start, end = processor.vision_start_token_id, processor.vision_end_token_id
    tokens, spans, positions, semantic_ids = [], [], [], []
    spatial = torch.stack(
        (
            torch.zeros(h * w, dtype=torch.int32),
            torch.arange(h, dtype=torch.int32).repeat_interleave(w),
            torch.arange(w, dtype=torch.int32).repeat(h),
        ),
        dim=-1,
    )
    for frame in range(t):
        if is_video:
            tokens.extend(
                processor.tokenizer.encode(
                    f"<{timestamps[frame]:.1f} seconds>", add_special_tokens=False
                )
            )
        tokens.append(start)
        spans.append((len(tokens), h * w))
        tokens.extend([token_id] * (h * w))
        tokens.append(end)
        positions.append(spatial.clone())
        semantic_ids.append(torch.tensor(token_id, dtype=torch.int32))
    return MMTokenPlan(tokens, spans), positions, semantic_ids


def preprocess_qwen4(mm_inputs, vit_config, processor):
    if len(mm_inputs) != 1:
        raise ValueError("Qwen4 preprocess expects one media item")
    item = mm_inputs[0]
    video = item.mm_type == MMUrlType.VIDEO
    if item.mm_type not in (MMUrlType.DEFAULT, MMUrlType.IMAGE, MMUrlType.VIDEO):
        raise ValueError("Qwen4 supports image and video inputs")
    config = item.mm_preprocess_config
    size = dict(
        (processor.video_processor if video else processor.image_processor).size
    )
    if config.min_pixels > 0:
        size["shortest_edge"] = config.min_pixels
    if config.max_pixels > 0:
        size["longest_edge"] = config.max_pixels
    content = get_bytes_io_from_url(
        item.url,
        vit_config.download_headers,
        max_file_size_kb=(
            vit_config.mm_video_max_file_size_kb
            if video
            else vit_config.mm_image_max_file_size_kb
        ),
    )
    timestamps = None
    if video:
        from decord import VideoReader, cpu
        from transformers.video_utils import VideoMetadata

        reader = VideoReader(content, ctx=cpu(0), num_threads=1)
        fps = float(reader.get_avg_fps())
        if fps <= 0 or len(reader) == 0:
            raise ValueError("video must have frames and a valid source FPS")
        vp = processor.video_processor
        metadata = VideoMetadata(total_num_frames=len(reader), fps=fps)
        desired_fps = config.fps if config.fps > 0 else vp.fps
        minimum = config.min_frames if config.min_frames > 0 else vp.min_frames
        maximum = config.max_frames if config.max_frames > 0 else vp.max_frames
        count = min(
            max(int(len(reader) / fps * desired_fps), minimum), maximum, len(reader)
        )
        indices = vp.sample_frames(metadata, num_frames=count).tolist()
        frames = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(
            0, 3, 1, 2
        )
        metadata.frames_indices = indices
        result = vp(
            videos=[frames],
            video_metadata=[metadata],
            do_sample_frames=False,
            return_metadata=True,
            size=size,
            return_tensors="pt",
        )
        pixels, grid = result["pixel_values_videos"], result["video_grid_thw"]
        timestamps = processor._calculate_timestamps(
            indices, fps, vp.temporal_patch_size
        )
    else:
        image = ImageOps.exif_transpose(Image.open(content)).convert("RGB")
        if config.height > 0 and config.width > 0:
            from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.qwen2_5_vl_mixin import (
                smart_resize,
            )

            factor = (
                processor.image_processor.patch_size
                * processor.image_processor.merge_size
            )
            height, width = smart_resize(config.height, config.width, factor=factor)
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            result = processor.image_processor(
                images=[image], do_resize=False, return_tensors="pt"
            )
        else:
            result = processor.image_processor(
                images=[image], size=size, return_tensors="pt"
            )
        pixels, grid = result["pixel_values"], result["image_grid_thw"]
    plan, positions, semantic = visual_token_plan(grid, processor, video, timestamps)
    return Qwen4VisualInput(pixels, grid, plan, positions, semantic)
