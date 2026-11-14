"""Qwen3-VL preprocessing and token layout for the released Qwen4 checkpoint."""

from dataclasses import dataclass

import torch
from PIL import Image, ImageOps

from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.utils.base_model_datatypes import MMUrlType


@dataclass
class Qwen4VisualInput:
    pixels: torch.Tensor
    grid: torch.Tensor
    timestamps: list | None = None


def assemble_visual_embedding(data, features, positions, processor, text_embedding):
    """Return one complete embedding/position/semantic-ID tensor per media item."""
    t, h, w = (int(x) for x in data.grid[0].tolist())
    merge = processor.image_processor.merge_size
    h, w = h // merge, w // merge
    if min(t, h, w) <= 0:
        raise ValueError("Qwen4 visual grid must have positive merged dimensions")
    is_video = data.timestamps is not None
    if not is_video and t != 1:
        raise ValueError("an image must produce one temporal grid cell")
    if is_video and len(data.timestamps) != t:
        raise ValueError("video timestamps must match temporal patches")
    if features.shape[0] != t * h * w:
        raise ValueError("visual features must match the merged grid")
    token_id = processor.video_token_id if is_video else processor.image_token_id
    start, end = processor.vision_start_token_id, processor.vision_end_token_id
    tokens, visual_mask, coordinates = [], [], []
    spatial = positions[: h * w].to(device="cpu", dtype=torch.int32)
    offset = 0
    for frame in range(t):
        text = []
        if is_video:
            text.extend(
                processor.tokenizer.encode(
                    f"<{data.timestamps[frame]:.1f} seconds>", add_special_tokens=False
                )
            )
        text.append(start)
        tokens.extend(text)
        visual_mask.extend([False] * len(text))
        coordinates.append(
            torch.arange(offset, offset + len(text), dtype=torch.int32)
            .unsqueeze(1)
            .expand(-1, 3)
        )
        offset += len(text)
        tokens.extend([token_id] * (h * w))
        visual_mask.extend([True] * (h * w))
        coordinates.append(spatial + offset)
        offset += max(h, w)
        tokens.append(end)
        visual_mask.append(False)
        coordinates.append(torch.full((1, 3), offset, dtype=torch.int32))
        offset += 1
    token_ids = torch.tensor(tokens, dtype=torch.int32)
    visual_mask = torch.tensor(visual_mask, dtype=torch.bool)
    # Only fetch the small set of text rows from the CPU checkpoint mapping.
    # No second vocabulary table is retained on the ViT GPU.
    text = text_embedding[token_ids[~visual_mask].long()]
    output = features.new_empty((len(tokens), features.shape[1]))
    device_mask = visual_mask.to(features.device)
    output[device_mask] = features
    output[~device_mask] = text.to(device=features.device, dtype=features.dtype)
    return output, torch.cat(coordinates), token_ids


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
    return Qwen4VisualInput(pixels, grid, timestamps)
