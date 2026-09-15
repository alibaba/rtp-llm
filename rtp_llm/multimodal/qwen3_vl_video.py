"""Qwen3-VL/Qwen3.5 video policy shared by CPU and NVDEC preprocessing.

Match vLLM's Qwen3VLVideoBackend sampling and the installed Transformers
Qwen3VLVideoProcessor sizing. Pixel budgets in processor.size are totals for
all sampled frames; do not apply Qwen2-VL's implicit per-frame pixel cap.
"""

import inspect
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers.image_utils import SizeDict
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import VideoMetadata


def _value(configs: Any, name: str, default: Any) -> Any:
    value = getattr(configs, name, -1)
    return default if value is None or value == -1 else value


def sample_frame_indices(
    total_frames: int, source_fps: float, configs: Any
) -> List[int]:
    """Use vLLM's integer frame count and float64 uniform frame indices.

    Odd counts are intentional: the video processor pads the final temporal
    patch, rather than dropping a frame before sampling as Qwen2-VL does.
    """
    fps = _value(configs, "fps", 2)
    min_frames = _value(configs, "min_frames", 4)
    max_frames = _value(configs, "max_frames", 768)
    if total_frames <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(
            "video requires a positive frame count and finite positive FPS"
        )
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video sampling FPS must be finite and positive")
    if min_frames <= 0 or max_frames < min_frames:
        raise ValueError("video frame limits must satisfy 0 < min_frames <= max_frames")
    count = min(
        max(int(total_frames / source_fps * fps), min_frames), max_frames, total_frames
    )
    return np.linspace(0, total_frames - 1, count).round().astype(int).tolist()


def resolve_video_size(
    processor: Any, total_min_pixels: int = 0, total_max_pixels: int = 0
) -> Dict[str, int]:
    """Resolve service overrides without mutating the shared HF processor."""
    if total_min_pixels < 0 or total_max_pixels < 0:
        raise ValueError("video total pixel overrides must be non-negative")
    size = {
        "shortest_edge": total_min_pixels or processor.size["shortest_edge"],
        "longest_edge": total_max_pixels or processor.size["longest_edge"],
    }
    if not 0 < size["shortest_edge"] <= size["longest_edge"]:
        raise ValueError("video total pixel limits must satisfy 0 < min <= max")
    return size


def video_resize_shape(
    configs: Any,
    num_frames: int,
    height: int,
    width: int,
    processor: Any,
    size: Dict[str, int],
) -> Tuple[int, int]:
    """Plan the exact HF video grid, including temporal/spatial rounding.

    Existing RTP request-level resized dimensions and per-frame min/max_pixels
    remain explicit overrides. Service defaults use HF's total-video budget.
    """
    factor = processor.patch_size * processor.merge_size
    requested_h = _value(configs, "height", None)
    requested_w = _value(configs, "width", None)
    if requested_h is not None or requested_w is not None:
        if (
            requested_h is None
            or requested_w is None
            or min(requested_h, requested_w) <= 0
        ):
            raise ValueError(
                "video resized_height and resized_width must both be positive"
            )
        return max(factor, round(requested_h / factor) * factor), max(
            factor, round(requested_w / factor) * factor
        )
    min_pixels = size["shortest_edge"]
    max_pixels = size["longest_edge"]
    # Match the optional policy in Transformers releases that expose it.
    if getattr(processor, "cap_pixels_per_frame", False):
        frame_cap = processor.max_video_tokens * factor * factor
        max_pixels = (
            max(min(frame_cap, max_pixels // num_frames), int(min_pixels * 1.05))
            * num_frames
        )
    per_frame_min = _value(configs, "min_pixels", None)
    per_frame_max = _value(configs, "max_pixels", None)
    if per_frame_min is not None:
        min_pixels = per_frame_min * num_frames
    if per_frame_max is not None:
        max_pixels = per_frame_max * num_frames
    if not 0 < min_pixels <= max_pixels:
        raise ValueError("video pixel limits must satisfy 0 < min <= max")
    return smart_resize(
        num_frames=num_frames,
        height=height,
        width=width,
        temporal_factor=processor.temporal_patch_size,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def decode_video(video_data: Any, configs: Any):
    """Decode sampled native RGB frames; resize only after the video policy resolves."""
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError("decord is required for Qwen3-VL video processing") from exc
    reader = VideoReader(video_data, ctx=cpu(0), num_threads=1)
    total, fps = len(reader), reader.get_avg_fps()
    indices = sample_frame_indices(total, fps, configs)
    video = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
    metadata = VideoMetadata(
        total_num_frames=total, fps=fps, duration=total / fps, frames_indices=indices
    )
    return video, metadata


def resize_video(
    video: torch.Tensor, configs: Any, processor: Any, size: Dict[str, int]
):
    """Use HF's torchvision resize semantics on uint8, before normalization."""
    height, width = video_resize_shape(
        configs, video.shape[0], video.shape[-2], video.shape[-1], processor, size
    )
    return resize_video_to_shape(video, processor, height, width)


def resize_video_to_shape(video: torch.Tensor, processor: Any, height: int, width: int):
    """Use the processor's underlying resize backend, including uint8 rounding."""
    # Transformers 5.2 takes interpolation; newer backends take resample.
    # Passing resample through the old **kwargs silently selects bilinear.
    if "interpolation" in inspect.signature(BaseVideoProcessor.resize).parameters:
        from transformers.image_utils import pil_torch_interpolation_mapping

        return BaseVideoProcessor.resize(
            processor,
            image=video,
            size=SizeDict(height=height, width=width),
            interpolation=pil_torch_interpolation_mapping.get(
                processor.resample, processor.resample
            ),
        )
    return BaseVideoProcessor.resize(
        processor,
        image=video.unsqueeze(0),
        size=SizeDict(height=height, width=width),
        resample=processor.resample,
    )[0]


def video_timestamps(frame_indices, source_fps, temporal_patch_size):
    """Match vLLM Qwen3VLProcessingInfo._calculate_timestamps."""
    indices = list(frame_indices)
    if not indices or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("video timestamps require frame indices and positive FPS")
    if temporal_patch_size <= 0:
        raise ValueError("temporal patch size must be positive")
    padding = (-len(indices)) % temporal_patch_size
    indices += [indices[-1]] * padding
    seconds = [index / source_fps for index in indices]
    return [
        (seconds[i] + seconds[i + temporal_patch_size - 1]) / 2
        for i in range(0, len(indices), temporal_patch_size)
    ]


def video_token_layout(grid_thw, frame_indices, source_fps, processor):
    """Compact prompt replacement: token IDs plus -N for N vision features.

    Each temporal patch gets its own timestamp and vision start/end pair.
    Timestamp strings are encoded independently, as in vLLM get_video_repl.
    """
    if tuple(grid_thw.shape) != (1, 3):
        raise ValueError("one video grid is required per video layout")
    t, h, w = grid_thw[0].tolist()
    video_processor = processor.video_processor
    timestamps = video_timestamps(
        frame_indices, source_fps, video_processor.temporal_patch_size
    )
    merge = video_processor.merge_size
    if len(timestamps) != t or h <= 0 or w <= 0 or h % merge or w % merge:
        raise ValueError("timestamps and video grid do not match")
    tokens_per_frame = (h // merge) * (w // merge)
    tokenizer = processor.tokenizer
    start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if start is None or end is None:
        raise ValueError("video tokenizer lacks vision delimiters")
    layout = []
    for timestamp in timestamps:
        layout.extend(
            tokenizer.encode(f"<{timestamp:.1f} seconds>", add_special_tokens=False)
        )
        layout.extend([start, -tokens_per_frame, end])
    return torch.tensor(layout, dtype=torch.int32)
