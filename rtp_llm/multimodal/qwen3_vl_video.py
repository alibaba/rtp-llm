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
    total_frames: int, source_fps: float, configs: Any, *, vit_config: Any = None
) -> List[int]:
    """Use vLLM's integer frame count and float64 uniform frame indices.

    Odd counts are intentional: the video processor pads the final temporal
    patch, rather than dropping a frame before sampling as Qwen2-VL does.
    Missing/None/-1 request fields inherit the service sampling defaults.
    Never mutate the request or read process-global environment in a worker.
    """
    fps = _value(configs, "fps", getattr(vit_config, "mm_video_fps", 2))
    min_frames = _value(configs, "min_frames", 4)
    max_frames = _value(
        configs, "max_frames", getattr(vit_config, "mm_video_max_frames", 768)
    )
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


def _rewind_video(video_data):
    if hasattr(video_data, "seek"):
        video_data.seek(0)


def _pyav_video_info(video_data):
    """Read indexed metadata, counting frames only for unindexed containers."""
    import av

    _rewind_video(video_data)
    with av.open(video_data) as container:
        if not container.streams.video:
            raise ValueError("video input has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or stream.guessed_rate or 0)
        total = stream.frames
        height, width = stream.height, stream.width
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("video requires a finite positive FPS")
        if total <= 0:
            total = 0
            for frame in container.decode(stream):
                total += 1
                height, width = frame.height, frame.width
        if total <= 0 or min(height, width) <= 0:
            raise ValueError("video input has no decodable frames")
    _rewind_video(video_data)
    return total, fps, height, width


def read_video_metadata(video_data):
    """Use the same decoder metadata as the CPU pixel path, including ARM."""
    try:
        from decord import VideoReader, cpu
    except ImportError:
        return _pyav_video_info(video_data)
    import av

    _rewind_video(video_data)
    with av.open(video_data) as container:
        if not container.streams.video:
            raise ValueError("video input has no video stream")
        stream = container.streams.video[0]
        height, width = stream.height, stream.width
    _rewind_video(video_data)
    reader = VideoReader(video_data, ctx=cpu(0), num_threads=1)
    return len(reader), reader.get_avg_fps(), height, width


def _decode_video_pyav(video_data, configs, vit_config):
    import av

    total, fps, _, _ = _pyav_video_info(video_data)
    indices = sample_frame_indices(total, fps, configs, vit_config=vit_config)
    wanted = set(indices)
    frames = []
    with av.open(video_data) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                frames.append(frame.to_ndarray(format="rgb24"))
            if index >= indices[-1]:
                break
    if len(frames) != len(indices):
        raise ValueError("decoded video frame count does not match metadata")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return video, VideoMetadata(
        total_num_frames=total, fps=fps, duration=total / fps, frames_indices=indices
    )


def decode_video(video_data: Any, configs: Any, *, vit_config: Any = None):
    """Decode sampled native RGB frames; resize only after the video policy resolves."""
    try:
        from decord import VideoReader, cpu
    except ImportError:
        return _decode_video_pyav(video_data, configs, vit_config)
    reader = VideoReader(video_data, ctx=cpu(0), num_threads=1)
    total, fps = len(reader), reader.get_avg_fps()
    indices = sample_frame_indices(total, fps, configs, vit_config=vit_config)
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


def video_timestamp_tokens(grid_thw, frame_indices, source_fps, processor):
    """Tokenize each temporal patch timestamp for mixin-local word embedding."""
    if tuple(grid_thw.shape) != (1, 3):
        raise ValueError("one video grid is required per video")
    t, h, w = grid_thw[0].tolist()
    vp = processor.video_processor
    timestamps = video_timestamps(frame_indices, source_fps, vp.temporal_patch_size)
    if (
        len(timestamps) != t
        or h <= 0
        or w <= 0
        or h % vp.merge_size
        or w % vp.merge_size
    ):
        raise ValueError("timestamps and video grid do not match")
    return [
        processor.tokenizer.encode(
            f"<{timestamp:.1f} seconds>", add_special_tokens=False
        )
        for timestamp in timestamps
    ]
