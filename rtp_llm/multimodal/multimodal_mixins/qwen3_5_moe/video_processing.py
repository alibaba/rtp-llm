"""Qwen3.5 frame sampling and resize policy for NVDEC preprocessing.

Sampling follows vLLM's Qwen3VLVideoBackend. Resizing follows the Qwen3-VL
video processor in Transformers 5.15: size bounds are T * H * W pixel budgets.
"""

import math

import numpy as np


def video_frame_indices(configs, total_frames, video_fps):
    """Sample once, without dropping an odd final frame to make T divisible by 2."""

    def option(name, default):
        value = getattr(configs, name, -1)
        return default if value in (None, -1) else value

    fps = option("fps", 2)
    min_frames = option("min_frames", 4)
    max_frames = option("max_frames", 768)
    if total_frames <= 0 or not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("video requires a positive frame count and frame rate")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("video fps must be positive")
    if (
        not isinstance(min_frames, int)
        or not isinstance(max_frames, int)
        or not 0 < min_frames <= max_frames
    ):
        raise ValueError("video frame limits must satisfy 0 < min_frames <= max_frames")

    # Match vLLM's qwen3_vl loader, keeping spatial resizing in the processor.
    count = int(total_frames / video_fps * fps)
    count = min(max(count, min_frames), max_frames, total_frames)
    return tuple(np.linspace(0, total_frames - 1, count).round().astype(int).tolist())


def video_processor_size(processor, configs=None):
    """Apply request pixel budgets over the model's video processor defaults."""
    size = dict(processor.size)
    for name, edge in (("min_pixels", "shortest_edge"), ("max_pixels", "longest_edge")):
        # Request pixel controls describe total video pixels, not pixels per frame.
        value = getattr(configs, name, -1)
        if value not in (None, -1):
            size[edge] = value
    minimum, maximum = size["shortest_edge"], size["longest_edge"]
    if not 0 < minimum <= maximum or not math.isfinite(maximum):
        raise ValueError(
            "video pixel budgets must satisfy 0 < shortest_edge <= longest_edge"
        )
    return size


def video_resize_shape(configs, num_frames, height, width, processor, factor=32):
    """Return H/W using the Qwen3-VL whole-video smart_resize algorithm."""
    size = video_processor_size(processor, configs)
    temporal_factor = processor.temporal_patch_size
    if num_frames < temporal_factor:
        raise ValueError(
            f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}"
        )
    # Preserve explicit RTP request dimensions, while still applying the total
    # video budget to them.
    if getattr(configs, "height", -1) != -1 and getattr(configs, "width", -1) != -1:
        height, width = configs.height, configs.width
    if height <= 0 or width <= 0:
        raise ValueError("video height and width must be positive")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute video aspect ratio must be smaller than 200")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > size["longest_edge"]:
        beta = math.sqrt(num_frames * height * width / size["longest_edge"])
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < size["shortest_edge"]:
        beta = math.sqrt(size["shortest_edge"] / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
