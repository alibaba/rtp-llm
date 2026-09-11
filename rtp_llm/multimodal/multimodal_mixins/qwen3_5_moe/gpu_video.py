"""Deferred NVDEC video input; GPU work executes on the embedding scheduler."""

import importlib
import io
import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import Tuple

import _imp
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.qwen2_5_vl_mixin import (
    smart_nframes,
    video_resize_shape,
)


@dataclass(frozen=True)
class GpuVideoInput:
    """Compressed media and CPU metadata, safe to return from a spawn worker."""

    encoded: bytes
    total_frames: int
    frame_indices: Tuple[int, ...]
    height: int
    width: int
    resized_height: int
    resized_width: int
    patch_size: int
    temporal_patch_size: int
    color_space: int = 2

    @property
    def shape(self):
        temporal = math.ceil(len(self.frame_indices) / self.temporal_patch_size)
        return (
            temporal
            * (self.resized_height // self.patch_size)
            * (self.resized_width // self.patch_size),
            3 * self.temporal_patch_size * self.patch_size**2,
        )

    @property
    def workspace_bytes(self):
        # NVDEC surfaces, expanded integer color-conversion intermediates,
        # selected frames, and float resize/processor buffers.
        return 48 * self.height * self.width + len(self.frame_indices) * (
            64 * self.height * self.width
            + 36 * self.resized_height * self.resized_width
        )


def prepare_gpu_video(encoded, configs, processor, factor):
    """Determine the same frame/grid policy without creating a CUDA context."""
    import av

    with av.open(io.BytesIO(encoded)) as container:
        if not container.streams.video:
            raise ValueError("video input has no video stream")
        stream = container.streams.video[0]
        total_frames = stream.frames
        fps = float(stream.average_rate or 0)
        height, width = stream.height, stream.width
        context = stream.codec_context
        color_space = int(context.colorspace)
        if (
            context.format.name != "yuv420p"
            or int(context.color_range) not in (0, 1)
            or color_space not in (1, 2, 5, 6)
        ):
            raise ValueError(
                "NVDEC video preprocessing supports limited-range 8-bit YUV420 with BT.601/709 color; use the cpu backend for this input"
            )
    if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            "NVDEC video preprocessing requires an indexed video with a frame "
            "count and average frame rate; use the cpu video backend for this input"
        )
    count = smart_nframes(configs, total_frames, fps)
    indices = tuple(torch.linspace(0, total_frames - 1, count).round().long().tolist())
    out_h, out_w = video_resize_shape(
        configs,
        count,
        height,
        width,
        factor,
        processor.size.get("longest_edge"),
    )
    data = GpuVideoInput(
        encoded,
        total_frames,
        indices,
        height,
        width,
        out_h,
        out_w,
        processor.patch_size,
        processor.temporal_patch_size,
        color_space,
    )
    grid = torch.tensor(
        [
            [
                math.ceil(count / processor.temporal_patch_size),
                out_h // processor.patch_size,
                out_w // processor.patch_size,
            ]
        ],
        dtype=torch.int64,
    )
    return data, grid


def _load_nvcodec():
    # Decord loads FFmpeg globally. Bind NVDEC to its own newer FFmpeg ABI
    # instead of resolving those symbols from Decord's incompatible version.
    # Serialize imports and restore the process flags even when loading fails.
    _imp.acquire_lock()
    flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags((flags | os.RTLD_DEEPBIND) & ~os.RTLD_GLOBAL)
        return importlib.import_module("PyNvVideoCodec")
    except ImportError as exc:
        raise RuntimeError(
            "QWEN35_VIDEO_BACKEND=nvdec requires PyNvVideoCodec and the matching "
            "NVIDIA libnvcuvid.so and libnvidia-encode.so driver libraries "
            "in LD_LIBRARY_PATH"
        ) from exc
    finally:
        sys.setdlopenflags(flags)
        _imp.release_lock()


def nv12_to_rgb(video, height, color_space):
    """Match Decord/x86 swscale's limited-range fixed-point RGB conversion."""
    # BT.709 and BT.601 coefficients scaled by 8192. Each contribution is
    # truncated before addition, matching the CPU's signed high-word multiply.
    cy, cr, cu, cgu, cgv = (
        (9539, 14686, 17305, -1747, -4366)
        if color_space == 1
        else (9539, 13075, 16525, -3209, -6660)
    )
    n, _, width = video.shape
    y = ((video[:, :height].to(torch.int32) - 16) * cy) >> 13
    chroma = video[:, height:].reshape(n, height // 2, width // 2, 2)
    chroma = chroma.permute(0, 3, 1, 2).to(torch.int32) - 128
    chroma = chroma.repeat_interleave(2, 2).repeat_interleave(2, 3)
    u, v = chroma[:, 0], chroma[:, 1]
    rgb = torch.stack(
        [
            y + ((v * cr) >> 13),
            y + ((u * cgu) >> 13) + ((v * cgv) >> 13),
            y + ((u * cu) >> 13),
        ],
        dim=1,
    )
    return rgb.clamp_(0, 255).to(torch.uint8)


# One hardware session per scheduler thread. No media, frames, or outputs are
# retained: every request creates a fresh demuxer and consumes through EOS.
_decode_state = threading.local()


@torch.inference_mode()
def decode_video_cuda(data: GpuVideoInput, device):
    """Decode in display order and copy sampled frames directly into CUDA."""
    nvc = _load_nvcodec()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("NVDEC preprocessing requires a CUDA embedding device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device), torch.profiler.record_function("video_nvdec"):
        stream = torch.cuda.current_stream(device)
        offset = 0

        def feed(buffer):
            nonlocal offset
            chunk = data.encoded[offset : offset + len(buffer)]
            buffer[: len(chunk)] = chunk
            offset += len(chunk)
            return len(chunk)

        demuxer = nvc.CreateDemuxer(feed)
        # Dimensions and codec changes create a new bounded hardware session.
        key = (
            device.index,
            stream.cuda_stream,
            demuxer.GetNvCodecId(),
            data.width,
            data.height,
        )
        if getattr(_decode_state, "key", None) != key:
            _decode_state.decoder = None
            _decode_state.key = None
            _decode_state.decoder = nvc.CreateDecoder(
                gpuid=device.index,
                codec=demuxer.GetNvCodecId(),
                cudacontext=0,
                cudastream=stream.cuda_stream,
                usedevicememory=True,
                outputColorType=nvc.OutputColorType.NATIVE,
            )
            _decode_state.key = key
        decoder = _decode_state.decoder
        wanted = set(data.frame_indices)
        selected = {}
        count = 0
        try:
            # Include the demuxer's EOS packet: draining delayed/B-frames is
            # required both for display-order indices and safe session reuse.
            for packet in demuxer:
                frames = decoder.Decode(packet)
                copied = False
                for frame in frames:
                    if count in wanted:
                        tensor = torch.from_dlpack(frame)
                        if (
                            tensor.device != device
                            or tensor.dtype != torch.uint8
                            or tuple(tensor.shape) != (data.height * 3 // 2, data.width)
                        ):
                            raise ValueError(
                                "NVDEC returned an unexpected frame shape/device/dtype"
                            )
                        selected[count] = tensor.clone()
                        copied = True
                    count += 1
                if copied:
                    # The next Decode may recycle its external surfaces.
                    stream.synchronize()
            if count != data.total_frames or selected.keys() != wanted:
                raise ValueError("NVDEC and container frame counts disagree")
            # Explicit indexing also preserves duplicates in a sampling policy.
            return nv12_to_rgb(
                torch.stack([selected[index] for index in data.frame_indices]),
                data.height,
                data.color_space,
            )
        except Exception:
            # A failed/incompletely drained session must not reach the next request.
            _decode_state.decoder = None
            _decode_state.key = None
            raise


@torch.inference_mode()
def preprocess_video_cuda(data: GpuVideoInput, processor, device):
    video = decode_video_cuda(data, device)
    with torch.profiler.record_function("video_resize"):
        # Match the CPU path: resize uint8 with antialias, then convert to FP32.
        video = resize(
            video,
            [data.resized_height, data.resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
    with torch.profiler.record_function("video_processor"):
        result = processor(
            video,
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        )
    pixels = result["pixel_values_videos"]
    if not pixels.is_cuda or tuple(pixels.shape) != data.shape:
        raise ValueError("video processor changed the device or expected patch shape")
    expected_grid = [
        [
            math.ceil(len(data.frame_indices) / data.temporal_patch_size),
            data.resized_height // data.patch_size,
            data.resized_width // data.patch_size,
        ]
    ]
    if result["video_grid_thw"].tolist() != expected_grid:
        raise ValueError("video processor changed the expected grid")
    return pixels
