"""Request-owned NVDEC inputs and bounded concurrent GPU preprocessing."""

import copy
import importlib
import io
import math
import os
import sys
import threading
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Tuple

import _imp
import torch

from rtp_llm.multimodal.qwen3_vl_video import (
    resize_video_to_shape,
    resolve_video_size,
    sample_frame_indices,
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
    source_fps: float = 0.0

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


@dataclass
class PreparedGpuVideo:
    """Request-owned patches; the embedding stream waits for their producer."""

    source: GpuVideoInput
    pixels: torch.Tensor
    ready: torch.cuda.Event

    @property
    def shape(self):
        return self.source.shape

    @property
    def frame_indices(self):
        return self.source.frame_indices

    @property
    def resized_height(self):
        return self.source.resized_height

    @property
    def resized_width(self):
        return self.source.resized_width

    def consume(self, stream):
        stream.wait_event(self.ready)
        self.pixels.record_stream(stream)
        return self.pixels


def prepare_gpu_video(encoded, configs, processor, factor, size=None):
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
    indices = tuple(sample_frame_indices(total_frames, fps, configs))
    count = len(indices)
    out_h, out_w = video_resize_shape(
        configs,
        count,
        height,
        width,
        processor,
        size if size is not None else resolve_video_size(processor),
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
        fps,
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
    if video.is_cuda and video.is_contiguous():
        from .vision_kernels import nv12_rgb

        return nv12_rgb(video, height, color_space)
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


# One hardware session per decoding thread. No media, frames, or outputs are
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
        consumer_stream = torch.cuda.current_stream(device)
        if getattr(_decode_state, "device", None) != device:
            _decode_state.stream = torch.cuda.Stream(device=device)
            _decode_state.device = device
        stream = _decode_state.stream
        with torch.cuda.stream(stream):
            return _decode_on_stream(data, device, nvc, stream, consumer_stream)


def _decode_on_stream(data, device, nvc, stream, consumer_stream):
    # A dedicated decode stream keeps surface-lifetime waits from draining the
    # preceding ViT forward on the consumer stream. Only sampled frames cross
    # the stream boundary; no frame or embedding is cached across requests.
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
    positions = {}
    for slot, index in enumerate(data.frame_indices):
        positions.setdefault(index, []).append(slot)
    wanted = set(positions)
    selected = set()
    # Copy sampled surfaces straight into their final ordered buffer. The
    # decoder owns/recycles its surfaces, so the copy and lifetime wait remain.
    sampled = torch.empty(
        (len(data.frame_indices), data.height * 3 // 2, data.width),
        device=device,
        dtype=torch.uint8,
    )
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
                    for slot in positions[count]:
                        sampled[slot].copy_(tensor)
                    selected.add(count)
                    copied = True
                count += 1
            if copied:
                # The next Decode may recycle its external surfaces.
                stream.synchronize()
        if count != data.total_frames or selected != wanted:
            raise ValueError("NVDEC and container frame counts disagree")
        # Explicit indexing also preserves duplicates in a sampling policy.
        consumer_stream.wait_stream(stream)
        with torch.cuda.stream(consumer_stream):
            sampled.record_stream(consumer_stream)
            return nv12_to_rgb(sampled, data.height, data.color_space)
    except Exception:
        # A failed/incompletely drained session must not reach the next request.
        _decode_state.decoder = None
        _decode_state.key = None
        raise


class VideoDecodePool:
    """Bounded worker pool for ordered decode and full GPU preprocessing."""

    def __init__(self, max_workers: int):
        if max_workers < 1:
            raise ValueError("video decode workers must be positive")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="qwen35-nvdec"
        )
        # The pool belongs to the vision module, without retaining the module.
        self._finalizer = weakref.finalize(
            self, self._executor.shutdown, wait=False, cancel_futures=True
        )

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._finalizer.detach()

    @contextmanager
    def decode(self, inputs, device):
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        with self._map(_decode_video_ready, inputs, device) as results:

            def recorded():
                for video in results:
                    video.record_stream(torch.cuda.current_stream(device))
                    yield video

            yield recorded()

    @contextmanager
    def preprocess(self, inputs, processor, device, dtype):
        with self._map(
            _preprocess_video_ready, inputs, processor, device, dtype
        ) as results:
            yield results

    @contextmanager
    def _map(self, function, inputs, *args):
        inputs = iter(inputs)
        pending = deque()

        def submit_next():
            data = next(inputs, None)
            if data is not None:
                pending.append(self._executor.submit(function, data, *args))

        def results():
            for _ in range(self.max_workers):
                submit_next()
            while pending:
                future = pending.popleft()
                video = future.result()
                yield video
                del video, future
                submit_next()

        iterator = results()
        try:
            yield iterator
        finally:
            iterator.close()
            for future in pending:
                future.cancel()
            # Drain work on failure/early exit before accepting another batch.
            wait(pending)
            pending.clear()


@torch.inference_mode()
def _decode_video_ready(data, device):
    with torch.cuda.device(device):
        if getattr(_decode_state, "output_device", None) != device:
            _decode_state.output_stream = torch.cuda.Stream(device=device)
            _decode_state.output_device = device
        stream = _decode_state.output_stream
        with torch.cuda.stream(stream):
            video = decode_video_cuda(data, device)
            stream.synchronize()
            return video


@torch.inference_mode()
def _preprocess_video_ready(data, processor, device, dtype):
    device = torch.device(device)
    with torch.cuda.device(device):
        if getattr(_decode_state, "preprocess_device", None) != device:
            _decode_state.preprocess_stream = torch.cuda.Stream(device=device)
            _decode_state.preprocess_device = device
        stream = _decode_state.preprocess_stream
        if (
            getattr(_decode_state, "processor_source", None) is not processor
            or getattr(_decode_state, "processor_device", None) != device
        ):
            # Transformers caches CUDA mean/std tensors by processor instance.
            # A thread-local copy keeps their initialization and reuse on this
            # worker's stream rather than racing another producer stream.
            _decode_state.preprocess_processor = copy.copy(processor)
            _decode_state.processor_source = processor
            _decode_state.processor_device = device
        with torch.cuda.stream(stream), torch.profiler.record_function("video_prepare"):
            pixels = preprocess_video_cuda(
                data, _decode_state.preprocess_processor, device
            ).to(dtype=dtype)
            ready = torch.cuda.Event()
            ready.record(stream)
            return PreparedGpuVideo(data, pixels, ready)


def process_video_pixels(video, processor):
    """Qwen3-VL single-video patch order without singleton stack/cat copies."""
    patches = processor.rescale_and_normalize(
        video.unsqueeze(0),
        processor.do_rescale,
        processor.rescale_factor,
        processor.do_normalize,
        (
            tuple(processor.image_mean)
            if isinstance(processor.image_mean, list)
            else processor.image_mean
        ),
        (
            tuple(processor.image_std)
            if isinstance(processor.image_std, list)
            else processor.image_std
        ),
    )
    temporal = processor.temporal_patch_size
    patch = processor.patch_size
    merge = processor.merge_size
    frames = patches.shape[1]
    if pad := -frames % temporal:
        patches = torch.cat(
            (patches, patches[:, -1:].expand(-1, pad, -1, -1, -1)), dim=1
        )
    batch, frames, channels, height, width = patches.shape
    grid_t, grid_h, grid_w = frames // temporal, height // patch, width // patch
    patches = patches.view(
        batch,
        grid_t,
        temporal,
        channels,
        grid_h // merge,
        merge,
        patch,
        grid_w // merge,
        merge,
        patch,
    ).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return patches.reshape(
        grid_t * grid_h * grid_w, channels * temporal * patch * patch
    )


@torch.inference_mode()
def preprocess_video_cuda(data: GpuVideoInput, processor, device, *, video=None):
    if video is None:
        video = decode_video_cuda(data, device)
    with torch.profiler.record_function("video_resize"):
        # Keep decoded frames on CUDA through resize and normalization.
        # Preserve the processor's bicubic/antialias and uint8 conversion policy;
        # CPU and CUDA backends can differ in arithmetic and uint8 rounding.
        if tuple(video.shape[-2:]) != (data.resized_height, data.resized_width):
            video = resize_video_to_shape(
                video, processor, data.resized_height, data.resized_width
            )
    with torch.profiler.record_function("video_processor"):
        pixels = process_video_pixels(video, processor)
    if not pixels.is_cuda or tuple(pixels.shape) != data.shape:
        raise ValueError("video processor changed the device or expected patch shape")
    expected_grid = [
        [
            math.ceil(len(data.frame_indices) / data.temporal_patch_size),
            data.resized_height // data.patch_size,
            data.resized_width // data.patch_size,
        ]
    ]
    expected_patches = math.prod(expected_grid[0])
    if pixels.shape[0] != expected_patches:
        raise ValueError("video processor changed the expected grid")
    return pixels
