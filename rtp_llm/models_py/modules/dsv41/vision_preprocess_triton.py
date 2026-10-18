# The geometry/filter/coefficient/uint8-rounding algorithms below are derived
# from Pillow 11.2.1 ImageOps.py and src/libImaging/Resample.c. This is a modified
# Python/Triton implementation, not an upstream Pillow implementation.
# Pillow 11.2.1 LICENSE (MIT-CMU), retained for these derived portions:
#
# The Python Imaging Library (PIL) is
#
#     Copyright © 1997-2011 by Secret Labs AB
#     Copyright © 1995-2011 by Fredrik Lundh and contributors
#
# Pillow is the friendly PIL fork. It is
#
#     Copyright © 2010 by Jeffrey A. Clark and contributors
#
# Like PIL, Pillow is licensed under the open source MIT-CMU License:
#
# By obtaining, using, and/or copying this software and/or its associated
# documentation, you agree that you have read, understood, and will comply
# with the following terms and conditions:
#
# Permission to use, copy, modify and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appears in all copies, and that
# both that copyright notice and this permission notice appear in supporting
# documentation, and that the name of Secret Labs AB or the author not be
# used in advertising or publicity pertaining to distribution of the software
# without specific, written prior permission.
#
# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
# SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
# IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

"""Experimental Pillow-compatible RGB bicubic preprocessing for DeepSeek V4.1.

The caller owns RGB conversion, the original-image H2D, the model's grid policy,
and the explicit GPU opt-in/fallback. This module takes a contiguous CUDA
uint8 HWC RGB image, applies ImageOps.pad(BICUBIC, color=127), and returns owned
contiguous BF16 patches. It does not decode images or copy pixels back to CPU.

Compatibility reference: Pillow 11.2.1, ImageOps.contain/pad and the RGB8 path in
https://github.com/python-pillow/Pillow/blob/11.2.1/src/libImaging/Resample.c
Both separable passes round/clip to uint8. Integer coefficients have 22 fraction
bits; CPU double arithmetic preserves the reference coefficient construction.
The small CPU normalization table uses the original three FP32 Torch operations
before BF16 rounding. No floating-point resize or fused normalization is used.

CPU tables are bounded. V41_GPU_PREPROCESS_TABLE_CACHE=1 additionally enables a
bounded device cache of immutable tables; cold uploads stay inside this call.
Output/workspace storage is never shared. Work is queued on the
input device's current stream, without a device-wide synchronization. As usual,
callers must make cross-stream inputs ready and maintain/record their lifetime.
CUDA graph capture is explicitly unsupported in this first implementation.
"""

from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from functools import lru_cache

import torch
import triton
import triton.language as tl
from PIL import __version__ as pillow_version

_PILLOW_VERSION = "11.2.1"
_PRECISION_BITS = 22
_MAX_DIMENSION = 65536
_MAX_TAPS = 1025
_MAX_COEFFICIENTS = 1 << 20
_MAX_ELEMENTS = (1 << 31) - 1


def _geometry(in_h: int, in_w: int, out_h: int, out_w: int):
    """Pillow contain dimensions and centered pad, including ties-to-even."""
    resize_h, resize_w = out_h, out_w
    source_ratio, target_ratio = in_w / in_h, out_w / out_h
    if source_ratio > target_ratio:
        resize_h = round(in_h / in_w * out_w)
    elif source_ratio < target_ratio:
        resize_w = round(in_w / in_h * out_h)
    return (
        resize_h,
        resize_w,
        round((out_h - resize_h) * 0.5),
        round((out_w - resize_w) * 0.5),
    )


def _tap_count(in_size: int, out_size: int) -> int:
    return math.ceil(2.0 * max(in_size / out_size, 1.0)) * 2 + 1


def is_supported_shape(
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    *,
    patch_size: int = 14,
) -> bool:
    """CPU metadata gate, usable before H2D; never touches the CUDA runtime."""
    if (
        pillow_version != _PILLOW_VERSION
        or type(patch_size) is not int
        or not 1 <= patch_size <= 64
        or not isinstance(source_hw, (tuple, list))
        or len(source_hw) != 2
        or any(type(size) is not int for size in source_hw)
        or not isinstance(target_hw, (tuple, list))
        or len(target_hw) != 2
        or any(type(size) is not int for size in target_hw)
    ):
        return False
    in_h, in_w = source_hw
    out_h, out_w = target_hw
    if (
        any(not 1 <= size <= _MAX_DIMENSION for size in (in_h, in_w, out_h, out_w))
        or out_h % patch_size
        or out_w % patch_size
        or max(in_h * in_w, out_h * out_w) * 3 > _MAX_ELEMENTS
    ):
        return False
    resize_h, resize_w, _, _ = _geometry(in_h, in_w, out_h, out_w)
    if min(resize_h, resize_w) < 1 or in_h * resize_w * 3 > _MAX_ELEMENTS:
        return False
    for old, new in ((in_w, resize_w), (in_h, resize_h)):
        if old != new:
            taps = _tap_count(old, new)
            if taps > _MAX_TAPS or new * taps > _MAX_COEFFICIENTS:
                return False
    return True


def is_supported(
    rgb: torch.Tensor, target_hw: tuple[int, int], *, patch_size: int = 14
) -> bool:
    """Non-mutating tensor/runtime gate; the caller retains the CPU fallback.

    This does not consume an opt-in flag: the production call site owns that
    policy. Restrictions bound table-cache memory and signed 32-bit indexing.
    """
    if (
        not isinstance(rgb, torch.Tensor)
        or not rgb.is_cuda
        or rgb.dtype != torch.uint8
        or rgb.layout != torch.strided
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or not rgb.is_contiguous()
        or not is_supported_shape(
            tuple(rgb.shape[:2]), target_hw, patch_size=patch_size
        )
    ):
        return False
    with torch.cuda.device(rgb.device):
        if torch.cuda.is_current_stream_capturing():
            return False
    return True


def _cubic(distance: float) -> float:
    """Catmull-Rom filter in the reference's double operation order."""
    x = abs(distance)
    if x < 1.0:
        return (1.5 * x - 2.5) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5
    return 0.0


@lru_cache(maxsize=16)
def _coefficients(in_size: int, out_size: int):
    """Private read-only CPU tensors: (start,count) and signed 22-bit weights.

    The full-box endpoints are integer-exact in float32 under our size gate.
    Python int truncates toward zero like the reference C cast; round() would
    be wrong here. Python's scalar double loop deliberately avoids NumPy's
    reduction reassociation. Public callers cannot mutate these cached tables.
    """
    scale = in_size / out_size
    filter_scale = max(scale, 1.0)
    support = 2.0 * filter_scale
    taps = _tap_count(in_size, out_size)
    inv_filter_scale = 1.0 / filter_scale
    bounds, coefficients = [], []
    for dest in range(out_size):
        center = (dest + 0.5) * scale
        first = max(int(center - support + 0.5), 0)
        last = min(int(center + support + 0.5), in_size)
        weights, total = [], 0.0
        for source in range(first, last):
            value = _cubic((source - center + 0.5) * inv_filter_scale)
            weights.append(value)
            total += value
        quantized = []
        for value in weights:
            if total != 0.0:
                value /= total
            fixed = value * (1 << _PRECISION_BITS)
            quantized.append(int(fixed + (-0.5 if value < 0.0 else 0.5)))
        bounds.append((first, last - first))
        coefficients.append(quantized + [0] * (taps - len(quantized)))
    return (
        torch.tensor(bounds, dtype=torch.int32, device="cpu"),
        torch.tensor(coefficients, dtype=torch.int32, device="cpu"),
    )


@lru_cache(maxsize=1)
def _normalization_lut():
    # Keep these exact CPU FP32 operation boundaries; do not replace by u/127.5-1.
    values = torch.arange(256, dtype=torch.float32, device="cpu") / 255
    return ((values - 0.5) / 0.5).to(torch.bfloat16)


class _DeviceTableCache:
    """Bounded, immutable tables, never request pixels or computed embeddings.

    Readiness events order first use on another stream. record_stream keeps
    every allocation alive through queued consumers even after LRU eviction.
    The bounds cover all devices together; no device is initialized at import.
    """

    def __init__(self, max_entries=32, max_bytes=8 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, device, key, factory):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("device tables require CUDA")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("device table lookup does not support graph capture")
            stream = torch.cuda.current_stream(device)
            # Version the table contract and resolve an implicit CUDA index.
            cache_key = (1, stream.device.index, *key)
            with self._lock:
                entry = self._entries.pop(cache_key, None)
                if entry is None:
                    tables = tuple(t.to(stream.device) for t in factory())
                    ready = torch.cuda.Event()
                    ready.record(stream)
                    size = sum(t.numel() * t.element_size() for t in tables)
                    entry = (tables, ready, size)
                    if size <= self.max_bytes and self.max_entries > 0:
                        while self._entries and (
                            len(self._entries) >= self.max_entries
                            or self._bytes + size > self.max_bytes
                        ):
                            _, retired = self._entries.popitem(last=False)
                            self._bytes -= retired[2]
                        self._entries[cache_key] = entry
                        self._bytes += size
                else:
                    self._entries[cache_key] = entry
                tables, ready, _ = entry
                stream.wait_event(ready)
                for table in tables:
                    table.record_stream(stream)
                return tables


_DEVICE_TABLES = _DeviceTableCache()


def _tables_on_device(device, key, factory):
    if os.environ.get("V41_GPU_PREPROCESS_TABLE_CACHE", "0") == "1":
        return _DEVICE_TABLES.get(device, key, factory)
    return tuple(table.to(device) for table in factory())


@triton.jit
def _horizontal_u8(
    RGB,
    TEMP,
    BOUNDS,
    COEFFICIENTS,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    RESIZE_W: tl.constexpr,
    TAPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < IN_H * RESIZE_W * 3
    channel = index % 3
    x = (index // 3) % RESIZE_W
    y = index // (RESIZE_W * 3)
    first = tl.load(BOUNDS + x * 2, valid, other=0)
    count = tl.load(BOUNDS + x * 2 + 1, valid, other=0)
    accum = tl.full((BLOCK,), 1 << 21, tl.int32)
    for tap in range(TAPS):
        active = valid & (tap < count)
        weight = tl.load(COEFFICIENTS + x * TAPS + tap, active, other=0)
        value = tl.load(
            RGB + (y * IN_W + first + tap) * 3 + channel, active, other=0
        ).to(tl.int32)
        accum += value * weight
    rounded = tl.minimum(tl.maximum(accum >> 22, 0), 255).to(tl.uint8)
    tl.store(TEMP + index, rounded, valid)


@triton.jit
def _vertical_pad_patchify(
    SOURCE,
    OUTPUT,
    BOUNDS,
    COEFFICIENTS,
    LUT,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    RESIZE_H: tl.constexpr,
    RESIZE_W: tl.constexpr,
    TOP: tl.constexpr,
    LEFT: tl.constexpr,
    PATCH: tl.constexpr,
    TAPS: tl.constexpr,
    RESAMPLE_Y: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < OUT_H * OUT_W * 3
    patch = index // (3 * PATCH * PATCH)
    channel = (index // (PATCH * PATCH)) % 3
    x = (patch % (OUT_W // PATCH)) * PATCH + index % PATCH - LEFT
    y = (patch // (OUT_W // PATCH)) * PATCH + (index // PATCH) % PATCH - TOP
    inside = valid & (x >= 0) & (x < RESIZE_W) & (y >= 0) & (y < RESIZE_H)
    if RESAMPLE_Y:
        first = tl.load(BOUNDS + y * 2, inside, other=0)
        count = tl.load(BOUNDS + y * 2 + 1, inside, other=0)
        accum = tl.full((BLOCK,), 1 << 21, tl.int32)
        for tap in range(TAPS):
            active = inside & (tap < count)
            weight = tl.load(COEFFICIENTS + y * TAPS + tap, active, other=0)
            value = tl.load(
                SOURCE + ((first + tap) * RESIZE_W + x) * 3 + channel,
                active,
                other=0,
            ).to(tl.int32)
            accum += value * weight
        value = tl.minimum(tl.maximum(accum >> 22, 0), 255)
    else:
        value = tl.load(SOURCE + (y * RESIZE_W + x) * 3 + channel, inside, other=0).to(
            tl.int32
        )
    pixel = tl.where(inside, value, 127)
    normalized = tl.load(LUT + pixel)
    tl.store(OUTPUT + index, normalized, valid)


def preprocess_rgb_u8(
    rgb: torch.Tensor, target_hw: tuple[int, int], *, patch_size: int = 14
) -> torch.Tensor:
    """Resize/pad and patchify RGB8 on CUDA, or raise ValueError if unsupported.

    Call is_supported_shape before H2D to reject unsupported geometry/version.
    The caller decides whether to reject or explicitly choose the CPU path;
    this function never silently falls back to CPU execution.
    A successful call is asynchronous on the current stream; the returned tensor
    has fresh storage and never aliases the input, tables, or previous outputs.
    """
    if not is_supported(rgb, target_hw, patch_size=patch_size):
        raise ValueError("unsupported input/runtime for V4.1 exact GPU preprocess")
    in_h, in_w = rgb.shape[:2]
    out_h, out_w = target_hw
    resize_h, resize_w, top, left = _geometry(in_h, in_w, out_h, out_w)
    block = 128
    with torch.cuda.device(rgb.device):
        source = rgb
        if in_w != resize_w:
            bounds, coefficients = _tables_on_device(
                rgb.device,
                ("axis", in_w, resize_w),
                lambda: _coefficients(in_w, resize_w),
            )
            source = torch.empty(
                (in_h, resize_w, 3), dtype=torch.uint8, device=rgb.device
            )
            _horizontal_u8[(triton.cdiv(source.numel(), block),)](
                rgb,
                source,
                bounds,
                coefficients,
                in_h,
                in_w,
                resize_w,
                coefficients.shape[1],
                block,
                num_warps=4,
            )
        if in_h != resize_h:
            bounds, coefficients = _tables_on_device(
                rgb.device,
                ("axis", in_h, resize_h),
                lambda: _coefficients(in_h, resize_h),
            )
            taps = coefficients.shape[1]
        else:
            # The compile-time identity branch never dereferences these pointers.
            bounds, coefficients, taps = rgb, rgb, 1
        (lut,) = _tables_on_device(
            rgb.device, ("normalization",), lambda: (_normalization_lut(),)
        )
        output = torch.empty(
            (out_h // patch_size * (out_w // patch_size), 3, patch_size, patch_size),
            dtype=torch.bfloat16,
            device=rgb.device,
        )
        _vertical_pad_patchify[(triton.cdiv(output.numel(), block),)](
            source,
            output,
            bounds,
            coefficients,
            lut,
            out_h,
            out_w,
            resize_h,
            resize_w,
            top,
            left,
            patch_size,
            taps,
            in_h != resize_h,
            block,
            num_warps=4,
        )
    return output
