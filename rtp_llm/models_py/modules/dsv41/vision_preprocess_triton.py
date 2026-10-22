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

"""Pillow-compatible RGB bicubic preprocessing for DeepSeek V4.1.

The caller owns RGB conversion, the original-image H2D, the model's grid policy,
and device/support fallback. This module takes a contiguous CUDA
uint8 HWC RGB image, applies ImageOps.pad(BICUBIC, color=127), and returns owned
contiguous BF16 patches. It does not decode images or copy pixels back to CPU.

Compatibility reference: Pillow 11.2.1, ImageOps.contain/pad and the RGB8 path in
https://github.com/python-pillow/Pillow/blob/11.2.1/src/libImaging/Resample.c
This defines the device algorithm, not a required installed Pillow version.
Pillow only decodes/converts RGB before this module. Newer CPU resize routines
may use a different pass order for very tall images; the device path retains
the reference's horizontal-then-vertical rounding semantics.
Both separable passes round/clip to uint8. Integer coefficients have 22 fraction
bits; CPU double arithmetic preserves the reference coefficient construction.
The small CPU normalization table uses the original three FP32 Torch operations
before BF16 rounding. No floating-point resize or fused normalization is used.

CPU tables and the device cache of immutable tables are bounded.
Cold uploads stay inside this call; cache hits preserve stream dependencies.
Output/workspace storage is never shared. Work is queued on the
input device's current stream, without a device-wide synchronization. As usual,
callers must make cross-stream inputs ready and maintain/record their lifetime.
CUDA graph capture is explicitly unsupported in this first implementation.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.vision_preprocess_tables import (
    _PRECISION_BITS,
    ResizeTables,
    _coefficients,
    _geometry,
    _normalization_lut,
    is_supported_shape,
)


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
    if torch.device(device).type == "cuda":
        return _DEVICE_TABLES.get(device, key, factory)
    return tuple(table.to(device) for table in factory())


# Packed RGB uploads may start at any byte offset; neither geometry nor source
# alignment should create a new kernel on the request path.
@triton.jit(
    do_not_specialize=["IN_H", "IN_W", "RESIZE_W", "TAPS"],
    do_not_specialize_on_alignment=["RGB"],
)
def _horizontal_u8(
    RGB,
    TEMP,
    BOUNDS,
    COEFFICIENTS,
    IN_H,
    IN_W,
    RESIZE_W,
    TAPS,
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


@triton.jit(
    do_not_specialize=["OUT_H", "OUT_W", "RESIZE_H", "RESIZE_W", "TOP", "LEFT", "TAPS"],
    do_not_specialize_on_alignment=["SOURCE", "BOUNDS", "COEFFICIENTS"],
)
def _vertical_pad_patchify(
    SOURCE,
    OUTPUT,
    BOUNDS,
    COEFFICIENTS,
    LUT,
    OUT_H,
    OUT_W,
    RESIZE_H,
    RESIZE_W,
    TOP,
    LEFT,
    PATCH: tl.constexpr,
    TAPS,
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
    rgb: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    patch_size: int = 14,
    tables: ResizeTables | None = None,
) -> torch.Tensor:
    """Resize/pad and patchify RGB8 on CUDA, or raise ValueError if unsupported.

    Call is_supported_shape before H2D to reject unsupported geometry.
    The caller decides whether to reject or explicitly choose the CPU path;
    this function never silently falls back to CPU execution.
    A successful call is asynchronous on the current stream; the returned tensor
    has fresh storage and never aliases the input, tables, or previous outputs.
    """
    if not is_supported(rgb, target_hw, patch_size=patch_size):
        raise ValueError("unsupported input/runtime for V4.1 exact GPU preprocess")
    if tables is not None and (
        tables.source_hw != tuple(rgb.shape[:2])
        or tables.target_hw != tuple(target_hw)
        or tables.patch_size != patch_size
    ):
        raise ValueError("resize tables do not match the input geometry")
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
                lambda: (
                    tables.horizontal
                    if tables is not None
                    else _coefficients(in_w, resize_w)
                ),
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
                lambda: (
                    tables.vertical
                    if tables is not None
                    else _coefficients(in_h, resize_h)
                ),
            )
            taps = coefficients.shape[1]
        else:
            # The compile-time identity branch never dereferences these pointers.
            bounds, coefficients, taps = rgb, rgb, 1
        (lut,) = _tables_on_device(
            rgb.device,
            ("normalization",),
            lambda: (
                tables.normalization if tables is not None else _normalization_lut(),
            ),
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


def warmup_preprocess(device, *, patch_size: int = 14) -> None:
    """Compile both resize branches before serving, independent of image sizes.

    Dimensions, padding and filter length are runtime values: unseen geometries
    reuse these kernels, including integer values of 1 and different alignment.
    Keep the integer coefficient construction and uint8 rounding unchanged.
    """
    from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("preprocess warmup requires a CUDA device")
    if type(patch_size) is not int or not 1 <= patch_size <= 64:
        raise ValueError("unsupported preprocess patch size")
    with cuda_graph_gate.operation(), torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("preprocess warmup does not support graph capture")
        # Both axes resized, then the identity vertical branch (different table
        # pointer types). No model weights or request tensors are needed.
        for source_hw, target_hw in (
            (
                (3 * patch_size + 1, 4 * patch_size + 3),
                (2 * patch_size, 3 * patch_size),
            ),
            ((patch_size, 2 * patch_size), (patch_size, 2 * patch_size)),
        ):
            rgb = torch.zeros((*source_hw, 3), dtype=torch.uint8, device=device)
            preprocess_rgb_u8(rgb, target_hw, patch_size=patch_size)
        torch.cuda.current_stream(device).synchronize()
