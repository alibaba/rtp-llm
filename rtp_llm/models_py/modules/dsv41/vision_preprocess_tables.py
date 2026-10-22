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

"""CPU-only Pillow resize metadata prepared by the media decoder.

Keep scalar double arithmetic and coefficient quantization identical to Pillow.
Tables are immutable by convention and retained by each decoded item, so LRU
churn cannot move this work back onto the GPU scheduler thread.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

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
        type(patch_size) is not int
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


@dataclass(frozen=True)
class ResizeTables:
    source_hw: tuple[int, int]
    target_hw: tuple[int, int]
    patch_size: int
    horizontal: tuple[torch.Tensor, torch.Tensor] | None
    vertical: tuple[torch.Tensor, torch.Tensor] | None
    normalization: torch.Tensor


def prepare_resize_tables(source_hw, target_hw, *, patch_size=14):
    source_hw, target_hw = tuple(source_hw), tuple(target_hw)
    if not is_supported_shape(source_hw, target_hw, patch_size=patch_size):
        raise ValueError("unsupported shape for V4.1 GPU preprocessing")
    in_h, in_w = source_hw
    resize_h, resize_w, _, _ = _geometry(in_h, in_w, *target_hw)
    return ResizeTables(
        source_hw,
        target_hw,
        patch_size,
        _coefficients(in_w, resize_w) if in_w != resize_w else None,
        _coefficients(in_h, resize_h) if in_h != resize_h else None,
        _normalization_lut(),
    )
