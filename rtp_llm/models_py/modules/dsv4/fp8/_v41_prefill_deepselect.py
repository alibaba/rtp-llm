"""Native FP32 DeepSelect for V4.1 prefill token K512.

Scores retain FP32 precision. The selecting kernel canonicalizes NaN to +inf
before ranking and optionally filters selected nonfinite values to -1, also
for short rows. Thus nonfinite scores consume selected slots, as in the
existing prefill selector. Output order and membership within ties are
unspecified; candidate-block selection has a separate contract.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


@lru_cache(maxsize=None)
def _device_supported(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


def is_available(device: torch.device) -> bool:
    if (
        os.environ.get("DSV41_PREFILL_DEEPSELECT", "0") != "1"
        or torch.version.hip is not None
        or device.type != "cuda"
    ):
        return False
    available = getattr(rtp_llm_ops, "deepselect_fp32_available", None)
    return available is not None and available() and _device_supported(device)


def is_supported(logits, ends, out=None, *, filter_finite=True) -> bool:
    """Metadata-only admission; no score or bound is read back to the host."""
    if not (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and 0 < logits.shape[0] <= 2**31 - 1
        and 512 <= logits.shape[1] < 2**23
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and logits.stride(0) % 256 == 0
        and logits.data_ptr() % 16 == 0
        and ends.device == logits.device
        and ends.dtype == torch.int32
        and ends.ndim == 1
        and ends.numel() == logits.shape[0]
        and ends.is_contiguous()
        and (filter_finite or out is not None)
    ):
        return False
    if logits.shape[1] % 32:
        # TMA describes complete 128-byte rows, including the final allocation
        # row. A strided view can end before that padding even when its stride
        # is aligned, so check the actual storage extent without a GPU read.
        padded_width = (logits.shape[1] + 31) // 32 * 32
        required_elements = (
            logits.storage_offset()
            + (logits.shape[0] - 1) * logits.stride(0)
            + padded_width
        )
        if required_elements * 4 > logits.untyped_storage().nbytes():
            return False
    if out is not None:
        if not (
            out.device == logits.device
            and out.dtype == torch.int32
            and out.shape == (logits.shape[0], 512)
            and out.stride(1) == 1
            and out.stride(0) >= 512
            and out.stride(0) % 8 == 0
            and out.data_ptr() % 32 == 0
        ):
            return False
        storage = out.untyped_storage().data_ptr()
        if any(t.untyped_storage().data_ptr() == storage for t in (logits, ends)):
            return False
    return is_available(logits.device)


def try_select_tokens(logits, ends, *, out=None, filter_finite=True):
    """Return K512 int32 indices, or None before an unsupported launch.

    Bounds are clamped in the native kernel. A caller using
    ``filter_finite=False`` must filter in its publication epilogue. Once a
    native launch is selected, execution errors propagate without fallback.
    """
    if not is_supported(logits, ends, out, filter_finite=filter_finite):
        return None
    if out is None:
        out = torch.empty(
            (logits.shape[0], 512), dtype=torch.int32, device=logits.device
        )
    rtp_llm_ops.deepselect_fp32(logits, ends, out, filter_finite)
    return out
