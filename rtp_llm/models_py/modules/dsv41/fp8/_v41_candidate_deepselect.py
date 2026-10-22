"""DeepSelect sorted K2048 publication for BF16 decode block scores.

Admission is metadata-only. Pool BF16 inputs directly into aligned BF16 score
storage: max pooling introduces no new rounding relative to the old FP32 max
of BF16 values. Genuine FP32 inputs retain the existing selector.
"""

import os
from functools import lru_cache

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


@lru_cache(maxsize=None)
def _device_supported(device):
    return torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


def is_supported(logits, lengths, block_size, count):
    if not (
        os.environ.get("DSV41_DECODE_CANDIDATE_DEEPSELECT", "1") == "1"
        and torch.version.hip is None
        and logits.is_cuda
        and logits.dtype == torch.bfloat16
        and logits.ndim == 2
        and 0 < logits.shape[0] <= 2**31 - 1
        and block_size == 8
        and count == 2048
        and 2048 <= (logits.shape[1] + block_size - 1) // block_size < 2**23 - 511
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and lengths.device == logits.device
        and lengths.dtype == torch.int32
        and lengths.ndim == 1
        and lengths.numel() == logits.shape[0]
        and lengths.is_contiguous()
    ):
        return False
    available = getattr(rtp_llm_ops, "deepselect_bf16_available", None)
    return (
        callable(getattr(rtp_llm_ops, "deepselect_bf16_candidates", None))
        and callable(available)
        and available()
        and _device_supported(logits.device)
    )


def select(scores, lengths):
    """Caller supplies padded BF16 storage and logical bounds from pooling.

    The native call uses the current (possibly auxiliary/capturing) CUDA stream.
    It overwrites every output slot on each replay, including zero/short rows;
    there is no GPU predicate read on the host and no launch-error fallback.
    """
    output = torch.empty((scores.shape[0], 2048), dtype=torch.int32, device=scores.device)
    rtp_llm_ops.deepselect_bf16_candidates(scores, lengths, output)
    return output
