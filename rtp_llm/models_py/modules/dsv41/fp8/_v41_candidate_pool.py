"""Bounded candidate pooling with work shared across long query rows.

The persistent TileLang path is restricted to aligned BF16 decode storage.
Other layouts and unavailable dependencies keep the Triton row-loop path.
"""

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def _device_supported(device):
    return torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


@lru_cache(maxsize=None)
def _num_sms(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def is_supported(logits, lengths, scores, block_lengths, block_size):
    if not (
        os.environ.get("DSV41_DECODE_CANDIDATE_POOL_PERSISTENT", "1") == "1"
        and torch.version.hip is None
        and logits.is_cuda
        and logits.dtype == torch.bfloat16
        and logits.ndim == 2
        and 128 <= logits.shape[0] <= 4096
        and 65536 <= logits.shape[1] <= 1048576
        and logits.shape[1] % 16384 == 0
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and logits.stride(0) % 16 == 0
        and logits.data_ptr() % 32 == 0
        and logits.shape[0] * logits.stride(0) <= 2**31 - 1
        and block_size == 8
        and lengths.device == logits.device
        and lengths.dtype == torch.int32
        and lengths.ndim == 1
        and lengths.numel() == logits.shape[0]
        and lengths.is_contiguous()
        and lengths.data_ptr() % 8 == 0
        and scores.device == logits.device
        and scores.dtype == torch.bfloat16
        and scores.ndim == 2
        and scores.shape[0] == logits.shape[0]
        and scores.shape[1] == logits.shape[1] // block_size
        and scores.stride(1) == 1
        and scores.stride(0) >= scores.shape[1]
        and scores.stride(0) % 16 == 0
        and scores.data_ptr() % 32 == 0
        and block_lengths.device == logits.device
        and block_lengths.dtype == torch.int32
        and block_lengths.shape == lengths.shape
        and block_lengths.is_contiguous()
        and block_lengths.data_ptr() % 8 == 0
    ):
        return False
    return _device_supported(logits.device)


@lru_cache(maxsize=1)
def _load_kernel():
    try:
        import tilelang
    except (ImportError, OSError):
        return None
    # An older TileLang may accept annotations but silently ignore this one.
    # Its registered CUDA intrinsic proves that NaN-preserving BF16 max is
    # available, before any kernel is compiled or any output is written.
    try:
        tilelang.tvm.ir.Op.get("tl.max_nan")
        if not all(
            hasattr(tilelang.PassConfigKey, name)
            for name in ("TL_DISABLE_DATA_RACE_CHECK", "TL_ENABLE_LOWER_LDGSTG")
        ):
            return None
    except (AttributeError, ValueError, RuntimeError):
        return None
    from ._v41_candidate_pool_tl import candidate_pool

    return candidate_pool


def try_pool(logits, lengths, scores, block_lengths, block_size):
    """Launch on the current stream, or return False before touching outputs.

    Admission never reads a device length. Task preparation and its storage
    allocation are captured together with pooling, so live lengths are consumed
    again on every graph replay. Compilation/launch errors remain visible.
    """
    if not is_supported(logits, lengths, scores, block_lengths, block_size):
        return False
    kernel = _load_kernel()
    if kernel is None:
        return False
    kernel(
        logits.shape[1],
        block_size,
        _num_sms(logits.device),
    )(logits, lengths, scores, block_lengths)
    return True
