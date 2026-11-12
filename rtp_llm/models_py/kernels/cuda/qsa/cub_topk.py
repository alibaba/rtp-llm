"""Opt-in SM120 QSA TopK with native CUDA ordering and token expansion.

The model-local kernel uses NVIDIA CUB's installed stable block sort APIs. It does
not vendor vLLM/CUB source. Stable uint32 keys retain the pinned Torch TopK
selection and raw radix order, including signed zero and both NaN signs.
Compilation occurs during eager warmup; an unprimed CUDA graph falls back.
"""

import hashlib
import logging
import os
import threading
from pathlib import Path

import torch

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
_MODULE = None
_BUILD_FAILED = False
_BUILD_LOCK = threading.Lock()
_INITIALIZED_DEVICES = set()
_LOGGER = logging.getLogger(__name__)

_CUDA_SOURCE = Path(__file__).with_name("topk.cuh").read_text()


def is_supported(scores, lengths):
    # Opt in only the two MTP3 target buckets captured with an 8K KV capacity.
    # Other row counts/column widths retain their existing dispatch domain.
    mode = os.getenv("RTP_QWEN4_TARGET_TOPK_IMPL", "native")
    if mode not in ("native", "cub"):
        raise ValueError("RTP_QWEN4_TARGET_TOPK_IMPL must be native or cub")
    mtp3_target = (
        mode == "cub"
        and scores.ndim == 2
        and scores.shape[0] in (64, 256)
        and scores.shape[1] == 2048
    )
    large_prefill = (
        scores.ndim == 2
        and 2144 < scores.shape[1] <= 16384
        and 1 <= scores.shape[0] <= 8192
    )
    return (
        scores.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and (large_prefill or mtp3_target or 1 <= scores.shape[0] <= 32)
        # The 8K MTP draft graph reserves three additional 128-token pages:
        # (64 + 3) * 32 compressed columns. Preserve the full-width Torch order.
        and (large_prefill or scores.shape[1] in (2047, 2048, 2144))
        and scores.is_contiguous()
        and lengths.dtype == torch.int32
        and lengths.shape == (scores.shape[0],)
        and lengths.device == scores.device
        and lengths.is_contiguous()
        and not (scores.requires_grad or lengths.requires_grad)
        and torch.cuda.get_device_capability(scores.device) == (12, 0)
    )


def _load_module(device):
    global _MODULE, _BUILD_FAILED
    if _MODULE is not None:
        with torch.cuda.device(device):
            index = torch.cuda.current_device()
            if index not in _INITIALIZED_DEVICES:
                if torch.cuda.is_current_stream_capturing():
                    return None
                _MODULE.qsa_cub_init()
                _INITIALIZED_DEVICES.add(index)
        return _MODULE
    if _BUILD_FAILED:
        return None
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return None
        with _BUILD_LOCK:
            if _MODULE is not None:
                return _MODULE
            if _BUILD_FAILED:
                return None
            try:
                from torch.utils.cpp_extension import load_inline

                flags = [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_120,code=sm_120",
                ]
                identity = hashlib.sha256(
                    (_CUDA_SOURCE + _TORCH_GIT + repr(flags)).encode()
                ).hexdigest()[:16]
                _MODULE = load_inline(
                    name="qwen4_qsa_topk_" + identity,
                    cpp_sources="void qsa_cub_init(); void qsa_cub_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, int64_t threads, bool finite_only, bool merge); void qsa_cub_topk_draft(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor row_lengths, torch::Tensor cu);",
                    cuda_sources=_CUDA_SOURCE,
                    functions=["qsa_cub_init", "qsa_cub_topk", "qsa_cub_topk_draft"],
                    extra_cflags=["-O2"],
                    extra_cuda_cflags=flags,
                    with_cuda=True,
                    verbose=False,
                )
                _MODULE.qsa_cub_init()
                _INITIALIZED_DEVICES.add(torch.cuda.current_device())
            except Exception:
                _MODULE = None
                _BUILD_FAILED = True
                _LOGGER.warning(
                    "Qwen4 QSA CUB compilation unavailable; using native selection",
                    exc_info=True,
                )
            return _MODULE


def try_cub_qsa_tokens(scores, lengths, *, finite_only=True):
    if not is_supported(scores, lengths):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_cub_topk(
        scores,
        lengths,
        output,
        256,
        finite_only,
        True,
    )
    return output


def try_cub_draft_qsa_tokens(scores, visible, row_lengths, cu):
    """Select directly into packed draft order; metadata is owner-validated.

    Fixed four-lane input rows and packed output rows have a bijection even
    for empty requests. Inactive rows fill the output padding with -1.
    """
    if not (
        is_supported(scores, visible)
        and scores.shape[1] == 2144
        and row_lengths.ndim == cu.ndim == 1
        and 1 <= row_lengths.numel() <= 8
        and scores.shape[0] == 4 * row_lengths.numel()
        and cu.numel() == row_lengths.numel() + 1
        and all(
            t.dtype == torch.int32
            and t.device == scores.device
            and t.is_contiguous()
            and not t.requires_grad
            for t in (row_lengths, cu)
        )
    ):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    module.qsa_cub_topk_draft(scores, visible, output, row_lengths, cu)
    return output
