"""Optional exact-order Qwen4 Prefill TopK with bounded coarse selection.

The FP16 histogram follows SGLang release297e54fd105e974cdf401e5b4d20253a22c16f90,
python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh (Apache-2.0).
Candidate buckets are never truncated. Sort complete candidates by the pinned
Torch score/index key; special values, short rows and overflow use the existing
exact CUB path. There is no SGLang runtime dependency.
SPDX-License-Identifier: Apache-2.0
"""

import hashlib
import logging
import threading
from pathlib import Path

import torch

from rtp_llm.models_py.kernels.cuda.qsa.cub_topk import _CUDA_SOURCE, _TORCH_GIT

_MODULE = None
_BUILD_FAILED = False
_BUILD_LOCK = threading.Lock()
_INITIALIZED_DEVICES = set()

_HISTOGRAM_SOURCE = Path(__file__).with_name("histogram.cu").read_text()


def is_supported(scores, lengths):
    return (
        scores.is_cuda
        and torch.version.hip is None
        and torch.version.git_version == _TORCH_GIT
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 33 <= scores.shape[0] <= 8192
        and 4096 < scores.shape[1] <= 16384
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
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
        capturing = torch.cuda.is_current_stream_capturing()
        if _MODULE is not None and index in _INITIALIZED_DEVICES:
            return _MODULE
        if capturing or _BUILD_FAILED:
            return None
        with _BUILD_LOCK:
            if _MODULE is not None:
                if index not in _INITIALIZED_DEVICES:
                    _MODULE.qsa_radix_init()
                    _INITIALIZED_DEVICES.add(index)
                return _MODULE
            try:
                from torch.utils.cpp_extension import load_inline

                source = (
                    "#define RTP_QSA_HISTOGRAM\n" + _CUDA_SOURCE + _HISTOGRAM_SOURCE
                )
                flags = [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_120,code=sm_120",
                ]
                identity = hashlib.sha256(
                    (source + _TORCH_GIT + repr(flags)).encode()
                ).hexdigest()[:16]
                _MODULE = load_inline(
                    name="qwen4_qsa_histogram_" + identity,
                    cpp_sources="void qsa_radix_init(); void qsa_radix_topk(torch::Tensor scores, torch::Tensor lengths, torch::Tensor output, torch::Tensor fallback, bool finite_only);",
                    cuda_sources=source,
                    functions=["qsa_radix_init", "qsa_radix_topk"],
                    extra_cflags=["-O2"],
                    extra_cuda_cflags=flags,
                    with_cuda=True,
                    verbose=False,
                )
                _MODULE.qsa_radix_init()
                _INITIALIZED_DEVICES.add(index)
                logging.info("Qwen4 histogram TopK initialized source=%s", identity)
            except Exception:
                _BUILD_FAILED = True
                _MODULE = None
                logging.exception(
                    "Qwen4 histogram TopK unavailable; retaining native selection"
                )
            return _MODULE


def try_histogram_qsa_tokens(scores, lengths, *, finite_only=True):
    if not is_supported(scores, lengths):
        return None
    module = _load_module(scores.device)
    if module is None:
        return None
    output = torch.empty(
        (scores.shape[0], 2051), device=scores.device, dtype=torch.int32
    )
    fallback = torch.empty(scores.shape[0], device=scores.device, dtype=torch.bool)
    module.qsa_radix_topk(scores, lengths, output, fallback, finite_only)
    return output
