"""Experimental TP=2 BF16->FP8 two-shot IPC all-reduce.

Native implementation is derived from TensorRT-LLM commit
c2220eef33f407fd3d805a6ab8a2725f54bdd3c4 under Apache-2.0.
It is deliberately loaded only on explicit use; callers own rank bootstrap and
current-stream/event ordering.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import ModuleType
from typing import Optional

_MODULE: Optional[ModuleType] = None


def is_supported(device_index: int | None = None) -> bool:
    """Return whether this process can JIT the experimental CUDA extension."""
    if os.environ.get("RTP_LLM_TP_FP8_ALLREDUCE", "0") != "1":
        return False
    try:
        import torch

        if not torch.cuda.is_available() or torch.version.hip is not None:
            return False
        index = torch.cuda.current_device() if device_index is None else device_index
        major, minor = torch.cuda.get_device_capability(index)
        return (major, minor) >= (8, 9)  # Ada FP8 or newer; not an SM120-only gate.
    except Exception:
        return False


def load_native() -> ModuleType:
    """JIT-load for the current runtime CUDA architecture; never called at import."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    import torch
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("tp_fp8_all_reduce.cu")
    capability = torch.cuda.get_device_capability()
    # cpp_extension honors this standard variable. Set only for this process when
    # the caller has not already selected an architecture, and derive it from GPU.
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if previous is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
    try:
        tag = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        _MODULE = load(
            name=f"rtp_tp_fp8_allreduce_{tag}",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
            verbose=os.environ.get("RTP_LLM_TP_FP8_ALLREDUCE_VERBOSE", "0") == "1",
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    return _MODULE


class TpFp8AllReduce:
    """Per-rank native context. ``rank`` is required because device ID is local."""

    def __init__(
        self, max_numel: int, device_index: int, rank: int, blocks: int = 16
    ) -> None:
        self._native = load_native().TpFp8AllReduce(
            max_numel, device_index, rank, blocks
        )

    def get_ipc_handle(self) -> bytes:
        return self._native.get_ipc_handle()

    def open_peer(self, handle: bytes) -> None:
        self._native.open_peer(handle)

    def close_peer(self) -> None:
        self._native.close_peer()

    def all_reduce(self, input, out) -> None:
        self._native.all_reduce(input, out)

    def close(self) -> None:
        self._native.close()


__all__ = ["TpFp8AllReduce", "is_supported", "load_native"]
