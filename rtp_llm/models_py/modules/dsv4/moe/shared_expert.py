"""Shared expert execution policies for DSV4 MoE.

Open-source MoE stacks such as vLLM and SGLang commonly overlap shared experts
with routed MoE work on an auxiliary CUDA stream.  They do not rely on BF16
direct accumulation by default.  RTP keeps the existing FP32 accumulate contract
and only fuses the final add+cast when possible.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range


def _mode() -> str:
    return os.environ.get("DSV4_SHARED_EXPERT_MODE", "auto").strip().lower()


class SharedExpertExecutor(ABC):
    name: str

    @abstractmethod
    def start(self, shared_experts: nn.Module, x: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> torch.Tensor:
        raise NotImplementedError


class SequentialSharedExpertExecutor(SharedExpertExecutor):
    name = "sequential"

    def __init__(self) -> None:
        self._out: torch.Tensor | None = None

    def start(self, shared_experts: nn.Module, x: torch.Tensor) -> None:
        with record_function_range("dsv4.moe.shared_expert"):
            self._out = shared_experts(x).float()

    def finish(self) -> torch.Tensor:
        assert self._out is not None
        out = self._out
        self._out = None
        return out


class OverlapSharedExpertExecutor(SharedExpertExecutor):
    """Run shared expert on an aux stream while routed MoE runs on current stream."""

    name = "overlap"

    def __init__(self) -> None:
        self._stream: torch.cuda.Stream | None = None
        self._active_stream: torch.cuda.Stream | None = None
        self._out: torch.Tensor | None = None

    def _can_overlap(self, x: torch.Tensor) -> bool:
        if not (x.is_cuda and torch.cuda.is_available()):
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if os.environ.get("MOEDBG", "0") != "0":
            return False
        threshold = int(
            os.environ.get("DSV4_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD", "4096")
        )
        return x.shape[0] <= threshold

    def start(self, shared_experts: nn.Module, x: torch.Tensor) -> None:
        if not self._can_overlap(x):
            self._active_stream = None
            with record_function_range("dsv4.moe.shared_expert"):
                self._out = shared_experts(x).float()
            return
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=x.device)
        stream = self._stream
        x.record_stream(stream)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream):
            with record_function_range("dsv4.moe.shared_expert"):
                self._out = shared_experts(x).float()
        self._active_stream = stream

    def finish(self) -> torch.Tensor:
        assert self._out is not None
        if self._active_stream is not None:
            torch.cuda.current_stream(self._out.device).wait_stream(self._active_stream)
        out = self._out
        self._out = None
        self._active_stream = None
        return out


def get_shared_expert_executor() -> SharedExpertExecutor:
    mode = _mode()
    if mode == "sequential":
        return SequentialSharedExpertExecutor()
    if mode in ("auto", "overlap"):
        return OverlapSharedExpertExecutor()
    raise ValueError(
        f"invalid DSV4_SHARED_EXPERT_MODE={mode!r}; expected auto|sequential|overlap"
    )


def combine_routed_and_shared(
    routed: torch.Tensor, shared: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    if os.environ.get("DSV4_SHARED_EXPERT_BF16_ADD", "0") == "1":
        return (routed.to(out_dtype) + shared.to(out_dtype)).to(out_dtype)
    try:
        from ._shared_expert_triton import fused_add_cast_bf16

        return fused_add_cast_bf16(routed, shared, out_dtype)
    except Exception:
        return (routed.float() + shared.float()).to(out_dtype)
