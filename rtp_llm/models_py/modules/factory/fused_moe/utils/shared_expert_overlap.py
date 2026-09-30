"""Run an independent shared expert beside routed MoE on a CUDA stream."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.config import (
    shared_expert_mode,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    cuda_graph_warmup_forward_enabled,
)


_STREAMS: dict[int, torch.cuda.Stream] = {}


def can_overlap_shared_expert(x: torch.Tensor) -> bool:
    """Use the existing generic MoE mode and stay outside CUDA Graph capture."""
    mode = shared_expert_mode()
    if mode not in ("sequential", "auto", "overlap"):
        raise ValueError(f"unsupported MOE_SHARED_EXPERT_MODE={mode!r}")
    if mode == "sequential" or not x.is_cuda or not torch.cuda.is_available():
        return False
    tokens = int(x.shape[0])
    lower = int(os.environ.get("MOE_SHARED_EXPERT_STREAM_MIN_TOKENS", "1"))
    upper = int(os.environ.get("MOE_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD", "4096"))
    if lower <= 0 or upper < lower:
        raise ValueError("invalid shared expert stream token bounds")
    if not lower <= tokens <= upper:
        return False
    if torch.cuda.is_current_stream_capturing() or cuda_graph_warmup_forward_enabled():
        return False
    if os.environ.get("MOEDBG", "0") != "0":
        return False
    return True


@dataclass
class PendingSharedExpert:
    output: torch.Tensor
    stream: torch.cuda.Stream
    input: torch.Tensor

    def finish(self) -> torch.Tensor:
        current = torch.cuda.current_stream(self.output.device)
        current.wait_stream(self.stream)
        self.output.record_stream(current)
        return self.output


def start_shared_expert(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> PendingSharedExpert:
    """Launch shared work after its input producer, then return immediately."""
    if not can_overlap_shared_expert(x):
        raise ValueError("shared expert overlap was not enabled for this input")
    device = x.device.index
    if device is None:
        device = torch.cuda.current_device()
    stream = _STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _STREAMS[device] = stream
    producer = torch.cuda.current_stream(x.device)
    stream.wait_stream(producer)
    # Keep producer-owned storage alive until the side-stream consumer finishes.
    x.record_stream(stream)
    try:
        with torch.cuda.stream(stream):
            output = fn(x)
    except Exception:
        producer.wait_stream(stream)
        raise
    return PendingSharedExpert(output, stream, x)
