"""Run the shared expert on a side stream when explicitly enabled.

Large batches, CUDA graph warmup, and unsupported devices use the sequential
path.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import torch


def _overlap_enabled() -> bool:
    if os.environ.get("MOE_SHARED_EXPERT_OVERLAP", "0") != "1":
        return False
    # Debug mode: disable overlap so profiler timeline is easier to read.
    if os.environ.get("MOEDBG", "0") != "0":
        return False
    return True


def _is_cuda_graph_warmup() -> bool:
    """Check if a CUDA graph warmup forward pass is in progress."""
    try:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
            cuda_graph_warmup_forward_enabled,
        )

        return cuda_graph_warmup_forward_enabled()
    except ImportError:
        return False


# Shared-expert work is serial within a layer, so one stream per GPU suffices.
_shared_expert_stream_cache: dict[int, Any] = {}


def _get_or_create_shared_expert_stream(device: torch.device) -> Any:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _shared_expert_stream_cache.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _shared_expert_stream_cache[device_index] = stream
    return stream


class SharedExpertOverlapExecutor:
    """Overlap shared-expert work with the routed-expert pipeline."""

    def __init__(self) -> None:
        self._shared_expert_output: Optional[torch.Tensor] = None
        # The auxiliary CUDA stream running shared-expert compute, or
        # None when the last start() fell back to sequential execution.
        self._shared_expert_stream: Optional[torch.cuda.Stream] = None

    # ------------------------------------------------------------------
    # Preparation (call before CUDA graph capture)
    # ------------------------------------------------------------------

    def prepare(self, device: torch.device) -> None:
        """Create the auxiliary stream before CUDA graph capture."""
        if not _overlap_enabled():
            return
        if torch.cuda.is_available() and device.type == "cuda":
            _get_or_create_shared_expert_stream(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, fn: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> None:
        """Launch ``fn(*args, **kwargs)`` on the auxiliary stream.

        If overlap is not possible (env var off, non-CUDA tensor, token
        count above threshold, CUDA graph warmup), *fn* is called
        synchronously on the current stream and the result is stored
        for :meth:`finish`.

        During CUDA graph capture the function falls back to synchronous
        execution on the current stream. This avoids recording one auxiliary
        stream lane per captured shared-expert call.
        """
        if not self._can_overlap(args):
            self._shared_expert_stream = None
            self._shared_expert_output = fn(*args, **kwargs)
            return

        hidden_states = args[0]
        device = hidden_states.device
        stream = _get_or_create_shared_expert_stream(device)

        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                arg.record_stream(stream)
        for v in kwargs.values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                v.record_stream(stream)

        # Synchronise: aux stream waits for main-stream producers.
        stream.wait_stream(torch.cuda.current_stream(device))

        with torch.cuda.stream(stream):
            self._shared_expert_output = fn(*args, **kwargs)

        self._shared_expert_stream = stream

    def finish(self) -> torch.Tensor:
        """Block until the auxiliary stream completes and return its result."""
        if (
            self._shared_expert_stream is not None
            and self._shared_expert_output is not None
        ):
            torch.cuda.current_stream(self._shared_expert_output.device).wait_stream(
                self._shared_expert_stream
            )
        shared_expert_output = self._shared_expert_output
        self._shared_expert_output = None
        self._shared_expert_stream = None
        assert shared_expert_output is not None, "finish() called before start()"
        return shared_expert_output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _can_overlap(args: tuple) -> bool:
        if not _overlap_enabled():
            return False
        if not torch.cuda.is_available():
            return False
        if not args or not isinstance(args[0], torch.Tensor):
            return False
        if not args[0].is_cuda:
            return False
        # CUDA graph warmup runs a single forward pass to prime kernel caches;
        # overlap would add unnecessary stream synchronisation overhead there.
        if _is_cuda_graph_warmup():
            return False
        # Do not capture the auxiliary shared-expert stream into decode CUDA
        # graphs. This keeps graph replay on the main stream and avoids dozens
        # of per-layer stream lanes when overlap is enabled globally.
        if torch.cuda.is_current_stream_capturing():
            return False
        token_count = int(args[0].shape[0])
        threshold = int(
            os.environ.get("MOE_SHARED_EXPERT_OVERLAP_TOKEN_THRESHOLD", "4096")
        )
        return token_count <= threshold
