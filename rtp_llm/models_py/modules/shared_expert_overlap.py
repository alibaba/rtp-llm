"""Shared expert overlap executor for GenericMoeLayer.

Runs the shared expert (DenseMLP) on an auxiliary CUDA stream while the
routed expert pipeline (DeepEP dispatch → execute → combine) runs on the
main stream.  The two overlap in wall-clock time: shared-expert compute
hides behind the all-to-all communication of the routed path.

Controlled by ``MOE_SHARED_EXPERT_OVERLAP`` (default ``"0"`` = off).
When overlap is disabled or unavailable the executor falls back to
sequential execution with no overhead.

Design notes (aligned with ``dsv4/moe/shared_expert.py:OverlapSharedExpertExecutor``):
  - Token-count threshold (``MOE_SHARED_EXPERT_OVERLAP_TOKEN_THRESHOLD``,
    default 4096): large batches make the shared expert heavy enough that
    overlap adds stream-switching cost without hiding it behind dispatch.
  - CUDA graph capture: ``prepare()`` pre-creates the auxiliary stream
    *before* capture so ``start()`` can still dispatch to it during
    capture.  ``record_stream`` is skipped during capture because the
    graph-replay allocator manages tensor lifetimes on its own.
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
        from rtp_llm.models_py.modules.dsv4.moe.warmup_sync import (
            cuda_graph_warmup_forward_enabled,
        )

        return cuda_graph_warmup_forward_enabled()
    except ImportError:
        return False


# Per-device auxiliary stream cache for shared-expert overlap work.
# One stream per GPU is sufficient because shared-expert work within a
# single layer is serial (start → routed → finish).
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
    """Run a shared expert on an auxiliary CUDA stream concurrently with
    the routed-expert pipeline on the main stream.

    Usage::

        executor = SharedExpertOverlapExecutor()
        # Call prepare() once at construction time (or before CUDA graph capture)
        # so the auxiliary stream exists before capture begins.
        executor.prepare(hidden_states_device)
        ...
        executor.start(shared_expert_fn, hidden_states, ...)
        # ... routed expert work on main stream ...
        shared_output = executor.finish()
    """

    def __init__(self) -> None:
        self._shared_expert_output: Optional[torch.Tensor] = None
        # The auxiliary CUDA stream running shared-expert compute, or
        # None when the last start() fell back to sequential execution.
        self._shared_expert_stream: Optional[torch.cuda.Stream] = None

    # ------------------------------------------------------------------
    # Preparation (call before CUDA graph capture)
    # ------------------------------------------------------------------

    def prepare(self, device: torch.device) -> None:
        """Pre-create the auxiliary stream for *device*.

        Must be called before any CUDA graph capture that will include
        the shared-expert forward.  Without this, ``start()`` would
        allocate a new stream during capture, which is not allowed and
        would silently fall back to sequential execution.

        No-op when overlap is disabled via ``MOE_SHARED_EXPERT_OVERLAP``.
        """
        if not _overlap_enabled():
            return
        if torch.cuda.is_available() and device.type == "cuda":
            _get_or_create_shared_expert_stream(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, fn: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> None:
        """Launch *fn(\*args, \*\*kwargs)* on the auxiliary stream.

        If overlap is not possible (env var off, non-CUDA tensor, token
        count above threshold, CUDA graph warmup), *fn* is called
        synchronously on the current stream and the result is stored
        for :meth:`finish`.

        During CUDA graph capture the function still dispatches to the
        pre-created auxiliary stream (the graph records the multi-stream
        topology), but ``record_stream`` is skipped because the
        graph-replay allocator manages tensor lifetimes independently.
        """
        capturing = torch.cuda.is_current_stream_capturing()
        if not self._can_overlap(args):
            self._shared_expert_stream = None
            self._shared_expert_output = fn(*args, **kwargs)
            return

        hidden_states = args[0]
        device = hidden_states.device
        stream = _get_or_create_shared_expert_stream(device)

        # During CUDA graph capture the replay allocator tracks tensor
        # lifetimes on its own; calling record_stream inside capture can
        # corrupt allocator state.
        if not capturing:
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
        # Token-count threshold: large batches make the shared expert itself
        # expensive enough that it no longer hides behind dispatch/combine.
        threshold = int(
            os.environ.get("MOE_SHARED_EXPERT_OVERLAP_TOKEN_THRESHOLD", "4096")
        )
        if args[0].shape[0] > threshold:
            return False
        return True
