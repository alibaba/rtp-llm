"""Shared CUDA-graph warmup-forward state probe without heavy imports."""

from __future__ import annotations

import os

_WARMUP_ENV = "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"


def cuda_graph_warmup_forward_enabled() -> bool:
    try:
        from rtp_llm.ops.compute_ops import cuda_graph_warmup_forward_enabled as enabled
    except ImportError:
        return os.environ.get(_WARMUP_ENV, "0") == "1"
    return enabled()
