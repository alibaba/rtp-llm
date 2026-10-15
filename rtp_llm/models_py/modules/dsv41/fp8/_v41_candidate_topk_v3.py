"""Shared raw K2048 selection for prefill and decode candidate blocks.

Callers pool into FP32 and publish the selected indices, retaining +inf newest
block pins while filtering NaN/-inf. Equal cutoff scores need not reproduce
PyTorch's block membership. Device lengths allow short rows to skip ranking.
"""

import os

import torch

from rtp_llm.models_py.modules.dsv41.fp8.indexer import (
    _TOPK_V3_OK,
    _get_topk_workspace,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def is_available(phase: str, device: torch.device) -> bool:
    return (
        _TOPK_V3_OK
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
        and torch.version.hip is None
        and os.environ.get(f"DSV41_{phase.upper()}_CANDIDATE_TOPK_V3", "1") == "1"
    )


def select(scores: torch.Tensor, block_lengths: torch.Tensor) -> torch.Tensor:
    """Raw contiguous int32[M,2048]; caller gates the backend and publishes."""
    indices = torch.empty(
        (scores.shape[0], 2048), dtype=torch.int32, device=scores.device
    )
    rtp_llm_ops.topk_v3(
        scores,
        block_lengths,
        indices,
        _get_topk_workspace(scores.device),
        2048,
        scores.shape[1],
    )
    return indices
