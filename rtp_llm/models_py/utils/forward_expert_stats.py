"""Per-forward model-side routing counters; no buffers are owned by the model.

Counts are actual local dispatch assignments, including TP-replicated work.
Logical/physical expert IDs coincide only for the uniform, non-redundant
MegaMoE layout supported here. Shared experts are not routed and not counted.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import logging
import os

import torch

_CURRENT_STATS = ContextVar("forward_expert_stats", default=None)


@contextmanager
def forward_expert_stats(stats):
    token = _CURRENT_STATS.set(stats)
    try:
        yield
        if stats is not None and os.environ.get("EPLB_STATS_VALIDATE") == "1":
            # Diagnostic only: deliberately synchronize to check every layer.
            logical = stats.log_stats_buf.sum(dim=1).cpu()
            loads = stats.gpu_loads_buf.sum(dim=1).cpu()
            if not torch.equal(logical, loads) or not bool((logical > 0).all()):
                raise RuntimeError("Expert stats: missing layer or inconsistent destination counts")
            logging.info(
                "EPLB_STATS_PRODUCED layers=%d ep=%d assignments=%d layer0=%s",
                logical.numel(), stats.gpu_loads_buf.shape[1], int(logical.sum()),
                stats.gpu_loads_buf[0].cpu().tolist(),
            )
    finally:
        _CURRENT_STATS.reset(token)


def record_expert_routes(topk_ids, *, layer_id, num_experts, ep_size):
    """Add real-token packed top-k IDs, after pack and before dispatch.

    Runs on the pack/dispatch stream. Repeated chunks add to the same row.
    Negative/out-of-range padding IDs contribute zero. No host readback is
    needed unless the explicit smoke diagnostic is enabled.
    """
    stats = _CURRENT_STATS.get()
    if stats is None:
        return
    if num_experts <= 0 or ep_size <= 0 or num_experts % ep_size:
        raise ValueError("Stats require a uniform non-redundant expert layout")
    if (stats.log_stats_buf.shape[1] != num_experts
            or stats.gpu_loads_buf.shape[1] != ep_size):
        raise ValueError("Stats layout differs from MegaMoE expert layout")
    if topk_ids.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Per-forward expert stats require eager execution")
    ids = topk_ids.reshape(-1).to(torch.int64)
    valid = (ids >= 0) & (ids < num_experts)
    safe_ids = ids.clamp(0, num_experts - 1)
    counts = valid.to(stats.log_stats_buf.dtype)
    stats.log_stats_buf[layer_id].scatter_add_(0, safe_ids, counts)
    owners = torch.div(safe_ids, num_experts // ep_size, rounding_mode="floor")
    stats.gpu_loads_buf[layer_id].scatter_add_(
        0, owners, counts.to(stats.gpu_loads_buf.dtype)
    )
