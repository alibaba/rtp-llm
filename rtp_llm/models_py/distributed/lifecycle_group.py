"""Dedicated CPU control group, constructed before model execution starts.

Only native lifecycle control uses this backend. Inference collectives and
NCCL suspend/rebuild must not own or destroy it.
"""

from datetime import timedelta

import torch.distributed as dist

_group = None


def init_lifecycle_group(store, rank: int, world_size: int, timeout_s: float):
    global _group
    if _group is not None:
        raise RuntimeError("CPU lifecycle group is already initialized")
    if world_size == 1:
        return
    if not dist.is_gloo_available():
        raise RuntimeError("multi-rank lifecycle control requires Gloo support")
    # Direct Backend construction avoids registering with the model WORLD group
    # or introducing collectives into model initialization/forward/restore.
    _group = dist.ProcessGroupGloo(
        dist.PrefixStore("rtp_llm_execution_control/", store),
        rank,
        world_size,
        timedelta(seconds=timeout_s),
    )


def get_lifecycle_group():
    return _group
