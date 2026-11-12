from types import SimpleNamespace

import torch

from rtp_llm.eplb.runtime import EplbLayer


def make_config(**updates):
    values = dict(
        eplb_mode=2,
        eplb_update_time=4,
        eplb_stats_window_size=4,
        eplb_rearrange_after_prefill=False,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def make_layer(
    physical=(0, 1, 2, 3, 0, 1),
    layer_id=0,
    ep_rank=0,
    ep_size=1,
    device="cpu",
    replica_offset=0,
    weight_width=6,
):
    logical = max(physical) + 1
    inverse = torch.full((logical, len(physical) - logical + 1), -1, dtype=torch.int32)
    counts = torch.zeros(logical, dtype=torch.int32)
    for slot, expert in enumerate(physical):
        inverse[expert, counts[expert]] = slot
        counts[expert] += 1
    local = len(physical) // ep_size
    ids = torch.tensor(physical[ep_rank * local : (ep_rank + 1) * local], device=device)
    weight = (ids[:, None] * 100 + torch.arange(weight_width, device=device)).to(
        torch.int32
    )
    scale = (
        (ids[:, None, None] * 10 + torch.arange(6, device=device).reshape(2, 3))
        .float()
        .transpose(1, 2)
    )
    return EplbLayer(
        layer_id,
        inverse.to(device),
        counts.to(device),
        (weight, scale),
        ep_rank,
        replica_offset,
    )
