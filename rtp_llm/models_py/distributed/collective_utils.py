"""Bounded host-side agreement before optional GPU communication setup."""

import json
from datetime import timedelta

import torch.distributed as dist

_exchange_epochs = {}


def exchange_group_state(group, state, tag):
    # All members execute the same initialization sequence. Include the group
    # and round so reinitialization cannot consume an earlier decision.
    name = getattr(group, "group_name", "default")
    key = (name, tag)
    epoch = _exchange_epochs.get(key, 0) + 1
    _exchange_epochs[key] = epoch
    prefix = f"rtp_llm/collectives/{name}/{epoch}/{tag}/"
    store = dist.distributed_c10d._get_default_store()
    keys = [prefix + str(rank) for rank in range(dist.get_world_size(group))]
    store.set(keys[dist.get_rank(group)], json.dumps(state).encode())
    store.wait(keys, timedelta(seconds=120))
    return [json.loads(store.get(key)) for key in keys]
