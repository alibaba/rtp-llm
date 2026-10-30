"""Bounded host-side agreement before optional GPU communication setup."""

import json
import logging
from datetime import timedelta

import torch.distributed as dist

_exchange_epochs = {}


def _exchange_prefix(group, tag):
    # All members execute the same initialization sequence. Include the group
    # and round so reinitialization cannot consume an earlier decision.
    name = getattr(group, "group_name", "default")
    key = (name, tag)
    epoch = _exchange_epochs.get(key, 0) + 1
    _exchange_epochs[key] = epoch
    return f"rtp_llm/collectives/{name}/{epoch}/{tag}/"


def exchange_group_state(group, state, tag):
    prefix = _exchange_prefix(group, tag)
    store = dist.distributed_c10d._get_default_store()
    keys = [prefix + str(rank) for rank in range(dist.get_world_size(group))]
    store.set(keys[dist.get_rank(group)], json.dumps(state).encode())
    store.wait(keys, timedelta(seconds=120))
    return [json.loads(store.get(key)) for key in keys]


def agree_on_optional_collective(group, local_ready, tag):
    """Enable the collective only if the group agrees before any rank falls back."""
    prefix = _exchange_prefix(group, tag)
    store = dist.distributed_c10d._get_default_store()
    keys = [prefix + str(rank) for rank in range(dist.get_world_size(group))]
    rank = dist.get_rank(group)
    decision_key = prefix + "decision"
    store.set(keys[rank], b"1" if local_ready else b"0")
    if store.check([decision_key]):
        return store.get(decision_key) == b"1"

    agreed = False
    if local_ready:
        try:
            store.wait(keys, timedelta(seconds=120))
            agreed = all(vote == b"1" for vote in store.multi_get(keys))
        except dist.DistStoreError:
            missing = [peer for peer, key in enumerate(keys) if not store.check([key])]
            logging.warning(
                "Optional collective %s timed out on TP rank %d; missing votes: %s",
                tag,
                rank,
                missing,
            )

    # A late vote must see the decision already used by ranks that timed out.
    return store.compare_set(decision_key, "", "1" if agreed else "0") == b"1"
