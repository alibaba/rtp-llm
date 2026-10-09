"""Reuse position conversion and rotary-table reads within one V4.1 forward."""

import os

import torch


def decode_rope_metadata(shared, table, positions):
    if os.environ.get("DSV41_REUSE_DECODE_ROPE", "1") != "1":
        converted = positions.reshape(-1).to(device=table.device, dtype=torch.long)
        return converted, table.index_select(0, converted).contiguous()
    # The first layer clears this dictionary every forward. Tensor references
    # keep storage alive; graph replay reruns the captured producers on updates.
    cache = shared.setdefault("decode_rope_metadata", {})
    position_key = (
        "positions",
        positions.data_ptr(),
        tuple(positions.shape),
        tuple(positions.stride()),
        positions.dtype,
        positions.device,
        table.device,
    )
    entry = cache.get(position_key)
    if entry is None:
        converted = positions.reshape(-1).to(device=table.device, dtype=torch.long)
        cache[position_key] = (positions, converted)
    else:
        _, converted = entry
    table_key = ("table", id(table), position_key)
    entry = cache.get(table_key)
    if entry is None:
        freqs = table.index_select(0, converted).contiguous()
        cache[table_key] = (table, freqs)
    else:
        _, freqs = entry
    return converted, freqs
