"""Scalar numerical oracle for indexer prefill and continuation on valid pages.

Build completed groups from logical token history, then scatter values into the
physical pools. Production validation, snapshots and dispatch are outside this
oracle. Normalization and RoPE use Torch without production kernel helpers.
"""

import torch


def write_indexer_cache(
    raw_keys: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_block_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_block_table: torch.Tensor,
    *,
    ratio: int,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    norm_eps: float = 1e-6,
) -> dict:
    """Update test pools and return per-token destinations and write counts."""
    state_slots = torch.full(
        (len(raw_keys),), -1, dtype=torch.long, device=raw_keys.device
    )
    kv_slots = torch.full_like(state_slots, -1)
    completed = torch.zeros_like(state_slots, dtype=torch.bool)
    ring_size = state_pool.shape[1]
    kv_width = kv_tokens_per_block // ratio
    rotary_dim = rope_cos.shape[-1]

    # Delay writes so continuation groups always read the original history.
    state_writes, kv_writes = [], []
    offsets = cu_seqlens.tolist()
    for request, start in enumerate(start_positions.tolist()):
        lo, hi = offsets[request : request + 2]
        keys = {start + i: key.float() for i, key in enumerate(raw_keys[lo:hi])}

        def state_location(position):
            block = int(state_block_table[request, position // state_tokens_per_block])
            assert 0 < block < len(state_pool)
            return block, position % ring_size

        for position, key in keys.items():
            block, entry = state_location(position)
            state_slots[lo + position - start] = block * ring_size + entry
            state_writes.append((block, entry, key))

        for end in range(start + ratio - 1 - start % ratio, start + hi - lo, ratio):
            begin = end - ratio + 1
            history = [
                keys[pos] if pos in keys else state_pool[state_location(pos)]
                for pos in range(begin, end + 1)
            ]
            mean = torch.stack(history).mean(0).to(raw_keys.dtype).float()
            normalized = (
                mean
                * torch.rsqrt(mean.square().mean() + norm_eps)
                * (1 + k_norm_gamma.float())
            ).to(raw_keys.dtype)
            cos, sin = (
                (table[begin] if table.ndim == 2 else table[request, begin])
                for table in (rope_cos, rope_sin)
            )
            left, right = normalized[:rotary_dim].chunk(2)
            rotated = torch.cat((-right, left))
            value = torch.cat(
                (
                    normalized[:rotary_dim] * cos + rotated * sin,
                    normalized[rotary_dim:],
                )
            ).bfloat16()
            block = int(kv_block_table[request, begin // kv_tokens_per_block])
            assert 0 < block < len(kv_pool)
            entry = (begin // ratio) % kv_width
            row = lo + end - start
            completed[row] = True
            kv_slots[row] = block * kv_width + entry
            kv_writes.append((block, entry, value))

    for pool, writes in ((state_pool, state_writes), (kv_pool, kv_writes)):
        for block, entry, value in writes:
            pool[block, entry].copy_(value)
    return {
        "state_slots": state_slots,
        "kv_slots": kv_slots,
        "completed": completed,
        "num_state_writes": len(state_writes),
        "num_kv_writes": len(kv_writes),
    }
