"""Exchange only the tails of the two zigzag chunks owned per request."""
from bisect import bisect_right
from functools import lru_cache

import torch.distributed as dist
import triton
import triton.language as tl

from .kcp_plan import make_plan
from .kcp_bm_pipeline import gather_buffer


@lru_cache(maxsize=128)
def halo_plan(cu_values, world, page, rank, history):
    plan = make_plan(cu_values, world, page)
    chunks = plan['chunks']
    starts = [c[0] for c in chunks]
    owned = sorted((c for c in chunks if c[3] == rank), key=lambda c: c[4])
    slots = {}
    for owner in range(world):
        row = sorted((c for c in chunks if c[3] == owner), key=lambda c: c[4])
        for slot, chunk in enumerate(row):
            slots[chunk[0]] = owner * len(owned) + slot
    kinds, indices = [], []
    for segment in plan['by_rank'][rank]:
        begin, _, seq, _, _ = plan['segments'][segment]
        for pos in range(begin - history, begin):
            if pos < cu_values[seq]:
                kinds.append(2)
                indices.append(seq * history + pos - cu_values[seq] + history)
                continue
            chunk = chunks[bisect_right(starts, pos) - 1]
            if chunk[3] == rank:
                kinds.append(0)
                indices.append(chunk[4] + pos - chunk[0])
            else:
                end = chunk[1]
                if not end - history <= pos < end:
                    raise ValueError('Convolution halo is outside the predecessor chunk tail')
                kinds.append(1)
                indices.append(slots[chunk[0]] * history + pos - end + history)
    tails = tuple(c[4] + c[1] - c[0] - history + j
                  for c in owned for j in range(history))
    return tuple(kinds), tuple(indices), tails, len(owned) * history


@triton.jit
def _pack_tails(X, INDEX, OUT, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                BLOCK: tl.constexpr):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = x < ROWS * WIDTH
    row = tl.load(INDEX + x // WIDTH, valid, 0).to(tl.int64)
    value = tl.load(X + row * WIDTH + x % WIDTH, valid, 0)
    tl.store(OUT + x, value, valid)


def exchange_halo(mixed, tail_index, rows_per_rank, group):
    gathered = gather_buffer(group, (rows_per_rank, mixed.shape[1]), mixed.dtype,
                             mixed.device, 'conv_halo_chunks')
    rank = dist.get_rank(group)
    local = gathered[rank]
    _pack_tails[(triton.cdiv(local.numel(), 512),)](
        mixed, tail_index, local, ROWS=rows_per_rank, WIDTH=mixed.shape[1], BLOCK=512)
    dist.all_gather_into_tensor(gathered.flatten(0, 1), local, group=group)
    return gathered
