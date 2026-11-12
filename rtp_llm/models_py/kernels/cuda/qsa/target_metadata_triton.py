"""Produce the target's per-query position geometry from live device prefixes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _target_metadata(
    PREFIX,
    VISIBLE,
    COMPRESSED,
    QUERY,
    START,
    N: tl.constexpr,
    Q: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    prefix = tl.load(PREFIX + row // Q, row < N, other=0)
    visible = (prefix + row % Q + 1).to(tl.int32)
    query = (visible - 1).to(tl.int32).to(tl.int64)
    tl.store(VISIBLE + row, visible, row < N)
    tl.store(COMPRESSED + row, visible >> 2, row < N)
    tl.store(QUERY + row, query, row < N)
    tl.store(START + row, query - (query & 3), row < N)


def try_target_metadata(prefixes, query_len, ratio):
    if not (
        prefixes.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(prefixes.device) == (12, 0)
        and prefixes.dtype == torch.int32
        and prefixes.ndim == 1
        and prefixes.is_contiguous()
        and prefixes.numel() > 0
        and 1 <= query_len <= 4
        and ratio == 4
        and prefixes.numel() * query_len <= 128
    ):
        return None
    shape = (prefixes.numel(), query_len)
    visible = torch.empty(shape, device=prefixes.device, dtype=torch.int32)
    compressed = torch.empty_like(visible)
    query = torch.empty(shape, device=prefixes.device, dtype=torch.int64)
    starts = torch.empty_like(query)
    _target_metadata[(triton.cdiv(visible.numel(), 128),)](
        prefixes,
        visible,
        compressed,
        query,
        starts,
        N=visible.numel(),
        Q=query_len,
        BLOCK=128,
        num_warps=4,
    )
    return visible, compressed, query.reshape(-1), starts.reshape(-1)
