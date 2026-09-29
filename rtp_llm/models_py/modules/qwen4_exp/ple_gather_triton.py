"""Fixed-shape, graph-capturable lookup into the TP-local PLE row shards."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_shards(
    ids_ptr,
    shard_ptrs,
    out_ptr,
    N: tl.constexpr,
    SHARD_ROWS: tl.constexpr,
    TOTAL_SHARDS: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BF16: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.arange(0, BD)
    ids = tl.load(ids_ptr + rows, mask=rows < N, other=0)
    shard = ids // SHARD_ROWS
    local_row = ids % SHARD_ROWS
    in_range = (rows < N) & (ids >= 0) & (shard < TOTAL_SHARDS)
    addr = tl.load(shard_ptrs + shard, mask=in_range, other=0)
    valid = in_range & (addr != 0)
    if BF16:
        base = addr.to(tl.pointer_type(tl.bfloat16))
    else:
        base = addr.to(tl.pointer_type(tl.float32))
    values = tl.load(
        base[:, None] + local_row[:, None] * D + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < D),
        other=0,
    )
    tl.store(
        out_ptr + rows[:, None] * D + cols[None, :],
        values,
        mask=(rows[:, None] < N) & (cols[None, :] < D),
    )


def is_supported(ids: torch.Tensor, shards: list[torch.Tensor]) -> bool:
    if not ids.is_cuda or torch.version.hip is not None or ids.dtype != torch.int64:
        return False
    if not shards or shards[0].dtype not in (torch.bfloat16, torch.float32):
        return False
    shape = shards[0].shape
    return (
        len(shape) == 2
        and 0 < shape[1] <= 256
        and all(
            shard.is_cuda
            and shard.device == ids.device
            and shard.dtype == shards[0].dtype
            and shard.shape == shape
            and shard.is_contiguous()
            for shard in shards
        )
    )


def gather_local(
    ids: torch.Tensor,
    shards: list[torch.Tensor],
    shard_indices: tuple[int, ...],
    total_shards: int,
    pointer_table: torch.Tensor,
) -> torch.Tensor:
    """Return local shard values, leaving remote shard values at zero."""
    if not is_supported(ids, shards):
        raise ValueError("unsupported PLE Triton shard layout")
    if pointer_table.shape != (total_shards,) or pointer_table.dtype != torch.int64:
        raise ValueError("PLE pointer table must cover every global shard")
    if len(shard_indices) != len(shards):
        raise ValueError("PLE shard index count mismatch")
    ids = ids.contiguous()
    out = torch.empty((*ids.shape, shards[0].shape[1]), device=ids.device, dtype=shards[0].dtype)
    n = ids.numel()
    if n:
        _gather_shards[(triton.cdiv(n, 16),)](
            ids,
            pointer_table,
            out,
            N=n,
            SHARD_ROWS=shards[0].shape[0],
            TOTAL_SHARDS=total_shards,
            D=shards[0].shape[1],
            BD=triton.next_power_of_2(shards[0].shape[1]),
            BF16=shards[0].dtype == torch.bfloat16,
            num_warps=4,
        )
    return out
