"""One-launch step-plan grid, selection-padding and score-weight prep.

``build_incremental_step_plan`` used to pay a chain of small aten launches
per step-plan window (arange/add/where/floor_divide/remainder/fill plus the
selection-padding host loops and their uploads).  This module writes every
[B, max_len]-grade output of that chain -- the padded visible lengths
(which double as the padded KV lengths), the compressed lengths, the
selection validity mask, the selection row map, the packed valid rows and
both constant score-weight tables -- in a single Triton launch whose only
inputs are the int32 prefix/length mirrors and the cu token offsets the
step plan already uploads as one pinned staging batch.
"""

import torch
import triton as tr
import triton.language as tl

# Lane bound keeps every linear index (including the weight-table row
# offsets) inside int32; larger geometries keep the eager fallback.
_MAX_GRID_LANES = 1 << 30
_MAX_HEAD_NUM = 128
_BLOCK = 64


def incremental_grid_prep_supported(
    device: torch.device, *, batch_size: int, max_len: int, head_num: int
) -> bool:
    """Whether the one-launch prep covers this step geometry.

    CPU fixtures, HIP builds and unusual head counts keep the eager aten
    fallback with identical values.
    """
    return bool(
        device.type == "cuda"
        and torch.version.hip is None
        and batch_size > 0
        and max_len > 0
        and batch_size * max_len <= _MAX_GRID_LANES
        and 0 < head_num <= _MAX_HEAD_NUM
    )


@tr.jit(do_not_specialize=["GRID_TOTAL", "TOKENS", "MAX_LEN"])
def _step_grid_prep_kernel(
    prefixes_ptr,
    lengths_ptr,
    cu_ptr,
    visible_ptr,
    compressed_ptr,
    valid_ptr,
    row_map_ptr,
    packed_rows_ptr,
    weights_ptr,
    padded_weights_ptr,
    score_scale,
    GRID_TOTAL,
    TOKENS,
    MAX_LEN,
    RATIO: tl.constexpr,
    HEAD: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    grid_active = lane < GRID_TOTAL
    request = lane // MAX_LEN
    row = lane % MAX_LEN
    length = tl.load(lengths_ptr + request, mask=grid_active, other=0)
    prefix = tl.load(prefixes_ptr + request, mask=grid_active, other=0)
    packed_base = tl.load(cu_ptr + request, mask=grid_active, other=0)
    valid = row < length
    # Non-negative ints: truncation and floor division agree, matching the
    # legacy aten where/floor_divide chain bit for bit.
    visible = tl.where(valid, prefix + row + 1, 0)
    tl.store(visible_ptr + lane, visible, mask=grid_active)
    tl.store(compressed_ptr + lane, visible // RATIO, mask=grid_active)
    tl.store(valid_ptr + lane, valid, mask=grid_active)
    source_row = packed_base + tl.minimum(row, length - 1)
    tl.store(row_map_ptr + lane, source_row.to(tl.int64), mask=grid_active)
    tl.store(
        packed_rows_ptr + packed_base + row,
        lane.to(tl.int64),
        mask=grid_active & valid,
    )
    # Constant score weights: the same 1/sqrt(head_dim) fill the legacy
    # torch.full/index_select pair produced, written by the same launch.
    head = tl.arange(0, HEAD_BLOCK)
    head_active = head < HEAD
    weight = tl.zeros((BLOCK, HEAD_BLOCK), tl.float32) + score_scale
    wide_lane = lane[:, None].to(tl.int64)
    tl.store(
        padded_weights_ptr + wide_lane * HEAD + head[None, :],
        weight,
        mask=grid_active[:, None] & head_active[None, :],
    )
    token = lane - GRID_TOTAL
    token_active = (lane >= GRID_TOTAL) & (token < TOKENS)
    tl.store(
        weights_ptr + token[:, None].to(tl.int64) * HEAD + head[None, :],
        weight,
        mask=token_active[:, None] & head_active[None, :],
    )


def build_incremental_grid(
    prefixes_i32: torch.Tensor,
    lengths_i32: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    max_len: int,
    token_count: int,
    ratio: int,
    head_num: int,
    score_scale: float,
):
    """Write every [B, max_len]-grade step-plan tensor in one launch.

    Returns ``(visible_lengths, compressed_lengths, selection_valid,
    selection_row_map, selection_valid_rows, score_weights,
    padded_score_weights)`` or ``None`` when the geometry is unsupported
    (the caller falls back to the byte-exact eager chain).

    ``visible_lengths`` is the int32 padded grid the plan also serves as
    ``padded_kv_lens`` and (flattened) ``visible_lengths_i32``; the
    selection row map and packed valid rows reproduce the legacy host loops
    from the same cu token offsets.
    """
    batch = int(prefixes_i32.numel())
    if not incremental_grid_prep_supported(
        prefixes_i32.device, batch_size=batch, max_len=max_len, head_num=head_num
    ):
        return None
    if (
        token_count <= 0
        or ratio <= 0
        or int(lengths_i32.numel()) != batch
        or int(cu_seqlens.numel()) != batch + 1
        or prefixes_i32.dtype != torch.int32
        or lengths_i32.dtype != torch.int32
        or cu_seqlens.dtype != torch.int32
        or not prefixes_i32.is_cuda
        or lengths_i32.device != prefixes_i32.device
        or cu_seqlens.device != prefixes_i32.device
    ):
        return None
    device = prefixes_i32.device
    grid_total = batch * max_len
    visible = torch.empty(grid_total, dtype=torch.int32, device=device)
    compressed = torch.empty(grid_total, dtype=torch.int32, device=device)
    valid = torch.empty(grid_total, dtype=torch.bool, device=device)
    row_map = torch.empty(grid_total, dtype=torch.int64, device=device)
    packed_rows = torch.empty(token_count, dtype=torch.int64, device=device)
    weights = torch.empty((token_count, head_num), dtype=torch.float32, device=device)
    padded_weights = torch.empty(
        (grid_total, head_num), dtype=torch.float32, device=device
    )
    head_block = tr.next_power_of_2(head_num)
    _step_grid_prep_kernel[(tr.cdiv(grid_total + token_count, _BLOCK),)](
        prefixes_i32,
        lengths_i32,
        cu_seqlens,
        visible,
        compressed,
        valid,
        row_map,
        packed_rows,
        weights,
        padded_weights,
        float(score_scale),
        grid_total,
        token_count,
        max_len,
        RATIO=ratio,
        HEAD=head_num,
        HEAD_BLOCK=head_block,
        BLOCK=_BLOCK,
        num_warps=8 if head_block > 32 else 4,
    )
    return (
        visible.view(batch, max_len),
        compressed.view(batch, max_len),
        valid.view(batch, max_len),
        row_map,
        packed_rows,
        weights,
        padded_weights,
    )
