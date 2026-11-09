"""Optional one-launch Triton validation of QSA paged-cache block tables."""

import torch
import triton
import triton.language as tl

_MAX_ELEMENTS = 4096


@triton.jit
def _validate_required_blocks_kernel(
    table_ptr,
    required_ptr,
    result_ptr,
    TABLE_ROW_STRIDE,
    TABLE_COL_STRIDE: tl.constexpr,
    REQUIRED_STRIDE: tl.constexpr,
    BATCH,
    WIDTH,
    POOL_BLOCKS: tl.constexpr,
    N: tl.constexpr,
):
    offset = tl.arange(0, N)
    row = offset // WIDTH
    column = offset % WIDTH
    active = offset < BATCH * WIDTH
    required = tl.load(required_ptr + row * REQUIRED_STRIDE, mask=active, other=0)
    block = tl.load(
        table_ptr + row * TABLE_ROW_STRIDE + column * TABLE_COL_STRIDE,
        mask=active,
        other=0,
    )
    invalid = active & (
        (required < 0)
        | (required > WIDTH)
        | (block >= POOL_BLOCKS)
        | ((column < required) & (block <= 0))
    )
    tl.store(result_ptr, tl.max(invalid.to(tl.int32), 0))


def is_supported(table: torch.Tensor, required_columns: torch.Tensor) -> bool:
    """Keep the one-program reduction bounded; other shapes use the Torch path."""
    return (
        table.is_cuda
        and torch.version.hip is None
        and required_columns.is_cuda
        and table.device == required_columns.device
        and table.dtype == torch.int32
        and required_columns.dtype == torch.int32
        and table.dim() == 2
        and required_columns.dim() == 1
        and table.shape[0] == required_columns.numel()
        and 0 < table.shape[0] * table.shape[1] <= _MAX_ELEMENTS
    )


def required_blocks_are_valid(
    table: torch.Tensor, required_columns: torch.Tensor, pool_blocks: int
) -> bool:
    """Return one device-computed verdict, synchronizing only for that scalar."""
    return bool(required_blocks_verdict(table, required_columns, pool_blocks).item())


def required_blocks_verdict(
    table: torch.Tensor, required_columns: torch.Tensor, pool_blocks: int
) -> torch.Tensor:
    """Launch the one-program reduction and return the raw 0-dim verdict.

    ``True`` means every required column is in range and allocated. Unlike
    :func:`required_blocks_are_valid` this never reads the verdict back to
    the host, so callers can consume it with a device-side assert.
    """
    batch, width = table.shape
    result = torch.empty((), dtype=torch.int32, device=table.device)
    _validate_required_blocks_kernel[(1,)](
        table,
        required_columns,
        result,
        TABLE_ROW_STRIDE=table.stride(0),
        TABLE_COL_STRIDE=table.stride(1),
        REQUIRED_STRIDE=required_columns.stride(0),
        BATCH=batch,
        WIDTH=width,
        POOL_BLOCKS=pool_blocks,
        N=triton.next_power_of_2(batch * width),
    )
    return result == 0


@triton.jit(
    do_not_specialize=[
        "KV_BATCH",
        "KV_WIDTH",
        "KV_POOL_BLOCKS",
        "STATE_BATCH",
        "STATE_WIDTH",
        "STATE_POOL_BLOCKS",
    ]
)
def _dual_table_required_blocks_kernel(
    kv_table_ptr,
    kv_required_ptr,
    state_table_ptr,
    state_required_ptr,
    result_ptr,
    KV_ROW_STRIDE,
    KV_COL_STRIDE,
    KV_REQUIRED_STRIDE,
    STATE_ROW_STRIDE,
    STATE_COL_STRIDE,
    STATE_REQUIRED_STRIDE,
    KV_BATCH,
    KV_WIDTH,
    KV_POOL_BLOCKS,
    STATE_BATCH,
    STATE_WIDTH,
    STATE_POOL_BLOCKS,
    N: tl.constexpr,
):
    # Both reductions share the fixed lane grid; each table's active window
    # is its own BATCH * WIDTH prefix of the same arange, with the same
    # per-table terms as _validate_required_blocks_kernel.
    offset = tl.arange(0, N)

    kv_row = offset // KV_WIDTH
    kv_column = offset % KV_WIDTH
    kv_active = offset < KV_BATCH * KV_WIDTH
    kv_required = tl.load(
        kv_required_ptr + kv_row * KV_REQUIRED_STRIDE, mask=kv_active, other=0
    )
    kv_block = tl.load(
        kv_table_ptr + kv_row * KV_ROW_STRIDE + kv_column * KV_COL_STRIDE,
        mask=kv_active,
        other=0,
    )
    kv_invalid = kv_active & (
        (kv_required < 0)
        | (kv_required > KV_WIDTH)
        | (kv_block >= KV_POOL_BLOCKS)
        | ((kv_column < kv_required) & (kv_block <= 0))
    )

    state_row = offset // STATE_WIDTH
    state_column = offset % STATE_WIDTH
    state_active = offset < STATE_BATCH * STATE_WIDTH
    state_required = tl.load(
        state_required_ptr + state_row * STATE_REQUIRED_STRIDE,
        mask=state_active,
        other=0,
    )
    state_block = tl.load(
        state_table_ptr + state_row * STATE_ROW_STRIDE + state_column * STATE_COL_STRIDE,
        mask=state_active,
        other=0,
    )
    state_invalid = state_active & (
        (state_required < 0)
        | (state_required > STATE_WIDTH)
        | (state_block >= STATE_POOL_BLOCKS)
        | ((state_column < state_required) & (state_block <= 0))
    )

    valid = (tl.max(kv_invalid.to(tl.int32), 0) == 0) & (
        tl.max(state_invalid.to(tl.int32), 0) == 0
    )
    tl.store(result_ptr, valid.to(tl.int8))


def dual_required_blocks_supported(
    kv_table: torch.Tensor,
    kv_required_columns: torch.Tensor,
    state_table: torch.Tensor,
    state_required_columns: torch.Tensor,
) -> bool:
    """Keep the shared one-program reduction bounded; others fall back."""
    return (
        kv_table.is_cuda
        and torch.version.hip is None
        and kv_required_columns.is_cuda
        and state_table.is_cuda
        and state_required_columns.is_cuda
        and kv_table.device == kv_required_columns.device
        and kv_table.device == state_table.device
        and kv_table.device == state_required_columns.device
        and kv_table.dtype == torch.int32
        and state_table.dtype == torch.int32
        and kv_required_columns.dtype == torch.int32
        and state_required_columns.dtype == torch.int32
        and kv_table.dim() == 2
        and state_table.dim() == 2
        and kv_required_columns.dim() == 1
        and state_required_columns.dim() == 1
        and kv_table.shape[0] == kv_required_columns.numel()
        and state_table.shape[0] == state_required_columns.numel()
        and 0 < kv_table.shape[0] * kv_table.shape[1] <= _MAX_ELEMENTS
        and 0 < state_table.shape[0] * state_table.shape[1] <= _MAX_ELEMENTS
    )


def dual_required_blocks_verdict(
    kv_table: torch.Tensor,
    kv_required_columns: torch.Tensor,
    state_table: torch.Tensor,
    state_required_columns: torch.Tensor,
    *,
    kv_pool_blocks: int,
    state_pool_blocks: int,
) -> torch.Tensor:
    """One-launch verdict over BOTH side-cache block tables (True=valid).

    The warm light-layer census paid six device launches per layer on the two
    separate verdict groups (two reductions, two ``result == 0`` conversions
    and two asserts).  Both groups validate step-frozen inputs, so this
    reduction covers both tables in one program and stores the combined
    boolean directly (no elementwise conversion), for one kernel plus one
    async assert per layer.  The fixed ``N`` lane bound and the
    ``do_not_specialize`` shape arguments keep one compiled variant per
    stride-alignment class for every supported table shape.
    """
    result = torch.empty((), dtype=torch.int8, device=kv_table.device)
    _dual_table_required_blocks_kernel[(1,)](
        kv_table,
        kv_required_columns,
        state_table,
        state_required_columns,
        result,
        KV_ROW_STRIDE=kv_table.stride(0),
        KV_COL_STRIDE=kv_table.stride(1),
        KV_REQUIRED_STRIDE=kv_required_columns.stride(0),
        STATE_ROW_STRIDE=state_table.stride(0),
        STATE_COL_STRIDE=state_table.stride(1),
        STATE_REQUIRED_STRIDE=state_required_columns.stride(0),
        KV_BATCH=kv_table.shape[0],
        KV_WIDTH=kv_table.shape[1],
        KV_POOL_BLOCKS=kv_pool_blocks,
        STATE_BATCH=state_table.shape[0],
        STATE_WIDTH=state_table.shape[1],
        STATE_POOL_BLOCKS=state_pool_blocks,
        N=_MAX_ELEMENTS,
    )
    return result.view(torch.bool)


@triton.jit
def _main_geometry_verdict_kernel(
    table_ptr,
    bases_ptr,
    lengths_ptr,
    result_ptr,
    TABLE_ROW_STRIDE,
    TABLE_COL_STRIDE,
    BASES_STRIDE,
    LENGTHS_STRIDE,
    BATCH,
    WIDTH,
    PAGE,
    POOL_BLOCKS,
    N: tl.constexpr,
):
    offset = tl.arange(0, N)
    row = offset // WIDTH
    column = offset % WIDTH
    active = offset < BATCH * WIDTH
    base = tl.load(bases_ptr + row * BASES_STRIDE, mask=active, other=0)
    length = tl.load(lengths_ptr + row * LENGTHS_STRIDE, mask=active, other=0)
    block = tl.load(
        table_ptr + row * TABLE_ROW_STRIDE + column * TABLE_COL_STRIDE,
        mask=active,
        other=0,
    )
    required = (base + length + PAGE - 1) // PAGE
    invalid = active & (
        (required > WIDTH)
        | (block >= POOL_BLOCKS)
        | ((column < required) & ((block <= 0) | (block >= POOL_BLOCKS)))
    )
    verdict = tl.min((invalid == 0).to(tl.int32), 0)
    tl.store(result_ptr, verdict.to(tl.int8))


def main_geometry_supported(
    table: torch.Tensor,
    sequence_bases: torch.Tensor,
    query_lengths: torch.Tensor,
) -> bool:
    """Keep the one-program main-table reduction bounded; others use Torch."""
    return (
        table.is_cuda
        and torch.version.hip is None
        and sequence_bases.is_cuda
        and query_lengths.is_cuda
        and table.device == sequence_bases.device
        and table.device == query_lengths.device
        and table.dtype == torch.int32
        and sequence_bases.dtype == torch.int32
        and query_lengths.dtype == torch.int32
        and table.dim() == 2
        and sequence_bases.dim() == 1
        and query_lengths.dim() == 1
        and table.shape[0] == sequence_bases.numel()
        and table.shape[0] == query_lengths.numel()
        and 0 < table.shape[0] * table.shape[1] <= _MAX_ELEMENTS
    )


def main_geometry_verdict(
    table: torch.Tensor,
    sequence_bases: torch.Tensor,
    query_lengths: torch.Tensor,
    *,
    pool_blocks: int,
    page_size: int,
) -> torch.Tensor:
    """Launch the one-program main-table reduction; return the 0-dim verdict.

    ``True`` means the table covers every request's visible KV and every
    required column resolves to an allocated in-range physical block — exactly
    the replaced aten chain's three terms (required-column coverage, the
    whole-table out-of-range scan that covers stale columns, and the
    required-and-invalid intersection). The caller checks
    :func:`main_geometry_supported` first; like :func:`required_blocks_verdict`
    this never reads the verdict back to the host, so it composes with a
    device-side assert. The fixed ``N`` bound and runtime shape arguments keep
    one compiled variant for every supported table shape.
    """
    result = torch.empty((), dtype=torch.int8, device=table.device)
    _main_geometry_verdict_kernel[(1,)](
        table,
        sequence_bases,
        query_lengths,
        result,
        TABLE_ROW_STRIDE=table.stride(0),
        TABLE_COL_STRIDE=table.stride(1),
        BASES_STRIDE=sequence_bases.stride(0),
        LENGTHS_STRIDE=query_lengths.stride(0),
        BATCH=table.shape[0],
        WIDTH=table.shape[1],
        PAGE=page_size,
        POOL_BLOCKS=pool_blocks,
        N=_MAX_ELEMENTS,
    )
    return result.view(torch.bool)
