"""Decode-only adapter from RTP's persistent 656-byte KV to TRT's FP8 KV.

All storage is supplied by the caller, including compacted physical indices.
The adapter neither changes persistent KV nor applies RoPE a second time.
Its reader contract matches the single-selected-token probes of the deployed
FlashMLA 1.0.0+cb10b79 Decode wheel: round scale to BF16, multiply the FP8 value,
then round the product to BF16 before saturated RNE FP8. This differs from
RTP's FP32-scale Prefill upconvert and is still a lossy conversion to native FP8.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_indices(
    TopK,
    Requests,
    Table,
    SeqLens,
    Sources,
    Indices,
    Counts,
    Lengths,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    REQUESTS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    Physical=None,
    HAS_PHYSICAL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK_K)
    req = tl.load(Requests + row).to(tl.int64)
    length = tl.load(SeqLens + row).to(tl.int64)
    logical = tl.load(TopK + row * K + col, col < K, other=-1).to(tl.int64)
    logical_page = logical // PAGE
    valid = (
        (col < K)
        & (req >= 0)
        & (req < REQUESTS)
        & (logical >= 0)
        & (logical < length)
        & (logical_page >= 0)
        & (logical_page < TABLE_COLS)
    )
    if HAS_PHYSICAL:
        # Pinned MLA has already mapped backing-store IDs into its resident
        # working set. These are token offsets, not logical pages; slot zero
        # is valid here, unlike reserved page zero in the ordinary KV cache.
        physical = tl.load(Physical + row * K + col, valid, other=-1).to(tl.int64)
        valid &= (physical >= 0) & (physical < PAGES * PAGE)
    else:
        page = tl.load(Table + req * TABLE_STRIDE + logical_page, valid, other=0).to(
            tl.int64
        )
        valid &= (page > 0) & (page < PAGES)
        physical = page * PAGE + logical % PAGE
    position = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    count = tl.sum(valid.to(tl.int32), axis=0)
    # Disjoint stores: valid entries occupy [0,count), the other store clears
    # [count,K). No initialization/scatter race or inter-CTA barrier is needed.
    tl.store(Sources + row * K + position, physical, valid)
    tl.store(Sources + row * K + col, -1, (col < K) & (col >= count))
    use_slot = (col < count) | ((count == 0) & (col == 0))
    tl.store(Indices + row * K + col, tl.where(use_slot, row * K + col, -1), col < K)
    tl.store(Counts + row, count)
    # TRT never receives a zero-length row; a zero-value sentinel represents
    # empty rows, and mask_empty_output enforces their exact-zero contract.
    tl.store(Lengths + row, tl.maximum(count, 1))


@triton.jit
def _fp8_sat_rne(value):
    value = tl.minimum(
        tl.maximum(value, -448.0, propagate_nan=tl.PropagateNan.ALL),
        448.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    return value.to(tl.float8e4nv, fp_downcast_rounding="rtne")


@triton.jit
def _convert_group(
    KV,
    Sources,
    Out,
    Q,
    QOut,
    row,
    tile,
    count,
    K: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    LOAD_CACHE: tl.constexpr,
):
    selected = tile * GROUP + tl.arange(0, GROUP)
    if tile * GROUP < tl.maximum(count, 1):
        valid = (selected < count) & (selected < K)
        active = valid | ((count == 0) & (selected == 0))
        source = tl.load(Sources + row * K + selected, valid, other=0)
        token = source * 656
        scale_group = tl.arange(0, 4)
        inner = tl.arange(0, 128)
        # Load each scale once before introducing the 128-element axis.
        scales = (
            tl.load(
                (KV + token[:, None] + 512 + scale_group[None, :] * 4).to(
                    tl.pointer_type(tl.float32)
                ),
                valid[:, None],
                other=0.0,
                cache_modifier=LOAD_CACHE,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        offsets = (
            token[:, None, None]
            + scale_group[None, :, None] * 128
            + inner[None, None, :]
        )
        value = tl.load(
            (KV + offsets).to(tl.pointer_type(tl.float8e4nv)),
            valid[:, None, None],
            other=0.0,
            cache_modifier=LOAD_CACHE,
        ).to(tl.float32)
        # Both BF16 boundaries are required by the actual FlashMLA Decode ABI.
        latent = (value * scales[:, :, None]).to(tl.bfloat16).to(tl.float32)
        destination = (row * K + selected) * 576
        tl.store(
            Out
            + destination[:, None, None]
            + scale_group[None, :, None] * 128
            + inner[None, None, :],
            _fp8_sat_rne(latent),
            active[:, None, None],
        )
        rope_dim = tl.arange(0, 64)
        rope = tl.load(
            (KV + token[:, None] + 528 + rope_dim[None, :] * 2).to(
                tl.pointer_type(tl.bfloat16)
            ),
            valid[:, None],
            other=0.0,
            cache_modifier=LOAD_CACHE,
        ).to(tl.float32)
        tl.store(
            Out + destination[:, None] + 512 + rope_dim[None, :],
            _fp8_sat_rne(rope),
            active[:, None],
        )
    # Q must be overwritten for every head, including empty rows. This branch
    # must never be nested under the selected-KV validity branch above.
    if tile < HEADS:
        column = tl.arange(0, 1024)
        value = tl.load(
            Q + (row * HEADS + tile) * 576 + column,
            (count > 0) & (column < 576),
            other=0.0,
        ).to(tl.float32)
        tl.store(
            QOut + (row * HEADS + tile) * 576 + column,
            _fp8_sat_rne(value),
            column < 576,
        )


@triton.jit
def _convert_qkv(
    KV,
    Sources,
    Counts,
    Out,
    Q,
    QOut,
    K: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    WORK_ITEMS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    PERSISTENT: tl.constexpr,
    LOAD_CACHE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    worker = tl.program_id(1).to(tl.int64)
    count = tl.load(Counts + row)
    if PERSISTENT:
        for tile in range(worker, WORK_ITEMS, PROGRAMS):
            _convert_group(
                KV, Sources, Out, Q, QOut, row, tile, count, K, HEADS, GROUP, LOAD_CACHE
            )
    else:
        _convert_group(
            KV, Sources, Out, Q, QOut, row, worker, count, K, HEADS, GROUP, LOAD_CACHE
        )


@triton.jit
def _mask_output(Out, Counts, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    if tl.load(Counts + row) == 0:
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Out + row * WIDTH + col, 0, col < WIDTH)


def _require_tensor(name, tensor, shape, dtype, device, *, contiguous=True):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {tuple(shape)} and dtype {dtype}")
    if tensor.device != device or tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on the same CUDA device as q")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def convert_selected_kv(
    q,
    kv,
    topk,
    req_ids,
    block_table,
    seq_lens,
    *,
    q_out,
    kv_out,
    source_indices,
    indices_out,
    counts_out,
    lengths_out,
    physical_indices=None,
):
    """Write caller-owned selected FP8 cache, query and metadata, without allocation.

    q/q_out: [T,H,576] BF16/FP8, already RoPE'd. kv: contiguous uint8
    [pages,page_size,656] (or [pages,page_size,1,656]); page zero is reserved.
    topk: [T,K] int32 request-local positions; req_ids/seq_lens: [T] int32,
    where seq_lens is the individual query's exclusive causal KV bound.
    block_table: [requests,max_pages] int32, contiguous within each row.
    kv_out: [T,K,576] FP8, with K a multiple of 64. source_indices: [T,K]
    int64; indices_out: [T,K] int32; counts_out/lengths_out: [T] int32.

    Optional physical_indices: [T,K] int32 token offsets in resident KV,
    aligned with the original topk entries. Bypass block-table translation
    but retain logical causal filtering; resident slot zero is valid. The
    caller must join the working set's prefetch/write before this call.
    This input mapping is distinct from indices_out into temporary kv_out.

    Compaction is stable and preserves duplicates. Each row owns K scratch
    slots. Only valid slots (or slot zero for an empty row) are overwritten;
    unused KV tails are intentionally unspecified and excluded by metadata.
    Input/output storage must not alias. The caller owns stream synchronization
    and lifetime: these buffers cannot be reused by concurrent graph replays.
    """
    if q.ndim != 3 or q.shape[-1] != 576:
        raise ValueError("q must have shape [T,H,576]")
    rows, heads, _ = q.shape
    if heads == 0 or topk.ndim != 2 or topk.shape[0] != rows:
        raise ValueError("q needs heads and topk must have shape [T,K]")
    k = topk.shape[1]
    if k == 0 or k % 64 or k > 8192 or rows * k > 2**31 - 1:
        raise ValueError(
            "K must be a positive multiple of 64 <= 8192; T*K must fit int32"
        )
    if kv.ndim == 4:
        if kv.shape[2:] != (1, 656):
            raise ValueError("kv must have shape [pages,page_size,1,656]")
    elif kv.ndim != 3 or kv.shape[-1] != 656:
        raise ValueError("kv must have shape [pages,page_size,656]")
    if kv.shape[1] == 0 or block_table.ndim != 2 or block_table.stride(1) != 1:
        raise ValueError(
            "kv needs positive page_size and block_table needs contiguous rows"
        )
    device = q.device
    for name, tensor, shape, dtype in (
        ("q", q, (rows, heads, 576), torch.bfloat16),
        ("kv", kv, kv.shape, torch.uint8),
        ("topk", topk, (rows, k), torch.int32),
        ("req_ids", req_ids, (rows,), torch.int32),
        ("seq_lens", seq_lens, (rows,), torch.int32),
        ("q_out", q_out, q.shape, torch.float8_e4m3fn),
        ("kv_out", kv_out, (rows, k, 576), torch.float8_e4m3fn),
        ("source_indices", source_indices, (rows, k), torch.int64),
        ("indices_out", indices_out, (rows, k), torch.int32),
        ("counts_out", counts_out, (rows,), torch.int32),
        ("lengths_out", lengths_out, (rows,), torch.int32),
    ):
        _require_tensor(name, tensor, shape, dtype, device)
    _require_tensor(
        "block_table",
        block_table,
        block_table.shape,
        torch.int32,
        device,
        contiguous=False,
    )
    if physical_indices is not None:
        _require_tensor(
            "physical_indices", physical_indices, (rows, k), torch.int32, device
        )
    if rows == 0:
        return
    _compact_indices[(rows,)](
        topk,
        req_ids,
        block_table,
        seq_lens,
        source_indices,
        indices_out,
        counts_out,
        lengths_out,
        K=k,
        BLOCK_K=triton.next_power_of_2(k),
        REQUESTS=block_table.shape[0],
        TABLE_COLS=block_table.shape[1],
        PAGE=kv.shape[1],
        PAGES=kv.shape[0],
        TABLE_STRIDE=block_table.stride(0),
        Physical=physical_indices,
        HAS_PHYSICAL=physical_indices is not None,
    )
    # Group four selected tokens per CTA and share the four scale loads.
    # Fuse Q conversion into the same launch. Use only static shape metadata;
    # counts/indices remain device-side and are refreshed on every execution.
    group = 4
    work_items = max(triton.cdiv(k, group), heads)
    _convert_qkv[(rows, work_items)](
        kv,
        source_indices,
        counts_out,
        kv_out,
        q,
        q_out,
        K=k,
        HEADS=heads,
        GROUP=group,
        WORK_ITEMS=work_items,
        PROGRAMS=work_items,
        PERSISTENT=False,
        LOAD_CACHE="",
        num_warps=1,
        enable_fp_fusion=False,
    )


def mask_empty_output(out, counts):
    """Enforce exact zero output for empty rows; do not read their old output."""
    if out.ndim != 3 or out.shape[-1] != 512 or not out.is_contiguous():
        raise ValueError("out must be contiguous [T,H,512]")
    if out.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("out must be BF16, FP16 or FP32")
    _require_tensor("counts", counts, (out.shape[0],), torch.int32, out.device)
    if out.numel():
        width = out.shape[1] * out.shape[2]
        _mask_output[(out.shape[0], triton.cdiv(width, 1024))](
            out,
            counts,
            WIDTH=width,
            BLOCK=1024,
        )
