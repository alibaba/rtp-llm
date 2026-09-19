"""Opt-in CP prefill with selected BF16 working pages.

Communication is unchanged: suffix is gathered in BF16 and the complete prefix
is gathered in persistent storage dtype. Only main BF16 materialization is
query-chunk local. Scratch page IDs are request-local; no physical alias dedup.
"""

import torch
import triton
import triton.language as tl


def validate_compact_mode(packed_overlap, prefix_prefetch):
    if packed_overlap or prefix_prefetch:
        raise ValueError(
            "M3_MSA_CP_COMPACT_PREFILL requires RTP_LLM_CP_PACKED_KV_OVERLAP=0 "
            "and RTP_LLM_CP_PREFIX_PREFETCH=0"
        )


def selected_page_map_reference(topk, page_table, query_boundaries):
    """Return selected scratch page IDs and a logical-to-compact page table."""
    selected = []
    for row, (lo, hi) in enumerate(zip(query_boundaries, query_boundaries[1:])):
        logical = topk[:, lo:hi].reshape(-1).long()
        logical = logical[logical >= 0].unique(sorted=True)
        selected.append((row, logical, page_table[row, logical].long()))
    ids = torch.cat([physical for _, _, physical in selected]).unique(sorted=True)
    width = page_table.shape[1]
    storage = torch.zeros(
        (page_table.shape[0], triton.cdiv(width, 4) * 4),
        dtype=torch.int32,
        device=page_table.device,
    )
    table = storage[:, :width]
    for row, logical, physical in selected:
        table[row, logical] = torch.searchsorted(ids, physical).to(torch.int32) + 1
    return ids, table


@triton.jit
def _mark_logical_pages_kernel(
    topk,
    query_segments,
    marked,
    ELEMENTS: tl.constexpr,
    QUERIES: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    MARK_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(topk + offset, offset < ELEMENTS, other=-1)
    query = (offset // TOPK) % QUERIES
    segment = tl.load(query_segments + query, offset < ELEMENTS, other=0).to(tl.int64)
    valid = (offset < ELEMENTS) & (logical >= 0) & (logical < WIDTH)
    tl.atomic_or(marked + segment * MARK_STRIDE + logical, 1, mask=valid, sem="relaxed")


@triton.jit
def _mark_scratch_pages_kernel(
    page_table,
    marked,
    flags,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    MARK_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, column = offset // WIDTH, offset % WIDTH
    active = tl.load(marked + row * MARK_STRIDE + column, row < ROWS, other=0)
    physical = tl.load(page_table + row * PAGE_STRIDE + column, row < ROWS, other=-1)
    valid = (row < ROWS) & (active != 0) & (physical >= 0) & (physical < CAPACITY)
    tl.atomic_or(flags + physical, 1, mask=valid, sem="relaxed")


@triton.jit
def _compact_page_table_kernel(
    page_table,
    marked,
    prefix_sum,
    result,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    MARK_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, column = offset // WIDTH, offset % WIDTH
    active = tl.load(marked + row * MARK_STRIDE + column, row < ROWS, other=0)
    physical = tl.load(page_table + row * PAGE_STRIDE + column, row < ROWS, other=-1)
    valid = (row < ROWS) & (active != 0) & (physical >= 0) & (physical < CAPACITY)
    compact_id = tl.load(prefix_sum + physical, valid, other=0)
    tl.store(result + row * MARK_STRIDE + column, compact_id, row < ROWS)


def selected_page_map(
    topk,
    page_table,
    query_boundaries,
    *,
    page_capacity=None,
    query_segments=None,
):
    """Use bounded GPU bitmaps when geometry supplies the scratch namespace.

    CPU/legacy callers retain the reference path. Logical-row marks preserve
    exact reference table contents even when two segments alias one request.
    Only nonzero produces a dynamic output; no device max/item sizes buffers.
    """
    if not topk.is_cuda or page_capacity is None:
        return selected_page_map_reference(topk, page_table, query_boundaries)
    if not 0 < page_capacity < 2**31:
        raise ValueError("page_capacity must fit a positive int32 page namespace")
    if topk.ndim != 3 or topk.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk must be an integer [heads, queries, topk] tensor")
    if page_table.ndim != 2 or min(page_table.shape) == 0 or page_table.stride(1) != 1:
        raise ValueError("page table must have nonempty rows and contiguous columns")
    if (
        len(query_boundaries) != page_table.shape[0] + 1
        or query_boundaries[0] != 0
        or query_boundaries[-1] != topk.shape[1]
        or any(lo > hi for lo, hi in zip(query_boundaries, query_boundaries[1:]))
    ):
        raise ValueError("query boundaries must match segment rows and query count")
    if query_segments is None:
        query_segments = _query_segment_ids(query_boundaries, topk.device)
    if (
        query_segments.shape != (topk.shape[1],)
        or query_segments.dtype != torch.int32
        or query_segments.device != topk.device
        or not query_segments.is_contiguous()
        or page_table.device != topk.device
        or page_table.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "query-segment map and page table must match topk device/layout"
        )
    if not topk.is_contiguous():
        topk = topk.contiguous()
    rows, width = page_table.shape
    stride = triton.cdiv(width, 4) * 4
    marked = torch.zeros((rows, stride), dtype=torch.int32, device=topk.device)
    flags = torch.zeros(page_capacity, dtype=torch.int32, device=topk.device)
    if topk.numel():
        _mark_logical_pages_kernel[(triton.cdiv(topk.numel(), 1024),)](
            topk,
            query_segments,
            marked,
            ELEMENTS=topk.numel(),
            QUERIES=topk.shape[1],
            TOPK=topk.shape[2],
            WIDTH=width,
            MARK_STRIDE=stride,
            BLOCK=1024,
        )
    _mark_scratch_pages_kernel[(triton.cdiv(rows * width, 1024),)](
        page_table,
        marked,
        flags,
        ROWS=rows,
        WIDTH=width,
        PAGE_STRIDE=page_table.stride(0),
        MARK_STRIDE=stride,
        CAPACITY=page_capacity,
        BLOCK=1024,
    )
    prefix_sum = torch.cumsum(flags, dim=0, dtype=torch.int32)
    ids = torch.nonzero(flags, as_tuple=False).reshape(-1)
    result = torch.zeros_like(marked)
    _compact_page_table_kernel[(triton.cdiv(rows * width, 1024),)](
        page_table,
        marked,
        prefix_sum,
        result,
        ROWS=rows,
        WIDTH=width,
        PAGE_STRIDE=page_table.stride(0),
        MARK_STRIDE=stride,
        CAPACITY=page_capacity,
        BLOCK=1024,
    )
    return ids, result[:, :width]


def _query_segment_ids(boundaries, device):
    segments = []
    for row, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        segments.extend([row] * (hi - lo))
    return torch.tensor(segments, dtype=torch.int32, device=device)


def build_source_metadata(
    prefix_lengths,
    kv_lens,
    scratch_seq_len,
    page_size,
    prefix_dst_pages,
    prefix_restore_rows,
):
    """Metadata uses the existing request-major scratch namespace, not CP order."""
    if scratch_seq_len % page_size:
        raise ValueError("scratch sequence length must be page aligned")
    if kv_lens.ndim != 1 or len(prefix_lengths) != kv_lens.numel():
        raise ValueError("prefix and KV lengths must have matching request rows")
    prefix = torch.tensor(prefix_lengths, dtype=torch.int64, device=kv_lens.device)
    if any(p < 0 or p % page_size for p in prefix_lengths):
        raise ValueError("compact prefix lengths must be nonnegative and page aligned")
    lens = kv_lens.to(torch.int64)
    suffix_lengths = lens - prefix
    offsets = torch.cat((lens.new_zeros(1), suffix_lengths.cumsum(0)[:-1]))
    rows = torch.full(
        (len(prefix_lengths) * (scratch_seq_len // page_size),),
        -1,
        dtype=torch.int64,
        device=kv_lens.device,
    )
    if prefix_restore_rows is None:
        prefix_restore_rows = torch.arange(
            prefix_dst_pages.numel(), device=kv_lens.device, dtype=torch.int64
        )
    rows[prefix_dst_pages.long()] = prefix_restore_rows
    return prefix, lens, offsets, rows


@triton.jit
def _restore_idx_pages_kernel(
    src,
    scales,
    source_rows,
    dst_pages,
    dst,
    PAGE_ELEMS: tl.constexpr,
    IDX_DIM: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0).to(tl.int64)
    off = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(source_rows + page).to(tl.int64)
    target = tl.load(dst_pages + page).to(tl.int64)
    values = tl.load(src + source * PAGE_ELEMS + off, off < PAGE_ELEMS, other=0.0)
    if HAS_SCALE:
        scale = tl.load(
            scales + source * (PAGE_ELEMS // IDX_DIM) + off // IDX_DIM,
            off < PAGE_ELEMS,
            other=0.0,
        )
        values = values * scale
    tl.store(dst + target * PAGE_ELEMS + off, values, off < PAGE_ELEMS)


def restore_idx_pages(
    idx_pages, dst_pages, idx_scratch, restore_rows=None, idx_scales=None
):
    if not dst_pages.numel():
        return
    if restore_rows is None:
        restore_rows = torch.arange(dst_pages.numel(), device=dst_pages.device)
    page_elems = idx_pages.shape[1] * idx_pages.shape[2]
    _restore_idx_pages_kernel[(dst_pages.numel(), triton.cdiv(page_elems, 1024))](
        idx_pages,
        idx_scales if idx_scales is not None else idx_pages,
        restore_rows,
        dst_pages,
        idx_scratch,
        PAGE_ELEMS=page_elems,
        IDX_DIM=int(idx_pages.shape[2]),
        HAS_SCALE=idx_scales is not None,
        BLOCK=1024,
    )


@triton.jit
def _materialize_pages_kernel(
    selected,
    prefix_main,
    packed,
    unpad,
    prefix,
    lens,
    suffix_offsets,
    prefix_rows,
    out_k,
    out_v,
    PAGES_PER_REQUEST: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    NK: tl.constexpr,
    NI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    selected_id = tl.program_id(0).to(tl.int64)
    off = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    page_elems = HEADS * PAGE_SIZE * DIM
    valid_off = off < page_elems
    head = off // (PAGE_SIZE * DIM)
    token = (off // DIM) % PAGE_SIZE
    d = off % DIM
    scratch_page = tl.load(selected + selected_id).to(tl.int64)
    request = scratch_page // PAGES_PER_REQUEST
    logical_page = scratch_page % PAGES_PER_REQUEST
    position = logical_page * PAGE_SIZE + token
    prefix_len = tl.load(prefix + request).to(tl.int64)
    kv_len = tl.load(lens + request).to(tl.int64)
    if logical_page * PAGE_SIZE < prefix_len:
        source_page = tl.load(prefix_rows + scratch_page).to(tl.int64)
        source = source_page * (2 * page_elems) + off
        k = tl.load(prefix_main + source, valid_off, other=0.0).to(tl.bfloat16)
        v = tl.load(prefix_main + source + page_elems, valid_off, other=0.0).to(
            tl.bfloat16
        )
    else:
        valid = valid_off & (position >= prefix_len) & (position < kv_len)
        suffix_token = (
            tl.load(suffix_offsets + request).to(tl.int64) + position - prefix_len
        )
        packed_row = tl.load(unpad + suffix_token, valid, other=0).to(tl.int64)
        source = packed_row * (2 * NK + NI) + head * DIM + d
        k = tl.load(packed + source, valid, other=0.0)
        v = tl.load(packed + source + NK, valid, other=0.0)
    # Page zero is a dedicated sentinel. Each selected page is fully written,
    # including zeros for the invalid suffix tail, on every invocation.
    target = (selected_id + 1) * page_elems + off
    tl.store(out_k + target, k, valid_off)
    tl.store(out_v + target, v, valid_off)


def materialize_pages(
    selected,
    prefix_main,
    packed,
    unpad,
    metadata,
    scratch_seq_len,
    page_size,
    heads,
    dim,
    idx_dim,
):
    if packed.dtype != torch.bfloat16:
        raise ValueError(
            "compact suffix must come from original BF16 packed activations"
        )
    if prefix_main.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("compact prefix storage must be BF16 or E4M3")
    if packed.ndim != 2 or packed.shape[1] != 2 * heads * dim + idx_dim:
        raise ValueError("packed suffix must have rows [K, V, idx-K]")
    if tuple(prefix_main.shape[1:]) != (2, heads, page_size, dim):
        raise ValueError("prefix source must use [page, 2, head, token, dim] layout")
    if not packed.is_contiguous() or not prefix_main.is_contiguous():
        raise ValueError("compact sources must be contiguous")
    prefix, lens, offsets, rows = metadata
    shape = (selected.numel() + 1, heads, page_size, dim)
    k = torch.empty(shape, dtype=torch.bfloat16, device=packed.device)
    v = torch.empty_like(k)
    k[0].zero_()
    v[0].zero_()
    if selected.numel():
        grid = (selected.numel(), triton.cdiv(heads * page_size * dim, 1024))
        _materialize_pages_kernel[grid](
            selected,
            prefix_main,
            packed,
            unpad,
            prefix,
            lens,
            offsets,
            rows,
            k,
            v,
            PAGES_PER_REQUEST=scratch_seq_len // page_size,
            PAGE_SIZE=page_size,
            HEADS=heads,
            DIM=dim,
            NK=heads * dim,
            NI=idx_dim,
            BLOCK=1024,
            num_warps=4,
        )
    return k, v


def _compact_tensor_key(tensor):
    # These tensors are created once per forward and remain immutable while the
    # sparse layers share their plan. Inference tensors do not expose a version
    # counter, but exact object identity plus a retained strong reference is
    # sufficient for this per-forward cache. Normal tensors keep mutation-based
    # invalidation through their version counter.
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return (
        id(tensor),
        version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


def _get_compact_geometry(
    plan,
    index_kv_indices,
    topk,
    page_size,
    chunk_size,
    num_q_heads,
    dim,
    partial_dtype,
    device,
):
    """Cache immutable geometry on this forward's plan, never selected pages.

    Retained tensor references plus object/shape checks detect replacements;
    version counters additionally detect in-place mutation for normal tensors.
    Inference tensors rely on the per-forward immutability contract. A copied
    plan cannot inherit the original cache. This cache is independent of the
    legacy ``_chunk_meta`` namespace.
    """
    from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
        _build_chunk_meta,
        _pack_segments_into_chunks,
    )

    tensors = (
        index_kv_indices,
        plan["qo_segment_lens"],
        plan["seqused_k"],
        plan["kv_segment_lens"],
    )
    tensor_keys = tuple(_compact_tensor_key(tensor) for tensor in tensors)
    key = (
        tensor_keys,
        topk,
        page_size,
        chunk_size,
        num_q_heads,
        dim,
        partial_dtype,
        torch.device(device),
        int(plan["num_kv_heads"]),
    )
    cached = plan.get("_compact_geometry")
    if cached is not None and cached["owner_id"] == id(plan) and cached["key"] == key:
        return cached["meta"], cached["boundaries"]

    meta = _build_chunk_meta(
        plan,
        index_kv_indices,
        topk,
        page_size,
        chunk_size,
        num_q_heads,
        dim,
        partial_dtype,
        device,
    )
    groups = _pack_segments_into_chunks(plan["qo_segment_lens"].tolist(), chunk_size)
    boundaries = []
    for group in groups:
        query_offsets = [0]
        for _, lo, hi in group:
            query_offsets.append(query_offsets[-1] + hi - lo)
        boundaries.append(query_offsets)
    meta["query_segments"] = [
        _query_segment_ids(offsets, device) for offsets in boundaries
    ]
    plan["_compact_geometry"] = {
        "owner_id": id(plan),
        "key": key,
        "tensors": tensors,
        "meta": meta,
        "boundaries": boundaries,
    }
    return meta, boundaries


@torch.no_grad()
def run_compact_attention(
    q,
    topk_idx,
    index_kv_indices,
    plan,
    prefix_main,
    packed,
    unpad,
    metadata,
    scratch_seq_len,
    page_size,
    heads,
    dim,
    idx_dim,
    topk,
):
    from rtp_llm.models_py.triton_kernels.sparse_msa.prefill.topk_bt_fused import (
        _get_or_create_chunk_ws,
        _sparse_attn_chunk_size,
        run_sparse_attn_chunk,
    )

    if q.dtype != torch.bfloat16:
        raise ValueError("compact CP prefill requires BF16 queries")
    partial_dtype = plan.get("partial_dtype", torch.bfloat16)
    chunk_size = _sparse_attn_chunk_size()
    meta, query_boundaries = _get_compact_geometry(
        plan,
        index_kv_indices,
        topk,
        page_size,
        chunk_size,
        q.shape[1],
        dim,
        partial_dtype,
        q.device,
    )
    fwd_bytes = meta["ws_fwd_bytes"]
    ws = _get_or_create_chunk_ws(fwd_bytes + meta["ws_csr_words"] * 4, q.device)
    ws_fwd, ws_csr = ws[:fwd_bytes], ws[fwd_bytes:].view(torch.int32)
    output = torch.empty_like(q)
    page_capacity = (scratch_seq_len // page_size) * metadata[0].numel()
    for chunk, boundaries, query_segments in zip(
        meta["chunks"], query_boundaries, meta["query_segments"]
    ):
        start, end = chunk["g0"], chunk["g1"]
        chosen = topk_idx[:, start:end].contiguous()
        selected, table = selected_page_map(
            chosen,
            chunk["pt"],
            boundaries,
            page_capacity=page_capacity,
            query_segments=query_segments,
        )
        k, v = materialize_pages(
            selected,
            prefix_main,
            packed,
            unpad,
            metadata,
            scratch_seq_len,
            page_size,
            heads,
            dim,
            idx_dim,
        )
        run_sparse_attn_chunk(
            q[start:end],
            k,
            v,
            chosen,
            chunk,
            table,
            builder=meta["builder"],
            topk=topk,
            block_size_k=page_size,
            sm_scale=dim**-0.5,
            causal=plan["causal"],
            partial_dtype=partial_dtype,
            usable_sm=int(plan.get("usable_SM_count", -1)),
            ws_csr=ws_csr,
            ws_fwd=ws_fwd,
            out=output[start:end],
        )
        del k, v, selected, table
    return output
