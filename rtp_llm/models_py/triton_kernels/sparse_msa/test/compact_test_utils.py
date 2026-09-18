"""Test-only reference helpers for compact sparse-prefill pages."""

import torch


def compact_bf16_pages_for_topk(k_pages, v_pages, topk_idx, page_table, cu_q):
    """Deduplicate selected BF16 pages into a sentinel-prefixed test pool."""
    if k_pages.dtype != torch.bfloat16 or v_pages.dtype != torch.bfloat16:
        raise ValueError("compact oracle requires BF16 K/V pages")
    if k_pages.shape != v_pages.shape or k_pages.ndim != 4:
        raise ValueError("K/V must have matching HND page shapes")
    if any(t.device != k_pages.device for t in (v_pages, topk_idx, page_table, cu_q)):
        raise ValueError("compact inputs must be on the same device")
    integer_dtypes = (torch.int32, torch.int64)
    if any(t.dtype not in integer_dtypes for t in (topk_idx, page_table, cu_q)):
        raise ValueError(
            "topk, page table and query boundaries must be integer tensors"
        )
    if (
        topk_idx.ndim != 3
        or topk_idx.shape[0] != k_pages.shape[1]
        or topk_idx.shape[2] == 0
    ):
        raise ValueError("topk must have shape [KV heads, queries, positive topk]")
    if page_table.ndim != 2 or page_table.shape[0] == 0:
        raise ValueError("page table must be rank 2 with a nonempty batch")
    if cu_q.ndim != 1:
        raise ValueError("query boundaries must be rank 1")
    boundaries = cu_q.tolist()
    if (
        len(boundaries) != page_table.shape[0] + 1
        or boundaries[0] != 0
        or boundaries[-1] != topk_idx.shape[1]
        or any(lo > hi for lo, hi in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError(
            "query boundaries must start at 0, be monotonic, and match batch/query rows"
        )

    selected = []
    for row, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
        logical = topk_idx[:, lo:hi].reshape(-1).long()
        if bool((logical < -1).any()):
            raise ValueError("topk padding must be -1")
        logical = logical[logical >= 0].unique(sorted=True)
        if bool((logical >= page_table.shape[1]).any()):
            raise ValueError("selected logical page is outside page table")
        physical = page_table[row, logical].long()
        if bool(((physical < 0) | (physical >= k_pages.shape[0])).any()):
            raise ValueError("selected physical page is outside input pages")
        selected.append((row, logical, physical))

    ids = torch.cat([entry[2] for entry in selected]).unique(sorted=True)
    width = page_table.shape[1]
    storage = torch.zeros(
        (page_table.shape[0], ((width + 3) // 4) * 4),
        dtype=torch.int32,
        device=page_table.device,
    )
    compact_map = storage[:, :width]
    for row, logical, physical in selected:
        compact_map[row, logical] = (
            torch.searchsorted(ids, physical).to(torch.int32) + 1
        )
    zero = k_pages.new_zeros((1, *k_pages.shape[1:]))
    compact_k = torch.cat((zero, k_pages.index_select(0, ids)))
    compact_v = torch.cat((zero, v_pages.index_select(0, ids)))
    return compact_k, compact_v, compact_map, ids


def compact_sparse_prefill_reference(
    module,
    q,
    k_pages,
    v_pages,
    topk_idx,
    page_map,
    plan,
    topk,
    block_size,
    sm_scale,
    chunk_size,
):
    """Run production step3 against test-only compact page materialization."""
    partial_dtype = plan.get("partial_dtype", torch.bfloat16)
    meta = module._build_chunk_meta(
        plan,
        page_map,
        topk,
        block_size,
        chunk_size,
        q.shape[1],
        q.shape[2],
        partial_dtype,
        q.device,
    )
    fwd_bytes, csr_words = meta["ws_fwd_bytes"], meta["ws_csr_words"]
    workspace = module._get_or_create_chunk_ws(fwd_bytes + csr_words * 4, q.device)
    ws_fwd = workspace[:fwd_bytes]
    ws_csr = workspace[fwd_bytes : fwd_bytes + csr_words * 4].view(torch.int32)
    output = torch.empty_like(q)
    for chunk in meta["chunks"]:
        start, end = chunk["g0"], chunk["g1"]
        chosen = topk_idx[:, start:end].contiguous()
        compact_k, compact_v, compact_map, _ = compact_bf16_pages_for_topk(
            k_pages, v_pages, chosen, chunk["pt"], chunk["cu_q"]
        )
        module.run_sparse_attn_chunk(
            q[start:end],
            compact_k,
            compact_v,
            chosen,
            chunk,
            compact_map,
            builder=meta["builder"],
            topk=topk,
            block_size_k=block_size,
            sm_scale=sm_scale,
            causal=plan["causal"],
            partial_dtype=partial_dtype,
            usable_sm=int(plan.get("usable_SM_count", -1)),
            ws_csr=ws_csr,
            ws_fwd=ws_fwd,
            out=output[start:end],
        )
    return output
