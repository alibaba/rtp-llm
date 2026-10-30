"""Model-owned native sparse Prefill over the current tagged NVFP4 cache."""

import torch

from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_cache import (
    M31CacheViews,
)
from rtp_llm.models_py.modules.minimax_m31.msa.common.minimax_m31_index_score import (
    q8kv4_cp_prefill_index_score,
    q8kv4_index_score,
)
from rtp_llm.models_py.modules.minimax_m31.msa.common.minimax_m31_topk import m31_topk16


def stage_m31_main_pages(
    views: M31CacheViews,
    table: torch.Tensor,
    sequence_lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage only active K/V pages into one shared physical map for FMHA.

    The main group owns one block table for its four planes. Interleaved
    per-page storage cannot be passed to FMHA as contiguous K/V; index_select
    makes one compact per-forward copy of the active pages.
    """
    if table.ndim != 2 or table.dtype != torch.int32:
        raise ValueError("MiniMax-M3.1 main page table must be int32 [batch,blocks]")
    if len(sequence_lengths) != table.shape[0]:
        raise ValueError("MiniMax-M3.1 Prefill request count mismatch")
    page_counts = [(length + 127) // 128 for length in sequence_lengths]
    if (
        any(length <= 0 for length in sequence_lengths)
        or max(page_counts, default=0) > table.shape[1]
    ):
        raise ValueError("MiniMax-M3.1 Prefill page table is too short")
    active_ids = torch.cat(
        [table[row, :count] for row, count in enumerate(page_counts)]
    ).to(torch.int64)
    if not table.is_cuda and torch.any(active_ids < 0):
        raise ValueError("MiniMax-M3.1 Prefill has an unallocated cache page")
    # FMHA accepts one page table for both K and V. Keep each request's
    # logical page number but remap it to compact, contiguous staging pages.
    compact_table = torch.zeros_like(table)
    offset = 0
    for row, count in enumerate(page_counts):
        compact_table[row, :count] = torch.arange(
            offset, offset + count, device=table.device, dtype=torch.int32
        )
        offset += count

    def copy_plane(plane: torch.Tensor, tail: tuple[int, ...]) -> torch.Tensor:
        return plane.index_select(0, active_ids).reshape(len(active_ids), *tail)

    k = copy_plane(views.main_k, (4, 128, 64))
    v = copy_plane(views.main_v, (4, 128, 64))
    k_scale = copy_plane(views.main_k_scale.view(torch.uint8), (4 * 128, 8)).reshape(
        -1, 8
    )
    v_scale = copy_plane(views.main_v_scale.view(torch.uint8), (4 * 128, 8)).reshape(
        -1, 8
    )
    return k, v, k_scale, v_scale, compact_table


def m31_sparse_prefill_fmha(
    q8: torch.Tensor,
    index_q8: torch.Tensor,
    views: M31CacheViews,
    main_table: torch.Tensor,
    index_table: torch.Tensor,
    request_ids: torch.Tensor,
    positions: torch.Tensor,
    input_lengths: list[int],
    prefix_lengths: list[int],
    *,
    init_blocks: int,
    local_blocks: int,
    query_segments: list[tuple[int, int, int]] | None = None,
) -> torch.Tensor:
    """Use existing Q8KV4 score/TopK and native FMHA on compact active pages."""
    from fmha_sm100.cute.interface import sparse_atten_nvfp4_kv_func
    from fmha_sm100.cute.src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100

    cp_score = query_segments is not None
    chunk_rows = 4096 if cp_score else 512
    full_lengths = [p + n for p, n in zip(prefix_lengths, input_lengths)]
    if query_segments is None:
        query_segments = [
            (request, 0, length) for request, length in enumerate(input_lengths)
        ]
    k, v, k_scale, v_scale, compact_table = stage_m31_main_pages(
        views, main_table, full_lengths
    )
    builder = SparseK2qCsrBuilderSm100()
    output = torch.empty(q8.shape, dtype=torch.bfloat16, device=q8.device)
    row_start = 0
    # Bound FMHA's TopK partial-output allocation. Each chunk may contain
    # several requests; its KV length ends at that chunk's final query row.
    while row_start < q8.shape[0]:
        row_end = min(row_start + chunk_rows, q8.shape[0])
        chunk_ids = request_ids[row_start:row_end]
        chunk_lens = (positions[row_start:row_end] + 1).to(torch.int32)
        scores = torch.empty(
            (4, row_end - row_start, index_table.shape[1]),
            dtype=torch.float32,
            device=q8.device,
        )
        topk = torch.empty(
            (4, row_end - row_start, 16), dtype=torch.int32, device=q8.device
        )
        if not cp_score:
            q8kv4_index_score(
                index_q8[row_start:row_end],
                views.index_k,
                views.index_k_scale,
                index_table.index_select(0, chunk_ids),
                chunk_lens,
                scores,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                sm_scale=128**-0.5,
                mma_scale_layout=True,
            )
        else:
            q8kv4_cp_prefill_index_score(
                index_q8[row_start:row_end],
                views.index_k,
                views.index_k_scale,
                index_table,
                chunk_ids,
                positions[row_start:row_end],
                scores,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
            )
        m31_topk16(scores, chunk_lens, topk)

        # Request boundaries come from the host Prefill lengths. Splitting a
        # request preserves causal offsets through k_len - q_len.
        pieces = []
        full_row = 0
        for request, begin, length in query_segments:
            lo = max(row_start, full_row)
            hi = min(row_end, full_row + length)
            if lo < hi:
                pieces.append((request, begin + hi - full_row, hi - lo))
            full_row += length
        cu_q = [0]
        cu_k = [0]
        max_k = 0
        for request, query_end, query_count in pieces:
            k_len = prefix_lengths[request] + query_end
            cu_q.append(cu_q[-1] + query_count)
            cu_k.append(cu_k[-1] + k_len)
            max_k = max(max_k, k_len)
        cu_q_d = torch.tensor(cu_q, dtype=torch.int32, device=q8.device)
        cu_k_d = torch.tensor(cu_k, dtype=torch.int32, device=q8.device)
        selected_table = compact_table.index_select(
            0,
            torch.tensor(
                [piece[0] for piece in pieces], dtype=torch.int64, device=q8.device
            ),
        )
        rows = sum((cu_k[i + 1] - cu_k[i] + 127) // 128 for i in range(len(pieces)))
        row_ptr, q_indices = builder(
            topk.contiguous(),
            cu_q_d,
            cu_k_d,
            total_k=cu_k[-1],
            blk_kv=128,
            max_seqlen_k=max_k,
            max_seqlen_q=max(piece[2] for piece in pieces),
            total_rows=rows,
            qhead_per_kv=16,
        )[:2]
        output[row_start:row_end] = sparse_atten_nvfp4_kv_func(
            q8[row_start:row_end],
            k,
            v,
            k_scale,
            v_scale,
            None,
            None,
            row_ptr,
            q_indices,
            16,
            cu_seqlens_q=cu_q_d,
            cu_seqlens_k=cu_k_d,
            max_seqlen_q=max(piece[2] for piece in pieces),
            max_seqlen_k=max_k,
            blk_kv=128,
            causal=True,
            softmax_scale=128**-0.5,
            page_table=selected_table,
            seqused_k=torch.tensor(
                [cu_k[i + 1] - cu_k[i] for i in range(len(pieces))],
                dtype=torch.int32,
                device=q8.device,
            ),
        )
        row_start = row_end
    return output
