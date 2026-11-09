"""MiniMax-M3.1 Q8KV4 decode using separate indexer and main KV page tables."""

import torch
import triton

from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_cache import (
    M31CacheViews,
)
from rtp_llm.models_py.modules.minimax_m31.msa.common.minimax_m31_index_score import (
    q8kv4_index_score,
)
from rtp_llm.models_py.modules.minimax_m31.msa.common.minimax_m31_topk import m31_topk16
from rtp_llm.models_py.modules.minimax_m31.msa.decode.minimax_m31_sparse_attention import (
    _q8kv4_sparse_decode_attention_mxfp8,
    q8kv4_sparse_decode_attention,
)


def m31_sparse_decode(
    q8: torch.Tensor,
    index_q8: torch.Tensor,
    views: M31CacheViews,
    main_table: torch.Tensor,
    index_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    init_blocks: int,
    local_blocks: int,
    output_mxfp8: bool = False,
    prefill_scores: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
    query_cu_seqlens: torch.Tensor | None = None,
    max_query_width: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Select logical pages through indexer storage, then read main KV storage."""
    batch = q8.shape[0]
    if tuple(q8.shape) != (batch, 64, 128):
        raise ValueError("M3.1 main Q must be [batch,64,128]")
    if tuple(index_q8.shape) != (batch, 4, 128):
        raise ValueError("M3.1 index Q must be [batch,4,128]")
    if q8.dtype != torch.float8_e4m3fn or index_q8.dtype != q8.dtype:
        raise ValueError("M3.1 sparse decode requires E4M3 queries")
    if main_table.ndim != 2 or index_table.shape != main_table.shape:
        raise ValueError("M3.1 main/indexer logical table shapes must match")
    if main_table.shape[0] != batch:
        raise ValueError("M3.1 table batch does not match query rows")
    if main_table.dtype != torch.int32 or index_table.dtype != torch.int32:
        raise ValueError("M3.1 page tables must use int32")
    if seq_lens.shape != (batch,) or seq_lens.dtype != torch.int32:
        raise ValueError("M3.1 sequence lengths must be int32 [batch]")
    if any(
        t.device != q8.device
        for t in (
            index_q8,
            main_table,
            index_table,
            seq_lens,
            views.main_k,
            views.index_k,
        )
    ):
        raise ValueError("M3.1 sparse decode tensors must share a device")
    if not q8.is_cuda:
        raise ValueError("M3.1 sparse decode requires CUDA")
    if batch == 0:
        if output_mxfp8:
            raise ValueError("MXFP8 combine requires at least one query row")
        return torch.empty((0, 64, 128), dtype=torch.bfloat16, device=q8.device)

    score_shape = (4, batch, index_table.shape[1])
    if prefill_scores is not None and (
        tuple(prefill_scores.shape) != score_shape
        or prefill_scores.dtype != torch.float32
        or prefill_scores.device != q8.device
    ):
        raise ValueError("M3.1 Prefill scores must match index table geometry")
    scores = (
        prefill_scores
        if prefill_scores is not None
        else torch.empty(score_shape, dtype=torch.float32, device=q8.device)
    )
    topk = torch.empty((4, batch, 16), dtype=torch.int32, device=q8.device)
    if prefill_scores is None:
        q8kv4_index_score(
            index_q8,
            views.index_k,
            views.index_k_scale,
            index_table,
            seq_lens,
            scores,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            sm_scale=128**-0.5,
            mma_scale_layout=True,
            query_cu_seqlens=query_cu_seqlens,
            max_query_width=max_query_width,
        )
    m31_topk16(scores, seq_lens, topk)
    if output_mxfp8:
        aligned_rows = (batch + 3) // 4 * 4
        out8 = torch.empty(
            (batch, 64, 128), dtype=torch.float8_e4m3fn, device=q8.device
        )
        scales = torch.empty_strided(
            (batch, 64), (1, aligned_rows), dtype=torch.int32, device=q8.device
        )
    partial = torch.empty(
        (batch, 4, 16, 16, 128), dtype=torch.bfloat16, device=q8.device
    )
    lse = torch.empty((batch, 4, 16, 16), dtype=torch.float32, device=q8.device)
    counts = torch.empty((batch, 4), dtype=torch.int32, device=q8.device)
    if output_mxfp8:
        _q8kv4_sparse_decode_attention_mxfp8(
            q8,
            views.main_k,
            views.main_v,
            views.main_k_scale,
            views.main_v_scale,
            main_table,
            topk,
            seq_lens,
            sm_scale=128**-0.5,
            out=out8,
            output_scales=scales,
            partial_out=partial,
            partial_lse=lse,
            counts=counts,
            mma_scale_layout=True,
            valid_token_mask=valid_token_mask,
        )
        return out8, scales
    out = torch.empty((batch, 64, 128), dtype=torch.bfloat16, device=q8.device)
    return q8kv4_sparse_decode_attention(
        q8,
        views.main_k,
        views.main_v,
        views.main_k_scale,
        views.main_v_scale,
        main_table,
        topk,
        seq_lens,
        sm_scale=128**-0.5,
        out=out,
        partial_out=partial,
        partial_lse=lse,
        counts=counts,
        mma_scale_layout=True,
        valid_token_mask=valid_token_mask,
    )
