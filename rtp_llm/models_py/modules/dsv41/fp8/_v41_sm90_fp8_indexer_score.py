"""SM90 DeepGEMM FP8 V4.1 indexer logits from the primary FP8 K cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._indexer_q_quant_triton import indexer_q_rope_fp8_quant_fold_indexed
from ._indexer_score import _get_num_sms


@dataclass(frozen=True)
class PagedFP8Scores:
    logits: torch.Tensor
    lengths: torch.Tensor
    query_len: int
    padded_query_len: int

    def compact_selected(self, selected: torch.Tensor) -> torch.Tensor:
        batch = selected.shape[0] // self.padded_query_len
        return selected.view(batch, self.padded_query_len, -1)[:, :self.query_len].reshape(
            batch * self.query_len, -1
        ).contiguous()


def score_paged_fp8_indexer(
    q: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    pages: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    max_ctx_len: int,
    rope_head_dim: int,
) -> PagedFP8Scores:
    """Return native FP32 logits and lengths for arbitrary speculative S.

    S>1 uses two-query K reuse per request, padding one query for odd S. The
    consumer selects on padded logits and compacts only the small TopK IDs.
    All operations are safe during CUDA graph capture.
    """
    import deep_gemm

    b, s, h, d = q.shape
    if s < 1 or h != 32 or d != 128:
        raise ValueError("unsupported V4.1 FP8 Q shape")
    if pages.dtype != torch.uint8 or pages.shape[1:] != (64, 132):
        raise ValueError("DeepGEMM FP8 indexer pages must be uint8 [P,64,132]")
    if block_table.dtype != torch.int32 or block_table.shape[0] != b:
        raise ValueError("DeepGEMM indexer table must be int32 [B,P]")
    if max_ctx_len > 64 * block_table.shape[1]:
        raise ValueError("FP8 indexer context exceeds the page table")
    lengths = lengths.to(torch.int32).contiguous().clamp(0, max_ctx_len)
    q_fp8, w_fold = indexer_q_rope_fp8_quant_fold_indexed(
        q, weights, freqs_cis, positions, rope_head_dim
    )
    next_n = 1 if s == 1 else 2
    groups = (s + next_n - 1) // next_n
    padded_s = groups * next_n
    if padded_s != s:
        q_padded = torch.zeros((b, padded_s, h, d), dtype=q_fp8.dtype, device=q.device)
        w_padded = torch.zeros((b, padded_s, h), dtype=w_fold.dtype, device=q.device)
        lengths_padded = torch.zeros((b, padded_s), dtype=torch.int32, device=q.device)
        q_padded[:, :s] = q_fp8
        w_padded[:, :s] = w_fold
        lengths_padded[:, :s] = lengths
    else:
        q_padded = q_fp8
        w_padded = w_fold
        lengths_padded = lengths
    rows = b * groups
    q_dg = q_padded.view(rows, next_n, h, d)
    w_dg = w_padded.view(b * padded_s, h)
    actual_lengths = lengths_padded.contiguous().view(-1)
    schedule_lengths = lengths_padded.clone()
    if next_n == 2:
        # SM90 metadata schedules each Q2 atom from its last query's length.
        # Cover the first query even for odd tails or nonmonotonic lengths.
        # TopK below retains the original per-query visible lengths.
        pairs = schedule_lengths.view(b, groups, 2)
        pairs[:, :, 1] = torch.maximum(pairs[:, :, 0], pairs[:, :, 1])
    context_lens = schedule_lengths.view(rows, next_n)
    table = block_table.repeat_interleave(groups, dim=0).contiguous() if groups > 1 else block_table.contiguous()
    metadata = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, 64, _get_num_sms(q.device)
    )
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_dg, None), pages.unsqueeze(2), w_dg, context_lens,
        table, metadata, max_ctx_len,
    )
    return PagedFP8Scores(logits, actual_lengths, s, padded_s)
