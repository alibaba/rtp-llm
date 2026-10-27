"""Test-only deterministic selection for exact offload on/off comparisons.

The native radix top-k emits IDs through atomics, so the same selected set can
arrive in different orders and change attention's floating-point reduction.
Keep native inference untouched; activate only in the model comparison servers.
"""

import os
import sys

if os.environ.get("RTP_SPARSE_TEST_STABLE_TOPK") == "1":
    import torch

    import rtp_llm.models_py.kernels.cuda.fast_topk as fast_topk

    selection_observed = False

    def select(score, lengths, topk, row_starts=None):
        global selection_observed
        if not selection_observed:
            print("SPARSE_OFFLOAD_TEST_STABLE_TOPK_SELECT", file=sys.stderr, flush=True)
            selection_observed = True
        positions = torch.arange(score.shape[1], device=score.device)
        starts = torch.zeros_like(lengths) if row_starts is None else row_starts
        local = positions[None, :] - starts[:, None]
        valid = (local >= 0) & (local < lengths[:, None])
        ranked = torch.argsort(
            score.masked_fill(~valid, -torch.inf), descending=True, stable=True
        )[:, : min(topk, score.shape[1])]
        indices = ranked - starts[:, None]
        indices = torch.where(valid.gather(1, ranked), indices, 0x7FFFFFFF)
        if indices.shape[1] < topk:
            indices = torch.nn.functional.pad(
                indices, (0, topk - indices.shape[1]), value=0x7FFFFFFF
            )
        indices = indices.sort(dim=-1).values
        return torch.where(indices == 0x7FFFFFFF, -1, indices).to(torch.int32)

    def paged(score, lengths, cu_seqlens_q, topk, row_starts=None):
        return select(score, lengths, topk, row_starts)

    def ragged(score, lengths, topk_indices_offset, topk, row_starts=None):
        indices = select(score, lengths, topk, row_starts)
        if topk_indices_offset is not None:
            indices = torch.where(
                indices >= 0, indices + topk_indices_offset[:, None], indices
            )
        return indices

    fast_topk.fast_topk_transform_fused = paged
    fast_topk.fast_topk_transform_ragged_fused = ragged

    print("SPARSE_OFFLOAD_TEST_STABLE_TOPK", file=sys.stderr, flush=True)
