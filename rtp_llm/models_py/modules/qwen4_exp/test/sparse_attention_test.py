"""GPU test: sparse attention vs an independent selected-mask reference.

Verifies that gathering the selected K/V entries matches dense attention after
masking it to exactly the same selected token set.  Also tests -1 padding and
ragged K per row.
"""

import math
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.sparse_attention import gather_and_attend

_H_Q, _H_KV, _D = 24, 2, 256  # qwen4 full-attention production geometry
_DEV = "cuda"


def _selected_mask_reference(q, k, v, selected):
    """Independent dense GQA reference masked to unique selected positions."""
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    T = k.shape[2]
    q_per_kv = H_q // H_kv
    selected_count = torch.zeros(B, S, T, dtype=torch.int32, device=q.device)
    valid = selected >= 0
    selected_count.scatter_add_(2, selected.clamp(min=0).long(), valid.to(torch.int32))
    selected_mask = selected_count > 0

    o = torch.zeros_like(q)
    for h_q in range(H_q):
        h_kv = h_q // q_per_kv
        k_h = k[:, h_kv].unsqueeze(1)  # [B, 1, T, D]
        v_h = v[:, h_kv].unsqueeze(1)
        q_h = q[:, h_q].unsqueeze(1)  # [B, 1, S, D]
        scores = (q_h @ k_h.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        scores = scores.masked_fill(~selected_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.where(
            selected_mask.any(dim=-1, keepdim=True).unsqueeze(1), attn, 0.0
        )
        o[:, h_q] = (attn @ v_h)[:, 0]
    return o


class SparseAttentionTest(unittest.TestCase):
    def test_sparse_matches_dense_on_the_selected_set(self):
        torch.manual_seed(0)
        B, S, T = 2, 8, 64
        K = 16
        q = (torch.randn(B, _H_Q, S, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        k = (torch.randn(B, _H_KV, T, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        v = (torch.randn(B, _H_KV, T, _D, device=_DEV) * 0.3).to(torch.bfloat16)

        causal = torch.triu(
            torch.ones(S, T, dtype=torch.bool, device=_DEV), diagonal=1
        ).logical_not_()
        # Select K random valid positions for each (b,s)
        sel = torch.zeros(B, S, K, dtype=torch.int32, device=_DEV)
        for b in range(B):
            for s in range(S):
                valid = causal[s].nonzero().squeeze(-1)
                n = min(K, len(valid))
                picks = valid[torch.randperm(len(valid), device=_DEV)[:n]]
                sel[b, s, :n] = picks.to(torch.int32)
                sel[b, s, n:] = -1

        sparse = gather_and_attend(q, k, v, sel)
        reference = _selected_mask_reference(q, k, v, sel)

        torch.testing.assert_close(sparse, reference, atol=1e-2, rtol=1e-2)

    def test_masked_positions_produce_no_contribution(self):
        """-1 selected indices are properly masked: their output is 0."""
        B, S, K = 1, 2, 4
        q = torch.randn(B, _H_Q, S, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(B, _H_KV, S + 1, _D, device=_DEV).to(torch.bfloat16)
        v = torch.randn(B, _H_KV, S + 1, _D, device=_DEV).to(torch.bfloat16)
        sel = torch.zeros(B, S, K, dtype=torch.int32, device=_DEV)
        sel.fill_(-1)  # all -1 — no attended tokens
        out = gather_and_attend(q, k, v, sel)
        self.assertTrue((out == 0).all())  # softmax(empty) -> 0 weight


if __name__ == "__main__":
    unittest.main()
