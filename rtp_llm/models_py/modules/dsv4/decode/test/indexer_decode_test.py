"""Unit tests for ``IndexerDecodeV4Op``.

Validates the reference path bit-exactly against a pure-Python einsum
oracle and (when ``deep_gemm.fp8_paged_mqa_logits`` is available) checks
the FP8 fast path's topk against the reference via topk-IoU.

Note: the fast-path FP8 paged layout integration is currently scaffolded
but not numerically verified end-to-end; its IoU test is marked
``@unittest.skip`` until the model integration site lands. See the
top-of-file comment in ``indexer_decode_op.py`` for context.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.decode.indexer_decode_op import (
    IndexerDecodeV4Op,
    _fast_path_available,
)


def _oracle_topk(
    q: torch.Tensor,  # [B, q_len, H, D] bf16
    kv: torch.Tensor,  # [B, T_max, D]   bf16
    weights: torch.Tensor,  # [B, q_len, H]   bf16 (already scaled)
    cmp_lens: torch.Tensor,  # [B] int32
    K: int,
) -> torch.Tensor:
    """Pure-Python einsum-then-topk oracle. Returns int32 [B, q_len, K]
    with -1 padding for invalid positions."""
    B, q_len, H, D = q.shape
    T_max = kv.shape[1]
    device = q.device
    score = torch.einsum("bshd,btd->bsht", q.float(), kv.float())
    score = (score.relu_() * weights.float().unsqueeze(-1)).sum(
        dim=2
    )  # [B, q_len, T_max]
    t_range = torch.arange(T_max, device=device, dtype=torch.int32)
    valid = t_range.view(1, 1, T_max) < cmp_lens.view(B, 1, 1).to(torch.int32)
    score = torch.where(valid, score, torch.full_like(score, float("-inf")))
    k_eff = min(K, T_max)
    idxs = score.topk(k_eff, dim=-1).indices.to(torch.int32)
    if k_eff < K:
        pad = torch.full((B, q_len, K - k_eff), -1, dtype=torch.int32, device=device)
        idxs = torch.cat([idxs, pad], dim=-1)
    cmp_len_b = cmp_lens.view(B, 1, 1).to(torch.int32)
    idxs = torch.where(idxs < cmp_len_b, idxs, torch.full_like(idxs, -1))
    return idxs


def _topk_iou(
    a: torch.Tensor,  # [B, q_len, K] int32 (-1 = invalid)
    b: torch.Tensor,  # same shape
) -> float:
    """Per-(B, q_len) row IoU between two topk index sets, averaged."""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    B, S, K = a.shape
    a_list = a.view(-1, K).cpu().tolist()
    b_list = b.view(-1, K).cpu().tolist()
    ious = []
    for ai, bi in zip(a_list, b_list):
        sa = {x for x in ai if x >= 0}
        sb = {x for x in bi if x >= 0}
        if not sa and not sb:
            ious.append(1.0)
            continue
        inter = len(sa & sb)
        union = len(sa | sb)
        ious.append(inter / union if union else 1.0)
    return sum(ious) / max(len(ious), 1)


class TestIndexerDecodeOp(unittest.TestCase):
    # Common shapes — V4 uses H_idx=64, D_idx=128, K=index_topk (here 32 to
    # exercise both topk dim and length-masking edge cases without making
    # the oracle slow).
    B = 4
    Q_LEN = 1
    H_IDX = 64
    D_IDX = 128
    T_COMPRESSED = 64
    K = 32

    def setUp(self) -> None:
        torch.manual_seed(0)

    # ----------------------------------------------------------------
    # 1) Reference path == pure-Python oracle (exact match expected)
    # ----------------------------------------------------------------
    def test_reference_path_matches_indexer_einsum(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        B, Q, H, D = self.B, self.Q_LEN, self.H_IDX, self.D_IDX
        T = self.T_COMPRESSED

        q = (torch.randn(B, Q, H, D, device=device) * 0.1).to(torch.bfloat16)
        kv = (torch.randn(B, T, D, device=device) * 0.1).to(torch.bfloat16)
        weights = (torch.randn(B, Q, H, device=device) * 0.5).to(torch.bfloat16)
        # All requests use the full T as their valid compressed length.
        cmp_lens = torch.full((B,), T, dtype=torch.int32, device=device)

        op = IndexerDecodeV4Op(
            index_n_heads=H,
            index_head_dim=D,
            index_topk=self.K,
            softmax_scale=1.0,
        )
        out_buffer = torch.full((B, Q, self.K), -1, dtype=torch.int32, device=device)
        result = op.forward(
            q,
            kv,
            weights,
            cmp_lens,
            out_buffer,
            force_reference=True,
        )
        self.assertIs(result, out_buffer)

        oracle = _oracle_topk(q, kv, weights, cmp_lens, self.K)

        # topk indices may differ in ORDER (topk is unsorted). Compare as sets per row.
        # For the reference path we expect identical set membership.
        for b in range(B):
            for s in range(Q):
                got = sorted(int(x) for x in result[b, s].tolist() if int(x) >= 0)
                want = sorted(int(x) for x in oracle[b, s].tolist() if int(x) >= 0)
                self.assertEqual(got, want, f"mismatch at b={b}, s={s}")

    # ----------------------------------------------------------------
    # 2) Fast path topk-IoU vs reference  -- WIP: paged FP8 layout
    # ----------------------------------------------------------------
    @unittest.skip(
        "fast-path WIP: paged FP8 (data || scale) packed layout "
        "needs end-to-end validation against deep_gemm. Reference "
        "path is the source of truth for now; integration into "
        "deepseek_v4_model.py will toggle the env flag once the "
        "fast-path layout is verified."
    )
    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    @unittest.skipUnless(_fast_path_available(), "no deep_gemm fp8_paged_mqa_logits")
    def test_fast_path_topk_iou(self):
        device = torch.device("cuda:0")
        B, Q, H, D = self.B, self.Q_LEN, self.H_IDX, self.D_IDX
        T = self.T_COMPRESSED

        q = (torch.randn(B, Q, H, D, device=device) * 0.1).to(torch.bfloat16)
        kv = (torch.randn(B, T, D, device=device) * 0.1).to(torch.bfloat16)
        weights = (torch.randn(B, Q, H, device=device) * 0.5).to(torch.bfloat16)
        cmp_lens = torch.full((B,), T, dtype=torch.int32, device=device)

        op = IndexerDecodeV4Op(
            index_n_heads=H,
            index_head_dim=D,
            index_topk=self.K,
            softmax_scale=1.0,
        )
        out_ref = torch.full((B, Q, self.K), -1, dtype=torch.int32, device=device)
        op.forward(q, kv, weights, cmp_lens, out_ref, force_reference=True)

        out_fast = torch.full((B, Q, self.K), -1, dtype=torch.int32, device=device)
        op.forward(q, kv, weights, cmp_lens, out_fast, force_reference=False)

        iou = _topk_iou(out_ref, out_fast)
        self.assertGreaterEqual(
            iou,
            0.95,
            f"fast-path topk-IoU={iou:.3f} below threshold 0.95",
        )

    # ----------------------------------------------------------------
    # 3) compressed_len_per_req < T_max masks invalid range
    # ----------------------------------------------------------------
    def test_short_compressed_len(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        B, Q, H, D = self.B, self.Q_LEN, self.H_IDX, self.D_IDX
        T = self.T_COMPRESSED  # 64
        valid_len = 10

        q = (torch.randn(B, Q, H, D, device=device) * 0.1).to(torch.bfloat16)
        kv = (torch.randn(B, T, D, device=device) * 0.1).to(torch.bfloat16)
        weights = (torch.randn(B, Q, H, device=device) * 0.5).to(torch.bfloat16)
        cmp_lens = torch.full((B,), valid_len, dtype=torch.int32, device=device)

        op = IndexerDecodeV4Op(
            index_n_heads=H,
            index_head_dim=D,
            index_topk=self.K,
            softmax_scale=1.0,
        )
        out_buffer = torch.full((B, Q, self.K), -1, dtype=torch.int32, device=device)
        op.forward(q, kv, weights, cmp_lens, out_buffer, force_reference=True)

        # Every non-(-1) entry must be < valid_len. Slots beyond valid_len
        # should be -1 (since K=32 > valid_len=10, the last 22 entries per row
        # must be -1).
        for b in range(B):
            for s in range(Q):
                row = out_buffer[b, s].tolist()
                non_neg = [x for x in row if x >= 0]
                self.assertEqual(
                    len(non_neg),
                    valid_len,
                    f"row b={b},s={s} has {len(non_neg)} valid entries; expected {valid_len}",
                )
                for x in non_neg:
                    self.assertLess(
                        x,
                        valid_len,
                        f"row b={b},s={s} has out-of-range index {x} >= {valid_len}",
                    )
                # Set membership = {0..valid_len-1}.
                self.assertEqual(
                    sorted(non_neg),
                    list(range(valid_len)),
                    f"row b={b},s={s} valid indices != [0..{valid_len-1}]",
                )

    # ----------------------------------------------------------------
    # 4) shape & dtype sanity
    # ----------------------------------------------------------------
    def test_returns_same_buffer_with_correct_shape_dtype(self):
        device = torch.device("cpu")
        B, Q, H, D = 2, 1, 4, 128
        T = 16
        K = 8
        q = torch.randn(B, Q, H, D, device=device).to(torch.bfloat16)
        kv = torch.randn(B, T, D, device=device).to(torch.bfloat16)
        weights = torch.randn(B, Q, H, device=device).to(torch.bfloat16)
        cmp_lens = torch.tensor([T, T // 2], dtype=torch.int32, device=device)
        op = IndexerDecodeV4Op(
            index_n_heads=H,
            index_head_dim=D,
            index_topk=K,
            softmax_scale=1.0,
        )
        out_buffer = torch.full((B, Q, K), -1, dtype=torch.int32, device=device)
        ret = op.forward(q, kv, weights, cmp_lens, out_buffer, force_reference=True)
        self.assertIs(ret, out_buffer)
        self.assertEqual(ret.shape, (B, Q, K))
        self.assertEqual(ret.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
