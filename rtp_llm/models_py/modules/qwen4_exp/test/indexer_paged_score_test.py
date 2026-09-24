import math
import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score import (
    qsa_paged_indexer_score,
)

_H, _D = 4, 128
_DEV = "cuda"


def _ref(q, w, pool, bt, ctx, bs):
    M, H, D = q.shape[0], q.shape[1], q.shape[2]
    max_ctx = int(ctx.max())
    out = torch.full((M, max_ctx), float("-inf"), device=_DEV)
    for row in range(M):
        T = int(ctx.view(-1)[row])
        if T == 0:
            continue
        for t in range(T):
            bi = t // bs
            ei = t % bs
            if bi >= bt.shape[1]:
                continue
            pbt = int(bt[row // ctx.shape[1], bi])
            if pbt <= 0:
                continue
            k = pool[pbt * bs + ei].float()
            score = float(
                sum(
                    w[row, h].item()
                    * torch.nn.functional.relu(q[row, h].float() @ k).item()
                    for h in range(H)
                )
            )
            out[row, t] = score
    return out


class PagedScoreTest(unittest.TestCase):
    def test_negative_weight_is_applied_after_relu(self):
        # dot(q, k) is positive.  The expected negative contribution proves
        # the kernel implements weight * relu(dot), not relu(weight * dot).
        q = torch.ones(1, 1, 1, _D, dtype=torch.bfloat16, device=_DEV)
        w = torch.tensor([[-0.5]], dtype=torch.float32, device=_DEV)
        pool = torch.zeros(2, _D, dtype=torch.bfloat16, device=_DEV)
        pool[1].fill_(1)
        bt = torch.tensor([[1]], dtype=torch.int32, device=_DEV)
        ctx = torch.tensor([[1]], dtype=torch.int32, device=_DEV)

        got = qsa_paged_indexer_score(q, w, pool, bt, ctx, block_size=1, max_ctx_len=1)

        self.assertEqual(float(got[0, 0]), -0.5 * _D)

    def test_missing_and_unallocated_logical_blocks_are_negative_infinity(self):
        q = torch.ones(1, 1, 1, _D, dtype=torch.bfloat16, device=_DEV)
        w = torch.ones(1, 1, dtype=torch.float32, device=_DEV)
        pool = torch.ones(2, _D, dtype=torch.bfloat16, device=_DEV)
        # Logical block zero is unallocated; the context also extends past the
        # one-entry block table.  Neither region may become a selectable zero.
        bt = torch.zeros(1, 1, dtype=torch.int32, device=_DEV)
        ctx = torch.tensor([[2]], dtype=torch.int32, device=_DEV)

        got = qsa_paged_indexer_score(q, w, pool, bt, ctx, block_size=1, max_ctx_len=2)

        self.assertTrue(torch.isneginf(got).all())

    def test_out_of_range_physical_block_is_rejected(self):
        q = torch.ones(1, 1, 1, _D, dtype=torch.bfloat16, device=_DEV)
        w = torch.ones(1, 1, dtype=torch.float32, device=_DEV)
        pool = torch.ones(2, _D, dtype=torch.bfloat16, device=_DEV)
        # Pool capacity is physical IDs [0, 2); ID 2 would read out of bounds.
        bt = torch.tensor([[2]], dtype=torch.int32, device=_DEV)
        ctx = torch.tensor([[1]], dtype=torch.int32, device=_DEV)

        with self.assertRaisesRegex(ValueError, "outside kv_pool capacity"):
            qsa_paged_indexer_score(q, w, pool, bt, ctx, block_size=1, max_ctx_len=1)

        # The runtime validates required blocks before scoring.  Its fast path
        # must still prevent a bad ID from becoming an out-of-bounds pool read.
        logits = qsa_paged_indexer_score(
            q,
            w,
            pool,
            bt,
            ctx,
            block_size=1,
            max_ctx_len=1,
            validate_block_table=False,
        )
        self.assertTrue(torch.isneginf(logits).all())

    def test_matches_torch_reference(self):
        torch.manual_seed(0)
        B, next_n = 3, 1
        bs = 64
        mblk = 16
        q = (torch.randn(B, next_n, _H, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        w = torch.full((B * next_n, _H), 1.0 / math.sqrt(_D), device=_DEV)
        pool = (torch.randn(mblk * bs, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        ctx = torch.randint(1, bs, (B, next_n), dtype=torch.int32, device=_DEV)
        bt = torch.zeros(B, mblk, dtype=torch.int32, device=_DEV)
        bt[:, 0] = 1  # block_id>0 valid; 0=sentinel

        paged = qsa_paged_indexer_score(
            q,
            w,
            pool,
            bt,
            ctx,
            block_size=bs,
            max_ctx_len=int(ctx.max()),
        )
        fast = qsa_paged_indexer_score(
            q,
            w,
            pool,
            bt,
            ctx,
            block_size=bs,
            max_ctx_len=int(ctx.max()),
            validate_block_table=False,
        )
        ref = _ref(q.reshape(-1, _H, _D), w, pool, bt, ctx, bs)
        torch.testing.assert_close(paged, ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(fast, paged)

    def test_next_n_rows_reuse_their_batch_block_table(self):
        torch.manual_seed(1)
        B, next_n = 2, 3
        bs = 4
        q = (torch.randn(B, next_n, _H, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        w = torch.full((B * next_n, _H), 1.0 / math.sqrt(_D), device=_DEV)
        pool = (torch.randn(5 * bs, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        # All tokens of one batch share its logical-to-physical block mapping.
        bt = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=_DEV)
        ctx = torch.tensor([[1, 5, 8], [2, 6, 7]], dtype=torch.int32, device=_DEV)

        paged = qsa_paged_indexer_score(
            q, w, pool, bt, ctx, block_size=bs, max_ctx_len=8
        )
        ref = _ref(q.reshape(-1, _H, _D), w, pool, bt, ctx, bs)

        torch.testing.assert_close(paged, ref, atol=1e-3, rtol=1e-3)

    def test_decode_shapes(self):
        B, mblk, bs, mcl = 4, 1024, 64, 4096
        q = (torch.randn(B, 1, _H, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        w = torch.full((B, _H), 1.0 / math.sqrt(_D), device=_DEV)
        pool = (torch.randn(mblk * bs, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        bt = torch.randint(0, mblk, (B, mblk), dtype=torch.int32, device=_DEV)
        ctx = torch.randint(0, mcl, (B, 1), dtype=torch.int32, device=_DEV)
        logits = qsa_paged_indexer_score(
            q,
            w,
            pool,
            bt,
            ctx,
            block_size=bs,
            max_ctx_len=mcl,
        )
        self.assertEqual(logits.shape, (B, mcl))
        self.assertTrue(torch.isfinite(logits).any())


if __name__ == "__main__":
    unittest.main()
