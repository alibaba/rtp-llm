"""CPU unit tests for the LSA semantic reference (no GPU required).

Run directly:

    python3 test_lsa_reference_test.py

or through Bazel (``//rtp_llm/models_py/triton_kernels/lsa_attention/test:lsa_reference_test``).
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.triton_kernels.lsa_attention.reference import (
    absorb_queries,
    dequantize_kv_fp8,
    quantize_kv_fp8,
    indexer_scores,
    lsa_forward,
    select_slots,
    sparse_mla_attention,
    sparse_mla_attention_naive,
)


class TestSparseMlaReference(unittest.TestCase):
    def test_matches_naive_loop(self) -> None:
        torch.manual_seed(0)
        t_count, heads, dq, dv = 4, 3, 80, 64
        n_count, slots = 96, 12
        q_abs = torch.randn(t_count, heads, dq) * 0.1
        kv = torch.randn(n_count, dq) * 0.1
        indices = torch.randint(0, n_count, (t_count, slots))
        indices[1, 5:] = -1  # padded row
        indices[3, :] = -1  # empty row

        out_ref, lse_ref = sparse_mla_attention(q_abs, kv, indices, kv_lora_rank=dv)
        out_naive, lse_naive = sparse_mla_attention_naive(
            q_abs, kv, indices, kv_lora_rank=dv
        )
        self.assertTrue(torch.allclose(out_ref, out_naive, atol=2e-5, rtol=1e-5))
        finite = lse_ref != float("-inf")
        self.assertTrue(torch.allclose(lse_ref[finite], lse_naive[finite], atol=2e-5))
        # Empty row: zeros output, -inf LSE.
        self.assertTrue(torch.all(out_ref[3] == 0))
        self.assertTrue(torch.all(lse_ref[3] == float("-inf")))

    def test_absorb_matches_explicit_form(self) -> None:
        """Absorbed attention must equal the explicit per-head W_uv form.

        Absorbed: softmax over ``[q_nope @ W_uk ; q_rope] . [c_kv ; k_rope]``,
        then project the latent output with ``W_uv``.
        Explicit: softmax over the same (identical) logits, but project every
        ``c_kv`` slot with ``W_uv`` *before* the weighted sum. Both must agree.
        """
        torch.manual_seed(1)
        t_count, heads = 3, 2
        nope, rope, lora, out_dim, slots = 16, 8, 32, 32, 10
        n_count = 40
        q_nope = torch.randn(t_count, heads, nope) * 0.2
        q_rope = torch.randn(t_count, heads, rope) * 0.2
        w_uk = torch.randn(heads, nope, lora) * 0.05
        w_uv = torch.randn(heads, lora, out_dim) * 0.05
        c_kv = torch.randn(n_count, lora) * 0.2
        k_rope = torch.randn(n_count, rope) * 0.2
        kv = torch.cat([c_kv, k_rope], dim=-1)
        indices = torch.sort(torch.randint(0, n_count, (t_count, slots)), dim=-1).values

        q_abs = absorb_queries(q_nope, q_rope, w_uk)
        out_abs, lse_abs = sparse_mla_attention(q_abs, kv, indices, kv_lora_rank=lora)
        self.assertTrue(torch.isfinite(lse_abs).all())

        # The absorbed and explicit dots are numerically identical; the softmax
        # scale is applied after the dot (RTP/FlashMLA convention: D ** -0.5
        # with D = kv_lora_rank + rope_head_dim).
        scale = (lora + rope) ** -0.5
        for h in range(heads):
            k_h = torch.cat([c_kv @ w_uk[h].transpose(0, 1), k_rope], dim=-1)
            q_h = torch.cat([q_nope[:, h], q_rope[:, h]], dim=-1).unsqueeze(1)
            for t in range(t_count):
                idx = indices[t]
                logits = (q_h[t] @ k_h[idx].T).squeeze(0) * scale  # [S]
                probs = torch.softmax(logits, dim=-1)
                explicit = probs @ (c_kv[idx] @ w_uv[h])  # [out_dim]
                absorbed = out_abs[t, h] @ w_uv[h]  # [out_dim]
                self.assertTrue(
                    torch.allclose(absorbed, explicit, atol=1e-5, rtol=1e-5),
                    msg=f"absorbed != explicit at t={t}, h={h}",
                )
                expected_lse = torch.logsumexp(logits, dim=-1)
                self.assertAlmostEqual(
                    float(lse_abs[t, h]), float(expected_lse), places=5
                )


class TestSelectSlots(unittest.TestCase):
    def test_budget_semantics(self) -> None:
        torch.manual_seed(2)
        topk, init, local = 32, 2, 8
        t_count, n_count = 4, 64
        scores = torch.randn(t_count, n_count)
        valid_lens = torch.tensor([10, 40, 64, 64])

        indices = select_slots(
            scores,
            valid_lens,
            topk=topk,
            init_tokens=init,
            local_tokens=local,
        )
        self.assertEqual(indices.shape, (t_count, topk))

        for t in range(t_count):
            row = indices[t]
            picked = row[row >= 0]
            expected = min(int(valid_lens[t]), topk)
            self.assertEqual(int((row >= 0).sum()), expected)
            self.assertEqual(int((row == -1).sum()), topk - expected)
            # Causality: nothing beyond the valid length.
            self.assertTrue(torch.all(picked < valid_lens[t]))
            self.assertEqual(len(set(picked.tolist())), len(picked.tolist()))
            if int(valid_lens[t]) > init + local:
                # Sink and local window are forced into the budget.
                for s in range(init):
                    self.assertIn(s, picked.tolist())
                start = int(valid_lens[t]) - local
                for s in range(start, int(valid_lens[t])):
                    self.assertIn(s, picked.tolist())

    def test_forced_slots_win_over_low_scores(self) -> None:
        topk, init, local = 16, 2, 4
        t_count, n_count = 2, 32
        scores = torch.zeros(t_count, n_count)
        # Give the streaming regions strongly negative scores; still forced in.
        scores[:, :init] = -100.0
        scores[:, n_count - local :] = -100.0
        valid_lens = torch.tensor([n_count, n_count])
        indices = select_slots(
            scores, valid_lens, topk=topk, init_tokens=init, local_tokens=local
        )
        for t in range(t_count):
            picked = set(indices[t].tolist())
            self.assertTrue(set(range(init)).issubset(picked))
            self.assertTrue(set(range(n_count - local, n_count)).issubset(picked))


class TestLsaForward(unittest.TestCase):
    def test_end_to_end_shapes(self) -> None:
        torch.manual_seed(3)
        t_count, heads, dq, dv = 6, 4, 96, 64
        n_count, slots = 128, 24
        q_abs = torch.randn(t_count, heads, dq) * 0.1
        kv = torch.randn(n_count, dq) * 0.1
        q_idx = torch.randn(t_count, 2, 16) * 0.1
        k_idx = torch.randn(n_count, 16) * 0.1
        weights = torch.randn(t_count, 2)
        valid_lens = torch.tensor([10, 20, 40, 64, 100, 128])

        out, lse, indices = lsa_forward(
            q_abs,
            kv,
            q_idx,
            k_idx,
            weights,
            valid_lens,
            topk=slots,
            init_tokens=2,
            local_tokens=8,
            kv_lora_rank=dv,
        )
        self.assertEqual(out.shape, (t_count, heads, dv))
        self.assertEqual(lse.shape, (t_count, heads))
        self.assertEqual(indices.shape, (t_count, slots))
        self.assertTrue(torch.isfinite(out).all())
        scores = indexer_scores(q_idx, k_idx, weights)
        self.assertTrue(torch.isfinite(scores).all())


class TestFp8KvHelpers(unittest.TestCase):
    def test_roundtrip_error_bounded(self) -> None:
        torch.manual_seed(6)
        kv = torch.randn(256, 576) * 0.3
        kv_fp8, scales = quantize_kv_fp8(kv)
        restored = dequantize_kv_fp8(kv_fp8, scales)
        rel = ((restored - kv).abs() / kv.abs().clamp_min(1e-3)).max().item()
        self.assertLess(rel, 0.15)
        self.assertEqual(tuple(kv_fp8.shape), (256, 576))
        self.assertEqual(tuple(scales.shape), (256, 5))
        self.assertEqual(kv_fp8.dtype, torch.float8_e4m3fn)

    def test_ragged_last_block(self) -> None:
        # DQ = 192 -> one full 128 block + a ragged 64-wide tail block.
        torch.manual_seed(7)
        kv = torch.randn(8, 192) * 0.2
        kv_fp8, scales = quantize_kv_fp8(kv)
        self.assertEqual(tuple(scales.shape), (8, 2))
        restored = dequantize_kv_fp8(kv_fp8, scales)
        self.assertLess(((restored - kv).abs() / kv.abs().clamp_min(1e-3)).max().item(), 0.15)
        with self.assertRaises(ValueError):
            dequantize_kv_fp8(kv_fp8, scales[:, :1])


if __name__ == "__main__":
    unittest.main()