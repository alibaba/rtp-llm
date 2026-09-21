"""CPU contract tests for the mean-pool indexer compression writer.

Verifies that the pooled keys produced by :func:`compress_prefill` match the
reference ``Qwen4ExpQSAIndexer.pooled_block_keys_all`` and expose the bf16 layout
consumed by the production paged scorer.
"""

import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.indexer import (
    Qwen4ExpQSAIndexer,
    apply_partial_rope,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer_compressor import (
    _write_indexer_cache_vectorized,
    compress_prefill,
    restore_indexer_cache,
    write_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.norm import exact_head_rms_norm

_H = 4
_D = 128
_HIDDEN = 32
_RATIO = 4
_BUDGET = 32
_EPS = 1e-6


def _build_indexer(device="cpu"):
    return Qwen4ExpQSAIndexer(
        qk_proj=torch.randn((_H + 1) * _D, _HIDDEN, device=device) * 0.05,
        q_norm_gamma=torch.rand(_D, device=device),
        k_norm_gamma=torch.rand(_D, device=device),
        n_heads=_H,
        kv_heads=1,
        head_dim=_D,
        token_budget=_BUDGET,
        compress_ratio=_RATIO,
        norm_eps=_EPS,
    )


class IndexerCompressorTest(unittest.TestCase):
    def test_pooled_keys_match_reference_block_by_block(self):
        torch.manual_seed(42)
        seq_len = 48
        dev = "cpu"

        indexer = _build_indexer(dev)
        hidden = torch.randn(2, seq_len, _HIDDEN, device=dev)
        angles = torch.rand(2, seq_len, _D // 2, device=dev) * 6.28
        cos = angles.cos().repeat_interleave(2, dim=-1)
        sin = angles.sin().repeat_interleave(2, dim=-1)

        ref_q, ref_raw = indexer.project(hidden)
        ref_pooled = indexer.pooled_block_keys_all(ref_raw, cos, sin)

        result = compress_prefill(
            hidden,
            indexer.qk_proj,
            indexer.k_norm_gamma,
            cos,
            sin,
            ratio=_RATIO,
            head_dim=_D,
            norm_eps=_EPS,
        )
        our_pooled = result["pooled_k"]

        self.assertEqual(our_pooled.dtype, torch.bfloat16)
        self.assertNotIn("pooled_scale", result)
        torch.testing.assert_close(
            our_pooled,
            ref_pooled.to(torch.bfloat16),
            atol=0,
            rtol=0,
            msg="pooled keys differ from reference",
        )

    def test_ragged_lengths(self):
        dev = "cpu"
        indexer = _build_indexer(dev)

        for T in (1, 3, 4, 5, 7):
            hidden = torch.randn(1, T, _HIDDEN, device=dev)
            angles = torch.rand(1, T, _D // 2, device=dev) * 6.28
            cos = angles.cos().repeat_interleave(2, dim=-1)
            sin = angles.sin().repeat_interleave(2, dim=-1)

            ref_q, ref_raw = indexer.project(hidden)
            ref_pooled = indexer.pooled_block_keys_all(ref_raw, cos, sin)

            result = compress_prefill(
                hidden,
                indexer.qk_proj,
                indexer.k_norm_gamma,
                cos,
                sin,
                ratio=_RATIO,
                head_dim=_D,
                norm_eps=_EPS,
            )
            nb = result["pooled_k"].shape[1]
            if nb > 0:
                torch.testing.assert_close(
                    result["pooled_k"],
                    ref_pooled.to(torch.bfloat16),
                    atol=0,
                    rtol=0,
                    msg=f"T={T} pooled keys mismatch",
                )
            self.assertEqual(ref_pooled.shape[1], nb)

    def test_contract_rejects_an_invalid_rotary_width(self):
        indexer = _build_indexer()
        hidden = torch.randn(1, 4, _HIDDEN)
        cos = torch.ones(1, 4, _D + 2)
        sin = torch.zeros_like(cos)

        with self.assertRaisesRegex(ValueError, "rotary_dim"):
            compress_prefill(
                hidden,
                indexer.qk_proj,
                indexer.k_norm_gamma,
                cos,
                sin,
                ratio=_RATIO,
                head_dim=_D,
                norm_eps=_EPS,
            )


class IndexerCacheWriterTest(unittest.TestCase):
    D = 8
    RATIO = 4
    KV_TOKENS_PER_BLOCK = 8
    STATE_TOKENS_PER_BLOCK = 8
    RING_ENTRIES = 8

    def setUp(self):
        torch.manual_seed(7)
        self.gamma = torch.rand(self.D)
        angles = torch.arange(32, dtype=torch.float32)[:, None] * torch.tensor(
            [[0.03, 0.05]]
        )
        self.cos = angles.cos().repeat_interleave(2, dim=-1)
        self.sin = angles.sin().repeat_interleave(2, dim=-1)

    def _pools(self, kv_blocks=7, state_blocks=8):
        kv_pool = torch.zeros(
            kv_blocks,
            self.KV_TOKENS_PER_BLOCK // self.RATIO,
            self.D,
            dtype=torch.bfloat16,
        )
        state_pool = torch.zeros(
            state_blocks, self.RING_ENTRIES, self.D, dtype=torch.float32
        )
        return kv_pool, state_pool

    def _expected(self, raw_keys, block_start):
        pooled = raw_keys.float().mean(dim=0).to(raw_keys.dtype)
        pooled = exact_head_rms_norm(pooled, self.gamma, _EPS)
        return apply_partial_rope(
            pooled, self.cos[block_start], self.sin[block_start]
        ).to(torch.bfloat16)

    def _write(
        self,
        raw_keys,
        cu_seqlens,
        start_positions,
        kv_pool,
        kv_block_table,
        state_pool,
        state_block_table,
        **kwargs,
    ):
        return write_indexer_cache(
            raw_keys,
            torch.tensor(cu_seqlens, dtype=torch.int32),
            torch.tensor(start_positions, dtype=torch.int64),
            self.cos,
            self.sin,
            self.gamma,
            kv_pool,
            torch.tensor(kv_block_table, dtype=torch.int32),
            state_pool,
            torch.tensor(state_block_table, dtype=torch.int32),
            ratio=self.RATIO,
            kv_tokens_per_block=self.KV_TOKENS_PER_BLOCK,
            state_tokens_per_block=self.STATE_TOKENS_PER_BLOCK,
            norm_eps=_EPS,
            **kwargs,
        )

    def test_ragged_batch_uses_each_requests_own_physical_blocks(self):
        first = torch.randn(5, self.D, dtype=torch.bfloat16)
        second = torch.randn(8, self.D, dtype=torch.bfloat16)
        raw = torch.cat([first, second])
        kv_pool, state_pool = self._pools()

        result = self._write(
            raw,
            [0, 5, 13],
            [0, 0],
            kv_pool,
            [[3, 5], [1, 4]],
            state_pool,
            [[6], [2]],
        )

        self.assertEqual(result["num_kv_writes"], 3)
        torch.testing.assert_close(kv_pool[3, 0], self._expected(first[:4], 0))
        torch.testing.assert_close(kv_pool[1, 0], self._expected(second[:4], 0))
        torch.testing.assert_close(kv_pool[1, 1], self._expected(second[4:8], 4))
        self.assertEqual(int(result["kv_slots"][3]), 6)
        self.assertEqual(int(result["kv_slots"][8]), 2)
        self.assertEqual(int(result["kv_slots"][12]), 3)
        torch.testing.assert_close(state_pool[6, 4], first[4].float())
        torch.testing.assert_close(state_pool[2, 7], second[7].float())

    def test_vectorized_writer_matches_reference_for_ragged_prefill(self):
        first = torch.randn(5, self.D, dtype=torch.bfloat16)
        second = torch.randn(8, self.D, dtype=torch.bfloat16)
        raw = torch.cat([first, second])
        reference_kv, reference_state = self._pools()
        vector_kv, vector_state = self._pools()
        cu = torch.tensor([0, 5, 13], dtype=torch.int32)
        starts = torch.tensor([0, 0], dtype=torch.int64)
        kv_table = torch.tensor([[3, 5], [1, 4]], dtype=torch.int32)
        state_table = torch.tensor([[6], [2]], dtype=torch.int32)
        positions = torch.cat([torch.arange(5), torch.arange(8)])
        block_starts = positions - positions.remainder(self.RATIO)
        token_rope_cos = self.cos.index_select(0, block_starts)
        token_rope_sin = self.sin.index_select(0, block_starts)

        reference = write_indexer_cache(
            raw,
            cu,
            starts,
            self.cos,
            self.sin,
            self.gamma,
            reference_kv,
            kv_table,
            reference_state,
            state_table,
            ratio=self.RATIO,
            kv_tokens_per_block=self.KV_TOKENS_PER_BLOCK,
            state_tokens_per_block=self.STATE_TOKENS_PER_BLOCK,
            norm_eps=_EPS,
            capture_undo=True,
        )
        vectorized = _write_indexer_cache_vectorized(
            raw,
            cu,
            starts,
            token_rope_cos,
            token_rope_sin,
            self.gamma,
            vector_kv,
            kv_table,
            vector_state,
            state_table,
            ratio=self.RATIO,
            kv_tokens_per_block=self.KV_TOKENS_PER_BLOCK,
            state_tokens_per_block=self.STATE_TOKENS_PER_BLOCK,
            norm_eps=_EPS,
            invalid_block_policy="raise",
            capture_undo=True,
            rope_is_token_aligned=True,
        )

        torch.testing.assert_close(vector_kv, reference_kv)
        torch.testing.assert_close(vector_state, reference_state)
        for key in ("state_slots", "kv_slots", "completed"):
            self.assertTrue(torch.equal(vectorized[key], reference[key]))
        self.assertEqual(vectorized["num_state_writes"], reference["num_state_writes"])
        self.assertEqual(vectorized["num_kv_writes"], reference["num_kv_writes"])

        restore_indexer_cache(vectorized["undo"])
        self.assertEqual(int(torch.count_nonzero(vector_kv)), 0)
        self.assertEqual(int(torch.count_nonzero(vector_state)), 0)

    def test_decode_completes_a_prefill_tail_from_the_state_ring(self):
        block = torch.randn(self.RATIO, self.D, dtype=torch.bfloat16)
        kv_pool, state_pool = self._pools()
        kv_table = [[4]]
        state_table = [[2]]

        prefill = self._write(
            block[:3],
            [0, 3],
            [0],
            kv_pool,
            kv_table,
            state_pool,
            state_table,
        )
        self.assertEqual(prefill["num_kv_writes"], 0)

        decode = self._write(
            block[3:],
            [0, 1],
            [3],
            kv_pool,
            kv_table,
            state_pool,
            state_table,
        )
        self.assertEqual(decode["num_kv_writes"], 1)
        self.assertEqual(int(decode["kv_slots"][0]), 8)
        torch.testing.assert_close(kv_pool[4, 0], self._expected(block, 0))

    def test_second_logical_block_obeys_noncontiguous_block_tables(self):
        raw = torch.randn(4, self.D, dtype=torch.bfloat16)
        kv_pool, state_pool = self._pools()

        result = self._write(
            raw,
            [0, 4],
            [8],
            kv_pool,
            [[2, 5]],
            state_pool,
            [[3, 4]],
        )

        self.assertEqual(int(result["kv_slots"][-1]), 10)
        torch.testing.assert_close(kv_pool[5, 0], self._expected(raw, 8))
        torch.testing.assert_close(state_pool[4, 3], raw[3].float())
        self.assertEqual(int(torch.count_nonzero(kv_pool[2])), 0)

    def test_state_logical_block_past_table_fails_before_either_pool_is_written(self):
        raw = torch.randn(4, self.D, dtype=torch.bfloat16)
        kv_pool, state_pool = self._pools()
        kv_pool.random_(-3, 3)
        state_pool.normal_()
        kv_before = kv_pool.clone()
        state_before = state_pool.clone()

        with self.assertRaisesRegex(ValueError, "outside block_table width"):
            self._write(
                raw,
                [0, 4],
                [8],
                kv_pool,
                [[2, 5]],
                state_pool,
                [[3]],
            )

        self.assertTrue(torch.equal(kv_pool, kv_before))
        self.assertTrue(torch.equal(state_pool, state_before))

    def test_invalid_blocks_raise_before_mutation_or_explicitly_skip(self):
        raw = torch.randn(4, self.D, dtype=torch.bfloat16)
        kv_pool, state_pool = self._pools()

        with self.assertRaisesRegex(ValueError, "block_id must be > 0"):
            self._write(
                raw,
                [0, 4],
                [0],
                kv_pool,
                [[1]],
                state_pool,
                [[0]],
            )
        self.assertEqual(int(torch.count_nonzero(kv_pool)), 0)
        self.assertEqual(int(torch.count_nonzero(state_pool)), 0)

        skipped = self._write(
            raw,
            [0, 4],
            [0],
            kv_pool,
            [[0]],
            state_pool,
            [[0]],
            invalid_block_policy="skip",
        )
        self.assertEqual(skipped["num_state_writes"], 0)
        self.assertEqual(skipped["num_kv_writes"], 0)
        self.assertTrue(skipped["completed"][-1])
        self.assertTrue((skipped["state_slots"] == -1).all())
        self.assertTrue((skipped["kv_slots"] == -1).all())

    def test_capture_undo_deduplicates_wrapped_destinations_and_restores(self):
        raw = torch.randn(12, self.D, dtype=torch.bfloat16)
        kv_pool, state_pool = self._pools()
        kv_pool.random_(-3, 3)
        state_pool.normal_()
        kv_before = kv_pool.clone()
        state_before = state_pool.clone()

        result = self._write(
            raw,
            [0, 12],
            [0],
            kv_pool,
            [[4, 4]],
            state_pool,
            [[2, 2]],
            capture_undo=True,
        )

        undo = result["undo"]
        self.assertEqual(int(undo.state_slots.numel()), self.RING_ENTRIES)
        self.assertEqual(int(undo.kv_slots.numel()), 2)
        self.assertFalse(torch.equal(kv_pool, kv_before))
        self.assertFalse(torch.equal(state_pool, state_before))
        restore_indexer_cache(undo)
        self.assertTrue(torch.equal(kv_pool, kv_before))
        self.assertTrue(torch.equal(state_pool, state_before))


if __name__ == "__main__":
    unittest.main()
