"""End-to-end prefill orchestration tests (prefill.py, L2 deliverable).

Mirrors the offline driver's synthetic path as unittest: a tiny 8-layer
config that covers every real-model layer role (SWA-only, SWA+Engram,
ratio-2 producer/consumers, ratio-1 producer/consumer). Checks finiteness,
per-pool entry counts for the token totals, the Engram n-gram windows
spanning the request tokens (incl. chunk-boundary lookback), cross-chunk
continuation through ``csa_state`` and determinism.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.ppu.engram import (
    build_token_windows,
    compute_engram_hash_ids,
)
from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _torch_sparse_attention
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)

GLOBAL_KV_ENTRY_BYTES = 288
INDEXER_KV_ENTRY_BYTES = 68
SWA_KV_ENTRY_BYTES = 528
CSA_STATE_FLOATS = 1024


class _EngramRecorder:
    """Delegating ops proxy recording engram_lookup call arguments."""

    def __init__(self, ops):
        self._ops = ops
        self.engram_calls = []

    def __getattr__(self, name):
        return getattr(self._ops, name)

    def engram_lookup(self, rows, *, layer_id, context=None):
        self.engram_calls.append((layer_id, rows.detach().clone()))
        return self._ops.engram_lookup(rows, layer_id=layer_id, context=context)


class PrefillEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_p1_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=77)
        cls.geometry = V41Geometry.from_config(cls.config)
        cls.bundle = load_ppu_weights(cls.config, cls.checkpoint)
        cls.ops = PPUOps(
            ep_rank=0,
            ep_size=1,
            num_routed_experts=cls.geometry.n_routed_experts,
            num_local_experts=cls.geometry.num_local_experts(1),
            hidden_size=cls.geometry.hidden_size,
            inter_dim=cls.geometry.moe_inter_dim,
            swiglu_limit=cls.geometry.swiglu_limit,
            mhc_config=cls.geometry.mhc_config(),
            sm_scale=cls.geometry.sm_scale(),
            window=cls.geometry.window,
        )
        cls.bundle.bind(cls.ops)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _core(self, ops=None) -> PPUModelCore:
        return PPUModelCore(self.config, self.bundle, ops or self.ops)

    def _ids(self, tokens, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(
            0, self.geometry.vocab_size, (tokens,), generator=generator
        )

    # ------------------------------------------------------------- e2e

    def test_end_to_end_finite(self):
        core = self._core()
        result = core.prefill(self._ids(33, seed=1), apply_lm_head=True)
        self.assertEqual(tuple(result.hidden.shape), (33, self.geometry.hidden_size))
        self.assertTrue(torch.isfinite(result.hidden).all())
        self.assertEqual(tuple(result.logits.shape), (33, self.geometry.vocab_size))
        self.assertTrue(torch.isfinite(result.logits).all())
        self.assertEqual(set(result.slots), {0})

    def test_cache_pool_entry_counts_for_token_total(self):
        core = self._core()
        core.prefill(self._ids(33, seed=2))
        regions = core.cache.valid_regions()
        self.assertEqual(regions["2:global_kv_2"], {"0": 33 // 2})
        self.assertEqual(regions["2:indexer_kv"], {"0": 33 // 2})
        self.assertEqual(regions["2:csa_state"], {"0": 33})
        self.assertEqual(regions["6:global_kv_1"], {"0": 33})
        self.assertEqual(regions["6:indexer_kv"], {"0": 33})
        for layer in range(self.geometry.num_layers):
            self.assertEqual(regions[f"{layer}:swa_kv"], {"0": 33}, layer)
        # Read shapes carry the seam-contract entry widths.
        global_rows, indexer_rows = core.cache.read_source(2, 0)
        self.assertEqual(tuple(global_rows.shape), (33 // 2, GLOBAL_KV_ENTRY_BYTES))
        self.assertEqual(tuple(indexer_rows.shape), (33 // 2, INDEXER_KV_ENTRY_BYTES))
        swa_rows = core.cache.read_swa(2, 0, torch.arange(33))
        self.assertEqual(tuple(swa_rows.shape), (33, SWA_KV_ENTRY_BYTES))
        csa_rows = core.cache.read_csa(2, 0, torch.arange(33))
        self.assertEqual(tuple(csa_rows.shape), (33, CSA_STATE_FLOATS))

    def test_all_layer_roles_are_exercised(self):
        """The synthetic config itself builds every real-model layer role."""
        geometry = self.geometry
        ratios = [geometry.ratio(layer) for layer in range(geometry.num_layers)]
        self.assertEqual(ratios, [0, 0, 2, 2, 2, 2, 1, 1])
        self.assertEqual(geometry.kv_source_layer_ids, (2, 6))
        self.assertEqual(geometry.index_source_layer_ids, (2, 6))
        self.assertEqual(geometry.engram_layer_ids, (1,))
        core = self._core()
        core.prefill(self._ids(16, seed=3))
        keys = {f"{owner}:{tag}" for owner, tag in core.cache.pool_keys()}
        for owner, tag in (
            (2, "global_kv_2"),
            (2, "indexer_kv"),
            (2, "csa_state"),
            (6, "global_kv_1"),
            (6, "indexer_kv"),
        ):
            self.assertIn(f"{owner}:{tag}", keys)
        for layer in range(8):
            self.assertIn(f"{layer}:swa_kv", keys)
        self.assertNotIn("6:csa_state", keys)  # ratio-1 producer: no CSA pool
        self.assertNotIn("3:global_kv_2", keys)  # consumers own no pool
        # SWA-only layer 0 sees only its own window pool in its cache view.
        view = core.cache.cache_view(0)
        self.assertEqual(set(view.pools), {"swa_kv"})

    def test_multi_request_entry_counts(self):
        core = self._core()
        tokens = 21
        cu_seqlens = torch.tensor([0, 12, 21], dtype=torch.int64)
        result = core.prefill(self._ids(tokens, seed=4), cu_seqlens=cu_seqlens)
        self.assertEqual(set(result.slots), {0, 1})
        self.assertTrue(torch.isfinite(result.hidden).all())
        regions = core.cache.valid_regions()
        expected = {0: 12, 1: 9}
        for request, length in expected.items():
            self.assertEqual(regions["2:global_kv_2"][str(request)], length // 2)
            self.assertEqual(regions["2:indexer_kv"][str(request)], length // 2)
            self.assertEqual(regions["2:csa_state"][str(request)], length)
            self.assertEqual(regions["6:global_kv_1"][str(request)], length)
            self.assertEqual(regions["6:indexer_kv"][str(request)], length)
            for layer in range(8):
                self.assertEqual(regions[f"{layer}:swa_kv"][str(request)], length)

    # ------------------------------------------------------------- engram

    def test_engram_lookup_runs_for_the_engram_layer(self):
        recorder = _EngramRecorder(self.ops)
        core = self._core(ops=recorder)
        core.prefill(self._ids(12, seed=5))
        self.assertEqual([layer_id for layer_id, _ in recorder.engram_calls], [1])
        _, rows = recorder.engram_calls[0]
        n_hash_cols = (
            self.geometry.engram_max_ngram_size - 1
        ) * self.geometry.engram_n_heads
        self.assertEqual(tuple(rows.shape), (12, n_hash_cols))
        table_rows = self.geometry.engram_num_embeddings[0]
        valid = (rows >= 0) & (rows < table_rows)
        self.assertTrue(bool(valid.any()))

    def test_engram_windows_span_request_tokens(self):
        ids = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)
        cu = torch.tensor([0, 3, 5], dtype=torch.int64)
        lookback = torch.tensor([[-1, -1, -1], [7, 8, 9]], dtype=torch.int64)
        windows = build_token_windows(ids, cu, lookback)
        expected = torch.tensor(
            [
                [10, -1, -1, -1],
                [11, 10, -1, -1],
                [12, 11, 10, -1],
                [13, 7, 8, 9],
                [14, 13, 7, 8],
            ],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(windows, expected))

    def test_engram_lookback_changes_the_hash_ids(self):
        ids = self._ids(8, seed=6)
        cu = torch.tensor([0, 8], dtype=torch.int64)
        real = build_token_windows(
            ids, cu, torch.tensor([[3, 2, 1]], dtype=torch.int64)
        )
        fresh = build_token_windows(ids, cu, torch.full((1, 3), -1, dtype=torch.int64))
        self.assertTrue(
            torch.equal(real[0], torch.tensor([[int(ids[0]), 3, 2, 1]]).squeeze(0))
        )  # the chunk's first token carries the lookback history
        self.assertTrue(
            torch.equal(fresh[0], torch.tensor([[int(ids[0]), -1, -1, -1]]).squeeze(0))
        )
        hashes_real = compute_engram_hash_ids(real, self.bundle.engram_hash_params)
        hashes_fresh = compute_engram_hash_ids(fresh, self.bundle.engram_hash_params)
        self.assertFalse(torch.equal(hashes_real, hashes_fresh))

    # ------------------------------------------------------------- chunks

    def test_continuation_chunk_crosses_ratio2_boundary_via_csa_state(self):
        core = self._core()
        first_ids = self._ids(5, seed=7)
        core.prefill(first_ids)
        # Second chunk starts at position 5 (odd): the ratio-2 producer must
        # read the missing left token from csa_state instead of raising.
        second_ids = self._ids(4, seed=8)
        # Most-recent-first predecessors: [id@4, id@3, id@2].
        lookback = first_ids[-3:].flip(0).unsqueeze(0).to(torch.int64)
        self.assertTrue(
            torch.equal(
                lookback,
                torch.tensor([[first_ids[4], first_ids[3], first_ids[2]]]),
            )
        )
        result = core.prefill(second_ids, lookback_token_ids=lookback)
        self.assertEqual(tuple(result.hidden.shape), (4, self.geometry.hidden_size))
        self.assertTrue(torch.isfinite(result.hidden).all())
        regions = core.cache.valid_regions()
        self.assertEqual(regions["2:global_kv_2"]["0"], 9 // 2)
        self.assertEqual(regions["2:csa_state"]["0"], 9)
        self.assertEqual(regions["6:global_kv_1"]["0"], 9)
        self.assertEqual(regions["0:swa_kv"]["0"], 9)

    def test_continuation_without_lookback_uses_fresh_windows(self):
        core = self._core()
        core.prefill(self._ids(6, seed=9))
        result = core.prefill(self._ids(3, seed=10))
        self.assertTrue(torch.isfinite(result.hidden).all())
        self.assertEqual(core.cache.request_length(0), 9)

    # ------------------------------------------------------------- request ids

    def test_multi_request_explicit_request_ids_are_honored(self):
        """wave2 fix ③ regression: explicit request_ids need not be 0..B-1.

        Slot allocation, the cache bookkeeping and the per-request production
        must all key by the ids the caller actually passed.
        """
        core = self._core()
        tokens = 21
        cu_seqlens = torch.tensor([0, 12, 21], dtype=torch.int64)
        ids = self._ids(tokens, seed=12)
        request_ids = torch.tensor([7] * 12 + [9] * 9, dtype=torch.int64)
        result = core.prefill(ids, cu_seqlens=cu_seqlens, request_ids=request_ids)
        self.assertEqual(set(result.slots), {7, 9})
        self.assertTrue(torch.isfinite(result.hidden).all())
        regions = core.cache.valid_regions()
        expected = {7: 12, 9: 9}
        for request, length in expected.items():
            self.assertEqual(regions["2:global_kv_2"][str(request)], length // 2)
            self.assertEqual(regions["2:csa_state"][str(request)], length)
            self.assertEqual(regions["6:global_kv_1"][str(request)], length)
            self.assertEqual(regions["0:swa_kv"][str(request)], length)
        # Continuing ONLY request 9 must extend exactly that request's pools
        # (its chunk starts at position 9 — odd — exercising the csa_state
        # continuation path under an explicit non-zero id).
        more = self._ids(4, seed=13)
        result2 = core.prefill(
            more,
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int64),
            request_ids=torch.full((4,), 9, dtype=torch.int64),
        )
        self.assertEqual(set(result2.slots), {9})
        self.assertTrue(torch.isfinite(result2.hidden).all())
        regions2 = core.cache.valid_regions()
        self.assertEqual(regions2["0:swa_kv"]["7"], 12)
        self.assertEqual(regions2["0:swa_kv"]["9"], 13)
        self.assertEqual(regions2["2:csa_state"]["9"], 13)

    def test_request_ids_not_aligned_with_cu_segments_raise(self):
        core = self._core()
        ids = self._ids(6, seed=14)
        cu = torch.tensor([0, 3, 6], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "constant within each cu_seqlens"):
            core.prefill(
                ids,
                cu_seqlens=cu,
                request_ids=torch.tensor([0, 1, 0, 1, 1, 1], dtype=torch.int64),
            )
        with self.assertRaisesRegex(ValueError, "distinct ids"):
            core.prefill(
                ids, cu_seqlens=cu, request_ids=torch.zeros(6, dtype=torch.int64)
            )

    # ------------------------------------------------------------- device

    def test_default_branch_derives_tensors_on_the_input_device(self):
        """wave2 fix ② regression: the default request_ids / positions / cu
        tensors are created on the INPUT tensor's device (cuda inputs used to
        get CPU bookkeeping and crashed after the long weight load)."""

        from unittest import mock

        import rtp_llm.models_py.modules.dsv41.ppu.prefill as prefill_module

        core = self._core()
        captured = {}
        original = prefill_module.build_step_context

        def spy(positions, request_ids, cu_seqlens, *args, **kwargs):
            captured["devices"] = (
                positions.device,
                request_ids.device,
                cu_seqlens.device,
            )
            return original(positions, request_ids, cu_seqlens, *args, **kwargs)

        ids = self._ids(9, seed=15)
        with mock.patch.object(prefill_module, "build_step_context", spy):
            core.prefill(ids)
        want = ids.device
        self.assertEqual(captured["devices"], (want, want, want))

    # ------------------------------------------------------------- indexer

    def test_pure_index_source_reads_the_kv_source_pool(self):
        """wave2 regression: index sources that are NOT kv sources (the real
        config's layers 24/28/32/36) must score their top-k against the KV
        SOURCE's indexer pool — the old code keyed ``global_slots[layer_id]``
        and raised KeyError on such layers."""
        config = dict(self.config)
        config["index_source_layer_ids"] = [2, 4, 6]  # 4 is a pure index source
        root = Path(tempfile.mkdtemp(prefix="dsv41_p1_idx_"))
        try:
            checkpoint = write_synthetic_checkpoint(config, root, seed=78)
            geometry = V41Geometry.from_config(config)
            self.assertNotIn(4, geometry.kv_source_layer_ids)
            bundle = load_ppu_weights(config, checkpoint)
            ops = PPUOps(
                ep_rank=0,
                ep_size=1,
                num_routed_experts=geometry.n_routed_experts,
                num_local_experts=geometry.num_local_experts(1),
                hidden_size=geometry.hidden_size,
                inter_dim=geometry.moe_inter_dim,
                swiglu_limit=geometry.swiglu_limit,
                mhc_config=geometry.mhc_config(),
                sm_scale=geometry.sm_scale(),
                window=geometry.window,
            )
            bundle.bind(ops)
            core = PPUModelCore(config, bundle, ops)
            result = core.prefill(self._ids(17, seed=16), apply_lm_head=True)
            self.assertTrue(torch.isfinite(result.logits).all())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # ------------------------------------------------------------- determinism

    def test_same_seed_same_logits(self):
        ids = self._ids(24, seed=11)
        first = self._core().prefill(ids, apply_lm_head=True)
        second = self._core().prefill(ids, apply_lm_head=True)
        torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
        torch.testing.assert_close(first.hidden, second.hidden, rtol=0, atol=0)


class SparseAttentionSinkTest(unittest.TestCase):
    def test_sink_changes_each_heads_denominator(self):
        # Zero queries give unit mass to each real key. The two heads' sink
        # masses are 1 and 3, with zero values. Padding contributes no mass.
        q = torch.zeros(2, 2, 2)
        kv = torch.tensor([[[2.0, 4.0]], [[6.0, 8.0]]])
        indices = torch.tensor([[[0, -1]], [[1, 0]]], dtype=torch.int32)
        lengths = torch.tensor([1, 2], dtype=torch.int32)
        sink = torch.tensor([1.0, 3.0]).log()
        result = _torch_sparse_attention(q, kv, indices, lengths, 1.0, sink)
        expected = torch.tensor(
            [[[1.0, 2.0], [0.5, 1.0]], [[8.0 / 3, 4.0], [1.6, 2.4]]]
        )
        torch.testing.assert_close(result, expected)

    def test_sink_limits_and_empty_rows(self):
        q = torch.zeros(2, 2, 2, dtype=torch.bfloat16)
        kv = torch.tensor([[[2.0, 4.0]]], dtype=torch.bfloat16)
        indices = torch.tensor([[[0]], [[-1]]], dtype=torch.int32)
        lengths = torch.tensor([1, 0], dtype=torch.int32)
        sink = torch.tensor([float("-inf"), float("inf")])
        result = _torch_sparse_attention(q, kv, indices, lengths, 1.0, sink)
        expected = torch.tensor(
            [[[2.0, 4.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(result, expected)

    def test_no_sink_keeps_regular_attention(self):
        q = torch.zeros(1, 1, 2)
        kv = torch.tensor([[[2.0, 4.0]], [[6.0, 8.0]]])
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
        lengths = torch.tensor([2], dtype=torch.int32)
        result = _torch_sparse_attention(q, kv, indices, lengths, 1.0, None)
        torch.testing.assert_close(result, torch.tensor([[[4.0, 6.0]]]))


if __name__ == "__main__":
    unittest.main()
