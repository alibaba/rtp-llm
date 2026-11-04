"""Decode-step orchestration tests (L3, D/DP8 role) — wave 2.

The orchestrator now runs on the REAL surfaces: L2's ``PPUWeightBundle`` +
``PPUKVCache`` (the wave-1 ``LocalDecodeCache`` stub is gone). The harness
uses the same synthetic checkpoint as ``test_prefill.py`` (every real-model
layer role: SWA-only, Engram+SWA, ratio-2 producer/consumers, ratio-1
producer/consumer) and seeds the cache with ``run_prefill`` — so the
strongest check here is decode-vs-prefill consistency: the hidden state a
decode step produces for token t must match the row-t hidden of a prefill
that processed the same tokens in one batch.
"""

import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.ppu import decode as decode_module
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.decode import (
    DecodeStepOrchestrator,
    PPUCacheSeam,
    cache_geometry_from_v41,
    make_decode_cache_seam,
    wire_decode,
)
from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import run_prefill
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)


class _RecordingOps:
    """Delegating ops proxy recording moe / engram_lookup call arguments."""

    def __init__(self, ops):
        self._ops = ops
        self.moe_calls = []
        self.engram_calls = []

    def __getattr__(self, name):
        return getattr(self._ops, name)

    def moe(self, x, ids, weights, *, layer_id, context=None):
        self.moe_calls.append(
            (layer_id, ids.detach().clone(), weights.detach().clone())
        )
        return self._ops.moe(x, ids, weights, layer_id=layer_id, context=context)

    def engram_lookup(self, rows, *, layer_id, context=None):
        self.engram_calls.append((layer_id, rows.detach().clone()))
        return self._ops.engram_lookup(rows, layer_id=layer_id, context=context)


class DecodeHarness(unittest.TestCase):
    """Shared synthetic model (same recipe as ``test_prefill.py``)."""

    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_d2_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=177)
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

    # -- helpers ------------------------------------------------------------

    def _ids(self, count, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(
            0, self.geometry.vocab_size, (count,), generator=generator
        ).tolist()

    def _fresh_cache(self):
        return PPUKVCache(geometry=cache_geometry_from_v41(self.geometry))

    def _prefill(self, cache, tokens, seed=0):
        input_ids = torch.tensor(tokens, dtype=torch.int64)
        return run_prefill(self.bundle, self.ops, cache, input_ids, apply_lm_head=True)

    def _orchestrator(self, cache, ops=None):
        return DecodeStepOrchestrator(
            ops=ops or self.ops, bundle=self.bundle, cache=cache
        )


class CacheSeamTest(DecodeHarness):
    def test_ppu_kv_cache_adapts(self):
        seam = make_decode_cache_seam(self._fresh_cache())
        self.assertIsInstance(seam, PPUCacheSeam)

    def test_none_fails_loud(self):
        with self.assertRaises(TypeError):
            make_decode_cache_seam(None)

    def test_unknown_object_fails_loud(self):
        with self.assertRaises(TypeError):
            make_decode_cache_seam(object())

    def test_seam_passthrough(self):
        seam = PPUCacheSeam(self._fresh_cache())
        self.assertIs(make_decode_cache_seam(seam), seam)


class RequestLifecycleTest(DecodeHarness):
    def test_add_request_requires_committed_cache(self):
        orchestrator = self._orchestrator(self._fresh_cache())
        with self.assertRaises(RuntimeError):
            orchestrator.add_request(0, [1, 2, 3])

    def test_foreign_owner_request_is_rejected(self):
        orchestrator = self._orchestrator(self._fresh_cache())
        with self.assertRaises(RuntimeError):
            orchestrator.add_request(8, [1], owner_rank=1)

    def test_missing_engram_params_fail_loud(self):
        bundle = self.bundle
        saved = bundle.engram_hash_params
        bundle.engram_hash_params = None
        try:
            with self.assertRaises(RuntimeError):
                self._orchestrator(self._fresh_cache())
        finally:
            bundle.engram_hash_params = saved

    def test_zero_rows_without_placeholders_fails_loud(self):
        cache = self._fresh_cache()
        tokens = self._ids(6, seed=3)
        self._prefill(cache, tokens)
        orchestrator = self._orchestrator(cache)
        orchestrator.add_request(0, tokens)
        with self.assertRaises(RuntimeError):
            orchestrator.step({})


class DecodeStepTest(DecodeHarness):
    def test_step_hidden_matches_prefill_last_row(self):
        """Decode of token t == prefill row t (the consistency contract)."""
        tokens = self._ids(9, seed=11)
        # Prefill over the prompt only; its last-row logits sample token 9.
        cache = self._fresh_cache()
        prompt = self._prefill(cache, tokens)
        next_token = int(prompt.logits[-1].argmax())
        # Uninterrupted prefill over prompt + generated token.
        full_cache = self._fresh_cache()
        full = self._prefill(full_cache, tokens + [next_token])
        # Decode continues the prompt-only cache by exactly one step.
        orchestrator = self._orchestrator(cache)
        orchestrator.add_request(0, tokens)
        result = orchestrator.step({0: next_token})
        self.assertEqual(tuple(result.hidden.shape), (1, self.geometry.hidden_size))
        self.assertTrue(torch.isfinite(result.hidden).all())
        reference = full.hidden[-1:].float()
        produced = result.hidden.float()
        self.assertTrue(
            torch.allclose(produced, reference, atol=2e-2, rtol=2e-2),
            msg=f"max diff {(produced - reference).abs().max().item():.3e}",
        )
        # Same next-token decision as the full prefill's last row.
        self.assertEqual(int(result.logits[0].argmax()), int(full.logits[-1].argmax()))

    def test_multi_step_state_advances(self):
        cache = self._fresh_cache()
        tokens = self._ids(8, seed=21)
        self._prefill(cache, tokens)
        orchestrator = self._orchestrator(cache)
        orchestrator.add_request(0, tokens)
        steps = 4
        logits = None
        for step in range(steps):
            token = int(logits[0].argmax()) if logits is not None else 5
            result = orchestrator.step({0: token})
            self.assertTrue(torch.isfinite(result.hidden).all(), step)
            self.assertTrue(torch.isfinite(result.logits).all(), step)
            self.assertEqual(result.row_positions, [8 + step])
            logits = result.logits
        state = orchestrator.requests[0]
        self.assertEqual(state.seq_len, 8 + steps)
        self.assertEqual(state.steps_done, steps)
        regions = cache.valid_regions()
        total = 8 + steps
        self.assertEqual(regions["2:global_kv_2"], {"0": total // 2})
        self.assertEqual(regions["2:csa_state"], {"0": total})
        self.assertEqual(regions["6:global_kv_1"], {"0": total})
        for layer in range(self.geometry.num_layers):
            self.assertEqual(regions[f"{layer}:swa_kv"], {"0": total}, layer)

    def test_ratio2_completion_boundary_from_odd_prompt(self):
        cache = self._fresh_cache()
        tokens = self._ids(5, seed=31)
        self._prefill(cache, tokens)
        regions = cache.valid_regions()
        self.assertEqual(regions["2:global_kv_2"], {"0": 2})  # 5 // 2
        self.assertEqual(regions["2:csa_state"], {"0": 5})  # one pending row
        orchestrator = self._orchestrator(cache)
        orchestrator.add_request(0, tokens)
        orchestrator.step({0: 3})
        regions = cache.valid_regions()
        self.assertEqual(regions["2:global_kv_2"], {"0": 3})  # pair completed
        orchestrator.step({0: 4})
        regions = cache.valid_regions()
        self.assertEqual(regions["2:global_kv_2"], {"0": 3})  # new pending row
        self.assertEqual(regions["2:csa_state"], {"0": 7})

    def test_placeholder_rows_are_masked_everywhere(self):
        ops = _RecordingOps(self.ops)
        cache = self._fresh_cache()
        tokens = self._ids(6, seed=41)
        self._prefill(cache, tokens)
        orchestrator = self._orchestrator(cache, ops=ops)
        orchestrator.add_request(0, tokens)
        result = orchestrator.step({0: 7}, placeholders=3)
        self.assertTrue(torch.isfinite(result.hidden).all())
        self.assertEqual(tuple(result.hidden.shape), (1, self.geometry.hidden_size))
        self.assertEqual(int(result.valid_mask.sum()), 1)
        self.assertEqual(result.num_placeholders, 3)
        invalid = ~result.valid_mask.bool()
        total = int(result.valid_mask.shape[0])
        placeholder_calls = [
            (ids, weights)
            for (_, ids, weights) in ops.moe_calls
            if ids.shape[0] == total
        ]
        self.assertTrue(placeholder_calls)
        for ids, weights in placeholder_calls:
            self.assertTrue(bool((ids[invalid] == -1).all()))
            self.assertTrue(bool((weights[invalid] == 0).all()))
            self.assertTrue(bool((ids[~invalid] >= 0).any()))
        # No cache pollution from placeholder rows (prompt + one token only).
        regions = cache.valid_regions()
        self.assertEqual(regions["2:global_kv_2"], {"0": 3})  # 7 tokens // 2
        self.assertEqual(regions["0:swa_kv"], {"0": 7})

    def test_finished_requests_drop_out_but_step_runs(self):
        cache = self._fresh_cache()
        tokens = self._ids(6, seed=51)
        self._prefill(cache, tokens)
        orchestrator = self._orchestrator(cache)
        orchestrator.add_request(0, tokens)
        orchestrator.mark_finished(0)
        result = orchestrator.step({0: 9}, placeholders=2)
        self.assertEqual(tuple(result.hidden.shape), (0, self.geometry.hidden_size))
        self.assertEqual(result.row_request_ids, [])
        self.assertEqual(orchestrator.requests[0].seq_len, 6)
        regions = cache.valid_regions()
        self.assertEqual(regions["0:swa_kv"], {"0": 6})

    def test_index_reuse_matches_source_layer(self):
        cache = self._fresh_cache()
        tokens = self._ids(10, seed=61)
        self._prefill(cache, tokens)
        orchestrator = self._orchestrator(cache)
        orchestrator.add_request(0, tokens)
        calls = []
        original = decode_module._layer_attention

        def wrapper(q, selected_blocks, **kwargs):
            calls.append((orchestrator._current_layer_id, selected_blocks))
            return original(q, selected_blocks, **kwargs)

        decode_module._layer_attention = wrapper
        try:
            orchestrator.step({0: 3})
        finally:
            decode_module._layer_attention = original
        by_layer = {layer: blocks for layer, blocks in calls}
        # Consumers reuse their index source's selection verbatim.
        for consumer, source in ((3, 2), (4, 2), (5, 2), (7, 6)):
            self.assertTrue(torch.equal(by_layer[consumer], by_layer[source]))
        self.assertFalse(torch.equal(by_layer[2], by_layer[6]))

    def test_batched_indexer_matches_per_request_reference(self):
        cache = self._fresh_cache()
        prompt_a = self._ids(9, seed=71)
        prompt_b = self._ids(7, seed=73)
        prompt_c = self._ids(11, seed=79)
        input_ids = torch.tensor(prompt_a + prompt_b + prompt_c, dtype=torch.int64)
        cu = torch.tensor([0, 9, 16, 27], dtype=torch.int64)
        run_prefill(self.bundle, self.ops, cache, input_ids, cu_seqlens=cu)
        orchestrator = self._orchestrator(cache)
        for rid, prompt in ((0, prompt_a), (1, prompt_b), (2, prompt_c)):
            orchestrator.add_request(rid, prompt)
        layer = self.bundle.layers[2]
        rows = 3
        generator = torch.Generator().manual_seed(83)
        x_pre = torch.randn(rows, self.geometry.hidden_size, generator=generator).to(
            torch.bfloat16
        )
        qr = torch.randn(rows, self.geometry.q_lora_rank, generator=generator).to(
            torch.bfloat16
        )
        active = [0, 1, 2]
        positions = [9, 7, 11]
        batched = orchestrator._indexer_topk(layer, x_pre, qr, active, positions)
        reference = orchestrator._indexer_topk_per_request(
            layer, x_pre, qr, active, positions
        )
        self.assertEqual(batched.shape, reference.shape)
        self.assertTrue(torch.equal(batched, reference))
        self.assertTrue(bool((batched >= 0).any()))

    def test_engram_windows_do_not_bleed_across_requests(self):
        ops = _RecordingOps(self.ops)
        cache = self._fresh_cache()
        prompt_a = [1, 2, 3, 4, 5, 6]
        prompt_b = [30, 29, 28, 27, 26, 25]
        input_ids = torch.tensor(prompt_a + prompt_b, dtype=torch.int64)
        cu = torch.tensor([0, 6, 12], dtype=torch.int64)
        run_prefill(self.bundle, self.ops, cache, input_ids, cu_seqlens=cu)
        orchestrator = self._orchestrator(cache, ops=ops)
        orchestrator.add_request(0, prompt_a)
        orchestrator.add_request(1, prompt_b)
        orchestrator.step({0: 7, 1: 7})
        layer_id, rows = ops.engram_calls[0]
        self.assertEqual(layer_id, 1)  # the synthetic config's Engram layer
        self.assertEqual(
            tuple(rows.shape),
            (
                2,
                (self.geometry.engram_max_ngram_size - 1)
                * self.geometry.engram_n_heads,
            ),
        )
        # Same current token, different predecessor history -> different
        # n-gram hashes from the 2-gram columns on.
        self.assertFalse(torch.equal(rows[0], rows[1]))


class WireDecodeTest(DecodeHarness):
    def _core(self, ops=None):
        return PPUModelCore(self.config, self.bundle, ops or self.ops)

    def test_unwired_core_decode_fails_loud(self):
        core = self._core()
        with self.assertRaises(NotImplementedError):
            core.decode(torch.zeros(1, dtype=torch.int64))

    def test_wire_decode_registers_impl_and_runs(self):
        fake_ops = types.ModuleType("rtp_llm.ops")
        fake_ops.PyModelOutputs = lambda hidden: types.SimpleNamespace(hidden=hidden)
        saved = sys.modules.get("rtp_llm.ops")
        sys.modules["rtp_llm.ops"] = fake_ops
        try:
            core = self._core()
            tokens = self._ids(6, seed=91)
            core.prefill(torch.tensor(tokens, dtype=torch.int64))
            orchestrator = wire_decode(core)
            self.assertIsNotNone(core.decode_impl)
            orchestrator.add_request(0, tokens)
            output = core.decode(torch.tensor([5], dtype=torch.int64))
            self.assertEqual(tuple(output.hidden.shape), (1, self.geometry.hidden_size))
            self.assertTrue(torch.isfinite(output.hidden).all())
            # The orchestrator shares the core's cache: state advanced by 1.
            self.assertEqual(orchestrator.requests[0].seq_len, 7)
        finally:
            if saved is None:
                sys.modules.pop("rtp_llm.ops", None)
            else:
                sys.modules["rtp_llm.ops"] = saved


if __name__ == "__main__":
    unittest.main()
