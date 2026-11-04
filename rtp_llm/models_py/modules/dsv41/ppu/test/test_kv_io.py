"""Fixture load/continue tests (kv_io.py, L3, plan section 7.5).

The offline P -> D contract: export the cache state after a prefill, reload
it into a fresh orchestrator, continue decode — and require the continuation
to match an uninterrupted run that never went through the round-trip.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.decode import (
    DecodeStepOrchestrator,
    cache_geometry_from_v41,
)
from rtp_llm.models_py.modules.dsv41.ppu.kv_io import (
    compare_against_uninterrupted,
    continue_decode,
    load_fixture_state,
)
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import run_prefill
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)


class FixtureContinueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_kvio_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=231)
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

    def _ids(self, count, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(
            0, self.geometry.vocab_size, (count,), generator=generator
        ).tolist()

    def _seeded_orchestrator(self, tokens, request_id=0):
        cache = PPUKVCache(geometry=cache_geometry_from_v41(self.geometry))
        result = run_prefill(
            self.bundle,
            self.ops,
            cache,
            torch.tensor(tokens, dtype=torch.int64),
            apply_lm_head=True,
        )
        orchestrator = DecodeStepOrchestrator(
            ops=self.ops, bundle=self.bundle, cache=cache
        )
        orchestrator.add_request(request_id, tokens)
        seed_token = int(result.logits[-1].argmax())
        return orchestrator, seed_token

    def test_load_fixture_continue_matches_uninterrupted(self):
        tokens = self._ids(9, seed=3)
        fixture_dir = Path(tempfile.mkdtemp(prefix="dsv41_fixture_"))
        try:
            reference, seed_token = self._seeded_orchestrator(tokens)
            # Export the reference's cache (the P -> D hand-off artifact).
            manifest = reference.cache.export_fixture(
                fixture_dir, meta={"source_cp": 1, "snapshot": "test"}
            )
            self.assertEqual(manifest["lengths"], {"0": len(tokens)})
            restored = load_fixture_state(
                fixture_dir,
                ops=self.ops,
                bundle=self.bundle,
                request_histories={0: tokens},
            )
            report = compare_against_uninterrupted(
                reference, restored, steps=4, seed_tokens={0: seed_token}
            )
            self.assertTrue(report["match"], msg=str(report))
            self.assertEqual(report["steps"], 4)
            # The restored cache really carries the prompt's pool state.
            regions = restored.cache.cache.valid_regions()
            # 9 prompt tokens + 4 continuation steps committed.
            self.assertEqual(regions["2:global_kv_2"], {"0": 13 // 2})
            self.assertEqual(regions["2:csa_state"], {"0": 13})
        finally:
            shutil.rmtree(fixture_dir, ignore_errors=True)

    def test_history_length_mismatch_is_refused(self):
        tokens = self._ids(8, seed=5)
        fixture_dir = Path(tempfile.mkdtemp(prefix="dsv41_fixture_"))
        try:
            orchestrator, _ = self._seeded_orchestrator(tokens)
            orchestrator.cache.export_fixture(fixture_dir)
            with self.assertRaises(ValueError):
                load_fixture_state(
                    fixture_dir,
                    ops=self.ops,
                    bundle=self.bundle,
                    request_histories={0: tokens + [1]},
                )
        finally:
            shutil.rmtree(fixture_dir, ignore_errors=True)

    def test_unknown_request_in_history_is_refused(self):
        tokens = self._ids(8, seed=7)
        fixture_dir = Path(tempfile.mkdtemp(prefix="dsv41_fixture_"))
        try:
            orchestrator, _ = self._seeded_orchestrator(tokens)
            orchestrator.cache.export_fixture(fixture_dir)
            with self.assertRaises(ValueError):
                load_fixture_state(
                    fixture_dir,
                    ops=self.ops,
                    bundle=self.bundle,
                    request_histories={0: tokens, 42: tokens},
                )
        finally:
            shutil.rmtree(fixture_dir, ignore_errors=True)

    def test_geometry_mismatch_is_refused(self):
        tokens = self._ids(8, seed=9)
        fixture_dir = Path(tempfile.mkdtemp(prefix="dsv41_fixture_"))
        try:
            orchestrator, _ = self._seeded_orchestrator(tokens)
            orchestrator.cache.export_fixture(fixture_dir)
            other_config = synthetic_config(num_layers=12)
            other_root = Path(tempfile.mkdtemp(prefix="dsv41_other_"))
            try:
                other_ckpt = write_synthetic_checkpoint(other_config, other_root)
                other_bundle = load_ppu_weights(other_config, other_ckpt)
                with self.assertRaises(ValueError):
                    load_fixture_state(
                        fixture_dir,
                        ops=self.ops,
                        bundle=other_bundle,
                        request_histories={0: tokens},
                    )
            finally:
                shutil.rmtree(other_root, ignore_errors=True)
        finally:
            shutil.rmtree(fixture_dir, ignore_errors=True)

    def test_continue_decode_runs_and_advances(self):
        tokens = self._ids(7, seed=11)
        orchestrator, seed_token = self._seeded_orchestrator(tokens)
        results = continue_decode(orchestrator, 3, {0: seed_token})
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(torch.isfinite(result.hidden).all())
            self.assertTrue(torch.isfinite(result.logits).all())
        self.assertEqual(orchestrator.requests[0].seq_len, 7 + 3)
        self.assertEqual(
            orchestrator.requests[0].history[-3:],
            [seed_token]
            + [int(results[0].logits[0].argmax()), int(results[1].logits[0].argmax())],
        )

    def test_continue_decode_requires_seed_tokens(self):
        tokens = self._ids(7, seed=13)
        orchestrator, _ = self._seeded_orchestrator(tokens)
        with self.assertRaises(RuntimeError):
            continue_decode(orchestrator, 2, {})

    def test_owner_filter_removes_foreign_pools_before_decode(self):
        cache = PPUKVCache(geometry=cache_geometry_from_v41(self.geometry))
        histories = {7: self._ids(9, seed=71), 42: self._ids(7, seed=72)}
        ids = torch.tensor(histories[7] + histories[42], dtype=torch.int64)
        prefill = run_prefill(
            self.bundle,
            self.ops,
            cache,
            ids,
            cu_seqlens=torch.tensor([0, 9, 16]),
            request_ids=torch.tensor([7] * 9 + [42] * 7),
            apply_lm_head=True,
        )
        fixture_dir = self.root / "multi-owner"
        cache.export_fixture(fixture_dir, meta={"source_cp": 8})
        restored = load_fixture_state(
            fixture_dir,
            ops=self.ops,
            bundle=self.bundle,
            dp_rank=1,
            dp_size=8,
            request_histories=histories,
            owner_map={7: 0, 42: 1},
        )
        self.assertEqual(set(restored.requests), {42})
        self.assertEqual(restored.cache.cache._request_tokens, {42: 7})
        self.assertEqual(
            restored.cache.cache.state_checksum(),
            cache.select_requests([42]).state_checksum(),
        )
        result = continue_decode(restored, 3, {42: int(prefill.logits[-1].argmax())})
        self.assertEqual(len(result), 3)
        self.assertEqual(restored.requests[42].seq_len, 10)


if __name__ == "__main__":
    unittest.main()
