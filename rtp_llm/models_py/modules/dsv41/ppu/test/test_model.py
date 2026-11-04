"""Tests for the PPU model classes (model.py, L2 deliverable).

``PPUModelCore`` construction from the synthetic checkpoint path and forward
(prefill) behavior: finite hidden of the right shape across token and request
counts, continuation prefill, and the fail-loud decode seam. The
``get_backend()`` integration is covered by ``test_backend.py`` and is not
duplicated here.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.model import PPUModelCore
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    PPUWeightBundle,
    V41Geometry,
    load_ppu_weights,
    synthetic_config,
    write_synthetic_checkpoint,
)


class PPUModelCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = synthetic_config()
        cls.root = Path(tempfile.mkdtemp(prefix="dsv41_m1_"))
        cls.checkpoint = write_synthetic_checkpoint(cls.config, cls.root, seed=42)
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

    def _core(self) -> PPUModelCore:
        return PPUModelCore(self.config, self.bundle, self.ops)

    def _ids(self, tokens, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.randint(
            0, self.geometry.vocab_size, (tokens,), generator=generator
        )

    def test_construction_from_synthetic_path(self):
        core = self._core()
        self.assertIsInstance(core.weights, PPUWeightBundle)
        self.assertIs(core.ops, self.ops)
        self.assertIsInstance(core.cache, PPUKVCache)
        self.assertEqual(core.geometry.num_layers, 8)
        self.assertIsNone(core.decode_impl)
        self.assertEqual(core.device.type, "cpu")

    def test_forward_shapes_and_finiteness_across_token_counts(self):
        # tokens=1 covers the ratio-2 producer's 0-entry path (fixed in
        # _quant.py; regression test in test__quant.py).
        for tokens in (1, 2, 33, 129):
            with self.subTest(tokens=tokens):
                core = self._core()
                result = core.prefill(
                    self._ids(tokens, seed=tokens), apply_lm_head=True
                )
                self.assertEqual(
                    tuple(result.hidden.shape),
                    (tokens, self.geometry.hidden_size),
                )
                self.assertTrue(torch.isfinite(result.hidden).all())
                self.assertEqual(
                    tuple(result.logits.shape), (tokens, self.geometry.vocab_size)
                )
                self.assertTrue(torch.isfinite(result.logits).all())
                self.assertEqual(set(result.slots), {0})
                self.assertIs(result.cache, core.cache)

    def test_forward_across_request_counts(self):
        for requests in (1, 2, 3):
            with self.subTest(requests=requests):
                tokens = 20 * requests
                core = self._core()
                cu_seqlens = torch.arange(0, tokens + 1, 20, dtype=torch.int64)
                result = core.prefill(
                    self._ids(tokens, seed=requests),
                    cu_seqlens=cu_seqlens,
                    apply_lm_head=True,
                )
                self.assertEqual(
                    tuple(result.hidden.shape), (tokens, self.geometry.hidden_size)
                )
                self.assertTrue(torch.isfinite(result.hidden).all())
                self.assertTrue(torch.isfinite(result.logits).all())
                self.assertEqual(set(result.slots), set(range(requests)))
                # Every request's own pools track its own length.
                regions = core.cache.valid_regions()
                for request in range(requests):
                    self.assertEqual(regions["0:swa_kv"][str(request)], 20)
                    self.assertEqual(regions["2:global_kv_2"][str(request)], 10)
                    self.assertEqual(regions["6:global_kv_1"][str(request)], 20)

    def test_continuation_prefill_extends_tracked_lengths(self):
        core = self._core()
        first = core.prefill(self._ids(5, seed=1))
        self.assertEqual(core.cache.request_length(0), 5)
        second = core.prefill(self._ids(4, seed=2))
        self.assertEqual(core.cache.request_length(0), 9)
        for result, tokens in ((first, 5), (second, 4)):
            self.assertEqual(tuple(result.hidden.shape), (tokens, 64))
            self.assertTrue(torch.isfinite(result.hidden).all())
        regions = core.cache.valid_regions()
        self.assertEqual(regions["2:global_kv_2"]["0"], 9 // 2)
        self.assertEqual(regions["6:global_kv_1"]["0"], 9)
        self.assertEqual(regions["0:swa_kv"]["0"], 9)

    def test_inconsistent_positions_fail_loud(self):
        core = self._core()
        positions = torch.tensor([3, 4, 5], dtype=torch.int64)  # cache tracks 0
        with self.assertRaises(ValueError):
            core.prefill(self._ids(3, seed=3), positions=positions)

    def test_positions_override_must_be_consistent(self):
        core = self._core()
        result = core.prefill(
            self._ids(4, seed=4), positions=torch.arange(4, dtype=torch.int64)
        )
        self.assertEqual(tuple(result.hidden.shape), (4, 64))

    def test_decode_seam_fails_loud_until_wired(self):
        core = self._core()
        with self.assertRaises(NotImplementedError) as ctx:
            core.decode(torch.zeros(1, dtype=torch.int64))
        self.assertIn("decode", str(ctx.exception))

    def test_decode_seam_delegates_once_wired(self):
        core = self._core()
        sentinel = object()
        core.decode_impl = lambda core_self, *a, **kw: sentinel
        self.assertIs(core.decode(1, 2, x=3), sentinel)

    def test_apply_lm_head_requires_lm_head_weight(self):
        bundle = self.bundle
        saved = bundle.lm_head
        bundle.lm_head = None
        try:
            core = self._core()
            with self.assertRaises(ValueError):
                core.prefill(self._ids(3, seed=5), apply_lm_head=True)
        finally:
            bundle.lm_head = saved

    def test_hidden_is_pre_lm_head_normed(self):
        core = self._core()
        result = core.prefill(self._ids(8, seed=6), apply_lm_head=True)
        logits = torch.nn.functional.linear(result.hidden, self.bundle.lm_head)
        torch.testing.assert_close(result.logits, logits.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
