"""Qwen4Exp QSA prepare-once host validation tests (CPU).

Covers the host-side de-duplication round: the per-layer python/aten
validation of step-invariant metadata (mirror ``any``/``tolist``/list
comparisons, the nonzero-prefix gate, the fused-writer step-geometry guard)
moves to one frozen facts pass per step plus fingerprint-only step-plan hit
checks.  Per-layer code re-checks only fingerprints and the per-layer tensors.

All cases run on CPU tensors with SimpleNamespace contexts, mirroring the
fixture style of qsa_step_plan_test.  The strict flag is pinned per test
because the frozen paths only exist with validation demoted; strict mode must
keep the legacy per-layer reads (covered by the strict-bypass cases).
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_step_plan
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.ops import RopeStyle


def _inputs(prefixes, lengths):
    batch = len(lengths)
    input_lengths = torch.tensor(lengths, dtype=torch.int32)
    prefix_lengths = torch.tensor(prefixes, dtype=torch.int32)
    cu = torch.zeros(batch + 1, dtype=torch.int32)
    cu[1:] = input_lengths.cumsum(0)
    cu_kv = torch.zeros(batch + 1, dtype=torch.int32)
    cu_kv[1:] = (prefix_lengths + input_lengths).cumsum(0)
    positions = []
    for prefix, length in zip(prefixes, lengths):
        positions.extend([pos, pos, pos] for pos in range(prefix, prefix + length))
    kv_table = torch.arange(1, batch * 3 + 1, dtype=torch.int32).view(batch, 3)
    state_table = kv_table.clone()
    return lambda: SimpleNamespace(
        is_prefill=True,
        is_target_verify=False,
        is_cuda_graph=False,
        is_s_padded=False,
        context_parallel_info=None,
        cache_store_inputs=None,
        input_lengths=input_lengths,
        prefix_lengths=prefix_lengths,
        sequence_lengths=torch.empty(0, dtype=torch.int32),
        cu_seqlens_device=cu,
        cu_kv_seqlens_device=cu_kv,
        combo_position_ids=torch.tensor(positions, dtype=torch.int32).reshape(-1),
        kv_cache_kernel_block_id_device=kv_table,
        kv_cache_block_id_device=state_table,
    )


class QsaPrepareOnceValidationTest(unittest.TestCase):
    D = 4
    RATIO = 4
    PAGE = 8

    def setUp(self):
        qsa_step_plan.clear_step_plan_cache()
        strict_off = patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        )
        strict_off.start()
        self.addCleanup(strict_off.stop)
        self.stats0 = qsa_step_plan.step_plan_cache_stats()
        self.rope = SimpleNamespace(
            style=RopeStyle.Mrope,
            index_factor=3,
            dim=self.D,
            mrope_dim1=1,
            mrope_dim2=1,
            mrope_dim3=0,
            mrope_interleaved=True,
            indexer_is_neox_style=True,
            base=10_000.0,
        )
        self.indexer = SimpleNamespace(
            head_dim=self.D,
            n_heads=2,
            compress_ratio=self.RATIO,
            token_budget=8,
            k_norm_gamma=torch.rand(self.D),
            norm_eps=1e-6,
        )

    def _context(self, prefixes, lengths, *, layer=0):
        factory = _inputs(prefixes, lengths)
        batch = len(lengths)
        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=factory(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=torch.zeros(
                    8 + layer, (self.PAGE // self.RATIO) * self.D * 2, dtype=torch.uint8
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_kv_inputs=factory(),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=torch.zeros(
                    8 + layer, 2 * self.RATIO * self.D, dtype=torch.float32
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_state_inputs=factory(),
            is_mtp_draft=False,
        )

    def _geometry(self, context, token_count):
        return context._draft_incremental_prefill_geometry(
            indexer=self.indexer,
            token_count=token_count,
            device=torch.device("cpu"),
            rope_config=self.rope,
            draft=False,
            draft_prefix_reuse=False,
        )

    def _fact_computes(self):
        return qsa_step_plan.step_plan_cache_stats()["fact_computes"]

    # ---- frozen mirror facts ----

    def test_mirror_facts_frozen_once_per_step(self):
        context = self._context([8, 0], [8, 4])
        inputs = context.main_inputs
        facts = qsa_step_plan.prefill_mirror_facts(inputs)
        self.assertIsNotNone(facts)
        self.assertEqual(facts.lengths, [8, 4])
        self.assertEqual(facts.prefixes, [8, 0])
        self.assertEqual(facts.lengths_sum, 12)
        self.assertTrue(facts.lengths_positive)
        self.assertTrue(facts.prefixes_nonneg)
        computes_after_build = self._fact_computes()
        # Simulated per-layer readers of one step: cached, no recompute.
        for _ in range(13):
            again = qsa_step_plan.prefill_mirror_facts(inputs)
            self.assertIs(again, facts)
        self.assertEqual(self._fact_computes(), computes_after_build)
        # In-place prefix edits bump the version and recompute the facts.
        inputs.prefix_lengths[0] = 16
        refreshed = qsa_step_plan.prefill_mirror_facts(inputs)
        self.assertIsNot(refreshed, facts)
        self.assertEqual(refreshed.prefixes, [16, 0])
        self.assertGreater(self._fact_computes(), computes_after_build)

    def test_has_nonzero_prefix_gate_frozen(self):
        for prefixes, expected in (
            ([0, 0], False),
            ([8, 0], True),
            ([0], False),
        ):
            with self.subTest(prefixes=prefixes):
                qsa_step_plan.clear_step_plan_cache()
                context = self._context(prefixes, [4] * len(prefixes))
                legacy = bool(context.main_inputs.prefix_lengths.numel()) and bool(
                    torch.any(context.main_inputs.prefix_lengths != 0).item()
                )
                self.assertEqual(legacy, expected)
                self.assertEqual(
                    qsa_step_plan.has_nonzero_prefill_prefix(context.main_inputs),
                    expected,
                )
                computes = self._fact_computes()
                # Repeated per-layer gate reads reuse the frozen facts.
                for _ in range(4):
                    self.assertEqual(
                        qsa_step_plan.has_nonzero_prefill_prefix(context.main_inputs),
                        expected,
                    )
                self.assertEqual(self._fact_computes(), computes)

    # ---- fingerprint-only step-plan hits ----

    def test_step_plan_hits_do_not_reread_mirror_values(self):
        context = self._context([8, 0], [8, 4])
        plan = qsa_step_plan.get_prefill_step_plan(
            context,
            indexer=self.indexer,
            token_count=12,
            device=torch.device("cpu"),
            rope_config=self.rope,
            draft=False,
            draft_prefix_reuse=False,
        )
        self.assertEqual(len(plan.lengths), 2)
        # Every later layer hits the plan without reading any mirror values.
        with patch.object(
            qsa_step_plan,
            "host_input_mirror",
            side_effect=AssertionError("per-layer mirror value read"),
        ):
            for _ in range(13):
                hit = qsa_step_plan.get_prefill_step_plan(
                    context,
                    indexer=self.indexer,
                    token_count=12,
                    device=torch.device("cpu"),
                    rope_config=self.rope,
                    draft=False,
                    draft_prefix_reuse=False,
                )
                self.assertIs(hit, plan)

    def test_prefill_step_plan_hit_helper(self):
        context = self._context([8, 0], [8, 4])
        plan = qsa_step_plan.get_prefill_step_plan(
            context,
            indexer=self.indexer,
            token_count=12,
            device=torch.device("cpu"),
            rope_config=self.rope,
            draft=False,
            draft_prefix_reuse=False,
        )
        inputs = context.main_inputs
        self.assertIs(
            qsa_step_plan.prefill_step_plan_hit(inputs, 12, batch_size=2), plan
        )
        self.assertIsNone(qsa_step_plan.prefill_step_plan_hit(inputs, 11))
        self.assertIsNone(
            qsa_step_plan.prefill_step_plan_hit(
                inputs, 12, device=torch.device("meta")
            )
        )
        # In-place length edits invalidate the fingerprint hit.
        inputs.input_lengths[0] = 4
        self.assertIsNone(qsa_step_plan.prefill_step_plan_hit(inputs, 12))
        with patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}
        ):
            self.assertIsNone(qsa_step_plan.prefill_step_plan_hit(inputs, 12))

    # ---- fused writer step-geometry gate ----

    def test_prefix_reuse_fused_geometry_gate(self):
        gate = qsa_step_plan.prefix_reuse_fused_geometry_supported
        supported = dict(
            batch_size=2,
            head_dim=128,
            head_num=8,
            ratio=4,
            lengths=[129, 7],
            prefixes=[128, 128],
            kv_tokens_per_block=128,
            state_tokens_per_block=128,
        )
        self.assertTrue(gate(**supported))
        # Every step-geometry precondition the writer's guard checked per
        # layer now fails the gate once per step instead.
        for field, value in (
            ("head_dim", 64),
            ("ratio", 8),
            ("kv_tokens_per_block", 64),
            ("state_tokens_per_block", 64),
            ("head_num", 70000),
        ):
            with self.subTest(field=field):
                declined = dict(supported)
                declined[field] = value
                self.assertFalse(gate(**declined))
        self.assertFalse(gate(**{**supported, "prefixes": [100, 128]}))
        self.assertFalse(gate(**{**supported, "lengths": [70000, 7]}))
        self.assertFalse(gate(**{**supported, "batch_size": 0}))

    def test_step_plan_carries_fused_geometry_gate(self):
        # The released deployment geometry (head_dim 128, 128-token pages)
        # is built with a full-geometry fixture in the triton integration
        # tests; here the small fixture must gate OFF the fused writer.
        plan = qsa_step_plan.get_prefill_step_plan(
            context=self._context([8, 0], [8, 4]),
            indexer=self.indexer,
            token_count=12,
            device=torch.device("cpu"),
            rope_config=self.rope,
            draft=False,
            draft_prefix_reuse=False,
        )
        self.assertFalse(plan.prefix_reuse_fused_supported)


if __name__ == "__main__":
    unittest.main()
