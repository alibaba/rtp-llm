"""Qwen4ExpPrefillStepPlan cache behavior tests (CPU).

Covers what the P2 frozen per-step plans add on top of the legacy per-layer
rebuild: cross-layer reuse (hit), the (id(main_inputs), token_count, mode)
key, in-place mutation invalidation, the strict-validation bypass and the
per-step selection constants the selection/FMHA consumers read. All cases
run on CPU tensors with SimpleNamespace contexts, mirroring the fixture
style of qsa_runtime_test.

Entry fidelity: the tests call the geometry seam exactly the way the
production selection router does (qsa_context.select_*_prefill_tokens ->
qsa_prefill._incremental_prefill_selection -> _draft_incremental_prefill_
geometry) with the same mode flags, the direct-call style qsa_runtime_test
uses; the full selection entry itself requires CUDA q/raw keys and mutates
the side pools, which these cache-semantics cases deliberately avoid. The
strict flag is pinned per test because the asserted cache behavior only
exists with validation demoted.
"""

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


class Qwen4ExpStepPlanTest(unittest.TestCase):
    D = 4
    RATIO = 4
    PAGE = 8

    def setUp(self):
        qsa_step_plan.clear_step_plan_cache()
        # Hermetic strict flag: the cache under test only exists with the
        # production default (validation demoted). An ambient
        # RTP_LLM_QSA_STRICT_VALIDATION=1 (e.g. exported for a strict bench
        # run) must not flip these assertions; the strict-bypass test
        # overrides this to "1" itself.
        strict_off = patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        )
        strict_off.start()
        self.addCleanup(strict_off.stop)
        self.stats0 = qsa_step_plan.step_plan_cache_stats()
        self.rope = SimpleNamespace(
            style=RopeStyle.Mrope,
            index_factor=3,
            dim=4,
            mrope_dim1=1,
            mrope_dim2=1,
            mrope_dim3=0,
            mrope_interleaved=True,
            indexer_is_neox_style=True,
            base=10_000.0,
        )
        self.base_rope = SimpleNamespace(
            style=RopeStyle.Base,
            index_factor=3,
            dim=4,
            indexer_is_neox_style=True,
            base=10_000.0,
            scale=1.0,
        )
        self.indexer = SimpleNamespace(
            head_dim=self.D,
            n_heads=2,
            compress_ratio=self.RATIO,
            token_budget=8,
            k_norm_gamma=torch.rand(self.D),
            norm_eps=1e-6,
        )

    def _stats_delta(self):
        stats = qsa_step_plan.step_plan_cache_stats()
        return {
            key: stats[key] - self.stats0[key]
            for key in ("hits", "misses", "invalidations")
        }

    def _mirror_readbacks(self):
        return qsa_step_plan.step_plan_cache_stats()["mirror_readbacks"]

    def _context(
        self, prefixes, lengths, *, shared=None, layer=0, is_mtp_draft=False
    ):
        """One per-layer context; ``shared`` reuses a previous step's inputs."""
        batch = len(lengths)
        input_lengths = torch.tensor(lengths, dtype=torch.int32)
        prefix_lengths = torch.tensor(prefixes, dtype=torch.int32)
        sequence_lengths = torch.empty(0, dtype=torch.int32)
        cu = torch.zeros(batch + 1, dtype=torch.int32)
        cu[1:] = input_lengths.cumsum(0)
        cu_kv = torch.zeros(batch + 1, dtype=torch.int32)
        cu_kv[1:] = (prefix_lengths + input_lengths).cumsum(0)
        positions = []
        for prefix, length in zip(prefixes, lengths):
            positions.extend(
                [pos, pos, pos] for pos in range(prefix, prefix + length)
            )
        position_ids = torch.tensor(positions, dtype=torch.int32).reshape(-1)
        # Both tables use in-range physical ids (< 8 pool blocks, id zero
        # reserved); the two pools may share the same logical->physical map.
        kv_table = torch.arange(1, batch * 3 + 1, dtype=torch.int32).view(batch, 3)
        state_table = kv_table.clone()

        def _inputs():
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu,
                cu_kv_seqlens_device=cu_kv,
                combo_position_ids=position_ids,
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=state_table,
            )

        main_inputs, kv_inputs, state_inputs = shared or (
            _inputs(),
            _inputs(),
            _inputs(),
        )
        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=main_inputs,
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                # One distinct pool per "layer" to prove the per-layer binding.
                kv_cache_base=torch.zeros(
                    8 + layer, (self.PAGE // self.RATIO) * self.D * 2, dtype=torch.uint8
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_kv_inputs=kv_inputs,
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=torch.zeros(
                    8 + layer, 2 * self.RATIO * self.D, dtype=torch.float32
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_state_inputs=state_inputs,
            is_mtp_draft=is_mtp_draft,
        )

    def _geometry(
        self,
        context,
        token_count,
        *,
        draft=False,
        draft_prefix_reuse=False,
        rope_config=None,
    ):
        """Enter the geometry exactly as the production selection router does.

        Production mode routing (qwen4_exp._qsa_selected_indices):
        target prefix reuse -> select_prefix_reuse_prefill_tokens ->
        _incremental_prefill_selection(draft=False) ->
        geometry(draft=False, draft_prefix_reuse=False)  [the default here];
        MTP post-rejection commit -> select_draft_incremental_prefill_tokens
        -> geometry(draft=True) on an is_mtp_draft context;
        MTP draft prefix hit -> select_draft_prefix_reuse_prefill_tokens ->
        geometry(draft=False, draft_prefix_reuse=True). The geometry default
        draft=True would instead require the explicit MTP draft context.
        """
        return context._draft_incremental_prefill_geometry(
            indexer=self.indexer,
            token_count=token_count,
            device=torch.device("cpu"),
            rope_config=rope_config if rope_config is not None else self.rope,
            draft=draft,
            draft_prefix_reuse=draft_prefix_reuse,
        )

    def test_second_layer_hits_the_frozen_plan_and_rebinds_its_pools(self):
        first = self._context([8], [5])
        plan_one = self._geometry(first, 5)
        second = self._context(
            [8],
            [5],
            shared=(
                first.main_inputs,
                first.indexer_kv_inputs,
                first.indexer_state_inputs,
            ),
            layer=1,
        )
        plan_two = self._geometry(second, 5)

        self.assertIs(plan_two.step, plan_one.step)
        self.assertIs(plan_two.cu_seqlens, plan_one.step.cu_seqlens)
        self.assertIs(plan_two.rope_cos, plan_one.step.rope_cos)
        # Per-layer pool views stay layer-local even on a plan hit.
        self.assertIsNot(plan_two.kv_pool, plan_one.kv_pool)
        self.assertIsNot(plan_two.state_pool, plan_one.state_pool)
        self.assertEqual(
            self._stats_delta(),
            {"hits": 1, "invalidations": 0, "misses": 1},
        )

    def test_inplace_position_edit_invalidates_the_entry_and_rebuilds(self):
        context = self._context([8], [5])
        plan_one = self._geometry(context, 5)
        # A value-preserving in-place write still bumps the tensor version.
        context.main_inputs.combo_position_ids.copy_(
            context.main_inputs.combo_position_ids
        )
        for inputs in (context.indexer_kv_inputs, context.indexer_state_inputs):
            inputs.combo_position_ids.copy_(inputs.combo_position_ids)
        plan_two = self._geometry(context, 5)

        self.assertIsNot(plan_two.step, plan_one.step)
        torch.testing.assert_close(plan_two.current_cos, plan_one.current_cos)
        torch.testing.assert_close(plan_two.rope_cos, plan_one.rope_cos)
        self.assertEqual(
            self._stats_delta(),
            {"hits": 0, "invalidations": 1, "misses": 2},
        )

    def test_strict_validation_bypasses_the_cache(self):
        context = self._context([8], [5])
        with patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "1"}
        ):
            plan_one = self._geometry(context, 5)
            plan_two = self._geometry(context, 5)
        self.assertIsNot(plan_two.step, plan_one.step)
        # Strict mode rebuilt the full legacy validation chain twice.
        self.assertEqual(
            self._stats_delta(),
            {"hits": 0, "invalidations": 0, "misses": 2},
        )
        # The bypassed cache serves no lookups either.
        self.assertIsNone(
            qsa_step_plan.lookup_prefill_step_plan(context.main_inputs, 5)
        )

    def test_tensor_fingerprint_accepts_inference_tensors(self):
        """Real requests run under inference_mode; their tensors track no version.

        Regression: reading ``._version`` raised "Inference tensors do not
        track version counter" from the first real (inference_mode) prefill
        forward and terminated the worker.
        """
        with torch.inference_mode():
            inference_tensor = torch.arange(8, dtype=torch.int32)
        self.assertEqual(
            qsa_step_plan._tensor_fingerprint(inference_tensor),
            (inference_tensor.data_ptr(), -1),
        )
        normal = torch.arange(8, dtype=torch.int32)
        pointer, version = qsa_step_plan._tensor_fingerprint(normal)
        self.assertEqual(pointer, normal.data_ptr())
        self.assertEqual(version, normal._version)
        self.assertIsNone(qsa_step_plan._tensor_fingerprint(object()))

    def test_modes_key_separately_and_lookup_serves_the_impl_side(self):
        context = self._context([8], [5])
        plan = self._geometry(context, 5)
        cold = qsa_step_plan.get_cold_step_plan(
            self._context([0, 0], [5, 8]),
            indexer=self.indexer,
            token_count=13,
            device=torch.device("cpu"),
            rope_config=self.rope,
            rope_dtype=torch.bfloat16,
        )
        # MTP post-rejection commit: Base RoPE on an explicit draft context.
        draft = self._context([8], [3], is_mtp_draft=True)
        draft_plan = self._geometry(draft, 3, draft=True, rope_config=self.base_rope)
        self.assertEqual(plan.step.mode, "prefix_reuse")
        self.assertEqual(cold.mode, "cold_prefill")
        self.assertEqual(draft_plan.step.mode, "draft_incremental")
        self.assertIsNot(cold, plan.step)
        self.assertIsNot(draft_plan.step, plan.step)

        found = qsa_step_plan.lookup_prefill_step_plan(context.main_inputs, 5)
        self.assertIs(found, plan.step)
        self.assertEqual(found.kv_lens_rows[0].reshape(-1).tolist(), [9, 10, 11, 12, 13])
        self.assertEqual(found.cu_host, [0, 5])
        self.assertEqual(
            found.padded_kv_lens.shape,
            (1, 5),
        )
        # A cold plan has no incremental selection constants to offer.
        self.assertIsNone(cold.kv_lens_rows)

    def test_selection_padding_matches_the_legacy_row_map_loop(self):
        context = self._context([8, 16], [5, 8])
        plan = self._geometry(context, 13)
        step = plan.step
        lengths = [5, 8]
        max_rows = 8
        offsets = []
        valid_rows = []
        offset = 0
        for request_idx, length in enumerate(lengths):
            offsets.extend(offset + min(row, length - 1) for row in range(max_rows))
            valid_rows.extend(request_idx * max_rows + row for row in range(length))
            offset += length
        self.assertEqual(step.selection_row_map.tolist(), offsets)
        self.assertEqual(step.selection_valid_rows.tolist(), valid_rows)
        self.assertEqual(
            step.selection_valid.tolist(),
            [
                [row < 5 for row in range(max_rows)],
                [row < 8 for row in range(max_rows)],
            ],
        )
        self.assertEqual(
            step.padded_kv_lens.tolist(),
            [
                [8 + row + 1 if row < 5 else 0 for row in range(max_rows)],
                [16 + row + 1 if row < 8 else 0 for row in range(max_rows)],
            ],
        )

    def test_cold_plan_hits_within_one_step_and_keeps_the_rope_table(self):
        context = self._context([0, 0], [5, 8])
        cold_one = qsa_step_plan.get_cold_step_plan(
            context,
            indexer=self.indexer,
            token_count=13,
            device=torch.device("cpu"),
            rope_config=self.rope,
            rope_dtype=torch.bfloat16,
        )
        cold_two = qsa_step_plan.get_cold_step_plan(
            context,
            indexer=self.indexer,
            token_count=13,
            device=torch.device("cpu"),
            rope_config=self.rope,
            rope_dtype=torch.bfloat16,
        )
        self.assertIs(cold_two, cold_one)
        self.assertEqual(cold_one.cu_checked, True)
        self.assertEqual(cold_one.cu_host, [0, 5, 13])
        self.assertEqual(tuple(cold_one.rope_table_cos.shape), (2, 8, 4))
        self.assertEqual(
            self._stats_delta(),
            {"hits": 1, "invalidations": 0, "misses": 1},
        )

    def test_host_mirror_passes_host_sources_through(self):
        inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([4, 8], dtype=torch.int32)
        )
        readbacks0 = self._mirror_readbacks()
        mirror = qsa_step_plan.host_input_mirror(inputs, "prefix_lengths")
        self.assertIs(mirror, inputs.prefix_lengths)
        self.assertEqual(self._mirror_readbacks(), readbacks0)

    @unittest.skipIf(not torch.cuda.is_available(), "device mirror needs CUDA")
    def test_host_mirror_reads_device_source_once_per_step(self):
        inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([4, 8], dtype=torch.int32, device="cuda")
        )
        readbacks0 = self._mirror_readbacks()
        mirror_one = qsa_step_plan.host_input_mirror(inputs, "prefix_lengths")
        self.assertEqual(mirror_one.device.type, "cpu")
        self.assertEqual(mirror_one.tolist(), [4, 8])
        self.assertEqual(self._mirror_readbacks(), readbacks0 + 1)
        # The layers of one step share the single readback.
        mirror_two = qsa_step_plan.host_input_mirror(inputs, "prefix_lengths")
        self.assertIs(mirror_two, mirror_one)
        self.assertEqual(self._mirror_readbacks(), readbacks0 + 1)
        # A rebound tensor (a fresh step materialising new inputs) refreshes
        # the mirror even when the values are identical.
        inputs.prefix_lengths = torch.tensor(
            [4, 8], dtype=torch.int32, device="cuda"
        )
        mirror_three = qsa_step_plan.host_input_mirror(inputs, "prefix_lengths")
        self.assertIsNot(mirror_three, mirror_one)
        self.assertEqual(mirror_three.tolist(), [4, 8])
        self.assertEqual(self._mirror_readbacks(), readbacks0 + 2)
        # Missing attributes stay None (the caller keeps its error semantics).
        self.assertIsNone(
            qsa_step_plan.host_input_mirror(SimpleNamespace(), "prefix_lengths")
        )

    @unittest.skipIf(not torch.cuda.is_available(), "device mirror needs CUDA")
    def test_host_mirror_evicts_beyond_its_bound(self):
        readbacks0 = self._mirror_readbacks()
        holders = [
            SimpleNamespace(
                prefix_lengths=torch.tensor([index], dtype=torch.int32, device="cuda")
            )
            for index in range(qsa_step_plan._MIRROR_CACHE_LIMIT + 1)
        ]
        for holder in holders:
            qsa_step_plan.host_input_mirror(holder, "prefix_lengths")
        self.assertEqual(
            len(qsa_step_plan._MIRROR_CACHE), qsa_step_plan._MIRROR_CACHE_LIMIT
        )
        # The oldest entry was evicted: touching it again costs a readback.
        mirror = qsa_step_plan.host_input_mirror(holders[0], "prefix_lengths")
        self.assertEqual(mirror.tolist(), [0])
        self.assertEqual(
            self._mirror_readbacks(),
            readbacks0 + qsa_step_plan._MIRROR_CACHE_LIMIT + 2,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "device mirror needs CUDA")
    def test_step_plan_builds_and_hits_with_device_length_sources(self):
        # The RTP_LLM_DEVICE_INPUT seam: input_lengths/prefix_lengths live on
        # the device while the rest of the step keeps the CPU fixtures.  The
        # build reads both sources back once; the second layer's hit reuses
        # those mirrors without another readback.
        context = self._context([8], [5])
        lengths_cuda = context.main_inputs.input_lengths.to("cuda")
        prefixes_cuda = context.main_inputs.prefix_lengths.to("cuda")
        for inputs in (
            context.main_inputs,
            context.indexer_kv_inputs,
            context.indexer_state_inputs,
        ):
            inputs.input_lengths = lengths_cuda
            inputs.prefix_lengths = prefixes_cuda
        readbacks0 = self._mirror_readbacks()
        plan_one = self._geometry(context, 5)
        self.assertEqual(plan_one.step.lengths, [5])
        self.assertEqual(plan_one.step.prefix_values, [8])
        self.assertEqual(self._mirror_readbacks(), readbacks0 + 2)
        second = self._context(
            [8],
            [5],
            shared=(
                context.main_inputs,
                context.indexer_kv_inputs,
                context.indexer_state_inputs,
            ),
        )
        plan_two = self._geometry(second, 5)
        self.assertIs(plan_two.step, plan_one.step)
        self.assertEqual(
            self._stats_delta(),
            {"hits": 1, "invalidations": 0, "misses": 1},
        )
        self.assertEqual(self._mirror_readbacks(), readbacks0 + 2)


if __name__ == "__main__":
    unittest.main()
