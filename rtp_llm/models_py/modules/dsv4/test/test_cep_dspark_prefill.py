"""Regressions of real CEP DSpARK config/placement/CP mapping boundaries.

No kernels or collectives are mocked: these tests do not claim GPU execution,
KV publication, end-to-end MTP parity, or cross-host qualification.
"""

import os
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.distributed.ep_stage_context import (
    resolve_dspark_prefill_opt_in,
    validate_pp_ep_target,
)
from rtp_llm.ops import CPRotateMethod, ParallelismConfig, RoleType, SpeculativeType
from rtp_llm.utils.model_weight import W


def proxy(rank=0, enabled=True):
    c = ParallelismConfig()
    c.pp_size = 2
    c.tp_size = c.ep_size = c.ffn_tp_size = 2
    c.world_size = c.local_world_size = 4
    c.world_rank = rank
    c.pp_rank = rank // 2
    c.tp_rank = c.ep_rank = c.ffn_tp_rank = rank % 2
    c.role_type = RoleType.PREFILL
    c.pp_stage_layer_counts = [22, 21]
    c.pp_ep_enabled = True
    c.pp_ep_backend = "fork_nccl_mxfp8"
    c.prefill_cp_config.method = CPRotateMethod.PREFILL_CP
    c.prefill_cp_config.prefill_cp_size = 2
    c.resolve_local_cp("deepseek_v4", enabled, False, False, enabled)
    return c


def draft_harness(c):
    from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import (
        DeepSeekV4DSparkModel,
    )

    model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
    torch.nn.Module.__init__(model)
    model.parallelism_config = c
    model.layer_num = 3
    model.tp_size, model.tp_rank = c.tp_size, c.tp_rank
    model.kv_cache = None
    model._configure_commit_cp()
    return model


class CEPDSparkPolicyTest(unittest.TestCase):
    def test_opt_in_is_algorithm_specific(self):
        with patch.dict(os.environ, {"DSV4_PP_EP_DSPARK_PREFILL": "1"}):
            self.assertTrue(
                resolve_dspark_prefill_opt_in(
                    "deepseek_v4", SpeculativeType.DSPARK, "deepseek_v4_dspark"
                )
            )
            for spec in (
                SpeculativeType.NONE,
                SpeculativeType.MTP,
                SpeculativeType.EAGLE,
            ):
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    resolve_dspark_prefill_opt_in(
                        "deepseek_v4", spec, "deepseek_v4_dspark"
                    )
            for model, draft in (
                ("other", "deepseek_v4_dspark"),
                ("deepseek_v4", "other"),
            ):
                with self.subTest(model=model, draft=draft), self.assertRaises(
                    ValueError
                ):
                    resolve_dspark_prefill_opt_in(model, SpeculativeType.DSPARK, draft)

    def test_opt_in_default_and_invalid_values(self):
        for value in ("", "0"):
            with patch.dict(os.environ, {"DSV4_PP_EP_DSPARK_PREFILL": value}):
                self.assertFalse(
                    resolve_dspark_prefill_opt_in("other", SpeculativeType.NONE, "")
                )
        for value in ("true", "2", "-1"):
            with patch.dict(
                os.environ, {"DSV4_PP_EP_DSPARK_PREFILL": value}
            ), self.assertRaises(ValueError):
                resolve_dspark_prefill_opt_in(
                    "deepseek_v4", SpeculativeType.DSPARK, "deepseek_v4_dspark"
                )

    def test_resolved_native_capability_roundtrips_all_stage_lanes(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                c = pickle.loads(pickle.dumps(proxy(rank)))
                self.assertTrue(c.dsv4_dspark_prefill_compat)
                self.assertTrue(c.local_cp_enabled())
                self.assertEqual(c.get_attn_tp_size(), 1)
                self.assertEqual(c.pp_stage_layer_counts, [22, 21])
                self.assertEqual(c.pp_rank, rank // 2)
                validate_pp_ep_target(
                    c,
                    hw_kernel_config=SimpleNamespace(
                        enable_cuda_graph=False, enable_native_cuda_graph=False
                    ),
                    is_sm120=True,
                    has_grouped_fp4=True,
                    is_speculative=True,
                )

    def test_ordinary_profile_cannot_admit_speculative_model(self):
        c = proxy(enabled=False)
        with self.assertRaisesRegex(ValueError, "speculative"):
            validate_pp_ep_target(
                c,
                hw_kernel_config=SimpleNamespace(
                    enable_cuda_graph=False, enable_native_cuda_graph=False
                ),
                is_sm120=True,
                has_grouped_fp4=True,
                is_speculative=True,
            )
        with self.assertRaises(ValueError):
            c.resolve_local_cp("deepseek_v4", True, False, False)
        self.assertFalse(c.dsv4_prefill_cp_compat)
        self.assertFalse(c.dsv4_dspark_prefill_compat)

    def test_only_last_stage_target_features_are_admitted(self):
        def setup(c, ids):
            sp = SimpleNamespace(gen_num_per_cycle=3, sp_dspark_mask_token_id=-1)
            target = SimpleNamespace(num_layers=43, capture_aux_hidden_layer_ids=None)
            draft = SimpleNamespace(
                dspark_noise_token_id=10,
                dspark_target_layer_ids=ids,
                dspark_markov_rank=256,
                vocab_size=100,
            )
            ModelFactory._setup_dspark_configs(sp, target, draft, parallelism_config=c)
            return target.capture_aux_hidden_layer_ids

        for rank in range(4):
            self.assertEqual(setup(proxy(rank), [40, 41, 42]), [40, 41, 42])
        with self.assertRaisesRegex(ValueError, "last stage"):
            setup(proxy(), [0, 41, 42])
        c = proxy()
        c.pp_stage_layer_counts = [42, 1]
        with self.assertRaisesRegex(ValueError, "last stage"):
            setup(c, [40, 41, 42])
        with self.assertRaisesRegex(ValueError, "opt-in"):
            setup(proxy(enabled=False), [40, 41, 42])


class CEPDSparkModelTest(unittest.TestCase):
    """CPU tensor assertions with the real CUDA-only model import closure."""

    def test_draft_keeps_three_layers_and_physical_stage_identity(self):
        for rank in (2, 3):
            model = draft_harness(proxy(rank))
            self.assertEqual(model.pp_layer_ids(), [0, 1, 2])
            self.assertEqual(model.pp_rank, 1)
            self.assertEqual(model.pp_size, 2)
            self.assertTrue(model._dspark_commit_cp_enabled)
            self.assertFalse(model._dspark_kv_cache_sharded)

    def test_commit_only_pp_draft_does_not_alias_missing_embedding(self):
        from rtp_llm.models.deepseek_v4 import DeepSeekV4, DeepSeekV4DSpark

        target = DeepSeekV4.__new__(DeepSeekV4)
        target.parallelism_config = proxy(2)
        cfg = SimpleNamespace(
            vocab_size=100, hidden_size=8, data_type="BF16", enable_fp32_lm_head=False
        )
        target.model_config = cfg
        self.assertEqual(
            DeepSeekV4DSpark.speculative_weight_alias_names(target, cfg), (W.lm_head,)
        )
        target.parallelism_config.pp_size = 1
        self.assertEqual(
            DeepSeekV4DSpark.speculative_weight_alias_names(target, cfg),
            (W.embedding, W.lm_head),
        )

    def test_pp_commit_descriptor_omits_embedding_but_retains_head_alias(self):
        from rtp_llm.models.deepseek_v4 import DeepSeekV4DSparkWeight, DeepSeekV4Weight

        weight = DeepSeekV4DSparkWeight.__new__(DeepSeekV4DSparkWeight)
        weight.role_type = RoleType.PREFILL
        weight.pp_size = 2
        names = [W.embedding, W.lm_head, W.v4_dspark_main_norm, W.v4_dspark_main_proj_w]
        info = SimpleNamespace(
            weights=[SimpleNamespace(name=n) for n in names], layer_weights=[]
        )
        with patch.object(DeepSeekV4Weight, "get_weight_info", return_value=info):
            actual = weight.get_weight_info()
        self.assertEqual([w.name for w in actual.weights], names[1:])

    def test_resolved_cep_commit_maps_real_zigzag_rows_and_padding(self):
        # Five real rows padded to eight: rank0 owns positions0,1,6,7;
        # rank1 owns2,3,4,5. No mocked CP context builder.
        cp_info = SimpleNamespace(
            prefill_qkv_padding_mask=torch.tensor(
                [1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool
            ),
            prefill_qkv_restore_indice=torch.tensor(
                [0, 1, 4, 5, 6, 7, 2, 3], dtype=torch.int32
            ),
            prefill_actual_input_lengths_cpu=torch.tensor([5], dtype=torch.int32),
            prefill_cp_chunk_lengths=torch.tensor([4], dtype=torch.int32),
        )
        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(
                context_parallel_info=cp_info,
                prefix_lengths=torch.tensor([10], dtype=torch.int32),
            )
        )
        found = []
        for rank, expected in ((2, [10, 11, -1, -1]), (3, [12, 13, 14, -1])):
            model = draft_harness(proxy(rank))
            self.assertFalse(model.parallelism_config.prefill_cp_config.is_enabled())
            req, positions, ctx = model.map_commit_rows(
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([5], dtype=torch.int32),
                torch.tensor([15], dtype=torch.int32),
                4,
                inputs,
            )
            self.assertIsNotNone(ctx)
            self.assertEqual(positions.tolist(), expected)
            self.assertEqual(req.tolist(), [0 if p >= 0 else -1 for p in expected])
            found.extend(p for p in positions.tolist() if p >= 0)
        self.assertEqual(sorted(found), list(range(10, 15)))

    def test_decode_remote_cp_does_not_split_commit_rows(self):
        c = proxy(2)
        c.role_type = RoleType.DECODE
        c.resolve_local_cp("deepseek_v4", True, False, False)
        model = draft_harness(c)
        self.assertFalse(model._dspark_commit_cp_enabled)
        self.assertFalse(model._dspark_kv_cache_sharded)

    def test_old_native_cp_still_enables_commit_mapping(self):
        c = ParallelismConfig()
        c.tp_size = 2
        c.role_type = RoleType.PREFILL
        c.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        self.assertTrue(draft_harness(c)._dspark_commit_cp_enabled)


if __name__ == "__main__":
    unittest.main()
