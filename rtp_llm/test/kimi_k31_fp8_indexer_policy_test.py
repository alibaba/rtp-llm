"""Indexer-only precision selection and replicated weight-storage contracts."""

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import torch

from rtp_llm.config.dsa_config import DsaConfig
from rtp_llm.models.kimi_k3.kimi_k31 import KimiK31ModelConfig
from rtp_llm.models.kimi_k3.fp8_weight import KimiK31LoadIndexerFp8Weight
from rtp_llm.ops import DataType
from rtp_llm.utils.model_weight import W


@dataclass(frozen=True)
class Runtime:
    dsa: DsaConfig


def config(kind="kimi_k31"):
    result = KimiK31ModelConfig()
    result.model_type = kind
    result.is_mtp = kind == "kimi_k31_mtp"
    result.config_dtype = "bf16"
    result.k3_runtime_config = Runtime(DsaConfig(
        layers=(3,), index_num_heads=32, index_head_dim=128, index_topk=2048,
        index_rope_dim=0, index_k_norm_type="layernorm", indexer_use_fp8=kind != "kimi_k31_mtp",
        index_topk_freq=1, index_skip_topk_offset=1,
    ))
    result.init_precision_config(None, "bf16")
    return result


class KimiK31Fp8IndexerPolicyTest(unittest.TestCase):
    def test_target_runtime_forces_independent_fp8_policy(self):
        cfg = config()
        self.assertTrue(cfg.k3_runtime_config.dsa.indexer_use_fp8)
        self.assertEqual(cfg.indexer_projection_quant_config.get_method(), "FP8_PER_BLOCK")
        self.assertEqual(cfg.k3_runtime_config.dsa.index_cache_entry_bytes, 132)

    def test_target_override_does_not_quantize_sfa_kda_or_a_separate_draft(self):
        target, draft = config(), config("kimi_k31_mtp")
        target.apply_override_args('{"indexer_use_fp8":true}')
        self.assertTrue(target.k3_runtime_config.dsa.indexer_use_fp8)
        self.assertEqual(target.indexer_projection_quant_config.get_method(), "FP8_PER_BLOCK")
        self.assertEqual(target.k3_runtime_config.dsa.index_cache_entry_bytes, 132)
        self.assertIsNone(target.attention_projection_quant_config)
        self.assertFalse(target.attn_config.mla_fp8_compute)
        self.assertEqual(target.data_type, DataType.TYPE_BF16)
        self.assertFalse(draft.k3_runtime_config.dsa.indexer_use_fp8)
        self.assertIsNone(draft.indexer_projection_quant_config)

    def test_draft_rejects_fp8_indexer(self):
        draft = config("kimi_k31_mtp")
        before = draft.k3_runtime_config
        with self.assertRaisesRegex(ValueError, "draft Indexer must remain BF16"):
            draft.apply_override_args('{"indexer_use_fp8":true}')
        self.assertIs(draft.k3_runtime_config, before)

    def test_override_requires_boolean(self):
        for value in ("1", "0", '"true"', "null", "[]"):
            cfg = config()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                cfg.apply_override_args('{"indexer_use_fp8":'+value+'}')
            self.assertTrue(cfg.k3_runtime_config.dsa.indexer_use_fp8)

    def test_target_rejects_bf16_without_changing_policy(self):
        cfg = config()
        before = cfg.k3_runtime_config
        with self.assertRaisesRegex(ValueError, "requires FP8 Indexer"):
            cfg.apply_override_args('{"indexer_use_fp8":false}')
        self.assertIs(cfg.k3_runtime_config, before)
        self.assertTrue(cfg.k3_runtime_config.dsa.indexer_use_fp8)

    def test_rejecting_fp8_sfa_does_not_partially_change_indexer_precision(self):
        cfg = config()
        before = cfg.k3_runtime_config
        cfg.attn_config.mla_fp8_compute = True
        with self.assertRaisesRegex(ValueError, "independent BF16 SFA/KDA"):
            cfg.apply_override_args('{"indexer_use_fp8":true}')
        self.assertIs(cfg.k3_runtime_config, before)
        self.assertEqual(cfg.indexer_projection_quant_config.get_method(), "FP8_PER_BLOCK")

    def test_static_quantization_scope_is_only_indexer_qb_and_k(self):
        self.assertEqual(KimiK31LoadIndexerFp8Weight.w8a8_weight_list, {
            W.mla_indexer_qb_w: W.mla_indexer_qb_s,
            W.mla_indexer_k_w: W.mla_indexer_k_s,
        })
        self.assertNotIn(W.mla_fusedqkrope_w, KimiK31LoadIndexerFp8Weight.w8a8_weight_list)
        self.assertNotIn(W.mla_indexer_weights_proj_w, KimiK31LoadIndexerFp8Weight.w8a8_weight_list)

    def test_all_indexer_heads_and_scales_are_replicated_with_owned_storage(self):
        policy = object.__new__(KimiK31LoadIndexerFp8Weight)
        policy.kernel = SimpleNamespace(name=W.mla_indexer_qb_w)
        for parallel in (4, 8):
            for rank in range(parallel):
                weight = torch.zeros(4096, 128, dtype=torch.float8_e4m3fn)
                scales = torch.ones(32, 1)
                result = policy._split({W.mla_indexer_qb_w:weight,W.mla_indexer_qb_s:scales},
                                       SimpleNamespace(tp_size=parallel,tp_rank=rank))
                self.assertEqual(result[W.mla_indexer_qb_w].shape, (4096, 128))
                self.assertEqual(result[W.mla_indexer_qb_s].shape, (32, 1))
                self.assertNotEqual(result[W.mla_indexer_qb_w].data_ptr(), weight.data_ptr())
                self.assertNotEqual(result[W.mla_indexer_qb_s].data_ptr(), scales.data_ptr())
                torch.testing.assert_close(result[W.mla_indexer_qb_s], scales, rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
