import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _check_factory_scenario(scenario: str) -> None:
    from rtp_llm.config.kv_cache_config import KVCacheConfig
    from rtp_llm.config.model_args import ModelArgs
    from rtp_llm.config.py_config_modules import LoraConfig
    from rtp_llm.model_factory import ModelFactory
    from rtp_llm.model_factory_register import (
        _hf_architecture_2_ft,
        _model_factory,
        get_lazy_model_module_path,
    )
    from rtp_llm.ops import ProfilingDebugLoggingConfig

    check = unittest.TestCase()
    check.assertNotIn("rtp_llm.models.hy_v4", sys.modules)
    check.assertNotIn("hy_v4", _model_factory)
    check.assertNotIn("hy_v4_mtp", _model_factory)
    check.assertEqual(get_lazy_model_module_path("hy_v4"), "rtp_llm.models.hy_v4")
    check.assertEqual(_hf_architecture_2_ft["HYV4ForCausalLM"], "hy_v4")
    raw = {
        "architectures": ["HYV4ForCausalLM"],
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "vocab_size": 128,
        "max_position_embeddings": 4096,
        "dtype": "bfloat16",
        "num_attention_heads": 4,
        "q_lora_rank": 16,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 6,
        "qk_rope_head_dim": 2,
        "v_head_dim": 8,
        "index_head_dim": 4,
        "index_n_heads": 2,
        "index_topk": 16,
        "indexer_types": ["full", "shared", "full"],
        "mlp_layer_types": ["dense", "sparse", "sparse"],
        "intermediate_size": 48,
        "moe_intermediate_size": 12,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
    }
    if scenario == "nondefault":
        raw.update(
            rms_norm_eps=1e-5,
            hc_magnitude=1.5,
            hc_eps=2e-6,
            learnable_sink_init=-0.5,
            n_shared_experts=2,
            norm_topk_prob=False,
            routed_scaling_factor=1.25,
            swiglu_limit=7.0,
            rope_interleave=False,
            indexer_rope_interleave=False,
            rope_parameters={"rope_theta": 123456.0},
            indexer_types=["full", "full", "shared"],
            mlp_layer_types=["dense", "dense", "sparse"],
        )
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "config.json").write_text(json.dumps(raw))
        args = ModelArgs()
        args.model_type = "hy_v4"
        args.ckpt_path = directory
        args.tokenizer_path = directory
        if scenario == "nondefault":
            args.max_seq_len = 2048
            args.enable_fp32_lm_head = False
        config = ModelFactory.create_model_config(
            args, LoraConfig(), KVCacheConfig(), ProfilingDebugLoggingConfig()
        )
        check.assertIn("rtp_llm.models.hy_v4", sys.modules)
        check.assertEqual(ModelFactory.get_model_cls("hy_v4").__name__, "Hy4")
        check.assertEqual(ModelFactory.get_weight_cls("hy_v4").__name__, "Hy4Weight")
        check.assertEqual(config.model_name, "Hy4")
        check.assertEqual(config.model_type, "hy_v4")
        check.assertTrue(config.use_kvcache)
        check.assertTrue(config.attn_config.use_mla)
        check.assertTrue(config.attn_config.is_sparse)
        check.assertTrue(config.force_sparse_mla)
        check.assertTrue(config.enable_ihc)
        check.assertEqual(config.hc_mult, 4)
        check.assertTrue(config.gated_mla)
        check.assertEqual(config.gating_type, "elementwise")
        check.assertTrue(config.learnable_sink)
        check.assertEqual(config.indexer_scale_fmt, "ue8m0")
        check.assertFalse(config.indexer_use_hadamard)
        check.assertAlmostEqual(config.indexer_layernorm_eps, 1e-6)
        check.assertEqual(config.indexer_types, raw["indexer_types"])
        check.assertEqual(config.mlp_layer_types, raw["mlp_layer_types"])
        check.assertEqual(config.dense_inter_size, 48)
        check.assertEqual(config.moe_inter_size, 12)
        if scenario == "nondefault":
            check.assertEqual(config.max_seq_len, 2048)
            check.assertFalse(config.enable_fp32_lm_head)
            check.assertEqual(list(config.moe_layer_index), [2])
            check.assertEqual(config.inter_size, 24)
            check.assertAlmostEqual(config.layernorm_eps, 1e-5)
            check.assertAlmostEqual(config.hc_magnitude, 1.5)
            check.assertAlmostEqual(config.hc_eps, 2e-6)
            check.assertAlmostEqual(config.learnable_sink_init, -0.5)
            check.assertFalse(config.has_moe_norm)
            check.assertAlmostEqual(config.routed_scaling_factor, 1.25)
            check.assertAlmostEqual(config.swiglu_limit, 7.0)
            check.assertEqual(config.attn_config.rope_config.base, 123456.0)
            check.assertTrue(config.attn_config.rope_config.is_neox_style)
            check.assertTrue(config.attn_config.rope_config.indexer_is_neox_style)
        else:
            check.assertEqual(config.max_seq_len, 4096)
            check.assertTrue(config.enable_fp32_lm_head)
            check.assertEqual(list(config.moe_layer_index), [1, 2])
            check.assertEqual(config.inter_size, 12)
            check.assertAlmostEqual(config.hc_magnitude, 2.0)
            check.assertAlmostEqual(config.hc_eps, 1e-6)
            check.assertAlmostEqual(config.learnable_sink_init, 0.0)
            check.assertTrue(config.has_moe_norm)
            check.assertAlmostEqual(config.routed_scaling_factor, 2.827, places=5)
            check.assertAlmostEqual(config.swiglu_limit, 10.0)
            check.assertFalse(config.attn_config.rope_config.is_neox_style)
            check.assertFalse(config.attn_config.rope_config.indexer_is_neox_style)

        args.model_type = "hy_v4_mtp"
        draft = ModelFactory.create_model_config(
            args, LoraConfig(), KVCacheConfig(), ProfilingDebugLoggingConfig()
        )
        check.assertEqual(ModelFactory.get_model_cls("hy_v4_mtp").__name__, "Hy4Mtp")
        check.assertEqual(draft.model_name, "Hy4Mtp")
        check.assertEqual(draft.num_layers, 1)
        check.assertTrue(draft.is_mtp)
        check.assertFalse(draft.enable_ihc)
        check.assertEqual(draft.hc_mult, 1)
        check.assertEqual(list(draft.moe_layer_index), [0])
        check.assertEqual(draft.mlp_layer_types, ["sparse"])
        check.assertEqual(draft.indexer_types, ["full"])
        check.assertFalse(draft.index_share_for_mtp_iteration)
    print(f"PASS real HY4 factory lazy loading and {scenario} target/draft config")


class Hy4FactoryTest(unittest.TestCase):
    def test_lazy_factory_configuration_in_fresh_processes(self):
        for scenario in ("default", "nondefault"):
            with self.subTest(scenario=scenario):
                result = subprocess.run(
                    [sys.executable, __file__, "--factory-scenario", scenario],
                    env={**os.environ, "RTP_LLM_GLM5_CMP": "0"},
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + "\n" + result.stderr
                )
                self.assertIn(f"{scenario} target/draft config", result.stdout)
                print(result.stdout)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--factory-scenario":
        _check_factory_scenario(sys.argv[2])
    else:
        unittest.main()
