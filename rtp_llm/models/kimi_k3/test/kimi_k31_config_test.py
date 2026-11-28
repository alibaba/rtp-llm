import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.py_config_modules import LoraConfig
from rtp_llm.config.py_config_modules import QuantizationConfig as QuantizationArgs
from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    ModelOptFp4Config,
    init_quant_config,
)
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import (
    _hf_architecture_2_ft,
    get_lazy_model_module_path,
)
from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.ops import (
    CPRotateMethod,
    DataType,
    KvCacheDataType,
    KVCacheSpecType,
    ParallelismConfig,
    ProfilingDebugLoggingConfig,
    RoleType,
    TaskType,
    TpCacheTransferMode,
)


def checkpoint_config(model_type="kimi_k31", *, draft=False):
    # Only configuration is materialized; this test never creates fake weights.
    text = {
        "model_type": "kimi_linear",
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "hidden_size": 256,
        "vocab_size": 256,
        "max_position_embeddings": 1024,
        "intermediate_size": 128,
        "num_attention_heads": 8,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "moe_intermediate_size": 128,
        "routed_expert_hidden_size": 128,
        "num_shared_experts": 1,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 2,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "linear_attn_config": {
            "num_heads": 8,
            "head_dim": 128,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
            "kda_layers": [1],
            "full_attn_layers": [2, 3] if draft else [2],
        },
        "quantization_config": {
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {"weights": {"num_bits": 4, "group_size": 32}}
            },
        },
    }
    if model_type == "kimi_k31":
        text["dsa_config"] = {
            "dsa_layers": [2],
            "index_num_heads": 32,
            "index_head_dim": 128,
            "index_topk": 512,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "indexer_use_fp8": True,
            "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        }
    return {"model_type": model_type, "text_config": text}


class KimiK31ConfigTest(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB", None)
        os.environ["RTP_LLM_FP4_GEMM_BACKEND"] = "cutlass"

    def build_config(
        self,
        model_type="kimi_k31",
        *,
        fp8_cache=False,
        override=None,
        quantization="",
        **cli_options
    ):
        draft = model_type.endswith("_mtp")
        source_type = "kimi_k3" if draft else model_type
        checkpoint = checkpoint_config(source_type, draft=draft)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        Path(temporary.name, "config.json").write_text(json.dumps(checkpoint))
        args = ModelArgs()
        args.model_type = model_type
        args.ckpt_path = args.tokenizer_path = temporary.name
        args.task_type = TaskType.LANGUAGE_MODEL
        args.act_type = "BF16"
        args.json_model_override_args = json.dumps(override or {})
        for name, value in cli_options.items():
            setattr(args, name, value)
        cache = KVCacheConfig()
        cache.fp8_kv_cache = int(fp8_cache)
        cache.seq_size_per_block = cache.kernel_seq_size_per_block = 128
        quantization_args = QuantizationArgs()
        quantization_args.quantization = quantization
        return ModelFactory.create_model_config(
            args,
            LoraConfig(),
            cache,
            ProfilingDebugLoggingConfig(),
            quantization_config=quantization_args,
        )

    def test_factory_and_architecture_registration_select_native_draft(self):
        self.assertEqual(
            get_lazy_model_module_path("kimi_k31"),
            "rtp_llm.models.kimi_k3.kimi_k31",
        )
        self.assertEqual(
            get_lazy_model_module_path("kimi_k31_mtp"),
            "rtp_llm.models.kimi_k3.kimi_k31",
        )
        target_class = ModelFactory.get_model_cls("kimi_k31")
        draft_class = ModelFactory.get_model_cls("kimi_k31_mtp")
        from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3Mtp
        from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3MtpWeight
        from rtp_llm.models.kimi_k3.kimi_k31 import KimiK31, KimiK31Mtp
        from rtp_llm.models.kimi_k3.kimi_k31_weight import KimiK31Weight

        self.assertIs(target_class, KimiK31)
        self.assertIs(draft_class, KimiK31Mtp)
        self.assertTrue(issubclass(draft_class, KimiK3Mtp))
        self.assertIs(ModelFactory.get_weight_cls("kimi_k31"), KimiK31Weight)
        self.assertIs(ModelFactory.get_weight_cls("kimi_k31_mtp"), KimiK3MtpWeight)
        self.assertEqual(
            _hf_architecture_2_ft["KimiK31ForConditionalGeneration"], "kimi_k31"
        )
        self.assertEqual(_hf_architecture_2_ft["KimiK31MTPModel"], "kimi_k31_mtp")
        self.assertIs(
            get_multimodal_mixin_cls("kimi_k31"),
            get_multimodal_mixin_cls("kimi_k3"),
        )

    def test_fp8_cache_retains_bf16_sfa_and_independent_indexer_descriptor(self):
        for fp8_cache in (False, True):
            with self.subTest(fp8_cache=fp8_cache):
                config = self.build_config(fp8_cache=fp8_cache)
                self.assertEqual(config.data_type, DataType.TYPE_BF16)
                self.assertFalse(config.attn_config.mla_fp8_compute)
                self.assertEqual(
                    config.attn_config.kv_cache_dtype,
                    KvCacheDataType.FP8 if fp8_cache else KvCacheDataType.BASE,
                )
                self.assertIsNotNone(config.indexer_projection_quant_config)
                self.assertTrue(config.use_page_rr_kv_cache)
                linear = config.kv_cache_spec_descs[0]
                self.assertEqual([desc.tag for desc in linear], ["linear"])
                self.assertEqual(
                    linear[0].tp_transfer_mode, TpCacheTransferMode.LINEAR_QKV
                )
                self.assertEqual(linear[0].dtype, DataType.TYPE_BF16)
                groups = {desc.tag: desc for desc in config.kv_cache_spec_descs[1]}
                self.assertEqual(set(groups), {"full", "indexer_kv"})
                self.assertEqual(groups["full"].cache_type, KVCacheSpecType.MLA)
                self.assertEqual(
                    groups["full"].tp_transfer_mode, TpCacheTransferMode.LEGACY
                )
                self.assertEqual(
                    groups["indexer_kv"].cache_type, KVCacheSpecType.OPAQUE_KV
                )
                self.assertEqual(groups["indexer_kv"].entry_dtype, DataType.TYPE_UINT8)
                self.assertEqual(groups["indexer_kv"].entry_elems, 132)
                self.assertEqual(
                    groups["full"].dtype,
                    DataType.TYPE_FP8_E4M3 if fp8_cache else DataType.TYPE_BF16,
                )

    def test_nvfp4_quantization_selects_only_target_projections(self):
        for name in ("KIMI_K31_NVFP4", "kimi_k31_nvfp4"):
            for fp8_cache in (False, True):
                with self.subTest(name=name, fp8_cache=fp8_cache):
                    config = self.build_config(quantization=name, fp8_cache=fp8_cache)
                    quant = config.attention_projection_quant_config
                    self.assertIsInstance(quant, ModelOptFp4Config)
                    self.assertEqual(quant.activation_scheme, "dynamic")
                    self.assertEqual(quant.supported_model_types, ("kimi_k31",))
                    self.assertEqual(
                        set(config.k31_nvfp4_modules),
                        {
                            "kda.input",
                            "kda.forget",
                            "dsa.input",
                            "dsa.query",
                        },
                    )
                    self.assertIsNone(config.quant_config)
                    self.assertFalse(config.quant_algo.isQuant())
                    self.assertEqual(config.expert_weight_dtype, "mxfp4")
                    self.assertEqual(config.data_type, DataType.TYPE_BF16)
                    self.assertFalse(config.attn_config.mla_fp8_compute)
                    self.assertIsInstance(
                        config.indexer_projection_quant_config, Fp8BlockWiseQuantConfig
                    )
                    self.assertEqual(
                        config.attn_config.kv_cache_dtype,
                        KvCacheDataType.FP8 if fp8_cache else KvCacheDataType.BASE,
                    )
                    self.assertFalse(hasattr(config, "kda_nvfp4"))
                    self.assertFalse(hasattr(config, "dsa_nvfp4"))

    def test_bf16_and_fp8_projection_modes_remain_available(self):
        for name in ("", "FP8_PER_BLOCK"):
            with self.subTest(name=name):
                config = self.build_config(quantization=name)
                self.assertEqual(config.k31_nvfp4_modules, ())
                self.assertEqual(
                    config.attention_projection_quant_config is not None, bool(name)
                )

    def test_nvfp4_weight_policy_keeps_outputs_fp8(self):
        from rtp_llm.models.kimi_k3.kimi_k31_weight import KimiK31Weight
        from rtp_llm.models.kimi_k3.nvfp4_weight import ATTENTION_PROJECTION_MODULES

        manifest = object.__new__(KimiK31Weight)
        manifest.model_config = self.build_config(quantization="KIMI_K31_NVFP4")
        for name, module in ATTENTION_PROJECTION_MODULES.items():
            with self.subTest(module=module):
                quant = manifest._attention_fp8_quant_config(SimpleNamespace(name=name))
                if module in ("kda.output", "dsa.output"):
                    self.assertIsInstance(quant, Fp8BlockWiseQuantConfig)
                else:
                    self.assertIsNone(quant)

    def test_retired_nvfp4_environment_is_ignored(self):
        with patch.dict(os.environ, KDA_NVFP4="invalid", DSA_NVFP4="invalid"):
            self.assertEqual(self.build_config().k31_nvfp4_modules, ())
            self.assertEqual(
                len(self.build_config(quantization="KIMI_K31_NVFP4").k31_nvfp4_modules),
                4,
            )

    def test_retired_nvfp4_json_options_are_rejected(self):
        for name in ("kda_nvfp4", "dsa_nvfp4"):
            for invalid in (False, True, 1, "1", None):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "KIMI_K31_NVFP4"):
                        self.build_config(override={name: invalid})

    def test_k31_nvfp4_does_not_enable_k3_or_native_draft_quantization(self):
        for model_type, name in (
            ("kimi_k3", "KIMI_K31_NVFP4"),
            ("kimi_k31_mtp", "KIMI_K31_NVFP4"),
            ("kimi_k31", "MODELOPT_FP4"),
        ):
            with self.subTest(model=model_type, quantization=name), self.assertRaises(
                ValueError
            ):
                self.build_config(model_type, quantization=name)

    def test_nvfp4_requires_cutlass_and_keeps_preset_private(self):
        with patch.dict(os.environ, RTP_LLM_FP4_GEMM_BACKEND="flashinfer"):
            with self.assertRaisesRegex(ValueError, "cutlass"):
                self.build_config(quantization="KIMI_K31_NVFP4")
            self.build_config(quantization="FP8_PER_BLOCK")
        config = self.build_config(quantization="KIMI_K31_NVFP4")
        self.assertIsNot(
            config.attention_projection_quant_config,
            init_quant_config("KIMI_K31_NVFP4"),
        )
        self.assertEqual(init_quant_config("MODELOPT_FP4").activation_scheme, "static")

    def test_workspace_defaults_and_json_override(self):
        self.assertEqual(
            self.build_config("kimi_k3").prefill_kv_workspace_budget_bytes, 6 * 1024**3
        )
        self.assertEqual(
            self.build_config().prefill_kv_workspace_budget_bytes, 8 * 1024**3
        )
        self.assertEqual(
            self.build_config(
                override={"prefill_kv_workspace_budget_bytes": 1024**2}
            ).prefill_kv_workspace_budget_bytes,
            1024**2,
        )
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.build_config(
                        override={"prefill_kv_workspace_budget_bytes": invalid}
                    )

    def test_workspace_explicit_default_value_remains_a_user_limit(self):
        config = self.build_config("kimi_k3")
        self.assertFalse(config._prefill_kv_workspace_budget_explicit)
        for options in (
            {"override": {"prefill_kv_workspace_budget_bytes": 6 * 1024**3}},
            {"prefill_kv_workspace_budget_bytes": 6 * 1024**3},
        ):
            with self.subTest(options=options):
                config = self.build_config("kimi_k3", **options)
                self.assertTrue(config._prefill_kv_workspace_budget_explicit)

    def test_workspace_and_shared_shard_use_model_args_instead_of_environment(self):
        with patch.dict(
            os.environ,
            {
                "RTP_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB": "1",
                "SHARED_EXPERT_WEIGHT_SHARD": "1",
                "KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD": "1",
            },
        ):
            for model_type, default_gib in (("kimi_k3", 6), ("kimi_k31", 8)):
                with self.subTest(model_type=model_type):
                    config = self.build_config(model_type)
                    self.assertEqual(
                        config.prefill_kv_workspace_budget_bytes, default_gib * 1024**3
                    )
                    self.assertFalse(config.shared_expert_weight_shard)
                    config = self.build_config(
                        model_type,
                        prefill_kv_workspace_budget_bytes=2 * 1024**3,
                        shared_expert_weight_shard=True,
                    )
                    self.assertEqual(
                        config.prefill_kv_workspace_budget_bytes, 2 * 1024**3
                    )
                    self.assertTrue(config.shared_expert_weight_shard)

    def test_explicit_model_args_override_json_options(self):
        config = self.build_config(
            override={
                "prefill_kv_workspace_budget_bytes": 1024**3,
                "shared_expert_weight_shard": True,
            },
            prefill_kv_workspace_budget_bytes=2 * 1024**3,
            shared_expert_weight_shard=False,
        )
        self.assertEqual(config.prefill_kv_workspace_budget_bytes, 2 * 1024**3)
        self.assertFalse(config.shared_expert_weight_shard)

    def test_invalid_shared_shard_option_is_rejected(self):
        for value in (0, 1, "1", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "shared_expert_weight_shard must be a boolean"
            ):
                self.build_config(override={"shared_expert_weight_shard": value})

    def test_draft_disables_query_cp_without_changing_target_or_page_rr(self):
        config = self.build_config()
        target = ParallelismConfig()
        target.role_type = RoleType.PREFILL
        target.tp_size = target.world_size = target.local_world_size = 8
        target.local_rank = 3
        target.prefill_cp_config.kv_cache_sharded = True
        config.configure_parallelism(target)
        self.assertTrue(target.prefill_cp_config.preserve_tp_layout)
        self.assertEqual(target.get_attn_tp_size(), 8)
        self.assertEqual(target.prefill_cp_config.prefill_cp_size, 8)

        draft_class = ModelFactory.get_model_cls("kimi_k31_mtp")
        draft = draft_class.speculative_parallelism_config(target)
        self.assertEqual(draft.prefill_cp_config.method, CPRotateMethod.DISABLED)
        self.assertFalse(draft.prefill_cp_config.preserve_tp_layout)
        self.assertTrue(draft.prefill_cp_config.kv_cache_sharded)
        self.assertEqual(draft.local_rank, 3)
        self.assertTrue(target.prefill_cp_config.is_enabled())
        self.assertTrue(target.prefill_cp_config.preserve_tp_layout)

    def test_native_draft_keeps_independent_embedding_and_head(self):
        target_class = ModelFactory.get_model_cls("kimi_k31")
        draft_class = ModelFactory.get_model_cls("kimi_k31_mtp")
        target = object.__new__(target_class)
        target.model_config = self.build_config()
        draft_config = self.build_config("kimi_k31_mtp")
        self.assertTrue(draft_config.is_mtp)
        self.assertFalse(draft_config.attn_config.is_sparse)
        self.assertTrue(draft_config.use_page_rr_kv_cache)
        self.assertEqual(
            draft_class.speculative_weight_alias_names(target, draft_config), ()
        )
        self.assertIsNone(getattr(draft_config.k3_runtime_config, "dsa", None))


if __name__ == "__main__":
    unittest.main()
