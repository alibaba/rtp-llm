"""Unit tests for Qwen 4.5-Flash config parsing, weight specs, and KV cache planning."""

from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path

from rtp_llm.config.qwen4_5_config import (
    Qwen4_5ModelConfig,
    compute_ple_ngram_head_vocab_sizes,
    compute_ple_ngram_shard_shape,
    estimate_qwen4_5_params_and_state,
    load_qwen4_5_config_json,
    parse_qwen4_5_config_json,
    parse_qwen4_5_quant_config,
    parse_qwen4_5_share_map,
)
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.qwen4_5 import (
    FFN_OUTPUT_SCONV_TAG,
    FULL_ATTN_KV_SCONV_TAG,
    FULL_ATTN_OUTPUT_SCONV_TAG,
    LINEAR_ATTN_OUTPUT_SCONV_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
    QWEN4_5_W,
    Qwen4_5,
    Qwen4_5Weight,
    build_qwen4_5_kv_cache_plan,
    expected_checkpoint_tensor_specs,
    summarize_checkpoint_specs,
    validate_qwen4_5_weight_map,
)

DEFAULT_REF_CONFIG = (
    "/data0/luohaocheng.lhc/artifacts/qwen45-flash-20261123/reference_config.json"
)
REF_CONFIG_PATH = Path(os.environ.get("QWEN4_5_REF_CONFIG", DEFAULT_REF_CONFIG))


class TestQwen4_5ConfigWeightKVCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_config = json.loads(REF_CONFIG_PATH.read_text(encoding="utf-8"))
        cls.arch_cfg = parse_qwen4_5_config_json(cls.raw_config)
        cls.quant_cfg = parse_qwen4_5_quant_config(cls.raw_config)
        cls.specs = expected_checkpoint_tensor_specs(cls.arch_cfg)
        cls.summary = summarize_checkpoint_specs(cls.specs, cls.arch_cfg)

    def test_1_reference_config_fields_and_share_map(self) -> None:
        raw = self.raw_config
        arch = self.arch_cfg
        qcfg = self.quant_cfg

        self.assertEqual(raw["architectures"], ["Qwen4_5ForCausalLM"])
        self.assertEqual(raw["model_type"], "qwen4_5")

        # 40 layers: 30 linear_attention, 10 full_attention at 3,7,...,39
        self.assertEqual(arch.num_hidden_layers, 40)
        self.assertEqual(len(arch.layer_types), 40)
        full_indices = tuple(
            i for i, t in enumerate(arch.layer_types) if t == "full_attention"
        )
        linear_indices = tuple(
            i for i, t in enumerate(arch.layer_types) if t == "linear_attention"
        )
        self.assertEqual(full_indices, (3, 7, 11, 15, 19, 23, 27, 31, 35, 39))
        self.assertEqual(len(linear_indices), 30)

        # share_map: 1-based 20->{24,28,32,36,40}, 21->{22,23,25..27,29..31,33..35,37..39}
        # -> 0-based {23,27,31,35,39: 19}, {21,22,24,25,26,28,29,30,32,33,34,36,37,38: 20}
        expected_sharing = {
            23: 19,
            27: 19,
            31: 19,
            35: 19,
            39: 19,
            21: 20,
            22: 20,
            24: 20,
            25: 20,
            26: 20,
            28: 20,
            29: 20,
            30: 20,
            32: 20,
            33: 20,
            34: 20,
            36: 20,
            37: 20,
            38: 20,
        }
        self.assertEqual(arch.get_attention_sharing(), expected_sharing)
        self.assertEqual(
            parse_qwen4_5_share_map(
                arch.share_map, arch.num_hidden_layers, arch.layer_types
            ),
            expected_sharing,
        )

        # Quantization config
        self.assertEqual(arch.quant_method, "mxfp4")
        self.assertEqual(qcfg.quant_method, "mxfp4")
        self.assertTrue(qcfg.is_mxfp4_routed_experts)
        self.assertEqual(qcfg.ple_quant_scheme, "fp8_pertensor")
        self.assertEqual(arch.ple_quant_scheme, "fp8_pertensor")
        self.assertTrue(qcfg.is_fp8_ple)
        self.assertEqual(
            qcfg.modules_to_convert, ("ple.ple_embedding.ngram_embedding",)
        )
        self.assertEqual(len(qcfg.modules_to_not_convert), 826)
        # Every BF16 excluded module/parameter must map to a BF16 tensor in the checkpoint specs
        # (bare nn.Parameter latent_* entries appear directly without a '.weight' suffix in real 1109 checkpoints)
        for mod_name in qcfg.modules_to_not_convert:
            tensor_key = mod_name if mod_name in self.specs else f"{mod_name}.weight"
            self.assertIn(tensor_key, self.specs)
            self.assertEqual(self.specs[tensor_key][0], "BF16")

        # MoE / FFN / SiTU-GLU
        self.assertEqual(arch.first_k_dense_replace, 1)
        self.assertEqual(arch.num_experts, 640)
        self.assertEqual(arch.num_experts_per_tok, 10)
        self.assertEqual(arch.routed_expert_hidden_size, 2560)
        self.assertEqual(arch.moe_intermediate_size, 2560)
        self.assertEqual(arch.shared_expert_intermediate_size, 1280)
        self.assertEqual(arch.hidden_act, "situ_glu")
        self.assertEqual(arch.hidden_act_gate_beta, 4.0)
        self.assertEqual(arch.hidden_act_up_beta, 25.0)
        self.assertEqual(arch.clip_mlp_output, 1000.0)

        # Hyper-Connections
        self.assertEqual(arch.hc_count, 4)
        self.assertEqual(arch.hc_lowrank, 128)
        self.assertTrue(arch.hc_prev_mix)
        self.assertFalse(arch.hc_prev_combine)

        # PLE
        self.assertEqual(arch.ple_layer_ids, (2,))
        self.assertEqual(arch.ngram_size, 3)
        self.assertEqual(arch.heads_per_ngram, 10)
        self.assertEqual(arch.split_ngram_parts, 128)
        self.assertEqual(arch.ple_embed_dim, 5120)
        self.assertEqual(arch.ple_conv_kernel_size, 4)
        self.assertEqual(len(compute_ple_ngram_head_vocab_sizes(arch)), 20)
        self.assertEqual(compute_ple_ngram_shard_shape(arch), (5468784, 256))

        # ShortConv flags
        self.assertTrue(arch.use_full_attn_kv_sconv)
        self.assertTrue(arch.use_full_attn_output_sconv)
        self.assertTrue(arch.use_linear_attn_output_sconv)
        self.assertTrue(arch.use_ffn_output_sconv)
        self.assertFalse(arch.hybrid_sharing_consumer_layer_no_output_sconv)

        # File loader helper round-trip
        loaded_arch, loaded_quant = load_qwen4_5_config_json(REF_CONFIG_PATH)
        self.assertEqual(loaded_arch, arch)
        self.assertEqual(loaded_quant, qcfg)

    def test_2_exact_checkpoint_tensor_specs_bytes_and_params(self) -> None:
        specs = self.specs
        summary = self.summary
        est = estimate_qwen4_5_params_and_state(self.arch_cfg)

        # Exact 1109 checkpoint numbers
        self.assertEqual(len(specs), 151_038)
        self.assertEqual(summary["tensor_count"], 151_038)
        self.assertEqual(summary["total_bytes"], 461_163_888_730)
        self.assertAlmostEqual(est.checkpoint_gib, 429.492, places=3)

        self.assertEqual(summary["logical_params"], 680_564_940_288)
        self.assertEqual(summary["core_params"], 501_363_826_176)
        self.assertEqual(summary["active_params_excl_embed"], 15_755_173_376)

        self.assertEqual(est.checkpoint_tensor_count, 151_038)
        self.assertEqual(est.checkpoint_total_bytes, 461_163_888_730)
        self.assertEqual(est.total_params, 680_564_940_288)
        self.assertEqual(est.core_params, 501_363_826_176)
        self.assertEqual(est.active_params_excl_embed, 15_755_173_376)

        # Top-level structure (NO model.norm!)
        self.assertNotIn("model.norm.weight", specs)
        self.assertEqual(specs["model.embed_tokens.weight"], ("BF16", (248320, 5120)))
        self.assertEqual(specs["lm_head.weight"], ("BF16", (248320, 5120)))
        self.assertEqual(
            specs["model.hyper_connection_mixer.hc_norm.weight"], ("BF16", (20480,))
        )
        self.assertEqual(
            specs["model.hyper_connection_mixer.hc_norm.bias"], ("BF16", (20480,))
        )
        self.assertEqual(
            specs["model.hyper_connection_mixer.input_mix_weight_down.weight"],
            ("BF16", (128, 20480)),
        )
        self.assertEqual(
            specs["model.hyper_connection_mixer.input_mix_weight_up.weight"],
            ("BF16", (20480, 128)),
        )

        # Per-layer HC edge cases: Layer 0 attn up [5120, 128], Layer 39 omits mlp down
        self.assertEqual(
            specs["model.layers.0.attn_hyper_connection.input_mix_weight_up.weight"],
            ("BF16", (5120, 128)),
        )
        self.assertEqual(
            specs["model.layers.1.attn_hyper_connection.input_mix_weight_up.weight"],
            ("BF16", (20480, 128)),
        )
        self.assertIn(
            "model.layers.38.mlp_hyper_connection.input_mix_weight_down.weight", specs
        )
        self.assertNotIn(
            "model.layers.39.mlp_hyper_connection.input_mix_weight_down.weight", specs
        )

        # Layer 1 PLE (128 FP8 shards + scale + 3 I64 metadata + 6 projections/norms/conv)
        for s in (0, 63, 127):
            self.assertEqual(
                specs[
                    f"model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight"
                ],
                ("F8_E4M3", (5468784, 256)),
            )
        self.assertEqual(
            specs["model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"],
            ("BF16", (1,)),
        )
        self.assertEqual(
            specs["model.layers.1.ple.ple_embedding.layer_multipliers"], ("I64", (3,))
        )
        self.assertEqual(
            specs["model.layers.1.ple.ple_embedding.ngram_heads_offsets"],
            ("I64", (20,)),
        )
        self.assertEqual(
            specs["model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"],
            ("I64", (20,)),
        )
        self.assertEqual(
            specs["model.layers.1.ple.key_proj.weight"], ("BF16", (20480, 5120))
        )
        self.assertEqual(
            specs["model.layers.1.ple.value_proj.weight"], ("BF16", (5120, 5120))
        )
        self.assertEqual(
            specs["model.layers.1.ple.conv1d.weight"], ("BF16", (20480, 1, 4))
        )

        # Linear Attn: source (layer 20) vs consumer (layer 21)
        self.assertEqual(
            specs["model.layers.20.linear_attn.in_proj_qkvz.weight"],
            ("BF16", (20480, 5120)),
        )
        self.assertEqual(
            specs["model.layers.20.linear_attn.in_proj_ba.weight"],
            ("BF16", (256, 5120)),
        )
        self.assertEqual(
            specs["model.layers.20.linear_attn.conv1d.weight"],
            ("BF16", (12288, 1, 4)),
        )
        self.assertEqual(specs["model.layers.20.linear_attn.A_log"], ("BF16", (128,)))
        self.assertEqual(specs["model.layers.20.linear_attn.dt_bias"], ("BF16", (128,)))

        self.assertEqual(
            specs["model.layers.21.linear_attn.in_proj_qkvz.weight"],
            ("BF16", (10240, 5120)),
        )
        self.assertEqual(
            specs["model.layers.21.linear_attn.conv1d.weight"],
            ("BF16", (2048, 1, 4)),
        )
        self.assertNotIn("model.layers.21.linear_attn.in_proj_ba.weight", specs)
        self.assertNotIn("model.layers.21.linear_attn.A_log", specs)
        self.assertNotIn("model.layers.21.linear_attn.dt_bias", specs)

        # Full Attn: producer (layer 19) vs consumer (layer 23)
        self.assertEqual(
            specs["model.layers.19.self_attn.q_proj.weight"], ("BF16", (16384, 5120))
        )
        self.assertEqual(
            specs["model.layers.19.self_attn.k_proj.weight"], ("BF16", (256, 5120))
        )
        self.assertEqual(
            specs["model.layers.19.self_attn.v_proj.weight"], ("BF16", (256, 5120))
        )
        self.assertEqual(
            specs["model.layers.19.self_attn.k_sconv.weight"], ("BF16", (256, 1, 4))
        )
        self.assertEqual(
            specs["model.layers.19.self_attn.v_sconv.weight"], ("BF16", (256, 1, 4))
        )
        self.assertEqual(
            specs["model.layers.23.self_attn.q_proj.weight"], ("BF16", (16384, 5120))
        )
        self.assertEqual(
            specs["model.layers.23.self_attn.q_norm.weight"], ("BF16", (256,))
        )
        self.assertEqual(
            specs["model.layers.23.self_attn.o_proj.weight"], ("BF16", (5120, 8192))
        )
        self.assertEqual(
            specs["model.layers.23.full_attn_output_sconv.weight"],
            ("BF16", (5120, 1, 4)),
        )
        for absent in (
            "k_proj.weight",
            "v_proj.weight",
            "k_norm.weight",
            "k_sconv.weight",
            "v_sconv.weight",
        ):
            self.assertNotIn(f"model.layers.23.self_attn.{absent}", specs)

        # FFN: Layer 0 dense vs Layer 1..39 Latent MoE + MXFP4 routed experts
        self.assertEqual(
            specs["model.layers.0.mlp.gate_proj.weight"], ("BF16", (12800, 5120))
        )
        self.assertEqual(specs["model.layers.1.mlp.expert_bias"], ("F32", (640,)))
        self.assertEqual(
            specs["model.layers.1.mlp.experts.0.gate_proj.weight"], ("U8", (2560, 1280))
        )
        self.assertEqual(
            specs["model.layers.1.mlp.experts.0.gate_proj.weight_scale"],
            ("U8", (2560, 80)),
        )
        self.assertEqual(
            specs["model.layers.39.mlp.experts.639.down_proj.weight"],
            ("U8", (2560, 1280)),
        )
        self.assertEqual(
            specs["model.layers.39.mlp.experts.639.down_proj.weight_scale"],
            ("U8", (2560, 80)),
        )

    def test_3_weight_map_validation_and_weight_class(self) -> None:
        # Exact coverage with (dtype, shape) metadata
        res = validate_qwen4_5_weight_map(self.arch_cfg, self.specs)
        self.assertEqual(res["expected_count"], 151_038)
        self.assertEqual(res["matched_count"], 151_038)
        self.assertEqual(res["extra_ignored_count"], 0)

        # Simulated safetensors index.json {tensor_name: shard_file}
        fake_index = {k: "model-00001-of-00120.safetensors" for k in self.specs}
        res_idx = validate_qwen4_5_weight_map(self.arch_cfg, fake_index)
        self.assertEqual(res_idx["matched_count"], 151_038)

        # Allowed extra prefixes (e.g. mtp.)
        with_mtp = dict(fake_index)
        with_mtp["mtp.fc.weight"] = "model-00120-of-00120.safetensors"
        res_mtp = validate_qwen4_5_weight_map(
            self.arch_cfg, with_mtp, allow_extra_prefixes=("mtp.",)
        )
        self.assertEqual(res_mtp["extra_ignored_count"], 1)

        # Unexpected tensor rejected when not allowed
        with self.assertRaisesRegex(ValueError, "unexpected 1 extra tensor"):
            validate_qwen4_5_weight_map(self.arch_cfg, with_mtp)

        # Missing tensor rejected
        missing_one = dict(fake_index)
        del missing_one["model.layers.1.mlp.expert_bias"]
        with self.assertRaisesRegex(ValueError, "missing 1 expected tensor"):
            validate_qwen4_5_weight_map(self.arch_cfg, missing_one)

        # Dtype/shape mismatch rejected
        bad_shape = dict(self.specs)
        bad_shape["model.layers.1.mlp.expert_bias"] = ("BF16", (640,))
        with self.assertRaisesRegex(ValueError, "dtype/shape mismatch"):
            validate_qwen4_5_weight_map(self.arch_cfg, bad_shape)

        # Qwen4_5Weight pure-Python descriptor builder covers every tensor 1-to-1
        weight = Qwen4_5Weight(arch_config=self.arch_cfg)
        weight._process_meta({}, list(self.specs.keys()))
        self.assertEqual(weight.prefix, "model.")
        desc_specs = {
            d.ckpt_name: (d.dtype, d.shape)
            for d in weight.get_global_weight_descriptors()
        }
        for layer_descs in weight.get_all_layer_weight_descriptors():
            for d in layer_descs:
                desc_specs[d.ckpt_name] = (d.dtype, d.shape)
        self.assertEqual(desc_specs, self.specs)
        self.assertEqual(QWEN4_5_W.embedding, "qwen4_5.embedding")

    def test_4_kv_cache_plan_yoco_and_non_yoco(self) -> None:
        plan = build_qwen4_5_kv_cache_plan(self.arch_cfg, enable_yoco_decode=True)

        # 5 FullAttn KV layers: 5.00 KiB/tok BF16
        self.assertEqual(plan.full_attn_kv_layer_indices, (3, 7, 11, 15, 19))
        self.assertEqual(plan.full_attn_shared_consumer_indices, (23, 27, 31, 35, 39))
        self.assertEqual(plan.num_full_attn_kv_layers, 5)
        self.assertEqual(plan.kv_bytes_per_token, 5120)
        self.assertAlmostEqual(plan.kv_kib_per_token, 5.00, places=6)

        # Consumer FullAttn layers share KV from layer 19 with 0 new KV bytes
        for idx in (23, 27, 31, 35, 39):
            lp = plan.layers[idx]
            self.assertFalse(lp.allocates_kv_cache)
            self.assertEqual(lp.kv_source_layer, 19)
            self.assertEqual(lp.kv_bytes_per_token, 0)
            self.assertFalse(lp.has_kv_sconv)
            tags = {r.tag for r in lp.opaque_regions}
            self.assertEqual(tags, {FULL_ATTN_OUTPUT_SCONV_TAG, FFN_OUTPUT_SCONV_TAG})

        # Producer FullAttn layers carry k_sconv + v_sconv + full_attn_output_sconv + ffn_output_sconv
        for idx in (3, 7, 11, 15, 19):
            lp = plan.layers[idx]
            self.assertTrue(lp.allocates_kv_cache)
            self.assertEqual(lp.num_kv_heads, 1)
            self.assertEqual(lp.head_dim, 256)
            self.assertTrue(lp.has_kv_sconv)
            self.assertEqual(lp.k_sconv_shape, (256, 3))
            self.assertEqual(lp.v_sconv_shape, (256, 3))
            tags = {r.tag for r in lp.opaque_regions}
            self.assertEqual(
                tags,
                {
                    FULL_ATTN_KV_SCONV_TAG,
                    FULL_ATTN_OUTPUT_SCONV_TAG,
                    FFN_OUTPUT_SCONV_TAG,
                },
            )

        # 16 YoCo GDN SSM layers: 32.0 MiB/req FP32
        self.assertEqual(plan.num_gdn_ssm_layers, 16)
        self.assertEqual(len(plan.gdn_shared_consumer_indices), 14)
        self.assertEqual(plan.ssm_bytes_per_req, 16 * 128 * 64 * 64 * 4)
        self.assertAlmostEqual(plan.ssm_mib_per_req, 32.0, places=6)

        # GDN source (20) vs consumer (21)
        src_gdn = plan.layers[20]
        self.assertEqual(src_gdn.gdn_conv_channels, 12288)
        self.assertEqual(src_gdn.gdn_conv_state_len, 3)
        self.assertTrue(src_gdn.allocates_ssm_state)
        self.assertEqual(src_gdn.ssm_state_shape, (128, 64, 64))

        cons_gdn = plan.layers[21]
        self.assertEqual(cons_gdn.gdn_conv_channels, 2048)
        self.assertEqual(cons_gdn.gdn_conv_state_len, 3)
        self.assertFalse(cons_gdn.allocates_ssm_state)
        self.assertEqual(cons_gdn.ssm_source_layer, 20)
        self.assertEqual(cons_gdn.ssm_bytes_per_req, 0)
        self.assertEqual(
            {r.tag for r in cons_gdn.opaque_regions},
            {LINEAR_ATTN_OUTPUT_SCONV_TAG, FFN_OUTPUT_SCONV_TAG},
        )

        # Layer 1 PLE state
        self.assertEqual(plan.ple_layer_indices, (1,))
        ple_lp = plan.layers[1]
        self.assertTrue(ple_lp.has_ple)
        self.assertEqual(ple_lp.ple_conv_state_shape, (20480, 3))
        self.assertEqual(ple_lp.ple_ngram_ctx_shape, (2,))
        self.assertIn(PLE_STATE_TAG, {r.tag for r in ple_lp.opaque_regions})
        self.assertIn(PLE_NGRAM_CTX_TAG, {r.tag for r in ple_lp.opaque_regions})
        self.assertEqual(plan.ple_ngram_ctx_bytes_per_req, 16)

        # 116 ShortConv states (121 individual conv tensors): 3.76 MiB/req
        self.assertEqual(plan.num_short_conv_states, 116)
        self.assertEqual(plan.num_short_conv_tensors, 121)
        self.assertEqual(plan.short_conv_bytes_per_req, 3_947_520)
        self.assertAlmostEqual(plan.short_conv_mib_per_req, 3.76, places=2)

        # Non-YoCo decode mode: consumers allocate their own SSM state (30 layers = 60 MiB)
        plan_no_yoco = build_qwen4_5_kv_cache_plan(
            self.arch_cfg, enable_yoco_decode=False
        )
        self.assertEqual(plan_no_yoco.num_full_attn_kv_layers, 5)
        self.assertEqual(plan_no_yoco.num_gdn_ssm_layers, 30)
        self.assertAlmostEqual(plan_no_yoco.ssm_mib_per_req, 60.0, places=6)
        self.assertEqual(plan_no_yoco.num_short_conv_states, 116)

    def test_5_invalid_share_map_and_config_rejection(self) -> None:
        layer_types = self.arch_cfg.layer_types
        n = self.arch_cfg.num_hidden_layers

        invalid_share_maps = [
            "20->20",          # self-share
            "24->20",          # backward share (src > dst)
            "0->4",            # 0-based index (must be 1-based >= 1)
            "20->41",          # out-of-range dst
            "20->21",          # cross-type: full_attention(20) -> linear_attention(21)
            "21->24",          # cross-type: linear_attention(21) -> full_attention(24)
            "20->24,24->28",   # chained sharing (source cannot also be consumer)
            "16->24,20->24",   # duplicate consumer
            "20:24",           # malformed separator
            "foo->bar",        # non-integer
            "20->24,",         # trailing empty item
        ]
        for bad in invalid_share_maps:
            with self.subTest(share_map=bad):
                with self.assertRaises(ValueError):
                    parse_qwen4_5_share_map(bad, n, layer_types)
                bad_cfg = copy.deepcopy(self.raw_config)
                bad_cfg["share_map"] = bad
                with self.assertRaises(ValueError):
                    parse_qwen4_5_config_json(bad_cfg)

        # Invalid model_type / architectures / MXFP4 alignment
        bad_mt = copy.deepcopy(self.raw_config)
        bad_mt["model_type"] = "qwen4_exp"
        with self.assertRaises(ValueError):
            parse_qwen4_5_config_json(bad_mt)

        bad_arch = copy.deepcopy(self.raw_config)
        bad_arch["architectures"] = ["LlamaForCausalLM"]
        with self.assertRaises(ValueError):
            parse_qwen4_5_config_json(bad_arch)

        bad_mxfp4 = copy.deepcopy(self.raw_config)
        bad_mxfp4["routed_expert_hidden_size"] = 2561
        with self.assertRaises(ValueError):
            parse_qwen4_5_config_json(bad_mxfp4)

    def test_6_model_wrapper_and_registration(self) -> None:
        self.assertIs(Qwen4_5.get_weight_cls(), Qwen4_5Weight)
        self.assertEqual(
            ModelDict.get_ft_model_type_by_hf_architectures("Qwen4_5ForCausalLM"),
            "qwen4_5",
        )
        model_cfg = Qwen4_5.create_config(REF_CONFIG_PATH)
        self.assertIsInstance(model_cfg, Qwen4_5ModelConfig)
        self.assertEqual(model_cfg.arch_config, self.arch_cfg)
        self.assertTrue(model_cfg.enable_qwen4_5_ple)
        self.assertEqual(model_cfg.qwen4_5_kv_cache_plan.num_full_attn_kv_layers, 5)
        self.assertEqual(
            model_cfg.qwen4_5_param_and_state_estimate.checkpoint_tensor_count,
            151_038,
        )


    def test_7_tp4_gdn_opaque_state_sharding_and_weight_split(self) -> None:
        from rtp_llm.config.qwen4_5_config import estimate_qwen4_5_params_and_state
        from rtp_llm.model_loader.load_config import LoadConfig
        from rtp_llm.model_loader.weight_module import AtomicWeight
        from rtp_llm.models.qwen4_5 import (
            GDN_CONSUMER_Q_CONV_TAG,
            GDN_CONV_STATE_TAG,
            GDN_SSM_STATE_TAG,
            build_qwen4_5_kv_cache_spec_descs,
        )
        from rtp_llm.ops import HWKernelConfig, KVCacheSpecType, ParallelismConfig, TpCacheTransferMode
        from rtp_llm.utils.database import BaseDatabase
        import torch

        plan_tp1 = build_qwen4_5_kv_cache_plan(
            self.arch_cfg, tp_size=1, enable_yoco_decode=True
        )
        plan_tp4 = build_qwen4_5_kv_cache_plan(
            self.arch_cfg, tp_size=4, enable_yoco_decode=True
        )
        # GDN 32K/128V -> 8K/32V per TP=4 rank; SSM state drops 4x (40 MiB -> 10 MiB)
        self.assertAlmostEqual(plan_tp4.ssm_mib_per_req, plan_tp1.ssm_mib_per_req / 4.0, places=6)
        self.assertEqual(plan_tp4.layers[0].ssm_state_shape, (32, 64, 64))
        self.assertEqual(plan_tp4.layers[0].gdn_conv_channels, 3072)  # 2*512 + 8192 -> 1024+8192? No, (1024+8192)/4 wait: 2*512+2048=3072
        self.assertEqual(plan_tp4.layers[21].gdn_conv_channels, 512)  # consumer Q-only: 8*64

        # FullAttn 1KV head + kv_sconv + output_sconv + PLE stay replicated on all TP ranks
        self.assertEqual(plan_tp4.kv_bytes_per_token, plan_tp1.kv_bytes_per_token)
        self.assertEqual(plan_tp4.layers[3].num_kv_heads, 1)
        self.assertEqual(plan_tp4.layers[3].k_sconv_shape, (256, 3))
        self.assertEqual(plan_tp4.ple_ngram_ctx_bytes_per_req, 16)
        for lp1, lp4 in zip(plan_tp1.layers, plan_tp4.layers):
            if lp1.has_kv_sconv:
                self.assertEqual(lp4.k_sconv_shape, lp1.k_sconv_shape)
                self.assertEqual(lp4.v_sconv_shape, lp1.v_sconv_shape)
            for r1, r4 in zip(lp1.opaque_regions, lp4.opaque_regions):
                if r1.tag not in (GDN_SSM_STATE_TAG, GDN_CONV_STATE_TAG, GDN_CONSUMER_Q_CONV_TAG):
                    self.assertEqual(r4.shape, r1.shape)

        # KVCacheSpecDescs reflect per-rank GDN sizing with REPLICATED transfer for 1:1 P_r->D_r
        mcfg = Qwen4_5.create_config(REF_CONFIG_PATH)
        descs = build_qwen4_5_kv_cache_spec_descs(mcfg, self.arch_cfg, tp_size=4)
        l0_by_tag = {d.tag: d for d in descs[0]}
        self.assertEqual(l0_by_tag[GDN_SSM_STATE_TAG].explicit_entry_count, 32)
        self.assertEqual(l0_by_tag[GDN_SSM_STATE_TAG].entry_elems, 64 * 64)
        self.assertEqual(l0_by_tag[GDN_SSM_STATE_TAG].tp_transfer_mode, TpCacheTransferMode.REPLICATED)
        self.assertEqual(l0_by_tag[GDN_CONV_STATE_TAG].explicit_entry_count, 3)
        self.assertEqual(l0_by_tag[GDN_CONV_STATE_TAG].entry_elems, 3072)
        self.assertEqual(l0_by_tag[GDN_CONV_STATE_TAG].tp_transfer_mode, TpCacheTransferMode.REPLICATED)
        l21_by_tag = {d.tag: d for d in descs[21]}
        self.assertEqual(l21_by_tag[GDN_CONSUMER_Q_CONV_TAG].entry_elems, 512)
        self.assertEqual(l21_by_tag[GDN_CONSUMER_Q_CONV_TAG].tp_transfer_mode, TpCacheTransferMode.REPLICATED)
        l3_full = [d for d in descs[3] if d.cache_type == KVCacheSpecType.MHA][0]
        self.assertEqual(l3_full.tp_transfer_mode, TpCacheTransferMode.MHA_HEADS)

        # Qwen4_5Weight CustomAtomicWeight _split safety under TP=4 + EP=4
        class _DummyDB(BaseDatabase):
            def get_pretrain_tensor_names(self): return []
            def get_lora_tensor_names(self, name): return []
            def load_tensor(self, name, data_type=torch.float16): return []

        pc = ParallelismConfig()
        pc.tp_size = 4; pc.tp_rank = 1; pc.ep_size = 4; pc.ep_rank = 1
        pc.world_size = 4; pc.world_rank = 1
        wb = Qwen4_5Weight(mcfg, parallelism_config=pc, hw_kernel_config=HWKernelConfig())
        winfo = wb._get_weight_info()
        lc = LoadConfig(
            database=_DummyDB(),
            num_layers=self.arch_cfg.num_hidden_layers,
            hidden_size=self.arch_cfg.hidden_size,
            head_num=self.arch_cfg.num_attention_heads,
            head_num_kv=self.arch_cfg.num_key_value_heads,
            size_per_head=self.arch_cfg.head_dim,
            moe_pure_tp_mode=False,
            align_size=1,
            moe_align_size=1,
            moe_layer_index=list(range(self.arch_cfg.first_k_dense_replace, self.arch_cfg.num_hidden_layers)),
            moe_n_group=1,
            expert_num=self.arch_cfg.num_experts,
            enable_eplb=False,
            phy_exp_num=self.arch_cfg.num_experts,
            tp_size=4, tp_rank=1, ep_size=4, ep_rank=1, dp_size=1, dp_rank=0,
            lm_head_tp_size=4, lm_head_tp_rank=1, ffn_tp_size=4, ffn_tp_rank=1,
            num_nodes=1, layer_range=range(self.arch_cfg.num_hidden_layers),
        )
        # Check normal GDN conv1d split [Q(2048), K(2048), V(8192)] -> [3072, 1, 4]
        w_conv0 = next(w for w in winfo.layer_weights[0] if w.name == QWEN4_5_W.linear_conv1d)
        conv_full = torch.arange(12288 * 4, dtype=torch.bfloat16).view(12288, 1, 4)
        conv_p1 = w_conv0._split(conv_full, lc)[QWEN4_5_W.linear_conv1d]
        self.assertEqual(tuple(conv_p1.shape), (3072, 1, 4))
        expected_p1 = torch.cat([conv_full[512:1024], conv_full[2048+512:2048+1024], conv_full[4096+2048:4096+4096]], dim=0)
        self.assertTrue(torch.equal(conv_p1, expected_p1))

    def test_8_multi_parallel_kv_plans_and_dflash2_fp8_kv_plan(self):
        from types import SimpleNamespace
        from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
            GDN_CONV_STATE_TAG,
            GDN_SSM_STATE_TAG,
            build_dflash2_kv_cache_plan,
            build_qwen4_5_kv_cache_plan,
        )

        p_tp4 = build_qwen4_5_kv_cache_plan(self.arch_cfg, parallel_mode="tp4_ep4", enable_yoco_decode=True)
        p_dp4 = build_qwen4_5_kv_cache_plan(self.arch_cfg, parallel_mode="dp4_ep4", enable_yoco_decode=True)
        p_dp2 = build_qwen4_5_kv_cache_plan(self.arch_cfg, parallel_mode="dp2_tp2_ep4", enable_yoco_decode=True)
        p_sp4 = build_qwen4_5_kv_cache_plan(self.arch_cfg, parallel_mode="sp4_tp4_ep4", enable_yoco_decode=True)

        self.assertEqual(p_tp4.layers[0].ssm_state_shape, (32, 64, 64))
        self.assertEqual(p_sp4.layers[0].ssm_state_shape, (32, 64, 64))
        self.assertEqual(p_dp2.layers[0].ssm_state_shape, (64, 64, 64))
        self.assertEqual(p_dp4.layers[0].ssm_state_shape, (128, 64, 64))
        self.assertEqual(p_dp2.layers[0].gdn_conv_channels, 6144)
        self.assertEqual(p_dp4.layers[0].gdn_conv_channels, 12288)
        self.assertAlmostEqual(p_dp2.ssm_mib_per_req, p_tp4.ssm_mib_per_req * 2.0, places=5)
        self.assertAlmostEqual(p_dp4.ssm_mib_per_req, p_tp4.ssm_mib_per_req * 4.0, places=5)
        self.assertEqual(p_dp4.kv_bytes_per_token, p_tp4.kv_bytes_per_token)
        self.assertTrue(any(lp.allocates_ssm_state for lp in p_dp2.layers))

        d_spec = SimpleNamespace(num_hidden_layers=5, num_key_value_heads=1, head_dim=256, swa_window_size=2048)
        d_bf16 = build_dflash2_kv_cache_plan(d_spec, parallel_mode="tp4_ep4", dflash_kv_dtype="bf16")
        d_fp8 = build_dflash2_kv_cache_plan(d_spec, parallel_mode="dp4_ep4", dflash_kv_dtype="fp8_e4m3")
        self.assertEqual(d_bf16.kv_bytes_per_token, 5 * 2 * 1 * 256 * 2)
        self.assertEqual(d_fp8.kv_bytes_per_token, 5 * 2 * 1 * 256 * 1)
        self.assertEqual(d_fp8.total_bytes_for_seq_len(20480, 1), d_bf16.total_bytes_for_seq_len(20480, 1) // 2)

    def test_9_fp8_e4m3_kv_cache_plan_alias(self):
        from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import build_qwen4_5_kv_cache_plan

        p_fp8 = build_qwen4_5_kv_cache_plan(self.arch_cfg, parallel_mode="tp4_ep4", kv_dtype="fp8")
        p_f8 = build_qwen4_5_kv_cache_plan(self.arch_cfg, parallel_mode="tp4_ep4", kv_dtype="f8_e4m3")
        p_fp8_e4m3 = build_qwen4_5_kv_cache_plan(
            self.arch_cfg, parallel_mode="tp4_ep4", kv_dtype="fp8_e4m3"
        )
        p_fp8_e4m3_upper = build_qwen4_5_kv_cache_plan(
            self.arch_cfg, parallel_mode="dp4_ep4", kv_dtype="FP8_E4M3"
        )
        self.assertEqual(p_fp8_e4m3.kv_dtype, "FP8_E4M3")
        self.assertEqual(p_fp8_e4m3.kv_dtype_bytes, 1)
        self.assertEqual(p_fp8_e4m3.kv_bytes_per_token, 2560)
        self.assertEqual(p_fp8_e4m3.kv_bytes_per_token, p_fp8.kv_bytes_per_token)
        self.assertEqual(p_fp8_e4m3.kv_bytes_per_token, p_f8.kv_bytes_per_token)
        self.assertEqual(p_fp8_e4m3_upper.kv_bytes_per_token, 2560)


    def test_10_dp2_tpsp2_ep4_and_sp4_topology_resolution(self) -> None:
        from rtp_llm.models_py.model_desc.qwen4_5 import (
            _SUPPORTED_PARALLEL_MODES,
            resolve_qwen4_5_parallel_topology,
        )
        from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import build_qwen4_5_kv_cache_plan

        self.assertIn("sp4_tp4_ep4", _SUPPORTED_PARALLEL_MODES)
        self.assertIn("dp2_tpsp2_ep4", _SUPPORTED_PARALLEL_MODES)

        t_dp2_tpsp2 = resolve_qwen4_5_parallel_topology(
            parallel_mode="dp2_tpsp2_ep4", dp_rank=1, tp_rank=1, ep_rank=3
        )
        self.assertEqual(t_dp2_tpsp2.parallel_mode, "dp2_tpsp2_ep4")
        self.assertEqual(t_dp2_tpsp2.dp_size, 2)
        self.assertEqual(t_dp2_tpsp2.tp_size, 2)
        self.assertEqual(t_dp2_tpsp2.sp_size, 2)
        self.assertEqual(t_dp2_tpsp2.ep_size, 4)
        self.assertEqual(t_dp2_tpsp2.dp_rank, 1)
        self.assertEqual(t_dp2_tpsp2.tp_rank, 1)
        self.assertEqual(t_dp2_tpsp2.sp_rank, 1)
        self.assertEqual(t_dp2_tpsp2.ep_rank, 3)
        self.assertFalse(t_dp2_tpsp2.dp_attn)

        t_sp4 = resolve_qwen4_5_parallel_topology(parallel_mode="sp4_tp4_ep4", tp_rank=2, ep_rank=2)
        self.assertEqual(t_sp4.parallel_mode, "sp4_tp4_ep4")
        self.assertEqual(t_sp4.dp_size, 1)
        self.assertEqual(t_sp4.tp_size, 4)
        self.assertEqual(t_sp4.sp_size, 4)
        self.assertEqual(t_sp4.ep_size, 4)
        self.assertEqual(t_sp4.sp_rank, 2)
        self.assertFalse(t_sp4.dp_attn)

        t_imp_dp_tpsp = resolve_qwen4_5_parallel_topology(
            dp_size=2, tp_size=2, sp_size=2, ep_size=4, dp_rank=1, tp_rank=1, ep_rank=3
        )
        self.assertEqual(t_imp_dp_tpsp.parallel_mode, "dp2_tpsp2_ep4")
        self.assertEqual(t_imp_dp_tpsp.sp_rank, 1)
        self.assertFalse(t_imp_dp_tpsp.dp_attn)

        with self.assertRaises(ValueError):
            resolve_qwen4_5_parallel_topology(parallel_mode="invalid_mode")

        p_dp2_tpsp2 = build_qwen4_5_kv_cache_plan(
            self.arch_cfg, tp_size=2, parallel_mode="dp2_tpsp2_ep4", enable_yoco_decode=True
        )
        p_dp2_tp2 = build_qwen4_5_kv_cache_plan(
            self.arch_cfg, parallel_mode="dp2_tp2_ep4", enable_yoco_decode=True
        )
        self.assertEqual(p_dp2_tpsp2.layers[0].ssm_state_shape, p_dp2_tp2.layers[0].ssm_state_shape)
        self.assertEqual(p_dp2_tpsp2.layers[0].gdn_conv_channels, 6144)


if __name__ == "__main__":
    unittest.main()
