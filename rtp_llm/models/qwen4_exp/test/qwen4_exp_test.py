import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_factory_register import ModelDict, _model_factory
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4ExpWeight
from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeMixin,
)
from rtp_llm.ops import (
    DataType,
    HWKernelConfig,
    HybridAttentionType,
    KvCacheDataType,
    KVCacheSpecType,
    ParallelismConfig,
    RoleType,
    RopeStyle,
)

_NUM_LAYERS = 48

# Key patterns taken from the real Qwen/Qwen3.8-Flash-Next weight map; see
# docs/design/qwen3.8_flash_next_support_design.md appendix C.
_CKPT_KEYS = (
    [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    ]
    + [
        f"model.language_model.layers.{i}.{module}.{suffix}"
        for i in range(_NUM_LAYERS)
        for module in ("attn_hyper_connection", "mlp_hyper_connection")
        for suffix in (
            "hc_norm.weight",
            "input_mix_weight_down.weight",
            "input_mix_weight_up.weight",
            "block_inject_weight.weight",
        )
    ]
    + [
        f"model.language_model.layers.{i}.mlp.experts.{proj}"
        for i in range(_NUM_LAYERS)
        for proj in ("gate_up_proj", "down_proj")
    ]
)


def _ckpt_keys(modules):
    """Collect ckpt tensor names, descending into composite weight modules."""
    keys = set()
    for module in modules:
        sub_weights = getattr(module, "sub_weights", None)
        if sub_weights:
            keys |= _ckpt_keys(sub_weights.values())
        else:
            keys |= {info.name for info in module.weights}
    return keys


class Qwen4ExpTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        Path(self._temp_dir.name, "config.json").write_text(json.dumps(self._config()))
        # Feature gates are process-wide development switches. Keep this suite
        # deterministic even when a developer exports them in the parent shell.
        with mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_ENABLE_QWEN4_EXP_PLE": "false",
                "RTP_LLM_ENABLE_QWEN4_EXP_QSA": "false",
            },
        ):
            self.config = Qwen4Exp.create_config(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_registration(self):
        self.assertIs(_model_factory["qwen4_exp"], Qwen4Exp)
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(
                {"architectures": ["Qwen4ExpForConditionalGeneration"]}
            ),
            "qwen4_exp",
        )
        self.assertIs(Qwen4Exp.get_weight_cls(), Qwen4ExpWeight)
        self.assertIs(get_multimodal_mixin_cls("qwen4_exp"), Qwen3_5MoeMixin)

    def test_serving_fails_before_weight_loading_without_experimental_opt_in(self):
        model = Qwen4Exp.__new__(Qwen4Exp)
        with mock.patch.dict(
            os.environ, {"RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "false"}
        ), mock.patch.object(Qwen35Moe, "load") as parent_load:
            with self.assertRaisesRegex(RuntimeError, "disabled by default"):
                model.load()
        parent_load.assert_not_called()

    def test_experimental_opt_in_reaches_the_parent_loader(self):
        model = Qwen4Exp.__new__(Qwen4Exp)
        with mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true",
                "RTP_LLM_ENABLE_QWEN4_EXP_QSA": "false",
            },
        ), mock.patch.object(Qwen35Moe, "load", return_value="loaded") as parent_load:
            self.assertEqual(model.load(skip_python_model=True), "loaded")
        parent_load.assert_called_once_with(skip_python_model=True)

    @staticmethod
    def _qsa_enabled_model(**config_overrides):
        model = Qwen4Exp.__new__(Qwen4Exp)
        config = dict(
            enable_qwen4_qsa=True,
            enable_qwen4_ple=False,
            data_type=DataType.TYPE_BF16,
            is_mtp=False,
            quant_config=None,
            attn_config=SimpleNamespace(
                is_sparse=True,
                use_sparse_gqa_fmha=True,
                kv_cache_dtype=KvCacheDataType.BASE,
            ),
        )
        config.update(config_overrides)
        model.model_config = SimpleNamespace(**config)
        model.parallelism_config = SimpleNamespace(
            role_type=RoleType.PDFUSION,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False,
                is_prefill_enabled=lambda: False,
            ),
        )
        return model

    def test_experimental_qsa_opt_in_reaches_the_parent_loader(self):
        model = self._qsa_enabled_model()
        with mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true",
                # The load guard consumes the parsed config, not a mutable env.
                "RTP_LLM_ENABLE_QWEN4_EXP_QSA": "false",
            },
        ), mock.patch.object(Qwen35Moe, "load", return_value="loaded") as parent_load:
            self.assertEqual(model.load(skip_python_model=True), "loaded")
        parent_load.assert_called_once_with(skip_python_model=True)

    def test_experimental_qsa_mtp_reaches_the_parent_loader(self):
        model = self._qsa_enabled_model(is_mtp=True)
        with mock.patch.dict(
            os.environ,
            {"RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true"},
        ), mock.patch.object(Qwen35Moe, "load", return_value="loaded") as parent_load:
            self.assertEqual(model.load(skip_python_model=True), "loaded")
        parent_load.assert_called_once_with(skip_python_model=True)

    def test_qsa_rejects_unsupported_modes_before_weight_loading(self):
        cases = (
            (
                "act_type=BF16",
                dict(data_type=DataType.TYPE_FP16),
                {},
            ),
            (
                "quantized indexer",
                dict(quant_config=SimpleNamespace(is_quanted=lambda: True)),
                {},
            ),
            (
                "base BF16 KV cache",
                dict(
                    attn_config=SimpleNamespace(
                        is_sparse=True,
                        use_sparse_gqa_fmha=True,
                        kv_cache_dtype=KvCacheDataType.FP8,
                    )
                ),
                {},
            ),
            (
                "routing flags are inconsistent",
                dict(
                    attn_config=SimpleNamespace(
                        is_sparse=True,
                        use_sparse_gqa_fmha=False,
                        kv_cache_dtype=KvCacheDataType.BASE,
                    )
                ),
                {},
            ),
            ("PD-separated", {}, dict(role_type=RoleType.DECODE)),
            (
                "context/prefill parallelism",
                {},
                dict(
                    prefill_cp_config=SimpleNamespace(
                        is_enabled=lambda: True,
                        is_prefill_enabled=lambda: False,
                    )
                ),
            ),
        )
        for expected, config_overrides, parallel_overrides in cases:
            with self.subTest(expected=expected):
                model = self._qsa_enabled_model(**config_overrides)
                for name, value in parallel_overrides.items():
                    setattr(model.parallelism_config, name, value)
                with mock.patch.dict(
                    os.environ,
                    {"RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true"},
                ), mock.patch.object(Qwen35Moe, "load") as parent_load:
                    with self.assertRaisesRegex(RuntimeError, expected):
                        model.load()
                parent_load.assert_not_called()

    def test_cuda_graph_remains_disabled(self):
        model = Qwen4Exp.__new__(Qwen4Exp)
        model.model_config = mock.Mock(enable_qwen4_ple=False, enable_qwen4_qsa=False)
        self.assertFalse(model.support_cuda_graph())

        model.model_config.enable_qwen4_ple = True
        self.assertFalse(model.support_cuda_graph())

        model.model_config.enable_qwen4_ple = False
        model.model_config.enable_qwen4_qsa = True
        self.assertFalse(model.support_cuda_graph())

    def test_ple_rejects_non_bf16_before_weight_loading(self):
        model = Qwen4Exp.__new__(Qwen4Exp)
        model.model_config = mock.Mock(
            enable_qwen4_qsa=False,
            enable_qwen4_ple=True,
            data_type=DataType.TYPE_FP16,
        )
        with mock.patch.dict(
            os.environ, {"RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true"}
        ), mock.patch.object(Qwen35Moe, "load") as parent_load:
            with self.assertRaisesRegex(RuntimeError, "requires act_type=BF16"):
                model.load()
        parent_load.assert_not_called()

    def test_ple_rejects_invalid_tp_split_before_weight_loading(self):
        model = Qwen4Exp.__new__(Qwen4Exp)
        model.model_config = mock.Mock(
            enable_qwen4_qsa=False,
            enable_qwen4_ple=True,
            data_type=DataType.TYPE_BF16,
            _qwen4_split_ngram_parts=128,
        )
        model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 1,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False,
                is_prefill_enabled=lambda: False,
            ),
        )
        with mock.patch.dict(
            os.environ, {"RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true"}
        ), mock.patch.object(Qwen35Moe, "load") as parent_load:
            with self.assertRaisesRegex(RuntimeError, "attention TP > 1"):
                model.load()
        parent_load.assert_not_called()

    def test_ple_rejects_cp_before_weight_loading(self):
        model = Qwen4Exp.__new__(Qwen4Exp)
        model.model_config = mock.Mock(
            enable_qwen4_qsa=False,
            enable_qwen4_ple=True,
            data_type=DataType.TYPE_BF16,
            _qwen4_split_ngram_parts=128,
        )
        model.parallelism_config = SimpleNamespace(
            tp_size=8,
            get_attn_tp_size=lambda: 1,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: True,
                is_prefill_enabled=lambda: False,
            ),
        )
        with mock.patch.dict(
            os.environ, {"RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true"}
        ), mock.patch.object(Qwen35Moe, "load") as parent_load:
            with self.assertRaisesRegex(RuntimeError, "does not support context"):
                model.load()
        parent_load.assert_not_called()

    def test_ple_rejects_pd_roles_before_weight_loading(self):
        for role_type in (RoleType.PREFILL, RoleType.DECODE):
            with self.subTest(role_type=role_type):
                model = Qwen4Exp.__new__(Qwen4Exp)
                model.model_config = mock.Mock(
                    enable_qwen4_qsa=False,
                    enable_qwen4_ple=True,
                    data_type=DataType.TYPE_BF16,
                    _qwen4_split_ngram_parts=128,
                )
                model.parallelism_config = SimpleNamespace(
                    role_type=role_type,
                    get_attn_tp_size=lambda: 8,
                    prefill_cp_config=SimpleNamespace(
                        is_enabled=lambda: False,
                        is_prefill_enabled=lambda: False,
                    ),
                )
                with mock.patch.dict(
                    os.environ,
                    {"RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL": "true"},
                ), mock.patch.object(Qwen35Moe, "load") as parent_load:
                    with self.assertRaisesRegex(RuntimeError, "PD-separated"):
                        model.load()
                parent_load.assert_not_called()

    def test_ple_weight_size_uses_attention_tp_not_ep(self):
        with mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_ENABLE_QWEN4_EXP_PLE": "true",
                "RTP_LLM_ENABLE_QWEN4_EXP_QSA": "false",
            },
        ):
            config = Qwen4Exp.create_config(self._temp_dir.name)
        # Production injects this through build_model_config() before invoking
        # the multimodal-aware size estimator. This focused parser test mirrors
        # that lifecycle without constructing the full server configuration.
        config.model_type = "qwen4_exp"

        table = config._tp_sharded_extra_weight_bytes
        replicated = config._replicated_extra_weight_bytes
        # The nominal n-gram table alone is 20M * 2560 * BF16 = 102.4GB.
        self.assertEqual(table, 102_400_000_000)
        self.assertEqual(replicated, 65_679_640)
        self.assertEqual(config._extra_weight_bytes, table + replicated)
        # The temporary checkpoint deliberately contains no tokenizer/image
        # processor assets. Isolate the PLE residency formula from the
        # independently tested multimodal parameter estimator.
        with mock.patch.object(Qwen3_5MoeMixin, "eval_mm_model_size", return_value=0):
            total = config.eval_model_weight_size()
            generic = total - config._extra_weight_bytes
            self.assertAlmostEqual(
                config.eval_model_weight_size_per_rank(tp_size=8, ep_size=16),
                generic / 16 + table / 8 + replicated,
            )

    def test_ple_checkpoint_size_is_reported_when_runtime_gate_is_off(self):
        self.assertFalse(self.config.enable_qwen4_ple)
        self.assertEqual(self.config._tp_sharded_extra_weight_bytes, 0)
        self.assertEqual(self.config._replicated_extra_weight_bytes, 0)
        self.assertEqual(self.config._extra_weight_bytes, 102_465_679_640)
        self.assertGreater(self.config._extra_weight_param_count, 51_200_000_000)

    def test_basic_config(self):
        config = self.config
        self.assertEqual(config.num_layers, _NUM_LAYERS)
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.vocab_size, 248320)
        self.assertEqual(config.max_seq_len, 262144)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.attn_config.head_num, 24)
        self.assertEqual(config.attn_config.kv_head_num, 2)
        self.assertEqual(config.attn_config.size_per_head, 256)
        self.assertEqual(config.layernorm_eps, 1e-6)
        self.assertTrue(config.qk_norm)

    def test_rope_config(self):
        rope_config = self.config.attn_config.rope_config
        self.assertEqual(rope_config.style, RopeStyle.Mrope)
        self.assertEqual(rope_config.base, 10_000_000)
        self.assertEqual(self.config.partial_rotary_factor, 0.25)
        # partial rotary: 256 * 0.25 == 64, so mrope_section must sum to 32 pairs.
        self.assertEqual(rope_config.dim, 64)
        self.assertEqual(rope_config.index_factor, 3)
        self.assertEqual(
            [rope_config.mrope_dim1, rope_config.mrope_dim2, rope_config.mrope_dim3],
            [11, 11, 10],
        )
        self.assertTrue(rope_config.mrope_interleaved)
        self.assertEqual(self.config.mm_model_config.mm_position_ids_style, 2)

    def test_linear_attn_norm_activation(self):
        # Upstream falls back to hidden_act when output_gate_type is unset
        # (this trimmed config has neither -> generic silu default).
        self.assertEqual(self.config.linear_attn_norm_activation, "silu")
        config_json = self._config()
        config_json["text_config"]["output_gate_type"] = "sigmoid"
        Path(self._temp_dir.name, "config.json").write_text(json.dumps(config_json))
        with mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_ENABLE_QWEN4_EXP_PLE": "false",
                "RTP_LLM_ENABLE_QWEN4_EXP_QSA": "false",
            },
        ):
            config = Qwen4Exp.create_config(self._temp_dir.name)
        self.assertEqual(config.linear_attn_norm_activation, "sigmoid")

    def test_moe_config(self):
        config = self.config
        self.assertEqual(config.expert_num, 512)
        self.assertEqual(config.moe_k, 10)
        self.assertEqual(config.moe_inter_size, 640)
        self.assertEqual(config.inter_size, 640)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.moe_style, 2)
        # No decoder_sparse_step in the checkpoint config: every layer is MoE.
        self.assertEqual(list(config.moe_layer_index), list(range(_NUM_LAYERS)))

    def test_layer_types_drive_hybrid_topology(self):
        hybrid_config = self.config.hybrid_attention_config
        self.assertTrue(hybrid_config.enable_hybrid_attention)
        layer_types = list(hybrid_config.hybrid_attention_types)
        self.assertEqual(len(layer_types), _NUM_LAYERS)
        self.assertEqual(layer_types.count(HybridAttentionType.NONE), 12)
        self.assertEqual(layer_types.count(HybridAttentionType.LINEAR), 36)
        for idx, layer_type in enumerate(layer_types):
            expected = (
                HybridAttentionType.NONE
                if (idx + 1) % 4 == 0
                else HybridAttentionType.LINEAR
            )
            self.assertEqual(layer_type, expected, f"layer {idx}")

    def test_linear_attention_config(self):
        linear_config = self.config.linear_attention_config
        self.assertEqual(linear_config.linear_num_key_heads, 16)
        self.assertEqual(linear_config.linear_num_value_heads, 48)
        self.assertEqual(linear_config.linear_key_head_dim, 128)
        self.assertEqual(linear_config.linear_value_head_dim, 128)
        self.assertEqual(linear_config.linear_conv_kernel_dim, 4)

    def test_multimodal_config(self):
        mm_config = self.config.mm_model_config
        self.assertTrue(mm_config.is_multimodal)
        self.assertEqual(
            [list(pair) for pair in mm_config.mm_sep_tokens], [[248053, 248054]]
        )
        self.assertEqual(
            self.config.mm_related_params.config["ckpt_path"], self._temp_dir.name
        )

    def test_kv_cache_spec_descs(self):
        descs = self.config.kv_cache_spec_descs
        self.assertEqual(len(descs), _NUM_LAYERS)
        tags = [layer_descs[0].tag for layer_descs in descs]
        self.assertEqual(tags.count("full"), 12)
        self.assertEqual({tags.count(f"linear{i}") for i in range(3)}, {12})
        for tag, layer_descs in zip(tags, descs):
            expected = KVCacheSpecType.MHA if tag == "full" else KVCacheSpecType.LINEAR
            self.assertEqual(layer_descs[0].cache_type, expected)
            self.assertEqual(len(layer_descs), 1)

    def test_unfinished_subsystems_are_disabled_by_default(self):
        self.assertFalse(self.config.enable_qwen4_ple)
        self.assertFalse(self.config.enable_qwen4_qsa)
        self.assertFalse(self.config.attn_config.is_sparse)
        self.assertFalse(self.config.attn_config.use_sparse_gqa_fmha)

    def test_qsa_routing_flag_propagates_through_real_attention_config(self):
        with mock.patch.dict(
            os.environ,
            {
                "RTP_LLM_ENABLE_QWEN4_EXP_PLE": "false",
                "RTP_LLM_ENABLE_QWEN4_EXP_QSA": "true",
            },
        ):
            config = Qwen4Exp.create_config(self._temp_dir.name)

        self.assertTrue(config.enable_qwen4_qsa)
        self.assertTrue(config.attn_config.is_sparse)
        self.assertTrue(config.attn_config.use_sparse_gqa_fmha)
        local_attn_config = config.getAttentionConfigs(8)
        self.assertTrue(local_attn_config.use_sparse_gqa_fmha)

    def test_disabled_qsa_does_not_request_indexer_weights(self):
        info = self._weight()._get_weight_info()
        requested = set()
        for layer in info.layer_weights:
            requested |= _ckpt_keys(layer)

        self.assertFalse([key for key in requested if ".indexer." in key])

    def test_enabled_qsa_requests_indexer_only_on_full_attention_layers(self):
        self.config.enable_qwen4_qsa = True
        info = self._weight()._get_weight_info()

        linear_keys = _ckpt_keys(info.layer_weights[0])
        full_keys = _ckpt_keys(info.layer_weights[3])
        self.assertFalse([key for key in linear_keys if ".indexer." in key])
        self.assertEqual(len([key for key in full_keys if ".indexer." in key]), 3)

    def test_hc_mult_is_plumbed_from_hc_count(self):
        self.assertEqual(self.config.hc_mult, 4)
        self.assertEqual(
            self.config.mtp_input_hidden_size,
            self.config.hc_mult * self.config.hidden_size,
        )
        self.assertEqual(
            self.config.getMtpInputHiddenSize(),
            self.config.hc_mult * self.config.hidden_size,
        )

    def _weight(self):
        weight = Qwen4ExpWeight(
            model_config=self.config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )
        weight._process_meta({}, _CKPT_KEYS)
        return weight

    def test_prefix_is_detected_from_gated_residual_anchor(self):
        # The Qwen3.5 anchor (layers.0.input_layernorm.weight) does not exist here.
        self.assertEqual(self._weight().prefix, "model.language_model.")
        self.assertTrue(self._weight()._has_stacked_ckpt)

    def test_prefix_detection_rejects_a_checkpoint_without_the_anchor(self):
        weight = Qwen4ExpWeight(
            model_config=self.config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )

        with self.assertRaisesRegex(ValueError, "cannot determine prefix"):
            weight._process_meta({}, ["model.language_model.embed_tokens.weight"])

    def test_gated_residual_weights_cover_every_hc_tensor(self):
        weight = self._weight()

        info = weight._get_weight_info()

        global_keys = _ckpt_keys(info.weights)
        self.assertEqual(
            global_keys,
            {
                "lm_head.weight",
                "model.language_model.embed_tokens.weight",
                "model.language_model.hyper_connection_mixer.hc_norm.weight",
                "model.language_model.hyper_connection_mixer."
                "input_mix_weight_down.weight",
                "model.language_model.hyper_connection_mixer."
                "input_mix_weight_up.weight",
            },
        )

        self.assertEqual(len(info.layer_weights), _NUM_LAYERS)
        for layer_id in (0, 3):
            layer_keys = _ckpt_keys(info.layer_weights[layer_id])
            for module_name in ("attn_hyper_connection", "mlp_hyper_connection"):
                prefix = f"model.language_model.layers.{{i}}.{module_name}."
                for suffix in (
                    "hc_norm.weight",
                    "input_mix_weight_down.weight",
                    "input_mix_weight_up.weight",
                    "block_inject_weight.weight",
                ):
                    self.assertIn(prefix + suffix, layer_keys)

    def test_no_layer_norm_keys_are_requested(self):
        info = self._weight()._get_weight_info()

        requested = _ckpt_keys(info.weights)
        for layer in info.layer_weights:
            requested |= _ckpt_keys(layer)

        # These three exist in Qwen3.5 but not in this checkpoint; requesting any
        # of them would fail the load.
        for absent in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "model.language_model.norm.weight",
        ):
            self.assertFalse(
                [key for key in requested if key.endswith(absent)],
                f"mapping must not request {absent}",
            )

    def test_linear_and_full_attention_layers_use_the_right_branch(self):
        info = self._weight()._get_weight_info()

        linear_keys = _ckpt_keys(info.layer_weights[0])
        full_keys = _ckpt_keys(info.layer_weights[3])

        self.assertIn(
            "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight",
            linear_keys,
        )
        self.assertNotIn(
            "model.language_model.layers.{i}.self_attn.q_proj.weight", linear_keys
        )
        self.assertIn(
            "model.language_model.layers.{i}.self_attn.q_proj.weight", full_keys
        )
        self.assertNotIn(
            "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight", full_keys
        )

    @staticmethod
    def _config():
        """Trimmed Qwen/Qwen3.8-Flash-Next config.json."""
        return {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "image_token_id": 248056,
            "model_type": "qwen4_exp",
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "num_hidden_layers": _NUM_LAYERS,
                "hidden_size": 2560,
                "vocab_size": 248320,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
                "full_attention_interval": 4,
                "layer_types": [
                    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                    for i in range(_NUM_LAYERS)
                ],
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "hc_count": 4,
                "hc_lowrank": 320,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "indexer_head_dim": 128,
                "indexer_kv_heads": 1,
                "indexer_n_heads": 4,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20000000,
                "ple_conv_kernel_size": 4,
                "ple_embed_dim": 2560,
                "ple_layer_ids": [2],
                "split_ngram_parts": 128,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default",
                },
            },
            "vision_config": {
                "deepstack_visual_indexes": [],
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "patch_size": 16,
            },
        }


if __name__ == "__main__":
    unittest.main()
