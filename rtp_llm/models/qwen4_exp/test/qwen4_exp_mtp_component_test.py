import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_factory_register import _model_factory, get_lazy_model_module_path
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models.qwen4_exp.qwen4_exp_mtp import Qwen4ExpMTP, Qwen4ExpMTPWeight
from rtp_llm.ops import (
    HWKernelConfig,
    HybridAttentionType,
    KVCacheSpecType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.utils.model_weight import W

# Shapes were read from the released Qwen/Qwen3.8-Flash-Next safetensors
# headers.  This is the complete mtp.* namespace: 7 globals, 8 residual-unit
# tensors, 7 MoE tensors, and 9 attention/indexer tensors.
_REAL_MTP_MANIFEST = {
    "mtp.fc_embedding.weight": (2560, 2560),
    "mtp.fc_hidden.weight": (2560, 2560),
    "mtp.hyper_connection_mixer.hc_norm.weight": (10240,),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (320, 10240),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (10240, 320),
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": (4, 10240),
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": (10240,),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": (
        320,
        10240,
    ),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": (10240, 320),
    "mtp.layers.0.mlp.experts.down_proj": (512, 2560, 640),
    "mtp.layers.0.mlp.experts.gate_up_proj": (512, 1280, 2560),
    "mtp.layers.0.mlp.gate.weight": (512, 2560),
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": (2560, 640),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (640, 2560),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": (640, 2560),
    "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 2560),
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": (4, 10240),
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": (10240,),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": (320, 10240),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": (10240, 320),
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": (640, 2560),
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": (128,),
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": (128,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (512, 2560),
    "mtp.layers.0.self_attn.o_proj.weight": (2560, 6144),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.q_proj.weight": (12288, 2560),
    "mtp.layers.0.self_attn.v_proj.weight": (512, 2560),
    "mtp.pre_fc_norm_embedding.weight": (2560,),
    "mtp.pre_fc_norm_hidden.weight": (10240,),
}


def _atomic_modules(modules):
    for module in modules:
        if isinstance(module, CompositeWeight):
            yield from _atomic_modules(module.sub_weights.values())
        else:
            yield module


def _source_keys(modules, layer_id=None):
    return {
        info.tensor_name(layer_id)
        for module in _atomic_modules(modules)
        for info in module.weights
    }


class Qwen4ExpMTPComponentTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        Path(self._temp_dir.name, "config.json").write_text(json.dumps(self._config()))
        with mock.patch.dict(
            os.environ,
            {
                # PLE is target-only, while QSA is mandatory for a serviceable
                # draft and must retain the explicit process opt-in.
                "RTP_LLM_ENABLE_QWEN4_EXP_PLE": "true",
                "RTP_LLM_ENABLE_QWEN4_EXP_QSA": "true",
            },
        ):
            self.config = Qwen4ExpMTP.create_config(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def _weight(self):
        return Qwen4ExpMTPWeight(
            model_config=self.config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
        )

    def test_component_is_registered_and_qsa_load_reaches_parent(self):
        self.assertIs(_model_factory["qwen4_exp_mtp"], Qwen4ExpMTP)
        self.assertEqual(
            get_lazy_model_module_path("qwen4_exp_mtp"),
            "rtp_llm.models.qwen4_exp.qwen4_exp_mtp",
        )
        model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
        model.model_config = self.config
        with mock.patch.object(Qwen4Exp, "load") as parent_load:
            model.load(skip_python_model=True)
        parent_load.assert_called_once_with(skip_python_model=True)

    def test_load_rejects_disabled_qsa_before_parent(self):
        model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
        model.model_config = SimpleNamespace(enable_qwen4_qsa=False)
        with mock.patch.object(Qwen4Exp, "load") as parent_load:
            with self.assertRaisesRegex(RuntimeError, "dense draft fallback"):
                model.load()
        parent_load.assert_not_called()

    def test_draft_config_is_one_full_attention_layer_without_ple_or_graph(self):
        self.assertEqual(self.config.model_type, "qwen4_exp_mtp")
        self.assertEqual(self.config.num_layers, 1)
        self.assertEqual(self.config.moe_layer_index, [0])
        self.assertTrue(self.config.is_mtp)
        self.assertEqual(
            self.config.mtp_input_hidden_size,
            self.config.hc_mult * self.config.hidden_size,
        )
        self.assertEqual(
            self.config.hybrid_attention_config.hybrid_attention_types,
            [HybridAttentionType.NONE],
        )
        self.assertFalse(self.config.enable_qwen4_ple)
        self.assertEqual(self.config._qwen4_ple_layer_ids, [])
        self.assertTrue(self.config.enable_qwen4_qsa)
        self.assertTrue(self.config.attn_config.is_sparse)
        self.assertTrue(self.config.attn_config.use_sparse_gqa_fmha)
        self.assertFalse(self.config.mm_model_config.is_multimodal)
        self.assertEqual(self.config.attn_config.rope_config.style, RopeStyle.Base)
        self.assertEqual(self.config.attn_config.rope_config.base, 10_000_000)
        self.assertEqual(len(self.config.kv_cache_spec_descs), 1)
        self.assertEqual(len(self.config.kv_cache_spec_descs[0]), 3)
        self.assertEqual(
            self.config.kv_cache_spec_descs[0][0].cache_type,
            KVCacheSpecType.MHA,
        )
        self.assertEqual(
            [desc.tag for desc in self.config.kv_cache_spec_descs[0][1:]],
            [INDEXER_KV_TAG, INDEXER_STATE_TAG],
        )
        self.assertTrue(
            self.config.hybrid_attention_config.enable_independent_kv_cache_pools
        )
        self.assertFalse(Qwen4ExpMTP.__new__(Qwen4ExpMTP).support_cuda_graph())

    def test_real_31_key_manifest_is_completely_and_only_mapped(self):
        self.assertEqual(len(_REAL_MTP_MANIFEST), 31)
        weight = self._weight()
        weight._process_meta({}, set(_REAL_MTP_MANIFEST))
        info = weight._get_weight_info()
        requested = {
            key for key in _source_keys(info.weights) if key.startswith("mtp.")
        }
        requested |= {
            key
            for key in _source_keys(info.layer_weights[0], layer_id=0)
            if key.startswith("mtp.")
        }
        self.assertEqual(requested, set(_REAL_MTP_MANIFEST))
        self.assertEqual(weight.prefix, "mtp.")
        self.assertTrue(weight._has_stacked_ckpt)

    def test_fc_weights_have_independent_runtime_tags_and_transpose(self):
        weight = self._weight()
        globals_by_name = {
            module.name: module
            for module in _atomic_modules(weight._create_global_weights())
        }
        embedding_fc = globals_by_name[W.qwen4_mtp_fc_embedding_w]
        hidden_fc = globals_by_name[W.qwen4_mtp_fc_hidden_w]
        self.assertNotEqual(embedding_fc.name, hidden_fc.name)
        self.assertEqual(embedding_fc.weights[0].name, "mtp.fc_embedding.weight")
        self.assertEqual(hidden_fc.weights[0].name, "mtp.fc_hidden.weight")
        source = torch.arange(6).reshape(2, 3)
        torch.testing.assert_close(
            embedding_fc.process_fun([source]), source.transpose(0, 1).contiguous()
        )
        torch.testing.assert_close(
            hidden_fc.process_fun([source]), source.transpose(0, 1).contiguous()
        )

    def test_pre_fc_norm_gamma_is_not_folded_in_bf16(self):
        globals_by_name = {
            module.name: module
            for module in _atomic_modules(self._weight()._create_global_weights())
        }
        gamma = torch.tensor([0.003], dtype=torch.bfloat16)
        for name in (W.multi_tokens_predict_enorm, W.multi_tokens_predict_hnorm):
            mapped = globals_by_name[name].process_fun([gamma])
            self.assertTrue(torch.equal(mapped, gamma))
            self.assertFalse(torch.equal(mapped, gamma + 1))

    def test_target_vocabulary_aliases_require_semantic_compatibility(self):
        draft = self.config
        target = Qwen4Exp.__new__(Qwen4Exp)
        target.model_config = SimpleNamespace(
            **{
                name: getattr(draft, name)
                for name in (
                    "vocab_size",
                    "hidden_size",
                    "data_type",
                    "enable_fp32_lm_head",
                )
            }
        )
        self.assertEqual(
            Qwen4ExpMTP.speculative_weight_alias_names(target, draft),
            (W.embedding, W.lm_head),
        )

        target.model_config.hidden_size += 1
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            Qwen4ExpMTP.speculative_weight_alias_names(target, draft)

        target.model_config.hidden_size = draft.hidden_size
        draft._qwen4_mtp_use_dedicated_embeddings = True
        with self.assertRaisesRegex(ValueError, "dedicated"):
            Qwen4ExpMTP.speculative_weight_alias_names(target, draft)

        draft._qwen4_mtp_use_dedicated_embeddings = False
        target.model_config.enable_output_vocab_pruning = True
        with self.assertRaisesRegex(ValueError, "output-vocab pruning"):
            Qwen4ExpMTP.speculative_weight_alias_names(target, draft)

        target.model_config.enable_output_vocab_pruning = False
        draft.enable_output_vocab_pruning = True
        with self.assertRaisesRegex(ValueError, "output-vocab pruning"):
            Qwen4ExpMTP.speculative_weight_alias_names(target, draft)

    def test_rejects_nonfinal_target_hidden_source(self):
        config = self._config()
        config["text_config"]["mtp"]["mtp_use_hidden_state_from_layer"] = 17
        Path(self._temp_dir.name, "config.json").write_text(json.dumps(config))

        with self.assertRaisesRegex(ValueError, "final hidden state"):
            Qwen4ExpMTP.create_config(self._temp_dir.name)

    @staticmethod
    def _config():
        num_layers = 48
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
                "num_hidden_layers": num_layers,
                "hidden_size": 2560,
                "vocab_size": 248320,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
                "layer_types": [
                    "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                    for i in range(num_layers)
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
                    "rope_theta": 10_000_000,
                    "rope_type": "default",
                },
                "mtp": {
                    "hybrid": True,
                    "layer_types": ["full_attention"],
                    "mtp_use_hidden_state_from_layer": None,
                    "num_hidden_layers": 1,
                    "rope_theta": 10_000_000,
                },
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": False,
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
