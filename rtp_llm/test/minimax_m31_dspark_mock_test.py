import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.minimax_m3 import MiniMaxM3
from rtp_llm.models.minimax_m3_vl import MiniMaxM3_VL
from rtp_llm.models.minimax_m31 import MiniMaxM31, MiniMaxM31Weight
from rtp_llm.models.minimax_m31_dspark import MiniMaxM31DSpark, MiniMaxM31DSparkWeight
from rtp_llm.models.minimax_m31_vl import MiniMaxM31_VL
from rtp_llm.openai.renderer_factory_register import _renderer_type_to_module


def _m31_config():
    layer_count = 60
    return {
        "model_type": "minimax_m3_vl",
        "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        "quantization_config": {
            "quant_method": "mxfp8",
            "moe_quant_algo": "NVFP4",
            "moe_quant_format": "nvfp4-pack-quantized",
        },
        "text_config": {
            "hidden_size": 6144,
            "head_dim": 128,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "vocab_size": 200064,
            "num_hidden_layers": layer_count,
            "intermediate_size": 3072,
            "dense_intermediate_size": 12288,
            "shared_intermediate_size": 3072,
            "num_local_experts": 128,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "moe_layer_freq": [0, 0, 0] + [1] * 57,
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 128,
                "sparse_num_index_heads": 4,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_attention_freq": [1] * layer_count,
                "sparse_disable_index_value": [1] * layer_count,
                "sparse_init_block": 1,
                "sparse_local_block": 1,
            },
        },
    }


class MiniMaxM31ConfigTest(unittest.TestCase):
    def test_dspark_has_only_m31_python_identity(self):
        self.assertEqual(MiniMaxM31DSpark.__name__, "MiniMaxM31DSpark")
        self.assertEqual(MiniMaxM31DSparkWeight.__name__, "MiniMaxM31DSparkWeight")

    def test_auto_model_version_defaults_to_m31(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                ModelDict.get_ft_model_type_by_config(_m31_config()),
                "minimax_m31_vl",
            )

    def test_auto_model_version_can_select_legacy_m3(self):
        with patch.dict(os.environ, {"MINIMAX_M3_VERSION": "3"}, clear=True):
            self.assertEqual(
                ModelDict.get_ft_model_type_by_config(_m31_config()),
                "minimax_m3_vl",
            )

    def test_auto_model_version_rejects_unknown_value(self):
        with patch.dict(os.environ, {"MINIMAX_M3_VERSION": "latest"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MINIMAX_M3_VERSION"):
                ModelDict.get_ft_model_type_by_config(_m31_config())

    def test_m31_frontend_reuses_m3_renderers(self):
        self.assertEqual(
            _renderer_type_to_module["minimax_m31"],
            "rtp_llm.openai.renderers.minimax_m3_renderer",
        )
        self.assertEqual(
            _renderer_type_to_module["minimax_m31_vl"],
            "rtp_llm.openai.renderers.minimax_m3_vl_renderer",
        )

    def test_real_shape_and_explicit_mock_gate(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(_m31_config()))
            with patch.dict(os.environ, {"M3_M31_MOCK_NVFP4_MOE": "1"}):
                config = MiniMaxM31._create_config(tmpdir)

        self.assertEqual(config.num_layers, 60)
        self.assertEqual(config.model_type, "minimax_m31")
        self.assertEqual(config.moe_layer_index, list(range(3, 60)))
        self.assertEqual(config.msa_sparse_config["sparse_layer_ids"], list(range(60)))
        self.assertTrue(config.mock_nvfp4_moe)

    def test_vl_registration_path_preserves_mock_gate(self):
        raw = _m31_config()
        raw["vision_config"] = {"hidden_size": 1152}
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(raw))
            with patch.dict(os.environ, {"M3_M31_MOCK_NVFP4_MOE": "1"}):
                config = MiniMaxM31_VL._create_config(tmpdir)

        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.model_type, "minimax_m31_vl")
        self.assertTrue(config.mock_nvfp4_moe)
        self.assertEqual(config.msa_sparse_config["sparse_layer_ids"], list(range(60)))

    def test_mock_gate_does_not_affect_non_nvfp4_draft(self):
        raw = _m31_config()
        raw["quantization_config"].pop("moe_quant_algo")
        raw["quantization_config"].pop("moe_quant_format")
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(raw))
            with patch.dict(os.environ, {"M3_M31_MOCK_NVFP4_MOE": "1"}):
                config = MiniMaxM31._create_config(tmpdir)

        self.assertFalse(config.mock_nvfp4_moe)

    def test_legacy_m3_does_not_parse_m31_mock_state(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(_m31_config()))
            with patch.dict(os.environ, {"M3_M31_MOCK_NVFP4_MOE": "1"}):
                text_config = MiniMaxM3._create_config(tmpdir)
                vl_config = MiniMaxM3_VL._create_config(tmpdir)

        self.assertFalse(text_config.mock_nvfp4_moe)
        self.assertFalse(vl_config.mock_nvfp4_moe)


class MiniMaxM31WeightContractTest(unittest.TestCase):
    @staticmethod
    def _weight():
        weight = object.__new__(MiniMaxM31Weight)
        weight._load_raw_mxfp8_idx = False
        weight._native_mxfp4_routed = False
        weight._prepacked_nvfp4_routed = False
        weight._mock_nvfp4_moe = False
        weight.prefix = "language_model."
        return weight

    @staticmethod
    def _keys():
        return {
            "language_model.model.layers.0.self_attn.index_q_proj.weight",
            "language_model.model.layers.3.self_attn.index_q_proj.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_packed",
            "language_model.model.layers.3.block_sparse_moe.e_score_correction_bias",
        }

    def test_prepacked_nvfp4_fails_closed_by_default(self):
        weight = self._weight()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "not supported yet"):
                weight._process_meta([], self._keys())

    def test_prepacked_nvfp4_is_mocked_only_when_requested(self):
        weight = self._weight()
        with patch.dict(os.environ, {"M3_M31_MOCK_NVFP4_MOE": "1"}, clear=True):
            weight._process_meta([], self._keys())

        self.assertTrue(weight._prepacked_nvfp4_routed)
        self.assertTrue(weight._mock_nvfp4_moe)
        self.assertEqual(weight._sparse_layer_set, {0, 3})

    def test_mock_weight_contract_keeps_only_shared_expert(self):
        weight = self._weight()
        weight._align_size = 0
        weight._is_gated_activation = True
        weight.moe_layer_index_ = [3]
        weight.expert_num_ = 128
        weight.has_e_score_correction_bias = True
        weight._prepacked_nvfp4_routed = True
        weight._mock_nvfp4_moe = True

        modules = weight._get_hf_ffn_layer_weight_info(3)
        checkpoint_names = {
            checkpoint_weight.name
            for module in modules
            for component in module.get_components()
            for checkpoint_weight in (getattr(component, "weights", None) or [])
        }

        self.assertEqual(len(modules), 1)
        self.assertEqual(
            checkpoint_names,
            {
                "language_model.model.layers.{i}.block_sparse_moe."
                "shared_experts.gate_proj.weight",
                "language_model.model.layers.{i}.block_sparse_moe."
                "shared_experts.down_proj.weight",
                "language_model.model.layers.{i}.block_sparse_moe."
                "shared_experts.up_proj.weight",
            },
        )
        self.assertFalse(
            any("experts.{expert_id}" in name for name in checkpoint_names)
        )


if __name__ == "__main__":
    unittest.main()
