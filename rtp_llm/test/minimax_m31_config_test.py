import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.model_factory_register import ModelDict
from rtp_llm.model_loader.ffn_weight import FfnWeight, MoeWeight
from rtp_llm.models.minimax_m3 import MiniMaxM3, MiniMaxM3Weight
from rtp_llm.models.minimax_m3_vl import MiniMaxM3_VL
from rtp_llm.models.minimax_m31 import MiniMaxM31, MiniMaxM31Weight
from rtp_llm.models.minimax_m31_dspark import MiniMaxM31DSpark, MiniMaxM31DSparkWeight
from rtp_llm.models.minimax_m31_vl import MiniMaxM31_VL
from rtp_llm.openai.renderer_factory_register import _renderer_type_to_module
from rtp_llm.utils.model_weight import W


class DSparkBackendTest(unittest.TestCase):
    def test_draft_gemma_weights_stay_raw_and_main_loader_keeps_offset(self):
        from rtp_llm.models.minimax_m3 import add_unit_offset
        from rtp_llm.utils.model_weight import identity

        norm_keys = (W.pre_ln_gamma, W.post_ln_gamma, W.q_ln_gamma, W.k_ln_gamma)
        raw = torch.tensor([-0.99609375, 0.37109375], dtype=torch.bfloat16)
        for cls in (MiniMaxM3Weight, MiniMaxM31DSparkWeight):
            weight = object.__new__(cls)
            weight.prefix = "language_model."
            weight._hidden_size = 6144
            weight._size_per_head = 128
            weight._head_num = 64
            weight._head_num_kv = 4
            weight._use_qk_norm = True
            weight._sparse_layer_set = set()
            with patch.object(
                weight, "_get_hf_ffn_layer_weight_info", return_value=[]
            ), patch.object(weight, "_should_load_msa_index", return_value=False):
                for layer_id in range(5):
                    modules = weight._get_hf_layer_weight_info(layer_id)
                    components = {
                        c.name: c for m in modules for c in m.get_components()
                    }
                    for name in norm_keys:
                        with self.subTest(
                            model=cls.__name__, layer=layer_id, norm=name
                        ):
                            component = components[name]
                            transform = (
                                identity
                                if cls is MiniMaxM31DSparkWeight
                                else add_unit_offset
                            )
                            self.assertIs(component.process_fun, transform)
                            torch.testing.assert_close(
                                component.process_fun([raw]),
                                transform([raw]),
                                rtol=0,
                                atol=0,
                            )
                            if cls is MiniMaxM31DSparkWeight:
                                self.assertTrue(
                                    component.weights[0].name.startswith(
                                        "language_model.model.dspark.layers.{i}.decoder_layer."
                                    )
                                )

    def test_hip_rejected_before_model_construction(self):
        with patch.object(torch.version, "hip", "test-hip"):
            with self.assertRaisesRegex(RuntimeError, "requires the CUDA backend"):
                MiniMaxM31DSpark._create_python_model(SimpleNamespace())


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

    def test_dspark_uses_five_swa_dense_layers(self):
        raw = _m31_config()
        raw["text_config"].update(
            num_hidden_layers=5,
            moe_layer_freq=[0] * 5,
            sparse_attention_config=None,
            dspark_noise_token_id=200058,
            dspark_markov_rank=256,
            dspark_block_size=7,
            sliding_window=4096,
            layer_types=["sliding_attention"] * 5,
            use_gemma_norm=True,
        )
        report = {
            "target_layer_ids": [3, 17, 31, 45, 59],
            "block_size": 7,
            "sliding_window": 4096,
            "layer_types": ["sliding_attention"] * 5,
            "use_gemma_norm": True,
        }
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(raw))
            with patch(
                "rtp_llm.models.minimax_m31_dspark.inspect_checkpoint",
                return_value=report,
            ):
                config = MiniMaxM31DSpark._create_config(tmpdir)

        self.assertEqual(config.model_type, "minimax_m31_dspark")
        self.assertEqual(config.num_layers, 5)
        self.assertEqual(config.physical_mtp_module_num, 1)
        self.assertEqual(config.moe_layer_index, [])
        self.assertIsNone(config.msa_sparse_config)
        self.assertEqual(config.dspark_target_layer_ids, [3, 17, 31, 45, 59])
        self.assertEqual(config.dspark_checkpoint_metadata["block_size"], 7)
        self.assertEqual(
            config.dspark_checkpoint_metadata["layer_types"],
            ["sliding_attention"] * 5,
        )
        self.assertTrue(config.dspark_checkpoint_metadata["use_gemma_norm"])
        self.assertEqual(config.dspark_checkpoint_metadata["sliding_window"], 4096)
        self.assertFalse(config.prepacked_nvfp4_moe)

    def test_dspark_execution_uses_validated_checkpoint_math(self):
        owner = SimpleNamespace(
            model_config=SimpleNamespace(
                gen_num_per_cycle=7,
                dspark_sample_from_anchor=True,
                dspark_target_layer_ids=[3, 17, 31, 45, 59],
                dspark_checkpoint_metadata={
                    "block_size": 7,
                    "sliding_window": 4096,
                    "layer_types": ["sliding_attention"] * 5,
                    "use_gemma_norm": True,
                },
            ),
            parallelism_config=Mock(),
            weight=Mock(),
            moe_config=Mock(),
            max_generate_batch_size=16,
            fmha_config=Mock(),
            hw_kernel_config=Mock(),
            device_resource_config=Mock(),
        )
        constructed = Mock()
        with patch.object(torch.version, "hip", None), patch(
            "rtp_llm.models_py.model_desc.minimax_m31_dspark.MiniMaxM31DSparkModel",
            return_value=constructed,
        ) as model:
            MiniMaxM31DSpark._create_python_model(owner)
        self.assertIs(owner.py_model, constructed)
        math = model.call_args.kwargs["math_contract"]
        self.assertTrue(math.hidden_norm_gemma)
        self.assertTrue(math.final_norm_gemma)
        self.assertFalse(math.causal_query)
        self.assertEqual(math.window_left, 4095)

    def test_dspark_shares_target_weights_before_dynamic_loading(self):
        from rtp_llm.models.minimax_m31_dspark import TargetSharedDSparkWeight

        for name, getter in (
            (W.embedding, "_get_target_embedding"),
            (W.lm_head, "_get_target_lm_head"),
        ):
            original = torch.ones(2, 3)
            with patch(
                "rtp_llm.models.minimax_m31_dspark." + getter, return_value=original
            ):
                weight = TargetSharedDSparkWeight(name)
                self.assertEqual(weight.weights, [])
                self.assertIs(weight.load(None, None, "cpu", None)[name], original)

    def test_dspark_fastsafetensors_loads_zero_dependency_shared_weights(self):
        from rtp_llm.model_loader.loader import ModelLoader
        from rtp_llm.model_loader.tensor_source import TensorCollector
        from rtp_llm.models.minimax_m31_dspark import TargetSharedDSparkWeight

        database = Mock()
        database.fastsafetensors_weights_iterator.return_value = iter(())
        items = [
            SimpleNamespace(
                weight=TargetSharedDSparkWeight(name),
                layer_id=None,
                collector=TensorCollector(set(), database),
            )
            for name in (W.embedding, W.lm_head)
        ]
        output = {}
        model_weights = SimpleNamespace(set_global_weight=output.__setitem__)
        loader = SimpleNamespace(
            _create_model_weights=lambda device: model_weights,
            _generate_weight_info=lambda: ({}, items),
            _build_stacked_key_config=lambda items: {},
            _load_config=SimpleNamespace(database=database),
        )
        embedding, head = torch.ones(2, 3), torch.zeros(2, 3)
        with patch(
            "rtp_llm.models.minimax_m31_dspark._get_target_embedding",
            return_value=embedding,
        ), patch(
            "rtp_llm.models.minimax_m31_dspark._get_target_lm_head", return_value=head
        ):
            ModelLoader._load_from_fastsafetensor(loader, "cpu")
        self.assertIs(output[W.embedding], embedding)
        self.assertIs(output[W.lm_head], head)
        database.load_tensor.assert_not_called()

    def test_dspark_missing_target_is_an_error(self):
        from rtp_llm.models.minimax_m31_dspark import TargetSharedDSparkWeight

        for name in (W.embedding, W.lm_head):
            with self.assertRaisesRegex(RuntimeError, "requires a live target"):
                TargetSharedDSparkWeight(name).load(
                    None, None, "missing-test-device", None
                )

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

    def test_real_shape_and_nvfp4_metadata(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(_m31_config()))
            config = MiniMaxM31._create_config(tmpdir)

        self.assertEqual(config.model_type, "minimax_m31")
        self.assertEqual(config.num_layers, 60)
        self.assertEqual(config.moe_layer_index, list(range(3, 60)))
        self.assertEqual(config.msa_sparse_config["sparse_layer_ids"], list(range(60)))
        self.assertTrue(config.prepacked_nvfp4_moe)

    def test_vl_registration_path_preserves_nvfp4_metadata(self):
        raw = _m31_config()
        raw["vision_config"] = {"hidden_size": 1152}
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(raw))
            config = MiniMaxM31_VL._create_config(tmpdir)

        self.assertEqual(config.model_type, "minimax_m31_vl")
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertTrue(config.prepacked_nvfp4_moe)
        self.assertEqual(config.msa_sparse_config["sparse_layer_ids"], list(range(60)))

    def test_non_nvfp4_draft_is_not_marked_prepacked(self):
        raw = _m31_config()
        raw["quantization_config"].pop("moe_quant_algo")
        raw["quantization_config"].pop("moe_quant_format")
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(raw))
            config = MiniMaxM31._create_config(tmpdir)

        self.assertFalse(config.prepacked_nvfp4_moe)

    def test_mock_nvfp4_moe_environment_variable_is_not_a_runtime_path(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(_m31_config()))
            with patch.dict(os.environ, {"M3_M31_MOCK_NVFP4_MOE": "1"}):
                for model_cls in (MiniMaxM31, MiniMaxM31_VL):
                    with self.subTest(model=model_cls.__name__):
                        config = model_cls._create_config(tmpdir)
                        self.assertTrue(config.prepacked_nvfp4_moe)
                        self.assertFalse(hasattr(config, "mock_nvfp4_moe"))
                        self.assertIs(model_cls.get_weight_cls(), MiniMaxM31Weight)

    def test_legacy_m3_does_not_parse_m31_nvfp4_state(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(_m31_config()))
            with patch.dict(os.environ, {"M3_M31_MOCK_NVFP4_MOE": "1"}):
                text_config = MiniMaxM3._create_config(tmpdir)
                vl_config = MiniMaxM3_VL._create_config(tmpdir)

        self.assertFalse(text_config.prepacked_nvfp4_moe)
        self.assertFalse(vl_config.prepacked_nvfp4_moe)


class MiniMaxM31WeightContractTest(unittest.TestCase):
    @staticmethod
    def _weight():
        weight = object.__new__(MiniMaxM31Weight)
        weight._raw_mxfp8_idx_layers = set()
        weight._native_mxfp4_routed = False
        weight._prepacked_nvfp4_routed = False
        weight.prefix = "language_model."
        weight._num_layers = 4
        weight._align_size = 0
        weight._is_gated_activation = True
        weight.moe_layer_index_ = [3]
        weight.expert_num_ = 128
        weight.has_e_score_correction_bias = True
        return weight

    @staticmethod
    def _keys():
        return {
            "language_model.model.layers.0.self_attn.index_q_proj.weight",
            "language_model.model.layers.3.self_attn.index_q_proj.weight",
            "language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_packed",
            "language_model.model.layers.3.block_sparse_moe.e_score_correction_bias",
        }

    @classmethod
    def _m31_keys(cls):
        keys = cls._keys()
        for layer_id in range(4):
            keys.add(
                f"language_model.model.layers.{layer_id}.self_attn.index_q_proj.weight"
            )
            keys.add(
                f"language_model.model.layers.{layer_id}.self_attn.index_k_proj.weight"
            )
        return keys

    def test_prepacked_nvfp4_is_detected_without_mock_gate(self):
        weight = self._weight()
        with patch.dict(os.environ, {}, clear=True):
            weight._process_meta([], self._m31_keys())

        self.assertTrue(weight._prepacked_nvfp4_routed)
        self.assertEqual(weight._sparse_layer_set, {0, 1, 2, 3})

    def test_m31_rejects_checkpoint_with_non_sparse_layers(self):
        weight = self._weight()
        with self.assertRaisesRegex(ValueError, "missing_q=\\[1, 2\\]"):
            weight._process_meta([], self._keys())

    def test_prepacked_weight_contract_loads_all_nvfp4_components(self):
        weight = self._weight()
        weight._prepacked_nvfp4_routed = True

        modules = weight._get_hf_ffn_layer_weight_info(3)
        checkpoint_names = {
            checkpoint_weight.name
            for module in modules
            for component in module.get_components()
            for checkpoint_weight in (getattr(component, "weights", None) or [])
        }

        self.assertEqual(len(modules), 9)
        self.assertIn(
            "language_model.model.layers.{i}.block_sparse_moe."
            "experts.{expert_id}.w1.weight_packed",
            checkpoint_names,
        )
        self.assertIn(
            "language_model.model.layers.{i}.block_sparse_moe."
            "experts.{expert_id}.w3.weight_scale",
            checkpoint_names,
        )
        self.assertIn(
            "language_model.model.layers.{i}.block_sparse_moe."
            "experts.{expert_id}.w2.weight_global_scale",
            checkpoint_names,
        )
        self.assertIn(
            "language_model.model.layers.{i}.block_sparse_moe."
            "shared_experts.gate_proj.weight",
            checkpoint_names,
        )
        self.assertIn(
            "language_model.model.layers.{i}.block_sparse_moe."
            "e_score_correction_bias",
            checkpoint_names,
        )
        components = {
            component.name: component
            for module in modules
            for component in module.get_components()
        }
        for name, dtype in (
            (W.moe_w1, torch.int8),
            (W.moe_w2, torch.int8),
            (W.moe_s1, torch.float8_e4m3fn),
            (W.moe_s2, torch.float8_e4m3fn),
            (W.moe_w1_s2, torch.float32),
            (W.moe_w2_s2, torch.float32),
        ):
            with self.subTest(weight=name):
                self.assertEqual(components[name].data_type, dtype)
                self.assertTrue(components[name].disable_quantization)

        # Distinct values expose accidental gate/up or expert reordering.
        for name in (W.moe_w1, W.moe_s1, W.moe_w1_s2):
            component = components[name]
            shape = () if name == W.moe_w1_s2 else (2, 4)
            inputs = [
                torch.full(shape, value, dtype=component.data_type)
                for value in (3, 5, 7, 11)
            ]
            actual = component.process_fun(inputs).float()
            expected = torch.tensor([[3, 7], [5, 11]], dtype=torch.float32)
            if shape:
                expected = expected.repeat_interleave(2, dim=1)
                expected = expected.unsqueeze(-1).expand(2, 4, 4)
            torch.testing.assert_close(actual, expected)

    def test_prepacked_checkpoint_keeps_dense_layers(self):
        weight = self._weight()
        weight._prepacked_nvfp4_routed = True
        modules = weight._get_hf_ffn_layer_weight_info(0)
        self.assertEqual(len(modules), 1)
        self.assertIsInstance(modules[0], FfnWeight)
        self.assertTrue(
            all(
                ".mlp." in ckpt.name
                for component in modules[0].get_components()
                for ckpt in component.weights
            )
        )

    def test_non_nvfp4_checkpoint_keeps_inherited_moe_loader(self):
        weight = self._weight()
        for native_mxfp4 in (False, True):
            with self.subTest(native_mxfp4=native_mxfp4):
                weight._native_mxfp4_routed = native_mxfp4
                modules = weight._get_hf_ffn_layer_weight_info(3)
                self.assertEqual(len(modules), 3)
                self.assertIsInstance(modules[0], FfnWeight)
                self.assertIsInstance(modules[1], MoeWeight)
                self.assertEqual(modules[2].name, W.e_score_correction_b)
                routed = modules[1].sub_weights[W.moe_w1]
                self.assertEqual(routed.stacked_ckpt_keys, native_mxfp4)
                self.assertEqual(routed.disable_quantization, native_mxfp4)

    def test_legacy_loader_does_not_detect_m31_packed_weights(self):
        weight = object.__new__(MiniMaxM3Weight)
        weight.prefix = "language_model."
        weight._process_meta([], self._keys())
        self.assertFalse(hasattr(weight, "_prepacked_nvfp4_routed"))
        self.assertEqual(weight._sparse_layer_set, {0, 3})


if __name__ == "__main__":
    unittest.main()
