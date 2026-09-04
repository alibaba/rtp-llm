import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import (
    CompressedW8A8Int8PerChannelQuantConfig,
    Fp8PerTensorCompressedQuantConfig,
    GPTQConfig,
    ModelOptFp4Config,
)
from rtp_llm.config.quant_config import QuantizationConfig as SourceQuantizationConfig
from rtp_llm.config.quant_config import WeightOnlyInt8PerChannelQuantConfig
from rtp_llm.metrics import GaugeMetrics
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod
from rtp_llm.ops import TaskType
from rtp_llm.utils.model_weight import CkptWeightInfo


def _model_config():
    return types.SimpleNamespace(
        model_type="qwen_3",
        num_layers=1,
        vocab_size=8,
        hidden_size=4,
        inter_size=4,
        expert_num=0,
        attn_config=types.SimpleNamespace(
            head_num=2,
            kv_head_num=1,
            size_per_head=2,
        ),
        layernorm_eps=1e-6,
        enable_fp32_lm_head=False,
        enable_output_vocab_pruning=False,
        tie_word_embeddings=True,
        compute_dtype=torch.float32,
        generate_env_config=None,
        ptuning_path="",
        lora_infos={},
        eplb_config=types.SimpleNamespace(enable_eplb=lambda: False),
        quant_config=types.SimpleNamespace(get_runtime_method_key=lambda: "none"),
        use_new_loader=True,
        require_weight_update=False,
        task_type=TaskType.LANGUAGE_MODEL,
    )


def _base_model(config, hw_kernel_config=None):
    if hw_kernel_config is None:
        hw_kernel_config = types.SimpleNamespace(enable_cuda_graph=False)
    kv_cache_config = types.SimpleNamespace(
        multi_task_prompt=False,
        multi_task_prompt_str="",
    )
    with patch.object(BaseModel, "load_tokenizer", return_value=None):
        model = BaseModel(
            model_config=config,
            parallelism_config=_parallelism_config(),
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            fmha_config=None,
            moe_config=None,
            max_generate_batch_size=0,
            load_method=LoadMethod.SCRATCH,
            vit_config=None,
            merge_lora=False,
            device_resource_config=None,
        )
    model.tokenizer = None
    return model


def _parallelism_config():
    return types.SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        local_rank=0,
        world_rank=0,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False,
            is_prefill_enabled=lambda: False,
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        get_ffn_tp_rank=lambda: 0,
    )


def _weights():
    return {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
            8, 4
        ),
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(2),
        "model.layers.0.self_attn.k_norm.weight": torch.ones(2),
        "model.layers.0.self_attn.o_proj.weight": torch.eye(4),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
        "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 4),
        "model.layers.0.mlp.up_proj.weight": torch.ones(4, 4),
        "model.layers.0.mlp.down_proj.weight": torch.ones(4, 4),
        "model.norm.weight": torch.ones(4),
    }


class Qwen3BaseModelIntegrationTest(unittest.TestCase):
    def test_base_model_loader_route_uses_registry_default_and_explicit_override(self):
        model = object.__new__(BaseModel)
        model.model_config = types.SimpleNamespace(
            model_type="qwen_3",
            use_new_loader=None,
            require_weight_update=False,
        )
        model._new_loader_unsupported_reason = lambda **kwargs: None

        self.assertTrue(model._use_new_loader())
        model.model_config.model_type = "legacy_only_test_model"
        self.assertFalse(model._use_new_loader())

        model.model_config.use_new_loader = True
        self.assertTrue(model._use_new_loader())
        model.model_config.model_type = "qwen_3"
        model.model_config.use_new_loader = False
        self.assertFalse(model._use_new_loader())

    def test_registry_default_falls_back_but_explicit_newloader_stays_strict(self):
        model = object.__new__(BaseModel)
        model.model_config = types.SimpleNamespace(
            model_type="qwen_3", use_new_loader=None
        )
        model._new_loader_unsupported_reason = (
            lambda **kwargs: "unsupported test configuration"
        )

        self.assertFalse(model._use_new_loader())
        model.model_config.use_new_loader = True
        self.assertTrue(model._use_new_loader())

    def test_load_initializes_custom_module_before_loader_routing(self):
        model = _base_model(_model_config())
        custom_module = object()

        def verify_route(**_kwargs):
            self.assertIs(model.custom_module, custom_module)
            return True

        with patch.object(
            model, "_init_custom_module", return_value=custom_module
        ), patch.object(
            model, "_use_new_loader", side_effect=verify_route
        ), patch.object(
            model, "_load_with_new_loader"
        ) as load_with_newloader:
            model.load()

        load_with_newloader.assert_called_once_with()

    def test_automatic_route_falls_back_for_fastsafetensors(self):
        config = _model_config()
        config.use_new_loader = None
        model = _base_model(config)
        model.load_method = LoadMethod.FASTSAFETENSORS

        self.assertIn(
            "fastsafetensors is not supported",
            model._new_loader_unsupported_reason(),
        )
        self.assertFalse(model._use_new_loader())

        config.use_new_loader = True
        self.assertTrue(model._use_new_loader())
        with self.assertRaisesRegex(ValueError, "fastsafetensors is not supported"):
            model._load_with_new_loader()

    def test_automatic_route_resolves_load_method_environment_override(self):
        config = _model_config()
        config.use_new_loader = None
        model = _base_model(config)
        model.load_method = LoadMethod.AUTO

        with patch.dict(os.environ, {"LOAD_METHOD": "fastsafetensors"}, clear=False):
            self.assertFalse(model._use_new_loader())

        with patch.dict(os.environ, {"LOAD_METHOD": "scratch"}, clear=False):
            self.assertTrue(model._use_new_loader())

    def test_automatic_route_falls_back_for_unsupported_custom_weights(self):
        config = _model_config()
        config.use_new_loader = None
        model = _base_model(config)
        custom_weight = CustomAtomicWeight(
            "__custom__.lm_head.weight",
            [CkptWeightInfo("lm_head.weight")],
        )
        model.custom_module = types.SimpleNamespace(
            get_custom_weight_info=lambda: [custom_weight]
        )

        self.assertIn(
            "does not support downstream custom weights",
            model._new_loader_unsupported_reason(),
        )
        self.assertFalse(model._use_new_loader())

        config.use_new_loader = True
        self.assertTrue(model._use_new_loader())
        with self.assertRaisesRegex(
            ValueError, "does not support downstream custom weights"
        ):
            model._load_with_new_loader()

        config.model_type = "bert"
        config.use_new_loader = None
        self.assertIsNone(model._new_loader_unsupported_reason())
        self.assertTrue(model._use_new_loader())

    def test_compressed_w8a8_preserves_all_exclusion_aliases(self):
        ignored_by_name = "model.layers.0.self_attn.o_proj"
        ignored_by_compat_name = ["model.layers.0.mlp.down_proj"]
        excluded = "lm_head"

        with tempfile.TemporaryDirectory() as path:
            with open(f"{path}/config.json", "w") as output:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "compressed-tensors",
                            "config_groups": {
                                "group_0": {
                                    "weights": {
                                        "type": "int",
                                        "num_bits": 8,
                                        "strategy": "channel",
                                        "dynamic": False,
                                        "symmetric": True,
                                    },
                                    "input_activations": {
                                        "type": "int",
                                        "num_bits": 8,
                                        "strategy": "token",
                                        "dynamic": True,
                                        "symmetric": True,
                                    },
                                }
                            },
                            "ignore": ignored_by_name,
                            "modules_to_not_convert": ignored_by_compat_name,
                            "exclude": excluded,
                        }
                    },
                    output,
                )

            source_config = SourceQuantizationConfig.load_from_ckpt(path)

        self.assertIsInstance(source_config, CompressedW8A8Int8PerChannelQuantConfig)
        self.assertEqual(
            source_config.ignored_layers,
            [ignored_by_name, *ignored_by_compat_name],
        )
        self.assertEqual(
            source_config.exclude_modules,
            {ignored_by_name, *ignored_by_compat_name, excluded},
        )

    def test_explicit_legacy_route_still_checks_checkpoint_compatibility(self):
        config = _model_config()
        config.use_new_loader = False
        model = _base_model(config)

        with patch.object(
            model,
            "_legacy_loader_unsupported_reason",
            return_value="test checkpoint has no legacy layout",
        ), self.assertRaisesRegex(ValueError, "no legacy layout"):
            model._use_new_loader()

    def test_registry_default_falls_back_for_unsupported_quantization(self):
        config = _model_config()
        config.use_new_loader = None
        model = _base_model(config)
        cases = {
            "W8A8_INT8": CompressedW8A8Int8PerChannelQuantConfig(),
            "INT8": WeightOnlyInt8PerChannelQuantConfig(),
            "GPTQ": GPTQConfig(bits=4, group_size=128, is_quanted=True),
            "MODELOPT_FP4": ModelOptFp4Config(bits=4, group_size=16, is_quanted=True),
        }

        for name, quant_config in cases.items():
            with self.subTest(quantization=name):
                config.quant_config = quant_config
                self.assertIn(
                    "does not provide a supported NewLoader runtime method",
                    model._new_loader_unsupported_reason(),
                )
                self.assertFalse(model._use_new_loader())

                config.use_new_loader = True
                self.assertTrue(model._use_new_loader())
                with self.assertRaisesRegex(ValueError, "quantization config"):
                    model._load_with_new_loader()
                config.use_new_loader = None

    def test_static_activation_fp8_moe_routes_to_legacy_or_fails_early(self):
        config = _model_config()
        config.model_type = "qwen_3_moe"
        config.expert_num = 8
        config.use_new_loader = None
        config.quant_config = Fp8PerTensorCompressedQuantConfig(
            is_quanted=True,
            dynamic=False,
        )
        model = _base_model(config)

        self.assertIn(
            "static-activation per-tensor FP8 MoE",
            model._new_loader_unsupported_reason(),
        )
        self.assertFalse(model._use_new_loader())

        config.use_new_loader = True
        self.assertTrue(model._use_new_loader())
        with self.assertRaisesRegex(ValueError, "static-activation per-tensor FP8 MoE"):
            model._load_with_new_loader()

        config.use_new_loader = None
        config.quant_config = Fp8PerTensorCompressedQuantConfig(
            is_quanted=True,
            dynamic=True,
        )
        self.assertIsNone(model._new_loader_unsupported_reason())
        self.assertTrue(model._use_new_loader())

    def test_automatic_newloader_preserves_legacy_until_policy_is_declared(self):
        config = _model_config()
        config.use_new_loader = None
        config.require_weight_update = None
        model = _base_model(config)

        self.assertFalse(model._use_new_loader())

        config.require_weight_update = False
        self.assertTrue(model._use_new_loader())

        config.require_weight_update = True
        with patch.object(
            model,
            "_legacy_loader_unsupported_reason",
            return_value="this checkpoint has no legacy layout",
        ), self.assertRaisesRegex(ValueError, "No compatible loader route"):
            model._use_new_loader()

        self.assertFalse(model._use_new_loader())
        config.use_new_loader = True
        self.assertTrue(model._use_new_loader())
        with self.assertRaisesRegex(ValueError, "online UpdateWeights is required"):
            model._load_with_new_loader()

    def test_source_configs_preserve_ignore_and_exclude(self):
        ignored = ["model.layers.0.self_attn.o_proj"]
        excluded = "model.layers.0.mlp.down_proj"
        cases = (
            (
                "fp8",
                {
                    "quant_method": "fp8",
                    "ignore": ignored,
                    "exclude": excluded,
                },
            ),
            (
                "fp8_block",
                {
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                    "ignore": ignored,
                    "exclude": excluded,
                },
            ),
            (
                "fp8_per_channel",
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "type": "float",
                                "num_bits": 8,
                                "strategy": "channel",
                            },
                            "input_activations": {"dynamic": True},
                        }
                    },
                    "ignore": ignored,
                    "exclude": excluded,
                },
            ),
            (
                "",
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "type": "int",
                                "num_bits": 8,
                                "strategy": "channel",
                                "dynamic": False,
                                "symmetric": True,
                            },
                            "input_activations": {
                                "type": "int",
                                "num_bits": 8,
                                "strategy": "token",
                                "dynamic": True,
                                "symmetric": True,
                            },
                        }
                    },
                    "modules_to_not_convert": ignored,
                    "exclude": excluded,
                },
            ),
            (
                "fp8_per_channel",
                {
                    "quant_method": "quark",
                    "global_quant_config": {
                        "weight": {
                            "dtype": "fp8_e4m3",
                            "qscheme": "per_channel",
                        }
                    },
                    "ignore": ignored,
                    "exclude": excluded,
                },
            ),
            (
                "",
                {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "ignore": ignored,
                    "exclude": excluded,
                },
            ),
        )

        for expected_method, quantization_config in cases:
            with self.subTest(
                method=expected_method
            ), tempfile.TemporaryDirectory() as path:
                with open(f"{path}/config.json", "w") as output:
                    json.dump({"quantization_config": quantization_config}, output)
                source_config = SourceQuantizationConfig.load_from_ckpt(path)

                self.assertEqual(
                    source_config.get_runtime_method_key(), expected_method
                )
                self.assertEqual(source_config.ignored_layers, ignored)
                expected_excluded = {excluded}
                if isinstance(source_config, CompressedW8A8Int8PerChannelQuantConfig):
                    expected_excluded.update(ignored)
                self.assertEqual(source_config.exclude_modules, expected_excluded)

    def test_checkpoint_ignore_reaches_real_newloader_projection(self):
        ignored = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "lm_head",
        ]
        excluded = "model.layers.0.mlp.down_proj"
        config = _model_config()
        base_model = _base_model(config)
        self.assertFalse(base_model.keep_mla_checkpoint_weights)

        with tempfile.TemporaryDirectory() as model_path:
            with open(f"{model_path}/config.json", "w") as output:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "compressed-tensors",
                            "config_groups": {
                                "group_0": {
                                    "weights": {
                                        "type": "float",
                                        "num_bits": 8,
                                        "strategy": "tensor",
                                    },
                                    "input_activations": {"dynamic": True},
                                }
                            },
                            "ignore": ignored,
                            "exclude": excluded,
                        }
                    },
                    output,
                )
            source_config = SourceQuantizationConfig.load_from_ckpt(model_path)
            self.assertEqual(source_config.ignored_layers, ignored)
            self.assertEqual(source_config.exclude_modules, {excluded})
            config.quant_config = source_config
            config.ckpt_path = model_path
            save_file(_weights(), f"{model_path}/model.safetensors")

            with patch.object(
                BaseModel, "_get_device_str", return_value="cpu"
            ), patch.object(
                BaseModel, "_init_custom_module", return_value=None
            ), patch.dict(
                os.environ, {"USE_NEW_LOADER": "1"}, clear=False
            ):
                base_model.load()

        layer = base_model.py_model.layers[0]
        self.assertIsInstance(
            layer.self_attn.o_proj.quant_method, UnquantizedLinearMethod
        )
        self.assertIsInstance(layer.mlp.down_proj.quant_method, UnquantizedLinearMethod)

    def test_static_and_dynamic_activation_scale_reach_real_newloader(self):
        ignored = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
            "lm_head",
        ]

        for activation_dynamic in (True, False):
            with self.subTest(
                activation_dynamic=activation_dynamic
            ), tempfile.TemporaryDirectory() as model_path:
                quantization_config = {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "type": "float",
                                "num_bits": 8,
                                "strategy": "tensor",
                            },
                            "input_activations": {
                                "dynamic": activation_dynamic,
                            },
                        }
                    },
                    "ignore": ignored,
                }
                with open(f"{model_path}/config.json", "w") as output:
                    json.dump({"quantization_config": quantization_config}, output)

                source_config = SourceQuantizationConfig.load_from_ckpt(model_path)
                self.assertEqual(source_config.is_dynamic(), activation_dynamic)

                checkpoint = _weights()
                weight_name = "model.layers.0.self_attn.o_proj.weight"
                source_weight = checkpoint[weight_name]
                weight_scale = (source_weight.abs().max() / 448.0).float().reshape(1)
                checkpoint[weight_name] = (source_weight / weight_scale).to(
                    torch.float8_e4m3fn
                )
                checkpoint[f"{weight_name}_scale"] = weight_scale
                if not activation_dynamic:
                    checkpoint["model.layers.0.self_attn.o_proj.input_scale"] = (
                        torch.tensor([0.125], dtype=torch.float32)
                    )
                save_file(checkpoint, f"{model_path}/model.safetensors")

                config = _model_config()
                config.quant_config = source_config
                config.ckpt_path = model_path
                hw_kernel_config = types.SimpleNamespace(
                    enable_cuda_graph=False,
                    use_swizzleA=True,
                )
                base_model = _base_model(config, hw_kernel_config)

                with patch.object(
                    BaseModel, "_get_device_str", return_value="cpu"
                ), patch.object(
                    BaseModel, "_init_custom_module", return_value=None
                ), patch(
                    "rtp_llm.models_py.quant_methods.fp8._select_fp8_runtime_backend",
                    return_value="cuda_scaled_mm",
                ), patch(
                    "rtp_llm.models_py.quant_methods.fp8._is_hip_runtime",
                    return_value=False,
                ), patch(
                    "rtp_llm.models_py.quant_methods.fp8._runtime_fp8_dtype",
                    return_value=torch.float8_e4m3fn,
                ), patch.dict(
                    os.environ, {"USE_NEW_LOADER": "1"}, clear=False
                ):
                    base_model.load()

                o_proj = base_model.py_model.layers[0].self_attn.o_proj
                self.assertEqual(
                    o_proj.quant_config.activation_dynamic,
                    activation_dynamic,
                )
                self.assertIs(
                    o_proj.quant_config.hw_kernel_config,
                    hw_kernel_config,
                )
                self.assertEqual(
                    "input_scale" in o_proj._loaded_parameter_names,
                    not activation_dynamic,
                )

    def test_quant_config_without_runtime_method_is_rejected(self):
        base_model = object.__new__(BaseModel)
        base_model.model_config = _model_config()
        base_model.model_config.quant_config = types.SimpleNamespace(
            get_runtime_method_key=lambda: ""
        )

        with self.assertRaisesRegex(ValueError, "is not supported"):
            base_model._new_loader_quant_type()

    def test_base_model_entry_loads_registered_qwen_runtime(self):
        config = _model_config()
        base_model = _base_model(config)

        with self.assertLogs(level="WARNING") as captured_logs, patch(
            "rtp_llm.models.base_model.kmonitor.report"
        ) as report_metric:
            with tempfile.TemporaryDirectory() as model_path:
                config.ckpt_path = model_path
                save_file(_weights(), f"{model_path}/model.safetensors")
                with patch.object(
                    BaseModel, "_get_device_str", return_value="cpu"
                ), patch.object(
                    BaseModel, "_init_custom_module", return_value=None
                ), patch.dict(
                    os.environ, {"USE_NEW_LOADER": "1"}, clear=False
                ):
                    base_model.load()

        self.assertIsInstance(base_model.py_model, Qwen3ForCausalLM)
        self.assertIsInstance(base_model.py_model, GptModelBase)
        self.assertIsInstance(base_model.py_model, RtpModule)
        self.assertFalse(base_model.py_model.training)
        self.assertIs(base_model.py_model.weight, base_model.weight)
        self.assertIs(base_model.weight_manager, None)
        self.assertIn("CAPABILITY_DISABLED", "\n".join(captured_logs.output))
        report_metric.assert_called_once_with(
            GaugeMetrics.UPDATE_WEIGHTS_AVAILABLE_METRIC,
            0,
            {"loader": "newloader", "model_type": "qwen_3"},
        )
        self.assertTrue(base_model.uses_new_loader)
        self.assertEqual(
            set(base_model.py_model.runtime_weight_view()),
            {"embedding", "final_layernorm.gamma", "lm_head"},
        )

    def test_ptuning_configuration_is_rejected(self):
        config = _model_config()
        config.ptuning_path = "/tmp/unsupported-ptuning"
        base_model = object.__new__(BaseModel)
        base_model.model_config = config
        base_model.parallelism_config = _parallelism_config()
        base_model.force_cpu_load_weights = False
        base_model.device_resource_config = None

        with self.assertRaisesRegex(ValueError, "p-tuning is not supported"):
            base_model._load_with_new_loader()

    def test_eplb_configuration_is_rejected_before_model_loading(self):
        config = _model_config()
        config.eplb_config = types.SimpleNamespace(enable_eplb=lambda: True)
        base_model = object.__new__(BaseModel)
        base_model.model_config = config
        base_model.parallelism_config = _parallelism_config()
        base_model.force_cpu_load_weights = False
        base_model.device_resource_config = None

        with self.assertRaisesRegex(ValueError, "EPLB is not supported"):
            base_model._load_with_new_loader()

    def test_output_vocab_pruning_is_rejected_before_model_loading(self):
        config = _model_config()
        config.enable_output_vocab_pruning = True
        base_model = object.__new__(BaseModel)
        base_model.model_config = config
        base_model.parallelism_config = _parallelism_config()
        base_model.force_cpu_load_weights = False
        base_model.device_resource_config = None

        with self.assertRaisesRegex(
            ValueError, "output vocabulary pruning is not supported"
        ):
            base_model._load_with_new_loader()

    def test_layer_micro_batch_is_rejected_by_public_load(self):
        config = _model_config()
        base_model = object.__new__(BaseModel)
        base_model.model_config = config
        base_model.parallelism_config = _parallelism_config()
        base_model.force_cpu_load_weights = False
        base_model.load_method = LoadMethod.SCRATCH
        base_model.fmha_config = None
        base_model.tokenizer = None
        base_model.hw_kernel_config = types.SimpleNamespace(enable_cuda_graph=False)
        base_model.device_resource_config = types.SimpleNamespace(
            enable_layer_micro_batch=1
        )

        with patch.dict(os.environ, {"USE_NEW_LOADER": "1"}, clear=False):
            with self.assertRaisesRegex(
                ValueError, "layer micro-batch is not supported"
            ):
                base_model.load()


if __name__ == "__main__":
    unittest.main()
