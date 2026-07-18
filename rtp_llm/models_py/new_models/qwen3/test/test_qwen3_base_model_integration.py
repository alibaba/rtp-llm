import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import QuantizationConfig as SourceQuantizationConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod


def _model_config():
    return types.SimpleNamespace(
        model_type="qwen_3",
        num_layers=1,
        vocab_size=8,
        hidden_size=4,
        inter_size=4,
        attn_config=types.SimpleNamespace(
            head_num=2,
            kv_head_num=1,
            size_per_head=2,
        ),
        layernorm_eps=1e-6,
        enable_fp32_lm_head=False,
        tie_word_embeddings=True,
        compute_dtype=torch.float32,
        lora_infos={},
        quant_config=types.SimpleNamespace(get_runtime_method_key=lambda: "none"),
    )


def _parallelism_config():
    return types.SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        local_rank=0,
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
                self.assertEqual(source_config.exclude_modules, {excluded})

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
        base_model = object.__new__(BaseModel)
        base_model.model_config = config
        base_model.parallelism_config = _parallelism_config()
        base_model.force_cpu_load_weights = False
        base_model.load_method = LoadMethod.SCRATCH
        base_model.fmha_config = None
        base_model.device_resource_config = None
        base_model.tokenizer = None
        base_model.hw_kernel_config = types.SimpleNamespace(enable_cuda_graph=False)

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

    def test_quant_config_without_runtime_method_is_rejected(self):
        base_model = object.__new__(BaseModel)
        base_model.model_config = _model_config()
        base_model.model_config.quant_config = types.SimpleNamespace(
            get_runtime_method_key=lambda: ""
        )

        with self.assertRaisesRegex(ValueError, "is not supported"):
            base_model._new_loader_quant_type()

    def test_gpt_runtime_base_rejects_invalid_graph_capture_state(self):
        runtime = GptModelBase(
            config=_model_config(),
            parallelism_config=_parallelism_config(),
            weight=None,
            max_generate_batch_size=0,
        )
        empty = torch.empty(0, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "No captured FMHA params"):
            runtime.fill_params(empty, empty, empty, 1, 4, 16)

        runtime.params_dict[4] = None
        with self.assertRaisesRegex(RuntimeError, "cannot be None"):
            runtime.fill_params(empty, empty, empty, 1, 4, 16)

    def test_base_model_entry_loads_registered_qwen_runtime(self):
        config = _model_config()
        base_model = object.__new__(BaseModel)
        base_model.model_config = config
        base_model.parallelism_config = _parallelism_config()
        base_model.force_cpu_load_weights = False
        base_model.load_method = LoadMethod.SCRATCH
        base_model.fmha_config = None
        base_model.device_resource_config = None
        base_model.tokenizer = None
        base_model.hw_kernel_config = types.SimpleNamespace(enable_cuda_graph=False)

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

        with self.assertRaisesRegex(ValueError, "p-tuning is not supported"):
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
