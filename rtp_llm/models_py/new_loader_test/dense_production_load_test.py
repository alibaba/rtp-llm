"""Production route -> real checkpoint reader -> registered runtime weights."""

import json
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.module_base import collect_loaded_tensor_ids
from rtp_llm.ops import TaskType


def service(model_type="qwen_2", *, use_new_loader=True, require_weight_update=None):
    model = BaseModel.__new__(BaseModel)
    model.model_config = NS(
        model_type=model_type,
        use_new_loader=use_new_loader,
        require_weight_update=require_weight_update,
        quant_config=None,
        expert_num=0,
        enable_output_vocab_pruning=False,
        eplb_config=NS(enable_eplb=lambda: False),
        mm_model_config=NS(is_multimodal=False),
        ptuning_path="",
        lora_infos={},
        task_type=TaskType.LANGUAGE_MODEL,
        ckpt_path="",
        num_layers=1,
        vocab_size=8,
        hidden_size=4,
        inter_size=4,
        attn_config=NS(head_num=2, kv_head_num=1, size_per_head=2),
        layernorm_eps=1e-6,
        enable_fp32_lm_head=False,
        tie_word_embeddings=False,
        compute_dtype=torch.float32,
        is_mtp=False,
    )
    model.parallelism_config = NS(
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        ep_rank=0,
        local_rank=0,
        prefill_cp_config=NS(
            is_enabled=lambda: False, is_prefill_enabled=lambda: False
        ),
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        get_ffn_tp_rank=lambda: 0,
    )
    model.hw_kernel_config = NS(enable_cuda_graph=False)
    model.fmha_config = None
    model.max_generate_batch_size = 4
    model.force_cpu_load_weights = False
    model.load_method = "scratch"
    model._weight_alias_names = ()
    model._uses_new_loader = False
    model.custom_module = None
    return model


def checkpoint(model_type):
    generator = torch.Generator().manual_seed(17)
    weights = {
        "model.embed_tokens.weight": torch.randn(8, 4, generator=generator),
        "lm_head.weight": torch.randn(8, 4, generator=generator),
        "model.norm.weight": torch.ones(4),
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
    }
    for projection, shape in (
        ("self_attn.q_proj", (4, 4)),
        ("self_attn.k_proj", (2, 4)),
        ("self_attn.v_proj", (2, 4)),
        ("self_attn.o_proj", (4, 4)),
        ("mlp.gate_proj", (4, 4)),
        ("mlp.up_proj", (4, 4)),
        ("mlp.down_proj", (4, 4)),
    ):
        weights[f"model.layers.0.{projection}.weight"] = torch.randn(
            *shape, generator=generator
        )
    if model_type == "qwen_2":
        for name, size in (("q", 4), ("k", 2), ("v", 2)):
            weights[f"model.layers.0.self_attn.{name}_proj.bias"] = torch.randn(
                size, generator=generator
            )
    if model_type == "qwen_3":
        for name in ("q", "k"):
            weights[f"model.layers.0.self_attn.{name}_norm.weight"] = torch.ones(2)
    return weights


class DenseProductionLoadTest(unittest.TestCase):
    def test_real_production_load_for_each_registered_dense_model(self):
        for model_type in ("qwen_2", "qwen_3", "llama"):
            with self.subTest(
                model_type=model_type
            ), tempfile.TemporaryDirectory() as directory:
                model = service(model_type)
                model.model_config.ckpt_path = directory
                weights = checkpoint(model_type)
                save_file(weights, directory + "/model.safetensors")
                with open(directory + "/config.json", "w") as handle:
                    json.dump(
                        {
                            "model_type": (
                                "llama" if model_type == "llama" else model_type
                            )
                        },
                        handle,
                    )
                # Only tokenizer/custom-head setup and device selection are
                # isolated. BaseModel.load and NewModelLoader.load are real.
                with patch.object(
                    model, "_init_custom_module", return_value=None
                ), patch.object(model, "_get_device_str", return_value="cpu"):
                    model.load()
                self.assertTrue(model.uses_new_loader)
                self.assertIsInstance(model.model_weights_loader, NewModelLoader)
                self.assertIsNone(model.weight_manager)
                self.assertFalse(model.py_model.training)
                self.assertEqual(model.weight.weights, [{}])
                self.assertIs(model.py_model.weight, model.weight)
                self.assertIs(
                    model.py_model.py_hw_kernel_config, model.hw_kernel_config
                )
                loaded = collect_loaded_tensor_ids(model.py_model)
                for parameter in model.py_model.parameters():
                    self.assertIn(id(parameter), loaded)
                torch.testing.assert_close(
                    model.py_model.embed_tokens.weight,
                    weights["model.embed_tokens.weight"],
                )
                torch.testing.assert_close(
                    model.py_model.lm_head.weight, weights["lm_head.weight"]
                )
                torch.testing.assert_close(
                    model.py_model.layers[0].self_attn.qkv_proj.weight,
                    torch.cat(
                        [
                            weights[f"model.layers.0.self_attn.{part}_proj.weight"]
                            for part in ("q", "k", "v")
                        ]
                    ),
                )

    def test_default_opt_in_and_explicit_routes(self):
        for explicit, hot_update, expected in (
            (None, None, False),
            (None, False, True),
            (None, True, False),
            (False, False, False),
            (True, None, True),
            (True, False, True),
        ):
            with self.subTest(explicit=explicit, hot_update=hot_update):
                model = service(
                    use_new_loader=explicit, require_weight_update=hot_update
                )
                self.assertIs(model._use_new_loader(), expected)
        with self.assertRaisesRegex(ValueError, "online UpdateWeights"):
            service(require_weight_update=True)._use_new_loader()

    def test_unsupported_modes_fallback_or_fail_before_loading(self):
        modifications = (
            ("model_config", "quant_config", object()),
            ("model_config", "ptuning_path", "prompt"),
            ("model_config", "lora_infos", {"adapter": "path"}),
            ("model_config", "expert_num", 8),
            ("model_config", "enable_output_vocab_pruning", True),
            ("model_config", "mm_model_config", NS(is_multimodal=True)),
            ("parallelism_config", "pp_size", 2),
            ("parallelism_config", "dp_size", 2),
            ("parallelism_config", "ep_size", 2),
        )
        for owner, field, value in modifications:
            for explicit in (None, True):
                with self.subTest(field=field, explicit=explicit):
                    model = service(
                        use_new_loader=explicit, require_weight_update=False
                    )
                    setattr(getattr(model, owner), field, value)
                    if explicit:
                        with self.assertRaisesRegex(ValueError, "unsupported"):
                            model._use_new_loader()
                    else:
                        self.assertFalse(model._use_new_loader())

    def test_fst_cpu_loading_and_unregistered_model_are_not_auto_enabled(self):
        for override in (
            {"load_method": "fastsafetensors"},
            {"force_cpu_load_weights": True},
        ):
            model = service(use_new_loader=None, require_weight_update=False)
            for name, value in override.items():
                setattr(model, name, value)
            self.assertFalse(model._use_new_loader())
            model.model_config.use_new_loader = True
            with self.assertRaisesRegex(ValueError, "unsupported"):
                model._use_new_loader()
        self.assertFalse(
            service(
                "not_registered", use_new_loader=None, require_weight_update=False
            )._use_new_loader()
        )
        with self.assertRaisesRegex(ValueError, "not registered"):
            service("not_registered")._use_new_loader()


if __name__ == "__main__":
    unittest.main()
