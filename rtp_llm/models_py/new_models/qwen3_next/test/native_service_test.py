"""Real service entry, shared checkpoint, both direct backends. Requires main native ABI."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Dense, Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_mtp import (
    Qwen3NextMTP,
    Qwen35DenseMTP,
    Qwen35MoeMTP,
)
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.qwen3_next.language import (
    Qwen3NextForCausalLM,
    Qwen3NextMTPForCausalLM,
    Qwen35MoeMTPForCausalLM,
    Qwen35DenseMTPForCausalLM,
)
from rtp_llm.models_py.new_models.qwen3_next.test.main_checkpoint_test import (
    _dense_weights,
    _moe_config,
    _parallelism,
)


class CpuServer:
    """Use the public service implementation, with tokenizer and GPU allocation excluded."""

    def _get_device_str(self):
        return "cpu"

    def load_tokenizer(self):
        pass

    def _init_custom_module(self):
        return None

    def create_model_loader(self):
        raise AssertionError("must not construct OldLoader")

    def _create_python_model(self):
        raise AssertionError("must publish the parameter-owning model")


def shared_checkpoint(kind):
    routed = kind != "qwen35_dense"
    weights = _dense_weights(include_lm_head=True)
    if routed:
        for part in ("gate", "up", "down"):
            del weights[f"model.layers.0.mlp.{part}_proj.weight"]
        weights["model.layers.0.mlp.gate.weight"] = torch.ones(2, 4)
        for expert in range(2):
            for index, part in enumerate(("up", "gate", "down")):
                weights[f"model.layers.0.mlp.experts.{expert}.{part}_proj.weight"] = (
                    torch.full((4, 4), float(1 + 3 * expert + index))
                )
        if kind == "qwen35_moe":
            for part in ("gate", "up", "down"):
                weights[f"model.layers.0.mlp.shared_expert.{part}_proj.weight"] = (
                    torch.ones(4, 4)
                )
            weights["model.layers.0.mlp.shared_expert_gate.weight"] = torch.ones(1, 4)
    prefix = "model." if kind == "qwen3_next" else "model.language_model."
    result = {}
    for key, value in weights.items():
        if not key.startswith("model."):
            result[key] = value
            continue
        name = key[len("model.") :]
        result[prefix + name] = value
        if name != "embed_tokens.weight":
            result["mtp." + name] = value + 10
    result["mtp.pre_fc_norm_embedding.weight"] = torch.ones(4)
    result["mtp.pre_fc_norm_hidden.weight"] = torch.ones(4)
    result["mtp.fc.weight"] = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    return result


class NativeServiceTest(unittest.TestCase):
    def test_public_target_and_draft_scratch_and_fst(self):
        for kind, target, draft, draft_owner in (
            ("qwen3_next", Qwen3Next, Qwen3NextMTP, Qwen3NextMTPForCausalLM),
            ("qwen35_dense", Qwen35Dense, Qwen35DenseMTP, Qwen35DenseMTPForCausalLM),
            ("qwen35_moe", Qwen35Moe, Qwen35MoeMTP, Qwen35MoeMTPForCausalLM),
        ):
            with tempfile.TemporaryDirectory() as root:
                text = dict(
                    num_hidden_layers=1,
                    hidden_size=4,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=2,
                    vocab_size=8,
                    max_position_embeddings=64,
                    full_attention_interval=1,
                    rms_norm_eps=1e-6,
                    intermediate_size=4,
                    moe_intermediate_size=4,
                    num_experts=2,
                    num_experts_per_tok=1,
                    linear_conv_kernel_dim=2,
                    linear_num_key_heads=2,
                    linear_num_value_heads=2,
                    linear_key_head_dim=2,
                    linear_value_head_dim=2,
                    rope_theta=10000,
                    partial_rotary_factor=1.0,
                    rope_parameters=dict(
                        rope_theta=10000,
                        partial_rotary_factor=1.0,
                        mrope_section=[1, 0, 0],
                    ),
                )
                if kind == "qwen35_moe":
                    text["shared_expert_intermediate_size"] = 4
                hf = (
                    text
                    if kind == "qwen3_next"
                    else {"text_config": text, "language_model_only": True}
                )
                (Path(root) / "config.json").write_text(json.dumps(hf))
                tensors = shared_checkpoint(kind)
                save_file(tensors, str(Path(root) / "model.safetensors"))
                for method in ("scratch", "fastsafetensors"):
                    for model_type, cls, owner in (
                        (kind, target, Qwen3NextForCausalLM),
                        (kind + "_mtp", draft, draft_owner),
                    ):
                        with self.subTest(model_type=model_type, method=method):
                            config = cls._create_config(root)
                            config.model_type = model_type
                            config.data_type = "fp32"
                            parallel = _parallelism()
                            parallel.local_world_size = 1
                            server_type = type("NativeQwenServer", (CpuServer, cls), {})
                            model = server_type.from_config(
                                model_config=config,
                                parallelism_config=parallel,
                                hw_kernel_config=SimpleNamespace(
                                    enable_cuda_graph=False, use_swizzleA=False
                                ),
                                kv_cache_config=SimpleNamespace(
                                    multi_task_prompt="", multi_task_prompt_str=""
                                ),
                                max_generate_batch_size=7,
                                fmha_config=None,
                                moe_config=_moe_config(),
                                load_method=method,
                                merge_lora=False,
                                vit_config=None,
                            )
                            self.assertIs(type(model.py_model), owner)
                            self.assertEqual(
                                model.model_weights_loader.loaded_method,
                                NewLoaderLoadMethod(method),
                            )
                            self.assertIs(
                                model.py_model.lm_head.weight,
                                model.weight.get_global_weight("lm_head"),
                            )
                            torch.testing.assert_close(
                                model.py_model.norm.weight,
                                torch.full((4,), 12.0 if cls is draft else 2.0),
                            )
                            if cls is draft:
                                torch.testing.assert_close(
                                    model.py_model.fc.weight, tensors["mtp.fc.weight"]
                                )
                            if kind != "qwen35_dense":
                                block = model.py_model.layers[0].mlp
                                self.assertIsNone(block.select_topk)
                                # Backend failures still surface in accelerator
                                # preflight, before serving the first request.
                                with patch.object(block, "_bind_router", side_effect=ImportError("router unavailable")):
                                    block.validate_runtime_device(torch.device("cpu"))
                                    with self.assertRaisesRegex(ImportError, "router unavailable"):
                                        block.validate_runtime_device(torch.device("cuda"))
                                self.assertEqual(
                                    block.shared_expert is not None,
                                    kind == "qwen35_moe",
                                )
                                self.assertEqual(
                                    block.shared_expert_gate is not None,
                                    kind == "qwen35_moe",
                                )


if __name__ == "__main__":
    unittest.main()
