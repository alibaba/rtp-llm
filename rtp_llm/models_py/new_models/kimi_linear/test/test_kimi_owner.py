"""Whole two-layer Kimi KDA/MLA + MoE checkpoint against main's typed config."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewLoaderLoadMethod, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v3.moe import DeepSeekV32MoeGate
from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearModel
from rtp_llm.models_py.new_models.kimi_linear.language import KimiLinearForCausalLM
from rtp_llm.ops import DataType, ParallelismConfig


def _raw_config():
    return {
        "model_type": "kimi_linear",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 16,
        "intermediate_size": 12,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": None,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "mla_use_nope": True,
        "rope_scaling": None,
        "linear_attn_config": {
            "kda_layers": [1],
            "full_attn_layers": [2],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 4,
        },
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "moe_intermediate_size": 4,
        "num_shared_experts": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 2.446,
        "num_expert_group": 1,
        "topk_group": 1,
    }


def _tensor(shape, value):
    return torch.full(shape, float(value), dtype=torch.float32)


def _checkpoint_weights():
    weights = {
        "model.embed_tokens.weight": _tensor((16, 8), 1),
        "model.norm.weight": _tensor((8,), 2),
        "lm_head.weight": _tensor((16, 8), 3),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = _tensor((8,), 4 + layer)
        weights[prefix + "post_attention_layernorm.weight"] = _tensor((8,), 6 + layer)
    kda = "model.layers.0.self_attn."
    for index, projection in enumerate(("q_proj", "k_proj", "v_proj"), 10):
        weights[kda + projection + ".weight"] = _tensor((8, 8), index)
    weights[kda + "b_proj.weight"] = _tensor((2, 8), 13)
    weights[kda + "f_a_proj.weight"] = _tensor((4, 8), 14)
    weights[kda + "f_b_proj.weight"] = _tensor((8, 4), 15)
    weights[kda + "g_a_proj.weight"] = _tensor((4, 8), 16)
    weights[kda + "g_b_proj.weight"] = _tensor((8, 4), 17)
    for index, projection in enumerate(("q_conv1d", "k_conv1d", "v_conv1d"), 18):
        weights[kda + projection + ".weight"] = _tensor((8, 1, 4), index)
    weights[kda + "dt_bias"] = _tensor((8,), 21)
    weights[kda + "A_log"] = _tensor((1, 1, 2, 1), 22)
    weights[kda + "o_norm.weight"] = _tensor((4,), 23)
    weights[kda + "o_proj.weight"] = _tensor((8, 8), 24)
    dense = "model.layers.0.mlp."
    weights[dense + "gate_proj.weight"] = _tensor((12, 8), 25)
    weights[dense + "up_proj.weight"] = _tensor((12, 8), 26)
    weights[dense + "down_proj.weight"] = _tensor((8, 12), 27)

    mla = "model.layers.1.self_attn."
    weights[mla + "q_proj.weight"] = _tensor((8, 8), 30)
    weights[mla + "kv_a_proj_with_mqa.weight"] = _tensor((6, 8), 31)
    weights[mla + "kv_a_layernorm.weight"] = _tensor((4,), 32)
    weights[mla + "kv_b_proj.weight"] = _tensor((8, 4), 33)
    weights[mla + "o_proj.weight"] = _tensor((8, 4), 34)
    moe = "model.layers.1.block_sparse_moe."
    weights[moe + "gate.weight"] = _tensor((4, 8), 35)
    weights[moe + "gate.e_score_correction_bias"] = _tensor((4,), 36)
    weights[moe + "shared_experts.gate_proj.weight"] = _tensor((4, 8), 37)
    weights[moe + "shared_experts.up_proj.weight"] = _tensor((4, 8), 38)
    weights[moe + "shared_experts.down_proj.weight"] = _tensor((8, 4), 39)
    for expert in range(4):
        expert_prefix = moe + f"experts.{expert}."
        weights[expert_prefix + "w1.weight"] = _tensor((4, 8), 40 + expert)
        weights[expert_prefix + "w3.weight"] = _tensor((4, 8), 50 + expert)
        weights[expert_prefix + "w2.weight"] = _tensor((8, 4), 60 + expert)
    return weights


class KimiModelCheckpointTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("KIMI_LINEAR_CHECKPOINT"), "release checkpoint not provided")
    def test_release_router_gemm_layout(self):
        ckpt = Path(os.environ["KIMI_LINEAR_CHECKPOINT"])
        index = json.loads((ckpt/"model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.layers.1.block_sparse_moe.gate."
        data = {}
        for name in ("weight", "e_score_correction_bias"):
            with safe_open(str(ckpt/index[prefix+name]), framework="pt", device="cpu") as handle:
                data[name] = handle.get_tensor(prefix+name)

        class GateLoader(NewModelLoader):
            def _create_model(self):
                e, h = data["weight"].shape
                return DeepSeekV32MoeGate(h, e, True, torch.bfloat16)

        with tempfile.TemporaryDirectory() as root:
            save_file(data, str(Path(root)/"model.safetensors"))
            for method in (NewLoaderLoadMethod.SCRATCH, NewLoaderLoadMethod.FASTSAFETENSORS):
                with self.subTest(method=method):
                    loader = GateLoader({"model_type": "kimi_linear"},
                                        NewLoaderConfig(device="cpu", load_method=method), model_path=root)
                    gate = loader.load()
                    self.assertEqual(loader.loaded_method, method)
                    # Main KimiLinearWeight uses transpose(...).contiguous();
                    # CudaF16Linear consumes the transposed view of that matrix.
                    ref = data["weight"].to(torch.bfloat16).T.contiguous().T
                    self.assertEqual(gate.weight.stride(), ref.stride())
                    torch.testing.assert_close(gate.weight, ref, rtol=0, atol=0)
                    torch.testing.assert_close(gate.e_score_correction_bias, data["e_score_correction_bias"].float())
                    x = torch.randn(17, ref.shape[1], generator=torch.Generator().manual_seed(13)).bfloat16()
                    torch.testing.assert_close(gate(x), torch.nn.functional.linear(x, ref), rtol=0, atol=0)

    def test_entire_typed_model(self):
        with tempfile.TemporaryDirectory() as root:
            raw = _raw_config()
            (Path(root) / "config.json").write_text(json.dumps(raw))
            data = _checkpoint_weights()
            save_file(data, str(Path(root) / "model.safetensors"))
            config = KimiLinear._create_config(root)
            config.ckpt_path = root
            config.model_type = "kimi_linear"
            config.max_seq_len = 128
            config.enable_fp32_lm_head = False
            config.data_type, config.lora_infos = "fp32", {}
            config.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
            config.linear_attention_config.conv_state_dtype = DataType.TYPE_FP32
            for method in (NewLoaderLoadMethod.SCRATCH, NewLoaderLoadMethod.FASTSAFETENSORS):
                with self.subTest(method=method):
                    service = SimpleNamespace(
                        model_config=config, _weight_alias_names=(), _weight_alias_owner=None,
                        _get_device_str=lambda: "cpu", parallelism_config=ParallelismConfig(),
                        hw_kernel_config=None, moe_config=None, fmha_config=None,
                        max_generate_batch_size=1, load_method=method, custom_module=None,
                    )
                    load_owned_model(service)
                    model = service.py_model
                    self.assertIsInstance(model, KimiLinearForCausalLM)
                    self.assertIsInstance(model, KimiLinearModel)
                    self.assertEqual(service.model_weights_loader.loaded_method, method)
                    kda, mla = model.layers
                    self.assertTrue(kda.is_linear)
                    self.assertFalse(mla.is_linear)
                    self.assertIs(kda.self_attn.prefill_kda.conv_weights, kda.self_attn.conv1d.weight)
                    self.assertIs(kda.self_attn.decode_kda.alog, kda.self_attn.A_log)
                    self.assertIsNotNone(mla.block_sparse_moe)
                    gate = mla.block_sparse_moe.gate
                    expected_gate = data["model.layers.1.block_sparse_moe.gate.weight"].T.contiguous().T
                    self.assertEqual(gate.weight.stride(), expected_gate.stride())
                    torch.testing.assert_close(gate.weight, expected_gate)
                    self.assertIs(model.runtime_weight_view()["lm_head"], model.lm_head.weight)
                    torch.testing.assert_close(model.embed_tokens.weight, data["model.embed_tokens.weight"])
                    torch.testing.assert_close(model.lm_head.weight, data["lm_head.weight"])
                    model._ensure_mla_kernel_layout()
                    self.assertEqual(model._mla_kernel_layout.weights[0], {})
                    self.assertTrue(model._mla_kernel_layout.weights[1])

            from rtp_llm.models.downstream_modules.classifier.classifier import ClassifierHandler
            from rtp_llm.models.downstream_modules.custom_module import CustomModule
            from rtp_llm.ops import TaskType

            raw["num_labels"] = 3
            (Path(root) / "config.json").write_text(json.dumps(raw))
            data["classifier.weight"] = torch.arange(24).float().reshape(3, 8)
            data["classifier.bias"] = torch.arange(3).float()
            save_file(data, str(Path(root) / "model.safetensors"))
            config.task_type = TaskType.SEQ_CLASSIFICATION
            service.custom_module = CustomModule(config, None)
            service.custom_module.handler = ClassifierHandler(config)
            service.custom_module.handler.device = torch.device("cpu")
            load_owned_model(service)
            self.assertIsNone(service.py_model.lm_head)
            linear = service.custom_module.handler.linear
            torch.testing.assert_close(linear.weight, data["classifier.weight"])
            torch.testing.assert_close(linear.bias, data["classifier.bias"])


if __name__ == "__main__":
    unittest.main()
