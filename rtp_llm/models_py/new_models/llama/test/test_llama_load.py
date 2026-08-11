import json
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import collect_loaded_tensor_ids
from rtp_llm.models_py.new_models.llama import LlamaForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.ops.compute_ops import PyModelInputs


def _model_config(*, tied=True, ckpt_path=""):
    return types.SimpleNamespace(
        model_type="llama",
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
        tie_word_embeddings=tied,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        ckpt_path=ckpt_path,
    )


def _load_config(*, tp_size=1, tp_rank=0, dtype=torch.float32, quant_config=None):
    parallelism = types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=1,
        ep_rank=0,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False,
            is_prefill_enabled=lambda: False,
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
        get_attn_tp_size=lambda: tp_size,
        get_attn_tp_rank=lambda: tp_rank,
        get_ffn_tp_size=lambda: tp_size,
        get_ffn_tp_rank=lambda: tp_rank,
    )
    return NewLoaderConfig(
        tp_size=tp_size,
        tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_tp_rank=tp_rank,
        ffn_tp_size=tp_size,
        ffn_tp_rank=tp_rank,
        lm_head_tp_size=tp_size,
        lm_head_tp_rank=tp_rank,
        compute_dtype=dtype,
        device="cpu",
        quant_config=quant_config or QuantizationConfig("none"),
        parallelism_config=parallelism,
    )


def _weights(*, include_lm_head=False, dtype=torch.float32):
    weights = {
        "model.embed_tokens.weight": torch.arange(32, dtype=dtype).reshape(8, 4),
        "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=dtype),
        "model.layers.0.self_attn.q_proj.weight": torch.arange(16, dtype=dtype).reshape(
            4, 4
        ),
        "model.layers.0.self_attn.k_proj.weight": torch.arange(8, dtype=dtype).reshape(
            2, 4
        ),
        "model.layers.0.self_attn.v_proj.weight": torch.arange(
            8, 16, dtype=dtype
        ).reshape(2, 4),
        "model.layers.0.self_attn.o_proj.weight": torch.eye(4, dtype=dtype),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4, dtype=dtype),
        "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 4, dtype=dtype),
        "model.layers.0.mlp.up_proj.weight": torch.full((4, 4), 2.0, dtype=dtype),
        "model.layers.0.mlp.down_proj.weight": torch.ones(4, 4, dtype=dtype),
        "model.norm.weight": torch.ones(4, dtype=dtype),
    }
    if include_lm_head:
        weights["lm_head.weight"] = torch.arange(32, 64, dtype=dtype).reshape(8, 4)
    return weights


class LlamaLoadTest(unittest.TestCase):
    def test_registry_resolves_only_the_declared_llama_route(self):
        self.assertIs(get_model_class("llama"), LlamaForCausalLM)

    def test_streaming_load_is_complete_and_ties_missing_lm_head(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(_weights(), f"{model_path}/model.safetensors")
            with open(f"{model_path}/config.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_act": "silu",
                        "attention_bias": False,
                        "mlp_bias": False,
                    },
                    handle,
                )
            model = NewModelLoader(
                _model_config(ckpt_path=model_path),
                _load_config(),
                model_path=model_path,
            ).load()

        self.assertIsInstance(model, LlamaForCausalLM)
        self.assertFalse(model.training)
        self.assertIsNone(model.layers[0].self_attn.qkv_proj.bias)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        self.assertIn(id(model.lm_head.weight), collect_loaded_tensor_ids(model))

    def test_untied_lm_head_is_required_and_loaded(self):
        model = LlamaForCausalLM(_model_config(tied=False), _load_config())
        model.load_weights(_weights())
        with self.assertRaisesRegex(RuntimeError, "ParallelLMHead.*weight"):
            NewModelLoader._validate_loaded_weights(model)

        model = LlamaForCausalLM(_model_config(tied=False), _load_config())
        weights = _weights(include_lm_head=True)
        model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)
        torch.testing.assert_close(model.lm_head.weight, weights["lm_head.weight"])

    def test_tp2_owns_rank_local_qkv_mlp_embedding_and_lm_head(self):
        weights = _weights()
        for rank in range(2):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as model_path:
                save_file(weights, f"{model_path}/model.safetensors")
                model = NewModelLoader(
                    _model_config(),
                    _load_config(tp_size=2, tp_rank=rank),
                    model_path=model_path,
                ).load()

                start = rank * 4
                torch.testing.assert_close(
                    model.embed_tokens.weight,
                    weights["model.embed_tokens.weight"][start : start + 4],
                )
                torch.testing.assert_close(
                    model.lm_head.weight, model.embed_tokens.weight
                )
                self.assertEqual(
                    model.layers[0].self_attn.qkv_proj.weight.shape, (6, 4)
                )
                self.assertEqual(model.layers[0].mlp.gate_up_proj.weight.shape, (4, 4))

    def test_fp16_bf16_and_fp32_checkpoint_loading(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                model = LlamaForCausalLM(
                    _model_config(tied=False), _load_config(dtype=dtype)
                )
                model.load_weights(_weights(include_lm_head=True, dtype=dtype))
                NewModelLoader._validate_loaded_weights(model)
                self.assertEqual(model.embed_tokens.weight.dtype, dtype)

    def test_quantization_dispatch_uses_fully_qualified_layer_prefixes(self):
        class RecordingQuantConfig(QuantizationConfig):
            def __init__(self):
                super().__init__("none")
                self.prefixes = []

            def get_quant_method(self, layer, prefix=""):
                self.prefixes.append(prefix)
                return super().get_quant_method(layer, prefix)

        quant_config = RecordingQuantConfig()
        LlamaForCausalLM(_model_config(), _load_config(quant_config=quant_config))
        self.assertEqual(
            quant_config.prefixes,
            [
                "layers.0.self_attn.qkv_proj",
                "layers.0.self_attn.o_proj",
                "layers.0.mlp.gate_up_proj",
                "layers.0.mlp.down_proj",
            ],
        )

    def test_forward_matches_independent_cpu_reference(self):
        weights = _weights()
        model = LlamaForCausalLM(_model_config(), _load_config())
        model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)

        class QOnlyFmha:
            fmha_params = None

            def __init__(self, q_size):
                self.q_size = q_size

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv[..., : self.q_size]

        def rms_norm(x, weight):
            normalized = x.float() * torch.rsqrt(
                x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
            )
            return (normalized * weight).to(x.dtype)

        input_ids = torch.tensor([0, 3, 7], dtype=torch.int32)
        inputs = PyModelInputs(input_ids=input_ids)
        fmha = QOnlyFmha(model.layers[0].self_attn.qkv_proj.q_size)
        with patch(
            "rtp_llm.models_py.new_models.qwen2.language.select_fmha_impl_for_layer",
            return_value=fmha,
        ) as select_fmha:
            outputs = model(inputs, fmha_impl=fmha)
        select_fmha.assert_called_once_with(fmha, model.kv_cache, 0)

        hidden = F.embedding(input_ids.long(), weights["model.embed_tokens.weight"])
        residual = hidden
        hidden = rms_norm(hidden, weights["model.layers.0.input_layernorm.weight"])
        hidden = F.linear(hidden, weights["model.layers.0.self_attn.q_proj.weight"])
        hidden = F.linear(hidden, weights["model.layers.0.self_attn.o_proj.weight"])
        hidden = residual + hidden
        residual = hidden
        hidden = rms_norm(
            hidden, weights["model.layers.0.post_attention_layernorm.weight"]
        )
        gate = F.linear(hidden, weights["model.layers.0.mlp.gate_proj.weight"])
        up = F.linear(hidden, weights["model.layers.0.mlp.up_proj.weight"])
        hidden = F.linear(
            F.silu(gate.float()) * up.float(),
            weights["model.layers.0.mlp.down_proj.weight"],
        )
        expected = rms_norm(residual + hidden, weights["model.norm.weight"])
        torch.testing.assert_close(outputs.hidden_states, expected)

    def test_unknown_missing_and_bias_tensors_fail_fast(self):
        bad_cases = []
        unknown = _weights()
        unknown["model.layers.0.mlp.gate_prjo.weight"] = torch.ones(4, 4)
        bad_cases.append((unknown, False, "could not dispatch"))
        missing = _weights()
        del missing["model.layers.0.self_attn.k_proj.weight"]
        bad_cases.append((missing, True, "QKVParallelLinear"))
        biased = _weights()
        biased["model.layers.0.self_attn.q_proj.bias"] = torch.ones(4)
        bad_cases.append((biased, False, "Unexpected bias tensor"))

        for weights, validate_after_load, pattern in bad_cases:
            with self.subTest(pattern=pattern):
                model = LlamaForCausalLM(_model_config(), _load_config())
                if validate_after_load:
                    model.load_weights(weights)
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        NewModelLoader._validate_loaded_weights(model)
                else:
                    with self.assertRaisesRegex(RuntimeError, pattern):
                        model.load_weights(weights)

    def test_attention_bias_config_fails_before_loading(self):
        config = _model_config()
        config.attention_bias = True
        with self.assertRaisesRegex(ValueError, "Attention projection bias"):
            LlamaForCausalLM(config, _load_config())

    def test_checkpoint_architecture_and_activation_fail_before_loading(self):
        bad_configs = (
            (
                {
                    "model_type": "llama",
                    "architectures": ["YiForCausalLM"],
                    "hidden_act": "silu",
                },
                "LlamaForCausalLM checkpoint layout",
            ),
            (
                {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                    "hidden_act": "gelu",
                },
                "hidden_act='silu'",
            ),
        )
        for raw_config, pattern in bad_configs:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as path:
                with open(f"{path}/config.json", "w", encoding="utf-8") as handle:
                    json.dump(raw_config, handle)
                with self.assertRaisesRegex(ValueError, pattern):
                    LlamaForCausalLM(
                        _model_config(ckpt_path=path),
                        _load_config(),
                    )

    def test_mlp_bias_and_non_silu_dict_configs_fail_before_loading(self):
        config = {
            "model_type": "llama",
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "enable_fp32_lm_head": False,
            "tie_word_embeddings": True,
            "mlp_bias": True,
        }
        with self.assertRaisesRegex(ValueError, "MLP projection bias"):
            LlamaForCausalLM(config, _load_config())

        config["mlp_bias"] = False
        config["hidden_act"] = "gelu"
        with self.assertRaisesRegex(ValueError, "hidden_act='silu'"):
            LlamaForCausalLM(config, _load_config())


if __name__ == "__main__":
    unittest.main()
