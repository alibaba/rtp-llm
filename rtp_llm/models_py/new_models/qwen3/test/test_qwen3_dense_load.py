import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import collect_loaded_tensor_ids
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig


def _validate(module) -> None:
    NewModelLoader._validate_loaded_weights(module)


class LinearPartitionTest(unittest.TestCase):
    def test_merged_separate_and_fused_weights_match_for_tp(self):
        gate = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        up = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)

        separate = MergedColumnParallelLinear(
            input_size=2,
            output_size=8,
            tp_size=2,
            tp_rank=1,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
        )
        separate.load_weights({"gate_proj.weight": gate, "up_proj.weight": up})

        fused = MergedColumnParallelLinear(
            input_size=2,
            output_size=8,
            tp_size=2,
            tp_rank=1,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
        )
        fused.load_weights({"gate_up_proj.weight": torch.cat((gate, up), dim=0)})

        expected = torch.cat((gate[2:], up[2:]), dim=0)
        torch.testing.assert_close(separate.weight, expected)
        torch.testing.assert_close(fused.weight, expected)
        _validate(separate)
        _validate(fused)

    def test_qkv_kv_replication_uses_contiguous_rank_groups(self):
        q = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        k = torch.tensor([[20.0, 21.0], [30.0, 31.0]])
        v = torch.tensor([[40.0, 41.0], [50.0, 51.0]])
        expected_kv_heads = [0, 0, 1, 1]

        for rank, kv_head in enumerate(expected_kv_heads):
            layer = QKVParallelLinear(
                hidden_size=2,
                num_heads=8,
                num_kv_heads=2,
                head_dim=1,
                tp_size=4,
                tp_rank=rank,
                params_dtype=torch.float32,
            )
            layer.load_weights(
                {
                    "q_proj.weight": q,
                    "k_proj.weight": k,
                    "v_proj.weight": v,
                }
            )
            expected = torch.cat(
                (
                    q[rank * 2 : rank * 2 + 2],
                    k[kv_head : kv_head + 1],
                    v[kv_head : kv_head + 1],
                ),
                dim=0,
            )
            torch.testing.assert_close(layer.weight, expected)
            _validate(layer)

    def test_row_parallel_reduces_before_adding_bias(self):
        layer = RowParallelLinear(
            input_size=4,
            output_size=2,
            tp_size=2,
            tp_rank=0,
            bias=True,
            params_dtype=torch.float32,
        )
        layer.load_weights(
            {
                "weight": torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
                "bias": torch.tensor([3.0, 5.0]),
            }
        )
        inputs = torch.tensor([[2.0, 4.0]])
        with patch(
            "rtp_llm.models_py.layers.linear.all_reduce",
            side_effect=lambda tensor, group: tensor * 2,
        ):
            output = layer(inputs)
        torch.testing.assert_close(output, torch.tensor([[7.0, 13.0]]))

    def test_merged_rejects_mixed_and_oversized_shards(self):
        layer = MergedColumnParallelLinear(
            input_size=2,
            output_size=8,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
        )
        with self.assertRaisesRegex(ValueError, "rows must be 4"):
            layer.load_weights({"gate_proj.weight": torch.ones(5, 2)})

        layer.load_weights({"gate_proj.weight": torch.ones(4, 2)})
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            layer.load_weights({"gate_proj.weight": torch.zeros(4, 2)})
        torch.testing.assert_close(layer.weight[:4], torch.ones(4, 2))
        with self.assertRaisesRegex(RuntimeError, "mix fused and per-shard"):
            layer.load_weights({"gate_up_proj.weight": torch.ones(8, 2)})

    def test_merged_rejects_non_floating_shard_conversion(self):
        layer = MergedColumnParallelLinear(
            input_size=2,
            output_size=8,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
        )

        with self.assertRaisesRegex(TypeError, "Dtype mismatch"):
            layer.load_weights(
                {"gate_proj.weight": torch.ones(4, 2, dtype=torch.int32)}
            )

    def test_qkv_rejects_invalid_gqa(self):
        with self.assertRaisesRegex(ValueError, "num_heads=8"):
            QKVParallelLinear(
                hidden_size=8,
                num_heads=8,
                num_kv_heads=6,
                head_dim=1,
                tp_size=4,
                params_dtype=torch.float32,
            )


class Qwen3LoadTest(unittest.TestCase):
    def _config(self):
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
        )

    def _load_config(self):
        parallelism = types.SimpleNamespace(tp_size=1, tp_rank=0)
        return NewLoaderConfig(
            compute_dtype=torch.float32,
            device="cpu",
            quant_config=QuantizationConfig("none"),
            parallelism_config=parallelism,
        )

    def _model(self):
        return Qwen3ForCausalLM(self._config(), self._load_config())

    def _weights(self):
        values = {
            "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
                8, 4
            ),
            "model.layers.0.input_layernorm.weight": torch.ones(4),
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                16, dtype=torch.float32
            ).reshape(4, 4),
            "model.layers.0.self_attn.k_proj.weight": torch.arange(
                8, dtype=torch.float32
            ).reshape(2, 4),
            "model.layers.0.self_attn.v_proj.weight": torch.arange(
                8, 16, dtype=torch.float32
            ).reshape(2, 4),
            "model.layers.0.self_attn.q_norm.weight": torch.ones(2),
            "model.layers.0.self_attn.k_norm.weight": torch.ones(2),
            "model.layers.0.self_attn.o_proj.weight": torch.eye(4),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
            "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 4),
            "model.layers.0.mlp.up_proj.weight": torch.full((4, 4), 2.0),
            "model.layers.0.mlp.down_proj.weight": torch.ones(4, 4),
            "model.norm.weight": torch.ones(4),
        }
        return values

    def test_real_model_load_validates_and_ties_missing_lm_head(self):
        model = self._model()
        weights = self._weights()

        model.load_weights(weights)
        _validate(model)

        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        self.assertIn(id(model.lm_head.weight), collect_loaded_tensor_ids(model))
        self.assertEqual(
            set(model.runtime_weight_view()),
            {"embedding", "final_layernorm.gamma", "lm_head"},
        )

    def test_new_model_loader_streams_real_safetensors(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(self._weights(), f"{model_path}/model.safetensors")
            loader = NewModelLoader(
                self._config(),
                self._load_config(),
                model_path=model_path,
            )

            model = loader.load()

        self.assertIsInstance(model, Qwen3ForCausalLM)
        self.assertFalse(model.training)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)

    def test_unknown_checkpoint_tensor_fails(self):
        model = self._model()
        weights = self._weights()
        weights["model.layers.0.mlp.gate_prjo.weight"] = torch.ones(4, 4)

        with self.assertRaisesRegex(RuntimeError, "could not dispatch"):
            model.load_weights(weights)

    def test_unsupported_quantization_fails_fast(self):
        config = QuantizationConfig("fp8")
        with self.assertRaisesRegex(ValueError, "not supported"):
            QKVParallelLinear(
                hidden_size=4,
                num_heads=2,
                num_kv_heads=1,
                head_dim=2,
                quant_config=config,
            )

    def test_missing_required_config_value_fails_fast(self):
        config = self._config()
        del config.hidden_size

        with self.assertRaisesRegex(ValueError, "hidden_size"):
            Qwen3ForCausalLM(config, self._load_config())

    def test_hf_dict_config_constructs_same_model_shape(self):
        config = {
            "model_type": "qwen_3",
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "enable_fp32_lm_head": False,
        }

        model = Qwen3ForCausalLM(config, self._load_config())

        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.embed_tokens.weight.shape, (8, 4))
        self.assertEqual(model.layers[0].self_attn.qkv_proj.weight.shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
