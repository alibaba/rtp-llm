import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import collect_loaded_tensor_ids
from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.ops.compute_ops import PyModelInputs
from safetensors.torch import save_file


def _validate(module) -> None:
    NewModelLoader._validate_loaded_weights(module)


class LinearPartitionTest(unittest.TestCase):
    def test_column_parallel_rejects_unimplemented_gather_output(self):
        with self.assertRaisesRegex(ValueError, "gather_output is not supported"):
            ColumnParallelLinear(
                input_size=2,
                output_size=4,
                gather_output=True,
                params_dtype=torch.float32,
            )

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

    def test_qkv_rejects_misaligned_gcd_topology(self):
        with self.assertRaisesRegex(ValueError, "mutually divisible"):
            QKVParallelLinear(
                hidden_size=12,
                num_heads=12,
                num_kv_heads=6,
                head_dim=1,
                tp_size=4,
                params_dtype=torch.float32,
            )


class Qwen2LoadTest(unittest.TestCase):
    def _config(self):
        return types.SimpleNamespace(
            model_type="qwen_2",
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
        )

    def _load_config(
        self,
        tp_size=1,
        tp_rank=0,
        attn_tp_size=None,
        attn_tp_rank=None,
        ffn_tp_size=None,
        ffn_tp_rank=None,
        lm_head_tp_size=None,
        lm_head_tp_rank=None,
        cp_enabled=False,
        prefill_cp_enabled=False,
        ffn_disaggregate=False,
    ):
        attn_tp_size = tp_size if attn_tp_size is None else attn_tp_size
        attn_tp_rank = tp_rank if attn_tp_rank is None else attn_tp_rank
        ffn_tp_size = tp_size if ffn_tp_size is None else ffn_tp_size
        ffn_tp_rank = tp_rank if ffn_tp_rank is None else ffn_tp_rank
        lm_head_tp_size = tp_size if lm_head_tp_size is None else lm_head_tp_size
        lm_head_tp_rank = tp_rank if lm_head_tp_rank is None else lm_head_tp_rank
        parallelism = types.SimpleNamespace(
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=1,
            ep_rank=0,
            prefill_cp_config=types.SimpleNamespace(
                is_enabled=lambda: cp_enabled,
                is_prefill_enabled=lambda: prefill_cp_enabled,
            ),
            ffn_disaggregate_config=types.SimpleNamespace(
                enable_ffn_disaggregate=ffn_disaggregate
            ),
            get_attn_tp_size=lambda: attn_tp_size,
            get_attn_tp_rank=lambda: attn_tp_rank,
            get_ffn_tp_size=lambda: ffn_tp_size,
            get_ffn_tp_rank=lambda: ffn_tp_rank,
        )
        return NewLoaderConfig(
            tp_size=tp_size,
            tp_rank=tp_rank,
            attn_tp_size=attn_tp_size,
            attn_tp_rank=attn_tp_rank,
            ffn_tp_size=ffn_tp_size,
            ffn_tp_rank=ffn_tp_rank,
            lm_head_tp_size=lm_head_tp_size,
            lm_head_tp_rank=lm_head_tp_rank,
            compute_dtype=torch.float32,
            device="cpu",
            quant_config=QuantizationConfig("none"),
            parallelism_config=parallelism,
        )

    def _model(self):
        return Qwen2ForCausalLM(self._config(), self._load_config())

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
            "model.layers.0.self_attn.q_proj.bias": torch.full((4,), 0.5),
            "model.layers.0.self_attn.k_proj.bias": torch.full((2,), -0.25),
            "model.layers.0.self_attn.v_proj.bias": torch.full((2,), 0.75),
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

    def test_untied_model_rejects_missing_lm_head(self):
        config = self._config()
        config.tie_word_embeddings = False
        model = Qwen2ForCausalLM(config, self._load_config())

        model.load_weights(self._weights())

        with self.assertRaisesRegex(RuntimeError, "ParallelLMHead.*weight"):
            _validate(model)

    def test_untied_model_loads_explicit_lm_head(self):
        config = self._config()
        config.tie_word_embeddings = False
        model = Qwen2ForCausalLM(config, self._load_config())
        weights = self._weights()
        weights["lm_head.weight"] = torch.arange(32, 64, dtype=torch.float32).reshape(
            8, 4
        )

        model.load_weights(weights)
        _validate(model)

        torch.testing.assert_close(model.lm_head.weight, weights["lm_head.weight"])

    def test_lm_head_postprocess_matches_legacy_logits_for_tp(self):
        hidden_states = torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 0.25, 2.0, 0.75]])
        explicit_lm_head = torch.arange(32, 64, dtype=torch.float32).reshape(8, 4)

        for tied in (True, False):
            config = self._config()
            config.tie_word_embeddings = tied
            config.normalize_lm_head_weight = True
            config.logit_scale = 0.25
            weights = self._weights()
            source = weights["model.embed_tokens.weight"]
            if not tied:
                weights["lm_head.weight"] = explicit_lm_head
                source = explicit_lm_head

            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as model_path:
                save_file(weights, f"{model_path}/model.safetensors")
                for rank in range(2):
                    model = NewModelLoader(
                        model_config=config,
                        load_config=self._load_config(tp_size=2, tp_rank=rank),
                        model_path=model_path,
                    ).load()
                    local_source = source[rank * 4 : (rank + 1) * 4]
                    expected_weight = F.normalize(local_source, dim=1) * 0.25
                    torch.testing.assert_close(model.lm_head.weight, expected_weight)
                    torch.testing.assert_close(
                        F.linear(hidden_states, model.runtime_weight_view()["lm_head"]),
                        F.linear(hidden_states, expected_weight),
                    )

    def test_checkpoint_lm_head_must_use_global_vocab_shape(self):
        config = self._config()
        config.tie_word_embeddings = False
        model = Qwen2ForCausalLM(config, self._load_config(tp_size=2))
        weights = self._weights()
        weights["lm_head.weight"] = torch.ones(4, 4)

        with self.assertRaisesRegex(ValueError, "checkpoint rows must be 8"):
            model.load_weights(weights)

    def test_parallel_topologies_are_applied_to_their_owners(self):
        model = Qwen2ForCausalLM(
            self._config(), self._load_config(tp_size=2, tp_rank=1)
        )

        self.assertEqual(
            (model.embed_tokens.tp_size, model.embed_tokens.tp_rank), (2, 1)
        )
        self.assertEqual(
            (
                model.layers[0].self_attn.qkv_proj.tp_size,
                model.layers[0].self_attn.qkv_proj.tp_rank,
            ),
            (2, 1),
        )
        self.assertEqual(
            (
                model.layers[0].mlp.gate_up_proj.tp_size,
                model.layers[0].mlp.gate_up_proj.tp_rank,
            ),
            (2, 1),
        )
        self.assertEqual((model.lm_head.tp_size, model.lm_head.tp_rank), (2, 1))

    def test_legacy_negative_one_kv_head_sentinel_uses_q_head_count(self):
        config = self._config()
        config.attn_config.kv_head_num = -1

        model = Qwen2ForCausalLM(config, self._load_config())

        self.assertEqual(model.layers[0].self_attn.num_kv_heads, 2)

    def test_context_parallel_topology_is_rejected(self):
        load_config = self._load_config(
            tp_size=2,
            tp_rank=1,
            attn_tp_size=1,
            attn_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            cp_enabled=True,
        )

        with self.assertRaisesRegex(ValueError, "Context parallelism"):
            Qwen2ForCausalLM(self._config(), load_config)

    def test_prefill_context_parallel_mode_is_rejected(self):
        load_config = self._load_config(prefill_cp_enabled=True)

        with self.assertRaisesRegex(ValueError, "Context parallelism"):
            Qwen2ForCausalLM(self._config(), load_config)

    def test_independent_ffn_topology_is_rejected(self):
        load_config = self._load_config(
            tp_size=2,
            tp_rank=1,
            ffn_tp_size=1,
            ffn_tp_rank=0,
        )

        with self.assertRaisesRegex(ValueError, "Independent FFN"):
            Qwen2ForCausalLM(self._config(), load_config)

    def test_ffn_disaggregation_is_rejected(self):
        load_config = self._load_config(ffn_disaggregate=True)

        with self.assertRaisesRegex(ValueError, "FFN disaggregation"):
            Qwen2ForCausalLM(self._config(), load_config)

    def test_separate_lm_head_topology_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lm_head_tp partition"):
            self._load_config(
                tp_size=2,
                tp_rank=1,
                lm_head_tp_size=1,
                lm_head_tp_rank=0,
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

        self.assertIsInstance(model, Qwen2ForCausalLM)
        self.assertFalse(model.training)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)

    def test_rank_local_consolidated_checkpoint_is_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as model_path:
            torch.save(self._weights(), f"{model_path}/consolidated.00.pth")
            loader = NewModelLoader(
                self._config(),
                self._load_config(),
                model_path=model_path,
            )

            with self.assertRaisesRegex(
                ValueError, "does not support rank-local consolidated checkpoints"
            ):
                loader.load()

    def test_real_model_forward_matches_cpu_reference(self):
        weights = self._weights()
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            model = NewModelLoader(
                self._config(), self._load_config(), model_path=model_path
            ).load()

        class QOnlyFmha:
            fmha_params = None

            def __init__(self, q_size):
                self.q_size = q_size

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv[..., : self.q_size]

        def rms_norm(x, weight, eps=1e-6):
            normalized = x.float() * torch.rsqrt(
                x.float().pow(2).mean(dim=-1, keepdim=True) + eps
            )
            return (normalized * weight).to(x.dtype)

        input_ids = torch.tensor([0, 3, 7], dtype=torch.int32)
        fmha = QOnlyFmha(model.layers[0].self_attn.qkv_proj.q_size)
        inputs = PyModelInputs(input_ids=input_ids)
        with patch(
            "rtp_llm.models_py.new_models.qwen2.language.select_block_map_for_layer"
        ) as select_block_map:
            outputs = model(inputs, fmha_impl=fmha)
        select_block_map.assert_called_once_with(inputs.attention_inputs, 0)

        hidden = F.embedding(input_ids.long(), weights["model.embed_tokens.weight"])
        residual = hidden
        hidden = rms_norm(hidden, weights["model.layers.0.input_layernorm.weight"])
        q = F.linear(
            hidden,
            weights["model.layers.0.self_attn.q_proj.weight"],
            weights["model.layers.0.self_attn.q_proj.bias"],
        )
        hidden = F.linear(q, weights["model.layers.0.self_attn.o_proj.weight"])
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
        hidden = residual + hidden
        expected = rms_norm(hidden, weights["model.norm.weight"])

        torch.testing.assert_close(outputs.hidden_states, expected)
        torch.testing.assert_close(
            model.lm_head(outputs.hidden_states),
            F.linear(expected, weights["model.embed_tokens.weight"]),
        )

    def test_unknown_checkpoint_tensor_fails(self):
        model = self._model()
        weights = self._weights()
        weights["model.layers.0.mlp.gate_prjo.weight"] = torch.ones(4, 4)

        with self.assertRaisesRegex(RuntimeError, "could not dispatch"):
            model.load_weights(weights)

    def test_unsupported_quantization_fails_fast(self):
        config = QuantizationConfig("definitely_unsupported")
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
            Qwen2ForCausalLM(config, self._load_config())

    def test_hf_dict_config_constructs_same_model_shape(self):
        config = {
            "model_type": "qwen_2",
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
        }

        model = Qwen2ForCausalLM(config, self._load_config())

        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.embed_tokens.weight.shape, (8, 4))
        self.assertEqual(model.layers[0].self_attn.qkv_proj.weight.shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
