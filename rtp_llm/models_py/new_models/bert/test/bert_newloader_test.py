import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.bert import BertForEmbedding, RobertaForEmbedding
from rtp_llm.models_py.new_models.bert.language import CUSTOM_WEIGHT_PREFIX
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.utils.model_weight import W
from safetensors.torch import save_file


def _config(model_type="bert"):
    config = ModelConfig()
    config.model_type = model_type
    config.hidden_size = 4
    config.inter_size = 8
    config.num_layers = 1
    config.vocab_size = 7
    config.max_seq_len = 6
    config.type_vocab_size = 2
    config.layernorm_eps = 1e-5
    config.activation_type = "gelu"
    config.norm_type = "layernorm"
    config.layernorm_type = "post_layernorm"
    config.has_positional_encoding = True
    config.has_pre_decoder_layernorm = True
    config.has_post_decoder_layernorm = False
    config.has_lm_head = False
    config.attn_config.head_num = 2
    config.attn_config.kv_head_num = 2
    config.attn_config.size_per_head = 2
    config.attn_config.is_causal = False
    config.attn_config.rope_config.dim = 0
    return config


def _weights(prefix="bert", layernorm_alias=False, include_token_type=True):
    ln_weight = "gamma" if layernorm_alias else "weight"
    ln_bias = "beta" if layernorm_alias else "bias"
    hidden = 4
    inter = 8
    values = {
        f"{prefix}.embeddings.word_embeddings.weight": torch.arange(
            7 * hidden, dtype=torch.float32
        ).reshape(7, hidden),
        f"{prefix}.embeddings.position_embeddings.weight": torch.arange(
            6 * hidden, dtype=torch.float32
        ).reshape(6, hidden),
        f"{prefix}.embeddings.LayerNorm.{ln_weight}": torch.ones(hidden),
        f"{prefix}.embeddings.LayerNorm.{ln_bias}": torch.zeros(hidden),
        f"{prefix}.encoder.layer.0.attention.self.query.weight": torch.full(
            (hidden, hidden), 1.0
        ),
        f"{prefix}.encoder.layer.0.attention.self.query.bias": torch.full(
            (hidden,), 2.0
        ),
        f"{prefix}.encoder.layer.0.attention.self.key.weight": torch.full(
            (hidden, hidden), 3.0
        ),
        f"{prefix}.encoder.layer.0.attention.self.key.bias": torch.full((hidden,), 4.0),
        f"{prefix}.encoder.layer.0.attention.self.value.weight": torch.full(
            (hidden, hidden), 5.0
        ),
        f"{prefix}.encoder.layer.0.attention.self.value.bias": torch.full(
            (hidden,), 6.0
        ),
        f"{prefix}.encoder.layer.0.attention.output.dense.weight": torch.full(
            (hidden, hidden), 7.0
        ),
        f"{prefix}.encoder.layer.0.attention.output.dense.bias": torch.full(
            (hidden,), 8.0
        ),
        f"{prefix}.encoder.layer.0.attention.output.LayerNorm.{ln_weight}": torch.ones(
            hidden
        ),
        f"{prefix}.encoder.layer.0.attention.output.LayerNorm.{ln_bias}": torch.zeros(
            hidden
        ),
        f"{prefix}.encoder.layer.0.intermediate.dense.weight": torch.full(
            (inter, hidden), 9.0
        ),
        f"{prefix}.encoder.layer.0.intermediate.dense.bias": torch.full((inter,), 10.0),
        f"{prefix}.encoder.layer.0.output.dense.weight": torch.full(
            (hidden, inter), 11.0
        ),
        f"{prefix}.encoder.layer.0.output.dense.bias": torch.full((hidden,), 12.0),
        f"{prefix}.encoder.layer.0.output.LayerNorm.{ln_weight}": torch.ones(hidden),
        f"{prefix}.encoder.layer.0.output.LayerNorm.{ln_bias}": torch.zeros(hidden),
    }
    if include_token_type:
        values[f"{prefix}.embeddings.token_type_embeddings.weight"] = torch.full(
            (2, hidden), 13.0
        )
    return values


class BertNewloaderTest(unittest.TestCase):
    def _load(
        self,
        config,
        weights,
        *,
        compute_dtype=torch.float16,
        **load_kwargs,
    ):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, os.path.join(model_path, "model.safetensors"))
            model = NewModelLoader(
                config,
                NewLoaderConfig(
                    device="cpu", compute_dtype=compute_dtype, **load_kwargs
                ),
                model_path=model_path,
            ).load()
        return model

    def test_registry_resolves_bert_and_roberta(self):
        self.assertIs(get_model_class("bert"), BertForEmbedding)
        self.assertIs(get_model_class("roberta"), RobertaForEmbedding)

    def test_real_bert_load_owns_checkpoint_parameters_in_model_tree(self):
        model = self._load(_config(), _weights())
        self.assertIsInstance(model, BertForEmbedding)
        self.assertFalse(model.training)
        qkv = model.layers[0].self_attn.qkv_proj.weight
        self.assertEqual(tuple(qkv.shape), (12, 4))
        self.assertTrue(
            torch.allclose(qkv[:4], torch.full((4, 4), 1.0, dtype=qkv.dtype))
        )
        self.assertTrue(
            torch.allclose(qkv[4:8], torch.full((4, 4), 3.0, dtype=qkv.dtype))
        )
        self.assertTrue(
            torch.allclose(qkv[8:], torch.full((4, 4), 5.0, dtype=qkv.dtype))
        )
        self.assertIs(
            model.runtime_weight_view()[W.embedding],
            model.embeddings.word_embeddings.weight,
        )
        self.assertIsNone(model.weight)
        self.assertFalse(hasattr(model, "weights"))
        self.assertFalse(hasattr(model, "model"))

    def test_roberta_and_layernorm_gamma_beta_are_supported(self):
        model = self._load(
            _config("roberta"),
            _weights(prefix="roberta", layernorm_alias=True),
        )
        self.assertIsInstance(model, RobertaForEmbedding)
        self.assertTrue(
            torch.allclose(
                model.runtime_weight_view()[W.pre_decoder_ln_gamma],
                torch.ones(4, dtype=torch.float16),
            )
        )

    def test_linear_layers_consume_native_checkpoint_orientation(self):
        weights = _weights()
        query = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        output = torch.arange(16, dtype=torch.float32).reshape(4, 4) + 20
        intermediate = torch.arange(32, dtype=torch.float32).reshape(8, 4) + 40
        ffn_output = torch.arange(32, dtype=torch.float32).reshape(4, 8) + 80
        weights["bert.encoder.layer.0.attention.self.query.weight"] = query
        weights["bert.encoder.layer.0.attention.output.dense.weight"] = output
        weights["bert.encoder.layer.0.intermediate.dense.weight"] = intermediate
        weights["bert.encoder.layer.0.output.dense.weight"] = ffn_output

        model = self._load(_config(), weights)
        layer = model.layers[0]
        self.assertTrue(
            torch.equal(layer.self_attn.qkv_proj.weight[:4], query.to(torch.float16))
        )
        self.assertTrue(
            torch.equal(layer.self_attn.o_proj.weight, output.to(torch.float16))
        )
        self.assertTrue(
            torch.equal(layer.mlp.intermediate.weight, intermediate.to(torch.float16))
        )
        self.assertTrue(
            torch.equal(layer.mlp.output.weight, ffn_output.to(torch.float16))
        )

    def test_missing_token_type_embedding_uses_device_local_zeros(self):
        model = self._load(_config(), _weights(include_token_type=False))
        token_type = model.runtime_weight_view()[W.token_type_embedding]
        self.assertEqual(token_type.device.type, "cpu")
        self.assertTrue(
            torch.allclose(token_type, torch.zeros(2, 4, dtype=token_type.dtype))
        )

    def test_missing_required_weight_fails_before_postprocess(self):
        weights = _weights()
        weights.pop("bert.encoder.layer.0.output.dense.weight")
        with self.assertRaisesRegex(RuntimeError, "missing checkpoint tensors.*weight"):
            self._load(_config(), weights)

    def test_duplicate_layernorm_alias_fails(self):
        weights = _weights()
        weights["bert.embeddings.LayerNorm.gamma"] = torch.ones(4)
        with self.assertRaisesRegex(RuntimeError, "Duplicate BERT checkpoint tensor"):
            self._load(_config(), weights)

    def test_custom_weights_use_exact_declared_contract(self):
        weights = _weights()
        weights.update(
            {
                "bert.pooler.dense.weight": torch.ones(4, 4),
                "bert.pooler.dense.bias": torch.zeros(4),
                "classifier.weight": torch.ones(2, 4),
                "classifier.bias": torch.zeros(2),
            }
        )
        mappings = tuple(
            (CUSTOM_WEIGHT_PREFIX + checkpoint_name, checkpoint_name)
            for checkpoint_name in (
                "bert.pooler.dense.weight",
                "bert.pooler.dense.bias",
                "classifier.weight",
                "classifier.bias",
            )
        )
        model = self._load(_config(), weights, custom_weight_mappings=mappings)
        self.assertTrue(
            torch.equal(
                model.runtime_weight_view()[mappings[2][0]],
                torch.ones(2, 4, dtype=torch.float16),
            )
        )

        weights.pop("bert.pooler.dense.bias")
        with self.assertRaisesRegex(RuntimeError, "missing required custom weights"):
            self._load(_config(), weights, custom_weight_mappings=mappings)

    def test_unknown_floating_tensor_is_not_silently_accepted(self):
        weights = _weights()
        weights["bert.encoder.layer.0.typo.weight"] = torch.ones(4, 4)
        with self.assertRaisesRegex(RuntimeError, "dropped unexpected"):
            self._load(_config(), weights)

    def test_wrong_model_prefix_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "dropped unexpected"):
            self._load(_config("roberta"), _weights(prefix="bert"))

    def test_attention_config_invariants_fail_fast(self):
        config = _config()
        config.attn_config.size_per_head = 3
        with self.assertRaisesRegex(ValueError, "hidden_size must equal"):
            BertForEmbedding(config, NewLoaderConfig(device="cpu"))

        config = _config()
        config.attn_config.kv_head_num = 1
        with self.assertRaisesRegex(ValueError, "kv_head_num == head_num"):
            BertForEmbedding(config, NewLoaderConfig(device="cpu"))

    def test_invalid_runtime_config_values_fail_fast(self):
        valid = _config()
        config = SimpleNamespace(
            hidden_size=valid.hidden_size,
            inter_size=valid.inter_size,
            num_layers=valid.num_layers,
            vocab_size=valid.vocab_size,
            max_seq_len=valid.max_seq_len,
            type_vocab_size=True,
            attn_config=valid.attn_config,
            quantization="",
            quant_config=None,
        )
        with self.assertRaisesRegex(TypeError, "type_vocab_size"):
            BertForEmbedding(config, NewLoaderConfig(device="cpu"))

        with self.assertRaisesRegex(TypeError, "custom_weight_mappings"):
            NewLoaderConfig(
                device="cpu",
                custom_weight_mappings=[("__custom__.a", "b")],
            )
        with self.assertRaisesRegex(ValueError, "Duplicate custom runtime"):
            NewLoaderConfig(
                device="cpu",
                custom_weight_mappings=(
                    ("__custom__.a", "b"),
                    ("__custom__.a", "c"),
                ),
            )

    def test_quantized_checkpoints_are_rejected_until_adapted(self):
        config = _config()
        config.quantization = "fp8"
        with self.assertRaisesRegex(
            NotImplementedError, "unquantized checkpoints only"
        ):
            BertForEmbedding(config, NewLoaderConfig(device="cpu"))

        config = _config()
        config.quant_config = object()
        with self.assertRaisesRegex(
            NotImplementedError, "unquantized checkpoints only"
        ):
            BertForEmbedding(config, NewLoaderConfig(device="cpu"))

    def test_tp2_loads_rank_local_attention_and_ffn_shards(self):
        weights = _weights()
        weights["bert.encoder.layer.0.attention.self.query.weight"] = torch.arange(
            16, dtype=torch.float32
        ).reshape(4, 4)
        for rank in (0, 1):
            model = self._load(
                _config(),
                weights,
                tp_size=2,
                tp_rank=rank,
            )
            qkv = model.layers[0].self_attn.qkv_proj.weight
            self.assertEqual(tuple(qkv.shape), (6, 4))
            expected_q = weights["bert.encoder.layer.0.attention.self.query.weight"][
                rank * 2 : (rank + 1) * 2
            ]
            self.assertTrue(torch.equal(qkv[:2], expected_q.to(torch.float16)))
            self.assertEqual(
                tuple(model.layers[0].mlp.intermediate.weight.shape), (4, 4)
            )
            self.assertEqual(tuple(model.layers[0].mlp.output.weight.shape), (4, 4))
            self.assertEqual(
                tuple(model.embeddings.word_embeddings.weight.shape), (7, 4)
            )

    def test_direct_model_tree_forward_matches_torch_reference(self):
        model = self._load(_config(), _weights(), compute_dtype=torch.float32)

        class QueryOnlyFmha:
            def forward(self, qkv, kv_cache, layer_idx):
                self.layer_idx = layer_idx
                return qkv[:, :4]

        input_ids = torch.tensor([1, 2], dtype=torch.long)
        position_ids = torch.tensor([0, 1], dtype=torch.long)
        token_type_ids = torch.tensor([0, 1], dtype=torch.long)
        with torch.inference_mode():
            hidden = model.embeddings(input_ids, position_ids, token_type_ids, 1.0)
            result = model.layers[0](hidden, QueryOnlyFmha(), None)

            layer = model.layers[0]
            qkv = torch.nn.functional.linear(
                hidden,
                layer.self_attn.qkv_proj.weight,
                layer.self_attn.qkv_proj.bias,
            )
            attention = torch.nn.functional.linear(
                qkv[:, :4],
                layer.self_attn.o_proj.weight,
                layer.self_attn.o_proj.bias,
            )
            after_attention = torch.nn.functional.layer_norm(
                attention + hidden,
                (4,),
                layer.attention_layernorm.weight,
                layer.attention_layernorm.bias,
                layer.attention_layernorm.eps,
            )
            intermediate = torch.nn.functional.gelu(
                torch.nn.functional.linear(
                    after_attention,
                    layer.mlp.intermediate.weight,
                    layer.mlp.intermediate.bias,
                )
            )
            output = torch.nn.functional.linear(
                intermediate,
                layer.mlp.output.weight,
                layer.mlp.output.bias,
            )
            expected = torch.nn.functional.layer_norm(
                output + after_attention,
                (4,),
                layer.output_layernorm.weight,
                layer.output_layernorm.bias,
                layer.output_layernorm.eps,
            )
        self.assertTrue(torch.allclose(result, expected))

    def test_ep_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "ep_size must be 1"):
            BertForEmbedding(_config(), NewLoaderConfig(device="cpu", ep_size=2))


if __name__ == "__main__":
    unittest.main()
