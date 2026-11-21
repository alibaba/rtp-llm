"""Real checkpoint/whole-decoder CPU tests; FMHA is an explicit torch reference.

These tests do not claim production GPU attention, model smoke, or CI execution.
No decoder layers are removed and no native imports are replaced.
"""

import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewLoaderLoadMethod, NewModelLoader
from rtp_llm.models_py.new_models.cosyvoice_qwen.language import (
    CosyVoiceQwenForCausalLM,
)
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.ops import FMHAConfig, HWKernelConfig, MoeConfig, ParallelismConfig


def load_config(method="scratch"):
    parallel = types.SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        pp_size=1,
        pp_rank=0,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False, is_prefill_enabled=lambda: False
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        get_ffn_tp_rank=lambda: 0,
    )
    return NewLoaderConfig(
        device="cpu",
        load_method=method,
        compute_dtype=torch.float32,
        quant_config=QuantizationConfig("none"),
        parallelism_config=parallel,
    )


def text_config():
    return dict(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=12,
        vocab_size=8,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        enable_fp32_lm_head=False,
    )


def language_weights(input_vocab=8, output_vocab=8):
    generator = torch.Generator().manual_seed(53)

    def normal(*shape):
        return torch.randn(shape, generator=generator) * 0.15

    root = "model.layers.0."
    return {
        "model.embed_tokens.weight": normal(input_vocab, 8),
        "lm_head.weight": normal(output_vocab, 8),
        "model.norm.weight": torch.ones(8),
        root + "input_layernorm.weight": torch.ones(8),
        root + "post_attention_layernorm.weight": torch.ones(8),
        root + "self_attn.q_proj.weight": normal(8, 8),
        root + "self_attn.k_proj.weight": normal(4, 8),
        root + "self_attn.v_proj.weight": normal(4, 8),
        root + "self_attn.q_proj.bias": normal(8),
        root + "self_attn.k_proj.bias": normal(4),
        root + "self_attn.v_proj.bias": normal(4),
        root + "self_attn.o_proj.weight": normal(8, 8),
        root + "mlp.gate_proj.weight": normal(12, 8),
        root + "mlp.up_proj.weight": normal(12, 8),
        root + "mlp.down_proj.weight": normal(8, 12),
    }


def checkpoint(directory, config, weights):
    vocab = {"<unk>": 0, **{f"tok{i}": i for i in range(1, config["vocab_size"])}}
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
    ).save_pretrained(directory)
    with open(os.path.join(directory, "config.json"), "w") as stream:
        json.dump(config, stream)
    names, index = list(weights), {}
    for shard_id in range(2):
        filename = f"model-{shard_id + 1:05d}-of-00002.safetensors"
        shard = {name: weights[name] for name in names[shard_id::2]}
        save_file(shard, os.path.join(directory, filename))
        index.update({name: filename for name in shard})
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as stream:
        json.dump({"weight_map": index}, stream)


class TorchCausalGQA:
    """Explicit CPU oracle implementing RoPE, grouped heads, and causal SDPA."""

    def forward(self, qkv, kv_cache, layer_idx):
        q, k, v = qkv.split((8, 4, 4), dim=-1)
        length = q.shape[0]
        q, k, v = (
            q.reshape(length, 2, 4),
            k.reshape(length, 1, 4),
            v.reshape(length, 1, 4),
        )
        phase = torch.outer(
            torch.arange(length, dtype=q.dtype), torch.tensor([1.0, 0.01])
        )
        phase = torch.cat((phase, phase), dim=-1).unsqueeze(1)

        def rope(x):
            first, second = x.chunk(2, dim=-1)
            return x * phase.cos() + torch.cat((-second, first), dim=-1) * phase.sin()

        q, k = rope(q), rope(k)
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
            v.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        return output.squeeze(0).transpose(0, 1).reshape(length, 8)


def reference_forward(weights, hidden):
    def rms(x, key):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weights[key]

    root = "model.layers.0."
    normed = rms(hidden, root + "input_layernorm.weight")
    qkv = torch.cat(
        [
            F.linear(
                normed,
                weights[root + "self_attn." + p + ".weight"],
                weights[root + "self_attn." + p + ".bias"],
            )
            for p in ("q_proj", "k_proj", "v_proj")
        ],
        dim=-1,
    )
    hidden = hidden + F.linear(
        TorchCausalGQA().forward(qkv, None, 0),
        weights[root + "self_attn.o_proj.weight"],
    )
    normed = rms(hidden, root + "post_attention_layernorm.weight")
    mlp = F.silu(F.linear(normed, weights[root + "mlp.gate_proj.weight"])) * F.linear(
        normed, weights[root + "mlp.up_proj.weight"]
    )
    hidden = hidden + F.linear(mlp, weights[root + "mlp.down_proj.weight"])
    return rms(hidden, "model.norm.weight")


def assert_language_forward(test, model, weights, input_ids, features=None):
    mask = torch.ones(len(input_ids), dtype=torch.bool)
    if features is not None:
        mask[1 : 1 + len(features)] = False
    safe_ids = torch.where(mask, input_ids, 0)
    expected_input = F.embedding(safe_ids.long(), weights["model.embed_tokens.weight"])
    if features is not None:
        expected_input[1 : 1 + len(features)] = features
    inputs = types.SimpleNamespace(
        input_ids=input_ids,
        capture_hidden_states=False,
        embedding_inputs=(
            None if features is None else types.SimpleNamespace(text_tokens_mask=mask)
        ),
        multimodal_inputs=(
            None
            if features is None
            else types.SimpleNamespace(
                multimodal_features=[features],
                mm_features_locs=torch.tensor([1]),
                mm_extra_input=[],
            )
        ),
    )
    with torch.inference_mode():
        actual = model(inputs, fmha_impl=TorchCausalGQA()).hidden_states
        expected = reference_forward(weights, expected_input)
        logits = model.lm_head(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        logits, F.linear(expected, weights["lm_head.weight"]), rtol=1e-5, atol=1e-6
    )
    test.assertEqual(
        tuple(logits.shape), (len(input_ids), len(weights["lm_head.weight"]))
    )


class MultimodalLanguageLoadTest(unittest.TestCase):
    def test_registered_language_class(self):
        self.assertEqual(
            ModelDict.get_ft_model_type_by_hf_architectures("CosyQwen2ForCausalLM"),
            "cosyvoice_qwen",
        )
        self.assertIs(get_model_class("cosyvoice_qwen"), CosyVoiceQwenForCausalLM)

    def test_default_input_vocab_matches_old_config(self):
        model = CosyVoiceQwenForCausalLM(text_config(), load_config())
        self.assertEqual(model.embed_tokens.weight.shape[0], 151946)
        self.assertEqual(model.lm_head.weight.shape[0], 8)

    def test_sharded_checkpoint_loads_without_decoder_runtime(self):
        raw = dict(text_config(), model_type="cosyvoice_qwen", input_vocab_size=15)
        weights = language_weights(15, 8)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint(directory, raw, weights)
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    loader = NewModelLoader(raw, load_config(method), model_path=directory)
                    model = loader.load()
                    self.assertEqual(loader.loaded_method.value, method)
                    torch.testing.assert_close(model.embed_tokens.weight, weights["model.embed_tokens.weight"])
                    torch.testing.assert_close(model.lm_head.weight, weights["lm_head.weight"])


class MultimodalLanguageNativeBoundaryTest(unittest.TestCase):
    def check_checkpoint(self, method, tied, explicit_head):
        from rtp_llm.model_factory import ModelFactory

        raw = dict(text_config(), model_type="cosyvoice_qwen", input_vocab_size=15)
        raw["tie_word_embeddings"] = tied
        weights = language_weights(15, 8)
        if not explicit_head:
            del weights["lm_head.weight"]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint(directory, raw, weights)
            service_cls = ModelFactory.get_model_cls("cosyvoice_qwen")
            config = service_cls.create_config(directory)
            config.model_type = "cosyvoice_qwen"
            config.tokenizer_path = directory
            config.enable_fp32_lm_head = False
            config.data_type, config.lora_infos = "fp32", {}
            self.assertEqual(config.input_vocab_size, 15)
            self.assertEqual(config.mm_model_config.mm_sep_tokens, [[-200]])
            service = service_cls(
                model_config=config,
                parallelism_config=ParallelismConfig(),
                hw_kernel_config=HWKernelConfig(),
                kv_cache_config=KVCacheConfig(),
                fmha_config=FMHAConfig(),
                moe_config=MoeConfig(),
                max_generate_batch_size=1,
                load_method=NewLoaderLoadMethod(method),
                vit_config=None,
                merge_lora=False,
            )
            # Device selection is the only substitution: execute BaseModel.load
            # and normal publication, not a test stand-in for the service.
            with patch.object(service, "_get_device_str", return_value="cpu"):
                service.load()
            model = service.py_model
            self.assertIsInstance(model, CosyVoiceQwenForCausalLM)
            self.assertEqual(service.model_weights_loader.loaded_method.value, method)
            self.assertIs(
                service.weight.global_weights["embedding"], model.embed_tokens.weight
            )
            self.assertIs(
                service.weight.global_weights["lm_head"], model.lm_head.weight
            )
        self.assertEqual(tuple(model.embed_tokens.weight.shape), (15, 8))
        expected_head = weights.get(
            "lm_head.weight", weights["model.embed_tokens.weight"][:8]
        )
        torch.testing.assert_close(model.lm_head.weight, expected_head, rtol=0, atol=0)
        expected = {**weights, "lm_head.weight": expected_head}
        assert_language_forward(self, model, expected, torch.tensor([14, 2, 3, 10]))
        assert_language_forward(
            self, model, expected, torch.tensor([14, -200, -200, 10]), torch.randn(2, 8)
        )

    def test_scratch_and_native_fst(self):
        for method in ("scratch", "fastsafetensors"):
            for tied, explicit in ((False, True), (True, False), (True, True)):
                with self.subTest(method=method, tied=tied, explicit=explicit):
                    self.check_checkpoint(method, tied, explicit)


if __name__ == "__main__":
    unittest.main()
