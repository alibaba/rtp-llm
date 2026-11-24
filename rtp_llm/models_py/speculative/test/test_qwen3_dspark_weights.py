"""CPU TorchSpec Qwen3 DSpark checkpoint loading and TP contracts."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models.qwen_3_dspark import Qwen3DSpark
from rtp_llm.models_py.speculative.test.draft_checkpoint_case import load_draft

_HIDDEN = 64
_VOCAB = 64
_LAYERS = 5
_TARGET_LAYERS = [39, 47, 55]
_HEADS = 32
_KV_HEADS = 8
_HEAD_DIM = 2
_INTERMEDIATE = 192
_MARKOV_RANK = 4
_CONFIDENCE_KEYS = {"confidence_head.proj.weight", "confidence_head.proj.bias"}


def _raw_config(*, lm_head_source="target"):
    config = {
        "architectures": ["Qwen3DSparkForCausalLM"],
        "model_type": "qwen_3_dspark",
        "hidden_size": _HIDDEN,
        "intermediate_size": _INTERMEDIATE,
        "num_attention_heads": _HEADS,
        "num_key_value_heads": _KV_HEADS,
        "head_dim": _HEAD_DIM,
        "num_hidden_layers": _LAYERS,
        "vocab_size": _VOCAB,
        "dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"},
        "layer_types": ["full_attention"] * _LAYERS,
        "block_size": 5,
        "mask_token_id": _VOCAB - 1,
        "aux_hidden_state_layer_ids": _TARGET_LAYERS,
        "markov_rank": _MARKOV_RANK,
        "sample_from_anchor": True,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    if lm_head_source is not None:
        config["lm_head_source"] = lm_head_source
    return config


def _checkpoint(*, include_lm_head=False):
    tensors = {}

    def add(name, shape):
        offset = len(tensors) * 17
        values = (torch.arange(math.prod(shape)) + offset).remainder(251).float() / 8
        tensors[name] = values.reshape(shape).to(torch.bfloat16)

    add("fc.weight", (_HIDDEN, len(_TARGET_LAYERS) * _HIDDEN))
    add("hidden_norm.weight", (_HIDDEN,))
    add("model.embed_tokens.weight", (_VOCAB, _HIDDEN))
    add("model.norm.weight", (_HIDDEN,))
    for layer in range(_LAYERS):
        prefix = f"model.layers.{layer}."
        add(prefix + "input_layernorm.weight", (_HIDDEN,))
        add(prefix + "post_attention_layernorm.weight", (_HIDDEN,))
        add(prefix + "self_attn.q_norm.weight", (_HEAD_DIM,))
        add(prefix + "self_attn.k_norm.weight", (_HEAD_DIM,))
        add(prefix + "self_attn.q_proj.weight", (_HEADS * _HEAD_DIM, _HIDDEN))
        add(prefix + "self_attn.k_proj.weight", (_KV_HEADS * _HEAD_DIM, _HIDDEN))
        add(prefix + "self_attn.v_proj.weight", (_KV_HEADS * _HEAD_DIM, _HIDDEN))
        add(prefix + "self_attn.o_proj.weight", (_HIDDEN, _HEADS * _HEAD_DIM))
        add(prefix + "mlp.gate_proj.weight", (_INTERMEDIATE, _HIDDEN))
        add(prefix + "mlp.up_proj.weight", (_INTERMEDIATE, _HIDDEN))
        add(prefix + "mlp.down_proj.weight", (_HIDDEN, _INTERMEDIATE))
    add("markov_head.markov_w1.weight", (_VOCAB, _MARKOV_RANK))
    add("markov_head.markov_w2.weight", (_VOCAB, _MARKOV_RANK))
    # Converter preserves these training tensors although the current runtime
    # deliberately has no descriptor for either confidence-head parameter.
    add("confidence_head.proj.weight", (1, _HIDDEN + _MARKOV_RANK))
    add("confidence_head.proj.bias", (1,))
    if include_lm_head:
        add("lm_head.weight", (_VOCAB, _HIDDEN))
    return tensors


class Qwen3DSparkWeightTest(unittest.TestCase):
    def _load(self, path, tp, rank):
        config = Qwen3DSpark._create_config(path)
        config.ckpt_path = path
        config.model_type = "qwen_3_dspark"
        config.data_type = "bf16"
        config.enable_fp32_lm_head = False
        config.gen_num_per_cycle = 3
        roles = ("lm_head",) if config.dspark_share_target_lm_head else ()
        model, target, read = load_draft(config, tp, rank, roles)
        return config, model, target, read

    def test_real_torchspec_converted_checkpoint_tp_contract(self):
        checkpoint = _checkpoint()
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            root.joinpath("config.json").write_text(json.dumps(_raw_config()))
            save_file(checkpoint, str(root / "model.safetensors"))
            for tp in (1, 2, 8):
                for rank in range(tp):
                    with self.subTest(tp=tp, rank=rank):
                        config, model, target, requested = self._load(path, tp, rank)
                        target_head = target.lm_head.weight
                        self.assertTrue(config.dspark_share_target_lm_head)
                        self.assertTrue(config.dspark_sample_from_anchor)
                        self.assertEqual(config.dspark_target_layer_ids, _TARGET_LAYERS)
                        self.assertIs(model.lm_head.weight, target_head)
                        self.assertEqual(
                            model.lm_head.weight.data_ptr(), target_head.data_ptr()
                        )
                        torch.testing.assert_close(
                            model.embed_tokens.weight,
                            checkpoint["model.embed_tokens.weight"].chunk(tp, dim=0)[
                                rank
                            ],
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            model.norm.weight,
                            checkpoint["model.norm.weight"],
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            model.fc.weight,
                            checkpoint["fc.weight"],
                            atol=0,
                            rtol=0,
                        )
                        for runtime_name, source_name in (
                            (model.hidden_norm.weight, "hidden_norm.weight"),
                            (
                                model.markov_head.markov_w1.weight,
                                "markov_head.markov_w1.weight",
                            ),
                            (
                                model.markov_head.markov_w2.weight,
                                "markov_head.markov_w2.weight",
                            ),
                        ):
                            rows = checkpoint[source_name].shape[0]
                            torch.testing.assert_close(
                                runtime_name[:rows],
                                checkpoint[source_name],
                                atol=0,
                                rtol=0,
                            )
                            self.assertEqual(
                                torch.count_nonzero(runtime_name[rows:]), 0
                            )
                        self.assertEqual(len(model.layers), _LAYERS)
                        layer = model.layers[0]
                        prefix = "model.layers.0."
                        expected_qkv = torch.cat(
                            [
                                checkpoint[prefix + f"self_attn.{part}_proj.weight"]
                                .chunk(tp, dim=0)[rank]
                                .T
                                for part in ("q", "k", "v")
                            ],
                            dim=1,
                        )
                        torch.testing.assert_close(
                            layer.self_attn.qkv_proj.weight.T,
                            expected_qkv,
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            layer.self_attn.o_proj.weight.T,
                            checkpoint[prefix + "self_attn.o_proj.weight"].T.chunk(
                                tp, dim=0
                            )[rank],
                            atol=0,
                            rtol=0,
                        )
                        expected_w13 = torch.cat(
                            [
                                checkpoint[prefix + "mlp.gate_proj.weight"].T.chunk(
                                    tp, dim=1
                                )[rank],
                                checkpoint[prefix + "mlp.up_proj.weight"].T.chunk(
                                    tp, dim=1
                                )[rank],
                            ],
                            dim=1,
                        )
                        torch.testing.assert_close(
                            layer.mlp.gate_up_proj.weight.T,
                            expected_w13,
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            layer.mlp.down_proj.weight.T,
                            checkpoint[prefix + "mlp.down_proj.weight"].T.chunk(
                                tp, dim=0
                            )[rank],
                            atol=0,
                            rtol=0,
                        )
                        self.assertEqual(requested, set(checkpoint) - _CONFIDENCE_KEYS)
                        self.assertNotIn("lm_head.weight", requested)

    def test_checkpoint_lm_head_remains_owned_when_alias_is_not_declared(self):
        checkpoint = _checkpoint(include_lm_head=True)
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            root.joinpath("config.json").write_text(
                json.dumps(_raw_config(lm_head_source="checkpoint"))
            )
            save_file(checkpoint, str(root / "model.safetensors"))
            config, model, _, requested = self._load(path, 2, 1)
        self.assertFalse(config.dspark_share_target_lm_head)
        torch.testing.assert_close(
            model.lm_head.weight,
            checkpoint["lm_head.weight"]
            .chunk(2, dim=0)[1]
            .to(model.lm_head.weight.dtype),
            atol=0,
            rtol=0,
        )
        self.assertIn("lm_head.weight", requested)

    def test_missing_lm_head_fails_when_alias_is_not_declared(self):
        checkpoint = _checkpoint()
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            root.joinpath("config.json").write_text(
                json.dumps(_raw_config(lm_head_source="checkpoint"))
            )
            save_file(checkpoint, str(root / "model.safetensors"))
            with self.assertRaisesRegex((RuntimeError, ValueError), "lm_head"):
                self._load(path, 1, 0)


if __name__ == "__main__":
    unittest.main()
