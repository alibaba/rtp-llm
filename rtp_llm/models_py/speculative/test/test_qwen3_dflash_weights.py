"""CPU checkpoint lookup/merge/TP contracts using synthetic DFlash weights."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models.qwen_3_dflash import Qwen3DFlash
from rtp_llm.models_py.speculative.test.draft_checkpoint_case import load_draft

# Preserve the three checkpoint families' layer/feature counts and H/Q ratio;
# reduce matrix dimensions only. No trained tensor or network access is needed.
_FAMILIES = (
    ("qwen36_27b", 80, 5, [1, 16, 31, 46, 61], 2048, 248070),
    ("qwen35_27b", 80, 6, [1, 10, 18, 27, 35, 44, 52, 61], 4096, 248077),
    ("qwen35_397b", 64, 6, [1, 9, 17, 25, 33, 41, 49, 57], 4096, 248077),
)


def _checkpoint(hidden, layers, feature_count):
    tensors = {}

    def add(name, shape):
        offset = len(tensors) * 17
        tensor = (torch.arange(math.prod(shape)) + offset).remainder(251).float() / 8
        tensors[name] = tensor.reshape(shape).to(torch.bfloat16)

    add("fc.weight", (hidden, feature_count * hidden))
    add("hidden_norm.weight", (hidden,))
    add("norm.weight", (hidden,))
    for index in range(layers):
        prefix = f"layers.{index}."
        add(prefix + "input_layernorm.weight", (hidden,))
        add(prefix + "post_attention_layernorm.weight", (hidden,))
        add(prefix + "self_attn.q_norm.weight", (2,))
        add(prefix + "self_attn.k_norm.weight", (2,))
        add(prefix + "self_attn.q_proj.weight", (64, hidden))
        add(prefix + "self_attn.k_proj.weight", (16, hidden))
        add(prefix + "self_attn.v_proj.weight", (16, hidden))
        add(prefix + "self_attn.o_proj.weight", (hidden, 64))
        add(prefix + "mlp.gate_proj.weight", (128, hidden))
        add(prefix + "mlp.up_proj.weight", (128, hidden))
        add(prefix + "mlp.down_proj.weight", (hidden, 128))
    return tensors


def _raw_config(hidden, layers, ids, window, mask):
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": hidden,
        "intermediate_size": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 2,
        "num_hidden_layers": layers,
        "vocab_size": 248320,
        "dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10_000_000, "rope_type": "default"},
        "layer_types": ["sliding_attention"] * (layers - 1) + ["full_attention"],
        "sliding_window": window,
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": mask,
            "target_layer_ids": ids,
        },
    }


class Qwen3DFlashWeightTest(unittest.TestCase):
    def test_real_checkpoint_mapper_and_all_tp_ranks(self):
        for name, hidden, layers, ids, window, mask in _FAMILIES:
            checkpoint = _checkpoint(hidden, layers, len(ids))
            with tempfile.TemporaryDirectory() as path:
                Path(path, "config.json").write_text(
                    json.dumps(_raw_config(hidden, layers, ids, window, mask))
                )
                save_file(checkpoint, str(Path(path, "model.safetensors")))
                config = Qwen3DFlash._create_config(path)
                config.ckpt_path = path
                config.phy2log_path = ""
                config.data_type = "bf16"
                config.model_type = "qwen_3_dflash"
                config.gen_num_per_cycle = 3
                config.enable_fp32_lm_head = False
                for tp in (1, 2, 8):
                    for rank in range(tp):
                        with self.subTest(family=name, tp=tp, rank=rank):
                            self._check_rank(config, checkpoint, tp, rank)

    def _check_rank(self, config, checkpoint, tp, rank):
        model, target, requested_names = load_draft(
            config, tp, rank, ("embed_tokens", "lm_head")
        )
        self.assertEqual(len(model.layers), config.num_layers)
        self.assertIs(model.embed_tokens.weight, target.embed_tokens.weight)
        self.assertIs(model.lm_head.weight, target.lm_head.weight)
        # FC and its output norm remain replicated, including TP8. A split
        # here would silently corrupt the target-feature combiner.
        torch.testing.assert_close(
            model.fc.weight, checkpoint["fc.weight"], atol=0, rtol=0
        )
        for runtime_name, checkpoint_name in (
            (model.hidden_norm.weight, "hidden_norm.weight"),
            (model.norm.weight, "norm.weight"),
        ):
            torch.testing.assert_close(
                runtime_name, checkpoint[checkpoint_name], atol=0, rtol=0
            )
        for index, loaded in enumerate(model.layers):
            prefix = f"layers.{index}.self_attn."
            expected_qkv = torch.cat(
                [
                    checkpoint[prefix + part + "_proj.weight"].chunk(tp, dim=0)[rank].T
                    for part in ("q", "k", "v")
                ],
                dim=1,
            )
            self.assertEqual(
                tuple(loaded.self_attn.qkv_proj.weight.T.shape),
                (config.hidden_size, 96 // tp),
            )
            torch.testing.assert_close(
                loaded.self_attn.qkv_proj.weight.T, expected_qkv, atol=0, rtol=0
            )
            torch.testing.assert_close(
                loaded.self_attn.o_proj.weight.T,
                checkpoint[prefix + "o_proj.weight"].T.chunk(tp, dim=0)[rank],
                atol=0,
                rtol=0,
            )
            for runtime_name, suffix in (
                (loaded.self_attn.q_norm.weight, "q_norm.weight"),
                (loaded.self_attn.k_norm.weight, "k_norm.weight"),
            ):
                torch.testing.assert_close(
                    runtime_name,
                    checkpoint[prefix + suffix],
                    atol=0,
                    rtol=0,
                )
        self.assertEqual(requested_names, set(checkpoint))
        self.assertFalse(any(name.startswith("model.") for name in requested_names))
        self.assertFalse(
            any("embed_tokens" in name or "lm_head" in name for name in requested_names)
        )


if __name__ == "__main__":
    unittest.main()
