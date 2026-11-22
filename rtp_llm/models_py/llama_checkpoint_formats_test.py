"""Rank-local FT and consolidated .pth archives must not require HF config.json."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import QuantizationConfig as CheckpointQuantization
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards


def values(*shape):
    return torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)


def load(root, tp=1, rank=0, kv_heads=1, dtype=torch.float32, task_type=None):
    config = SimpleNamespace(
        ckpt_path=str(root), model_type="llama", num_layers=1, vocab_size=8,
        hidden_size=8, inter_size=8, layernorm_eps=1e-6, is_mtp=False,
        enable_fp32_lm_head=False, tie_word_embeddings=False,
        attn_config=SimpleNamespace(head_num=2, kv_head_num=kv_heads, size_per_head=4),
        capture_aux_hidden_layer_ids=[], hidden_state_capture_layer_ids=[],
    )
    if task_type is not None:
        config.task_type = task_type
    parallel = SimpleNamespace(
        tp_size=tp, tp_rank=rank, dp_size=1, dp_rank=0, ep_size=1, ep_rank=0,
        pp_size=1, pp_rank=0, world_size=tp, world_rank=rank,
        get_attn_tp_size=lambda: tp, get_attn_tp_rank=lambda: rank,
        get_ffn_tp_size=lambda: tp, get_ffn_tp_rank=lambda: rank,
    )
    quant = CheckpointQuantization.load_from_ckpt(str(root))
    options = NewLoaderConfig(
        tp_size=tp, tp_rank=rank, compute_dtype=dtype, device="cpu",
        parallelism_config=parallel, quant_config=QuantizationConfig(
            quant.get_method() if quant else "none", source_config=quant
        ),
        load_method="scratch",
    )
    return NewModelLoader(config, options, model_path=str(root)).load()


class LlamaCheckpointFormatsTest(unittest.TestCase):
    def test_explicitly_disabled_window_does_not_limit_full_attention(self):
        from rtp_llm.models_py.new_models.qwen2.llama import _check_window

        for family in ("llama", "yi"):
            hf = dict(model_type=family, sliding_window=4, use_sliding_window=False)
            _check_window(hf, 8)
            for change in (
                {"use_sliding_window": True},
                {"use_sliding_window": None},
                {"layer_types": ["full_attention", "sliding_attention"]},
                {"model_type": "mistral"},
            ):
                active = {**hf, **change}
                _check_window(active, 4)
                with self.assertRaisesRegex(ValueError, "sliding_window"):
                    _check_window(active, 8)

    def test_calibrated_decoder_does_not_quantize_routed_experts(self):
        from rtp_llm.models_py.quant_methods.unquantized import UnquantizedFusedMoEMethod

        cfg = QuantizationConfig("pertensor_quant")
        experts = SimpleNamespace(PROJ_NAMES=("gate_proj", "up_proj", "down_proj"))
        self.assertIsInstance(
            cfg.get_moe_quant_method(experts, "model.layers.0.mlp.experts"),
            UnquantizedFusedMoEMethod,
        )


    def test_real_llama_smooth_converter_projection_layout(self):
        from rtp_llm.utils.smooth_quant_convert.llama.convert import (
            split_and_save_weight,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text('{"model_type":"llama"}')
            (root / "smoothquant.ini").touch()
            tensors = {
                "model.embed_tokens.weight": values(8, 8),
                "lm_head.weight": values(8, 8),
                "model.norm.weight": values(8) + 1,
                "model.layers.0.input_layernorm.weight": values(8) + 1,
                "model.layers.0.post_layernorm.weight": values(8) + 1,
            }
            for proj in (
                "attention.query_key_value",
                "attention.dense",
                "mlp.fc",
                "mlp.gate",
                "mlp.proj",
            ):
                width = 24 if proj == "attention.query_key_value" else 8
                weight = values(8, width) / 128 - 0.5
                calibration = {
                    "x": torch.ones(8),
                    "w": weight.abs().amax(0),
                    "y": torch.ones(width),
                }
                opts = {"int8_outputs": "all", "multi_query_mode": False}
                tensors.update(
                    split_and_save_weight(
                        0,
                        root,
                        1,
                        f"layers.0.{proj}.weight",
                        weight.numpy(),
                        calibration,
                        opts,
                    )
                )
                if proj in ("attention.dense", "mlp.proj"):
                    tensors.update(
                        split_and_save_weight(
                            0,
                            root,
                            1,
                            f"layers.0.{proj}.smoother",
                            torch.ones(8).numpy(),
                            None,
                            opts,
                        )
                    )
            torch.save(tensors, root / "llama2.pt")
            model = load(root, kv_heads=2, dtype=torch.float16)
            attention = model.layers[0].self_attn.qkv_proj
            source = "model.layers.0.attention.query_key_value."
            torch.testing.assert_close(
                attention.weight, tensors[source + "weight.int8.col"].reshape(8, 24).T
            )
            torch.testing.assert_close(
                attention.weight_scale.flatten(),
                tensors[source + "scale_w_quant_orig.col"].flatten(),
            )

    def test_calibrated_decoder_int8_legacy_suffixes(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text('{"model_type":"llama"}')
            (root / "pertensorquant.ini").touch()
            layer = "model.layers.0."
            tensors = {
                "model.embed_tokens.weight": values(8, 8),
                "lm_head.weight": values(8, 8),
                "model.norm.weight": values(8) + 1,
                layer + "input_layernorm.weight": values(8) + 1,
                layer + "post_attention_layernorm.weight": values(8) + 1,
            }
            for i, (proj, width) in enumerate(
                (
                    ("self_attn.q_proj", 8),
                    ("self_attn.k_proj", 4),
                    ("self_attn.v_proj", 4),
                    ("self_attn.o_proj", 8),
                    ("mlp.gate_proj", 8),
                    ("mlp.up_proj", 8),
                    ("mlp.down_proj", 8),
                )
            ):
                base = layer + proj
                tensors[base + ".qweight"] = (values(8, width) % 11 - 5 + i).to(
                    torch.int8
                )
                tensors[base + ".scales"] = (
                    torch.tensor([0.25]) if width == 4 else (values(1, width) + 1) / 8
                )
            for proj in (
                "self_attn.qkv",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
            ):
                tensors[layer + proj + "_input_scale"] = torch.tensor([0.5])
            path = root / "model.safetensors"
            save_file(tensors, path)
            for rank in range(2):
                model = load(root, 2, rank)
                qkv = model.layers[0].self_attn.qkv_proj
                x = (values(3, 8) - 5) / 3
                outputs = []
                for proj in ("q_proj", "k_proj", "v_proj"):
                    base = layer + "self_attn." + proj
                    w, s = tensors[base + ".qweight"], tensors[base + ".scales"]
                    expected = ((x / 0.5).round().int() @ w.int()) * 0.5 * s
                    outputs.append(
                        expected.chunk(2, -1)[rank] if proj == "q_proj" else expected
                    )
                torch.testing.assert_close(
                    qkv(x), torch.cat(outputs, -1), atol=0, rtol=0
                )
                gate = tensors[layer + "mlp.gate_proj.qweight"].T.chunk(2)[rank]
                up = tensors[layer + "mlp.up_proj.qweight"].T.chunk(2)[rank]
                torch.testing.assert_close(
                    model.layers[0].mlp.gate_up_proj.weight, torch.cat([gate, up])
                )
                torch.testing.assert_close(
                    model.layers[0].self_attn.o_proj.weight,
                    tensors[layer + "self_attn.o_proj.qweight"].chunk(2)[rank].T,
                )
            del tensors[layer + "self_attn.qkv_input_scale"]
            save_file(tensors, path)
            with self.assertRaisesRegex(RuntimeError, "input_scale"):
                load(root)

    def test_meta_consolidated_pth_without_hf_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            params = {"dim": 8, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "norm_eps": 1e-6}
            (root / "params.json").write_text(json.dumps(params))
            tensors = {
                "tok_embeddings.weight": values(8, 8),
                "output.weight": values(8, 8) + 99,
                "norm.weight": values(8) + 1,
                "layers.0.attention_norm.weight": values(8) + 7,
                "layers.0.ffn_norm.weight": values(8) + 9,
            }
            for i, name in enumerate(("wq", "wk", "wv", "wo")):
                tensors[f"layers.0.attention.{name}.weight"] = values(
                    4 if name in ("wk", "wv") else 8, 8
                ) + 20 * i
            for i in (1, 2, 3):
                tensors[f"layers.0.feed_forward.w{i}.weight"] = values(8, 8) + i
            for rank in range(2):
                part = {}
                for name, tensor in tensors.items():
                    axis = 1 if name.endswith(("wo.weight", "w2.weight", "tok_embeddings.weight")) else 0
                    part[name] = tensor if tensor.ndim == 1 else tensor.chunk(2, axis)[rank].clone()
                torch.save(part, root / f"consolidated.{rank:02d}.pth")
            model = load(root)
            attn = model.layers[0].self_attn
            q = tensors["layers.0.attention.wq.weight"][[0, 2, 1, 3, 4, 6, 5, 7]]
            k = tensors["layers.0.attention.wk.weight"][[0, 2, 1, 3]]
            v = tensors["layers.0.attention.wv.weight"]
            torch.testing.assert_close(attn.qkv_proj.weight, torch.cat((q, k, v)))
            torch.testing.assert_close(model.embed_tokens.weight, tensors["tok_embeddings.weight"])
            torch.testing.assert_close(model.lm_head.weight, tensors["output.weight"])
            from rtp_llm.ops import TaskType

            backbone = load(root, task_type=TaskType.DENSE_EMBEDDING)
            self.assertIsNone(backbone.lm_head)
            torch.testing.assert_close(backbone.norm.weight, tensors["norm.weight"])
            torch.testing.assert_close(backbone.layers[0].self_attn.qkv_proj.weight, attn.qkv_proj.weight)
            for rank in range(2):
                path = root / f"consolidated.{rank:02d}.pth"
                part = torch.load(path, weights_only=True)
                del part["output.weight"]
                torch.save(part, path)
            self.assertIsNone(load(root, task_type=TaskType.DENSE_EMBEDDING).lm_head)
            with self.assertRaisesRegex(ValueError, "Missing FT sources for lm_head.weight"):
                load(root)
            for rank in range(2):
                path = root / f"consolidated.{rank:02d}.pth"
                part = torch.load(path, weights_only=True)
                part["unexpected.weight"] = torch.ones(1)
                torch.save(part, path)
            with self.assertRaisesRegex(ValueError, "Unbound FT tensors.*unexpected.weight"):
                load(root, task_type=TaskType.DENSE_EMBEDDING)

    def test_rank_local_ft_without_hf_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = values(8, 8)
            for rank in range(2):
                records = [
                    (None, "embedding", base.chunk(2, 1)[rank]),
                    (None, "lm_head", (base + 91).chunk(2, 0)[rank]),
                    (None, "final_layernorm.gamma", values(8)),
                    (0, "pre_layernorm_weights.gamma", values(8) + 1),
                    (0, "post_layernorm_weights.gamma", values(8) + 2),
                    (0, "self_attention_weights.query_weight.kernel", values(8, 12) + 100 * rank),
                    (0, "self_attention_weights.attention_output_weight.kernel", values(4, 8)),
                    (0, "ffn_weights.intermediate_weight13.kernel", values(8, 8)),
                    (0, "ffn_weights.intermediate_weight2.kernel", values(4, 8)),
                ]
                write_ft_shards(records, str(root), tp_rank=rank, ep_rank=0)
            generate_ft_index(str(root), {"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": 1})
            for rank in range(2):
                model = load(root, 2, rank)
                self.assertIsNone(model.layers[0].self_attn.qkv_proj.bias)
                torch.testing.assert_close(
                    model.layers[0].self_attn.qkv_proj.weight,
                    (values(8, 12) + 100 * rank).T,
                )
                torch.testing.assert_close(model.embed_tokens.weight, base.chunk(2)[rank])
                torch.testing.assert_close(model.lm_head.weight, (base + 91).chunk(2)[rank])


if __name__ == "__main__":
    unittest.main()
