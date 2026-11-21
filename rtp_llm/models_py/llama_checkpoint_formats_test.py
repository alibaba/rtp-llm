"""Rank-local FT and consolidated .pth archives must not require HF config.json."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards


def values(*shape):
    return torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)


def load(root, tp=1, rank=0):
    config = SimpleNamespace(
        ckpt_path=str(root), model_type="llama", num_layers=1, vocab_size=8,
        hidden_size=8, inter_size=8, layernorm_eps=1e-6, is_mtp=False,
        enable_fp32_lm_head=False, tie_word_embeddings=False,
        attn_config=SimpleNamespace(head_num=2, kv_head_num=1, size_per_head=4),
        capture_aux_hidden_layer_ids=[], hidden_state_capture_layer_ids=[],
    )
    parallel = SimpleNamespace(
        tp_size=tp, tp_rank=rank, dp_size=1, dp_rank=0, ep_size=1, ep_rank=0,
        pp_size=1, pp_rank=0, world_size=tp, world_rank=rank,
        get_attn_tp_size=lambda: tp, get_attn_tp_rank=lambda: rank,
        get_ffn_tp_size=lambda: tp, get_ffn_tp_rank=lambda: rank,
    )
    options = NewLoaderConfig(
        tp_size=tp, tp_rank=rank, compute_dtype=torch.float32, device="cpu",
        parallelism_config=parallel, quant_config=QuantizationConfig("none"),
        load_method="scratch",
    )
    return NewModelLoader(config, options, model_path=str(root)).load()


class LlamaCheckpointFormatsTest(unittest.TestCase):
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
