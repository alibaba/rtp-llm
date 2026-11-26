"""Raw Meta archives must finish GQA/rotary conversion before online FP8."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.llama_checkpoint_formats_test import load, values, write_meta_source, write_q_adapter
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards


class LlamaMetaOnlineFp8Test(unittest.TestCase):
    def test_hf_adapter_is_merged_before_online_quantization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({
                "model_type": "llama", "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                "hidden_size": 8, "intermediate_size": 8, "vocab_size": 8,
            }))
            weights = {
                "model.embed_tokens.weight": values(8, 8),
                "lm_head.weight": values(8, 8),
                "model.norm.weight": torch.ones(8),
            }
            layer = "model.layers.0."
            for norm in ("input_layernorm", "post_attention_layernorm"):
                weights[layer + norm + ".weight"] = torch.ones(8)
            for name, rows in (
                ("self_attn.q_proj", 8), ("self_attn.k_proj", 4),
                ("self_attn.v_proj", 4), ("self_attn.o_proj", 8),
                ("mlp.gate_proj", 8), ("mlp.up_proj", 8), ("mlp.down_proj", 8),
            ):
                weights[layer + name + ".weight"] = values(rows, 8)
            save_file(weights, str(root / "model.safetensors"))
            adapter, delta = write_q_adapter(root, gain=512)
            logical = torch.cat(((values(8, 8) + delta), values(4, 8), values(4, 8))).bfloat16().float()
            step = logical.abs().amax() / 448
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    model = load(
                        root, dtype=torch.bfloat16, device="cuda:0", online_quant="fp8_online",
                        startup_lora=adapter, load_method=method,
                    )
                    proj = model.layers[0].self_attn.qkv_proj
                    torch.testing.assert_close(proj.weight_scale.cpu().reshape(()), step)
                    torch.testing.assert_close(
                        proj.weight.cpu().view(torch.uint8),
                        (logical / step).to(torch.float8_e4m3fn).view(torch.uint8),
                    )

    def test_meta_startup_lora_precedes_both_fp8_quantizers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = write_meta_source(root)
            adapter, delta = write_q_adapter(root, gain=512)
            q = src["layers.0.attention.wq.weight"][[4, 6, 5, 7]].bfloat16() + delta[4:].bfloat16()
            kv = [src["layers.0.attention.wk.weight"][[0, 2, 1, 3]], src["layers.0.attention.wv.weight"]]
            logical = torch.cat([q, *kv]).bfloat16().float()
            step = logical.abs().max() / 448
            for host in (True, False):
                with self.subTest(force_cpu=host):
                    model = load(root, tp=2, rank=1, dtype=torch.bfloat16, device="cuda:0",
                                 online_quant="fp8_online", force_cpu=host, startup_lora=adapter)
                    proj = model.layers[0].self_attn.qkv_proj
                    torch.testing.assert_close(proj.weight_scale.cpu().reshape(()), step)
                    torch.testing.assert_close(proj.weight.cpu().view(torch.uint8),
                                               (logical / step).to(torch.float8_e4m3fn).view(torch.uint8))

    def test_startup_lora_reencodes_per_tensor_ft_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                (None, "embedding", values(8, 8).bfloat16()), (None, "lm_head", values(8, 8).bfloat16()),
                (None, "final_layernorm.gamma", torch.ones(8, dtype=torch.bfloat16)),
                (0, "pre_layernorm_weights.gamma", torch.ones(8, dtype=torch.bfloat16)),
                (0, "post_layernorm_weights.gamma", torch.ones(8, dtype=torch.bfloat16)),
            ]
            for stem, rows in (("self_attention_weights.query_weight", 16),
                               ("self_attention_weights.attention_output_weight", 8),
                               ("ffn_weights.intermediate_weight13", 16),
                               ("ffn_weights.intermediate_weight2", 8)):
                codes = (values(rows, 8) % 13 - 6).to(torch.float8_e4m3fn)
                records.extend([(0, stem + ".kernel", codes),
                                (0, stem + ".weight_only_quant_scale", torch.tensor([0.25]))])
            write_ft_shards(records, str(root), tp_rank=0, ep_rank=0)
            generate_ft_index(str(root), {"TP_SIZE": 1, "DP_SIZE": 1, "EP_SIZE": 1})
            adapter, delta = write_q_adapter(root, gain=512)
            model = load(root, dtype=torch.bfloat16, device="cuda:0", online_quant="fp8", startup_lora=adapter)
            qkv = model.layers[0].self_attn.qkv_proj
            weight = (values(16, 8) % 13 - 6) * 0.25
            weight[:8] += delta
            step = weight.abs().max() / 448
            torch.testing.assert_close(qkv.weight_scale.cpu().reshape(()), step)
            torch.testing.assert_close(qkv.weight.cpu().view(torch.uint8),
                                       (weight / step).to(torch.float8_e4m3fn).view(torch.uint8))
            torch.testing.assert_close(qkv.input_scale.cpu(), torch.ones(1))

    def test_real_meta_fp8_after_tp_rotary(self):
        self.assertEqual(torch.cuda.get_device_capability(), (9, 0))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_meta_source(root)
            q = source["layers.0.attention.wq.weight"][[4, 6, 5, 7]]
            k = source["layers.0.attention.wk.weight"][[0, 2, 1, 3]]
            v = source["layers.0.attention.wv.weight"]
            logical = torch.cat((q, k, v)).bfloat16().float()
            expected_scale = logical.abs().amax() / 448
            codes = (logical / expected_scale).to(torch.float8_e4m3fn)
            for force_cpu in (False, True):
                with self.subTest(force_cpu=force_cpu):
                    model = load(
                        root,
                        tp=2,
                        rank=1,
                        dtype=torch.bfloat16,
                        device="cuda:0",
                        online_quant="fp8_online",
                        force_cpu=force_cpu,
                    )
                    leaf = model.layers[0].self_attn.qkv_proj
                    torch.testing.assert_close(
                        leaf.weight_scale.cpu().reshape(()), expected_scale
                    )
                    torch.testing.assert_close(
                        leaf.weight.cpu().view(torch.uint8), codes.view(torch.uint8)
                    )


if __name__ == "__main__":
    unittest.main()
