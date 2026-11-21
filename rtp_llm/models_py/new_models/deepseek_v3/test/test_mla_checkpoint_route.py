"""DeepSeek V2 direct-Q and V3 compressed-Q checkpoint reads and runtime views."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.new_models.deepseek_v3.attention import DeepSeekV32MlaAttention


class AttentionLoader(NewModelLoader):
    """Use the production loader, with the release layer as its ownership root."""

    def _create_model(self):
        return DeepSeekV32MlaAttention(**self.model_config["attention"])


def attention_kwargs(cfg):
    return dict(
        hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_attention_heads"],
        q_lora_rank=cfg["q_lora_rank"] or 0,
        kv_lora_rank=cfg["kv_lora_rank"],
        nope_head_dim=cfg["qk_nope_head_dim"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        v_head_dim=cfg["v_head_dim"],
        layer_idx=0,
        params_dtype=torch.bfloat16,
    )


class MlaCheckpointRouteTest(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(
            hidden_size=16, num_attention_heads=4, q_lora_rank=None,
            kv_lora_rank=4, qk_nope_head_dim=2, qk_rope_head_dim=2, v_head_dim=2,
        )
        self.weights = {}
        g = torch.Generator().manual_seed(712)
        for name, shape in {
            "q_proj.weight": (16, 16),
            "kv_a_proj_with_mqa.weight": (6, 16),
            "kv_a_layernorm.weight": (4,),
            "kv_b_proj.weight": (16, 4),
            "o_proj.weight": (16, 8),
        }.items():
            self.weights[name] = torch.randn(shape, generator=g).bfloat16()

    def _run_case(self, cfg, weights, method, size=1, rank=0):
        args = dict(attention_kwargs(cfg), tp_size=size, tp_rank=rank)
        with tempfile.TemporaryDirectory() as root:
            save_file(weights, str(Path(root) / "model.safetensors"))
            loader = AttentionLoader(
                {"model_type": "deepseek2", "attention": args},
                NewLoaderConfig(device="cpu", compute_dtype=torch.bfloat16,
                                tp_size=size, tp_rank=rank, load_method=method),
                model_path=root,
            )
            layer = loader.load()
            self.assertEqual(loader.loaded_method, method)
        if cfg["q_lora_rank"]:
            q = weights["q_a_proj.weight"]
            qb = weights["q_b_proj.weight"].chunk(size, dim=0)[rank]
            torch.testing.assert_close(layer.q_b_proj.weight, qb)
        else:
            q = weights["q_proj.weight"].chunk(size, dim=0)[rank]
        kv = weights["kv_a_proj_with_mqa.weight"]
        torch.testing.assert_close(layer._fused_qkv_a_w, torch.cat([q, kv]))
        x = torch.linspace(-1, 1, q.shape[1], dtype=torch.bfloat16).view(1, -1)
        fused = torch.nn.functional.linear(x, layer._fused_qkv_a_w)
        torch.testing.assert_close(fused, torch.cat([
            torch.nn.functional.linear(x, q), torch.nn.functional.linear(x, kv)
        ], dim=-1))
        kv_b = weights["kv_b_proj.weight"].chunk(size, dim=0)[rank]
        heads = cfg["num_attention_heads"] // size
        k, v = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
        kv_b = kv_b.reshape(heads, k + v, cfg["kv_lora_rank"])
        torch.testing.assert_close(layer._kc_w, kv_b[:, :k])
        torch.testing.assert_close(layer._vc_w, kv_b[:, k:].transpose(1, 2))

    def test_direct_q_tp_read_contract(self):
        for rank in (0, 1):
            layer = DeepSeekV32MlaAttention(**attention_kwargs(self.cfg), tp_size=2, tp_rank=rank)
            q = self.weights["q_proj.weight"]
            spec = layer.checkpoint_read_spec("q_proj.weight", tuple(q.shape))
            self.assertIsNotNone(spec)
            torch.testing.assert_close(spec.materialize(q).tensor, q.chunk(2)[rank])
            self.assertIsNone(layer.checkpoint_read_spec("kv_a_proj_with_mqa.weight", (6, 16)))
            self._run_case(self.cfg, self.weights, NewLoaderLoadMethod.SCRATCH, 2, rank)

    def test_compressed_q_replication_and_q_b_sharding(self):
        cfg = dict(self.cfg, q_lora_rank=4)
        layer = DeepSeekV32MlaAttention(**attention_kwargs(cfg), tp_size=2, tp_rank=1)
        self.assertIsNone(layer.checkpoint_read_spec("q_a_proj.weight", (4, 16)))
        qb = torch.arange(64).view(16, 4)
        spec = layer.checkpoint_read_spec("q_b_proj.weight", tuple(qb.shape))
        torch.testing.assert_close(spec.materialize(qb).tensor, qb.chunk(2)[1])

    def test_native_fst_projection(self):
        self._run_case(self.cfg, self.weights, NewLoaderLoadMethod.FASTSAFETENSORS)

    def _release_layer(self, path):
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.layers.0.self_attn."
        weights = {}
        for key, shard in index.items():
            if key.startswith(prefix):
                with safe_open(str(path / shard), framework="pt", device="cpu") as f:
                    weights[key[len(prefix):]] = f.get_tensor(key)
        self._run_case(cfg, weights, NewLoaderLoadMethod.FASTSAFETENSORS)

    @unittest.skipUnless(os.getenv("DEEPSEEK_V2_LITE_CHECKPOINT"), "release fixture not provided")
    def test_release_layer_fst(self):
        self._release_layer(os.environ["DEEPSEEK_V2_LITE_CHECKPOINT"])

    @unittest.skipUnless(os.getenv("GLM4_MOE_LITE_CHECKPOINT"), "release fixture not provided")
    def test_release_compressed_q_fst(self):
        self._release_layer(os.environ["GLM4_MOE_LITE_CHECKPOINT"])


if __name__ == "__main__":
    unittest.main()
