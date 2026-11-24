import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewLoaderLoadMethod, NewModelLoader
from rtp_llm.models_py.new_models.kimi_linear.kda_parameters import KimiKDAParameters


class KdaLoader(NewModelLoader):
    def _create_model(self):
        return KimiKDAParameters(self.model_config["kda"], "self_attn")


def kda_config(hidden=8, heads=2, head_dim=4):
    return dict(
        hidden_size=hidden, kda_num_heads=heads, kda_head_dim=head_dim, conv_kernel=4, rms_norm_eps=1e-5,
        attn_tp_size=1, attn_tp_rank=0, quant_config=None, params_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32, conv_state_dtype=torch.bfloat16,
    )


class KimiKdaCheckpointTest(unittest.TestCase):
    def check(self, values, cfg, method):
        n = cfg["attn_tp_size"]
        rank = cfg["attn_tp_rank"]
        with tempfile.TemporaryDirectory() as root:
            save_file(values, str(Path(root) / "model.safetensors"))
            loader = KdaLoader(
                {"model_type": "kimi_linear", "kda": cfg},
                NewLoaderConfig(device="cpu", load_method=method, tp_size=n, tp_rank=rank),
                model_path=root,
            )
            owner = loader.load()
            self.assertEqual(loader.loaded_method, method)
        owner.to(dtype=torch.bfloat16)
        qkv, conv = [], []
        for name in ("q", "k", "v"):
            qkv.append(values[name + "_proj.weight"].chunk(n)[rank])
            conv.append(values[name + "_conv1d.weight"].squeeze(1).chunk(n)[rank])
        torch.testing.assert_close(owner.conv1d.weight, torch.cat(conv))
        torch.testing.assert_close(owner.qkv_proj.weight, torch.cat(qkv))
        x = torch.linspace(-1, 1, cfg["hidden_size"], dtype=torch.bfloat16)[None, :]
        expected = torch.cat([torch.nn.functional.linear(x, w) for w in qkv], dim=-1)
        torch.testing.assert_close(owner.qkv_proj(x), expected)
        torch.testing.assert_close(owner.dt_bias, values["dt_bias"].reshape(-1).chunk(n)[rank].float())
        torch.testing.assert_close(owner.A_log, values["A_log"].reshape(-1).chunk(n)[rank].float())

    def test_two_rank_kda_weights(self):
        cfg = kda_config()
        g = torch.Generator().manual_seed(57)
        shapes = {"b_proj.weight": (2, 8), "o_proj.weight": (8, 8), "A_log": (1, 1, 2, 1),
                  "dt_bias": (8,), "o_norm.weight": (4,)}
        for p in ("q", "k", "v"):
            shapes[p + "_proj.weight"] = (8, 8)
            shapes[p + "_conv1d.weight"] = (8, 1, 4)
        for p in ("f", "g"):
            shapes[p + "_a_proj.weight"] = (4, 8)
            shapes[p + "_b_proj.weight"] = (8, 4)
        values = {k: torch.randn(shape, generator=g).bfloat16() for k, shape in shapes.items()}
        # Values between adjacent BF16 numbers expose a round-then-promote bug.
        for key in ("A_log", "dt_bias"):
            values[key] = torch.randn(shapes[key], generator=g)
        for r in (0, 1):
            self.check(values, dict(cfg, attn_tp_size=2, attn_tp_rank=r), NewLoaderLoadMethod.SCRATCH)
        self.check(values, cfg, NewLoaderLoadMethod.FASTSAFETENSORS)

    @unittest.skipUnless(os.environ.get("KIMI_LINEAR_CHECKPOINT"), "release checkpoint not provided")
    def test_release_first_kda_layer(self):
        path = Path(os.environ["KIMI_LINEAR_CHECKPOINT"])
        raw = json.loads((path / "config.json").read_text())
        spec = raw["linear_attn_config"]
        cfg = kda_config(raw["hidden_size"], spec["num_heads"], spec["head_dim"])
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.layers.0.self_attn."
        values = {}
        for name, shard in index.items():
            if name.startswith(prefix):
                with safe_open(str(path / shard), framework="pt", device="cpu") as f:
                    values[name[len(prefix):]] = f.get_tensor(name)
        self.check(values, cfg, NewLoaderLoadMethod.FASTSAFETENSORS)


if __name__ == "__main__":
    unittest.main()
