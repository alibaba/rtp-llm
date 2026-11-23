import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewLoaderLoadMethod, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v3.indexer_parameters import SparseIndexerParameters


class IndexerLoader(NewModelLoader):
    def _create_model(self):
        owner = SparseIndexerParameters(**self.model_config["indexer"])
        owner.weights_proj.fused_gate = self.model_config.get("fused_gate", False)
        return owner


class SparseIndexerCheckpointTest(unittest.TestCase):
    def _check(self, data, cfg, method, fused=False):
        args = dict(
            hidden_size=cfg["hidden_size"], q_lora_rank=cfg["q_lora_rank"],
            index_n_heads=cfg["index_n_heads"], index_head_dim=cfg["index_head_dim"],
            layernorm_eps=cfg["rms_norm_eps"], params_dtype=torch.bfloat16, quant_config=None,
        )
        with tempfile.TemporaryDirectory() as root:
            save_file(data, str(Path(root) / "model.safetensors"))
            loader = IndexerLoader(
                {"model_type": "glm_5", "indexer": args, "fused_gate": fused}, NewLoaderConfig(device="cpu", load_method=method),
                model_path=root,
            )
            owner = loader.load()
            self.assertEqual(loader.loaded_method, method)
        owner.to(dtype=torch.bfloat16)
        if torch.version.hip is None:
            self.assertEqual(owner.weights_proj.weight.is_contiguous(), fused)
        self.assertEqual(owner.weights_proj.weight.dtype, torch.float32)
        torch.testing.assert_close(owner.weights_proj.weight, data["weights_proj.weight"].float(), rtol=0, atol=0)
        x = torch.linspace(-1, 1, cfg["hidden_size"]).bfloat16().reshape(1, -1)
        qc = torch.linspace(1, -1, cfg["q_lora_rank"]).bfloat16().reshape(1, -1)
        torch.testing.assert_close(owner.weights_proj(x.float()), F.linear(x.float(), data["weights_proj.weight"].float()))
        torch.testing.assert_close(owner.wq_b(qc), F.linear(qc, data["wq_b.weight"].bfloat16()))
        k = F.linear(x, data["wk.weight"].bfloat16())
        torch.testing.assert_close(
            owner.k_norm(owner.wk(x)), F.layer_norm(k, (cfg["index_head_dim"],),
                data["k_norm.weight"].bfloat16(), data["k_norm.bias"].bfloat16(), cfg["rms_norm_eps"]),
        )

    def test_bfloat16_migration_preserves_float_gate(self):
        cfg = dict(hidden_size=8, q_lora_rank=4, index_n_heads=2, index_head_dim=4, rms_norm_eps=1e-6)
        shapes = {"wk.weight": (4,8), "wq_b.weight": (8,4), "k_norm.weight": (4,),
                  "k_norm.bias": (4,), "weights_proj.weight": (2,8)}
        gen = torch.Generator().manual_seed(2026)
        data = {key: torch.randn(shape, generator=gen) for key, shape in shapes.items()}
        for method in (NewLoaderLoadMethod.SCRATCH, NewLoaderLoadMethod.FASTSAFETENSORS):
            for fused in (False, True):
                self._check(data, cfg, method, fused=fused)

    @unittest.skipUnless(os.environ.get("GLM5_BF16_CHECKPOINT"), "release checkpoint not provided")
    def test_real_glm5_sparse_indexer(self):
        path = Path(os.environ["GLM5_BF16_CHECKPOINT"])
        cfg = json.loads((path / "config.json").read_text())
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = "model.layers.0.self_attn.indexer."
        data = {}
        for name, shard in index.items():
            if name.startswith(prefix):
                with safe_open(str(path / shard), framework="pt", device="cpu") as f:
                    data[name[len(prefix):]] = f.get_tensor(name)
        self._check(data, cfg, NewLoaderLoadMethod.FASTSAFETENSORS)


if __name__ == "__main__":
    unittest.main()
