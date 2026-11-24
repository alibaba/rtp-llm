"""Opt-in metadata contract against a release V4 checkpoint; no large tensors read."""

import json
import os
import struct
import unittest
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

from rtp_llm.models_py.new_models.deepseek_v4.moe import OwnedFp8Fp4MoE


@lru_cache(maxsize=8)
def _metadata(shard_path):
    with shard_path.open("rb") as shard:
        size = struct.unpack("<Q", shard.read(8))[0]
        if not 0 < size < 100_000_000:
            raise ValueError(f"Invalid safetensors header size {size}")
        return json.loads(shard.read(size))


class ReleaseV4MoeHeaderTest(unittest.TestCase):
    @unittest.skipUnless(os.getenv("V4_RELEASE_CHECKPOINT"), "release path not set")
    def test_ep_cut_and_shapes_match_release_bytes(self):
        ckpt = Path(os.environ["V4_RELEASE_CHECKPOINT"])
        config = json.loads((ckpt / "config.json").read_text())
        self.assertEqual(config["model_type"], "deepseek_v4")
        weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        dtypes = {
            torch.bfloat16: "BF16",
            torch.float32: "F32",
            torch.int32: "I32",
            torch.int8: "I8",
            torch.float8_e4m3fn: "F8_E4M3",
            torch.float8_e8m0fnu: "F8_E8M0",
        }
        for layer in (0, config["num_hash_layers"]):
            self.assertLess(layer, config["num_hidden_layers"])
            prefix = f"layers.{layer}.ffn"
            keys = {name for name in weight_map if name.startswith(prefix + ".")}
            for rank in range(4):
                with torch.device("meta"):
                    owner = OwnedFp8Fp4MoE(
                        layer,
                        config["hidden_size"],
                        config["moe_intermediate_size"],
                        config["n_routed_experts"],
                        config["num_experts_per_tok"],
                        config["n_shared_experts"],
                        n_hash_layers=config["num_hash_layers"],
                        vocab_size=config["vocab_size"],
                        prefix=prefix,
                        ep_rank=rank,
                        ep_size=4,
                    )
                found = set()
                for name in keys:
                    metadata = _metadata(ckpt / weight_map[name])[name]
                    spec = owner.checkpoint_read_spec(name, metadata["shape"])
                    if not spec.load:
                        continue
                    if name.endswith(".gate.tid2eid"):
                        self.assertIn(metadata["dtype"], ("I32", "I64"))
                        self.assertEqual(owner._target(name).dtype, torch.int32)
                    else:
                        self.assertEqual(
                            dtypes[owner._target(name).dtype], metadata["dtype"], name
                        )
                    found.add(name.removeprefix(prefix + "."))
                self.assertEqual(found, owner._expected)
                if layer == 0 and rank == 0:
                    # Read just the released token-to-expert table via mmap;
                    # all routed/shared stacks above stay on the meta device.
                    key = f"{prefix}.gate.tid2eid"
                    with safe_open(ckpt / weight_map[key], framework="pt") as handle:
                        table = handle.get_tensor(key)
                        owner.router_tid2eid = torch.nn.Parameter(
                            torch.empty_like(owner.router_tid2eid, device="cpu"),
                            requires_grad=False,
                        )
                        owner.load_weights({key: table})
                    torch.testing.assert_close(owner.router_tid2eid, table.int())


if __name__ == "__main__":
    unittest.main()
