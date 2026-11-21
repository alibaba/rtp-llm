"""Opt-in Pro *source* regression; this is not a model-forward/golden test.

Pass a target or combined Pro release via DSV4_PRO_SOURCE. For a combined
target/draft release, check target namespace isolation, not path equivalence
to the Pro smoke fixture. Only four projection tensors are materialized.
"""

import json
import os
import struct
import unittest
from collections import Counter
from pathlib import Path

import torch

from rtp_llm.models_py.model_loader import _ExpertRangeFilter
from rtp_llm.models_py.new_models.deepseek_v4.checkpoint_namespace import (
    V4CheckpointNamespace,
)
from rtp_llm.models_py.weight_mapper import get_all_weights


@unittest.skipUnless(
    os.environ.get("DSV4_PRO_SOURCE"), "requires DeepSeek-V4-Pro weights"
)
class V4ProReleaseSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["DSV4_PRO_SOURCE"])
        cls.config = json.loads((cls.root / "config.json").read_text())
        cls.index = json.loads((cls.root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        cls.owns = V4CheckpointNamespace(
            layer_prefix="layers",
            global_prefix="",
            layer_count=cls.config["num_hidden_layers"],
            commit_only=False,
        ).accepts

    def test_target_release_expert_partition_all_layers(self):
        cfg, index = self.config, self.index
        n, experts, hidden = 61, 384, 7168
        self.assertEqual(cfg["model_type"], "deepseek_v4")
        self.assertEqual(
            (cfg["num_hidden_layers"], cfg["n_routed_experts"], cfg["hidden_size"]),
            (n, experts, hidden),
        )
        self.assertEqual(cfg["moe_intermediate_size"], 3072)
        self.assertEqual(cfg["o_groups"], 16)
        self.assertGreaterEqual(len(cfg["compress_ratios"]), n)
        counter = Counter()
        routers = [_ExpertRangeFilter(experts, 4, r) for r in range(4)]
        for key in index:
            layer, sep, suffix = key.partition(".ffn.experts.")
            if not self.owns(key):
                self.assertFalse(key.startswith("layers."), key)
                continue
            if not sep:
                continue
            expert, proj, kind = suffix.split(".")
            layer_id, expert_id = int(layer.split(".")[1]), int(expert)
            self.assertIn(proj, ("w1", "w2", "w3"))
            self.assertIn(kind, ("weight", "scale"))
            self.assertEqual(
                tuple(r for r, router in enumerate(routers) if router.should_load(key)),
                (expert_id // 96,),
                key,
            )
            counter[layer_id, expert_id // 96] += 1
        self.assertEqual(counter, {(i, r): 96 * 6 for i in range(n) for r in range(4)})

    def test_packed_projection_bytes_per_ep_rank(self):
        # One real I8 packed projection per rank. Source of truth is the
        # independently parsed safetensors header + bytes, not another loader.
        names = {f"layers.60.ffn.experts.{96 * r + 95}.w2.weight" for r in range(4)}
        sources = sorted({str(self.root / self.index[name]) for name in names})
        seen = set()
        for rank in range(4):
            router = _ExpertRangeFilter(384, 4, rank)
            loaded = list(
                get_all_weights(
                    sources,
                    name_filter=lambda key: self.owns(key)
                    and key in names
                    and router.should_load(key),
                )
            )
            self.assertEqual(len(loaded), 1)
            name, value = loaded[0]
            self.assertEqual(name, f"layers.60.ffn.experts.{96 * rank + 95}.w2.weight")
            self.assertEqual(value.dtype, torch.int8)
            self.assertEqual(tuple(value.shape), (7168, 1536))
            with (self.root / self.index[name]).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(size))
                self.assertEqual(header[name]["dtype"], "I8")
                scale = header[name.replace(".weight", ".scale")]
                self.assertEqual(
                    (scale["dtype"], scale["shape"]), ("F8_E8M0", [7168, 96])
                )
                start, end = header[name]["data_offsets"]
                stream.seek(8 + size + start)
                self.assertEqual(
                    value.contiguous().numpy().tobytes(), stream.read(end - start)
                )
            seen.add(name)
        self.assertEqual(seen, names)


if __name__ == "__main__":
    unittest.main()
