import json
import struct
import tempfile
import unittest
from pathlib import Path

from rtp_llm.tools.checkpoint_inventory import (
    inventory_checkpoint,
    read_safetensors_header,
)


class CheckpointInventoryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        fixture = (
            Path(__file__).parents[2] / "config/test/testdata/deepseek_v41_flash.json"
        )
        self.config = json.loads(fixture.read_text())
        self.config["text_config"].update(engram_layer_ids=[], engram_num_embeddings=[])
        (self.path / "config.json").write_text(json.dumps(self.config))

    def write_shard(self, name, tensors):
        header = json.dumps(tensors).encode()
        end = max((t["data_offsets"][1] for t in tensors.values()), default=0)
        (self.path / name).write_bytes(
            struct.pack("<Q", len(header)) + header + bytes(end)
        )

    def write_index(self, names):
        (self.path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": names})
        )

    def test_stage_and_expert_ownership(self):
        names = [
            "embed.weight",
            "norm.weight",
            "head.weight",
            "layers.20.attn.wkv.weight",
            "layers.21.attn.wkv.weight",
            "mtp.0.ffn.gate.weight",
            "vision.norm.weight",
        ]
        # Unequal per-expert bytes would be assigned to the correct rank, not averaged.
        names += [
            f"layers.{layer}.ffn.experts.{expert}.w1.weight"
            for layer in (20, 21)
            for expert in range(384)
        ]
        tensors = {}
        offset = 0
        for name in names:
            elements = 2 if name == "layers.20.ffn.experts.383.w1.weight" else 1
            tensors[name] = {
                "dtype": "BF16",
                "shape": [elements],
                "data_offsets": [offset, offset + 2 * elements],
            }
            offset += 2 * elements
        self.write_shard("one.safetensors", tensors)
        self.write_index({name: "one.safetensors" for name in names})
        report = inventory_checkpoint(self.path, include_tensors=True)
        self.assertEqual(report["tensor_bytes"], 2 * len(names) + 2)
        self.assertEqual(report["categories"]["target_other"], 10)
        self.assertEqual(
            report["stored_weight_bytes_per_rank"]["decode_dp8ep8"], [202] * 7 + [204]
        )
        self.assertEqual(
            report["stored_weight_bytes_per_rank"]["prefill_cep4pp2"],
            [[196] * 3 + [198], [198] * 4],
        )
        self.assertEqual(report["shards"][0]["tensors"], tensors)
        self.assertFalse(report["payload_hashes_computed"])
        self.assertIn("config.json", report["assets_sha256"])

    def test_rejects_index_pointing_to_missing_tensor(self):
        self.write_shard("one.safetensors", {})
        self.write_index({"head.weight": "one.safetensors"})
        with self.assertRaisesRegex(ValueError, "missing from headers"):
            inventory_checkpoint(self.path)

    def test_rejects_header_tensor_absent_from_index(self):
        self.write_shard(
            "one.safetensors",
            {"norm.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}},
        )
        self.write_index({"head.weight": "one.safetensors"})
        with self.assertRaisesRegex(ValueError, "index/header mismatch"):
            inventory_checkpoint(self.path)

    def test_rejects_shape_mismatch_and_overlapping_offsets(self):
        for tensors, message in (
            (
                {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 2]}},
                "byte count mismatch",
            ),
            (
                {
                    "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                    "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
                },
                "overlapping",
            ),
        ):
            with self.subTest(message=message):
                self.write_shard("one.safetensors", tensors)
                with self.assertRaisesRegex(ValueError, message):
                    read_safetensors_header(self.path / "one.safetensors")

    def test_rejects_truncated_payload(self):
        header = json.dumps(
            {"a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
        ).encode()
        shard = self.path / "one.safetensors"
        shard.write_bytes(struct.pack("<Q", len(header)) + header)
        with self.assertRaisesRegex(ValueError, "file payload"):
            read_safetensors_header(shard)

    def test_rejects_duplicate_json_keys(self):
        shard = self.path / "one.safetensors"
        raw = b'{"a": {}, "a": {}}'
        shard.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            read_safetensors_header(shard)


if __name__ == "__main__":
    unittest.main()
