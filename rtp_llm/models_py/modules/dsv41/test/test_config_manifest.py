import json
import tempfile
import unittest
from pathlib import Path

from fixture import flash_config

from rtp_llm.config.dsv41_config import (
    V41Config,
    derive_checkpoint_revision,
    resolve_checkpoint_revision,
)
from rtp_llm.config.dsv41_weights import build_v41_manifest, validate_inventory


class ConfigManifestTest(unittest.TestCase):
    def test_owner_schedule_and_capacity(self):
        config = V41Config.from_dict(flash_config())
        self.assertEqual(
            config.global_owner_by_layer,
            (-1, -1) + (2,) * 6 + (8,) * 6 + (14,) * 6 + (20,) * 20,
        )
        self.assertEqual(
            config.topk_owner_by_layer[-20:],
            (20,) * 4 + (24,) * 4 + (28,) * 4 + (32,) * 4 + (36,) * 4,
        )
        self.assertEqual(config.engram_host_bytes, 202758032400)
        config.validate_parallelism(tp_size=1, ep_size=16)
        config.validate_parallelism(tp_size=4, ep_size=8)
        with self.assertRaisesRegex(ValueError, "o_groups"):
            config.validate_parallelism(tp_size=16, ep_size=16)

    def test_derived_revision_tracks_checkpoint_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(flash_config()), encoding="utf-8"
            )
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}', encoding="utf-8"
            )
            first = derive_checkpoint_revision(root)
            self.assertEqual(first, derive_checkpoint_revision(root))
            self.assertEqual(len(first), 40)
            self.assertTrue(all(c in "0123456789abcdef" for c in first))
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map": {"layers.0.attn.wq.weight": "model-00001.safetensors"}}',
                encoding="utf-8",
            )
            second = derive_checkpoint_revision(root)
            self.assertNotEqual(first, second)
            changed = flash_config()
            changed["bos_token_id"] = 3
            (root / "config.json").write_text(json.dumps(changed), encoding="utf-8")
            self.assertNotEqual(second, derive_checkpoint_revision(root))

    def test_resolve_revision_explicit_override_and_derived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(flash_config()), encoding="utf-8"
            )
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}', encoding="utf-8"
            )
            self.assertEqual(
                resolve_checkpoint_revision(root), derive_checkpoint_revision(root)
            )
            override = "a" * 40
            self.assertEqual(resolve_checkpoint_revision(root, override), override)
            with self.assertRaisesRegex(ValueError, "immutable checkpoint"):
                resolve_checkpoint_revision(root, "not-a-40-hex-revision")

    def test_invalid_contracts_fail(self):
        for field, value in (
            ("compress_ratios", [0] * 43),
            ("kv_source_layer_ids", [2, 8, 14, 24]),
            ("num_hidden_layers", True),
            ("num_hash_layers", 3),
            ("rms_norm_eps", 1e-6),
            ("engram_pad_token_id", 1),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                raw = flash_config()
                raw["text_config"][field] = value
                V41Config.from_dict(raw)
        raw = flash_config()
        raw["quantization_config"]["weight_block_size"] = [128, 128]
        with self.assertRaises(ValueError):
            V41Config.from_dict(raw)

    def test_exact_inventory_roles_and_wo_a(self):
        specs = build_v41_manifest(V41Config.from_dict(flash_config()))
        self.assertEqual(len(specs), 96085)
        self.assertEqual(
            len([spec for spec in specs.values() if spec.placement == "host_shared"]), 4
        )
        self.assertEqual(
            specs["layers.0.attn.wo_a.weight"].conversion, "dequantize_bf16"
        )
        self.assertEqual(specs["mtp.0.attn.wo_a.weight"].conversion, "dequantize_bf16")
        self.assertNotIn("layers.24.attn.indexer.wk.weight", specs)
        self.assertIn("layers.24.attn.indexer.wq_b.weight", specs)
        self.assertNotIn("layers.20.attn.compressor.wgate.weight", specs)
        self.assertNotIn("hc_head_base", specs)
        result = validate_inventory(specs, specs, include_draft=False)
        self.assertTrue(result["ignored"])
        self.assertTrue(all(name.startswith("mtp.") for name in result["ignored"]))
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_inventory(specs, set(specs) - {"layers.14.engram.embed.scale"})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_inventory(specs, set(specs) | {"hc_head_base"})


if __name__ == "__main__":
    unittest.main()
