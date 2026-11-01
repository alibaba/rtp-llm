"""CPU-only owner input/publication contracts; no fake model or GPU results."""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from tools.engram_sidecar.framework.checkpoint_support import (
    digest, failure_fields, private_write, read_source,
)


class OwnerInputTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.raw = {"model_type": "deepseek_v41", "image_token_id": 42,
                    "text_config": {"engram_layer_ids": [0, 2], "hidden_size": 128}}
        self.write_config(self.raw)
        (self.checkpoint / "model.safetensors.index.json").write_text('{"weight_map":{"a":"a.safetensors"}}')

    def write_config(self, raw):
        (self.checkpoint / "config.json").write_text(json.dumps(raw))

    def read(self, **kwargs):
        return read_source(checkpoint=self.checkpoint, model_revision="release-immutable-123", **kwargs)

    def test_real_config_merges_text_but_keeps_top_level_fields(self):
        source = self.read()
        self.assertEqual(source.config["engram_layer_ids"], [0, 2])
        self.assertEqual(source.config["image_token_id"], 42)
        self.assertNotIn("text_config", source.config)
        self.assertEqual(source.config["checkpoint_path"], str(self.checkpoint))
        self.assertFalse(source.component_fixture)
        self.assertIsNone(source.expected_hash_abi)

    def test_metadata_digests_bind_exact_files(self):
        source = self.read()
        self.assertEqual(source.config_sha256, digest((self.checkpoint / "config.json").read_bytes()))
        self.assertEqual(source.index_sha256, digest((self.checkpoint / "model.safetensors.index.json").read_bytes()))
        self.write_config({**self.raw, "image_token_id": 43})
        self.assertNotEqual(source.config_sha256, self.read().config_sha256)

    def test_real_checkpoint_requires_revision(self):
        for revision in (None, "", "  ", "bad\nrevision", "x" * 513):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                read_source(checkpoint=self.checkpoint, model_revision=revision)

    def test_relative_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            read_source(checkpoint=Path("relative"), model_revision="release")

    def test_fixture_remains_compatible(self):
        self.checkpoint.rename(self.root / "owner_checkpoint")
        (self.root / "fixture.json").write_text(json.dumps({"model_revision": "fixture-v2", "hash_abi": "sha256:" + "a" * 64}))
        source = read_source(fixture_root=self.root)
        self.assertEqual(source.model_revision, "fixture-v2")
        self.assertEqual(source.expected_hash_abi, "sha256:" + "a" * 64)
        self.assertTrue(source.component_fixture)

    def test_source_modes_cannot_be_mixed(self):
        with self.assertRaises(ValueError):
            self.read(fixture_root=self.root)
        with self.assertRaises(ValueError):
            read_source()
        with self.assertRaises(ValueError):
            read_source(fixture_root=self.root, model_revision="override")

    def test_component_fixture_cannot_claim_real_checkpoint_mode(self):
        self.write_config({**self.raw, "fixture_component_only": True})
        with self.assertRaisesRegex(ValueError, "component fixture"):
            self.read()

    def test_bad_text_config_rejected(self):
        self.write_config({"text_config": []})
        with self.assertRaises(ValueError):
            self.read()

    def test_cuda_failure_diagnostics_do_not_copy_arbitrary_text(self):
        error = RuntimeError("owner_create: CUDA_ERROR_OUT_OF_MEMORY (code=2): private secret payload")
        self.assertEqual(failure_fields(error), dict(error_type="RuntimeError", code=None,
                         cuda_stage="owner_create", cuda_error="CUDA_ERROR_OUT_OF_MEMORY", cuda_code=2))
        for message in ("private capability data", "owner_create: CUDA_UNKNOWN (code=-1): secret"):
            self.assertEqual(failure_fields(RuntimeError(message)), dict(error_type="RuntimeError", code=None))

    def test_capabilities_write_is_private_atomic_replace(self):
        path = self.root / "private.json"
        private_write(path, b"first")
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        private_write(path, b"second")
        self.assertEqual(path.read_bytes(), b"second")
        self.assertFalse(list(self.root.glob("*.tmp-*")))


if __name__ == "__main__":
    unittest.main()
