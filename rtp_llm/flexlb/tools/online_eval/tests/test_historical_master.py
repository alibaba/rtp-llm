import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runtime import historical_master as hm


class HistoricalMasterTest(unittest.TestCase):
    def test_pin_validation_and_runtime_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jar = root / "master.jar"
            jar.write_bytes(b"test-jar")
            config = root / "config.json"
            raw = '{"schemaVersion":1,"dispatcher":{"type":"NON_BATCH"}}'
            config.write_text(raw)
            manifest = {"source_commit": "a" * 40}
            for key, path in (("jar", jar), ("config", config)):
                manifest[key] = str(path)
                manifest[key + "_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with patch.dict(
                os.environ,
                {"FLEXLB_FT_HISTORICAL_MASTER_MANIFEST": str(manifest_path)},
            ):
                self.assertEqual(hm.load_manifest(), manifest)
                with patch.object(hm, "MANIFEST", manifest):
                    env = SimpleNamespace(
                        spec=SimpleNamespace(
                            discovery="discovery_file", raw_config=None
                        ),
                        discovery_file=root / "discovery.json",
                        run_dir=root,
                    )
                    adapted = hm.adapt_env(
                        env,
                        {
                            "MODEL_SERVICE_CONFIG": (
                                '{"discovery_file":"x","role_endpoints":[]}'
                            )
                        },
                    )
                    self.assertEqual(adapted["FLEXLB_CONFIG"], raw)
                    self.assertNotIn(
                        "discovery_file", json.loads(adapted["MODEL_SERVICE_CONFIG"])
                    )
                    self.assertEqual(
                        adapted["MOCK_DISCOVERY_FILE"], str(env.discovery_file)
                    )
                jar.write_bytes(b"different-build")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    hm.load_manifest()

    def test_default_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(hm.load_manifest())
        with patch.object(hm, "MANIFEST", None):
            env = {"original": 1}
            self.assertIs(hm.adapt_env(None, env), env)
