import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from runtime.master_artifact import configure_master


class MasterArtifactTest(unittest.TestCase):
    def test_ordinary_run_records_effective_inputs_without_version_pin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jar = root / "master.jar"
            jar.write_bytes(b"unversioned-jar")
            env = SimpleNamespace(
                run_dir=root,
                discovery_file=root / "discovery.json",
                spec=SimpleNamespace(discovery="discovery_file"),
            )
            menv = dict(FLEXLB_CONFIG='{"schemaVersion":3}',
                        MODEL_SERVICE_CONFIG='{"discovery_file":"x"}')
            with patch.dict(os.environ, {}, clear=True):
                result = configure_master(env, menv, jar)
            self.assertIs(result, menv)
            identity = json.loads((root / "master-artifact.json").read_text())
            self.assertEqual(identity["jar_sha256"], hashlib.sha256(jar.read_bytes()).hexdigest())
            self.assertIsNone(identity["source_commit"])
            self.assertEqual(json.loads((root / "actual-master-config.json").read_text()),
                             {"schemaVersion": 3})

    def test_plain_jar_and_schema_one_config_uses_file_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jar = root / "master.jar"
            jar.write_bytes(b"old-jar")
            config = root / "config.json"
            config.write_text('{"schemaVersion":1,"dispatcher":{"type":"NON_BATCH"}}')
            env = SimpleNamespace(
                run_dir=root, discovery_file=root / "discovery.json",
                spec=SimpleNamespace(discovery="discovery_file"),
            )
            menv = dict(FLEXLB_CONFIG='{"schemaVersion":3}',
                        MODEL_SERVICE_CONFIG='{"discovery_file":"x","role_endpoints":[]}')
            with patch.dict(os.environ, {"FLEXLB_FT_MASTER_CONFIG_FILE": str(config),
                                      "FLEXLB_FT_MASTER_SOURCE_COMMIT": "a" * 40}, clear=True):
                configure_master(env, menv, jar)
            self.assertEqual(json.loads(menv["FLEXLB_CONFIG"])["schemaVersion"], 1)
            self.assertNotIn("discovery_file", json.loads(menv["MODEL_SERVICE_CONFIG"]))
            self.assertEqual(menv["MOCK_DISCOVERY_FILE"], str(env.discovery_file))
            identity = json.loads((root / "master-artifact.json").read_text())
            self.assertEqual(identity["source_commit"], "a" * 40)
            self.assertEqual(identity["source_commit_origin"], "declared")
