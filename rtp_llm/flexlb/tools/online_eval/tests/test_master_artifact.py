import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from adapters.master_artifact import configure_master
from artifacts.provenance import gate_provenance
from context_fixtures import plan


@pytest.mark.parametrize("missing", [None, "master-artifact.json", "actual-master-config.json", "perf.json"])
def test_gate_provenance_consumes_launch_outputs_and_rejects_missing_observed_inputs(tmp_path, monkeypatch, missing):
    jar = tmp_path / "master.jar"
    jar.write_bytes(b"test-master")
    monkeypatch.setattr("runtime.paths.API_JAR", jar)
    monkeypatch.setattr("runtime.paths.MOCK_JAR", jar)
    configuration = {"schemaVersion": 3, "dispatcher": {"type": "NON_BATCH"}}
    override = tmp_path / "override.json"
    override.write_text(json.dumps(configuration))
    monkeypatch.setenv("FLEXLB_FT_MASTER_CONFIG_FILE", str(override))
    spec = SimpleNamespace(
        discovery="discovery_file", n_prefill=1, n_decode=1, prefill_cache_blocks=64, decode_cache_blocks=64, mock_extra_args=()
    )
    env = SimpleNamespace(run_dir=tmp_path, discovery_file=tmp_path / "discovery.json", spec=spec)
    configure_master(env, dict(FLEXLB_CONFIG='{"schemaVersion":3}', MODEL_SERVICE_CONFIG="{}"), jar)
    (tmp_path / "perf.json").write_text('{"prefill":{"fixed_ms":1}}')
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"rid":"one","input_len":64,"output_len":2}\n')
    (tmp_path / "flow-input.json").write_text('{"environment":{"QPS":"10"}}')
    flow = SimpleNamespace(directory=tmp_path, trace_manifest=dict(path=str(trace), sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
    ctx = SimpleNamespace(
        artifact_dir=tmp_path, env=env, env_epoch=1, plan=plan(dict(id="run", implementation=dict(configuration_sha256="a" * 64)))
    )
    if missing:
        (tmp_path / missing).unlink()
        with pytest.raises(ValueError, match="required provenance file missing"):
            gate_provenance(ctx, flow, source_files=(), analyzer_file=__file__)
    else:
        evidence = gate_provenance(ctx, flow, source_files=(), analyzer_file=__file__)
        assert evidence["actual_master_config"] == configuration
        assert evidence["master_artifact"]["effective_config_sha256"] == hashlib.sha256(override.read_bytes()).hexdigest()
        assert evidence["master_artifact"]["jar_sha256"] == hashlib.sha256(jar.read_bytes()).hexdigest()


class MasterArtifactTest(unittest.TestCase):
    def test_ordinary_run_records_effective_inputs_without_version_pin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jar = root / "master.jar"
            jar.write_bytes(b"unversioned-jar")
            env = SimpleNamespace(run_dir=root, discovery_file=root / "discovery.json", spec=SimpleNamespace(discovery="discovery_file"))
            menv = dict(FLEXLB_CONFIG='{"schemaVersion":3}', MODEL_SERVICE_CONFIG='{"discovery_file":"x"}')
            with patch.dict(os.environ, {}, clear=True):
                result = configure_master(env, menv, jar)
            self.assertIs(result, menv)
            identity = json.loads((root / "master-artifact.json").read_text())
            self.assertEqual(identity["jar_sha256"], hashlib.sha256(jar.read_bytes()).hexdigest())
            self.assertIsNone(identity["source_commit"])
            self.assertEqual(json.loads((root / "actual-master-config.json").read_text()), {"schemaVersion": 3})

    def test_plain_jar_and_schema_one_config_uses_file_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jar = root / "master.jar"
            jar.write_bytes(b"old-jar")
            config = root / "config.json"
            config.write_text('{"schemaVersion":1,"dispatcher":{"type":"NON_BATCH"}}')
            env = SimpleNamespace(run_dir=root, discovery_file=root / "discovery.json", spec=SimpleNamespace(discovery="discovery_file"))
            menv = dict(FLEXLB_CONFIG='{"schemaVersion":3}', MODEL_SERVICE_CONFIG='{"discovery_file":"x","role_endpoints":[]}')
            with patch.dict(
                os.environ, {"FLEXLB_FT_MASTER_CONFIG_FILE": str(config), "FLEXLB_FT_MASTER_SOURCE_COMMIT": "a" * 40}, clear=True
            ):
                configure_master(env, menv, jar)
            self.assertEqual(json.loads(menv["FLEXLB_CONFIG"])["schemaVersion"], 1)
            self.assertNotIn("discovery_file", json.loads(menv["MODEL_SERVICE_CONFIG"]))
            self.assertEqual(menv["MOCK_DISCOVERY_FILE"], str(env.discovery_file))
            identity = json.loads((root / "master-artifact.json").read_text())
            self.assertEqual(identity["source_commit"], "a" * 40)
            self.assertEqual(identity["source_commit_origin"], "declared")
