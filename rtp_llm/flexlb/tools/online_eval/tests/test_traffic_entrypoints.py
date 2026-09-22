"""Both stress inputs materialize through the same registered source contract."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic.prefix_lineage import encode


class TrafficEntrypointTest(unittest.TestCase):
    def test_stress_rejects_raw_trace_and_synthetic_replay(self):
        script = ROOT / "scripts/stress/run_online_eval.sh"
        raw = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                             env=dict(os.environ, TRACE_FILE="/tmp/raw.jsonl"))
        self.assertEqual(2, raw.returncode)
        self.assertIn("TRACE_FILE is generated", raw.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "source.json"
            spec.write_text(json.dumps(dict(kind="synthetic", model="realistic", version="1",
                parameters=dict(seed=1, count=2))))
            replay = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                                    env=dict(os.environ, TRAFFIC_SOURCE_SPEC=str(spec),
                                             SEND_MODE="replay", PROMETHEUS_BIN="python3"))
            self.assertEqual(2, replay.returncode)
            self.assertIn("synthetic traffic has ordinal timestamps", replay.stderr)

    def test_pinned_dag_and_synthetic_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = directory / "small.xz"
            model.write_bytes(encode([(0, 512, -1, 0), (10, 1024, 0, 1)]))
            model.with_suffix(".manifest.json").write_text(json.dumps(dict(
                bytes=model.stat().st_size,
                sha256=hashlib.sha256(model.read_bytes()).hexdigest(), count=2,
            )))
            lineage = directory / "lineage.jsonl"
            subprocess.run([sys.executable, str(ROOT / "scripts/materialize_traffic.py"),
                            "--lineage-model", str(model), "--namespace", "check",
                            "--max-requests", "2", "--out", str(lineage)], check=True, capture_output=True)
            rows = [json.loads(line) for line in lineage.read_text().splitlines()]
            self.assertEqual(["check:0", "check:1"], [row["rid"] for row in rows])
            self.assertEqual(rows[0]["input_token_blocks"][0],
                             rows[1]["input_token_blocks"][0])
            self.assertEqual(420, rows[0]["ol"])
            manifest = json.loads(lineage.with_suffix(".manifest.json").read_text())
            self.assertEqual("EMPIRICAL_PREFIX_STRUCTURE", manifest["realism"])
            self.assertEqual(2, manifest["projection"]["selected_requests"])

            spec = directory / "synthetic.json"
            spec.write_text(json.dumps(dict(kind="synthetic", model="realistic", version="1",
                parameters=dict(seed=7, count=3, families=2, prefix_blocks=1,
                    suffix_blocks=1, zipf_alpha=1, cold_fraction=0,
                    output_tokens=4))))
            synthetic = directory / "synthetic.jsonl"
            subprocess.run([sys.executable, str(ROOT / "scripts/materialize_traffic.py"),
                            "--spec", str(spec), "--namespace", "check",
                            "--max-requests", "2", "--out", str(synthetic)], check=True, capture_output=True)
            self.assertEqual(2, len(synthetic.read_text().splitlines()))
            model.write_bytes(model.read_bytes() + b"tampered")
            failed = subprocess.run([sys.executable, str(ROOT / "scripts/materialize_traffic.py"),
                                     "--lineage-model", str(model), "--namespace", "bad",
                                     "--out", str(directory / "bad.jsonl")], capture_output=True)
            self.assertNotEqual(0, failed.returncode)
            self.assertFalse((directory / "bad.jsonl").exists())

    def test_java_templates_are_reproducible_from_model(self):
        from scripts.derive_master_templates import derive

        model = ROOT / "data/traffic_models/frontend_20260921.xz"
        stored = json.loads((ROOT / "data/traffic_models/master_batch_templates.json").read_text())
        self.assertEqual(stored, derive(model))
        self.assertGreaterEqual(len({row["il"] for row in stored["templates"]}), 32)


if __name__ == "__main__":
    unittest.main()
