"""File-backed mock calibration and checked historical expression snapshots."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import re
import tempfile
import unittest

from flexlb_cfg import render_env
from flexlb_profile_data import load_mock_calibration
from runtime.perf_presets import ROOT, load_preset, load_performance_file
from runtime.harness import _write_master_config
from scenario.backend import make_env_spec
from scenario.compiler import environment

FLEXLB = ROOT.parents[1]
SNAPSHOTS = (
    ROOT / "tests/fixtures/environment_expectations.json",
    FLEXLB / "flexlb-mock-engine/src/test/resources/master-config-stress-na130.json",
    FLEXLB / "docs/config-examples/flexlb-queue-priority-batch.json",
    FLEXLB / "docs/config-examples/flexlb-queue-priority-non-batch.json",
)
JAVA_SNAPSHOT = FLEXLB / "flexlb-sync/src/test/java/org/flexlb/balance/prediction/FormulaPredictorTest.java"


def expressions(document, expected):
    if isinstance(document, dict):
        for child in document.values():
            yield from expressions(child, expected)
    elif isinstance(document, list):
        for child in document:
            yield from expressions(child, expected)
    elif isinstance(document, str):
        if document.startswith(expected.split(",", 1)[0]):
            yield document
        elif document.startswith("{"):
            try:
                yield from expressions(json.loads(document), expected)
            except json.JSONDecodeError:
                pass


def assert_snapshot_matches(path, expected):
    found = list(expressions(json.loads(Path(path).read_text()), expected))
    if not found or any(expression != expected for expression in found):
        raise AssertionError(f"stale mock expression snapshot: {path}")


class MockCalibrationTest(unittest.TestCase):
    def test_default_and_fault_presets_materialize_named_file_values(self):
        calibration = load_mock_calibration()
        expected_sha = hashlib.sha256((ROOT / "config/mock_calibrations/dsv4_l20.json").read_bytes()).hexdigest()
        for preset in ("default", "fault_env"):
            with self.subTest(preset=preset):
                performance, runtime = load_preset(preset)
                self.assertEqual(runtime, {})
                for key in ("id", "model", "hardware", "status"):
                    self.assertEqual(performance["calibration_" + key], calibration[key])
                self.assertEqual(performance["calibration_sha256"], expected_sha)
                self.assertEqual({key: performance["decode"][key]
                                  for key in calibration["decode"]}, calibration["decode"])
        config = json.loads(render_env("single-nonbatch"))
        expression = config["router"]["roles"]["prefill"]["executionTimeEstimator"]["expression"]
        self.assertEqual(expression, calibration["prefill_expression"])
        self.assertTrue((ROOT / "data/performance/synthetic_baseline.json").is_file())

    def test_run_files_expose_effective_default_and_fault_calibration(self):
        calibration = load_mock_calibration()
        expected_sha = hashlib.sha256((ROOT / "config/mock_calibrations/dsv4_l20.json").read_bytes()).hexdigest()
        for preset in ("default", "fault_env"):
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as directory:
                plan = environment({"perf_preset": preset}, "test", "single-nonbatch")
                spec = make_env_spec(plan, "single-nonbatch", {"master_base": 28000})
                run_dir = Path(directory)
                perf_path = run_dir / "perf.json"
                perf_path.write_text(json.dumps(spec.perf))
                master_path = _write_master_config(SimpleNamespace(run_dir=run_dir, spec=spec))
                observed_perf = json.loads(perf_path.read_text())
                envelope = json.loads(master_path.read_text())
                envs = dict(envelope["zone_process_setting"]["process_info"]["envs"])
                master = json.loads(envs["FLEXLB_CONFIG"])
                estimator = master["router"]["roles"]["prefill"]["executionTimeEstimator"]
                self.assertEqual(estimator["expression"], calibration["prefill_expression"])
                self.assertEqual(observed_perf["calibration_id"], calibration["id"])
                self.assertEqual(observed_perf["calibration_sha256"],
                                 hashlib.sha256((ROOT / "config/mock_calibrations/dsv4_l20.json").read_bytes()).hexdigest())
                for key, value in calibration["decode"].items():
                    self.assertEqual(observed_perf["decode"][key], value)
                if preset == "fault_env":
                    self.assertEqual(observed_perf["prefill"]["fixed_ms"], 100.0)

    def test_missing_or_invalid_calibration_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            with self.assertRaises(FileNotFoundError):
                load_mock_calibration(path)
            bad = dict(load_mock_calibration(), decode=dict(step_base_ms=-1))
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "decode calibration"):
                load_mock_calibration(path)
            source = Path(directory) / "performance.json"
            source.write_text(json.dumps({"calibration": "missing.json"}))
            with self.assertRaisesRegex(ValueError, "missing registered mock calibration"):
                load_performance_file(source)

    def test_historical_copies_are_guarded_against_drift(self):
        expression = load_mock_calibration()["prefill_expression"]
        for path in SNAPSHOTS:
            with self.subTest(path=path):
                assert_snapshot_matches(path, expression)
        source = JAVA_SNAPSHOT.read_text()
        block = source.split("private static final String DSV4_PRODUCTION_EXPRESSION =", 1)[1].split(";", 1)[0]
        java_expression = "".join(re.findall(r'"([^"\n]*)"', block))
        self.assertEqual(java_expression, expression)
        fixture = json.loads(SNAPSHOTS[0].read_text())
        calibration = load_mock_calibration()
        for entry in fixture.values():
            performance = entry.get("perf")
            if performance is None:
                continue
            for key, value in calibration["decode"].items():
                self.assertEqual(performance["decode"][key], value)
            for key in ("id", "model", "hardware", "status"):
                self.assertEqual(performance["calibration_" + key], calibration[key])
            self.assertEqual(performance["calibration_sha256"],
                             hashlib.sha256((ROOT / "config/mock_calibrations/dsv4_l20.json").read_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "drift.json"
            path.write_text(json.dumps({"expression": expression.replace(expression.split(",", 1)[1].split("+", 1)[0].strip(), "-68.7", 1)}))
            with self.assertRaisesRegex(AssertionError, "stale mock expression snapshot"):
                assert_snapshot_matches(path, expression)
