"""Ordinary Bazel decode keeps one complete local execution plan."""

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_execution,
    fixture_model,
)


class DecodeRemoteTest(unittest.TestCase):
    def setUp(self):
        module_name = "rtp_llm.test.perf_test.decode_remote_test"
        self.assertIsNotNone(
            importlib.util.find_spec(module_name), "decode wrapper is missing"
        )
        self.wrapper = importlib.import_module(module_name)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.outputs = self.root / "outputs"
        self.source = {
            "schema_version": 1,
            "model": fixture_model(),
            "engine": fixture_engine(),
            "workload": {
                "kind": "grid",
                "input_lengths": [128, 256],
                "output_lengths": [32],
                "batch": {"strategy": "fixed", "sizes": [1, 2]},
                "measure_runs": 1,
                "warmup_runs": 0,
                "generate_config": {"top_k": 1},
            },
            "execution": fixture_execution(),
            "output": {"result_root": str(self.root / "must-not-be-used")},
            "regression": {},
            "publish": {},
        }
        self.plans = []
        self.exit_code = 0

    def execute(self, plan):
        self.plans.append(plan)
        return SimpleNamespace(exit_code=self.exit_code)

    def invoke(self, source=None, outputs=None):
        self.config.write_text(json.dumps(self.source if source is None else source))
        self.errors = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {"TEST_UNDECLARED_OUTPUTS_DIR": str(outputs or self.outputs)},
            ),
            patch(
                "rtp_llm.test.perf_test.execution.local.create_default_local_executor",
                return_value=self.execute,
            ),
            contextlib.redirect_stderr(self.errors),
        ):
            return self.wrapper.main([f"--config={self.config}"])

    def test_one_action_runs_all_cases_as_local_and_preserves_declared_config(self):
        self.assertEqual(0, self.invoke())
        self.assertEqual("", self.errors.getvalue())
        self.assertEqual(1, len(self.plans))
        plan = self.plans[0]
        self.assertEqual(
            ("decode", "run", "local"), (plan.mode, plan.action, plan.backend)
        )
        self.assertEqual(4, len(plan.cases))
        run_dir = Path(plan.result_directory)
        self.assertEqual(self.outputs, run_dir.parent)
        persisted = json.loads((run_dir / "execution_plan.json").read_text())
        self.assertEqual("local", persisted["backend"])
        self.assertEqual(4, len(persisted["cases"]))
        self.assertEqual(self.source, json.loads(self.config.read_text()))
        self.assertFalse((self.root / "must-not-be-used").exists())

    def test_relative_distribution_stays_relative_to_declared_config(self):
        (self.root / "lengths.json").write_text("[128, 256]")
        backing = self.root / "backing"
        backing.mkdir()
        self.config.symlink_to(backing / "source.json")
        self.source["workload"].pop("input_lengths")
        self.source["workload"].pop("output_lengths")
        self.source["workload"]["output_length"] = 32
        self.source["workload"].update(
            kind="distribution", distribution_file="lengths.json"
        )
        self.assertEqual(0, self.invoke())
        self.assertEqual(2, len(self.plans[0].cases))
        self.assertEqual(self.source, json.loads(self.config.read_text()))

    def test_executor_failure_is_test_failure(self):
        self.exit_code = 7
        self.assertEqual(7, self.invoke())
        self.assertEqual(1, len(self.plans))

    def test_nonlocal_config_fails_before_execution(self):
        self.source["execution"]["backend"] = "aone"
        self.assertNotEqual(0, self.invoke())
        self.assertIn("local", self.errors.getvalue())
        self.assertEqual([], self.plans)
        self.assertFalse(self.outputs.exists())

    def test_missing_config_fails_clearly(self):
        errors = io.StringIO()
        with (
            patch.dict(os.environ, {"TEST_UNDECLARED_OUTPUTS_DIR": str(self.outputs)}),
            contextlib.redirect_stderr(errors),
        ):
            self.assertNotEqual(0, self.wrapper.main([f"--config={self.config}"]))
        self.assertIn("config", errors.getvalue())
        self.assertFalse(self.outputs.exists())

    def test_missing_output_environment_fails_clearly(self):
        self.config.write_text(json.dumps(self.source))
        errors = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(errors):
            self.assertNotEqual(0, self.wrapper.main([f"--config={self.config}"]))
        self.assertIn("TEST_UNDECLARED_OUTPUTS_DIR", errors.getvalue())
        self.assertFalse((self.root / "must-not-be-used").exists())

    def test_unavailable_output_directory_fails_before_execution(self):
        self.outputs.write_text("a file cannot hold test artifacts")
        self.assertNotEqual(0, self.invoke())
        self.assertIn(str(self.outputs), self.errors.getvalue())
        self.assertEqual([], self.plans)


class DecodeRemoteMacroTest(unittest.TestCase):
    def test_macro_declares_one_unsharded_gpu_action_with_config_runfile(self):
        declarations = []
        namespace = {
            "native": SimpleNamespace(
                py_test=lambda **kwargs: declarations.append(kwargs)
            )
        }
        exec((Path(__file__).parent / "aone_defs.bzl").read_text(), namespace)
        self.assertIn("unified_decode_remote_test", namespace)
        namespace["unified_decode_remote_test"](
            name="decode",
            config=":config.json",
            gpu_type="H20",
            gpu_count=2,
            data=[":lengths.json"],
            deps=[":extra_lib"],
        )
        self.assertEqual(1, len(declarations))
        action = declarations[0]
        self.assertIn(":config.json", action["data"])
        self.assertIn(":lengths.json", action["data"])
        self.assertIn(":extra_lib", action["deps"])
        self.assertEqual(["--config=$(location :config.json)"], action["args"])
        self.assertEqual({"gpu": "H20", "gpu_count": "2"}, action["exec_properties"])
        self.assertNotIn("shard_count", action)
        self.assertIn("manual", action["tags"])


if __name__ == "__main__":
    unittest.main()
