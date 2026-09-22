"""Default routing and relocated command paths remain usable from any cwd."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pipeline import execute_cases as parallel_runner
from runtime import instance_runner


class RepositoryLayoutTest(unittest.TestCase):
    def test_default_routes_to_bundled_python_programs(self):
        with tempfile.TemporaryDirectory() as cwd, mock.patch.object(
            sys, "argv", ["scripts/pipeline/execute_cases.py", "--dry-run"]
        ), mock.patch.object(
            instance_runner, "run_structured", return_value=0
        ) as run, mock.patch.object(
            parallel_runner, "_mock_base", return_value=55151
        ), mock.patch.object(
            parallel_runner, "_master_base", return_value=18080
        ):
            with mock.patch("os.getcwd", return_value=cwd):
                self.assertEqual(parallel_runner.main(), 0)
            args, _ = run.call_args.args
            self.assertEqual(args.source, "yaml")
            self.assertEqual(Path(args.case_dir), ROOT / "config/scenarios")
            self.assertEqual(args.shard, "case")

    def test_removed_source_is_rejected(self):
        with mock.patch.object(
            sys,
            "argv",
            ["scripts/pipeline/execute_cases.py", "--source", "legacy", "--case-dir", "config/scenarios"],
        ), self.assertRaises(SystemExit) as error:
            parallel_runner.main()
        self.assertEqual(error.exception.code, 2)

    def test_commands_work_outside_repository(self):
        commands = [
            ("scripts/commands/run_stress.py", ["--help"]),
            ("scripts/commands/run_cases.py", ["--help"]),
            ("scripts/commands/list_cases.py", ["--help"]),
            ("scripts/commands/compare_runs.py", ["--help"]),
            ("scripts/probes/check_mock_fidelity.py", ["--help"]),
            ("scripts/commands/render_stress_report.py", ["--help"]),
        ]
        with tempfile.TemporaryDirectory() as cwd:
            for command, arguments in commands:
                with self.subTest(command=command):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / command), *arguments],
                        cwd=cwd,
                        text=True,
                        capture_output=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(result.stdout.strip())

    def test_inventory_stays_complete_without_retired_modules(self):
        from scenario import compile_scenarios, load_scenarios
        from scenario.catalog import handlers

        with mock.patch("scenario.compiler.VICTIM_OFFSETS", (700, 701, 702)):
            plans = compile_scenarios(
                load_scenarios(ROOT / "config/scenarios"), handlers=handlers()
            )
        expected = json.loads((ROOT / "tests/fixtures/instance_ids.json").read_text())
        self.assertEqual(expected, sorted(p["id"] for p in plans))
        for path in [
            "legacy",
            "migration",
            "archive",
            "flexlb_test_framework/cases",
            "flexlb_test_framework/support",
            "flexlb_functional_tests.py",
            "paired_case_runner.py",
            "legacy_paired_runner.py",
        ]:
            self.assertFalse((ROOT / path).exists(), path)
        self.assertFalse(
            any(name.startswith("flexlb_test_framework.cases") for name in sys.modules)
        )

    def test_source_components_are_unified(self):
        expected = {
            "monitoring", "traffic", "runtime", "cases", "scenario",
            "workload", "analysis", "reporting", "artifacts",
        }
        self.assertEqual(expected, {p.name for p in (ROOT / "src").iterdir() if p.is_dir()})
        for old_package in ("online_eval", "flexlb_test_framework", "stress", "flexlb_eval"):
            self.assertFalse((ROOT / "src" / old_package).exists(), old_package)
        self.assertFalse((ROOT / "scenarios").exists())
        self.assertTrue((ROOT / "config/scenarios").is_dir())

    def test_stress_entry_resolves_roots_outside_repository(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/commands/run_stress.py"), "--dry-run"],
                cwd=cwd, text=True, capture_output=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["mode"]["runtime"], "stress")


if __name__ == "__main__":
    unittest.main()
