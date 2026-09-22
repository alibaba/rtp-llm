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

import parallel_runner
from flexlb_eval.runtime import instance_runner


class RepositoryLayoutTest(unittest.TestCase):
    def test_default_routes_to_bundled_python_programs(self):
        with tempfile.TemporaryDirectory() as cwd, mock.patch.object(
            sys, "argv", ["parallel_runner.py", "--dry-run"]
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
            ["parallel_runner.py", "--source", "legacy", "--case-dir", "config/scenarios"],
        ), self.assertRaises(SystemExit) as error:
            parallel_runner.main()
        self.assertEqual(error.exception.code, 2)

    def test_commands_work_outside_repository(self):
        commands = [
            ("scripts/compare_ab.py", ["--help"]),
            ("scripts/compare_twin.py", ["--help"]),
            ("scripts/render_report.py", ["--help"]),
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
        from flexlb_eval.scenario import compile_scenarios, load_scenarios
        from flexlb_eval.scenario.catalog import handlers

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
        self.assertEqual(expected, {p.name for p in (ROOT / "src/flexlb_eval").iterdir() if p.is_dir()})
        for old_package in ("online_eval", "flexlb_test_framework", "stress"):
            self.assertFalse((ROOT / "src" / old_package).exists(), old_package)

    def test_stress_shell_resolves_roots_before_starting_services(self):
        # Stop before sourcing the Java helper, after the real path assignments.
        script = r"""
set -T
trap 'if [[ "$BASH_COMMAND" == source\ *load_client.sh* ]]; then
  printf "%s\n" "$SCRIPT_DIR" "$ONLINE_EVAL_DIR" "$FLEXLB_DIR" "$REPO_ROOT"
  exit 0
fi' DEBUG
source "$1"
"""
        with tempfile.TemporaryDirectory() as cwd:
            for entry in ("scripts/stress/run_online_eval.sh",):
                result = subprocess.run(
                    ["bash", "-c", script, "layout-check", str(ROOT / entry)],
                    cwd=cwd,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout.splitlines(),
                    [
                        str(ROOT / "scripts/stress"),
                        str(ROOT),
                        str(ROOT.parents[1]),
                        str(ROOT.parents[3]),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
