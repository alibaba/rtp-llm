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
from flexlb_test_framework import instance_runner


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
            self.assertEqual(Path(args.case_dir), ROOT / "scenarios")
            self.assertEqual(args.shard, "case")

    def test_removed_source_is_rejected(self):
        with mock.patch.object(
            sys,
            "argv",
            ["parallel_runner.py", "--source", "legacy", "--case-dir", "scenarios"],
        ), self.assertRaises(SystemExit) as error:
            parallel_runner.main()
        self.assertEqual(error.exception.code, 2)

    def test_canonical_and_compatibility_commands_work_outside_repository(self):
        pairs = [
            ("stress/compare_ab.py", "compare_ab.py", ["--help"]),
            ("stress/compare_twin.py", "compare_twin.py", ["--help"]),
            ("stress/canvas_report_gen.py", "canvas_report_gen.py", ["--help"]),
        ]
        with tempfile.TemporaryDirectory() as cwd:
            for canonical, alias, arguments in pairs:
                with self.subTest(command=canonical):
                    target = ROOT / alias
                    self.assertTrue(target.is_symlink())
                    self.assertEqual(target.resolve(), ROOT / canonical)
                    for name in (canonical, alias):
                        result = subprocess.run(
                            [sys.executable, str(ROOT / name), *arguments],
                            cwd=cwd,
                            text=True,
                            capture_output=True,
                            timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertTrue(result.stdout.strip())

    def test_inventory_stays_complete_without_retired_modules(self):
        from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
        from flexlb_test_framework.scenario.catalog import handlers

        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios"), handlers=handlers()
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

    def test_stress_shell_resolves_roots_before_starting_services(self):
        # Stop before sourcing the Java helper, after the real path assignments.
        script = r"""
set -T
trap 'if [[ "$BASH_COMMAND" == source\ *lib_load_client.sh* ]]; then
  printf "%s\n" "$SCRIPT_DIR" "$ONLINE_EVAL_DIR" "$FLEXLB_DIR" "$REPO_ROOT"
  exit 0
fi' DEBUG
source "$1"
"""
        with tempfile.TemporaryDirectory() as cwd:
            for entry in ("run_online_eval.sh", "stress/run_online_eval.sh"):
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
                        str(ROOT / "stress"),
                        str(ROOT),
                        str(ROOT.parents[1]),
                        str(ROOT.parents[3]),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
