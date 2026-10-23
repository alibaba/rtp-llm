"""CLI ownership tests; child execution is stubbed, so no service is started."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_prefill_traffic_suite as runner


class TrafficCliOwnershipTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "catalog.json"
        self.case = {
            "name": "sample",
            "suites": ["quick", "acceptance"],
            "jitters": ["stable"],
            "seeds": [1],
            "cold_fraction": 0.5,
            "long_fraction": 0.2,
            "session_fraction": 0.5,
            "phases": [{"name": "steady", "seconds": 1, "load": 0.2}],
        }
        self.write_catalog()

    def write_catalog(self):
        self.catalog.write_text(json.dumps({"schema_version": 1, "cases": [self.case]}))

    def argv(self, output, *extras):
        return [
            "traffic-suite",
            "--cases",
            str(self.catalog),
            "--case",
            "sample",
            "--runner",
            "/not-used",
            "--jar",
            "/not-used",
            "--mock-jar",
            "/not-used",
            "--out-dir",
            str(output),
            "--suite",
            "acceptance",
            "--dispatchers",
            "NON_BATCH",
            *extras,
        ]

    def invoke(self, arguments):
        with mock.patch.object(sys, "argv", arguments), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            runner.main()

    def test_duplicate_cli_dimensions_fail_before_directory_or_child_creation(self):
        for option, values in (
            ("--seeds", ["1", "1"]),
            ("--dispatchers", ["NON_BATCH", "NON_BATCH"]),
            ("--jitters", ["stable", "stable"]),
        ):
            with self.subTest(option=option):
                output = self.root / option[2:]
                with mock.patch.object(runner.subprocess, "run") as child:
                    with self.assertRaises(SystemExit) as error:
                        self.invoke(self.argv(output, option, *values))
                    self.assertEqual(2, error.exception.code)
                    child.assert_not_called()
                self.assertFalse(output.exists())

    def test_duplicate_catalog_dimensions_are_also_rejected(self):
        for field, values in (("seeds", [1, 1]), ("jitters", ["stable", "stable"])):
            with self.subTest(field=field):
                self.case["seeds"], self.case["jitters"] = [1], ["stable"]
                self.case[field] = values
                self.write_catalog()
                output = self.root / field
                with mock.patch.object(runner.subprocess, "run") as child:
                    with self.assertRaises(SystemExit):
                        self.invoke(self.argv(output))
                    child.assert_not_called()
                self.assertFalse(output.exists())

    def test_unique_matrix_preserves_all_requested_combinations(self):
        output = self.root / "matrix"
        with mock.patch.object(
            runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ) as child:
            self.invoke(
                self.argv(
                    output, "--seeds", "1", "2", "--dispatchers", "NON_BATCH", "BATCH"
                )
            )
        self.assertEqual(4, child.call_count)
        results = json.loads((output / "results.json").read_text())
        self.assertEqual(4, len(results))
        self.assertEqual(4, len({row["directory"] for row in results}))

    def test_children_use_saved_snapshot_when_original_catalog_changes(self):
        output = self.root / "snapshot-matrix"
        observed_loads = []

        def execute(command, **kwargs):
            path = Path(command[command.index("--cases") + 1])
            self.assertEqual((output / "cases.json").resolve(), path)
            saved = json.loads(path.read_text())
            observed_loads.append(saved["cases"][0]["phases"][0]["load"])
            self.case["phases"][0]["load"] = 0.9
            self.write_catalog()
            return SimpleNamespace(returncode=0)

        with mock.patch.object(runner.subprocess, "run", side_effect=execute):
            self.invoke(self.argv(output, "--seeds", "1", "2"))
        self.assertEqual([0.2, 0.2], observed_loads)
        self.assertEqual(
            0.9,
            json.loads(self.catalog.read_text())["cases"][0]["phases"][0]["load"],
        )

    def test_existing_child_directory_keeps_all_prior_evidence_unchanged(self):
        output = self.root / "existing"
        output.mkdir()
        old = b'{"prior_evidence":true}\n'
        (output / "failure.json").write_bytes(old)
        (output / "report.json").write_bytes(b'{"status":"PASS"}\n')
        with mock.patch.object(runner, "run_case") as execute:
            with self.assertRaises(FileExistsError):
                self.invoke(self.argv(output, "--run-case", "sample"))
            execute.assert_not_called()
        self.assertEqual(old, (output / "failure.json").read_bytes())
        self.assertEqual(b'{"status":"PASS"}\n', (output / "report.json").read_bytes())

    def test_existing_parent_matrix_is_not_overwritten(self):
        output = self.root / "matrix-existing"
        output.mkdir()
        old = b'{"prior_cases":true}\n'
        (output / "cases.json").write_bytes(old)
        with mock.patch.object(runner.subprocess, "run") as child:
            with self.assertRaises(FileExistsError):
                self.invoke(self.argv(output))
            child.assert_not_called()
        self.assertEqual(old, (output / "cases.json").read_bytes())

    def test_failed_child_writes_failure_only_into_its_new_directory(self):
        output = self.root / "new-failure"

        def fail(args, case):
            self.assertTrue(output.is_dir())
            raise RuntimeError("deliberate case failure")

        with mock.patch.object(runner, "run_case", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "deliberate case failure"):
                self.invoke(self.argv(output, "--run-case", "sample"))
        failure = json.loads((output / "failure.json").read_text())
        self.assertIn("deliberate case failure", failure["error"])

    def test_successful_child_receives_new_directory_without_failure_artifact(self):
        output = self.root / "new-success"
        with mock.patch.object(runner, "run_case") as execute:
            self.invoke(self.argv(output, "--run-case", "sample"))
            execute.assert_called_once()
        self.assertTrue(output.is_dir())
        self.assertFalse((output / "failure.json").exists())

    def test_competing_creator_does_not_transfer_directory_ownership(self):
        output = self.root / "racing-directory"
        original_mkdir = Path.mkdir
        old = b'{"created_by_other_process":true}\n'

        def competing_mkdir(path, *args, **kwargs):
            original_mkdir(path, *args, **kwargs)
            (path / "failure.json").write_bytes(old)
            raise FileExistsError("competing process created this directory")

        with mock.patch.object(Path, "mkdir", competing_mkdir), mock.patch.object(
            runner, "run_case"
        ) as execute:
            with self.assertRaises(FileExistsError):
                self.invoke(self.argv(output, "--run-case", "sample"))
            execute.assert_not_called()
        self.assertEqual(old, (output / "failure.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
