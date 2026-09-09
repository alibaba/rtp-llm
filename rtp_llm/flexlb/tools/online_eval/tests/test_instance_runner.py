"""CLI protocol/aggregation tests with fixture children, never mock engine load."""

import argparse
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parallel_runner as ports
from flexlb_test_framework import instance_runner as runner
from flexlb_test_framework.instance_plan import parse_catalog
from test_instance_plan import catalog


class InstanceRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.child = self.root / "scenario_runner.py"
        self.child.touch()
        self.data = catalog()
        patches = [
            mock.patch.object(ports, "PORT_WINDOW_LOCK_DIR", self.root / "locks"),
            mock.patch.object(ports, "port_in_use", return_value=False),
            mock.patch.object(runner, "SCENARIO_RUNNER", self.child),
            mock.patch.dict(
                os.environ,
                {"FLEXLB_FT_INSTANCE_TIMING_BASELINE": str(self.root / "timing.json")},
                clear=True,
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.drop_locks)

    def drop_locks(self):
        ports._close_window_locks(ports._WINDOW_LOCK_FILES)
        ports._WINDOW_LOCK_FILES = None
        if ports._RUN_LOCK_FILE is not None:
            ports._RUN_LOCK_FILE.close()
            ports._RUN_LOCK_FILE = None

    def args(self, **kwargs):
        values = dict(
            source="yaml",
            case_dir=str(self.root / "scenarios"),
            instances=None,
            categories=None,
            profile="batch-window",
            parallel=2,
            mock_stride=500,
            dry_run=False,
            out_dir=str(self.root / "output"),
            json=None,
            grade="normal",
            timing_json=None,
        )
        values.update(kwargs)
        return argparse.Namespace(**values)

    def fixture_child(self, command, env, log_path, **kwargs):
        self.assertEqual(
            5,
            len(
                [
                    key
                    for key in env
                    if key.startswith("FLEXLB_FT_") and key.endswith("PORT")
                ]
            ),
        )
        if "--instances" in command:
            lease_path = Path(command[command.index("--lease-json") + 1])
            lease = json.loads(lease_path.read_text())
            self.assertEqual(1, lease["schema_version"])
            self.assertEqual(
                lease["child_env"], {key: env[key] for key in lease["child_env"]}
            )
            selected = command[command.index("--instances") + 1].split(",")
            rows = [
                {
                    **row,
                    "status": "PASS",
                    "duration_ms": 100,
                    "stages": [
                        {
                            "id": "check",
                            "status": "PASS",
                            "checks": [{"id": "done", "status": "PASS"}],
                        }
                    ],
                    "cleanup": [],
                    "error": None,
                }
                for row in self.data["instances"]
                if row["id"] in selected
            ]
            path = Path(command[command.index("--out-dir") + 1]) / "scenarios.json"
            payload = {
                "schema_version": 1,
                "summary": {"exit_code": 0},
                "instances": rows,
            }
        path.write_text(json.dumps(payload))
        return 0

    def run_fixture(self, args, spawn=None):
        for row in self.data["instances"]:
            row["grade"] = args.grade
        listed = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(self.data), stderr=""
        )
        with mock.patch.object(
            runner.subprocess, "run", return_value=listed
        ) as listing, mock.patch.object(
            runner.ChildProcesses, "run", side_effect=spawn or self.fixture_child
        ) as child, contextlib.redirect_stdout(
            io.StringIO()
        ):
            rc = runner.run_structured(args, ports)
        return rc, listing, child

    def test_real_child_compiler_and_runtime_use_selected_grade(self):
        # The real child CLI/compiler/runtime run in a subprocess. Only Java
        # process startup is replaced; this is a protocol fixture, not a JVM test.
        module_root = Path(__file__).resolve().parents[1]
        self.child.write_text(
            "import sys\nsys.path.insert(0, "
            + repr(str(module_root))
            + ")\n"
            + """
from types import SimpleNamespace, ModuleType
import scenario_runner as child
import flexlb_test_framework.scenario.backend as backend
from flexlb_test_framework.scenario.contracts import StageHandler, StageOutput, CheckResult
class FixtureBackend:
    def __init__(self, lease): pass
    def setup(self, ctx, environment, deadline): return SimpleNamespace(), None
    def teardown(self, ctx, deadline): pass
backend.JavaMockBackend = FixtureBackend
def validate(params, plan): return params
def evaluate(ctx, params, deadline):
    grade = ctx.instance['grade']
    status = 'FAIL' if grade == 'strict' else 'PASS'
    return StageOutput({'observed': grade}, [CheckResult('selected', status, actual=grade)])
child.handlers = lambda: {'grade_probe': StageHandler('grade_probe', validate, evaluate, {'observed': 'string'}, checks=frozenset({'selected'}))}
from flexlb_test_framework.case_programs import PROGRAMS
fixture = ModuleType('grade_protocol_fixture')
fixture.__file__ = __file__
def build(case):
    for name, action in [('setup', 'setup'), ('probe', 'grade_probe'), ('cleanup', 'teardown')]:
        case.step(name, action)
build.__module__ = fixture.__name__
fixture.default = build
sys.modules[fixture.__name__] = fixture
PROGRAMS['grade_protocol'] = fixture.__name__
raise SystemExit(child.main())
"""
        )
        source = self.root / "scenarios"
        source.mkdir()
        (source / "grade.yaml").write_text(
            """
schema_version: 2
case: grade_protocol
metadata: {description: Grade protocol fixture, category: status}
profiles: [batch-window]
environment: {backend: java_mock, n_prefill: 1, n_decode: 1}
variants: [{id: default}]
"""
        )
        for grade, expected in (("strict", "FAIL"), ("loose", "PASS")):
            with self.subTest(grade=grade):
                self.drop_locks()
                output = self.root / grade
                args = self.args(grade=grade, parallel=1, out_dir=str(output))
                args.dry_run = True
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    self.assertEqual(0, runner.run_structured(args, ports))
                manifest = json.loads(
                    captured.getvalue()[captured.getvalue().index("{") :]
                )
                self.assertEqual(grade, manifest["grade"])
                self.assertEqual(grade, manifest["instances"][0]["grade"])
                self.drop_locks()
                args.dry_run = False
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = runner.run_structured(args, ports)
                self.assertEqual(int(expected == "FAIL"), rc)
                result = json.loads((output / "aggregate.json").read_text())
                row = result["instances"][0]
                self.assertEqual(expected, row["status"])
                self.assertEqual(grade, row["grade"])
                self.assertEqual(grade, row["stages"][1]["checks"][0]["actual"])

    def test_same_id_from_wrong_grade_cannot_be_reused_as_success(self):
        def wrong_grade(command, env, log, **kwargs):
            self.fixture_child(command, env, log, **kwargs)
            path = Path(command[command.index("--out-dir") + 1]) / "scenarios.json"
            data = json.loads(path.read_text())
            for row in data["instances"]:
                row["grade"] = "loose"
            path.write_text(json.dumps(data))
            return 0

        rc, _, _ = self.run_fixture(self.args(grade="strict"), spawn=wrong_grade)
        self.assertEqual(1, rc)
        result = json.loads((self.root / "output/aggregate.json").read_text())
        self.assertTrue(all(row["status"] == "ERROR" for row in result["instances"]))
        self.assertTrue(
            all("grade mismatch" in row["error"] for row in result["instances"])
        )

    def test_grade_is_identical_for_listing_execution_and_manifest(self):
        for grade in ("strict", "normal", "loose"):
            with self.subTest(grade=grade):
                self.drop_locks()
                rc, listing, child = self.run_fixture(self.args(grade=grade))
                self.assertEqual(0, rc)
                command = listing.call_args.args[0]
                self.assertEqual(grade, command[command.index("--grade") + 1])
                for call in child.call_args_list:
                    command = call.args[0]
                    self.assertEqual(grade, command[command.index("--grade") + 1])
                manifest = json.loads((self.root / "output/manifest.json").read_text())
                aggregate = json.loads(
                    (self.root / "output/aggregate.json").read_text()
                )
                self.assertEqual(grade, manifest["grade"])
                self.assertEqual(grade, aggregate["summary"]["grade"])

    def test_fixture_child_protocol_and_manifest_match_lane_environment(self):
        rc, listing, child = self.run_fixture(self.args())
        self.assertEqual(0, rc)
        command = listing.call_args.args[0]
        self.assertIn("--list-json", command)
        self.assertEqual(
            str((self.root / "scenarios").resolve()),
            command[command.index("--source") + 1],
        )
        doc = json.loads((self.root / "output/manifest.json").read_text())
        result = json.loads((self.root / "output/aggregate.json").read_text())
        self.assertEqual(7, result["summary"]["total"])
        self.assertCountEqual(
            [row["id"] for row in self.data["instances"]],
            [row["id"] for row in result["instances"]],
        )
        for call in child.call_args_list:
            command, env, _ = call.args
            self.assertEqual(str(self.child), command[1])
            lease = next(
                row
                for row in doc["lanes"]
                if row["child_env"]["FLEXLB_FT_MASTER_HTTP_PORT"]
                == env["FLEXLB_FT_MASTER_HTTP_PORT"]
            )
            self.assertEqual(
                lease["child_env"], {key: env[key] for key in lease["child_env"]}
            )
        timings = json.loads((self.root / "timing.json").read_text())
        self.assertEqual(1, timings["schema_version"])
        self.assertEqual(7, len(timings["instances"]))

    def test_dry_run_never_creates_results_or_starts_child(self):
        rc, _, child = self.run_fixture(self.args(dry_run=True))
        self.assertEqual(0, rc)
        child.assert_not_called()
        self.assertFalse((self.root / "output").exists())
        self.assertFalse((self.root / "timing.json").exists())
        self.assertIsNone(ports._WINDOW_LOCK_FILES)

    def test_invalid_budget_fails_before_child_start(self):
        self.data["instances"][0]["resource_budget"]["max_dynamic_additions"] = 200
        with contextlib.redirect_stderr(io.StringIO()):
            rc, _, child = self.run_fixture(self.args())
        self.assertEqual(2, rc)
        child.assert_not_called()

    def test_missing_duplicate_and_unexpected_results_are_errors(self):
        rows = parse_catalog(self.data, source="yaml", profile="batch-window")[:2]
        path = self.root / "result.json"
        for fixture in [
            [],
            [rows[0].metadata] * 2,
            [{"id": "unselected", "status": "PASS"}],
        ]:
            path.write_text(json.dumps({"schema_version": 1, "instances": fixture}))
            result = runner._read_results(path, rows)
            self.assertEqual(2, len(result))
            self.assertTrue(all(row["status"] == "ERROR" for row in result))

    def test_cleanup_error_cannot_be_swallowed_as_finding(self):
        rows = parse_catalog(self.data, source="yaml", profile="batch-window")[:1]
        path = self.root / "result.json"
        payload = {
            **rows[0].metadata,
            "status": "FINDING-CONFIRMED",
            "cleanup": [{"status": "ERROR", "error": "leaked resource"}],
            "stages": [],
        }
        path.write_text(json.dumps({"schema_version": 1, "instances": [payload]}))
        result = runner._read_results(path, rows)
        self.assertEqual("ERROR", result[0]["status"])
        self.assertEqual("FINDING-CONFIRMED", result[0]["original_status"])

    def test_empty_or_unexecuted_checks_cannot_report_success(self):
        rows = parse_catalog(self.data, source="yaml", profile="batch-window")[:1]
        path = self.root / "result.json"
        for stages in (
            [],
            [{"status": "PASS", "checks": []}],
            [{"status": "SKIP", "checks": [{"status": "PASS"}]}],
            [{"status": "BLOCKED", "checks": []}],
        ):
            payload = {**rows[0].metadata, "status": "PASS", "stages": stages}
            path.write_text(json.dumps({"schema_version": 1, "instances": [payload]}))
            result = runner._read_results(path, rows)
            self.assertEqual("ERROR", result[0]["status"])
            self.assertTrue(result[0]["error"])

    def test_nested_execution_errors_cannot_be_hidden_by_green_rows(self):
        group = parse_catalog(self.data, source="yaml", profile="batch-window")[:1]
        path = self.root / "nested.json"
        base = {
            **group[0].metadata,
            "status": "PASS",
            "error": None,
            "stages": [
                {
                    "id": "verify",
                    "status": "PASS",
                    "checks": [{"id": "ok", "status": "PASS"}],
                }
            ],
            "cleanup": [],
        }
        mutations = []
        for status in ("FAIL", "ERROR", "TIMEOUT", "BLOCKED", "UNKNOWN", None):
            payload = copy.deepcopy(base)
            payload["cleanup"] = [
                {"id": "environment", "status": status, "error": None}
            ]
            mutations.append(payload)
        for status in ("ERROR", "TIMEOUT", "UNKNOWN", "FAIL"):
            payload = copy.deepcopy(base)
            payload["stages"][0]["checks"].append({"id": "bad", "status": status})
            mutations.append(payload)
        for payload in mutations:
            with self.subTest(payload=payload):
                path.write_text(
                    json.dumps({"schema_version": 1, "instances": [payload]})
                )
                rows = runner._read_results(path, group)
                aggregate = runner._aggregate(
                    [{"lane": 0, "instances": rows, "segments": [{"exit_code": 0}]}],
                    group,
                    self.args(),
                    1,
                )
                self.assertEqual("ERROR", rows[0]["status"])
                self.assertEqual(1, aggregate["summary"]["exit_code"])

    def test_only_matching_normal_fail_checks_can_be_findings(self):
        group = parse_catalog(self.data, source="yaml", profile="batch-window")[:1]
        path = self.root / "finding.json"
        payload = {
            **group[0].metadata,
            "status": "FINDING-CONFIRMED",
            "cleanup": [],
            "stages": [
                {
                    "id": "verify",
                    "status": "FAIL",
                    "checks": [{"id": "known", "status": "FAIL"}],
                }
            ],
            "finding_confirmed": ["verify.known"],
        }
        path.write_text(json.dumps({"schema_version": 1, "instances": [payload]}))
        self.assertEqual(
            "FINDING-CONFIRMED", runner._read_results(path, group)[0]["status"]
        )
        for bad in ("ERROR", "TIMEOUT"):
            broken = copy.deepcopy(payload)
            broken["stages"][0]["checks"].append({"id": "bad", "status": bad})
            path.write_text(json.dumps({"schema_version": 1, "instances": [broken]}))
            self.assertEqual("ERROR", runner._read_results(path, group)[0]["status"])

    def test_child_nonzero_exit_cannot_be_hidden_by_pass_rows(self):
        def fail(command, env, log, **kwargs):
            self.fixture_child(command, env, log)
            return 7

        rc, _, _ = self.run_fixture(self.args(), spawn=fail)
        self.assertEqual(1, rc)

    def test_second_output_owner_fails_before_catalog(self):
        args = self.args()
        Path(args.out_dir).mkdir()
        ports._acquire_run_lock(Path(args.out_dir))
        with mock.patch.object(runner, "_catalog") as listing:
            with self.assertRaisesRegex(SystemExit, "held by another"):
                runner.run_structured(args, ports)
            listing.assert_not_called()


class ChildProcessesTest(unittest.TestCase):
    def test_watchdog_returns_timeout_and_reaps_owned_child(self):
        children = runner.ChildProcesses()
        with tempfile.TemporaryDirectory() as tmp:
            rc = children.run(
                [
                    sys.executable,
                    "-c",
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
                ],
                dict(os.environ),
                Path(tmp) / "child.log",
                timeout=0.2,
                cleanup_timeout=0.05,
            )
        self.assertEqual(124, rc)
        self.assertFalse(children.processes)

    def test_stop_terminates_owned_child_and_refuses_further_launch(self):
        children = runner.ChildProcesses()
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(
            max_workers=1
        ) as pool:
            log = Path(tmp) / "child.log"
            future = pool.submit(
                children.run,
                [sys.executable, "-c", "import time; time.sleep(60)"],
                dict(os.environ),
                log,
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with children.lock:
                    if children.processes:
                        break
                time.sleep(0.01)
            children.stop()
            self.assertNotEqual(0, future.result(timeout=5))
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                children.run([sys.executable, "-c", "pass"], dict(os.environ), log)


if __name__ == "__main__":
    unittest.main()
