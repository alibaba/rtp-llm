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
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters import port_windows as ports
from scenario import instance_runner as runner
from scenario import result_validation, timing_store
from scenario.instance_plan import parse_catalog
from test_instance_plan import catalog


class InstanceRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.child = self.root / "scripts/commands/list_cases.py"
        self.child.parent.mkdir(parents=True)
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

    def args(self, **kwargs):
        values = dict(
            source="yaml",
            suite=None,
            kind=None,
            all=True,
            case_dir=str(self.root / "config/scenarios"),
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
            self.assertEqual(1, lease["lease_schema_version"])
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
                "scenario_results_schema_version": 1,
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
        output = io.StringIO()
        with mock.patch.object(
            runner.subprocess, "run", return_value=listed
        ) as listing, mock.patch.object(
            runner.ChildProcesses, "run", side_effect=spawn or self.fixture_child
        ) as child, contextlib.redirect_stdout(
            output
        ):
            rc = runner.run_structured(args)
        self.stdout = output.getvalue()
        return rc, listing, child

    def test_workload_report_views_are_previewed_and_all_paths_are_printed(self):
        identity = self.data["instances"][0]["id"]
        from document_io import load_document

        source = Path(__file__).resolve().parents[1] / "config/scenarios/master_ha_failover.yaml"
        names = load_document(source)["reports"]
        self.assertEqual(["default.yaml", "master_ha_failover.yaml"], names)
        self.data["instances"][0].update(
            test_kind="workload",
            reports=names,
        )
        dry = self.args(dry_run=True, instances=identity, parallel=1)
        rc, _, child = self.run_fixture(dry)
        self.assertEqual(0, rc)
        child.assert_not_called()
        self.assertIn(f"planned reports {identity}: default.yaml, master_ha_failover.yaml", self.stdout)

        paths = {
            "default.yaml": "/tmp/run/reports/run/default/report.html",
            "master_ha_failover.yaml": "/tmp/run/reports/run/ha-core/report.html",
        }

        def completed_child(command, env, log, **kwargs):
            self.fixture_child(command, env, log, **kwargs)
            path = Path(command[command.index("--out-dir") + 1]) / "scenarios.json"
            result = json.loads(path.read_text())
            result["instances"][0]["workload"] = {
                "telemetry_gaps": {"large": "x" * 5000}, "reports": paths,
            }
            path.write_text(json.dumps(result))
            return 0

        target = self.root / "output/selected-aggregate.json"
        rc, _, _ = self.run_fixture(
            self.args(instances=identity, parallel=1, json=str(target)), spawn=completed_child
        )
        self.assertEqual(0, rc)
        lines = [line.removeprefix("workload reports: ") for line in self.stdout.splitlines()
                 if line.startswith("workload reports: ")]
        self.assertEqual(1, len(lines))
        self.assertEqual({"instance": identity, "status": "PASS", "reports": paths}, json.loads(lines[0]))
        self.assertEqual(paths, json.loads(target.read_text())["instances"][0]["workload"]["reports"])

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
from scripts.commands import list_cases as child
import scenario.backend as backend
from scenario.contracts import StageHandler, StageOutput, CheckResult
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
from case_api import registry
fixture = ModuleType('grade_protocol_fixture')
fixture.__file__ = __file__
def build(case):
    for name, action in [('setup', 'setup'), ('probe', 'grade_probe'), ('cleanup', 'teardown')]:
        case.step(name, action)
build.__module__ = fixture.__name__
fixture.default = build
sys.modules[fixture.__name__] = fixture
definition = registry.CaseDefinition({'default': build})
registry._snapshot = registry.CaseRegistry.build([registry.RegisteredCase('grade_protocol', fixture.__name__, __import__('pathlib').Path(__file__), definition)])
raise SystemExit(child.main())
"""
        )
        source = self.root / "config/scenarios"
        source.mkdir(parents=True)
        (source / "grade.yaml").write_text(
            """
case_schema_version: 2
case: grade_protocol
metadata: {kind: functional, description: Grade protocol fixture, category: status}
execution: {collection: diagnostic}
profiles: [batch-window]
environment: {backend: java_mock, n_prefill: 1, n_decode: 1}
program: default
"""
        )
        for grade, expected in (("strict", "FAIL"), ("loose", "PASS")):
            with self.subTest(grade=grade):
                output = self.root / grade
                args = self.args(grade=grade, parallel=1, out_dir=str(output))
                args.dry_run = True
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    self.assertEqual(0, runner.run_structured(args))
                manifest = json.loads(
                    captured.getvalue()[captured.getvalue().index("{") :]
                )
                self.assertEqual(grade, manifest["grade"])
                self.assertEqual(grade, manifest["instances"][0]["grade"])
                args.dry_run = False
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = runner.run_structured(args)
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
            str((self.root / "config/scenarios").resolve()),
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
        self.assertEqual(1, timings["timings_schema_version"])
        self.assertEqual(7, len(timings["instances"]))

    def test_dry_run_never_creates_results_or_starts_child(self):
        rc, _, child = self.run_fixture(self.args(dry_run=True))
        self.assertEqual(0, rc)
        child.assert_not_called()
        self.assertFalse((self.root / "output").exists())
        self.assertFalse((self.root / "timing.json").exists())
        holders = ports._try_window_lock(18080, 55151, 500, 2)
        self.assertIsNotNone(holders)
        ports._close_window_locks(holders)

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
            path.write_text(json.dumps({"scenario_results_schema_version": 1, "instances": fixture}))
            result = result_validation.read_results(path, rows)
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
        path.write_text(json.dumps({"scenario_results_schema_version": 1, "instances": [payload]}))
        result = result_validation.read_results(path, rows)
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
            path.write_text(json.dumps({"scenario_results_schema_version": 1, "instances": [payload]}))
            result = result_validation.read_results(path, rows)
            self.assertEqual("ERROR", result[0]["status"])
            self.assertTrue(result[0]["error"])

    def test_diagnostic_skip_is_valid_but_does_not_replace_executed_checks(self):
        group = parse_catalog(self.data, source="yaml", profile="batch-window")[:1]
        path = self.root / "diagnostic.json"
        diagnostic = {
            "id": "retired",
            "status": "PASS",
            "checks": [
                {"id": "diagnostic", "status": "SKIP", "detail": "retired contract"}
            ],
        }
        behavioral = {
            "id": "completion",
            "status": "PASS",
            "checks": [{"id": "complete", "status": "PASS"}],
        }
        for stages, expected in (
            ([diagnostic, behavioral], "PASS"),
            ([diagnostic], "ERROR"),
        ):
            payload = {
                **group[0].metadata,
                "status": "PASS",
                "stages": stages,
                "cleanup": [],
            }
            path.write_text(json.dumps({"scenario_results_schema_version": 1, "instances": [payload]}))
            result = result_validation.read_results(path, group)
            self.assertEqual(expected, result[0]["status"])

    def test_warning_checks_remain_visible_without_failing_aggregate(self):
        group = parse_catalog(self.data, source="yaml", profile="batch-window")[:1]
        path = self.root / "warning.json"
        check = {"id": "known_error", "status": "WARNING", "actual": 2}
        base = {
            **group[0].metadata,
            "status": "PASS",
            "stages": [{"id": "handover", "status": "PASS", "checks": [check]}],
            "cleanup": [],
        }
        for execution_error, expected in ((None, "PASS"), ("query failed", "ERROR")):
            payload = copy.deepcopy(base)
            payload["stages"][0]["checks"][0]["error"] = execution_error
            path.write_text(json.dumps({"scenario_results_schema_version": 1, "instances": [payload]}))
            rows = result_validation.read_results(path, group)
            self.assertEqual(expected, rows[0]["status"])
            self.assertEqual("WARNING", rows[0]["stages"][0]["checks"][0]["status"])
            aggregate = runner._aggregate(
                [{"lane": 0, "instances": rows, "segments": [{"exit_code": 0}]}],
                group, self.args(), 1,
            )
            self.assertEqual(int(expected == "ERROR"), aggregate["summary"]["exit_code"])

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
                    json.dumps({"scenario_results_schema_version": 1, "instances": [payload]})
                )
                rows = result_validation.read_results(path, group)
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
        path.write_text(json.dumps({"scenario_results_schema_version": 1, "instances": [payload]}))
        self.assertEqual(
            "FINDING-CONFIRMED", result_validation.read_results(path, group)[0]["status"]
        )
        for bad in ("ERROR", "TIMEOUT"):
            broken = copy.deepcopy(payload)
            broken["stages"][0]["checks"].append({"id": "bad", "status": bad})
            path.write_text(json.dumps({"scenario_results_schema_version": 1, "instances": [broken]}))
            self.assertEqual("ERROR", result_validation.read_results(path, group)[0]["status"])

    def test_child_nonzero_exit_cannot_be_hidden_by_pass_rows(self):
        def fail(command, env, log, **kwargs):
            self.fixture_child(command, env, log)
            return 7

        rc, _, _ = self.run_fixture(self.args(), spawn=fail)
        self.assertEqual(1, rc)

    def test_second_output_owner_fails_before_catalog(self):
        args = self.args()
        Path(args.out_dir).mkdir()
        with ports.acquire_run_lock(Path(args.out_dir)):
            with mock.patch.object(runner, "_catalog") as listing:
                with self.assertRaisesRegex(ports.PortAdmissionError, "held by another"):
                    runner.run_structured(args)
                listing.assert_not_called()


    def test_dynamic_idle_lane_claims_work_before_slow_lane_finishes(self):
        instances = parse_catalog(self.data, source="yaml", profile="batch-window")
        slow, fast, third = runner.order_instances(instances)[:3]
        third_started = threading.Event()
        assignments = {}
        mutex = threading.Lock()

        def spawn(command, env, log, **kwargs):
            identity = command[command.index("--instances") + 1]
            with mutex:
                assignments[identity] = env["FLEXLB_FT_MASTER_HTTP_PORT"]
            if identity == slow.id:
                self.assertTrue(third_started.wait(5), "idle lane did not claim another run")
            elif identity == third.id:
                third_started.set()
            return self.fixture_child(command, env, log, **kwargs)

        rc, _, child = self.run_fixture(self.args(), spawn=spawn)
        self.assertEqual(0, rc)
        self.assertEqual(len(instances), child.call_count)
        self.assertEqual(assignments[fast.id], assignments[third.id])
        self.assertNotEqual(assignments[slow.id], assignments[third.id])
        manifest = json.loads((self.root / "output/manifest.json").read_text())
        self.assertEqual("dynamic", manifest["scheduling"])
        self.assertTrue(all(not lane["instance_ids"] for lane in manifest["lanes"]))
        self.assertEqual([row.id for row in runner.order_instances(instances)], manifest["dispatch_queue"])
        for lane in manifest["lanes"]:
            self.assertEqual(max(row.budget.worker_capacity for row in instances), lane["worker_capacity"])

    def test_all_failed_cleanup_lanes_retire_and_account_for_every_pending_run(self):
        def spawn(command, env, log, **kwargs):
            self.fixture_child(command, env, log, **kwargs)
            path = Path(command[command.index("--out-dir") + 1]) / "scenarios.json"
            result = json.loads(path.read_text())
            result["instances"][0]["cleanup"] = [dict(id="environment", status="ERROR", error="leaked")]
            path.write_text(json.dumps(result))
            return 1
        rc, _, child = self.run_fixture(self.args(), spawn=spawn)
        self.assertEqual(1, rc)
        self.assertEqual(2, child.call_count)
        result = json.loads((self.root / "output/aggregate.json").read_text())
        self.assertEqual(7, result["summary"]["total"])
        self.assertEqual(5, result["summary"]["blocked"])
        self.assertTrue(all(not lane["reusable"] for lane in result["lanes"]))

    def test_static_lane_stops_on_occupied_port_after_completed_run(self):
        occupied = set()
        def probe(port, host=None):
            return port in occupied
        def spawn(command, env, log, **kwargs):
            rc = self.fixture_child(command, env, log, **kwargs)
            occupied.add(int(env["FLEXLB_FT_MASTER_HTTP_PORT"]))
            return rc
        with mock.patch.object(ports, "port_in_use", side_effect=probe):
            rc, _, child = self.run_fixture(self.args(parallel=1, scheduling="static"), spawn=spawn)
        self.assertEqual(1, rc)
        self.assertEqual(1, child.call_count)
        result = json.loads((self.root / "output/aggregate.json").read_text())
        self.assertEqual(6, result["summary"]["blocked"])
        self.assertIn("ports not released", result["lanes"][0]["retirement_reason"])

    def test_retiring_after_last_run_is_still_an_infrastructure_failure(self):
        occupied = set()
        def spawn(command, env, log, **kwargs):
            rc = self.fixture_child(command, env, log, **kwargs)
            occupied.add(int(env["FLEXLB_FT_MASTER_HTTP_PORT"]))
            return rc
        with mock.patch.object(ports, "port_in_use", side_effect=lambda port, host=None: port in occupied):
            rc, _, child = self.run_fixture(self.args(parallel=1, instances=self.data["instances"][0]["id"]), spawn=spawn)
        self.assertEqual(1, rc)
        self.assertEqual(1, child.call_count)
        result = json.loads((self.root / "output/aggregate.json").read_text())
        self.assertEqual(1, result["summary"]["retired_lanes"])
        self.assertEqual(0, result["summary"]["blocked"])
        self.assertEqual("PASS", result["instances"][0]["status"])

    def test_failed_gate_with_clean_resources_does_not_retire_lane(self):
        def spawn(command, env, log, **kwargs):
            self.fixture_child(command, env, log, **kwargs)
            path = Path(command[command.index("--out-dir") + 1]) / "scenarios.json"
            result = json.loads(path.read_text())
            row = result["instances"][0]
            row["status"] = "FAIL"
            row["stages"][0]["checks"][0]["status"] = "FAIL"
            path.write_text(json.dumps(result))
            return 1
        rc, _, child = self.run_fixture(self.args(parallel=1, scheduling="static"), spawn=spawn)
        self.assertEqual(1, rc)
        self.assertEqual(7, child.call_count)
        result = json.loads((self.root / "output/aggregate.json").read_text())
        self.assertEqual(0, result["summary"]["blocked"])
        self.assertTrue(result["lanes"][0]["reusable"])


class ChildProcessesTest(unittest.TestCase):
    def test_exited_runner_cannot_leave_a_live_descendant(self):
        import socket
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ready = root / "descendant-port"
            descendant = ("import socket,time;from pathlib import Path;"
                          "s=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
                          "Path(" + repr(str(ready)) + ").write_text(str(s.getsockname()[1]));time.sleep(60)")
            parent = ("import subprocess,sys,time;from pathlib import Path;"
                      "subprocess.Popen([sys.executable,'-c'," + repr(descendant) + "]);"
                      "p=Path(" + repr(str(ready)) + ")\nwhile not p.exists(): time.sleep(.01)")
            children = runner.ChildProcesses()
            with self.assertRaisesRegex(RuntimeError, "live descendant"):
                children.run([sys.executable, "-c", parent], dict(os.environ), root / "runner.log", timeout=5)
            port = int(ready.read_text())
            end = time.monotonic() + 2
            while time.monotonic() < end:
                with socket.socket() as client:
                    if client.connect_ex(("127.0.0.1", port)) != 0:
                        break
                time.sleep(.01)
            else:
                self.fail("owned descendant socket remained live")
            self.assertFalse(children.processes)

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
