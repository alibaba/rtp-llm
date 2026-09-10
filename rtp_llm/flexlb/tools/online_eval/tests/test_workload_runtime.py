"""Different execution policies retain original failures and cleanup evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.runtime import execute_instance
from flexlb_test_framework.workload.runtime import execute_workload
from flexlb_test_framework.suites import classify
from test_scenario_runtime import source, Backend, Clock

ROOT = Path(__file__).resolve().parents[1]


class WorkloadRuntimeTests(unittest.TestCase):
    def run_plan(
        self, workload, observation=True, setup_error=False, cleanup_error=False
    ):
        doc = source()
        doc["stages"][-1]["purpose"] = "observation" if observation else "operation"
        doc["stages"].append(dict(doc["stages"][-1], id="independent"))
        plan = compile_scenarios([("fixture", doc)])[0]
        plan.update(
            test_kind="workload" if workload else "functional",
            workload_runtime={
                "capture_metrics": False,
                "sample_interval_s": 1,
                "collector_shutdown_s": 2,
            },
        )
        clock = Clock()
        backend = Backend(clock)
        backend.completed = False
        backend.fail_setup = setup_error
        backend.fail_cleanup = cleanup_error
        with tempfile.TemporaryDirectory() as out:
            result = (execute_workload if workload else execute_instance)(
                plan, backend, artifact_dir=out, clock=clock, sleeper=clock.sleep
            )
            if workload:
                self.assertTrue((Path(out) / "workload-report.html").is_file())
                report = json.loads((Path(out) / "workload-report.json").read_text())
                self.assertEqual(result["status"], report["status"])
            return result

    def test_functional_failure_blocks(self):
        r = self.run_plan(False)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["stages"][-1]["status"], "BLOCKED")

    def test_workload_preserves_independent_failures(self):
        r = self.run_plan(True)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual([s["status"] for s in r["stages"][-2:]], ["FAIL", "FAIL"])
        self.assertEqual(r["workload"]["runtime_validity"], "VALID")
        self.assertEqual(r["workload"]["performance_verdict"], "NOT_EVALUATED")

    def test_prerequisite_failure_blocks(self):
        self.assertEqual(
            self.run_plan(True, observation=False)["stages"][-1]["status"], "BLOCKED"
        )

    def test_setup_error_is_invalid(self):
        r = self.run_plan(True, setup_error=True)
        self.assertEqual(r["status"], "ERROR")
        self.assertEqual(r["workload"]["runtime_validity"], "INVALID")

    def test_cleanup_error_is_invalid_even_after_fail(self):
        r = self.run_plan(True, cleanup_error=True)
        self.assertEqual(r["workload"]["runtime_validity"], "INVALID")
        self.assertTrue(any(c["status"] != "PASS" for c in r["cleanup"]))

    def test_catalog_is_disjoint_and_complete(self):
        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios"), handlers=handlers()
        )
        f = {p["id"] for p in classify(plans, "functional")}
        w = {p["id"] for p in classify(plans, "workload")}
        self.assertFalse(f & w)
        self.assertEqual(f | w, {p["id"] for p in plans})
        self.assertTrue(any("wraparound" in x for x in w))
        self.assertTrue(any("client_no_fetch" in x for x in f))
        self.assertEqual(len(plans), 385)
