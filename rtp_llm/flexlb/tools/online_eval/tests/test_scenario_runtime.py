"""Cooperative deadline and ownership tests use no threads, sleeps or services."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.contracts import (
    CheckResult,
    StageHandler,
    StageOutput,
)
from flexlb_test_framework.scenario.runtime import (
    Deadline,
    RuntimeContext,
    StageTimeout,
    execute_instance,
)


def source():
    return {
        "schema_version": 1,
        "id": "lifecycle",
        "description": "Request lifecycle",
        "category": "status",
        "profiles": ["batch-window"],
        "environment": {},
        "execution": {"timeout_s": 3, "stage_timeout_s": 2, "cleanup_timeout_s": 5},
        "stages": [
            {"id": "setup", "action": "setup"},
            {"id": "submit", "action": "request"},
            {
                "id": "wait",
                "action": "wait",
                "params": {"requests": {"$ref": "stages.submit.output.requests"}},
            },
            {
                "id": "check",
                "action": "check",
                "params": {
                    "actual": {"$ref": "stages.wait.output.completed"},
                    "op": "eq",
                    "expected": True,
                },
            },
        ],
    }


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Backend:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.fail_setup = self.fail_submit = self.fail_cleanup = self.timeout_wait = (
            False
        )
        self.completed = True

    def setup(self, ctx, environment, deadline):
        self.calls.append("setup")
        if self.fail_setup:
            raise RuntimeError("partial setup")
        return object(), object()

    def start_requests(self, ctx, params, deadline):
        self.calls.append("submit")
        records = [1]

        def cleanup(d):
            self.calls.append("request_cleanup")
            d.sleep(0.5)
            if self.fail_cleanup:
                raise RuntimeError("cancel failed")

        handle = ctx.register_resource("requests", records, cleanup)
        if self.fail_submit:
            raise RuntimeError("partial submit")
        return handle

    def wait_requests(self, ctx, resource, deadline):
        self.calls.append("wait")
        if self.timeout_wait:
            deadline.sleep(4)
        return {"completed": self.completed, "error_count": int(not self.completed)}

    def cancel_requests(self, ctx, resource, deadline):
        return len(resource)

    def teardown(self, ctx, deadline):
        self.calls.append("teardown")
        deadline.sleep(0.5)


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = Backend(self.clock)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_plan(self, doc=None, handlers=None):
        plan = compile_scenarios(
            [("fixture.json", doc or source())], handlers=handlers
        )[0]
        return execute_instance(
            plan,
            self.backend,
            handlers=handlers,
            artifact_dir=self.tmp.name,
            clock=self.clock,
            sleeper=self.clock.sleep,
        )

    def test_order_resources_cross_stage_and_reverse_cleanup(self):
        result = self.run_plan()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            self.backend.calls,
            ["setup", "submit", "wait", "request_cleanup", "teardown"],
        )
        self.assertEqual([r["status"] for r in result["cleanup"]], ["PASS", "PASS"])
        self.assertTrue((Path(self.tmp.name) / "result.json").is_file())

    def test_partial_setup_and_partial_submit_are_cleaned(self):
        self.backend.fail_setup = True
        result = self.run_plan()
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(self.backend.calls, ["setup", "teardown"])
        self.assertTrue(all(r["status"] == "BLOCKED" for r in result["stages"][1:]))
        self.backend.fail_setup = False
        self.backend.fail_submit = True
        self.backend.calls.clear()
        result = self.run_plan()
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(
            self.backend.calls, ["setup", "submit", "request_cleanup", "teardown"]
        )

    def test_timeout_has_independent_cleanup_budget(self):
        self.backend.timeout_wait = True
        result = self.run_plan()
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["stages"][-1]["status"], "BLOCKED")
        self.assertEqual([r["status"] for r in result["cleanup"]], ["PASS", "PASS"])
        self.assertEqual(self.backend.calls[-2:], ["request_cleanup", "teardown"])

    def test_finding_is_check_specific_and_does_not_hide_cleanup_or_stage_errors(self):
        doc = source()
        doc["findings"] = ["check.comparison"]
        self.backend.completed = False
        result = self.run_plan(doc)
        self.assertEqual(result["status"], "FINDING-CONFIRMED")
        self.backend.fail_cleanup = True
        result = self.run_plan(doc)
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["cleanup"][-1]["status"], "PASS")
        self.backend.fail_cleanup = False
        self.backend.fail_submit = True
        self.assertEqual(self.run_plan(doc)["status"], "ERROR")

    def test_known_finding_and_normal_failure_coexist(self):
        doc = source()
        doc["findings"] = ["check.comparison"]
        doc["stages"].append(
            {
                "id": "normal",
                "action": "check",
                "params": {
                    "actual": {"$ref": "stages.wait.output.error_count"},
                    "op": "eq",
                    "expected": 0,
                },
            }
        )
        self.backend.completed = False
        result = self.run_plan(doc)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["finding_confirmed"], ["check.comparison"])

    def test_adapter_output_and_check_contract_errors_are_not_findings(self):
        handler = StageHandler(
            "inspect",
            lambda p, plan: p,
            lambda ctx, p, d: StageOutput(
                {"ok": True}, [CheckResult("complete", "ERROR", "missing sample")]
            ),
            {"ok": "boolean"},
            checks=frozenset({"complete"}),
        )
        doc = source()
        doc["stages"] = [doc["stages"][0], {"id": "inspect", "action": "inspect"}]
        doc["findings"] = ["inspect.complete"]
        result = self.run_plan(doc, {"inspect": handler})
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["finding_confirmed"], [])

    def test_skipped_probe_does_not_claim_finding_resolved(self):
        handler = StageHandler(
            "inspect",
            lambda p, plan: p,
            lambda ctx, p, d: StageOutput(
                {}, [CheckResult("sample", "SKIP", "no matching sample")]
            ),
            {},
            checks=frozenset({"sample"}),
        )
        doc = source()
        doc["stages"].append({"id": "inspect", "action": "inspect"})
        doc["findings"] = ["inspect.sample"]
        result = self.run_plan(doc, {"inspect": handler})
        self.assertEqual("PASS", result["status"])
        self.assertEqual([], result["finding_resolved"])
        self.assertEqual("SKIP", result["stages"][-1]["checks"][0]["status"])

    def test_explicit_teardown_cleans_requests_before_environment_once(self):
        doc = source()
        doc["stages"].append({"id": "end", "action": "teardown"})
        result = self.run_plan(doc)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(self.backend.calls.count("teardown"), 1)
        self.assertEqual(len(result["cleanup"]), 2)

    def test_handle_authenticity_epoch_and_historical_read(self):
        ctx = RuntimeContext(
            {}, self.backend, self.tmp.name, self.clock, self.clock.sleep
        )
        live = ctx.register_resource("requests", [1])
        historic = ctx.register_resource("snapshot", {"count": 1}, historical=True)
        with self.assertRaisesRegex(ValueError, "forged"):
            ctx.resource({**live, "id": "unknown"}, "requests")
        ctx.env_epoch += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            ctx.resource(live, "requests", allow_stale=True)
        self.assertEqual(
            ctx.resource(historic, "snapshot", allow_stale=True), {"count": 1}
        )

    def test_cleanup_errors_preserve_primary_error_and_all_attempts(self):
        self.backend.fail_submit = self.backend.fail_cleanup = True
        result = self.run_plan()
        self.assertIn("partial submit", result["error"])
        self.assertEqual([r["status"] for r in result["cleanup"]], ["ERROR", "PASS"])


if __name__ == "__main__":
    unittest.main()
