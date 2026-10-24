"""Cooperative deadline and ownership tests use no threads, sleeps or services."""

import tempfile
import unittest
from pathlib import Path

from program_fixtures import compile_scenarios
from scenario.contracts import CheckResult, StageHandler, StageOutput
from scenario.runtime import RuntimeContext, execute_instance


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = Backend(self.clock)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_plan(self, doc=None, handlers=None):
        plan = compile_scenarios([("fixture.json", doc or source())], handlers=handlers)[0]
        return execute_instance(
            plan, self.backend, handlers=handlers, artifact_dir=self.tmp.name, clock=self.clock, sleeper=self.clock.sleep
        )

    def test_order_resources_cross_stage_and_reverse_cleanup(self):
        result = self.run_plan()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(self.backend.calls, ["setup", "submit", "wait", "request_cleanup", "teardown"])
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
        self.assertEqual(self.backend.calls, ["setup", "submit", "request_cleanup", "teardown"])

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
            {"id": "normal", "action": "check", "params": {"actual": {"$ref": "stages.wait.output.error_count"}, "op": "eq", "expected": 0}}
        )
        self.backend.completed = False
        result = self.run_plan(doc)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["finding_confirmed"], ["check.comparison"])

    def test_warning_is_visible_and_does_not_fail_execution(self):
        handler = StageHandler(
            "inspect",
            lambda p, plan: p,
            lambda ctx, p, d: StageOutput({"ok": True}, [CheckResult("complete", "WARNING", "known issue")]),
            {"ok": "boolean"},
            checks=frozenset({"complete"}),
        )
        doc = source()
        doc["stages"].append({"id": "advisory", "action": "inspect"})
        result = self.run_plan(doc, {"inspect": handler})
        self.assertEqual(result["status"], "PASS")
        check = result["stages"][-1]["checks"][0]
        self.assertEqual(check["status"], "WARNING")
        self.assertEqual(check["detail"], "known issue")

    def test_adapter_output_and_check_contract_errors_are_not_findings(self):
        handler = StageHandler(
            "inspect",
            lambda p, plan: p,
            lambda ctx, p, d: StageOutput({"ok": True}, [CheckResult("complete", "ERROR", "missing sample")]),
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
            lambda ctx, p, d: StageOutput({}, [CheckResult("sample", "SKIP", "no matching sample")]),
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
        ctx = RuntimeContext({}, self.backend, self.tmp.name, self.clock, self.clock.sleep)
        live = ctx.register_resource("requests", [1])
        historic = ctx.register_resource("snapshot", {"count": 1}, historical=True)
        with self.assertRaisesRegex(ValueError, "forged"):
            ctx.resource({**live, "id": "unknown"}, "requests")
        ctx.env_epoch += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            ctx.resource(live, "requests", allow_stale=True)
        self.assertEqual(ctx.resource(historic, "snapshot", allow_stale=True), {"count": 1})

    def test_cleanup_errors_preserve_primary_error_and_all_attempts(self):
        self.backend.fail_submit = self.backend.fail_cleanup = True
        result = self.run_plan()
        self.assertIn("partial submit", result["error"])
        self.assertEqual([r["status"] for r in result["cleanup"]], ["ERROR", "PASS"])


if __name__ == "__main__":
    unittest.main()

from runtime_fixtures import Backend, Clock, source
