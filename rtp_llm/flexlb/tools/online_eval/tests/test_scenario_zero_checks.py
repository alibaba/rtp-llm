"""A successful setup is not a successful validation."""

import copy
import tempfile
import unittest

from flexlb_test_framework.scenario import ScenarioError, compile_scenarios
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_runtime import Backend, Clock, source


class ZeroChecksTest(unittest.TestCase):
    def test_compiler_rejects_setup_only(self):
        doc = source()
        doc["stages"] = doc["stages"][:1]
        with self.assertRaisesRegex(ScenarioError, "at least one check"):
            compile_scenarios([("zero.yaml", doc)])

    def test_runtime_defends_unchecked_plan_and_still_cleans(self):
        plan = compile_scenarios([("valid.yaml", source())])[0]
        clock, backend = Clock(), Backend(Clock())
        plan["stages"] = plan["stages"][:1]
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_instance(
                plan, backend, artifact_dir=tmp, clock=clock, sleeper=clock.sleep
            )
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error"], "no checks executed")
        self.assertIn("teardown", backend.calls)
        self.assertEqual(result["cleanup"][0]["status"], "PASS")

    def test_blocked_checks_keep_original_setup_error(self):
        plan = compile_scenarios([("valid.yaml", source())])[0]
        clock = Clock()
        backend = Backend(clock)
        backend.fail_setup = True
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_instance(
                plan, backend, artifact_dir=tmp, clock=clock, sleeper=clock.sleep
            )
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("partial setup", result["error"])
        self.assertTrue(all(r["status"] == "BLOCKED" for r in result["stages"][1:]))
        self.assertIn("teardown", backend.calls)

    def test_checked_plan_can_pass(self):
        plan = compile_scenarios([("valid.yaml", source())])[0]
        clock = Clock()
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_instance(
                plan, Backend(clock), artifact_dir=tmp, clock=clock, sleeper=clock.sleep
            )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(sum(len(r["checks"]) for r in result["stages"]), 1)


if __name__ == "__main__":
    unittest.main()
