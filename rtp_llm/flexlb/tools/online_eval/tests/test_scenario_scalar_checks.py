"""Typed primitive comparisons preserve numeric finiteness and boolean separation."""

import tempfile
import unittest

from flexlb_test_framework.scenario import ScenarioError, compile_scenarios
from flexlb_test_framework.scenario.contracts import StageHandler, StageOutput
from flexlb_test_framework.scenario.runtime import execute_instance
from test_scenario_runtime import Backend, Clock, source


class ScalarCheckTest(unittest.TestCase):
    def fixture(self, kind, actual, expected, op="eq"):
        doc = source()
        doc["stages"] = [
            doc["stages"][0],
            dict(id="metric", action="metric"),
            dict(
                id="assertion",
                action="check",
                params=dict(
                    actual={"$ref": "stages.metric.output.value"},
                    expected=expected,
                    op=op,
                ),
            ),
        ]
        registry = {
            "metric": StageHandler(
                "metric",
                lambda params, plan: params,
                lambda *args: StageOutput({"value": actual}),
                {"value": kind},
            )
        }
        return doc, registry

    def test_numeric_rate_and_string_equality_execute(self):
        for kind, actual, expected, op in [
            ("number", 0.95, 0.9, "ge"),
            ("number", 1, 0.9, "ge"),
            ("string", "ready", "ready", "eq"),
        ]:
            doc, handlers = self.fixture(kind, actual, expected, op)
            plan = compile_scenarios([("scalar.yaml", doc)], handlers=handlers)[0]
            clock = Clock()
            with tempfile.TemporaryDirectory() as tmp:
                result = execute_instance(
                    plan, Backend(clock), handlers, tmp, clock, clock.sleep
                )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["stages"][-1]["checks"][0]["actual"], actual)

    def test_boolean_nonfinite_and_string_ordering_are_rejected(self):
        for kind, expected, op in [
            ("number", True, "eq"),
            ("number", float("nan"), "ge"),
            ("number", float("inf"), "le"),
            ("string", "x", "ge"),
            ("integer", 0.9, "ge"),
        ]:
            doc, handlers = self.fixture(kind, 1, expected, op)
            with self.assertRaises(ScenarioError):
                compile_scenarios([("scalar.yaml", doc)], handlers=handlers)

    def test_nonfinite_runtime_output_is_error_before_comparison(self):
        doc, handlers = self.fixture("number", float("nan"), 0.9, "ge")
        plan = compile_scenarios([("scalar.yaml", doc)], handlers=handlers)[0]
        clock = Clock()
        with tempfile.TemporaryDirectory() as tmp:
            result = execute_instance(
                plan, Backend(clock), handlers, tmp, clock, clock.sleep
            )
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["stages"][-1]["status"], "BLOCKED")
        self.assertEqual(result["cleanup"][0]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
