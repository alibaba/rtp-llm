"""Only the explicitly typed metric whitelist can enter the Master environment."""

import unittest

from flexlb_ft.scenario import compile_scenarios
from flexlb_ft.scenario.backend import make_env_spec
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.scenario.compiler import environment
from test_scenario_runtime import source


class MetricConfigTest(unittest.TestCase):
    def test_typed_metric_environment_keeps_debug_and_full_config(self):
        raw = {
            "metric_whitelist": "flexlb_auto_tpm_request_count",
            "debug_enabled": True,
        }
        plan = environment(raw, "test", "single-nonbatch")
        spec = make_env_spec(plan, "single-nonbatch", {"master_base": 28000})
        self.assertEqual(
            spec.master_env,
            {
                "FLEXLB_MONITOR_METRIC_WHITELIST": "flexlb_auto_tpm_request_count",
                "FLEXLB_DEBUG_ENABLED": "true",
            },
        )
        self.assertEqual(
            plan["resolved_config"],
            environment({"debug_enabled": True}, "test", "single-nonbatch")[
                "resolved_config"
            ],
        )

    def test_variant_whitelist_is_explicit_and_other_variants_do_not_inherit_it(self):
        doc = source()
        doc["variants"] = [
            {"id": "ordinary"},
            {
                "id": "metric",
                "environment_overrides": {"metric_whitelist": "flexlb_auto_tpm"},
            },
        ]
        plans = compile_scenarios([("test", doc)], handlers=handlers())
        self.assertNotIn("metric_whitelist", plans[0]["environment"])
        self.assertEqual(plans[1]["environment"]["metric_whitelist"], "flexlb_auto_tpm")
        self.assertEqual(
            plans[0]["environment"]["resolved_config"],
            plans[1]["environment"]["resolved_config"],
        )

    def test_arbitrary_environment_text_and_unbounded_lists_are_rejected(self):
        for value in (
            "",
            "*",
            "foo\nbar",
            "foo,",
            "foo;bar",
            None,
            ["foo"],
            ",".join(["foo"] * 17),
            "a" * 1025,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                environment({"metric_whitelist": value}, "test", "single-nonbatch")


if __name__ == "__main__":
    unittest.main()
