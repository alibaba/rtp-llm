"""Birth settings preserve legacy performance and Master logging ownership."""

import copy
import unittest

from flexlb_test_framework.harness import default_perf, fault_env_perf
from flexlb_test_framework.scenario import compile_scenarios
from flexlb_test_framework.scenario.backend import make_env_spec
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.compiler import environment
from test_scenario_runtime import source


def spec(raw):
    return make_env_spec(
        environment(raw, "test", "single-nonbatch"),
        "single-nonbatch",
        {"master_base": 28000},
    )


class StartupOptionsTest(unittest.TestCase):
    def test_waiting_cap_merges_birth_perf_without_adding_batch_limits(self):
        for preset, base in (
            ("default", default_perf()),
            ("fault_env", fault_env_perf()),
        ):
            for cap in (0, 16):
                with self.subTest(preset=preset, cap=cap):
                    expected = copy.deepcopy(base)
                    expected.setdefault("prefill", {})["max_waiting_batches"] = cap
                    actual = spec(
                        {"perf_preset": preset, "prefill_max_waiting_batches": cap}
                    )
                    self.assertEqual(actual.perf, expected)
                    self.assertNotIn("max_batch_tokens", actual.perf["prefill"])
                    self.assertNotIn("max_batch_requests", actual.perf["prefill"])
        self.assertEqual(spec({"perf_preset": "fault_env"}).perf, fault_env_perf())

    def test_explicit_perf_replacement_and_cap_compose_without_mutating_input(self):
        perf = dict(fixed_ms=3000, scale=1, max_batch_tokens=1024, max_batch_requests=0)
        raw = dict(prefill_perf=perf, prefill_max_waiting_batches=16)
        before = copy.deepcopy(raw)
        self.assertEqual(spec(raw).perf["prefill"], dict(perf, max_waiting_batches=16))
        self.assertEqual(raw, before)

    def test_master_logging_and_diagnostic_api_are_independent(self):
        baseline = environment({}, "test", "single-nonbatch")["resolved_config"]
        for logging in (False, True):
            for diagnostic in (False, True):
                raw = dict(master_debug_log=logging, debug_enabled=diagnostic)
                actual = spec(raw)
                self.assertIs(actual.master_debug_log, logging)
                self.assertEqual(
                    actual.master_env,
                    {"FLEXLB_DEBUG_ENABLED": "true"} if diagnostic else {},
                )
                self.assertEqual(
                    environment(raw, "test", "single-nonbatch")["resolved_config"],
                    baseline,
                )
        self.assertFalse(spec({}).master_debug_log)

    def test_variant_options_survive_environment_epochs_without_leaking(self):
        doc = source()
        doc["profiles"] = ["single-nonbatch"]
        doc["variants"] = [
            dict(id="ordinary"),
            dict(
                id="configured",
                environment_overrides=dict(
                    master_debug_log=True,
                    prefill_max_waiting_batches=16,
                    perf_preset="fault_env",
                ),
            ),
        ]
        doc["stages"].insert(
            1,
            dict(
                id="replace",
                action="environment_reconfigure",
                params=dict(config_overrides=dict(ordering="fifo")),
            ),
        )
        plans = compile_scenarios([("test", doc)], handlers=handlers())
        for index, plan in enumerate(plans):
            initial = plan["environment"]
            later = plan["stages"][1]["params"]["environments"]["single-nonbatch"]
            self.assertEqual(
                initial.get("master_debug_log"), later.get("master_debug_log")
            )
            self.assertEqual(
                initial.get("prefill_max_waiting_batches"),
                later.get("prefill_max_waiting_batches"),
            )
            actual = make_env_spec(later, "single-nonbatch", {"master_base": 28000})
            if index:
                self.assertTrue(actual.master_debug_log)
                self.assertEqual(
                    actual.perf["prefill"],
                    dict(fixed_ms=100.0, scale=1.0, max_waiting_batches=16),
                )
            else:
                self.assertNotIn("prefill_max_waiting_batches", initial)
                self.assertNotIn("master_debug_log", initial)
                self.assertEqual(actual.perf, default_perf())

    def test_strict_types_reject_before_process_construction(self):
        for field, values in (
            ("master_debug_log", (None, 0, 1, "true", [], {})),
            (
                "prefill_max_waiting_batches",
                (None, True, False, -1, 1.5, 16.0, "16", [], {}),
            ),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(
                    ValueError
                ):
                    environment({field: value}, "test", "single-nonbatch")


if __name__ == "__main__":
    unittest.main()
