"""Explicit variant programs and bounded, declared master layouts."""

import copy
import tempfile
import unittest
from pathlib import Path

from flexlb_test_framework.scenario import ScenarioError, compile_scenarios
from flexlb_test_framework.scenario.backend import (
    configure_master_sync_log,
    make_env_spec,
)
from flexlb_test_framework.scenario.contracts import StageHandler
from test_scenario_compile import scenario


class VariantProgramsTest(unittest.TestCase):
    def test_sync_log_is_private_per_epoch_and_does_not_create_fake_evidence(self):
        doc = scenario()
        doc["variants"] = [
            dict(id="logged", environment_overrides=dict(master_sync_log=True))
        ]
        plan = compile_scenarios([("logged.yaml", doc)], "batch-window")[0]
        self.assertTrue(plan["environment"]["master_sync_log"])
        spec = make_env_spec(
            plan["environment"], plan["profile"], dict(master_base=28000)
        )
        with tempfile.TemporaryDirectory() as root:
            path = configure_master_sync_log(spec, root, 1)
            self.assertEqual(path, Path(root).resolve() / "master-sync-1" / "sync.log")
            self.assertEqual(
                spec.master_extra_args, [f"--flexlb.log.path={path.parent}"]
            )
            self.assertFalse(path.exists())
            with self.assertRaises(FileExistsError):
                configure_master_sync_log(spec, root, 1)
        for value in (1, "true", dict(path="/tmp/shared")):
            doc["environment"]["master_sync_log"] = value
            doc["variants"] = [dict(id="bad")]
            with self.assertRaises(ScenarioError):
                compile_scenarios([("bad.yaml", doc)])
        doc["environment"].update(master_sync_log=True, master_layout="dual_standalone")
        with self.assertRaises(ScenarioError):
            compile_scenarios([("dual.yaml", doc)])

    def test_startup_prefill_budget_preserves_disabled_zero_and_variant_values(self):
        from flexlb_test_framework.harness import default_perf

        doc = scenario()
        disabled = dict(
            fixed_ms=3000, scale=1, max_batch_tokens=0, max_batch_requests=0
        )
        limited = dict(disabled, max_batch_tokens=1024, max_batch_requests=2)
        doc["environment"]["prefill_perf"] = disabled
        doc["variants"] = [
            dict(id="disabled"),
            dict(id="limited", environment_overrides=dict(prefill_perf=limited)),
        ]
        for plan, expected in zip(
            compile_scenarios([("budget.yaml", doc)], "batch-window"),
            (disabled, limited),
        ):
            spec = make_env_spec(
                plan["environment"], plan["profile"], dict(master_base=28000)
            )
            self.assertEqual(spec.perf["prefill"], expected)
            self.assertEqual(spec.perf["decode"], default_perf()["decode"])
            spec.perf["prefill"]["fixed_ms"] = 0
            self.assertEqual(plan["environment"]["prefill_perf"]["fixed_ms"], 3000)
        self.assertEqual(doc["environment"]["prefill_perf"], disabled)

    def test_startup_prefill_budget_rejects_incomplete_unknown_and_nonfinite_values(
        self,
    ):
        valid = dict(fixed_ms=3000, scale=1, max_batch_tokens=0, max_batch_requests=0)
        invalid = [{}, dict(valid, unknown=1)]
        for key in valid:
            for value in (True, -1, float("nan"), float("inf"), "1"):
                invalid.append(dict(valid, **{key: value}))
        invalid.append(dict(valid, max_batch_requests=1.5))
        for perf in invalid:
            with self.subTest(perf=perf), self.assertRaises(ScenarioError):
                doc = scenario()
                doc["environment"]["prefill_perf"] = perf
                compile_scenarios([("invalid.yaml", doc)])

    def test_variant_programs_are_local_and_explicit(self):
        doc = scenario()
        alternate = copy.deepcopy(doc["stages"])
        alternate[1]["id"] = "other_submit"
        alternate[2]["params"]["requests"][
            "$ref"
        ] = "stages.other_submit.output.requests"
        doc["variants"] = [
            dict(id="base"),
            dict(id="alternate", stages=alternate, execution=dict(timeout_s=33)),
        ]
        original = copy.deepcopy(doc)
        plans = compile_scenarios([("variants.yaml", doc)], "batch-window")
        self.assertEqual(
            [p["stages"][1]["id"] for p in plans], ["submit", "other_submit"]
        )
        self.assertEqual(plans[1]["execution"]["timeout_s"], 33)
        self.assertEqual(doc, original)
        doc["variants"][1]["stages"][2]["params"]["requests"][
            "$ref"
        ] = "stages.submit.output.requests"
        with self.assertRaisesRegex(ScenarioError, "unknown or forward"):
            compile_scenarios([("variants.yaml", doc)])

    def test_variant_only_program_requires_each_variant_to_supply_stages(self):
        doc = scenario()
        steps = doc.pop("stages")
        doc["variants"] = [dict(id="one", stages=steps)]
        self.assertEqual(len(compile_scenarios([("variants.yaml", doc)])), 4)
        doc["variants"].append(dict(id="missing"))
        with self.assertRaises(ScenarioError):
            compile_scenarios([("variants.yaml", doc)])
        doc["variants"] = [dict(id="ambiguous", stages=steps, stage_overrides={})]
        with self.assertRaisesRegex(ScenarioError, "mutually exclusive"):
            compile_scenarios([("variants.yaml", doc)])

    def test_dual_layout_and_coldstart_are_rendered_not_ambient(self):
        doc = scenario()
        doc["environment"].update(
            master_layout="dual_standalone",
            master_stable_window_s=0,
            debug_enabled=True,
        )
        plan = compile_scenarios([("dual.yaml", doc)], "batch-window")[0]
        spec = make_env_spec(
            plan["environment"], plan["profile"], dict(master_base=28000)
        )
        self.assertEqual(
            [(m.name, m.http_port) for m in spec.masters], [("A", 28000), ("B", 28003)]
        )
        self.assertIsNone(spec.zk_consistency)
        self.assertEqual(spec.master_stable_window_s, 0)
        self.assertEqual(spec.master_env, {"FLEXLB_DEBUG_ENABLED": "true"})
        doc["environment"]["master_layout"] = "ambient_tier3"
        with self.assertRaises(ScenarioError):
            compile_scenarios([("dual.yaml", doc)])
        doc["environment"]["master_layout"] = "single"
        doc["environment"]["master_stable_window_s"] = False
        with self.assertRaises(ScenarioError):
            compile_scenarios([("dual.yaml", doc)])

    def test_validator_receives_its_own_variant_environment_and_profiles(self):
        seen = []

        def validate(params, plan):
            seen.append((dict(plan.environment), plan.profiles))
            plan.environment["master_layout"] = "mutated copy"
            return params

        doc = scenario()
        doc["stages"].insert(1, dict(id="probe", action="probe"))
        handler = StageHandler("probe", validate, lambda *_: None, {})
        compile_scenarios([("context.yaml", doc)], handlers={"probe": handler})
        self.assertEqual(
            seen[0][1],
            ("batch-window", "single-nonbatch", "single-batch", "window-nonbatch"),
        )
        self.assertNotIn("master_layout", doc["environment"])


if __name__ == "__main__":
    unittest.main()
