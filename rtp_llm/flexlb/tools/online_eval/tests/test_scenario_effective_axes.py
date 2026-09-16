import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_cfg import render_env
from flexlb_test_framework.scenario import compile_scenarios, load_scenarios
from flexlb_test_framework.scenario.backend import make_env_spec


class EffectiveAxesTests(unittest.TestCase):
    def source(self, overrides):
        source = copy.deepcopy(load_scenarios(ROOT / "scenarios/core")[0][1])
        source["stages"] = source.pop("variants")[0]["stages"]
        source["environment"]["config_overrides"] = overrides
        return source

    def test_actual_nonbatch_priority_axes_are_not_inferred_from_profile_label(self):
        source = self.source(
            dict(
                ordering="priority",
                decision="single",
                dispatcher="non_batch",
                max_inflight_per_prefill_worker=1,
            )
        )
        source["requires"] = ["priority", "single", "generate_stream"]
        plans = compile_scenarios([("axes.yaml", source)])
        self.assertEqual(len(plans), 4)
        for plan in plans:
            self.assertEqual(
                plan["effective_axes"],
                dict(
                    scheduler="QUEUE",
                    ordering="PRIORITY",
                    decision="SINGLE",
                    dispatcher="NON_BATCH",
                ),
            )
            self.assertIn("generate_stream", plan["effective_capabilities"])
            self.assertNotIn("enqueue_batch", plan["effective_capabilities"])
            self.assertNotIn("fifo", plan["effective_capabilities"])
        source["stages"][1]["params"]["consume"] = "deferred"
        with self.assertRaisesRegex(ValueError, "deferred.*enqueue_batch"):
            compile_scenarios([("axes.yaml", source)])

    def test_declared_requirements_must_match_effective_capabilities(self):
        source = self.source(dict(dispatcher="non_batch"))
        source["profiles"] = ["batch-window"]
        source["requires"] = ["enqueue_batch"]
        with self.assertRaisesRegex(ValueError, "effective environment lacks"):
            compile_scenarios([("axes.yaml", source)])

    def test_preemption_is_preserved_through_backend_ssot_rendering(self):
        preemption = dict(
            allowed_victim_stages=[
                "PREFILL_QUEUED",
                "DECODE_RESERVED",
                "DECODE_ENGINE_OWNED",
            ],
            timeout_ms=2000,
        )
        source = self.source(dict(ordering="priority", preemption=preemption))
        source["requires"] = ["preemption", "engine_cancellation"]
        plan = compile_scenarios([("axes.yaml", source)])[0]
        spec = make_env_spec(
            plan["environment"], plan["profile"], dict(master_base=28000)
        )
        self.assertEqual(spec.config_overrides.preemption, preemption)
        self.assertEqual(
            json.loads(render_env(plan["profile"], spec.config_overrides)),
            plan["environment"]["resolved_config"],
        )

    def test_nested_preemption_rejects_unknown_duplicate_or_cross_field_errors(self):
        for preemption in (
            {"allowed_victim_stages": []},
            {"allowed_victim_stages": ["invented"]},
            {"allowed_victim_stages": ["PREFILL_QUEUED"] * 2},
            {
                "allowed_victim_stages": ["PREFILL_QUEUED"],
                "engine_cancellation": {
                    "ack_timeout_ms": 1,
                    "completion_timeout_ms": 2,
                },
            },
            {
                "allowed_victim_stages": ["DECODE_ENGINE_OWNED"],
                "engine_cancellation": {
                    "ack_timeout_ms": True,
                    "completion_timeout_ms": 2,
                },
            },
            {"allowed_victim_stages": ["PREFILL_QUEUED"], "invented": 1},
        ):
            with self.subTest(preemption=preemption), self.assertRaises(ValueError):
                compile_scenarios(
                    [
                        (
                            "axes.yaml",
                            self.source(
                                dict(ordering="priority", preemption=preemption)
                            ),
                        )
                    ]
                )


if __name__ == "__main__":
    unittest.main()
