"""Compare compiled Master configs with the legacy environment render path."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from flexlb_ft.harness import OMIT, ConfigOverride, render_env
from flexlb_ft.scenario import compile_scenarios, load_scenarios
from flexlb_ft.scenario.catalog import handlers
from flexlb_ft.support.ha import tier1_dual_spec


class MasterConfigTests(unittest.TestCase):
    def test_all_lifecycle_variants_match_legacy_rendered_configuration(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "scenarios/master/master_lifecycle.yaml"
        )
        plans = compile_scenarios(load_scenarios(path), handlers=handlers())
        self.assertEqual(6, len(plans))
        for plan in plans:
            with self.subTest(variant=plan["variant_id"], profile=plan["profile"]):
                if plan["variant_id"] == "kill_single":
                    override = ConfigOverride(
                        ordering="priority", queue_timeout_ms=OMIT
                    )
                else:
                    spec = tier1_dual_spec(SimpleNamespace(profile=plan["profile"]))
                    override = spec.config_overrides
                    self.assertIsNone(override)
                expected = json.loads(render_env(plan["profile"], override))
                self.assertEqual(expected, plan["environment"]["resolved_config"])

    def test_single_nonbatch_freeze_retains_fifo_and_queue_deadline(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "scenarios/master/master_lifecycle.yaml"
        )
        plans = compile_scenarios(load_scenarios(path), handlers=handlers())
        freeze = next(
            p
            for p in plans
            if p["variant_id"] == "freeze_short_long"
            and p["profile"] == "single-nonbatch"
        )
        scheduler = freeze["environment"]["resolved_config"]["scheduler"]
        self.assertEqual("FIFO", scheduler["ordering"]["type"])
        self.assertEqual(60000, scheduler["queueTimeoutMs"])
