"""Compare compiled Master configs with the legacy environment render path."""

import json
import unittest
from pathlib import Path

from runtime.harness import render_env
from scenario import compile_scenarios, load_scenarios
from scenario.catalog import handlers


class MasterConfigTests(unittest.TestCase):
    def test_request_completion_matches_expected_rendered_configuration(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "config/scenarios/request_completion.yaml"
        )
        plans = compile_scenarios(load_scenarios(path), handlers=handlers())
        self.assertEqual(4, len(plans))
        for plan in plans:
            with self.subTest(variant=plan["variant_id"], profile=plan["profile"]):
                expected = json.loads(render_env(plan["profile"]))
                self.assertEqual(expected, plan["environment"]["resolved_config"])
