"""Action ownership, registration and strict parameter boundaries."""

import ast
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cases.config import configure_program
from cases.master_ha_failover import actions as master_ha
from cases.registry import PROGRAMS
from cases.master_performance import program as master_performance
from scenario import ScenarioError, compile_scenarios
from scenario.catalog import FOUNDATION_HANDLERS, handlers
from scenario.loader import load_document
from scenario.parameters import validate_fields

ROOT = Path(__file__).resolve().parents[1]


class ActionContractsTest(unittest.TestCase):
    def test_case_catalog_contains_foundations_and_only_declared_capabilities(self):
        foundation = {h.name for h in FOUNDATION_HANDLERS}
        catalog = handlers("master_performance")
        self.assertEqual(foundation | {"performance_observe", "performance_finish"}, set(catalog))
        self.assertTrue(all(not catalog[name].owners for name in foundation))
        self.assertEqual(frozenset({"master_performance"}), catalog["performance_finish"].owners)
        self.assertEqual(foundation, set(handlers("request_completion")))
        with self.assertRaisesRegex(ValueError, "unknown registered Python case"):
            handlers("unknown")

    def test_another_case_cannot_use_owned_action_even_with_spoofed_id(self):
        path = ROOT / "config/scenarios/request_completion.yaml"
        document = configure_program(load_document(path), str(path))
        document["id"] = "master_performance"
        document["variants"][0]["stages"][1]["action"] = "performance_finish"
        with self.assertRaisesRegex(ScenarioError, "belongs to cases.*master_performance"):
            compile_scenarios([(str(path), document)], handlers=handlers())

    def test_duplicate_foundational_or_case_action_is_rejected(self):
        duplicate = FOUNDATION_HANDLERS[0]
        with patch.object(master_performance, "ACTION_HANDLERS", (duplicate,)):
            with self.assertRaisesRegex(ValueError, "duplicate action"):
                handlers("master_performance")
        with patch.object(master_performance, "ACTION_HANDLERS", (replace(duplicate, owners=frozenset({"another_case"})),)):
            with self.assertRaisesRegex(ValueError, "conflicting case owner"):
                handlers("master_performance")

    def test_field_validation_rejects_unknown_missing_and_non_mapping_input(self):
        plan = SimpleNamespace(path="case.stage")
        for value in (None, [], {"extra": 1}, {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "case.stage: invalid action parameters"):
                validate_fields(value, plan, {"required"}, {"required"})
        source = {"required": {"nested": [1]}}
        result = validate_fields(source, plan, {"required"}, {"required"})
        result["required"]["nested"].append(2)
        self.assertEqual([1], source["required"]["nested"])

    def test_shared_implementation_requires_each_program_to_declare_it(self):
        with patch.dict(PROGRAMS, second_performance=PROGRAMS["master_performance"]):
            catalog = handlers()
            self.assertEqual(
                frozenset({"master_performance", "second_performance"}),
                catalog["performance_finish"].owners,
            )
        descriptor = master_performance.ACTION_HANDLERS[0]
        with patch.object(master_performance, "ACTION_HANDLERS", (descriptor, descriptor)):
            with self.assertRaisesRegex(ValueError, "duplicate action"):
                handlers("master_performance")

    def test_ha_time_marker_uses_bounded_wait_and_epoch_clock(self):
        deadline = SimpleNamespace(sleep=Mock())
        from scenario.runtime import RuntimeContext
        ctx = RuntimeContext({}, None, ".", lambda: 10, lambda _: None, wall_clock=lambda: 1234)
        result = master_ha._mark(ctx, {"wait_s": 2, "event": "baseline_end"}, deadline)
        self.assertEqual("baseline_end", ctx.report_events[0]["id"])
        deadline.sleep.assert_called_once_with(2)
        self.assertEqual({"epoch_s": 1234}, result.output)

    def test_actions_share_runtime_helpers_without_importing_other_actions(self):
        for directory in (ROOT / "src/scenario/actions", ROOT / "src/cases"):
            for path in directory.rglob("*.py"):
                with self.subTest(path=path.name):
                    tree = ast.parse(path.read_text())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            self.assertFalse((node.module or "").startswith("scenario.actions"))
                        elif isinstance(node, ast.Import):
                            self.assertFalse(any(alias.name.startswith("scenario.actions") for alias in node.names))
