"""Action ownership, registration and strict parameter boundaries."""

import ast
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from case_registry_fixtures import entry, snapshot
from cases.master_ha_failover import actions as master_ha
from cases.master_performance import program as master_performance
from context_fixtures import runtime_context
from document_io import load_document
from program_fixtures import compile_scenarios
from scenario import ScenarioError, registry
from scenario.catalog import FOUNDATION_HANDLERS, handlers
from scenario.parameters import validate_fields
from scenario.program_compiler import configure_program

ROOT = Path(__file__).resolve().parents[1]


class ActionContractsTest(unittest.TestCase):
    def test_case_catalog_contains_foundations_and_only_declared_capabilities(self):
        foundation = {h.name for h in FOUNDATION_HANDLERS}
        catalog = handlers("master_performance")
        self.assertEqual(foundation | {"performance_observe", "performance_finish"}, set(catalog))
        self.assertTrue(all(not catalog[name].owners for name in foundation))
        self.assertEqual(frozenset({"master_performance"}), catalog["performance_finish"].owners)
        self.assertEqual(foundation | {"request_complete"}, set(handlers("request_completion")))
        with self.assertRaisesRegex(ValueError, "unknown registered Python case"):
            handlers("unknown")

    def test_another_case_cannot_use_owned_action_even_with_spoofed_id(self):
        path = ROOT / "config/scenarios/request_completion.yaml"
        document = configure_program(load_document(path), str(path))
        document = replace(document, id="master_performance")
        variant = document.variants[0]
        variant = replace(variant, stages=(variant.stages[0], replace(variant.stages[1], action="performance_finish"), *variant.stages[2:]))
        document = replace(document, variants=(variant,))
        with self.assertRaisesRegex(ScenarioError, "belongs to cases.*master_performance"):
            compile_scenarios([(str(path), document)], handlers=handlers())

    def test_duplicate_foundational_or_case_action_is_rejected(self):
        duplicate = FOUNDATION_HANDLERS[0]
        with patch.object(
            registry,
            "_snapshot",
            snapshot(entry("master_performance", replace(master_performance.CASE, actions=(duplicate,))), include_existing=True),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate action"):
                handlers("master_performance")
        with patch.object(
            registry,
            "_snapshot",
            snapshot(
                entry(
                    "master_performance",
                    replace(master_performance.CASE, actions=(replace(duplicate, owners=frozenset({"another_case"})),)),
                ),
                include_existing=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "conflicting case owner"):
                handlers("master_performance")

    def test_field_validation_rejects_unknown_missing_and_non_mapping_input(self):
        plan = SimpleNamespace(path="case.stage")
        for value, message in (
            (None, "case.stage: expected mapping"),
            ([], "case.stage: expected mapping"),
            ({"extra": 1}, "case.stage: unknown configuration fields"),
            ({}, "missing YAML parameter case.stage.required"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                validate_fields(value, plan, {"required"}, {"required"})
        source = {"required": {"nested": [1]}}
        result = validate_fields(source, plan, {"required"}, {"required"})
        result["required"]["nested"].append(2)
        self.assertEqual([1], source["required"]["nested"])

    def test_shared_implementation_requires_each_program_to_declare_it(self):
        with patch.object(registry, "_snapshot", snapshot(entry("second_performance", master_performance.CASE), include_existing=True)):
            catalog = handlers()
            self.assertEqual(frozenset({"master_performance", "second_performance"}), catalog["performance_finish"].owners)
        descriptor = master_performance.ACTION_HANDLERS[0]
        with patch.object(
            registry,
            "_snapshot",
            snapshot(
                entry("master_performance", replace(master_performance.CASE, actions=(descriptor, descriptor))), include_existing=True
            ),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate action"):
                handlers("master_performance")

    def test_ha_period_records_the_same_bounded_event_boundaries(self):
        from runtime.deadline import Deadline

        now = [10.0]

        def sleep(seconds):
            now[0] += seconds

        ctx = runtime_context({}, None, ".", lambda: now[0], sleep, wall_clock=lambda: 1234 + now[0] - 10)
        deadline = Deadline(30, ctx.clock, sleep)
        result = master_ha._period(ctx, dict(name="b", settle_s=2, duration_s=3, operation_timeout_s=6), deadline)
        self.assertEqual(["b_start", "b_end"], [e["id"] for e in ctx.report_events])
        self.assertAlmostEqual(1236, result.output["start"])
        self.assertAlmostEqual(1239, result.output["end"])

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
