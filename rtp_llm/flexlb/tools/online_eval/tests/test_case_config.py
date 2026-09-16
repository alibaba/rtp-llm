"""Configuration changes data; Python retains the execution and acceptance contracts."""

import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexlb_test_framework.case_config import configure_program
from flexlb_test_framework.scenario import (
    ScenarioError,
    compile_scenarios,
    load_scenarios,
)
from flexlb_test_framework.scenario.catalog import handlers
from flexlb_test_framework.scenario.loader import load_document


class CaseConfigTest(unittest.TestCase):
    def test_debug_and_no_fetch_keep_retained_smoke_construction(self):
        from flexlb_cfg import ConfigOverride, render_env

        plans = compile_scenarios(
            load_scenarios(ROOT / "scenarios/status/status_protocol.yaml"),
            handlers=handlers(),
        )
        selected = [
            p for p in plans if p["variant_id"] in ("debug_snapshot", "normal_no_fetch")
        ]
        self.assertEqual(len(selected), 2)
        for plan in selected:
            self.assertEqual(
                (plan["environment"]["n_prefill"], plan["environment"]["n_decode"]),
                (2, 4),
            )
            self.assertEqual(
                plan["environment"]["resolved_config"],
                json.loads(
                    render_env("batch-window", ConfigOverride(request_timeout_ms=30000))
                ),
            )

    def config(self):
        config = load_document(ROOT / "scenarios/core/request_completion.yaml")
        # These tests vary request cohort count/PD scale. The dedicated
        # missing-Fetch experiment has its own fixed topology and parameters.
        config["variants"] = [
            v for v in config["variants"] if v["id"] in ("immediate", "deferred_fetch")
        ]
        return config

    def compile(self, config):
        return compile_scenarios(
            [("config.yaml", configure_program(config, "config.yaml"))],
            handlers=handlers(),
        )

    def test_pd_scale_reaches_environment_and_resource_budget(self):
        config = self.config()
        config["environment"].update(n_prefill=5, n_decode=7)
        for plan in self.compile(config):
            self.assertEqual(plan["environment"]["n_prefill"], 5)
            self.assertEqual(plan["environment"]["n_decode"], 7)
            self.assertEqual(plan["resource_budget"]["initial_workers"], 12)
            self.assertEqual(plan["implementation"]["language"], "python")

    def test_extra_configuration_reuses_python_case_without_new_code(self):
        config = self.config()
        config["variants"] = [
            {
                "id": "large_pd",
                "program": "immediate",
                "profiles": ["single-nonbatch"],
                "environment": {"n_prefill": 4, "n_decode": 8},
                "parameters": {"input_len": 8192, "output_len": 16, "count": 3},
            }
        ]
        (plan,) = self.compile(config)
        self.assertEqual(plan["id"], "request_completion::large_pd::single-nonbatch")
        self.assertEqual(plan["environment"]["n_decode"], 8)
        self.assertEqual(plan["stages"][1]["params"]["input_len"], 8192)
        self.assertEqual(plan["stages"][1]["params"]["count"], 3)
        self.assertEqual(plan["stages"][4]["params"]["expected"], 0)

    def test_parameters_are_scoped_to_their_configuration(self):
        config = self.config()
        config["parameters"]["count"] = 2
        config["variants"][0]["parameters"] = {"count": 4}
        original = copy.deepcopy(config)
        plans = self.compile(config)
        self.assertEqual(
            {
                p["stages"][1]["params"]["count"]
                for p in plans
                if p["variant_id"] == "immediate"
            },
            {4},
        )
        self.assertEqual(
            {
                p["stages"][1]["params"]["count"]
                for p in plans
                if p["variant_id"] == "deferred_fetch"
            },
            {2},
        )
        self.assertEqual(config, original)
        self.assertEqual(
            self.compile(self.config())[0]["stages"][1]["params"]["count"], 1
        )

    def test_new_yaml_file_can_select_an_existing_python_program(self):
        config = self.config()
        config["id"] = "another_pd_scale"
        with tempfile.TemporaryDirectory() as directory:
            for name, data in (
                ("original.json", self.config()),
                ("extra.json", config),
            ):
                (Path(directory) / name).write_text(json.dumps(data))
            plans = compile_scenarios(load_scenarios(directory), handlers=handlers())
        self.assertEqual(len(plans), 12)
        self.assertEqual(
            {p["scenario_id"] for p in plans},
            {"request_completion", "another_pd_scale"},
        )

    def test_rejects_orchestration_anywhere_in_yaml(self):
        for key in (
            "stages",
            "steps",
            "stage_overrides",
            "action",
            "$ref",
            "when",
            "needs",
        ):
            with self.subTest(key=key):
                config = self.config()
                config["variants"][0]["parameters"] = {key: "anything"}
                with self.assertRaisesRegex(ScenarioError, "cannot orchestrate"):
                    self.compile(config)

    def test_rejects_missing_parameters_and_invalid_numeric_values(self):
        for parameters in (
            {"coutn": 2},
            {"count": True},
            {"count": 0},
            {"count": 10001},
            {"input_len": "2048"},
        ):
            with self.subTest(parameters=parameters):
                config = self.config()
                config["parameters"] = parameters
                with self.assertRaises(ScenarioError):
                    self.compile(config)

    def test_configuration_cannot_import_or_orchestrate(self):
        for patch in (
            {"case": "os.system"},
            {"findings": ["no_errors.comparison"]},
            {"stages": []},
        ):
            with self.subTest(patch=patch):
                config = self.config()
                config.update(patch)
                with self.assertRaises(ScenarioError):
                    self.compile(config)

    def test_rejects_unknown_and_duplicate_programs(self):
        variants = (
            [{"id": "unknown"}],
            [{"id": "immediate"}, {"id": "immediate"}],
        )
        for value in variants:
            with self.subTest(variants=value):
                config = self.config()
                config["variants"] = value
                with self.assertRaises(ScenarioError):
                    self.compile(config)

    def test_old_yaml_execution_language_is_rejected_at_file_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.yaml"
            path.write_text("schema_version: 1\nid: old\nstages: []\n")
            with self.assertRaisesRegex(ScenarioError, "cannot orchestrate"):
                load_scenarios(path)

    def test_yaml_controls_timeouts_expectations_and_metadata(self):
        config = self.config()
        config["metadata"]["description"] = "Configured in YAML"
        config["parameters"]["completion"]["setup_timeout_s"] = 211
        config["parameters"]["completion"]["expected"]["no_errors"] = 7
        for plan in self.compile(config):
            self.assertEqual(plan["description"], "Configured in YAML")
            self.assertEqual(plan["stages"][0]["timeout_s"], 211)
            self.assertEqual(plan["stages"][4]["params"]["expected"], 7)

    def test_missing_yaml_data_has_no_python_fallback(self):
        config = self.config()
        del config["parameters"]["completion"]["setup_timeout_s"]
        with self.assertRaisesRegex(ScenarioError, "missing YAML parameter"):
            self.compile(config)

    def test_numeric_constraints_and_profiles_are_yaml_owned(self):
        config = self.config()
        config["variants"] = [{"id": "custom", "program": "immediate"}]
        config["profiles"] = ["single-nonbatch"]
        config["parameters"]["count"] = 10001
        config["parameter_schema"]["count"]["maximum"] = 10001
        (plan,) = self.compile(config)
        self.assertEqual(plan["stages"][1]["params"]["count"], 10001)
        del config["profiles"]
        with self.assertRaisesRegex(ScenarioError, "profiles must"):
            self.compile(config)

    def test_nested_variant_data_is_isolated(self):
        config = self.config()
        config["variants"][0]["parameters"] = {
            "completion": {"expected": {"no_errors": 3}}
        }
        for plan in self.compile(config):
            expected = 3 if plan["variant_id"] == "immediate" else 0
            self.assertEqual(plan["stages"][4]["params"]["expected"], expected)
            self.assertTrue(plan["stages"][3]["params"]["expected"])

    def test_programs_do_not_declare_configuration_tables_or_defaults(self):
        for path in (ROOT / "flexlb_test_framework/case_programs").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.assertNotIn(
                                target.id,
                                {"VARIANTS", "PROFILES", "METADATA"},
                                str(path),
                            )
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "number":
                        self.assertEqual(len(node.args), 1, str(path))
                        self.assertEqual(node.keywords, [], str(path))

    def test_shipped_inventory_is_data_only_and_keeps_all_checks(self):
        documents = load_scenarios(ROOT / "scenarios")
        plans = compile_scenarios(documents, handlers=handlers())
        expected = json.loads((ROOT / "tests/fixtures/instance_ids.json").read_text())
        from flexlb_test_framework.scenario.loader import load_document

        aliases = load_document(ROOT / "suites.yaml")["covered_instances"]
        self.assertEqual(len(aliases), 9)
        self.assertTrue(set(aliases).issubset(expected))
        expected = sorted({aliases.get(identity, identity) for identity in expected})
        self.assertEqual(sorted(p["id"] for p in plans), expected)
        self.assertTrue(all(any(s["check_ids"] for s in p["stages"]) for p in plans))
        for source, _ in documents:
            config = load_document(source)
            self.assertEqual(config["schema_version"], 2)
            self.assertNotIn('"$ref"', json.dumps(config))
            self.assertNotIn('"stages"', json.dumps(config))


if __name__ == "__main__":
    unittest.main()
