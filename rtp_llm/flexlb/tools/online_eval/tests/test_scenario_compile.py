"""No-process compilation: errors, reference types, explicit variants and SSOT."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from flexlb_cfg import render_env
from flexlb_test_framework.scenario import (
    ScenarioError,
    compile_scenarios,
    load_scenarios,
)
from flexlb_test_framework.scenario.loader import load_document


def scenario():
    return {
        "schema_version": 1,
        "id": "completion",
        "description": "A submitted request reaches terminal state",
        "category": "status",
        "environment": {"backend": "java_mock"},
        "stages": [
            {"id": "setup", "action": "setup"},
            {"id": "submit", "action": "request", "params": {"output_len": 2}},
            {
                "id": "finish",
                "action": "wait",
                "params": {"requests": {"$ref": "stages.submit.output.requests"}},
            },
            {
                "id": "complete",
                "action": "check",
                "params": {
                    "actual": {"$ref": "stages.finish.output.completed"},
                    "op": "eq",
                    "expected": True,
                },
            },
        ],
    }


class CompileTest(unittest.TestCase):
    def compile(self, doc, profile=None):
        return compile_scenarios([("test.yaml", doc)], profile)

    def test_four_profiles_render_ssot_and_no_import_of_process_framework(self):
        plans = self.compile(scenario())
        self.assertEqual(len(plans), 4)
        for plan in plans:
            self.assertEqual(
                plan["environment"]["resolved_config"],
                json.loads(render_env(plan["profile"])),
            )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from flexlb_test_framework.scenario import compile_scenarios; assert not any(x in sys.modules for x in ['flexlb_test_framework.harness','flexlb_test_framework.engine_ops','grpc'])",
            ],
            cwd=TOOLS,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_explicit_variants_do_not_form_cartesian_products_or_mutate_inputs(self):
        doc = scenario()
        doc["variants"] = [
            {
                "id": "short",
                "profiles": ["single-batch"],
                "stage_overrides": {"submit": {"output_len": 1}},
            },
            {
                "id": "long",
                "profiles": ["window-nonbatch"],
                "stage_overrides": {"submit": {"output_len": 20}},
                "environment_overrides": {"n_prefill": 3},
            },
        ]
        original = copy.deepcopy(doc)
        plans = self.compile(doc)
        self.assertEqual(
            [p["id"] for p in plans],
            ["completion::short::single-batch", "completion::long::window-nonbatch"],
        )
        self.assertEqual(
            [p["stages"][1]["params"]["output_len"] for p in plans], [1, 20]
        )
        plans[0]["stages"][1]["params"]["output_len"] = 999
        self.assertEqual(plans[1]["stages"][1]["params"]["output_len"], 20)
        self.assertEqual(doc, original)

    def test_profile_filter_does_not_relabel_environment(self):
        plans = self.compile(scenario(), "single-nonbatch")
        self.assertEqual(len(plans), 1)
        self.assertEqual(
            plans[0]["environment"]["resolved_config"]["dispatcher"]["type"],
            "NON_BATCH",
        )

    def test_unknown_field_and_version_are_errors_with_source(self):
        for patch in (
            {"schema_version": True},
            {"schema_version": 2},
            {"python": "exec()"},
            {1: "bad"},
        ):
            with self.subTest(patch=patch), self.assertRaisesRegex(
                ScenarioError, "test.yaml"
            ):
                self.compile({**scenario(), **patch})

    def test_setup_order_and_references_are_checked_before_runtime(self):
        for ref in (
            "stages.later.output.requests",
            "stages.setup.output.environment",
            "__import__('os')",
            "stages.submit.output.missing",
        ):
            doc = scenario()
            doc["stages"][2]["params"]["requests"] = {"$ref": ref}
            with self.subTest(ref=ref), self.assertRaisesRegex(
                ScenarioError, r"stages\[2\]"
            ):
                self.compile(doc)
        doc = scenario()
        doc["stages"][0]["action"] = "request"
        with self.assertRaisesRegex(ScenarioError, "setup"):
            self.compile(doc)

    def test_forged_handle_or_ref_with_extra_keys_is_rejected(self):
        for value in (
            {"kind": "requests", "id": "fake", "env_epoch": 0},
            {"$ref": "stages.submit.output.requests", "value": 1},
        ):
            doc = scenario()
            doc["stages"][2]["params"]["requests"] = value
            with self.assertRaises(ScenarioError):
                self.compile(doc)

    def test_integer_boolean_finite_and_positive_constraints(self):
        for value in (True, -1, 0, 1.5, "1"):
            doc = scenario()
            doc["stages"][1]["params"]["count"] = value
            with self.subTest(value=value), self.assertRaises(ScenarioError):
                self.compile(doc)
        for value in (float("nan"), float("inf"), False, 0):
            doc = scenario()
            doc["execution"] = {"cleanup_timeout_s": value}
            with self.subTest(value=value), self.assertRaises(ScenarioError):
                self.compile(doc)

    def test_deferred_fetch_requires_batch_and_capability_is_not_silently_ignored(self):
        doc = scenario()
        doc["stages"][1]["params"]["consume"] = "deferred"
        with self.assertRaisesRegex(ScenarioError, "deferred.*enqueue_batch"):
            self.compile(doc)
        doc["profiles"] = ["batch-window", "single-batch"]
        self.assertEqual(len(self.compile(doc)), 2)
        doc["requires"] = ["generate_stream"]
        with self.assertRaisesRegex(ScenarioError, "lacks capabilities"):
            self.compile(doc)

    def test_unknown_backend_axes_or_unimplemented_action_rejected(self):
        for environment in (
            {"backend": "gpu"},
            {"tp": 2},
            {"config_overrides": {"dispatcher": "invented"}},
        ):
            doc = scenario()
            doc["environment"] = environment
            with self.subTest(environment=environment), self.assertRaises(
                ScenarioError
            ):
                self.compile(doc)
        doc = scenario()
        doc["stages"][1]["action"] = "press"
        with self.assertRaisesRegex(ScenarioError, "unsupported action"):
            self.compile(doc)

    def test_duplicate_scenario_variant_stage_and_invalid_findings(self):
        with self.assertRaisesRegex(ScenarioError, "duplicate scenario"):
            compile_scenarios([("a", scenario()), ("b", scenario())])
        doc = scenario()
        doc["variants"] = [{"id": "same"}, {"id": "same"}]
        with self.assertRaisesRegex(ScenarioError, "duplicate variant"):
            self.compile(doc)
        doc = scenario()
        doc["stages"][2]["id"] = "submit"
        with self.assertRaisesRegex(ScenarioError, "duplicate stage"):
            self.compile(doc)
        doc = scenario()
        doc["findings"] = ["setup"]
        with self.assertRaises(ScenarioError):
            self.compile(doc)
        doc["findings"] = ["complete.comparison"]
        self.assertEqual(self.compile(doc)[0]["findings"], ["complete.comparison"])

    def test_variants_cannot_expand_profiles_or_change_stage_action(self):
        doc = scenario()
        doc["profiles"] = ["batch-window"]
        doc["variants"] = [{"id": "bad", "profiles": ["single-batch"]}]
        with self.assertRaises(ScenarioError):
            self.compile(doc)
        doc["variants"] = [
            {"id": "bad", "stage_overrides": {"submit": {"action": "probe"}}}
        ]
        with self.assertRaises(ScenarioError):
            self.compile(doc)


class LoaderTest(unittest.TestCase):
    def config(self):
        config = load_document(TOOLS / "scenarios/core/request_completion.yaml")
        config["environment"].update(n_prefill=1, n_decode=1)
        config["profiles"] = ["batch-window", "single-batch"]
        config["variants"] = [{"id": "immediate", "program": "immediate"}]
        return config

    def test_json_yaml_same_compile_plan(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a.json", root / "b.yaml"
            a.write_text(json.dumps(self.config()))
            b.write_text(yaml.safe_dump(self.config()))
            self.assertEqual(load_document(a), load_document(b))
            docs = load_scenarios(root)
            self.assertEqual([Path(s).name for s, _ in docs], ["a.json", "b.yaml"])
            with self.assertRaisesRegex(ScenarioError, "duplicate scenario"):
                compile_scenarios(docs)

    def test_duplicate_keys_aliases_tags_and_nonfinite_rejected(self):
        examples = [
            ("bad.json", '{"a":1,"a":2}'),
            ("bad.yaml", "a: 1\na: 2\n"),
            ("bad.yaml", "a: &x [1]\nb: *x\n"),
            ("bad.yaml", "a: !!python/object:os.system {}"),
            ("bad.json", '{"a":NaN}'),
            ("bad.yaml", "a: .inf"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in examples:
                p = Path(tmp) / name
                p.write_text(content)
                with self.subTest(content=content), self.assertRaisesRegex(
                    ScenarioError, name
                ):
                    load_document(p)

    def test_depth_size_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            p = root / "a.json"
            p.write_text('{"a":' + "[" * 40 + "1" + "]" * 40 + "}")
            with self.assertRaisesRegex(ScenarioError, "depth"):
                load_document(p)
            p.write_text(" " * (1024 * 1024 + 1))
            with self.assertRaisesRegex(ScenarioError, "1 MiB"):
                load_document(p)
            p.unlink()
            outside = Path(tmp) / "outside.json"
            outside.write_text("{}")
            p.symlink_to(outside)
            with self.assertRaisesRegex(ScenarioError, "escapes root"):
                load_scenarios(root)

    def test_cli_reports_compile_mode_and_zero_selection_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scenario.json"
            doc = self.config()
            doc["profiles"] = ["single-batch"]
            p.write_text(json.dumps(doc))
            result = subprocess.run(
                [sys.executable, "-m", "flexlb_test_framework.scenario", str(p)],
                cwd=TOOLS,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["mode"], "compile")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "flexlb_test_framework.scenario",
                    str(p),
                    "--profile",
                    "single-nonbatch",
                ],
                cwd=TOOLS,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
