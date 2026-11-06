"""Core performance behavior contracts."""

import ast
import builtins
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.cli import parse_cli
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.entry import (
    build_request_plan,
    create_preview,
    dispatch_plan,
    main,
    run_request,
)
from rtp_llm.test.perf_test.errors import ConfigurationError, PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan, build_execution_plan
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_execution,
    fixture_model,
    fixture_workload,
)

entry_ROOT = Path(__file__).resolve().parents[3]
entry_SCRIPT = entry_ROOT / "tools" / "perf_test"
entry_LOCAL = (
    entry_ROOT / "rtp_llm" / "test" / "perf_test" / "examples" / "prefill_local.json"
)


def dry_run_config(backend):
    execution = {"backend": "local", "devices": [0]}
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model", type="registered_model", checkpoint="/models/test"
        ),
        "engine": fixture_engine(),
        "workload": fixture_workload(),
        "execution": execution,
        "output": {},
        "regression": {},
        "publish": {},
    }


def dry_run_cases():
    return [
        {"input_len": 512, "cache_len": 0, "batch_size": 1},
        {"input_len": 1024, "cache_len": 512, "batch_size": 1},
    ]


public_boundary_ROOT = Path(__file__).parent


class UnifiedEntryTest(unittest.TestCase):

    def test_multi_dp_batch_rejected_before_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = json.loads(entry_LOCAL.read_text())
            source["model"]["topology"]["dp"] = 2
            source["execution"]["devices"] = [0, 1]
            source["output"]["result_root"] = str(root / "results")
            path = root / "config.json"
            for batch_size in (4, 1):
                source["workload"]["batch_sizes"] = [batch_size]
                path.write_text(json.dumps(source))
                request = parse_cli(
                    ["--mode=prefill", "--action=run", f"--config={path}"]
                )
                if batch_size > 1:
                    with self.assertRaisesRegex(PlanningError, "DP.*batch"):
                        build_request_plan(request)
                else:
                    self.assertTrue(build_request_plan(request)[0].cases)
                self.assertEqual(
                    ["config.json"], sorted((item.name for item in root.iterdir()))
                )

    def _run(self, *arguments, cwd=None):
        return subprocess.run(
            [str(entry_SCRIPT), *arguments],
            cwd=cwd or entry_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_repository_entry_help(self):
        completed = self._run("--help")
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--mode", completed.stdout)
        self.assertIn("--action", completed.stdout)
        self.assertIn("--skip-test", completed.stdout)
        self.assertNotIn("--partial", completed.stdout)

    def test_local_dry_run_prints_plan_without_creating_result_root(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = self._run(
                "--mode=prefill",
                "--action=run",
                f"--config={entry_LOCAL}",
                "--dry-run",
                cwd=directory,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            preview = json.loads(completed.stdout)
            self.assertTrue(preview["dry_run"])
            self.assertEqual("local", preview["backend"])
            self.assertGreater(preview["cases"]["count"], 0)
            self.assertFalse((Path(directory) / "perf_test_results").exists())

    def test_bazel_main_module_uses_the_same_help_contract(self):
        completed = subprocess.run(
            ["python", "-m", "rtp_llm.test.perf_test.main", "--help"],
            cwd=entry_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--dry-run", completed.stdout)


class EntryExecutionTest(unittest.TestCase):

    def test_run_directory_is_initialized_before_backend_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            config = {
                "schema_version": 1,
                "model": fixture_model(label="model"),
                "engine": fixture_engine(
                    max_seq_len=1024, max_batch_tokens=1024, concurrency_limit=1
                ),
                "workload": fixture_workload(max_input_len=512, cache_lengths=[0]),
                "execution": fixture_execution(),
                "output": {"result_root": str(result_root)},
                "regression": {},
                "publish": {},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            observed = {}

            def execute(plan):
                run_root = Path(plan.result_directory)
                observed["root"] = run_root
                observed["files"] = {
                    name: (run_root / name).is_file()
                    for name in (
                        "source_config.json",
                        "resolved_config.json",
                        "execution_plan.json",
                        "manifest.json",
                    )
                }
                return Mock(exit_code=0)

            request = parse_cli(
                ["--mode=prefill", "--action=run", f"--config={config_path}"]
            )
            original_import = builtins.__import__

            def entry_import(name, *args, **kwargs):
                if name == "rtp_llm.test.perf_test.prefill.checkpoint":
                    raise ModuleNotFoundError(name)
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=entry_import):
                run_request(request, executor_factory=lambda: {"local": execute})
            self.assertTrue(all(observed["files"].values()))
            source = json.loads(
                (observed["root"] / "source_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config, source)


class DryRunTest(unittest.TestCase):

    def _assert_side_effect_free_preview(self, backend):
        create_run_directory = Mock()
        build_target = Mock()
        start_engine = Mock()
        submit_job = Mock()
        publisher = Mock()

        def execute(_plan):
            create_run_directory()
            build_target()
            start_engine()
            submit_job()
            publisher()

        with tempfile.TemporaryDirectory() as temp_dir:
            result_directory = Path(temp_dir) / "formal-run"
            plan = build_execution_plan(
                mode="prefill",
                action="run",
                resolved=resolve_config(dry_run_config(backend)),
                cases=dry_run_cases(),
                result_directory=str(result_directory),
            )
            preview = dispatch_plan(plan, dry_run=True, execute=execute)
            self.assertFalse(result_directory.exists())
            self.assertTrue(preview["dry_run"])
            self.assertEqual(backend, preview["backend"])
            self.assertEqual(
                plan.to_dict()["resolved_config"], preview["normalized_config"]
            )
            self.assertEqual(2, preview["cases"]["count"])
            self.assertEqual(
                [case.case_id for case in plan.cases], preview["cases"]["case_ids"]
            )
            self.assertEqual(plan.resource_plan.to_dict(), preview["resource_summary"])
        create_run_directory.assert_not_called()
        build_target.assert_not_called()
        start_engine.assert_not_called()
        submit_job.assert_not_called()
        publisher.assert_not_called()
        return preview

    def test_local_dry_run_is_complete_and_side_effect_free(self):
        preview = self._assert_side_effect_free_preview("local")
        self.assertEqual(1, len(preview["commands"]))
        self.assertEqual("local-run", preview["commands"][0]["stage"])
        self.assertIn("execution_plan.json", preview["commands"][0]["argv"][-1])


class PublicBoundaryTest(unittest.TestCase):

    def test_aone_rejected_before_output_or_executor_for_both_modes(self):
        for mode in ("prefill", "decode"):
            for dry in (False, True):
                with self.subTest(
                    mode=mode, dry=dry
                ), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source = json.loads(
                        (
                            public_boundary_ROOT
                            / "examples"
                            / (
                                "prefill_local.json"
                                if mode == "prefill"
                                else "decode_grid_local.json"
                            )
                        ).read_text()
                    )
                    source["execution"] = {
                        "backend": "aone",
                        "resource": "daily",
                        "gpu_type": "H20",
                        "max_parallel": 1,
                        "shard_count": 1,
                    }
                    source["output"]["result_root"] = str(root / "results")
                    config = root / "config.json"
                    config.write_text(json.dumps(source))
                    args = [f"--mode={mode}", "--action=run", f"--config={config}"] + (
                        ["--dry-run"] if dry else []
                    )
                    executor_factory = Mock()
                    self.assertEqual(2, main(args, executor_factory=executor_factory))
                    executor_factory.assert_not_called()
                    self.assertEqual(
                        ["config.json"], sorted((x.name for x in root.iterdir()))
                    )

    def test_historical_plan_roundtrip_is_read_only(self):
        payload = json.loads(
            (public_boundary_ROOT / "testdata/historical_aone_plan.json").read_text()
        )
        plan = ExecutionPlan.from_dict(payload)
        self.assertEqual(payload, plan.to_dict())
        with self.assertRaises(PlanningError):
            create_preview(plan)

    def test_decode_follow_up_actions_rejected_before_output_or_executor(self):
        for action in ("resume", "profile", "report"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_directory = root / "run"
                (plan, source) = build_request_plan(
                    parse_cli(
                        [
                            "--mode=decode",
                            "--action=run",
                            f"--config={public_boundary_ROOT / 'examples/decode_grid_local.json'}",
                        ]
                    )
                )
                plan = replace(plan, result_directory=str(run_directory))
                initialize_run_directory(
                    run_directory,
                    source_config=source,
                    plan=plan,
                    run_id="run",
                    code_version="test",
                )
                before = {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                }
                executor_factory = Mock()
                self.assertEqual(
                    2,
                    main(
                        [
                            "--mode=decode",
                            f"--action={action}",
                            f"--run-dir={run_directory}",
                        ],
                        executor_factory=executor_factory,
                    ),
                )
                executor_factory.assert_not_called()
                self.assertEqual(
                    before,
                    {
                        str(path.relative_to(root)): path.read_bytes()
                        for path in root.rglob("*")
                        if path.is_file()
                    },
                )


class RuntimeBuildClosureTest(unittest.TestCase):

    def test_declared_runtime_sources_import_in_isolated_tree(self):
        root = Path(__file__).parent
        build = ast.parse((root / "BUILD").read_text())
        libraries = {}
        for node in build.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == "py_library":
                    attrs = {item.arg: item.value for item in call.keywords}
                    libraries[ast.literal_eval(attrs["name"])] = ast.literal_eval(
                        attrs["srcs"]
                    )
        sources = set(
            libraries["perf_test_runtime_lib"] + libraries["perf_test_entry_lib"]
        )
        prefix = "rtp_llm.test.perf_test."
        missing = set()
        for source in sources:
            for node in ast.walk(ast.parse((root / source).read_text())):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                    modules += [node.module + "." + alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module in modules:
                    if module.startswith(prefix):
                        candidate = module[len(prefix) :].replace(".", "/") + ".py"
                        if (root / candidate).is_file() and candidate not in sources:
                            missing.add(candidate)
        self.assertEqual(
            set(), missing, "runtime imports missing from declared closure"
        )
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory)
            package = isolated / "rtp_llm/test/perf_test"
            for source in sources:
                target = package / source
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / source, target)
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys, types; sys.path.insert(0, sys.argv[1]); sys.modules['yaml'] = types.ModuleType('yaml'); tokenizers = types.ModuleType('transformers'); tokenizers.AutoTokenizer = object; sys.modules['transformers'] = tokenizers; fuser = types.ModuleType('rtp_llm.utils.fuser'); fuser.fetch_remote_file_to_local = lambda path: path; sys.modules['rtp_llm.utils.fuser'] = fuser; from rtp_llm.test.perf_test.multi_node import multi_benchmark; assert callable(multi_benchmark.validate_benchmark)",
                    str(isolated),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)


class CliParsingTest(unittest.TestCase):

    def test_skip_test_alias_selects_offline_prefill_report(self):
        request = parse_cli(["--mode=prefill", "--skip-test", "--run-dir=runs/example"])
        self.assertEqual("report", request.action)
        self.assertEqual("runs/example", request.run_dir)
        explicit = parse_cli(
            [
                "--mode=prefill",
                "--action=report",
                "--skip-test",
                "--run-dir=runs/example",
            ]
        )
        self.assertEqual(request, explicit)

    def test_action_source_conflicts_are_exit_code_two(self):
        invalid = [
            ["--mode", "prefill", "--action", "run"],
            [
                "--mode",
                "prefill",
                "--action",
                "run",
                "--config",
                "config.json",
                "--run-dir",
                "runs/example",
            ],
        ]
        for argv in invalid:
            with self.subTest(argv=argv):
                with self.assertRaises(ConfigurationError) as context:
                    parse_cli(argv)
                self.assertEqual(2, context.exception.exit_code)


if __name__ == "__main__":
    unittest.main()
