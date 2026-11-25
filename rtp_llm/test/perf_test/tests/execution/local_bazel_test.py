"""Local submissions enter Bazel once and workers consume frozen plans."""

import io
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test import entry, local_main
from rtp_llm.test.perf_test.cli import CliRequest, parse_cli
from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.execution.local_bazel import (
    execute_local_bazel,
    local_command,
)
from rtp_llm.test.perf_test.tests.support.fixtures import fixture_config


class LocalBazelTest(unittest.TestCase):
    def plan(self, directory, **execution):
        return entry.plan_from_source(
            fixture_config(
                execution={
                    "backend": "local",
                    "platform": "cuda13",
                    "test_timeout_seconds": 19500,
                    **execution,
                },
                output={"result_root": str(Path(directory) / "results")},
            ),
            source_dir=Path(directory),
        )

    def request(self):
        return parse_cli(["--config=unused.json", "--bazel-arg=--jobs=2"])

    def test_local_command_uses_real_test_environment_and_existing_env_names(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _ = self.plan(directory)
            command = local_command(
                plan, self.request(), Path(directory) / "bundle.json"
            )
        self.assertEqual("bazelisk", command[0])
        self.assertEqual(
            "//rtp_llm/test/perf_test:perf_test_local",
            command[command.index("test") + 1],
        )
        self.assertEqual(
            1, sum(arg.startswith("--output_user_root=") for arg in command)
        )
        for flag in (
            "--config=cuda13",
            "--jobs=2",
            "--test_timeout=19500",
            "--test_env=GPU_COUNT=1",
            "--test_env=WORLD_SIZE=1",
            "--test_env=RTP_TEST_WATCHDOG_SECONDS=19500",
            "--strategy=TestRunner=local",
            "--nocache_test_results",
        ):
            self.assertIn(flag, command)
        self.assertIn(
            "--test_arg=--run-bundle=" + str(Path(directory) / "bundle.json"), command
        )

    def test_worker_reads_frozen_cases_without_replanning_or_submitting_again(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source = self.plan(directory)
            request = self.request()

            def run(command, **kwargs):
                bundle = Path(
                    next(
                        arg.split("=", 2)[2]
                        for arg in command
                        if arg.startswith("--test_arg=--run-bundle=")
                    )
                )
                frozen = json.loads(bundle.read_text())
                self.assertEqual(plan.to_dict(), frozen["plan"])
                with patch.object(
                    entry, "plan_from_source", side_effect=AssertionError("replanned")
                ), patch.object(
                    local_main,
                    "worker_executors",
                    return_value={"local": lambda active: {"exit_code": 5}},
                ):
                    self.assertEqual(
                        5, local_main.main(["--run-bundle=" + str(bundle)])
                    )
                return subprocess.CompletedProcess(command, 3)

            with patch(
                "rtp_llm.test.perf_test.execution.local_bazel.subprocess", Mock(run=run)
            ):
                result = execute_local_bazel(plan, source, request)
            self.assertEqual(5, result["exit_code"])
            self.assertTrue(
                (Path(plan.result_directory) / "launch_environment.json").is_file()
            )
            environment = json.loads(
                (Path(plan.result_directory) / "launch_environment.json").read_text()
            )["environment"]
            self.assertEqual(os.environ.get("CXX"), environment.get("CXX"))
            if os.environ.get("TEST_SRCDIR"):
                print("Local worker CXX from Bazel:", environment.get("CXX"))

    def test_default_entry_submits_local_before_initializing_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source = self.plan(directory)
            with patch(
                "rtp_llm.test.perf_test.execution.local_bazel.execute_local_bazel",
                return_value={"exit_code": 0},
            ) as submit:
                self.assertEqual(
                    0, entry.execute_plan(plan, source, self.request())["exit_code"]
                )
            submit.assert_called_once()
            self.assertFalse(Path(plan.result_directory).exists())

    def test_local_exit_codes_normalize_signals_and_preserve_worker_failures(self):
        for child_code, worker_code, expected in (
            (-15, None, 143),
            (-9, None, 137),
            (-2, None, 130),
            (-15, 0, 143),
            (-9, 5, 5),
            (3, 0, 3),
            (0, None, 1),
            (0, 0, 0),
            (0, 5, 5),
            (0, -15, 143),
        ):
            with self.subTest(
                child_code=child_code, worker_code=worker_code
            ), tempfile.TemporaryDirectory() as directory:
                plan, source = self.plan(directory)

                def run(command, **kwargs):
                    if worker_code is not None:
                        bundle = Path(
                            next(
                                arg.split("=", 2)[2]
                                for arg in command
                                if arg.startswith("--test_arg=--run-bundle=")
                            )
                        )
                        result_path = Path(
                            json.loads(bundle.read_text())["result_path"]
                        )
                        result_path.write_text(json.dumps({"exit_code": worker_code}))
                    return subprocess.CompletedProcess(command, child_code)

                with patch(
                    "rtp_llm.test.perf_test.execution.local_bazel.subprocess",
                    Mock(run=run),
                ):
                    result = execute_local_bazel(plan, source, self.request())
                self.assertEqual(expected, result["exit_code"])

    def test_dry_run_shows_bazel_command_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source = self.plan(directory)
            request = replace(self.request(), dry_run=True)
            preview = entry.execute_plan(plan, source, request)
            self.assertIn("--config=cuda13", preview["commands"][0]["argv"])
            self.assertFalse(Path(plan.result_directory).exists())

    def test_reserved_bazel_flags_cannot_break_platform_or_worker_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _ = self.plan(directory)
            for flag in (
                "--config=cuda13_arm",
                "--run_under=other",
                "--test_timeout=1",
                "--test_env=GPU_COUNT=0",
                "--test_arg=--config=other.json",
                "--strategy=TestRunner=remote",
            ):
                with self.subTest(flag=flag), self.assertRaises(ConfigurationError):
                    local_command(
                        plan,
                        replace(self.request(), bazel_args=(flag,)),
                        Path(directory) / "bundle",
                    )

    def test_injected_executor_and_report_do_not_launch_bazel(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source = self.plan(directory)
            with patch(
                "rtp_llm.test.perf_test.execution.local_bazel.execute_local_bazel"
            ) as submit:
                result = entry.execute_plan(
                    plan,
                    source,
                    executor_factory=lambda: {"local": lambda active: {"exit_code": 0}},
                )
            self.assertEqual(0, result["exit_code"])
            submit.assert_not_called()

    def test_runtime_bundle_rejects_nonlocal_and_does_not_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, source = self.plan(directory)
            payload = plan.to_dict()
            payload["backend"] = "aone"
            path = Path(directory) / "bundle.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "plan": payload,
                        "source": source,
                        "request": {},
                    }
                )
            )
            self.assertEqual(2, local_main.main(["--run-bundle=" + str(path)]))

    def test_explicit_output_root_and_unknown_bazel_options_pass_through(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, _ = self.plan(directory)
            request = replace(
                self.request(),
                bazel_startup_args=("--output_user_root=" + directory,),
                bazel_args=("--some_future_bazel_option=value",),
            )
            command = local_command(plan, request, Path(directory) / "bundle")
            self.assertEqual(
                1, sum(arg.startswith("--output_user_root=") for arg in command)
            )
            self.assertIn("--some_future_bazel_option=value", command)

    def test_profile_uses_persisted_plan_and_selected_cases_without_reinitializing(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            plan, source = self.plan(directory)
            root = Path(plan.result_directory)
            root.mkdir(parents=True)
            marker = root / "existing-artifact"
            marker.write_text("preserve")
            (root / "launch_request.json").write_text('{"overrides":["original"]}')
            plan = replace(plan, action="profile")
            request = CliRequest(
                plan.mode, "profile", None, str(root), (plan.cases[0].case_id,), False
            )
            bundle = Path(directory) / "bundle.json"
            from dataclasses import asdict

            bundle.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "plan": plan.to_dict(),
                        "source": source,
                        "request": asdict(request),
                        "result_path": str(Path(directory) / "result.json"),
                    }
                )
            )
            with patch.object(
                local_main,
                "worker_executors",
                return_value={"local": lambda active: {"status": "failed"}},
            ) as factory, patch.object(
                entry,
                "initialize_run_directory",
                side_effect=AssertionError("reinitialized"),
            ):
                self.assertEqual(5, local_main.main(["--run-bundle=" + str(bundle)]))
            factory.assert_called_once_with([plan.cases[0].case_id])
            self.assertEqual("preserve", marker.read_text())
            self.assertEqual(
                ["original"],
                json.loads((root / "launch_request.json").read_text())["overrides"],
            )
            self.assertEqual(
                5,
                json.loads((Path(directory) / "result.json").read_text())["exit_code"],
            )

    def test_typed_worker_failure_and_interruption_are_preserved(self):
        from dataclasses import asdict
        from types import SimpleNamespace

        from rtp_llm.test.perf_test.errors import EngineLifecycleError
        from rtp_llm.test.perf_test.state import RunState

        for failure in (
            EngineLifecycleError("startup failed"),
            ValueError("preflight failed"),
            RuntimeError(),
            SimpleNamespace(
                exit_code=143, state=RunState.INTERRUPTED, signal_number=15
            ),
        ):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                plan, source = self.plan(directory)
                bundle = Path(directory) / "bundle.json"
                result_path = Path(directory) / "result.json"
                bundle.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "plan": plan.to_dict(),
                            "source": source,
                            "request": asdict(self.request()),
                            "result_path": str(result_path),
                        }
                    )
                )
                options = (
                    {"side_effect": failure}
                    if isinstance(failure, Exception)
                    else {"return_value": failure}
                )
                with patch.object(entry, "execute_plan", **options):
                    self.assertEqual(
                        getattr(failure, "exit_code", 5),
                        local_main.main(["--run-bundle=" + str(bundle)]),
                    )
                result = json.loads(result_path.read_text())
                self.assertEqual(getattr(failure, "exit_code", 5), result["exit_code"])
                if not isinstance(failure, Exception):
                    self.assertEqual("interrupted", result["state"])
                    self.assertEqual(15, result["signal_number"])

    def test_relative_local_model_paths_are_frozen_in_submission_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").mkdir()
            source = fixture_config()
            source["model"].update(checkpoint="./weights", tokenizer="./weights")
            os.chdir(root)
            try:
                plan, _ = entry.plan_from_source(source, source_dir=root)
            finally:
                os.chdir(original)
            self.assertEqual(
                str(root / "weights"), plan.resolved_config["model"]["checkpoint"]
            )
            self.assertEqual(
                str(root / "weights"), plan.resolved_config["model"]["tokenizer"]
            )


if __name__ == "__main__":
    unittest.main()
