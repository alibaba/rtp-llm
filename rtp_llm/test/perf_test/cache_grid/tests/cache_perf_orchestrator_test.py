import argparse
import hashlib
import io
import json
import shlex
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli
from rtp_llm.test.perf_test.cache_grid.runner import (
    cache_perf_orchestrator as orchestrator,
)


class CachePerfOrchestratorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name) / "workspace"
        self.repo.mkdir()

    def local_args(self, **updates):
        values = dict(
            mode="local-run",
            model_json='{"type":"qwen_2","checkpoint":"/models/qwen"}',
            engine_json='{"max_seq_len":4096,"seq_size_per_block":64,'
            '"args":{"act_type":"BF16"}}',
            grid_json='{"cache_alignment":64,"compute_step":[1024,256]}',
            engine_env_json="{}",
            runtime_env_json="{}",
            result_root=None,
            devices="0",
            bazel="bazelisk",
            bazel_arg=[],
            dry_run=False,
        )
        values.update(updates)
        return argparse.Namespace(**values)

    def aone_args(self, **updates):
        values = dict(
            mode="aone-run",
            resource="daily",
            gpu_type="SM103_CU13",
            build_jobs=None,
            max_parallel=1,
        )
        values.update(updates)
        return self.local_args(**values)

    def local_argv(self):
        args = self.local_args()
        return [
            "local-run",
            "--devices=0",
            f"--model-json={args.model_json}",
            f"--engine-json={args.engine_json}",
            f"--grid-json={args.grid_json}",
        ]

    def aone_argv(self):
        return [
            "aone-run",
            "--resource=daily",
            "--gpu-type=SM103_CU13",
            *self.local_argv()[2:],
        ]

    def test_cli_rejects_cross_mode_options_before_planning(self):
        cases = [
            (self.local_argv(), "--resource=daily"),
            (self.local_argv(), "--gpu-type=SM103_CU13"),
            (self.local_argv(), "--build-jobs=2"),
            (self.local_argv(), "--max-parallel=2"),
            (self.aone_argv(), "--devices=0"),
        ]
        with patch.object(orchestrator, "build_local_plan") as local, patch.object(
            orchestrator, "build_aone_plan"
        ) as aone:
            for argv, option in cases:
                with self.subTest(option=option), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        orchestrator.main([*argv, option])
                    self.assertEqual(error.exception.code, 2)
        local.assert_not_called()
        aone.assert_not_called()
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_cli_requires_json_arguments_and_mode_specific_selection(self):
        for argv in (self.local_argv(), self.aone_argv()):
            for option in argv[1:]:
                with self.subTest(mode=argv[0], missing=option.split("=")[0]):
                    with redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            orchestrator.parser().parse_args(
                                [value for value in argv if value != option]
                            )
                    self.assertEqual(error.exception.code, 2)

    def test_cli_defaults_environment_and_rejects_unknown_resource(self):
        for argv in (self.local_argv(), self.aone_argv()):
            args = orchestrator.parser().parse_args(argv)
            self.assertEqual(args.engine_env_json, "{}")
            self.assertEqual(args.runtime_env_json, "{}")
            self.assertEqual(args.bazel, "bazelisk")
            self.assertEqual(args.bazel_arg, [])
        args = orchestrator.parser().parse_args(self.aone_argv())
        self.assertEqual(args.max_parallel, 1)
        self.assertIsNone(args.build_jobs)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            orchestrator.parser().parse_args([*self.aone_argv(), "--resource=unknown"])
        self.assertEqual(error.exception.code, 2)

    def test_cli_dry_run_prints_normalized_plans_without_writes_or_execution(self):
        for argv, stages in (
            (self.local_argv(), ["run"]),
            (self.aone_argv(), ["build", "test"]),
        ):
            output = io.StringIO()
            result_root = self.repo / "results with spaces"
            with self.subTest(mode=argv[0]), patch.object(
                orchestrator, "REPO", self.repo
            ), patch.object(orchestrator.subprocess, "run") as run, patch.object(
                orchestrator, "write_artifacts"
            ) as write, patch.object(
                orchestrator, "write_run_state"
            ) as state, redirect_stdout(
                output
            ):
                code = orchestrator.main(
                    [
                        *argv,
                        f"--result-root={result_root}",
                        "--bazel-arg=--jobs=7",
                        "--dry-run",
                    ]
                )
                self.assertEqual(code, 0)
                run.assert_not_called()
                write.assert_not_called()
                state.assert_not_called()
                self.assertEqual(list(self.repo.iterdir()), [])
                rendered = output.getvalue()
                summary, _ = json.JSONDecoder().raw_decode(rendered)
                self.assertEqual(summary["world_size"], 1)
                self.assertEqual(summary["grid"]["max_input_len"], 4095)
                self.assertEqual(summary["result_root"], str(result_root))
                lines = [
                    line
                    for line in rendered.splitlines()
                    if any(line.startswith(f"{stage}: ") for stage in stages)
                ]
                self.assertEqual(len(lines), len(stages))
                for stage, line in zip(stages, lines):
                    command = shlex.split(line.split(": ", 1)[1])
                    self.assertEqual(command, summary["commands"][stage])
                    if stage == "run":
                        self.assertEqual(command[command.index("--") - 1], "--jobs=7")
                    else:
                        self.assertEqual(command[-1], "--jobs=7")

    def test_cli_invalid_request_becomes_usage_error_without_side_effects(self):
        with patch.object(orchestrator, "REPO", self.repo), patch.object(
            orchestrator.subprocess, "run"
        ) as run, redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                orchestrator.main([*self.local_argv(), "--model-json={}"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("error:", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())
        self.assertEqual(list(self.repo.iterdir()), [])
        run.assert_not_called()

    def credential_engine_json(self):
        engine = json.loads(self.local_args().engine_json)
        engine["args"].update(
            {
                "api_key": "review-sentinel-only",
                "REMOTE_PASSWORD": "review-sentinel-only",
                "client_secret": "review-sentinel-only",
                "access_key_id": "review-sentinel-only",
                "credential_file": "review-sentinel-only",
                "token": "review-sentinel-only",
                "hf_token": "review-sentinel-only",
                "max_tokens": 128,
                "tokenizer_mode": "auto",
            }
        )
        return json.dumps(engine)

    def test_plan_summaries_redact_credentials_without_changing_runtime_artifacts(self):
        engine_json = self.credential_engine_json()
        original_args = json.loads(engine_json)["args"]
        for args, builder, profile_name in (
            (
                self.local_args(engine_json=engine_json),
                orchestrator.build_local_plan,
                "profile.snapshot.json",
            ),
            (
                self.aone_args(engine_json=engine_json),
                orchestrator.build_aone_plan,
                "profile.json",
            ),
        ):
            with self.subTest(mode=args.mode):
                plan = builder(args, repo=self.repo, run_id="redaction", inherited={})
                runtime_profile = json.loads(
                    plan.artifacts[plan.paths.inputs_dir / profile_name]
                )
                for key, value in original_args.items():
                    self.assertEqual(plan.resolved.engine["args"][key], value)
                    self.assertEqual(runtime_profile["engine_args"][key], value)
                summary_args = plan.summary["engine"]["args"]
                self.assertEqual(set(summary_args), set(plan.resolved.engine["args"]))
                self.assertEqual(summary_args["act_type"], "BF16")
                self.assertEqual(summary_args["max_tokens"], 128)
                self.assertEqual(summary_args["tokenizer_mode"], "auto")
                self.assertEqual(summary_args["reuse_cache"], 1)
                for key, value in original_args.items():
                    if value == "review-sentinel-only":
                        self.assertEqual(summary_args[key], "<redacted>")
                self.assertNotIn("review-sentinel-only", json.dumps(plan.summary))

    def test_cli_dry_run_never_prints_credential_engine_values(self):
        for argv in (self.local_argv(), self.aone_argv()):
            output = io.StringIO()
            with self.subTest(mode=argv[0]), patch.object(
                orchestrator, "REPO", self.repo
            ), redirect_stdout(output):
                self.assertEqual(
                    orchestrator.main(
                        [
                            *argv,
                            f"--engine-json={self.credential_engine_json()}",
                            "--dry-run",
                        ]
                    ),
                    0,
                )
                rendered = output.getvalue()
                self.assertNotIn("review-sentinel-only", rendered)
                summary, _ = json.JSONDecoder().raw_decode(rendered)
                self.assertEqual(summary["engine"]["args"]["api_key"], "<redacted>")
                self.assertEqual(summary["engine"]["args"]["act_type"], "BF16")
                self.assertEqual(list(self.repo.iterdir()), [])

    def test_cli_executes_selected_plan_and_propagates_exit_code(self):
        for argv, executor, other in (
            (self.local_argv(), "execute_local_plan", "execute_aone_plan"),
            (self.aone_argv(), "execute_aone_plan", "execute_local_plan"),
        ):
            with self.subTest(mode=argv[0]), patch.object(
                orchestrator, "REPO", self.repo
            ), patch.object(
                orchestrator, executor, return_value=9
            ) as execute, patch.object(
                orchestrator, other
            ) as unused:
                self.assertEqual(orchestrator.main(argv), 9)
                self.assertEqual(execute.call_args.args[0].mode, argv[0])
                unused.assert_not_called()

    def test_aone_plan_expands_resource_and_separates_concurrency(self):
        args = self.aone_args(build_jobs=64, max_parallel=2)
        plan = orchestrator.build_aone_plan(
            args, repo=self.repo, run_id="unit", inherited={}
        )
        common = [
            "--config=cuda13",
            "--config=cicd",
            "--config=daily_aone_bazel_cache",
            "--config=daily_aone_bazel_remote",
        ]
        self.assertEqual(
            plan.commands["build"],
            [
                "bazelisk",
                "build",
                plan.ci_plan["suite_target"],
                *common,
                "--jobs=64",
                "--remote_download_outputs=minimal",
                "--keep_going",
            ],
        )
        self.assertEqual(
            plan.commands["test"],
            [
                "bazelisk",
                "test",
                plan.ci_plan["suite_target"],
                *common,
                "--strategy=TestRunner=remote",
                "--jobs=2",
                "--test_timeout=7000",
                "--test_output=errors",
                "--keep_going",
                "--remote_download_outputs=all",
                "--nocache_test_results",
                "--run_under=//rtp_llm/test/utils:gpu_lock",
            ],
        )
        self.assertEqual(plan.ci_plan["gpu_count"], 1)
        self.assertEqual(plan.mode, "aone-run")
        self.assertEqual(plan.summary["effective_case_count"], len(plan.request_cases))

    def test_aone_plan_maps_supported_pools_and_online_resource(self):
        for gpu_type, platform in (
            ("SM100_ARM_CU13", "cuda13_arm"),
            ("SM103_CU13", "cuda13"),
            ("L20_CU13", "cuda13"),
            ("L20D_DEV", "cuda13"),
        ):
            with self.subTest(gpu_type=gpu_type):
                plan = orchestrator.build_aone_plan(
                    self.aone_args(resource="online", gpu_type=gpu_type),
                    repo=self.repo,
                    run_id="pool",
                    inherited={},
                )
                expected = [
                    f"--config={platform}",
                    "--config=cicd",
                    "--config=online_aone_bazel_cache",
                    "--config=online_aone_bazel_remote",
                ]
                for command in plan.commands.values():
                    self.assertEqual(command[3:7], expected)
                profile = json.loads(
                    plan.artifacts[plan.paths.inputs_dir / "profile.json"]
                )
                self.assertEqual(
                    profile["bazel"]["configs"],
                    [
                        platform,
                        "cicd",
                        "online_aone_bazel_cache",
                        "online_aone_bazel_remote",
                    ],
                )
                self.assertEqual(plan.ci_plan["gpu_type"], gpu_type)

    def test_aone_plan_rejects_unknown_gpu_pool_before_artifacts(self):
        with patch.object(orchestrator.ci, "build_input_artifacts") as inputs:
            with self.assertRaisesRegex(ValueError, "supported GPU types"):
                orchestrator.build_aone_plan(
                    self.aone_args(gpu_type="UNKNOWN_CU13"),
                    repo=self.repo,
                    run_id="bad",
                )
        inputs.assert_not_called()
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_aone_plan_rejects_unknown_resource(self):
        with self.assertRaisesRegex(ValueError, "supported resources"):
            orchestrator.build_aone_plan(
                self.aone_args(resource="unknown"), repo=self.repo, run_id="bad"
            )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_aone_plan_defaults_build_jobs_to_available_cpus(self):
        for cpu_count, expected in (
            (8, "--jobs=8"),
            (None, "--jobs=1"),
            (0, "--jobs=1"),
        ):
            with self.subTest(cpu_count=cpu_count), patch.object(
                orchestrator.os, "cpu_count", return_value=cpu_count
            ):
                plan = orchestrator.build_aone_plan(
                    self.aone_args(), repo=self.repo, run_id="cpus", inherited={}
                )
                self.assertIn(expected, plan.commands["build"])
                self.assertIn("--jobs=1", plan.commands["test"])

    def test_aone_plan_rejects_nonpositive_concurrency_before_paths_or_artifacts(self):
        for option in ("build_jobs", "max_parallel"):
            for value in (0, -1):
                with self.subTest(option=option, value=value), patch.object(
                    orchestrator, "make_run_paths", wraps=orchestrator.make_run_paths
                ) as paths, patch.object(
                    orchestrator.ci,
                    "build_input_artifacts",
                    wraps=orchestrator.ci.build_input_artifacts,
                ) as artifacts, patch.object(
                    Path, "mkdir"
                ) as mkdir:
                    with self.assertRaisesRegex(
                        ValueError, option.replace("_", "-") + ".*positive"
                    ):
                        orchestrator.build_aone_plan(
                            self.aone_args(**{option: value}), repo=self.repo
                        )
                    paths.assert_not_called()
                    artifacts.assert_not_called()
                    mkdir.assert_not_called()
                    self.assertEqual(list(self.repo.iterdir()), [])

    def test_plans_reject_managed_environment_before_paths_or_artifacts(self):
        for builder, args in (
            (orchestrator.build_local_plan, self.local_args),
            (orchestrator.build_aone_plan, self.aone_args),
        ):
            for category in ("engine_env", "runtime_env"):
                for name in (
                    "GPU_COUNT",
                    "LOCAL_WORLD_SIZE",
                    "WORLD_RANK",
                    "TP_SIZE",
                    "DP_SIZE",
                    "EP_SIZE",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                ):
                    with self.subTest(
                        mode=builder.__name__, category=category, name=name
                    ), patch.object(
                        orchestrator,
                        "make_run_paths",
                        wraps=orchestrator.make_run_paths,
                    ) as paths, patch.object(
                        orchestrator.ci,
                        "build_input_artifacts",
                        wraps=orchestrator.ci.build_input_artifacts,
                    ) as artifacts, patch.object(
                        Path, "mkdir"
                    ) as mkdir:
                        with self.assertRaisesRegex(ValueError, rf"{category}\.{name}"):
                            builder(
                                args(**{f"{category}_json": json.dumps({name: "0"})}),
                                repo=self.repo,
                            )
                        paths.assert_not_called()
                        artifacts.assert_not_called()
                        mkdir.assert_not_called()
                        self.assertEqual(list(self.repo.iterdir()), [])

    def test_derived_topology_environment_overrides_inherited_launch_values(self):
        expected = {
            "WORLD_SIZE": "4",
            "GPU_COUNT": "4",
            "LOCAL_WORLD_SIZE": "4",
            "WORLD_RANK": "0",
            "TP_SIZE": "2",
            "DP_SIZE": "2",
            "EP_SIZE": "2",
        }
        model = '{"type":"qwen_2","checkpoint":"/models/qwen","topology":{"tp":2,"dp":2,"ep":2}}'
        for builder, args in (
            (
                orchestrator.build_local_plan,
                self.local_args(devices="0,1,2,3", model_json=model),
            ),
            (orchestrator.build_aone_plan, self.aone_args(model_json=model)),
        ):
            with self.subTest(mode=args.mode):
                plan = builder(
                    args, repo=self.repo, inherited={name: "0" for name in expected}
                )
                profile_name = (
                    "profile.snapshot.json"
                    if args.mode == "local-run"
                    else "profile.json"
                )
                profile = json.loads(
                    plan.artifacts[plan.paths.inputs_dir / profile_name]
                )
                for name, value in expected.items():
                    self.assertEqual(profile["engine_env"].get(name), value)
                    if args.mode == "local-run":
                        self.assertEqual(plan.process_env[name], value)
                        self.assertIn(
                            f"--engine_env={name}={value}", plan.commands["run"]
                        )
                    else:
                        build = plan.artifacts[
                            plan.paths.generated_dir / "BUILD.bazel"
                        ].decode()
                        self.assertIn(
                            json.dumps(name) + ": " + json.dumps(value), build
                        )
                self.assertEqual(plan.resolved.gpu_count, 4)

    def test_aone_plan_derives_gpu_count_from_topology_and_keeps_env_ownership(self):
        inherited = {"PATH": "/bin", "CC": "old-gcc", "API_TOKEN": "inherited-only"}
        args = self.aone_args(
            model_json='{"type":"qwen_2","checkpoint":"/models/qwen",'
            '"topology":{"tp":2,"dp":2,"pp":2,"ep":4}}',
            engine_env_json='{"MODEL_SETTING":"00123"}',
            runtime_env_json='{"CC":"/usr/bin/gcc"}',
        )
        plan = orchestrator.build_aone_plan(
            args, repo=self.repo, run_id="env", inherited=inherited
        )
        self.assertEqual(plan.ci_plan["gpu_count"], 8)
        self.assertEqual(
            plan.process_env,
            {
                "PATH": "/bin",
                "CC": "/usr/bin/gcc",
                "API_TOKEN": "inherited-only",
            },
        )
        self.assertEqual(inherited["CC"], "old-gcc")
        build_file = plan.artifacts[plan.paths.generated_dir / "BUILD.bazel"].decode()
        self.assertIn("gpu_count = 8", build_file)
        self.assertIn('"WORLD_SIZE": "8"', build_file)
        self.assertIn('"MODEL_SETTING": "00123"', build_file)
        self.assertIn('"CC": "/usr/bin/gcc"', build_file)
        self.assertNotIn("inherited-only", b"".join(plan.artifacts.values()).decode())

    def test_aone_plan_is_write_free_and_assigns_each_effective_case_once(self):
        args = self.aone_args(bazel_arg=["--jobs=7", "--remote_timeout=45m"])
        with patch.object(Path, "mkdir") as mkdir, patch.object(
            Path, "open", side_effect=AssertionError("planning must not open files")
        ), patch.object(Path, "write_bytes") as write_bytes, patch.object(
            Path, "write_text"
        ) as write_text, patch.object(
            orchestrator.subprocess, "run"
        ) as run:
            plan = orchestrator.build_aone_plan(
                args, repo=self.repo, run_id="dry", inherited={}
            )
        mkdir.assert_not_called()
        write_bytes.assert_not_called()
        write_text.assert_not_called()
        run.assert_not_called()
        assigned = [
            case_id for shard in plan.ci_plan["shards"] for case_id in shard["case_ids"]
        ]
        self.assertEqual(sorted(assigned), plan.ci_plan["case_ids"])
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(
            sorted(case["case_id"] for case in plan.request_cases), sorted(assigned)
        )
        self.assertEqual(plan.commands["build"][-2:], args.bazel_arg)
        self.assertEqual(plan.commands["test"][-2:], args.bazel_arg)
        self.assertEqual(list(self.repo.iterdir()), [])
        self.assertEqual(
            {path.parent for path in plan.artifacts},
            {
                plan.paths.inputs_dir,
                plan.paths.generated_dir,
            },
        )
        self.assertEqual(
            plan.ci_plan["suite_target"], "//cache_grid_aone_generated_qwen_dry:all"
        )
        self.assertEqual(
            plan.ci_plan["profile_source"], str(plan.paths.inputs_dir / "profile.json")
        )
        self.assertEqual(
            plan.ci_plan["grid_source"], str(plan.paths.inputs_dir / "grid.json")
        )
        for shard in plan.ci_plan["shards"]:
            payload = json.loads(
                plan.artifacts[plan.paths.generated_dir / shard["grid_file"]]
            )
            self.assertEqual(
                [case["case_id"] for case in payload["cases"]], shard["case_ids"]
            )

    def test_aone_plan_automatically_shards_large_grid_independent_of_parallelism(self):
        args = self.aone_args(
            engine_json='{"max_seq_len":1302,"seq_size_per_block":1,"args":{"act_type":"BF16"}}',
            grid_json='{"cache_alignment":1,"compute_step":[1],"fixed_cache_len":0,"max_cases":1400}',
            max_parallel=1,
        )
        plan = orchestrator.build_aone_plan(
            args, repo=self.repo, run_id="large", inherited={}
        )
        self.assertEqual(len(plan.request_cases), 1301)
        self.assertEqual(plan.ci_plan["shard_count"], 2)
        self.assertIn("--jobs=1", plan.commands["test"])
        assigned = [
            case_id for shard in plan.ci_plan["shards"] for case_id in shard["case_ids"]
        ]
        self.assertEqual(sorted(assigned), list(range(1301)))

    def test_local_plan_uses_one_bazel_run_without_test_timeout(self):
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="unit",
            machine="x86_64",
            inherited={},
        )
        command = plan.commands["run"]
        self.assertEqual(command[:3], ["bazelisk", "run", cli.TARGET])
        self.assertEqual(set(plan.commands), {"run"})
        self.assertIn("--config=cuda13", command)
        self.assertIn("--run_under=//rtp_llm/test/utils:gpu_lock", command)
        self.assertNotIn("--test_timeout", " ".join(command))
        self.assertNotIn("shard", " ".join(command))
        self.assertEqual(plan.summary["effective_case_count"], len(plan.request_cases))
        self.assertGreater(len(plan.request_cases), 0)
        self.assertEqual(plan.mode, "local-run")
        self.assertIsNone(plan.ci_plan)
        self.assertFalse(plan.paths.result_root.exists())

    def test_local_plan_rejects_device_count_and_duplicates(self):
        for devices, message in (
            ("0,1", "requires 1 devices"),
            ("0,0", "must be unique"),
        ):
            with self.subTest(devices=devices), self.assertRaisesRegex(
                ValueError, message
            ):
                orchestrator.build_local_plan(
                    self.local_args(devices=devices), repo=self.repo
                )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_rejects_invalid_device_syntax_before_side_effects(self):
        for devices in ("", "0,", ",0", "gpu0", "-1", "+1", "0,,1", "２", "²", "١"):
            with self.subTest(devices=devices), self.assertRaisesRegex(
                ValueError, "GPU indices"
            ):
                orchestrator.build_local_plan(
                    self.local_args(devices=devices), repo=self.repo
                )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_canonicalizes_devices_in_all_launch_contracts(self):
        args = self.local_args(
            devices=" 02, 005 ",
            model_json='{"type":"qwen_2","checkpoint":"/models/qwen","topology":{"tp":2}}',
        )
        plan = orchestrator.build_local_plan(args, repo=self.repo, inherited={})
        launch = json.loads(plan.artifacts[plan.paths.result_root / cli.MANIFEST])
        self.assertEqual(plan.summary["devices"], ["2", "5"])
        self.assertEqual(launch["devices"], ["2", "5"])
        self.assertEqual(plan.process_env["CUDA_VISIBLE_DEVICES"], "2,5")
        self.assertEqual(launch["env"]["CUDA_VISIBLE_DEVICES"], "2,5")
        self.assertIn("--engine_env=CUDA_VISIBLE_DEVICES=2,5", plan.commands["run"])
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_rejects_duplicates_after_device_canonicalization(self):
        for devices in ("0,00", "02,2"):
            with self.subTest(devices=devices), patch.object(
                orchestrator, "make_run_paths"
            ) as paths, patch.object(
                orchestrator.ci, "build_input_artifacts"
            ) as artifacts:
                with self.assertRaisesRegex(ValueError, "must be unique"):
                    orchestrator.build_local_plan(
                        self.local_args(devices=devices), repo=self.repo
                    )
                paths.assert_not_called()
                artifacts.assert_not_called()
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_saved_resume_and_retest_preserve_gpu_mask_and_environment_fingerprint(
        self,
    ):
        from rtp_llm.test.perf_test.batch_decode_test import (
            _build_cache_resume_config,
            parse_args,
        )
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            resume_config_fingerprint,
        )

        plan = orchestrator.build_local_plan(
            self.local_args(
                devices="2,5",
                model_json='{"type":"qwen_2","checkpoint":"/models/qwen","topology":{"tp":2}}',
                engine_env_json='{"PERF_PROFILE_RUNS":"0"}',
                runtime_env_json='{"CC":"/usr/bin/gcc"}',
            ),
            repo=self.repo,
            inherited={"CUDA_VISIBLE_DEVICES": "7"},
        )
        root = plan.paths.result_root
        orchestrator.write_artifacts(plan.artifacts)
        (root / "cache_grid_results.json").write_text("{}")
        saved = cli.load_saved(root)
        self.assertEqual(saved["env"].get("CUDA_VISIBLE_DEVICES"), "2,5")
        self.assertIn("--engine_env=CUDA_VISIBLE_DEVICES=2,5", saved["runner_args"])
        expected_env = {k: plan.process_env[k] for k in saved["env"]}

        def environment_fingerprint(runner_args, process_env):
            parsed, remaining = parse_args(runner_args)
            with patch.dict("os.environ", process_env, clear=True):
                config = _build_cache_resume_config(
                    parsed, remaining, list(saved["env"]), 64
                )
            environment = config["engine"]["environment"]
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2,5")
            return resume_config_fingerprint(environment)

        command = plan.commands["run"]
        initial_fingerprint = environment_fingerprint(
            command[command.index("--") + 1 :], plan.process_env
        )
        for mode in ("resume", "retest"):
            argv = [mode, "--result-dir", str(root)]
            if mode == "retest":
                argv += ["--cases", "0"]
            replay = cli.build_plan(
                cli.parser().parse_args(argv), {"CUDA_VISIBLE_DEVICES": "7"}
            )
            orchestrator.write_artifacts(
                {
                    replay["destination"] / name: content
                    for name, content in replay["artifacts"].items()
                }
            )
            self.assertEqual(replay["process_env"]["CUDA_VISIBLE_DEVICES"], "2,5")
            self.assertEqual(replay["summary"]["environment"], expected_env)
            runner_args = [
                value[len("--test_arg=") :]
                for value in replay["command"]
                if value.startswith("--test_arg=")
            ]
            self.assertEqual(
                environment_fingerprint(runner_args, replay["process_env"]),
                initial_fingerprint,
            )
            self.assertIn("--test_env=CUDA_VISIBLE_DEVICES=2,5", replay["command"])
            self.assertIn(
                "--test_arg=--engine_env=CUDA_VISIBLE_DEVICES=2,5", replay["command"]
            )
            if mode == "retest":
                frozen = json.loads(replay["artifacts"][cli.MANIFEST])
                self.assertEqual(frozen["env"], expected_env)

    def test_local_plan_derives_device_count_from_tp_dp_pp(self):
        args = self.local_args(
            devices=" 2, 5 ",
            model_json='{"type":"qwen_2","checkpoint":"/models/qwen",'
            '"topology":{"tp":2,"ep":2}}',
        )
        plan = orchestrator.build_local_plan(
            args, repo=self.repo, run_id="tp", machine="x86_64", inherited={}
        )
        self.assertEqual(plan.resolved.gpu_count, 2)
        self.assertEqual(plan.process_env["CUDA_VISIBLE_DEVICES"], "2,5")

    def test_local_plan_selects_arm_config_and_rejects_other_architectures(self):
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="arm",
            machine="aarch64",
            inherited={},
        )
        self.assertIn("--config=cuda13_arm", plan.commands["run"])
        with self.assertRaisesRegex(ValueError, "unsupported local architecture"):
            orchestrator.build_local_plan(
                self.local_args(), repo=self.repo, machine="ppc64le"
            )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_rejects_explicit_device_environment_before_paths_or_artifacts(
        self,
    ):
        result_root = self.repo.parent / "external-results"
        for category in ("engine_env", "runtime_env"):
            for value in ("0", "7"):
                with self.subTest(category=category, value=value):
                    args = self.local_args(
                        result_root=result_root,
                        **{
                            f"{category}_json": json.dumps(
                                {"CUDA_VISIBLE_DEVICES": value}
                            )
                        },
                    )
                    with patch.object(
                        orchestrator,
                        "make_run_paths",
                        wraps=orchestrator.make_run_paths,
                    ) as paths, patch.object(
                        orchestrator.ci,
                        "build_input_artifacts",
                        wraps=orchestrator.ci.build_input_artifacts,
                    ) as artifacts:
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"{category}\.CUDA_VISIBLE_DEVICES.*derived",
                        ):
                            orchestrator.build_local_plan(
                                args,
                                repo=self.repo,
                                run_id="device-env",
                                machine="x86_64",
                                inherited={},
                            )
                    paths.assert_not_called()
                    artifacts.assert_not_called()
                    self.assertFalse(result_root.exists())
                    self.assertEqual(list(self.repo.iterdir()), [])

    def test_local_plan_freezes_categorized_environment_and_launch_manifest(self):
        inherited = {
            "PATH": "/bin",
            "CC": "old-gcc",
            "CUDA_VISIBLE_DEVICES": "7",
            "API_TOKEN": "inherited-only",
        }
        args = self.local_args(
            engine_env_json='{"MODEL_SETTING":"00123"}',
            runtime_env_json='{"CC":"/usr/bin/gcc"}',
        )
        plan = orchestrator.build_local_plan(
            args, repo=self.repo, run_id="env", machine="x86_64", inherited=inherited
        )
        self.assertEqual(plan.process_env["CC"], "/usr/bin/gcc")
        self.assertEqual(plan.process_env["MODEL_SETTING"], "00123")
        self.assertEqual(plan.process_env["API_TOKEN"], "inherited-only")
        self.assertEqual(plan.process_env["PATH"], "/bin")
        self.assertEqual(plan.process_env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(inherited["CC"], "old-gcc")
        command = plan.commands["run"]
        self.assertIn("--engine_env=CC=/usr/bin/gcc", command)
        self.assertIn("--engine_env=MODEL_SETTING=00123", command)
        self.assertNotIn("inherited-only", " ".join(command))
        root = plan.paths.result_root
        profile = json.loads(plan.artifacts[root / "profile.snapshot.json"])
        self.assertEqual(profile["runtime_env"], {"CC": "/usr/bin/gcc"})
        self.assertEqual(profile["engine_env"]["MODEL_SETTING"], "00123")
        self.assertNotIn("CC", profile["engine_env"])
        self.assertNotIn("MODEL_SETTING", profile["runtime_env"])
        launch = json.loads(plan.artifacts[root / cli.MANIFEST])
        self.assertEqual(launch["schema_version"], 2)
        self.assertEqual(launch["mode"], "run")
        self.assertEqual(launch["profile_file"], "profile.snapshot.json")
        self.assertEqual(launch["grid"], str(root / "grid.snapshot.json"))
        self.assertEqual(launch["devices"], ["0"])
        self.assertEqual(
            launch["bazel"],
            {
                "command": "run",
                "configs": ["cuda13"],
                "args": [],
                "executable": "bazelisk",
            },
        )
        self.assertFalse(
            any(value.startswith("--engine_env=") for value in launch["runner_args"])
        )
        self.assertEqual(launch["env"]["CC"], "/usr/bin/gcc")
        self.assertEqual(launch["engine_env"], profile["engine_env"])
        self.assertEqual(launch["runtime_env"], profile["runtime_env"])
        self.assertNotIn("inherited-only", b"".join(plan.artifacts.values()).decode())
        for name in ("profile.snapshot.json", "grid.snapshot.json"):
            self.assertEqual(
                launch["snapshots"][name],
                hashlib.sha256(plan.artifacts[root / name]).hexdigest(),
            )
        grid = json.loads(plan.artifacts[root / "grid.snapshot.json"])
        self.assertEqual(plan.request_cases, grid["cases"])

    def test_local_runner_args_and_bazel_overrides_are_forwarded_in_order(self):
        args = self.local_args(
            bazel="/opt/bazelisk", bazel_arg=["--jobs=2", "--remote_timeout=45m"]
        )
        plan = orchestrator.build_local_plan(
            args, repo=self.repo, run_id="argv", machine="x86_64", inherited={}
        )
        command = plan.commands["run"]
        separator = command.index("--")
        self.assertEqual(command[0], "/opt/bazelisk")
        self.assertEqual(command[separator - 2 : separator], args.bazel_arg)
        runner_args = command[separator + 1 :]
        for value in (
            "--partial=2",
            "--decode_test_length=1",
            "--batch_size=1",
            "--cache_request_transport=dashsc_input_ids",
            "--cache_measure_runs=3",
            "--cache_commit_tail_tokens=64",
        ):
            self.assertIn(value, runner_args)
        self.assertIn(
            f"--profile={plan.paths.result_root / 'profile.snapshot.json'}", runner_args
        )
        self.assertIn(
            f"--cache_grid_json={plan.paths.result_root / 'grid.snapshot.json'}",
            runner_args,
        )
        self.assertIn(f"--result_dir={plan.paths.result_root}", runner_args)

    def test_local_paths_use_sanitized_label_and_no_extra_directories(self):
        paths = orchestrator.make_run_paths(
            self.local_args(),
            repo=self.repo,
            mode="local",
            run_id="unit",
            model_label="Qwen /中文.2",
        )
        self.assertEqual(
            paths.result_root, self.repo / "cache_grid_local_results_Qwen____.2_unit"
        )
        self.assertEqual(paths.inputs_dir, paths.result_root)
        self.assertIsNone(paths.generated_dir)
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_aone_paths_reserve_top_level_input_and_generated_names(self):
        paths = orchestrator.make_run_paths(
            self.local_args(),
            repo=self.repo,
            mode="aone",
            run_id="unit",
            model_label="qwen",
        )
        self.assertEqual(
            paths.result_root, self.repo / "cache_grid_aone_results_qwen_unit"
        )
        self.assertEqual(
            paths.inputs_dir, self.repo / "cache_grid_aone_inputs_qwen_unit"
        )
        self.assertEqual(
            paths.generated_dir, self.repo / "cache_grid_aone_generated_qwen_unit"
        )
        self.assertEqual(list(self.repo.iterdir()), [])

    def test_explicit_result_root_can_be_outside_workspace_but_must_be_empty(self):
        root = self.repo.parent / "external-results"
        args = self.local_args(result_root=root)
        plan = orchestrator.build_local_plan(
            args, repo=self.repo, run_id="external", machine="x86_64", inherited={}
        )
        self.assertEqual(plan.paths.result_root, root)
        self.assertFalse(root.exists())
        root.mkdir()
        orchestrator.build_local_plan(
            args, repo=self.repo, run_id="empty", machine="x86_64", inherited={}
        )
        (root / "existing").write_text("keep")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            orchestrator.build_local_plan(args, repo=self.repo)
        self.assertEqual((root / "existing").read_text(), "keep")

    def test_result_root_file_is_rejected(self):
        root = self.repo / "file"
        root.write_text("keep")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            orchestrator.build_local_plan(
                self.local_args(result_root=root), repo=self.repo
            )

    def test_planning_does_not_write_or_launch_and_resolves_request_once(self):
        with patch.object(
            orchestrator, "resolve_run_request", wraps=orchestrator.resolve_run_request
        ) as resolve, patch.object(Path, "mkdir") as mkdir, patch.object(
            orchestrator.subprocess, "run"
        ) as run:
            orchestrator.build_local_plan(
                self.local_args(),
                repo=self.repo,
                run_id="dry",
                machine="x86_64",
                inherited={},
            )
        self.assertEqual(resolve.call_count, 1)
        mkdir.assert_not_called()
        run.assert_not_called()

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_local_writes_snapshots_before_launch_and_postprocesses(
        self, pipeline, run
    ):
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="ok",
            machine="x86_64",
            inherited={},
        )

        def launch(command, **kwargs):
            self.assertEqual(command, plan.commands["run"])
            self.assertEqual(
                kwargs,
                {"cwd": orchestrator.REPO, "env": plan.process_env, "check": False},
            )
            for path, content in plan.artifacts.items():
                self.assertEqual(path.read_bytes(), content)
            return CompletedProcess(command, 0)

        run.side_effect = launch
        pipeline.return_value = 0
        self.assertEqual(orchestrator.execute_local_plan(plan), 0)
        self.assertEqual(run.call_count, 1)
        pipeline.assert_called_once()
        namespace = pipeline.call_args.args[0]
        self.assertEqual(namespace.result_dir, plan.paths.result_root)
        self.assertTrue(namespace.skip_test)
        self.assertFalse(namespace.partial)
        self.assertFalse(namespace.skip_fit)
        saved = cli.load_saved(plan.paths.result_root)
        self.assertEqual(
            saved["grid"], str(plan.paths.result_root / "grid.snapshot.json")
        )

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_local_preserves_runner_failure(self, pipeline, run):
        run.return_value = CompletedProcess([], 9)
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="fail",
            machine="x86_64",
            inherited={},
        )
        self.assertEqual(orchestrator.execute_local_plan(plan), 9)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("timeout", run.call_args.kwargs)
        pipeline.assert_not_called()
        self.assertTrue((plan.paths.result_root / cli.MANIFEST).is_file())

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_local_returns_postprocess_failure(self, pipeline, run):
        run.return_value = CompletedProcess([], 0)
        pipeline.return_value = 23
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="reportfail",
            machine="x86_64",
            inherited={},
        )
        self.assertEqual(orchestrator.execute_local_plan(plan), 23)

    @patch.object(orchestrator.subprocess, "run")
    def test_execute_local_checks_all_artifact_collisions_before_writing(self, run):
        plan = orchestrator.build_local_plan(
            self.local_args(),
            repo=self.repo,
            run_id="collision",
            machine="x86_64",
            inherited={},
        )
        root = plan.paths.result_root
        root.mkdir()
        (root / cli.MANIFEST).write_bytes(b"existing manifest")
        with self.assertRaises(FileExistsError):
            orchestrator.execute_local_plan(plan)
        self.assertFalse((root / "profile.snapshot.json").exists())
        self.assertFalse((root / "grid.snapshot.json").exists())
        self.assertEqual((root / cli.MANIFEST).read_bytes(), b"existing manifest")
        run.assert_not_called()

    def test_write_artifacts_preflights_every_target_before_creating_parents(self):
        first = self.repo / "new" / "snapshot.json"
        existing = self.repo / "existing.json"
        existing.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            orchestrator.write_artifacts({first: b"new", existing: b"replace"})
        self.assertFalse(first.parent.exists())
        self.assertEqual(existing.read_bytes(), b"keep")

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.ci, "collect_results")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_aone_builds_tests_collects_and_reports(
        self, pipeline, collect, run
    ):
        plan = orchestrator.build_aone_plan(
            self.aone_args(runtime_env_json='{"CC":"/usr/bin/gcc"}'),
            repo=self.repo,
            run_id="ok",
            inherited={"INHERITED": "keep"},
        )
        events = []

        def launch(command, **kwargs):
            events.append(command[1])
            self.assertEqual(command, plan.commands[command[1]])
            self.assertEqual(
                kwargs,
                {"cwd": orchestrator.REPO, "env": plan.process_env, "check": False},
            )
            for path, content in plan.artifacts.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertFalse(plan.paths.result_root.exists())
            return CompletedProcess(command, 0)

        def collect_outputs(args, *, repo):
            events.append("collect")
            self.assertEqual(repo, orchestrator.REPO)
            self.assertEqual(
                args.plan, plan.paths.generated_dir / "cache_perf_ci_plan.json"
            )
            self.assertEqual(args.result_root, plan.paths.result_root)
            self.assertIsNone(args.bazel_testlogs)
            self.assertFalse(args.partial)
            self.assertFalse(args.skip_fit)
            return {"missing_shards": []}, 0

        def report(args):
            events.append("pipeline")
            self.assertTrue(args.skip_test)
            self.assertFalse(args.partial)
            self.assertFalse(args.skip_fit)
            self.assertEqual(args.result_root, plan.paths.result_root)
            self.assertEqual(args.cards, 1)
            return 0

        run.side_effect = launch
        collect.side_effect = collect_outputs
        pipeline.side_effect = report
        self.assertEqual(orchestrator.execute_aone_plan(plan), 0)
        self.assertEqual(events, ["build", "test", "collect", "pipeline"])
        state = json.loads((plan.paths.result_root / "cache_perf_run.json").read_text())
        self.assertEqual(state["schema_version"], 1)
        self.assertEqual(state["mode"], "aone-run")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["commands"], plan.commands)
        self.assertEqual(state["summary"], plan.summary)
        self.assertEqual(
            state["paths"],
            {
                "inputs": str(plan.paths.inputs_dir),
                "generated": str(plan.paths.generated_dir),
                "results": str(plan.paths.result_root),
            },
        )
        self.assertEqual(
            state["stages"],
            {
                "build": {"returncode": 0},
                "test": {"returncode": 0},
                "collect": {"returncode": 0, "missing_shards": []},
                "pipeline": {"returncode": 0},
            },
        )

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.ci, "collect_results")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_aone_stops_after_build_failure(self, pipeline, collect, run):
        plan = orchestrator.build_aone_plan(
            self.aone_args(), repo=self.repo, run_id="buildfail"
        )
        run.return_value = CompletedProcess([], 7)
        self.assertEqual(orchestrator.execute_aone_plan(plan), 7)
        self.assertEqual(run.call_count, 1)
        collect.assert_not_called()
        pipeline.assert_not_called()
        state = json.loads((plan.paths.result_root / "cache_perf_run.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stages"], {"build": {"returncode": 7}})

    def test_execute_aone_records_build_launch_errors_and_interruptions(self):
        for error, expected, status in (
            (OSError("cannot launch build"), 1, "failed"),
            (RuntimeError("build launch failed"), 1, "failed"),
            (KeyboardInterrupt(), 130, "interrupted"),
        ):
            with self.subTest(error=type(error).__name__):
                plan = orchestrator.build_aone_plan(self.aone_args(), repo=self.repo)
                events = []

                def launch(command, **kwargs):
                    events.append(command[1])
                    self.assertFalse(plan.paths.result_root.exists())
                    raise error.with_traceback(None)

                with patch.object(
                    orchestrator.subprocess, "run", side_effect=launch
                ), patch.object(
                    orchestrator.ci,
                    "collect_results",
                    side_effect=AssertionError("must not collect"),
                ), patch.object(
                    orchestrator.unified_pipeline,
                    "run",
                    side_effect=AssertionError("must not report"),
                ):
                    try:
                        code = orchestrator.execute_aone_plan(plan)
                    except (Exception, KeyboardInterrupt) as unhandled:
                        self.fail(
                            f"build launch escaped without terminal state: {unhandled!r}"
                        )
                self.assertEqual(code, expected)
                self.assertEqual(events, ["build"])
                state = json.loads(
                    (plan.paths.result_root / "cache_perf_run.json").read_text()
                )
                self.assertEqual(state["status"], status)
                self.assertEqual(set(state["stages"]), {"build"})
                self.assertEqual(state["stages"]["build"]["returncode"], expected)
                self.assertEqual(state["stages"]["build"]["status"], status)
                self.assertEqual(state["stages"]["build"]["error"], repr(error))

    def test_execute_aone_test_launch_errors_collect_partial_and_preserve_priority(
        self,
    ):
        for error, expected, status in (
            (OSError("cannot launch test"), 1, "failed"),
            (RuntimeError("test launch failed"), 1, "failed"),
            (KeyboardInterrupt(), 130, "interrupted"),
        ):
            for failure in (None, "collect", "pipeline"):
                with self.subTest(error=type(error).__name__, failure=failure):
                    plan = orchestrator.build_aone_plan(
                        self.aone_args(), repo=self.repo
                    )
                    events = []

                    def launch(command, **kwargs):
                        events.append(command[1])
                        self.assertFalse(plan.paths.result_root.exists())
                        if command[1] == "test":
                            raise error.with_traceback(None)
                        return CompletedProcess(command, 0)

                    def collect_outputs(args, *, repo):
                        events.append("collect")
                        self.assertFalse(plan.paths.result_root.exists())
                        self.assertTrue(args.partial)
                        self.assertTrue(args.skip_fit)
                        if failure == "collect":
                            raise ValueError("partial collection failed")
                        return {"missing_shards": ["shard_01"]}, 0

                    def report(args):
                        events.append("pipeline")
                        self.assertTrue(args.partial)
                        self.assertTrue(args.skip_fit)
                        if failure == "pipeline":
                            raise RuntimeError("partial report failed")
                        return 0

                    with patch.object(
                        orchestrator.subprocess, "run", side_effect=launch
                    ), patch.object(
                        orchestrator.ci, "collect_results", side_effect=collect_outputs
                    ), patch.object(
                        orchestrator.unified_pipeline, "run", side_effect=report
                    ):
                        try:
                            code = orchestrator.execute_aone_plan(plan)
                        except (Exception, KeyboardInterrupt) as unhandled:
                            self.fail(
                                f"test launch escaped without partial collection: {unhandled!r}"
                            )
                    self.assertEqual(code, expected)
                    self.assertEqual(
                        events,
                        ["build", "test", "collect"]
                        + ([] if failure == "collect" else ["pipeline"]),
                    )
                    state = json.loads(
                        (plan.paths.result_root / "cache_perf_run.json").read_text()
                    )
                    self.assertEqual(state["status"], status)
                    self.assertEqual(
                        state["stages"]["test"],
                        {
                            "returncode": expected,
                            "status": status,
                            "error": repr(error),
                        },
                    )
                    self.assertEqual(
                        state["stages"]["collect"]["returncode"],
                        int(failure == "collect"),
                    )
                    if failure == "collect":
                        self.assertIn(
                            "partial collection failed",
                            state["stages"]["collect"]["error"],
                        )
                        self.assertNotIn("pipeline", state["stages"])
                    else:
                        self.assertEqual(
                            state["stages"]["collect"]["missing_shards"], ["shard_01"]
                        )
                        self.assertEqual(
                            state["stages"]["pipeline"]["returncode"],
                            int(failure == "pipeline"),
                        )
                        if failure == "pipeline":
                            self.assertIn(
                                "partial report failed",
                                state["stages"]["pipeline"]["error"],
                            )

    def test_execute_aone_does_not_swallow_unrelated_base_exceptions(self):
        plan = orchestrator.build_aone_plan(self.aone_args(), repo=self.repo)
        with patch.object(orchestrator.subprocess, "run", side_effect=SystemExit(42)):
            with self.assertRaises(SystemExit) as error:
                orchestrator.execute_aone_plan(plan)
        self.assertEqual(error.exception.code, 42)

    def write_case_results(self, destination, cases, **metadata):
        metrics = []
        for case in cases:
            inp, cache, batch = (
                case[key] for key in ("input_len", "cache_len", "batch_size")
            )
            requests = [
                dict(
                    success=True,
                    input_len=inp,
                    output_len=1,
                    reuse_len=cache,
                    prefill_time_ms=10 + index,
                )
                for index in range(batch)
            ]
            metrics.append(
                {
                    **case,
                    "request_groups": [
                        dict(count=batch, input_len=inp, cache_len=cache)
                    ],
                    "status": "ok",
                    "measure_runs": 3,
                    "success_runs": 3,
                    "runs": [
                        dict(
                            valid=True,
                            completed_requests=batch,
                            requests=requests,
                            batch_wall_time_ms=20,
                        )
                        for _ in range(3)
                    ],
                }
            )
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "cache_grid_results.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "mode": "prefix_cache_grid",
                    "complete": True,
                    "metrics": metrics,
                    "measure_runs": 3,
                    **metadata,
                }
            )
        )

    def test_aone_test_launch_failure_collects_real_available_partial_outputs(self):
        real_collect = orchestrator.ci.collect_results
        real_pipeline = orchestrator.unified_pipeline.run
        for error, code, status in (
            (OSError("test launch failed"), 1, "failed"),
            (KeyboardInterrupt(), 130, "interrupted"),
        ):
            with self.subTest(error=type(error).__name__):
                plan = orchestrator.build_aone_plan(
                    self.aone_args(
                        grid_json='{"cache_alignment":64,"compute_step":[256],"max_input_len":512,"fixed_cache_len":0}'
                    ),
                    repo=self.repo,
                )
                events = []

                def launch(command, **kwargs):
                    events.append(command[1])
                    if command[1] == "test":
                        shard = plan.ci_plan["shards"][0]
                        self.write_case_results(
                            self.repo
                            / "bazel-testlogs"
                            / plan.ci_plan["package"]
                            / shard["name"]
                            / "test.outputs",
                            plan.request_cases[:1],
                            complete=False,
                            grid_sha256=shard["grid_sha256"],
                            profile_sha256=plan.ci_plan["profile_fingerprint"],
                        )
                        raise error.with_traceback(None)
                    return CompletedProcess(command, 0)

                def collect_outputs(args, *, repo):
                    events.append("collect")
                    self.assertFalse(plan.paths.result_root.exists())
                    self.assertTrue(args.partial)
                    return real_collect(args, repo=repo)

                def report(args):
                    events.append("pipeline")
                    self.assertTrue(args.partial)
                    return real_pipeline(args)

                with patch.object(orchestrator, "REPO", self.repo), patch.object(
                    orchestrator.subprocess, "run", side_effect=launch
                ), patch.object(
                    orchestrator.ci, "collect_results", side_effect=collect_outputs
                ), patch.object(
                    orchestrator.unified_pipeline, "run", side_effect=report
                ), redirect_stdout(
                    io.StringIO()
                ):
                    try:
                        actual = orchestrator.execute_aone_plan(plan)
                    except (Exception, KeyboardInterrupt) as unhandled:
                        self.fail(
                            f"partial outputs lost on test launch failure: {unhandled!r}"
                        )
                self.assertEqual(actual, code)
                self.assertEqual(events, ["build", "test", "collect", "pipeline"])
                state = json.loads(
                    (plan.paths.result_root / "cache_perf_run.json").read_text()
                )
                self.assertEqual(state["status"], status)
                self.assertEqual(state["stages"]["test"]["returncode"], code)
                self.assertEqual(state["stages"]["collect"]["returncode"], 0)
                self.assertEqual(state["stages"]["pipeline"]["returncode"], 0)
                batch = json.loads(
                    (plan.paths.result_root / "batch_runs.json").read_text()
                )
                self.assertEqual(batch["collected_case_ids"], [0])
                self.assertEqual(batch["expected_case_ids"], [0, 1])
                summary = json.loads(
                    (plan.paths.result_root / "pipeline_summary.json").read_text()
                )
                self.assertEqual(summary["status"], "partial")

    def test_aone_records_collection_and_pipeline_interruptions(self):
        for stage in ("collect", "pipeline"):
            with self.subTest(stage=stage):
                plan = orchestrator.build_aone_plan(self.aone_args(), repo=self.repo)
                with patch.object(
                    orchestrator.subprocess, "run", return_value=CompletedProcess([], 0)
                ), patch.object(
                    orchestrator.ci,
                    "collect_results",
                    return_value=({"missing_shards": []}, 0),
                ) as collect, patch.object(
                    orchestrator.unified_pipeline, "run", return_value=0
                ) as pipeline:
                    (collect if stage == "collect" else pipeline).side_effect = (
                        KeyboardInterrupt()
                    )
                    try:
                        code = orchestrator.execute_aone_plan(plan)
                    except KeyboardInterrupt:
                        self.fail(
                            f"{stage} interruption escaped without terminal state"
                        )
                self.assertEqual(code, 130)
                state = json.loads(
                    (plan.paths.result_root / "cache_perf_run.json").read_text()
                )
                self.assertEqual(state["status"], "interrupted")
                self.assertEqual(
                    state["stages"][stage],
                    {
                        "returncode": 130,
                        "status": "interrupted",
                        "error": "KeyboardInterrupt()",
                    },
                )
                if stage == "collect":
                    self.assertNotIn("pipeline", state["stages"])

    def test_nondefault_batch_reaches_real_observations_through_both_executors(self):
        real_pipeline = orchestrator.unified_pipeline.run
        for mode in ("local", "aone"):
            with self.subTest(mode=mode):
                args = getattr(self, f"{mode}_args")(
                    grid_json='{"cache_alignment":64,"compute_step":[256],"max_input_len":512,"fixed_cache_len":0,"batch_size":2}'
                )
                plan = getattr(orchestrator, f"build_{mode}_plan")(
                    args, repo=self.repo, inherited={}
                )

                def launch(command, **kwargs):
                    if mode == "local":
                        self.write_case_results(
                            plan.paths.result_root, plan.request_cases
                        )
                    elif command[1] == "test":
                        for shard in plan.ci_plan["shards"]:
                            cases = [
                                case
                                for case in plan.request_cases
                                if case["case_id"] in shard["case_ids"]
                            ]
                            self.write_case_results(
                                self.repo
                                / "bazel-testlogs"
                                / plan.ci_plan["package"]
                                / shard["name"]
                                / "test.outputs",
                                cases,
                                grid_sha256=shard["grid_sha256"],
                                profile_sha256=plan.ci_plan["profile_fingerprint"],
                            )
                    return CompletedProcess(command, 0)

                def report(args):
                    # Tiny synthetic results exercise collection/charts, not fitting.
                    args.skip_fit = True
                    return real_pipeline(args)

                with patch.object(orchestrator, "REPO", self.repo), patch.object(
                    orchestrator.subprocess, "run", side_effect=launch
                ), patch.object(
                    orchestrator.unified_pipeline, "run", side_effect=report
                ), redirect_stdout(
                    io.StringIO()
                ):
                    try:
                        code = getattr(orchestrator, f"execute_{mode}_plan")(plan)
                    except ValueError as error:
                        self.fail(f"batch=2 observations were rejected: {error}")
                self.assertEqual(code, 0)
                audit = json.loads(
                    (plan.paths.result_root / "report/audit.json").read_text()
                )
                self.assertEqual(audit["rejected_counts"], {})
                self.assertEqual(audit["counts_by_batch"], {"2": 2})
                rows = json.loads(
                    (plan.paths.result_root / "report/observations.json").read_text()
                )
                self.assertEqual([row["batch_size"] for row in rows], [2, 2])
                self.assertEqual([row["target_ms"] for row in rows], [11, 11])
                self.assertEqual(sorted(row["case_id"] for row in rows), [0, 1])
                self.assertTrue(
                    (
                        plan.paths.result_root / "report/latency.interactive.html"
                    ).is_file()
                )

    @patch.object(orchestrator.subprocess, "run")
    @patch.object(orchestrator.ci, "collect_results")
    @patch.object(orchestrator.unified_pipeline, "run")
    def test_execute_aone_partial_collect_preserves_test_failure(
        self, pipeline, collect, run
    ):
        plan = orchestrator.build_aone_plan(
            self.aone_args(), repo=self.repo, run_id="partial"
        )
        run.side_effect = [CompletedProcess([], 0), CompletedProcess([], 9)]
        collect.return_value = ({"missing_shards": ["shard_01"]}, 0)
        pipeline.return_value = 0
        self.assertEqual(orchestrator.execute_aone_plan(plan), 9)
        self.assertTrue(collect.call_args.args[0].partial)
        namespace = pipeline.call_args.args[0]
        self.assertTrue(namespace.partial)
        self.assertTrue(namespace.skip_fit)
        self.assertTrue(namespace.skip_test)
        self.assertIsNone(namespace.oss_destination)
        for name in (
            "formula_output_dir",
            "svg_output",
            "cold_svg_output",
            "html_output",
        ):
            self.assertIsNone(getattr(namespace, name))
        state = json.loads((plan.paths.result_root / "cache_perf_run.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["stages"]["test"]["returncode"], 9)
        self.assertEqual(state["stages"]["collect"]["missing_shards"], ["shard_01"])

    def test_execute_aone_collect_failures_preserve_exit_priority_and_state(self):
        for test_code, collect_code, raises, expected in (
            (9, 0, True, 9),
            (0, 0, True, 1),
            (9, 6, False, 9),
            (0, 6, False, 6),
        ):
            with self.subTest(
                test_code=test_code, collect_code=collect_code, raises=raises
            ):
                plan = orchestrator.build_aone_plan(
                    self.aone_args(),
                    repo=self.repo,
                    run_id=f"collect-{test_code}-{collect_code}-{raises}",
                )
                with patch.object(orchestrator.subprocess, "run") as run, patch.object(
                    orchestrator.ci, "collect_results"
                ) as collect, patch.object(
                    orchestrator.unified_pipeline, "run"
                ) as pipeline:
                    run.side_effect = [
                        CompletedProcess([], 0),
                        CompletedProcess([], test_code),
                    ]
                    if raises:
                        collect.side_effect = ValueError(
                            "no cache-grid results were collected"
                        )
                    else:
                        collect.return_value = (
                            {"missing_shards": ["shard_00"]},
                            collect_code,
                        )
                    self.assertEqual(orchestrator.execute_aone_plan(plan), expected)
                pipeline.assert_not_called()
                self.assertEqual(collect.call_args.args[0].partial, test_code != 0)
                state = json.loads(
                    (plan.paths.result_root / "cache_perf_run.json").read_text()
                )
                self.assertEqual(state["status"], "failed")
                self.assertEqual(state["commands"], plan.commands)
                self.assertEqual(
                    state["stages"]["collect"]["returncode"], 1 if raises else 6
                )
                if raises:
                    self.assertIn(
                        "no cache-grid results", state["stages"]["collect"]["error"]
                    )
                self.assertNotIn("pipeline", state["stages"])

    def test_execute_aone_pipeline_failures_preserve_exit_priority_and_state(self):
        for test_code, pipeline_code, raises, expected in (
            (9, 0, True, 9),
            (0, 0, True, 1),
            (9, 23, False, 9),
            (0, 23, False, 23),
        ):
            with self.subTest(
                test_code=test_code, pipeline_code=pipeline_code, raises=raises
            ):
                plan = orchestrator.build_aone_plan(
                    self.aone_args(),
                    repo=self.repo,
                    run_id=f"pipeline-{test_code}-{pipeline_code}-{raises}",
                )
                with patch.object(orchestrator.subprocess, "run") as run, patch.object(
                    orchestrator.ci, "collect_results"
                ) as collect, patch.object(
                    orchestrator.unified_pipeline, "run"
                ) as pipeline:
                    run.side_effect = [
                        CompletedProcess([], 0),
                        CompletedProcess([], test_code),
                    ]
                    collect.return_value = ({"missing_shards": []}, 0)
                    if raises:
                        pipeline.side_effect = RuntimeError("report failed")
                    else:
                        pipeline.return_value = pipeline_code
                    self.assertEqual(orchestrator.execute_aone_plan(plan), expected)
                state = json.loads(
                    (plan.paths.result_root / "cache_perf_run.json").read_text()
                )
                self.assertEqual(state["status"], "failed")
                self.assertEqual(
                    state["stages"]["pipeline"]["returncode"], 1 if raises else 23
                )
                if raises:
                    self.assertIn("report failed", state["stages"]["pipeline"]["error"])

    def test_write_run_state_replaces_complete_json_atomically(self):
        path = self.repo / "results" / "cache_perf_run.json"
        state = {"status": "running", "label": "中文"}
        orchestrator.write_run_state(path, state)
        replacement = {"status": "complete", "stages": {"test": {"returncode": 0}}}
        replace = orchestrator.os.replace

        def publish(source, destination):
            self.assertEqual(json.loads(path.read_text()), state)
            self.assertEqual(json.loads(source.read_text()), replacement)
            self.assertEqual(source.parent, destination.parent)
            replace(source, destination)

        with patch.object(orchestrator.os, "replace", side_effect=publish):
            orchestrator.write_run_state(path, replacement)
        self.assertEqual(json.loads(path.read_text()), replacement)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_write_artifacts_rejects_dangling_symlink_before_writes(self):
        first = self.repo / "new" / "snapshot.json"
        link = self.repo / "link.json"
        link.symlink_to(self.repo / "absent")
        with self.assertRaises(FileExistsError):
            orchestrator.write_artifacts({first: b"new", link: b"replace"})
        self.assertFalse(first.parent.exists())
        self.assertFalse((self.repo / "absent").exists())

    def test_postprocess_namespace_matches_pipeline_defaults_and_partial_behavior(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                namespace = orchestrator.postprocess_namespace(
                    self.repo, partial=partial
                )
                self.assertEqual(namespace.mode, "pipeline")
                self.assertEqual(namespace.result_dir, self.repo)
                self.assertTrue(namespace.skip_test)
                self.assertEqual(namespace.partial, partial)
                self.assertEqual(namespace.skip_fit, partial)
                self.assertEqual(namespace.batch_size, 1)
                self.assertEqual(namespace.profile_backend, "kineto")
                self.assertEqual(namespace.estimator, "median")
                self.assertIsNone(namespace.result_root)
                self.assertIsNone(namespace.profile)
                self.assertIsNone(namespace.grid)
                self.assertIsNone(namespace.bazel)
                self.assertIsNone(namespace.oss_destination)


if __name__ == "__main__":
    unittest.main()
