import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    ProfileError,
    fingerprint,
    load_profile,
    profile_environment,
    strip_json_comments,
)
from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli


class CachePerfTest(unittest.TestCase):
    def test_launcher_resolves_repository_after_move(self):
        self.assertEqual(cli.REPO, Path(__file__).resolve().parents[5])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = self.root / "local.jsonc"
        self.profile.write_text(
            """{
          // local paths are literal, not shell expressions
          "schema_version":1,
          "engine":{"checkpoint_path":"/model","tokenizer_path":"/model"},
          "cache_grid":{"measure_runs":5},
          "engine_env":{"DSV4_CHUNK_TOKENS":"8192"},
          "runtime_env":{"CC":"/gcc"}
        }"""
        )
        self.grid = self.root / "grid.json"
        self.grid.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generator": {"cache_alignment": 4096},
                    "cases": [
                        {
                            "case_id": 7,
                            "batch_size": 1,
                            "input_len": 8192,
                            "cache_len": 4096,
                        },
                        {
                            "case_id": 9,
                            "batch_size": 1,
                            "input_len": 16384,
                            "cache_len": 0,
                        },
                    ],
                }
            )
        )
        self.result = self.root / "results"

    def args(self, mode="run", extra=(), explicit=True):
        args = [mode, "--result-dir", str(self.result), *extra]
        if explicit:
            args += ["--profile", str(self.profile), "--grid", str(self.grid)]
        return cli.parser().parse_args(args)

    def save_run(self):
        plan = cli.build_plan(
            self.args(), {"PATH": "/usr/bin", "DSV4_CHUNK_TOKENS": "1"}
        )
        self.result.mkdir()
        for name, data in plan["artifacts"].items():
            (self.result / name).write_bytes(data)
        (self.result / "cache_grid_results.json").write_text("{}")
        return plan

    def test_jsonc_preserves_urls_strings_and_fingerprint(self):
        text = '{"schema_version":1,"url":"https://a/b/*c*/","quote":"a\\"//b" /* x */}'
        parsed = json.loads(strip_json_comments(text))
        self.assertEqual(parsed["url"], "https://a/b/*c*/")
        self.assertEqual(
            fingerprint(parsed), fingerprint(json.loads(json.dumps(parsed)))
        )
        with self.assertRaises(ProfileError):
            strip_json_comments("{ /* missing close")

    def test_nsys_profile_wraps_bazel_and_forwards_capture_options(self):
        plan = cli.build_plan(
            self.args(
                "profile",
                extra=[
                    "--cases=7",
                    "--runs=3",
                    "--profile-backend=nsys",
                    "--nsys-path=/opt/nsys tools/nsys",
                    "--nsys-session=case19314",
                    "--nsys-tail-seconds=0.2",
                ],
            ),
            {},
        )
        command = plan["command"]
        wrapper = shlex.split(
            next(x.split("=", 1)[1] for x in command if x.startswith("--run_under="))
        )
        self.assertEqual(wrapper[:2], ["/opt/nsys tools/nsys", "launch"])
        self.assertIn("--session-new=case19314", wrapper)
        self.assertIn("--wait=all", wrapper)
        self.assertIn("--test_arg=--cache_profile_backend=nsys", command)
        self.assertIn("--test_arg=--cache_profile_runs=3", command)
        self.assertIn("--test_arg=--cache_nsys_tail_seconds=0.2", command)
        self.assertIn("--test_env=GEN_TIMELINE_SYNC=0", command)
        self.assertIn("--test_arg=--gen_timeline_sync=False", command)
        manifest = json.loads(plan["artifacts"][cli.MANIFEST])
        self.assertIn("--cache_nsys_session=case19314", manifest["runner_args"])
        self.assertFalse(plan["destination"].exists())

    def test_nsys_default_session_is_unique_and_kineto_has_no_wrapper(self):
        args = self.args("profile", extra=["--cases=7", "--profile-backend=nsys"])
        commands = [cli.build_plan(args, {})["command"] for _ in range(2)]
        sessions = [
            next(x for x in c if x.startswith("--test_arg=--cache_nsys_session="))
            for c in commands
        ]
        self.assertNotEqual(*sessions)
        normal = cli.build_plan(self.args("profile", extra=["--cases=7"]), {})
        self.assertFalse(any(x.startswith("--run_under=") for x in normal["command"]))

    def test_nsys_rejects_nonprofile_and_invalid_tail(self):
        for mode, extra in [
            ("run", ["--profile-backend=nsys"]),
            ("profile", ["--cases=7", "--nsys-tail-seconds=-1"]),
            ("profile", ["--cases=7", "--nsys-tail-seconds=nan"]),
        ]:
            with self.subTest(mode=mode, extra=extra), self.assertRaises(ValueError):
                cli.build_plan(self.args(mode, extra=extra), {})

    def test_standard_json_still_rejects_comments(self):
        path = self.root / "strict.json"
        path.write_text(self.profile.read_text())
        with self.assertRaises(ProfileError):
            load_profile(path)

    def test_invalid_or_sensitive_env_rejected(self):
        for values in (
            {"DSV4_X": True},
            {"BAD-NAME": "x"},
            {"API_KEY": "secret"},
            {"PATH": "$PATH:/x"},
        ):
            with self.assertRaises(ProfileError):
                profile_environment({"engine_env": values})
        with self.assertRaises(ProfileError):
            profile_environment({"runtime_env": {"CC": "a"}, "engine_env": {"CC": "b"}})

    def test_env_priority_and_bazel_command(self):
        plan = cli.build_plan(
            self.args(
                extra=["--env", "DSV4_CHUNK_TOKENS=4096", "--output-base", "/tmp/build"]
            ),
            {"PATH": "/usr/bin", "DSV4_CHUNK_TOKENS": "1"},
        )
        self.assertEqual(plan["process_env"]["DSV4_CHUNK_TOKENS"], "4096")
        self.assertEqual(
            plan["command"][:3], ["bazelisk", "--output_base=/tmp/build", "test"]
        )
        self.assertIn("--test_env=CC=/gcc", plan["command"])
        self.assertIn("--test_arg=--engine_env=CC=/gcc", plan["command"])
        self.assertFalse(self.result.exists())

    def test_inherited_credentials_reach_process_without_being_recorded(self):
        plan = cli.build_plan(
            self.args(),
            {"PATH": "/usr/bin", "CACHE_API_TOKEN": "secret-value"},
        )
        self.assertEqual(plan["process_env"]["CACHE_API_TOKEN"], "secret-value")
        self.assertNotIn("CACHE_API_TOKEN", plan["summary"]["environment"])
        self.assertNotIn("secret-value", " ".join(plan["command"]))
        self.assertIn("--test_env=CACHE_API_TOKEN", plan["command"])
        manifest = json.loads(plan["artifacts"][cli.MANIFEST])
        self.assertNotIn("CACHE_API_TOKEN", manifest["env"])

    def test_dry_run_does_not_write_or_launch(self):
        with patch.object(cli.subprocess, "run") as run, patch("builtins.print"):
            rc = cli.main(
                [
                    "run",
                    "--profile",
                    str(self.profile),
                    "--grid",
                    str(self.grid),
                    "--result-dir",
                    str(self.result),
                    "--dry-run",
                ]
            )
        self.assertEqual(rc, 0)
        run.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_run_forwards_and_resume_freezes_skip_reuse_validation(self):
        first = cli.build_plan(
            self.args(extra=["--skip-reuse-validation"]),
            {"PATH": "/usr/bin"},
        )
        self.assertIn("--test_arg=--cache_skip_reuse_validation", first["command"])
        self.assertTrue(first["summary"]["skip_reuse_validation"])
        self.result.mkdir()
        for name, data in first["artifacts"].items():
            (self.result / name).write_bytes(data)
        (self.result / "cache_grid_results.json").write_text("{}")
        resumed = cli.build_plan(self.args("resume", explicit=False), {})
        self.assertIn("--test_arg=--cache_skip_reuse_validation", resumed["command"])
        with self.assertRaisesRegex(ValueError, "overrides"):
            cli.build_plan(
                self.args(
                    "resume",
                    extra=["--skip-reuse-validation"],
                    explicit=False,
                ),
                {},
            )

    def test_resume_rejects_bazel_launch_overrides(self):
        self.save_run()
        for option in (
            "--config=changed",
            "--output-base=/tmp/changed-bazel-output",
            "--bazel=changed-bazel",
        ):
            with self.subTest(option=option), self.assertRaisesRegex(
                ValueError, "launch overrides"
            ):
                cli.build_plan(
                    self.args("resume", extra=[option], explicit=False),
                    {},
                )

    def test_launch_v2_stores_profile_and_environment_once(self):
        plan = self.save_run()
        raw = json.loads(plan["artifacts"][cli.MANIFEST])
        self.assertEqual(raw["schema_version"], 2)
        self.assertNotIn("profile", raw)
        self.assertEqual(raw["profile_file"], "profile.snapshot.json")
        self.assertFalse(any(a.startswith("--engine_env=") for a in raw["runner_args"]))
        saved = cli.load_saved(self.result)
        self.assertEqual(saved["profile"], load_profile(self.profile))
        for key, value in saved["env"].items():
            self.assertEqual(
                saved["runner_args"].count(f"--engine_env={key}={value}"), 1
            )

    def test_compact_test_info_preserves_attempts_and_resume_fingerprint(self):
        from rtp_llm.test.perf_test.batch_decode_test import (
            _build_cache_resume_config,
            _write_test_info,
            parse_args,
        )
        from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import (
            resume_config_fingerprint,
        )

        plan = self.save_run()
        saved = cli.load_saved(self.result)
        args, remaining = parse_args(saved["runner_args"])
        with patch.dict("os.environ", saved["env"], clear=True):
            config = _build_cache_resume_config(
                args, remaining, list(saved["env"]), 4096
            )
            for status in ("running", "completed", "running", "completed"):
                _write_test_info(
                    args,
                    remaining,
                    list(saved["env"]),
                    status=status,
                    expected_cache_block_size=4096,
                    resume_config=config,
                )
        text = (self.result / "test_info.json").read_text()
        info = json.loads(text)
        self.assertEqual(info["schema_version"], 4)
        self.assertEqual(info["attempt_count"], 2)
        self.assertEqual(info["status"], "completed")
        self.assertEqual(
            info["resume_config_sha256"], resume_config_fingerprint(config)
        )
        for field in (
            "engine_env_names",
            "engine_environment",
            "engine_args",
            "argv",
            "resume_config",
            "profile",
        ):
            self.assertNotIn(field, info)
        self.assertNotIn("DSV4_CHUNK_TOKENS", text)
        self.assertEqual(
            info["config_file_sha256"][cli.MANIFEST],
            cli.sha(plan["artifacts"][cli.MANIFEST]),
        )
        resumed = cli.build_plan(self.args("resume", explicit=False), {})
        self.assertEqual(resumed["summary"]["environment"], saved["env"])
        resumed_args = [
            a[len("--test_arg=") :]
            for a in resumed["command"]
            if a.startswith("--test_arg=")
        ]
        new_args, new_remaining = parse_args(resumed_args)
        with patch.dict("os.environ", saved["env"], clear=True):
            new_config = _build_cache_resume_config(
                new_args, new_remaining, list(saved["env"]), 4096
            )
        self.assertEqual(
            resume_config_fingerprint(new_config), resume_config_fingerprint(config)
        )
        raw = json.loads((self.result / cli.MANIFEST).read_text())
        raw["env"]["DSV4_CHUNK_TOKENS"] = "999"
        (self.result / cli.MANIFEST).write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "saved configuration changed"):
            cli.load_saved(self.result)
        (self.result / cli.MANIFEST).unlink()
        with self.assertRaisesRegex(ValueError, "current run requires"):
            cli.load_saved(self.result)

    def test_run_refuses_existing_results(self):
        self.save_run()
        with self.assertRaisesRegex(ValueError, "empty/new"):
            cli.build_plan(self.args(), {})

    def test_resume_restores_env_and_requires_checkpoint(self):
        before = self.save_run()
        after = cli.build_plan(
            self.args("resume", explicit=False), {"DSV4_CHUNK_TOKENS": "999"}
        )
        self.assertEqual(
            before["summary"]["environment"], after["summary"]["environment"]
        )
        self.assertIn("--test_arg=--require_cache_resume", after["command"])
        self.assertEqual(after["artifacts"], {})

    def test_profile_limits_survive_grid_sizes_and_workspace_modes(self):
        from rtp_llm.test.perf_test.batch_decode_test import (
            _configure_cache_batch_limits,
            _load_cache_grid_cases,
            extract_arg,
            parse_args,
        )

        profile = json.loads(
            self.profile.read_text().replace(
                "// local paths are literal, not shell expressions", ""
            )
        )
        profile["engine"].update(
            concurrency_limit=17, max_context_batch_size=9, dp_size=1
        )
        self.profile.write_text(json.dumps(profile))
        for workspace in (None, 1048576):
            for batch in (2, 4):
                with self.subTest(workspace=workspace, batch=batch):
                    grid = {
                        "schema_version": 2,
                        "generator": {"cache_alignment": 4096},
                        "cases": [
                            {
                                "case_id": 0,
                                "batch_size": batch,
                                "prefix_policy": "independent",
                                "request_groups": [
                                    {"count": batch, "input_len": 8192, "cache_len": 0}
                                ],
                            }
                        ],
                    }
                    if workspace:
                        grid["generator"]["workspace_tokens"] = workspace
                    self.grid.write_text(json.dumps(grid))
                    plan = cli.build_plan(self.args(), {})
                    argv = [
                        a[len("--test_arg=") :]
                        for a in plan["command"]
                        if a.startswith("--test_arg=")
                    ]
                    self.assertFalse(
                        any(
                            a.startswith(
                                ("--concurrency_limit", "--max_context_batch_size")
                            )
                            for a in argv
                        )
                    )
                    argv = [a for a in argv if not a.startswith("--profile=")]
                    args, remaining = parse_args(
                        [*argv, "--profile", str(self.profile)]
                    )
                    _configure_cache_batch_limits(
                        args,
                        remaining,
                        _load_cache_grid_cases(str(self.grid)),
                        cache_alignment=4096,
                    )
                    self.assertEqual(args.concurrency_limit, 17)
                    self.assertEqual(
                        extract_arg(remaining, "max_context_batch_size"), "9"
                    )

    def test_fixed_workspace_capacities_are_frozen_and_replayed(self):
        payload = json.loads(self.grid.read_text())
        payload["generator"] = {
            "workspace_tokens": 1048576,
            "cache_alignment": 4096,
            "parameters": {"max_batch_tokens": 65536},
        }
        for case in payload["cases"]:
            case["batch_size"] = 4
            case["prefix_policy"] = "independent"
            case["request_groups"] = [
                {
                    "count": 4,
                    "input_len": case["input_len"],
                    "cache_len": case["cache_len"],
                }
            ]
        self.grid.write_text(json.dumps(payload))
        first = self.save_run()
        plans = [first]
        for mode in ("resume", "retest"):
            extra = [] if mode == "resume" else ["--cases", "7"]
            plans.append(
                cli.build_plan(self.args(mode, extra=extra, explicit=False), {})
            )
        for plan in plans:
            for arg in (
                "--max_seq_len=1048576",
                "--max_batch_tokens_size=65536",
                "--cache_workspace_tokens=1048576",
            ):
                self.assertIn("--test_arg=" + arg, plan["command"])

    def test_fixed_workspace_accepts_packed_long_short_batch_before_launch(self):
        self.grid.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generator": {
                        "workspace_tokens": 1048576,
                        "cache_alignment": 4096,
                    },
                    "cases": [
                        {
                            "case_id": 1,
                            "batch_size": 8,
                            "prefix_policy": "independent",
                            "request_groups": [
                                {"count": 1, "input_len": 200000, "cache_len": 0},
                                {"count": 7, "input_len": 8192, "cache_len": 0},
                            ],
                        }
                    ],
                }
            )
        )
        plan = cli.build_plan(self.args(), {})
        self.assertIn("--test_arg=--cache_workspace_tokens=1048576", plan["command"])
        self.assertFalse(self.result.exists())

    def test_fixed_workspace_batch_one_profile_keeps_capacity(self):
        payload = json.loads(self.grid.read_text())
        payload["generator"] = {
            "workspace_tokens": 1048576,
            "cache_alignment": 4096,
        }
        self.grid.write_text(json.dumps(payload))
        self.save_run()
        plan = cli.build_plan(
            self.args("profile", extra=["--cases", "7"], explicit=False), {}
        )
        self.assertIn("--test_arg=--cache_workspace_tokens=1048576", plan["command"])
        self.assertFalse(
            any(
                "--max_context_batch_size" in a or "--concurrency_limit" in a
                for a in plan["command"]
            )
        )
        self.assertIn("--test_arg=--max_seq_len=1048576", plan["command"])

    def test_resume_missing_checkpoint_or_overrides_rejected(self):
        with self.assertRaisesRegex(ValueError, "existing"):
            cli.build_plan(self.args("resume", explicit=False), {})
        self.save_run()
        with self.assertRaisesRegex(ValueError, "overrides"):
            cli.build_plan(
                self.args("resume", extra=["--runs", "3"], explicit=False), {}
            )

    def test_changed_snapshot_rejected(self):
        self.save_run()
        (self.result / "grid.snapshot.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            cli.build_plan(self.args("resume", explicit=False), {})

    def test_retest_selects_only_requested_cases(self):
        self.save_run()
        plan = cli.build_plan(
            self.args("retest", extra=["--cases", "9", "--runs", "3"], explicit=False),
            {},
        )
        grid = json.loads(plan["artifacts"]["grid.snapshot.json"])
        self.assertEqual([c["case_id"] for c in grid["cases"]], [9])
        self.assertIn("--test_arg=--cache_profile_runs=0", plan["command"])
        self.assertIn("--test_arg=--cache_measure_runs=3", plan["command"])
        self.assertNotIn("--test_arg=--require_cache_resume", plan["command"])
        self.assertTrue(
            plan["destination"].is_relative_to(self.result / "cache_perf_replays")
        )

    def test_profile_never_resumes_formal_grid(self):
        plan = cli.build_plan(
            self.args("profile", extra=["--cases", "7", "--runs", "2"]), {}
        )
        self.assertIn("--test_arg=--cache_profile_only", plan["command"])
        self.assertIn("--test_arg=--cache_profile_flat_output", plan["command"])
        self.assertIn("--test_arg=--cache_profile_runs=2", plan["command"])
        self.assertEqual(plan["summary"]["planned_cases"], 1)
        self.assertFalse(plan["summary"]["reads_checkpoint"])

    def test_unknown_case_and_invalid_runs(self):
        for options in (["--cases", "888"], ["--cases", "7", "--runs", "0"]):
            with self.assertRaises(ValueError):
                cli.build_plan(self.args("retest", extra=options), {})

    def test_profile_plan_uses_same_isolated_directory_in_backend(self):
        from rtp_llm.test.perf_test.batch_decode_test import (
            _prepare_cache_profile_result_dir,
            parse_args,
        )

        plan = cli.build_plan(self.args("profile", extra=["--cases", "7"]), {})
        directory = plan["destination"]
        directory.mkdir(parents=True)
        for name, data in plan["artifacts"].items():
            (directory / name).write_bytes(data)
        argv = json.loads(plan["artifacts"][cli.MANIFEST])["runner_args"]
        args, _ = parse_args(argv)
        _prepare_cache_profile_result_dir(args)
        self.assertEqual(Path(args.result_dir), directory)
        self.assertTrue(args.cache_profile_flat_output)
        self.assertTrue(args.cache_profile_only)
        self.assertFalse((directory / "cache_profile_replays").exists())
        for name, data in plan["artifacts"].items():
            self.assertEqual((directory / name).read_bytes(), data)

    def test_main_propagates_bazel_exit_code(self):
        with patch.object(cli.subprocess, "run") as run, patch("builtins.print"):
            run.return_value.returncode = 9
            rc = cli.main(
                [
                    "run",
                    "--profile",
                    str(self.profile),
                    "--grid",
                    str(self.grid),
                    "--result-dir",
                    str(self.result),
                ]
            )
        self.assertEqual(rc, 9)
        self.assertTrue((self.result / cli.MANIFEST).exists())
        self.assertEqual(run.call_args.kwargs["cwd"], cli.REPO)

    def test_explicit_default_value_overrides_profile(self):
        from rtp_llm.test.perf_test.batch_decode_test import parse_args

        args, _ = parse_args(
            ["--profile", str(self.profile), "--cache_measure_runs", "3"]
        )
        self.assertEqual(args.cache_measure_runs, 3)


if __name__ == "__main__":
    unittest.main()
