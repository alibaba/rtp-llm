import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli
from rtp_llm.test.perf_test.cache_grid.tests.result_schema_test import grouped_metric


class CachePerfPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.result = self.root / "results"
        self.profile = self.root / "profile.json"
        self.profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_label": "Original Model",
                    "engine": {"model_type": "qwen_2", "tp_size": 1},
                    "engine_env": {"PERF_PROFILE_RUNS": "0"},
                    "bazel": {"configs": ["cuda12"]},
                }
            )
        )
        self.grid = self.root / "grid.json"
        self.grid.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generator": {"cache_alignment": 4096},
                    "cases": [
                        {
                            "case_id": 0,
                            "batch_size": 1,
                            "input_len": 8192,
                            "cache_len": 0,
                        },
                    ],
                }
            )
        )
        output = patch("sys.stdout", new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)

    def argv(self, *extra, launch=True):
        argv = ["pipeline", "--result-dir", str(self.result), *extra]
        if launch:
            argv += ["--profile", str(self.profile), "--grid", str(self.grid)]
        return argv

    def result_file(self, complete=True):
        self.result.mkdir(exist_ok=True)
        (self.result / "cache_grid_results.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "mode": "prefix_cache_grid",
                    "metrics": [grouped_metric()],
                    "complete": complete,
                    "completed_cases": 1,
                    "total_cases": 1,
                }
            )
        )

    def manifest(self):
        return json.loads((self.result / "pipeline_summary.json").read_text())

    def test_pipeline_uses_bazel_snapshots_and_frozen_profile(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[1] == "test":
                self.assertIn(cli.TARGET, command)
                self.assertIn("--config=cuda12", command)
                self.assertIn("--test_arg=--cache_measure_runs=4", command)
                self.assertEqual(kwargs["env"]["PERF_PROFILE_RUNS"], "0")
                self.assertTrue((self.result / cli.MANIFEST).exists())
                self.assertTrue((self.result / "grid.snapshot.json").exists())
                self.profile.write_text("{}")
                self.result_file()
            else:
                self.assertIn(
                    "--profile=" + str(self.result / "profile.snapshot.json"), command
                )
                frozen = json.loads((self.result / "profile.snapshot.json").read_text())
                self.assertEqual(frozen["model_label"], "Original Model")
            return subprocess.CompletedProcess(command, 0)

        with patch.object(cli.subprocess, "run", side_effect=run):
            self.assertEqual(cli.main(self.argv("--runs", "4", "--skip-fit")), 0)
        self.assertEqual(len(calls), 1)
        self.assertIn(
            "Original Model",
            (self.result / "report/latency.interactive.html").read_text(),
        )
        self.assertEqual(self.manifest()["status"], "completed")

    def test_dry_run_writes_nothing_and_launches_nothing(self):
        with patch.object(cli.subprocess, "run") as run:
            self.assertEqual(cli.main(self.argv("--dry-run")), 0)
            run.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_skip_test_keeps_quality_gate_and_custom_outputs(self):
        self.result_file()
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(cli.subprocess, "run", side_effect=run):
            code = cli.main(
                self.argv(
                    "--skip-test",
                    "--estimator",
                    "min",
                    "--svg-output",
                    str(self.root / "custom.svg"),
                    launch=False,
                )
            )
        self.assertEqual(code, 3)
        self.assertEqual(len(calls), 0)
        self.assertTrue((self.root / "custom.svg").exists())
        self.assertEqual(self.manifest()["status"], "fit_rejected")
        self.assertTrue(self.manifest()["stages"]["test"]["skipped"])

    def test_test_failure_stops_postprocessing_and_preserves_code(self):
        with patch.object(
            cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)
        ) as run:
            self.assertEqual(cli.main(self.argv()), 7)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(self.manifest()["stages"]["test"]["returncode"], 7)

    def test_incomplete_results_are_rejected_without_postprocessing(self):
        self.result_file(complete=False)
        args = cli.parser().parse_args(self.argv("--skip-test", launch=False))
        with patch.object(cli.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                cli.run_pipeline(args)
            run.assert_not_called()
        self.assertEqual(self.manifest()["status"], "failed")
        args.partial = True
        self.assertEqual(cli.run_pipeline(args), 0)
        self.assertEqual(self.manifest()["status"], "partial")

    def test_postprocessing_failure_is_not_a_quality_gate(self):
        self.result_file()
        with patch(
            "rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit.run_fit",
            return_value=2,
        ):
            self.assertEqual(cli.main(self.argv("--skip-test", launch=False)), 2)
        self.assertEqual(self.manifest()["status"], "failed")

    def test_chart_failure_is_recorded(self):
        self.result_file()
        with patch(
            "rtp_llm.test.perf_test.cache_grid.plot.unified_report.write_charts",
            side_effect=RuntimeError("chart failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "chart failed"):
                cli.run_pipeline(
                    cli.parser().parse_args(
                        self.argv("--skip-test", "--skip-fit", launch=False)
                    )
                )
        self.assertEqual(self.manifest()["status"], "failed")

    def test_pipeline_falls_back_to_svg_when_plotly_is_unavailable(self):
        self.result_file()
        with patch.dict("sys.modules", {"plotly": None, "plotly.offline": None}):
            self.assertEqual(
                cli.main(self.argv("--skip-test", "--skip-fit", launch=False)),
                0,
            )

        summary = self.manifest()
        self.assertEqual(summary["status"], "completed")
        fallback = Path(summary["artifacts"]["fallback_svg"])
        self.assertTrue(fallback.exists())
        self.assertIn(
            "Interactive charts are unavailable",
            Path(summary["artifacts"]["html"]).read_text(),
        )

    def test_pipeline_archives_and_uploads_completed_results(self):
        self.result_file()
        destination = "oss://bucket/reports/results.tar.gz"
        with patch(
            "rtp_llm.test.perf_test.cache_grid.runner.archive_upload.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as upload:
            self.assertEqual(
                cli.main(
                    self.argv(
                        "--skip-test",
                        "--skip-fit",
                        "--oss-destination",
                        destination,
                        launch=False,
                    )
                ),
                0,
            )
        summary = self.manifest()
        archive = Path(summary["artifacts"]["archive"]["path"])
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["stages"]["upload"]["status"], "completed")
        self.assertEqual(archive.parent, self.result.parent)
        self.assertEqual(
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            summary["artifacts"]["archive"]["sha256"],
        )
        upload.assert_called_once_with(
            ["ossutil", "cp", str(archive), destination], check=False
        )
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
            self.assertIn("results/cache_grid_results.json", names)
            self.assertIn("results/report/latency.interactive.html", names)
            self.assertIn("results/pipeline_summary.json", names)
            self.assertNotIn(archive.name, names)

    def test_upload_failure_retains_archive_and_records_failure(self):
        self.result_file()
        args = cli.parser().parse_args(
            self.argv(
                "--skip-test",
                "--skip-fit",
                "--oss-destination",
                "oss://bucket/reports/results.tar.gz",
                launch=False,
            )
        )
        with patch(
            "rtp_llm.test.perf_test.cache_grid.runner.archive_upload.subprocess.run",
            return_value=subprocess.CompletedProcess([], 7),
        ):
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                cli.run_pipeline(args)
        summary = self.manifest()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["stages"]["upload"]["status"], "failed")
        self.assertTrue(Path(summary["artifacts"]["archive"]["path"]).exists())

    def test_fit_rejection_still_uploads_report(self):
        self.result_file()
        with patch(
            "rtp_llm.test.perf_test.cache_grid.runner.archive_upload.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as upload:
            self.assertEqual(
                cli.main(
                    self.argv(
                        "--skip-test",
                        "--oss-destination",
                        "oss://bucket/reports/rejected.tar.gz",
                        launch=False,
                    )
                ),
                3,
            )
        upload.assert_called_once()
        self.assertEqual(self.manifest()["status"], "fit_rejected")
        self.assertEqual(self.manifest()["stages"]["upload"]["status"], "completed")

    def test_resume_uses_existing_launch_configuration(self):
        args = cli.parser().parse_args(self.argv())
        args.mode = "run"
        plan = cli.build_plan(args, {})
        self.result.mkdir()
        for name, data in plan["artifacts"].items():
            (self.result / name).write_bytes(data)
        self.result_file(complete=False)
        frozen = (self.result / "profile.snapshot.json").read_bytes()
        self.profile.unlink()
        self.grid.unlink()

        def run(command, **kwargs):
            if command[1] == "test":
                self.assertIn("--test_arg=--require_cache_resume", command)
                self.result_file()
            return subprocess.CompletedProcess(command, 0)

        with patch.object(cli.subprocess, "run", side_effect=run):
            self.assertEqual(
                cli.main(
                    self.argv("--test-mode", "resume", "--skip-fit", launch=False)
                ),
                0,
            )
        self.assertEqual((self.result / "profile.snapshot.json").read_bytes(), frozen)

    def test_skip_test_rejects_launch_overrides(self):
        args = cli.parser().parse_args(self.argv("--skip-test"))
        with self.assertRaisesRegex(ValueError, "launch overrides"):
            cli.run_pipeline(args)

    def test_multi_grid_merge_and_completed_resume(self):
        grids = self.root / "grids"
        grids.mkdir()
        for name in ("b1", "b2"):
            (grids / f"{name}.json").write_bytes(self.grid.read_bytes())
        argv = ["pipeline", "--result-root", str(self.result), "--skip-fit"]
        original = cli.execute_plan

        def execute(plan):
            with patch.object(
                cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ):
                code = original(plan)
            destination = plan["destination"]
            (destination / "cache_grid_results.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mode": "prefix_cache_grid",
                        "complete": True,
                        "metrics": [grouped_metric()],
                    }
                )
            )
            return code

        with patch.object(cli, "execute_plan", side_effect=execute) as launch:
            self.assertEqual(
                cli.main(
                    argv + ["--profile", str(self.profile), "--grid-dir", str(grids)]
                ),
                0,
            )
            self.assertEqual(launch.call_count, 2)
        rows = json.loads((self.result / "report/observations.json").read_text())
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["source_run"], rows[1]["source_run"])
        self.assertEqual(rows[0]["geometry_key"], rows[1]["geometry_key"])
        with patch.object(cli, "execute_plan") as launch:
            self.assertEqual(cli.main(argv + ["--test-mode", "resume"]), 0)
            launch.assert_not_called()

    def test_heterogeneous_requests_features_and_split(self):
        from rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit import (
            load_observations,
            split_rows,
        )
        from rtp_llm.test.perf_test.cache_grid.formula.restricted_symbolic_fit import (
            build_candidate_library,
        )
        from rtp_llm.test.perf_test.cache_grid.runner.observations import (
            normalize_metric,
        )

        item = grouped_metric(input_len=8192, cache_len=4096)
        item["batch_size"] = 2
        item["request_groups"].append(dict(count=1, input_len=16384, cache_len=4096))
        for run in item["runs"]:
            run["requests"].append(
                dict(run["requests"][0], input_len=16384, prefill_time_ms=2000)
            )
        normalized = normalize_metric(item, "source", 0)
        self.assertEqual(normalized["target_ms"], 2000)
        self.assertEqual(normalized["compute_len"], 16384)
        self.assertEqual(
            normalized["request_distribution"],
            [[8192, 4096, 4096], [16384, 4096, 12288]],
        )
        reverse = json.loads(json.dumps(item))
        reverse["request_groups"].reverse()
        for run in reverse["runs"]:
            run["requests"].reverse()
        self.assertEqual(
            normalize_metric(reverse, "other", 0)["geometry_key"],
            normalized["geometry_key"],
        )
        self.result_file()
        path = self.result / "cache_grid_results.json"
        data = json.loads(path.read_text())
        data["metrics"] = [item, reverse]
        path.write_text(json.dumps(data))
        rows, audit = load_observations([path])
        self.assertEqual(audit["unique_geometry_count"], 1)
        self.assertEqual(sorted(map(len, split_rows(rows).values())), [0, 0, 2])
        terms = {t.name: t for t in build_candidate_library(1024)}
        self.assertEqual(terms["u**2"].evaluate(rows[0]), 4**2 + 12**2)
        self.assertEqual(terms["batchSize"].evaluate(rows[0]), 2)
        self.assertEqual(terms["maxComputeTokens"].evaluate(rows[0]), 12)

    def test_multi_grid_dry_run_has_no_side_effects(self):
        grids = self.root / "grids"
        grids.mkdir()
        (grids / "b1.json").write_bytes(self.grid.read_bytes())
        argv = [
            "pipeline",
            "--result-root",
            str(self.result),
            "--grid-dir",
            str(grids),
            "--profile",
            str(self.profile),
            "--dry-run",
        ]
        with patch.object(cli, "execute_plan") as launch:
            self.assertEqual(cli.main(argv), 0)
            launch.assert_not_called()
        self.assertFalse(self.result.exists())

    def test_multi_grid_pending_resume_freezes_inputs(self):
        grids = self.root / "grids"
        grids.mkdir()
        for name in ("a", "b"):
            (grids / f"{name}.json").write_bytes(self.grid.read_bytes())
        argv = ["pipeline", "--result-root", str(self.result), "--skip-fit"]
        original = cli.execute_plan

        def interrupted(plan):
            with patch.object(
                cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)
            ):
                code = original(plan)
            dest = plan["destination"]
            (dest / "cache_grid_results.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mode": "prefix_cache_grid",
                        "complete": False,
                        "metrics": [grouped_metric()],
                    }
                )
            )
            return code

        with patch.object(cli, "execute_plan", side_effect=interrupted):
            self.assertEqual(
                cli.main(
                    argv
                    + [
                        "--profile",
                        str(self.profile),
                        "--grid-dir",
                        str(grids),
                        "--runs",
                        "4",
                    ]
                ),
                7,
            )
        self.profile.unlink()
        for path in grids.glob("*.json"):
            path.write_text("invalid modified input")
        commands = []

        def resumed(plan):
            commands.append(plan["command"])
            with patch.object(
                cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ):
                code = original(plan)
            (plan["destination"] / "cache_grid_results.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mode": "prefix_cache_grid",
                        "complete": True,
                        "metrics": [grouped_metric()],
                    }
                )
            )
            return code

        with patch.object(cli, "execute_plan", side_effect=resumed):
            self.assertEqual(cli.main(argv + ["--test-mode", "resume"]), 0)
        self.assertEqual(len(commands), 2)
        self.assertIn("--test_arg=--require_cache_resume", commands[0])
        self.assertNotIn("--test_arg=--require_cache_resume", commands[1])
        self.assertIn("--test_arg=--cache_measure_runs=4", commands[1])

    def test_different_models_cannot_be_merged(self):
        from rtp_llm.test.perf_test.cache_grid.runner.unified_pipeline import (
            validate_merge_inputs,
        )

        paths = []
        for name, model in (("a", "qwen_2"), ("b", "other")):
            dest = self.root / name
            dest.mkdir()
            path = dest / "cache_grid_results.json"
            path.write_text(json.dumps({"profile": {"engine": {"model_type": model}}}))
            paths.append(path)
        with self.assertRaisesRegex(ValueError, "profiles differ"):
            validate_merge_inputs(paths)

    def test_embedded_null_sections_do_not_crash_merge_validation(self):
        from rtp_llm.test.perf_test.cache_grid.runner.unified_pipeline import (
            validate_merge_inputs,
        )

        dest = self.root / "nullable"
        dest.mkdir()
        path = dest / "cache_grid_results.json"
        path.write_text(
            json.dumps(
                {
                    "profile": {
                        "engine": None,
                        "engine_args": None,
                        "engine_env": None,
                        "cache_grid": None,
                    }
                }
            )
        )
        records, profile = validate_merge_inputs([path])
        self.assertEqual(len(records), 1)
        self.assertIsNone(profile["engine"])

    def test_pipeline_does_not_accept_profiler_mode(self):
        args = cli.parser().parse_args(self.argv("--profile-backend", "nsys"))
        with self.assertRaisesRegex(ValueError, "formal measurements"):
            cli.run_pipeline(args)


if __name__ == "__main__":
    unittest.main()
