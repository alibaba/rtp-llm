#!/usr/bin/env python3

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit import (
    DEFAULT_TOKEN_UNIT,
    Observation,
    build_parser,
    load_observations,
    run_analyze_anomalies,
    run_fit,
    split_rows,
)


def _make_observations(count: int = 20) -> list[Observation]:
    rows = []
    for i in range(count):
        input_len = 1024 * (i + 1)
        cache_len = 512 * i
        compute_len = input_len - cache_len
        target_ms = 10.0 + 0.5 * (compute_len / 1024.0) - 0.1 * (cache_len / 1024.0)
        rows.append(
            Observation(
                batch_size=1,
                input_len=input_len,
                cache_len=cache_len,
                target_ms=max(target_ms, 1.0),
                source=f"synth:{i}",
                requested_cache_len=cache_len,
            )
        )
    return rows


def _write_cache_grid_result(
    path: Path, rows: list[Observation], measure_runs: int = 3
):
    metrics = []
    for idx, row in enumerate(rows):
        runs = []
        for r in range(measure_runs):
            runs.append(
                {
                    "prefill_time_ms": row.target_ms,
                    "success": True,
                    "input_len": row.input_len,
                    "output_len": 1,
                    "reuse_len": row.cache_len,
                    "ttft_ms": row.target_ms,
                }
            )
        metrics.append(
            {
                "case_key": f"bs1_seq{row.input_len}_cache{row.cache_len}",
                "case_id": idx,
                "batch_size": 1,
                "input_len": row.input_len,
                "cache_len_requested": row.cache_len,
                "cache_len_observed": [row.cache_len] * measure_runs,
                "expected_reuse_len": row.cache_len,
                "success_runs": measure_runs,
                "measure_runs": measure_runs,
                "status": "ok",
                "reuse_exact": True,
                "runs": runs,
            }
        )
    payload = {
        "schema_version": 2,
        "mode": "prefix_cache_grid",
        "complete": True,
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class MeasurementContractTest(unittest.TestCase):
    def test_server_contract_allows_both_client_transports(self):
        rows = _make_observations(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index, source in enumerate(
                (
                    "client_http_wall_max_new_tokens_1",
                    "client_dashsc_grpc_input_ids_wall_max_new_tokens_1",
                )
            ):
                path = Path(tmpdir) / f"result-{index}.json"
                _write_cache_grid_result(path, rows)
                payload = json.loads(path.read_text(encoding="utf-8"))
                for metric in payload["metrics"]:
                    for run in metric["runs"]:
                        run["ttft_source"] = source
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            observations, audit = load_observations(paths)
            self.assertTrue(observations)
            self.assertEqual(
                audit["measurement_contracts"],
                ["batch_max_server_first_token_cost_time_ms"],
            )


class RunFitTest(unittest.TestCase):
    def test_fit_without_profile(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                min_valid_rows=6,
                max_mape_pct=100.0,
                max_p95_ape_pct=100.0,
                max_max_ape_pct=100.0,
                estimator="median",
                allow_insufficient_data=False,
                profile=None,
                model_label=None,
                token_unit=None,
                formula_filename=None,
                formula_key=None,
            )
            exit_code = run_fit(args)
            self.assertIn(exit_code, (0, 3))
            report = json.loads((output_dir / "fit_report.json").read_text())
            self.assertIsNone(report["profile"])
            self.assertIsNone(report["profile_sha256"])
            self.assertEqual(report["token_unit"], DEFAULT_TOKEN_UNIT)
            self.assertEqual(report["model_family"], "restricted-symbolic")
            self.assertEqual(report["objective"], "mean_squared_relative_error")
            self.assertIsNotNone(report["symbolic_search"])
            self.assertTrue((output_dir / "Model_prefill_formula.txt").exists())
            formula_content = (output_dir / "Model_prefill_formula.txt").read_text()
            self.assertTrue(formula_content.startswith("PREFILL_TIME_FORMULA="))

    def test_default_filename_uses_top_level_model_label(self):
        for label, filename in (
            ("DeepSeek-V4-Pro", "DeepSeek-V4-Pro_prefill_formula.txt"),
            ("../Qwen/Model", "Qwen_Model_prefill_formula.txt"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                source = root / "results.json"
                _write_cache_grid_result(source, _make_observations(40))
                profile = root / "profile.json"
                profile.write_text(
                    json.dumps({"schema_version": 1, "model_label": label})
                )
                output = root / "formula"
                args = build_parser().parse_args(
                    [
                        "fit",
                        "--inputs",
                        str(source),
                        "--output-dir",
                        str(output),
                        "--profile",
                        str(profile),
                        "--min-valid-rows",
                        "6",
                    ]
                )
                self.assertIn(run_fit(args), (0, 3))
                self.assertTrue((output / filename).is_file())
                self.assertEqual(len(list(output.glob("*.txt"))), 1)

    def test_fit_with_profile(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            profile = {
                "schema_version": 1,
                "label": "Test Model",
                "chart": {
                    "model_label": "Test Model",
                    "token_unit": 2048,
                    "formula_filename": "test_formula.txt",
                    "formula_key": "TEST_KEY",
                },
            }
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                min_valid_rows=6,
                max_mape_pct=100.0,
                max_p95_ape_pct=100.0,
                max_max_ape_pct=100.0,
                estimator="median",
                allow_insufficient_data=False,
                profile=str(profile_path),
                model_label=None,
                token_unit=None,
                formula_filename=None,
                formula_key=None,
            )
            exit_code = run_fit(args)
            self.assertIn(exit_code, (0, 3))
            report = json.loads((output_dir / "fit_report.json").read_text())
            self.assertEqual(report["model"], "Test Model")
            gap = (output_dir / "fit_gap.svg").read_text()
            self.assertIn("Test Model", gap)
            self.assertNotIn("DeepSeek", gap)
            self.assertIsNotNone(report["profile"])
            self.assertIsNotNone(report["profile_sha256"])
            self.assertEqual(report["token_unit"], 2048)
            self.assertTrue((output_dir / "test_formula.txt").exists())
            formula_content = (output_dir / "test_formula.txt").read_text()
            self.assertTrue(formula_content.startswith("TEST_KEY="))
            self.assertIn("2048", formula_content)

    def test_cli_overrides_profile(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            profile = {
                "schema_version": 1,
                "label": "Profile Label",
                "chart": {
                    "model_label": "Profile Label",
                    "token_unit": 2048,
                },
            }
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                min_valid_rows=6,
                max_mape_pct=100.0,
                max_p95_ape_pct=100.0,
                max_max_ape_pct=100.0,
                estimator="median",
                allow_insufficient_data=False,
                profile=str(profile_path),
                model_label="CLI Label",
                token_unit=512,
                formula_filename="cli_formula.txt",
                formula_key="CLI_KEY",
            )
            exit_code = run_fit(args)
            self.assertIn(exit_code, (0, 3))
            report = json.loads((output_dir / "fit_report.json").read_text())
            self.assertEqual(report["model"], "CLI Label")
            self.assertEqual(report["token_unit"], 512)
            self.assertTrue((output_dir / "cli_formula.txt").exists())
            formula_content = (output_dir / "cli_formula.txt").read_text()
            self.assertTrue(formula_content.startswith("CLI_KEY="))
            self.assertIn("512", formula_content)


class AnalyzeAnomaliesTest(unittest.TestCase):
    def test_empty_validation_skips_residual_fit_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "results.json"
            _write_cache_grid_result(source, _make_observations(1))
            output = Path(tmp) / "out"
            args = build_parser().parse_args(
                [
                    "analyze-anomalies",
                    "--inputs",
                    str(source),
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(run_analyze_anomalies(args), 0)
            report = json.loads((output / "anomaly_report.json").read_text())
            self.assertEqual(report["residual_check"]["status"], "skipped")
            self.assertEqual(
                report["residual_check"]["model_family"], "restricted-symbolic"
            )

    def test_basic_analysis(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                estimator="median",
                min_rt_ms=5.0,
                min_compute_tokens=16384,
                max_anomaly_ape_pct=25.0,
                profile=None,
                model_label=None,
                token_unit=None,
            )
            exit_code = run_analyze_anomalies(args)
            self.assertEqual(exit_code, 0)
            report = json.loads((output_dir / "anomaly_report.json").read_text())
            self.assertIn("summary", report)
            self.assertIn("anomalies", report)
            self.assertIn("total_anomalies", report["summary"])
            self.assertIsNone(report["profile"])

    def test_with_profile(self):
        rows = _make_observations(40)
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            profile = {
                "schema_version": 1,
                "label": "Anomaly Test Model",
                "chart": {
                    "model_label": "Anomaly Test Model",
                },
            }
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                estimator="median",
                min_rt_ms=5.0,
                min_compute_tokens=16384,
                max_anomaly_ape_pct=25.0,
                profile=str(profile_path),
                model_label=None,
                token_unit=None,
            )
            exit_code = run_analyze_anomalies(args)
            self.assertEqual(exit_code, 0)
            report = json.loads((output_dir / "anomaly_report.json").read_text())
            self.assertEqual(report["model"], "Anomaly Test Model")
            self.assertIsNotNone(report["profile"])
            self.assertIsNotNone(report["profile_sha256"])

    def test_detects_cache_monotonicity_violation(self):
        rows = [
            Observation(1, 4096, 0, 20.0, "a", 0),
            Observation(1, 4096, 2048, 25.0, "b", 2048),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "cache_grid_results.json"
            _write_cache_grid_result(result_path, rows)
            output_dir = Path(tmpdir) / "output"
            args = argparse.Namespace(
                inputs=[str(result_path)],
                output_dir=str(output_dir),
                batch_size=1,
                estimator="median",
                min_rt_ms=1.0,
                min_compute_tokens=1024,
                max_anomaly_ape_pct=25.0,
                profile=None,
                model_label=None,
                token_unit=None,
            )
            run_analyze_anomalies(args)
            report = json.loads((output_dir / "anomaly_report.json").read_text())
            cache_anomalies = [
                a for a in report["anomalies"] if a["check"] == "cache_monotonicity"
            ]
            self.assertTrue(len(cache_anomalies) > 0)


class SplitTest(unittest.TestCase):
    def test_splits_preserve_geometry_and_hold_out_test_rows(self):
        rows = _make_observations(100)
        splits = split_rows(rows)
        self.assertEqual(splits, split_rows(list(reversed(rows))))
        self.assertTrue(all(splits.values()))
        self.assertEqual(
            set(rows), set().union(*(set(group) for group in splits.values()))
        )
        input_sets = [set(row.input_len for row in group) for group in splits.values()]
        self.assertTrue(
            all(
                not left & right
                for i, left in enumerate(input_sets)
                for right in input_sets[i + 1 :]
            )
        )


class ParserTest(unittest.TestCase):
    def test_removed_algorithm_options_are_rejected(self):
        import contextlib
        import io

        for option, value in (
            ("--model-family", "quadratic"),
            ("--objective", "mae"),
            ("--split-mode", "random-50-50"),
            ("--split-seed", "17"),
        ):
            with self.subTest(option=option), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        "fit",
                        "--inputs",
                        "a.json",
                        "--output-dir",
                        "/tmp/out",
                        option,
                        value,
                    ]
                )

    def test_fit_has_profile_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            ["fit", "--inputs", "a.json", "--output-dir", "/tmp/out"]
        )
        self.assertIsNone(args.profile)
        self.assertIsNone(args.token_unit)
        self.assertIsNone(args.formula_filename)
        self.assertIsNone(args.formula_key)

    def test_analyze_anomalies_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "analyze-anomalies",
                "--inputs",
                "a.json",
                "--output-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.command, "analyze-anomalies")
        self.assertEqual(args.min_rt_ms, 5.0)
        self.assertEqual(args.min_compute_tokens, 16384)
        self.assertEqual(args.max_anomaly_ape_pct, 25.0)

    def test_validate_has_profile_flags(self):
        parser = build_parser()
        args = parser.parse_args(["validate-inputs", "--inputs", "a.json"])
        self.assertIsNone(args.profile)
        self.assertIsNone(args.model_label)


if __name__ == "__main__":
    unittest.main()
