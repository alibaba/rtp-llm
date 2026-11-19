"""Performance framework behavior and failure regressions."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.results import (
    DecodeMetrics,
    PrefillMetrics,
    ResultSchemaError,
    parse_result_envelope,
)
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.results import (
    decode_metrics_case,
    decode_metrics_measurements,
    decode_metrics_selected,
    decode_metrics_source,
    decode_metrics_TOPOLOGY,
    prefill_adapter_finalize_plan,
    result_envelope_decode_result,
    result_envelope_prefill_result,
)


class ResultEnvelopeTest(unittest.TestCase):

    def test_ssh_result_uses_the_common_envelope(self):
        for value in (
            result_envelope_prefill_result(),
            result_envelope_decode_result(),
        ):
            value["backend"] = "ssh"
            self.assertEqual("ssh", parse_result_envelope(value).to_dict()["backend"])

    def test_prefill_result_preserves_server_ttft_and_wait(self):
        value = result_envelope_prefill_result()
        metrics = value["cases"][0]["metrics"]
        metrics["ttft_ms"] = 10.5
        metrics["wait_time_ms"] = 1.25
        metrics["measurements"][0].update(ttft_ms=10.5, wait_time_ms=1.25)
        parsed = parse_result_envelope(value)
        self.assertEqual(value, parsed.to_dict())

    @staticmethod
    def _concurrent_prefill_result():
        value = result_envelope_prefill_result()
        case = value["cases"][0]
        case["identity"].update(
            batch_size=4, measure_runs=2, measurement_contract="concurrent_batch_v1"
        )
        case["runs"] = 2
        case["metrics"].update(latency_ms=4.5, throughput_tokens_per_second=153600.0)
        case["metrics"]["measurements"] = [
            {
                "run": run,
                "request_index": member,
                "latency_ms": run * 4 + member + 1,
                "observed_reuse_length": 512,
                "round_elapsed_ms": elapsed,
                "measurement_contract": "concurrent_batch_v1",
            }
            for (run, elapsed) in enumerate((10.0, 20.0))
            for member in range(4)
        ]
        return value

    def test_concurrent_prefill_rejects_missing_entire_round(self):
        value = self._concurrent_prefill_result()
        case = value["cases"][0]
        case["runs"] = 1
        case["metrics"]["measurements"] = case["metrics"]["measurements"][:4]
        case["metrics"].update(latency_ms=2.5, throughput_tokens_per_second=204800.0)
        with self.assertRaisesRegex(ResultSchemaError, "measure_runs"):
            parse_result_envelope(value)

    def test_prefill_warmup_metadata_round_trips_and_checks_types(self):
        value = result_envelope_prefill_result()
        metrics = value["cases"][0]["metrics"]
        metrics.update(
            warmup_used=False,
            warmup_anomaly={"deviation_ratio": 0.9, "dropped_runs": [0]},
        )
        self.assertEqual(value, parse_result_envelope(value).to_dict())
        for field, invalid in (("warmup_used", 1), ("warmup_anomaly", [])):
            with self.subTest(field=field):
                bad = copy.deepcopy(value)
                bad["cases"][0]["metrics"][field] = invalid
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(bad)

    def test_concurrent_prefill_accepts_complete_round_after_sentinel_drop(self):
        value = self._concurrent_prefill_result()
        case = value["cases"][0]
        case["metrics"]["measurements"] = case["metrics"]["measurements"][4:]
        case["metrics"].update(
            latency_ms=6.5,
            throughput_tokens_per_second=102400.0,
            warmup_used=False,
            warmup_anomaly={"deviation_ratio": 0.9, "dropped_runs": [0]},
        )
        self.assertEqual(value, parse_result_envelope(value).to_dict())

    def test_concurrent_prefill_rejects_no_remaining_rounds(self):
        value = self._concurrent_prefill_result()
        value["cases"][0]["metrics"]["measurements"] = []
        with self.assertRaises(ResultSchemaError):
            parse_result_envelope(value)

    def test_concurrent_prefill_evidence_rejects_tampering(self):
        mutations = {
            "duplicate": lambda c: c["metrics"]["measurements"][1].update(
                request_index=0
            ),
            "missing": lambda c: c["metrics"]["measurements"].pop(),
            "elapsed_inconsistent": lambda c: c["metrics"]["measurements"][0].update(
                round_elapsed_ms=11
            ),
            "reuse": lambda c: c["metrics"]["measurements"][0].update(
                observed_reuse_length=511
            ),
            "throughput_summary": lambda c: c["metrics"].update(
                throughput_tokens_per_second=204800
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                value = self._concurrent_prefill_result()
                mutate(value["cases"][0])
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(value)

    def test_decode_rejects_removed_search_policy(self):
        value = result_envelope_decode_result()
        value["cases"][0]["identity"]["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 4,
            "target_tpot_ms": 10.0,
        }
        value["cases"][0]["metrics"]["search"] = {
            "target_tpot_ms": 10.0,
            "max_size": 4,
            "candidate_step": 4,
            "outcome": "found",
            "best_batch_size": 4,
            "observations": [{"batch_size": 4, "outcome": "satisfied"}],
        }
        with self.assertRaisesRegex(ResultSchemaError, "batch_policy.strategy"):
            parse_result_envelope(value)

    def test_decode_rejects_removed_search_field_even_when_null(self):
        value = result_envelope_decode_result()
        value["cases"][0]["metrics"]["search"] = None
        with self.assertRaisesRegex(
            ResultSchemaError, "metrics\\.search: unknown field"
        ):
            parse_result_envelope(value)

    def test_envelope_round_trips_as_golden_schema(self):
        for factory, metric_type in (
            (result_envelope_prefill_result, PrefillMetrics),
            (result_envelope_decode_result, DecodeMetrics),
        ):
            with self.subTest(mode=metric_type.__name__):
                source = factory()
                envelope = parse_result_envelope(source)
                self.assertIsInstance(envelope.cases[0].metrics, metric_type)
                self.assertEqual(source, envelope.to_dict())
                if metric_type is DecodeMetrics:
                    self.assertEqual(16, envelope.cases[0].metrics.output_length)


class DecodeMetricsTest(unittest.TestCase):

    def test_valid_and_partially_failed_batches_preserve_all_metrics(self):
        valid = build_decode_case_result(
            decode_metrics_case(),
            selected=decode_metrics_selected(),
            topology=decode_metrics_TOPOLOGY,
        )
        self.assertEqual("ok", valid["status"])
        self.assertEqual(2, valid["metrics"]["successful_requests"])
        self.assertEqual(0, valid["metrics"]["failed_requests"])
        self.assertEqual(
            {"wait": 1.5, "prefill": 2.5, "decode": 8.5, "tpot": 2.5, "total": 11.5},
            valid["metrics"]["latency_ms"],
        )
        self.assertEqual(640.0, valid["metrics"]["throughput_tokens_per_second"])
        partial = build_decode_case_result(
            decode_metrics_case(),
            selected=decode_metrics_selected(
                decode_metrics_measurements(second_success=False)
            ),
            topology=decode_metrics_TOPOLOGY,
        )
        self.assertEqual("request_failed", partial["status"])
        self.assertEqual(1, partial["metrics"]["successful_requests"])
        self.assertEqual(1, partial["metrics"]["failed_requests"])
        self.assertFalse(
            partial["metrics"]["measurements"][0]["requests"][1]["success"]
        )

    def test_finalize_round_trip_preserves_distribution_and_profile_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            case = decode_metrics_case()
            case["profiling"]["enabled"] = True
            plan = build_execution_plan(
                mode="decode",
                action="run",
                resolved=resolve_config(decode_metrics_source()),
                cases=[
                    {key: value for (key, value) in case.items() if key != "case_id"}
                ],
                result_directory=str(root),
            )
            case["case_id"] = plan.cases[0].case_id
            result = build_decode_case_result(
                case,
                selected=decode_metrics_selected(),
                topology=decode_metrics_TOPOLOGY,
                profiling={
                    "status": "failed",
                    "manifest_path": "profiles/test/profile_manifest.json",
                    "error": "trace flush timeout",
                },
            )
            initialize_run_directory(
                root,
                source_config=decode_metrics_source(),
                plan=plan,
                run_id="decode-run",
                code_version="rev",
            )
            envelope = finalize_decode_run(
                root, plan=plan, run_id="decode-run", case_results=[result]
            )
            parsed = parse_result_envelope(envelope)
            metrics = parsed.cases[0].metrics.to_dict()
            self.assertEqual([64, 128], metrics["input_lengths"])
            self.assertEqual(4, metrics["output_length"])
            self.assertEqual(2, metrics["global_batch_size"])
            self.assertEqual(decode_metrics_TOPOLOGY, metrics["topology"])
            self.assertEqual(
                [64, 128], metrics["distribution"]["actual_sequence_lengths"]
            )
            self.assertNotIn("search", metrics)
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())
            report = root / "reports" / "decode_report.json"
            self.assertEqual("formal", json.loads(report.read_text())["report_kind"])
            self.assertEqual("completed", envelope["status"])
            self.assertEqual(1, envelope["summary"]["profiling_failed_cases"])
            self.assertEqual(
                1,
                json.loads((root / "results" / "summary.json").read_text())[
                    "profiling_failed_cases"
                ],
            )
            paths = [
                root / "results" / "result.json",
                root / "results" / "summary.json",
                report,
                root / "results" / "cases" / f"{case['case_id']}.json",
            ]
            before = {path: path.read_bytes() for path in paths}
            invalid = copy.deepcopy(result)
            invalid["metrics"]["profiling"]["manifest_path"] = None
            with self.assertRaises(ResultSchemaError):
                finalize_decode_run(
                    root, plan=plan, run_id="decode-run", case_results=[invalid]
                )
            self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_metrics_count_every_formal_round_and_preserve_fixed_shape(self):
        case = decode_metrics_case()
        case["measure_runs"] = 2
        measurements = decode_metrics_measurements() + decode_metrics_measurements(
            second_success=False
        )
        measurements[1]["run"] = 1
        selected = decode_metrics_selected(measurements)
        result = build_decode_case_result(
            case, selected=selected, topology=decode_metrics_TOPOLOGY
        )
        self.assertEqual("request_failed", result["status"])
        metrics = result["metrics"]
        self.assertEqual(2, metrics["global_batch_size"])
        self.assertEqual([64, 128], metrics["input_lengths"])
        self.assertEqual(3, metrics["successful_requests"])
        self.assertEqual(1, metrics["failed_requests"])
        self.assertEqual(2, len(metrics["measurements"]))
        self.assertEqual([64, 128], metrics["distribution"]["actual_sequence_lengths"])
        measurements[0]["requests"][0]["latency_ms"]["tpot"] = 999.0
        self.assertEqual(
            2.0, metrics["measurements"][0]["requests"][0]["latency_ms"]["tpot"]
        )


class PrefillFinalizeTest(unittest.TestCase):

    def test_ssh_missing_measurements_remain_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            plan = replace(prefill_adapter_finalize_plan(root), backend="ssh")
            envelope = finalize_prefill_run(
                root,
                plan=plan,
                run_id="ssh-run",
                case_results=[],
                terminal_state=RunState.FAILED,
            )
            parsed = parse_result_envelope(envelope)
            self.assertEqual("ssh", parsed.backend)
            self.assertEqual("failed", parsed.status.value)

    def test_writes_valid_common_envelope_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir(parents=True)
            plan = prefill_adapter_finalize_plan(root)
            identity = plan.cases[0].to_dict()["identity"]
            result = {
                "case_id": plan.cases[0].case_id,
                "identity": identity,
                "status": "ok",
                "runs": 1,
                "errors": [],
                "metrics": {
                    "input_length": 512,
                    "expected_reuse_length": 0,
                    "observed_reuse_length": 0,
                    "compute_length": 512,
                    "latency_ms": 10.0,
                    "throughput_tokens_per_second": 51200.0,
                    "validation": {"shape": True, "reuse": True, "timing": True},
                    "seed": 1,
                    "prefix_policy": {"kind": "shared_seed"},
                    "measurements": [{"run": 0, "latency_ms": 10.0}],
                },
            }
            envelope = finalize_prefill_run(
                root, plan=plan, run_id="prefill-run", case_results=[result]
            )
            stream = root / "results" / "cases.jsonl"
            before = stream.read_bytes()
            finalize_prefill_run(
                root, plan=plan, run_id="prefill-run", case_results=[result]
            )
            self.assertEqual(before, stream.read_bytes())
            self.assertEqual(1, len(before.splitlines()))
            parsed = parse_result_envelope(envelope)
            self.assertEqual("prefill", parsed.mode)
            self.assertEqual("completed", parsed.status.value)
            self.assertTrue((root / "results" / "result.json").is_file())
            self.assertTrue((root / "results" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
