"""Core performance behavior contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.results import (
    DecodeMetrics,
    PrefillMetrics,
    ResultSchemaError,
    parse_result_envelope,
)
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_execution,
    fixture_model,
    fixture_workload,
)


def result_envelope_base_result(mode):
    return {
        "schema_version": 2,
        "run_id": "run-001",
        "mode": mode,
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": {
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        },
        "summary": {"planned_cases": 1, "successful_cases": 1, "failed_cases": 0},
        "cases": [],
    }


def result_envelope_prefill_result():
    value = result_envelope_base_result("prefill")
    value["cases"] = [
        {
            "case_id": "prefill-case-a",
            "identity": {"input_len": 1024, "cache_len": 512, "batch_size": 1},
            "status": "ok",
            "runs": 3,
            "errors": [],
            "metrics": {
                "input_length": 1024,
                "expected_reuse_length": 512,
                "observed_reuse_length": 512,
                "compute_length": 512,
                "latency_ms": 10.5,
                "throughput_tokens_per_second": 48761.9,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 7,
                "prefix_policy": {"kind": "shared_seed", "commit_tail_tokens": 64},
                "measurements": [{"run": 0, "latency_ms": 10.5}],
            },
        }
    ]
    return value


def result_envelope_decode_result():
    value = result_envelope_base_result("decode")
    value["summary"]["profiling_failed_cases"] = 0
    value["cases"] = [
        {
            "case_id": "decode-case-a",
            "identity": {
                "workload_kind": "grid",
                "input_lengths": [128] * 4,
                "output_length": 16,
                "batch_policy": {"strategy": "fixed", "size": 4},
                "measure_runs": 1,
                "warmup_runs": 0,
                "generate_config": {},
                "profiling": {
                    "enabled": False,
                    "runs": 1,
                    "steps": 3,
                    "enable_all_ranks": True,
                    "flush_timeout_seconds": 60,
                },
                "distribution": None,
            },
            "status": "ok",
            "runs": 1,
            "errors": [],
            "metrics": {
                "workload_kind": "grid",
                "input_lengths": [128, 128, 128, 128],
                "output_length": 16,
                "global_batch_size": 4,
                "topology": {"tp": 1, "dp": 2, "pp": 1, "ep": 1},
                "successful_requests": 4,
                "failed_requests": 0,
                "latency_ms": {
                    "wait": 1.0,
                    "prefill": 2.0,
                    "decode": 8.0,
                    "tpot": 0.5,
                    "total": 11.0,
                },
                "throughput_tokens_per_second": 8000.0,
                "measurements": [
                    {
                        "run": 0,
                        "throughput_tokens_per_second": 8000.0,
                        "worker_allocations": [],
                        "requests": [
                            {
                                "request_index": index,
                                "success": True,
                                "input_length": 128,
                                "output_length": 16,
                                "latency_ms": {
                                    "wait": 1.0,
                                    "prefill": 2.0,
                                    "decode": 8.0,
                                    "tpot": 0.5,
                                    "total": 11.0,
                                },
                            }
                            for index in range(4)
                        ],
                    }
                ],
                "search": None,
                "distribution": None,
                "profiling": {
                    "status": "disabled",
                    "manifest_path": None,
                    "error": None,
                },
            },
        }
    ]
    return value


decode_metrics_TOPOLOGY = {"tp": 2, "dp": 2, "pp": 1, "ep": 4}


def decode_metrics_case():
    spec = DecodeCaseSpec(
        workload_kind="distribution",
        input_lengths=(64, 128),
        output_length=4,
        batch_policy=ExactBatchPolicy("fixed", 2),
        measure_runs=1,
        warmup_runs=0,
        distribution={"source_kind": "sequence_lengths"},
    )
    return {"case_id": "decode-case", **spec.to_identity()}


def decode_metrics_selected(measurements=None):
    return DecodeBatchMeasurement(
        batch_size=2,
        input_lengths=(64, 128),
        worker_allocations=(),
        measurements=tuple(
            decode_metrics_measurements() if measurements is None else measurements
        ),
    )


def decode_metrics_request(index, input_length, *, success=True):
    value = {
        "request_index": index,
        "success": success,
        "input_length": input_length,
        "output_length": 4,
    }
    if success:
        value["latency_ms"] = {
            "wait": 1.0 + index,
            "prefill": 2.0 + index,
            "decode": 8.0 + index,
            "tpot": 2.0 + index,
            "total": 11.0 + index,
        }
    else:
        value["error"] = "request failed"
    return value


def decode_metrics_measurements(second_success=True):
    return [
        {
            "run": 0,
            "throughput_tokens_per_second": 640.0,
            "requests": [
                decode_metrics_request(0, 64),
                decode_metrics_request(1, 128, success=second_success),
            ],
            "worker_allocations": [
                {"worker_index": 0, "request_indices": [0]},
                {"worker_index": 1, "request_indices": [1]},
            ],
        }
    ]


def decode_metrics_source():
    return {
        "schema_version": 1,
        "model": fixture_model(topology=decode_metrics_TOPOLOGY),
        "engine": fixture_engine(concurrency_limit=2),
        "workload": {
            "kind": "grid",
            "input_lengths": [64],
            "output_lengths": [4],
            "batch": {"strategy": "fixed", "sizes": [2]},
        },
        "execution": fixture_execution(devices=[0, 1, 2, 3]),
        "output": {},
        "regression": {},
        "publish": {},
    }


def prefill_adapter_finalize_plan(root):
    config = {
        "schema_version": 1,
        "model": fixture_model(label="model"),
        "engine": fixture_engine(
            max_seq_len=1024, max_batch_tokens=1024, concurrency_limit=1
        ),
        "workload": fixture_workload(max_input_len=512, cache_lengths=[0]),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=[
            {
                "input_len": 512,
                "cache_len": 0,
                "compute_len": 512,
                "batch_size": 1,
                "measure_runs": 1,
                "seed": 1,
                "prefix_policy": "shared_seed",
            }
        ],
        result_directory=str(root),
    )


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

    def test_finalize_round_trip_preserves_distribution_and_search_trajectory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            case = decode_metrics_case()
            case["input_lengths"] = [64, 128, 256, 512]
            case["batch_policy"] = {
                "strategy": "target_tpot",
                "max_size": 4,
                "candidate_step": 2,
                "target_tpot_ms": 10.0,
            }
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
            search = {
                "target_tpot_ms": 10.0,
                "max_size": 4,
                "candidate_step": 2,
                "outcome": "found",
                "best_batch_size": 2,
                "observations": [
                    {"batch_size": 2, "outcome": "satisfied"},
                    {"batch_size": 4, "outcome": "target_exceeded"},
                ],
            }
            result = build_decode_case_result(
                case,
                selected=decode_metrics_selected(),
                topology=decode_metrics_TOPOLOGY,
                search=search,
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
            self.assertEqual(
                [2, 4],
                [item["batch_size"] for item in metrics["search"]["observations"]],
            )
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

    def test_metrics_count_every_formal_round_and_use_selected_shape(self):
        case = decode_metrics_case()
        case["measure_runs"] = 2
        case["input_lengths"] = [64, 128, 256, 512]
        case["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 2,
            "target_tpot_ms": 10.0,
        }
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


class ResultEnvelopeTest(unittest.TestCase):

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

    def test_failed_tpot_keeps_only_structured_search_diagnostics(self):
        value = result_envelope_decode_result()
        case = value["cases"][0]
        case["identity"]["batch_policy"] = {
            "strategy": "target_tpot",
            "max_size": 4,
            "candidate_step": 4,
            "target_tpot_ms": 10.0,
        }
        search = {
            "target_tpot_ms": 10.0,
            "max_size": 4,
            "candidate_step": 4,
            "outcome": "no_satisfying_candidate",
            "best_batch_size": None,
            "observations": [{"batch_size": 1, "outcome": "target_exceeded"}],
        }
        case.update(
            status="target_tpot_not_met",
            metrics=None,
            errors=[
                {
                    "code": "target_tpot_not_met",
                    "message": "no candidate",
                    "details": {"search": search},
                }
            ],
        )
        value["status"] = "partial"
        value["summary"].update(successful_cases=0, failed_cases=1)
        self.assertEqual(value, parse_result_envelope(value).to_dict())
        for key, bad in (
            ("observations", [{"batch_size": 5, "outcome": "satisfied"}]),
        ):
            with self.subTest(key=key):
                invalid = copy.deepcopy(value)
                invalid["cases"][0]["errors"][0]["details"]["search"][key] = bad
                with self.assertRaises(ResultSchemaError):
                    parse_result_envelope(invalid)

    def test_prefill_envelope_round_trips_as_golden_schema(self):
        source = result_envelope_prefill_result()
        envelope = parse_result_envelope(source)
        self.assertIsInstance(envelope.cases[0].metrics, PrefillMetrics)
        self.assertEqual(source, envelope.to_dict())

    def test_decode_envelope_round_trips_as_golden_schema(self):
        source = result_envelope_decode_result()
        envelope = parse_result_envelope(source)
        self.assertIsInstance(envelope.cases[0].metrics, DecodeMetrics)
        self.assertEqual(16, envelope.cases[0].metrics.output_length)
        self.assertEqual(source, envelope.to_dict())


class PrefillFinalizeTest(unittest.TestCase):

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
