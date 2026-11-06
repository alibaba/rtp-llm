"""Core performance behavior contracts."""

import copy
import itertools
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.report import generate_decode_report
from rtp_llm.test.perf_test.execution.engine_session import EngineSession
from rtp_llm.test.perf_test.multi_node.analyzer import merge_decode_results as merge
from rtp_llm.test.perf_test.plan import build_execution_plan, case_id, thaw
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_execution,
    fixture_model,
    fixture_workload,
)

decode_report_TOPOLOGY = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
decode_report_DISABLED = {"status": "disabled", "manifest_path": None, "error": None}


def decode_report_exact_case(batch):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy("fixed", batch),
        measure_runs=1,
        warmup_runs=0,
    )


def decode_report_measurement(batch, inputs, output):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=inputs,
        worker_allocations=(),
        measurements=(
            {
                "run": 0,
                "throughput_tokens_per_second": 100.0,
                "worker_allocations": [],
                "requests": [
                    {
                        "request_index": index,
                        "success": True,
                        "input_length": length,
                        "output_length": output,
                        "latency_ms": {
                            "wait": 1.0,
                            "prefill": 2.0,
                            "decode": 15.0,
                            "tpot": 1.0,
                            "total": 17.0,
                        },
                    }
                    for (index, length) in enumerate(inputs)
                ],
            },
        ),
    )


def decode_report_result(spec, batch, search=None):
    identity = spec.to_identity()
    return build_decode_case_result(
        {"case_id": case_id("decode", identity), **identity},
        selected=decode_report_measurement(
            batch, spec.input_lengths[:batch], spec.output_length
        ),
        topology=decode_report_TOPOLOGY,
        search=search,
        profiling=decode_report_DISABLED,
    )


def decode_report_exact_cases(inputs, outputs, batches):
    return [
        decode_report_result(
            replace(
                decode_report_exact_case(batch),
                input_lengths=(input_length,) * batch,
                output_length=output_length,
            ),
            batch,
        )
        for (batch, input_length, output_length) in itertools.product(
            batches, inputs, outputs
        )
    ]


def decode_report_tpot_cases(inputs, outputs):
    cases = []
    for index, (input_length, output_length) in enumerate(
        itertools.product(inputs, outputs)
    ):
        batch = (8, 12)[index]
        spec = replace(
            decode_report_exact_case(16),
            input_lengths=(input_length,) * 16,
            output_length=output_length,
            batch_policy=TargetTPOTPolicy("target_tpot", 16, 4, 10.0),
        )
        search = {
            "target_tpot_ms": 10.0,
            "max_size": 16,
            "candidate_step": 4,
            "outcome": "found",
            "best_batch_size": batch,
            "observations": [{"batch_size": batch, "outcome": "satisfied"}],
        }
        cases.append(decode_report_result(spec, batch, search))
    return cases


def prefill_report_case_bytes(root):
    return {
        path.name: path.read_bytes() for path in (root / "results").glob("cases.jsonl")
    }


def decode_multi_node_analyzer_decode_case_v2(
    *, case_id, policy, batch, profile, search
):
    inputs = [128] * batch
    return {
        "case_id": case_id,
        "identity": {
            "workload_kind": "grid",
            "input_lengths": [128] * (policy.get("max_size") or batch),
            "output_length": 16,
            "batch_policy": policy,
            "measure_runs": 1,
            "warmup_runs": 0,
            "generate_config": {},
            "profiling": {
                "enabled": profile["status"] != "disabled",
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
            "input_lengths": inputs,
            "output_length": 16,
            "global_batch_size": batch,
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
            "successful_requests": batch,
            "failed_requests": 0,
            "latency_ms": {
                "wait": 1.0,
                "prefill": 2.0,
                "decode": 8.0,
                "tpot": 1.0,
                "total": 11.0,
            },
            "throughput_tokens_per_second": 1000.0,
            "measurements": [
                {
                    "run": 0,
                    "worker_allocations": [],
                    "throughput_tokens_per_second": 1000.0,
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
                                "tpot": 1.0,
                                "total": 11.0,
                            },
                        }
                        for index in range(batch)
                    ],
                }
            ],
            "search": search,
            "distribution": None,
            "profiling": profile,
        },
    }


def decode_multi_node_analyzer_decode_envelope_v2():
    return {
        "schema_version": 2,
        "run_id": "decode-run",
        "mode": "decode",
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": {
            "label": "test",
            "type": "registered",
            "checkpoint": "/m",
            "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        },
        "summary": {
            "planned_cases": 2,
            "successful_cases": 2,
            "failed_cases": 0,
            "profiling_failed_cases": 1,
        },
        "cases": [
            decode_multi_node_analyzer_decode_case_v2(
                case_id="decode-exact",
                policy={"strategy": "fixed", "size": 4},
                batch=4,
                profile={"status": "disabled", "manifest_path": None, "error": None},
                search=None,
            ),
            decode_multi_node_analyzer_decode_case_v2(
                case_id="decode-tpot",
                policy={
                    "strategy": "target_tpot",
                    "max_size": 16,
                    "candidate_step": 4,
                    "target_tpot_ms": 10.0,
                },
                batch=8,
                profile={
                    "status": "failed",
                    "manifest_path": "profiles/decode-tpot/profile_manifest.json",
                    "error": "trace flush timeout",
                },
                search={
                    "target_tpot_ms": 10.0,
                    "max_size": 16,
                    "candidate_step": 4,
                    "outcome": "found",
                    "best_batch_size": 8,
                    "observations": [{"batch_size": 8, "outcome": "satisfied"}],
                },
            ),
        ],
    }


def decode_multi_node_analyzer_write_collected_run(
    root,
    *,
    input_length=128,
    output_length=64,
    strategy="target_tpot",
    batch_size=8,
    nested=False,
):
    task = root / "Task_dp_size-2"
    run = task / "run" if nested else task
    result_path = run / "results" / "result.json"
    result_path.parent.mkdir(parents=True)
    envelope = decode_multi_node_analyzer_decode_envelope_v2()
    envelope["cases"] = [envelope["cases"][1]]
    envelope["summary"].update(planned_cases=1, successful_cases=1)
    case = envelope["cases"][0]
    case["identity"]["input_lengths"] = [input_length] * 16
    case["identity"]["output_length"] = output_length
    case["identity"]["batch_policy"]["strategy"] = strategy
    case["metrics"].update(
        input_lengths=[input_length] * batch_size,
        output_length=output_length,
        global_batch_size=batch_size,
        successful_requests=batch_size,
    )
    case["metrics"]["search"]["best_batch_size"] = batch_size
    requests = case["metrics"]["measurements"][0]["requests"]
    template = copy.deepcopy(requests[0])
    requests[:] = [
        {
            **copy.deepcopy(template),
            "request_index": index,
            "input_length": input_length,
            "output_length": output_length,
        }
        for index in range(batch_size)
    ]
    result_path.write_text(json.dumps(envelope), encoding="utf-8")
    return result_path


def _setup_prefill_report_run(
    root, statuses=("ok", "ok"), *, concurrent=False, measure_runs=1, grid_names=None
):
    config = {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(max_seq_len=4097, concurrency_limit=1),
        "workload": fixture_workload(compute_steps=[256], cache_lengths=[0]),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }
    identities = [
        {"input_len": 512, "cache_len": 0, "compute_len": 512, "batch_size": 1},
        {"input_len": 1024, "cache_len": 512, "compute_len": 512, "batch_size": 1},
    ]
    if grid_names:
        identities = [
            dict(identity, grid_name=name)
            for (identity, name) in zip(identities, grid_names)
        ]
    if concurrent:
        identities = [
            {
                **identity,
                "measure_runs": measure_runs,
                "measurement_contract": "concurrent_batch_v1",
            }
            for identity in identities
        ]
    plan = build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=identities,
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=config, plan=plan, run_id="run", code_version="rev"
    )
    store = RunStore(root)
    for index, (case, status) in enumerate(zip(plan.cases, statuses)):
        identity = thaw(case.identity)
        metrics = None
        errors = [{"code": status, "message": "invalid"}]
        if status == "ok":
            metrics = {
                "input_length": identity["input_len"],
                "expected_reuse_length": identity["cache_len"],
                "observed_reuse_length": identity["cache_len"],
                "compute_length": identity["compute_len"],
                "latency_ms": 10.0 + index,
                "throughput_tokens_per_second": 1000.0,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 1,
                "prefix_policy": {"kind": "shared_seed"},
                "measurements": [{"run": 0, "latency_ms": 10.0 + index}],
            }
            if concurrent:
                metrics["throughput_tokens_per_second"] = (
                    identity["compute_len"] * 1000.0 / 20.0
                )
                metrics["measurements"][0].update(
                    {
                        "request_index": 0,
                        "round_elapsed_ms": 20.0,
                        "observed_reuse_length": identity["cache_len"],
                        "measurement_contract": "concurrent_batch_v1",
                    }
                )
                metrics["measurements"] = [
                    {**metrics["measurements"][0], "run": run}
                    for run in range(measure_runs)
                ]
            errors = []
        store.write_prefill_case(
            case.case_id,
            {
                "case_id": case.case_id,
                "identity": identity,
                "status": status,
                "runs": measure_runs if concurrent else 1,
                "errors": errors,
                "metrics": metrics,
            },
        )
    return plan


class PrefillReportTest(unittest.TestCase):

    def test_complete_input_generates_formal_observations_formula_and_chart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = _setup_prefill_report_run(root)
            report = generate_prefill_report(root)
            self.assertEqual("formal", report["report_kind"])
            self.assertEqual(2, report["counts"]["valid"])
            self.assertEqual([], report["excluded"])
            self.assertEqual(
                [case.case_id for case in plan.cases],
                [row["case_id"] for row in report["observations"]],
            )
            self.assertTrue((root / "reports" / "prefill_report.json").is_file())
            self.assertTrue((root / "reports" / "prefill_observations.json").is_file())
            html = (root / "reports" / "prefill_report.html").read_text()
            self.assertIn("formal", html)

    def test_invalid_case_is_excluded_and_report_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup_prefill_report_run(root, statuses=("ok", "invalid_reuse"))
            report = generate_prefill_report(root)
            self.assertEqual("diagnostic", report["report_kind"])
            self.assertEqual(1, report["counts"]["valid"])
            self.assertEqual(1, report["counts"]["excluded"])
            self.assertEqual("invalid_reuse", report["excluded"][0]["status"])

    def test_rerun_never_launches_engine_or_changes_case_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            _setup_prefill_report_run(root)
            before = prefill_report_case_bytes(root)
            with patch.object(
                EngineSession, "start", side_effect=AssertionError("no engine")
            ) as start:
                first = generate_prefill_report(root)
                second = generate_prefill_report(root)
            self.assertEqual(first, second)
            start.assert_not_called()
            self.assertEqual(before, prefill_report_case_bytes(root))


class PrefillChartTest(unittest.TestCase):

    def test_standalone_charts_and_static_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            _setup_prefill_report_run(root, concurrent=True)
            report = generate_prefill_report(root)
            self.assertTrue(report["observations"])
            for row in report["observations"]:
                self.assertGreater(row["latency_ms"], 0)
                self.assertGreater(row["throughput_tokens_per_second"], 0)
            for metric in ("latency", "tpm"):
                document = (root / "reports" / f"{metric}.interactive.html").read_text()
                self.assertIn("<html", document.lower())
                self.assertEqual(
                    "{http://www.w3.org/2000/svg}svg",
                    ElementTree.parse(root / "reports" / f"{metric}.svg").getroot().tag,
                )
            with patch.dict(
                "sys.modules",
                {"plotly": None, "plotly.graph_objects": None, "plotly.offline": None},
            ):
                fallback_report = generate_prefill_report(root)
            self.assertEqual(report["observations"], fallback_report["observations"])
            for metric in ("latency", "tpm"):
                document = (root / "reports" / f"{metric}.interactive.html").read_text()
                self.assertIn("<svg", document)
                self.assertIn("request_index", document)


class DecodeReportTest(unittest.TestCase):

    def test_exact_report_groups_input_output_matrix_by_batch(self):
        cases = decode_report_exact_cases((128, 512), (16, 64), (1, 4))
        before = copy.deepcopy(cases)
        report = generate_decode_report(cases)
        self.assertEqual(2, report["schema_version"])
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual({"planned": 8, "valid": 8, "excluded": 0}, report["counts"])
        self.assertEqual(
            [1, 4], [matrix["batch_size"] for matrix in report["matrices"]]
        )
        first = report["matrices"][0]
        self.assertEqual([128, 512], first["input_lengths"])
        self.assertEqual([16, 64], first["output_lengths"])
        self.assertEqual(4, len(first["cells"]))
        self.assertEqual(1.0, first["cells"][0]["tpot_ms"])
        self.assertEqual(100.0, first["cells"][0]["throughput_tokens_per_second"])
        self.assertEqual(17.0, first["cells"][0]["total_latency_ms"])
        self.assertEqual(1.0, first["cells"][0]["success_rate"])
        self.assertEqual(
            [item["case_id"] for item in cases],
            [item["case_id"] for item in report["observations"]],
        )
        self.assertEqual(before, cases)
        first["cells"][0]["errors"].append({"message": "changed"})
        self.assertEqual([], cases[0]["errors"])

    def test_tpot_report_cells_expose_selected_batch(self):
        report = generate_decode_report(decode_report_tpot_cases((128,), (16, 64)))
        self.assertEqual("formal", report["report_kind"])
        self.assertEqual(1, len(report["matrices"]))
        self.assertEqual(
            [8, 12],
            [cell["best_batch_size"] for cell in report["matrices"][0]["cells"]],
        )
        self.assertEqual(
            ["found", "found"],
            [cell["search_outcome"] for cell in report["matrices"][0]["cells"]],
        )


class DecodeMultiNodeAnalyzerTest(unittest.TestCase):

    def test_merger_deduplicates_copied_run_identity_but_keeps_distinct_runs(self):
        for same_run in (False, True):
            with self.subTest(
                same_run=same_run
            ), tempfile.TemporaryDirectory() as directory:
                path = decode_multi_node_analyzer_write_collected_run(Path(directory))
                envelope = json.loads(path.read_text())
                if not same_run:
                    envelope["run_id"] = "another-run"
                nested = path.parents[1] / "run" / "results" / "result.json"
                nested.parent.mkdir(parents=True)
                nested.write_text(json.dumps(envelope))
                rows = merge.merge_all_results(merge.scan_result_files(directory))
                self.assertEqual(
                    1 if same_run else 2, sum((len(metrics) for (_, metrics) in rows))
                )


if __name__ == "__main__":
    unittest.main()
