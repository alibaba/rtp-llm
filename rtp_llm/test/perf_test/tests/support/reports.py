"""Fixtures shared by tests split from prefill_report_test."""

import copy
import itertools
from dataclasses import replace

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.metrics import build_decode_case_result
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.plan import build_execution_plan, case_id, thaw
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_config,
    fixture_engine,
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


def decode_report_result(spec, batch):
    identity = spec.to_identity()
    return build_decode_case_result(
        {"case_id": case_id("decode", identity), **identity},
        selected=decode_report_measurement(
            batch, spec.input_lengths[:batch], spec.output_length
        ),
        topology=decode_report_TOPOLOGY,
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


def prefill_report_case_bytes(root):
    return {
        path.name: path.read_bytes() for path in (root / "results").glob("cases.jsonl")
    }


def _setup_prefill_report_run(
    root, statuses=("ok", "ok"), *, concurrent=False, measure_runs=1, grid_names=None
):
    config = fixture_config(
        engine=fixture_engine(max_seq_len=4097, concurrency_limit=1),
        workload=fixture_workload(compute_steps=[256], cache_lengths=[0]),
    )
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
