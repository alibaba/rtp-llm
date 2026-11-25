"""Fixtures shared by tests split from result_envelope_test."""

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.plan import build_execution_plan
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_config,
    fixture_engine,
    fixture_envelope,
    fixture_execution,
    fixture_model,
    fixture_workload,
)


def default_executor_request_round(input_lengths, output_length, *, latency=None):
    return tuple(
        {
            "request_index": index,
            "success": True,
            "input_length": input_length,
            "output_length": output_length,
            "latency_ms": dict(
                latency
                or {
                    "wait": 1.0,
                    "prefill": 2.0,
                    "decode": 15.0,
                    "tpot": 1.0,
                    "total": 17.0,
                }
            ),
        }
        for index, input_length in enumerate(input_lengths)
    )


def result_envelope_base_result(mode):
    return fixture_envelope(
        mode, [], summary={"planned_cases": 1, "successful_cases": 1, "failed_cases": 0}
    )


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
    latency = {"wait": 1.0, "prefill": 2.0, "decode": 8.0, "tpot": 0.5, "total": 11.0}
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
                "latency_ms": dict(latency),
                "throughput_tokens_per_second": 8000.0,
                "measurements": [
                    {
                        "run": 0,
                        "throughput_tokens_per_second": 8000.0,
                        "worker_allocations": [],
                        "requests": list(
                            default_executor_request_round(
                                (128,) * 4, 16, latency=latency
                            )
                        ),
                    }
                ],
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
    return fixture_config(
        model=fixture_model(topology=decode_metrics_TOPOLOGY),
        engine=fixture_engine(concurrency_limit=2),
        workload={
            "kind": "grid",
            "input_lengths": [64],
            "output_lengths": [4],
            "batch": {"strategy": "fixed", "sizes": [2]},
        },
        execution=fixture_execution(devices=[0, 1, 2, 3]),
    )


def prefill_adapter_finalize_plan(root):
    config = fixture_config(
        model=fixture_model(label="model"),
        engine=fixture_engine(
            max_seq_len=1024, max_batch_tokens=1024, concurrency_limit=1
        ),
        workload=fixture_workload(max_input_len=512, cache_lengths=[0]),
    )
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
