from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION


def _result():
    identity = {
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
    }
    latency = {"wait": 1.0, "prefill": 2.0, "decode": 8.0, "tpot": 0.5, "total": 11.0}
    key = case_id("decode", identity)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": "run-001",
        "mode": "decode",
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
        "summary": {
            "planned_cases": 1,
            "successful_cases": 1,
            "failed_cases": 0,
            "profiling_failed_cases": 0,
        },
        "cases": [
            {
                "case_id": key,
                "identity": identity,
                "status": "ok",
                "runs": 1,
                "errors": [],
                "metrics": {
                    "workload_kind": "grid",
                    "input_lengths": [128] * 4,
                    "output_length": 16,
                    "global_batch_size": 4,
                    "topology": {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
                    "successful_requests": 4,
                    "failed_requests": 0,
                    "latency_ms": latency,
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
                                    "latency_ms": dict(latency),
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
        ],
    }


def _baseline(*entries):
    return {"schema_version": 1, "entries": list(entries)}


def _entry(key, metric, value, direction, threshold=0.05):
    return {
        "case_key": key,
        "metric": metric,
        "value": value,
        "direction": direction,
        "threshold": threshold,
    }
