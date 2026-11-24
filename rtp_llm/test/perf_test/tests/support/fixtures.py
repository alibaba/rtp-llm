"""Performance framework behavior and failure regressions."""

from pathlib import Path


def fixture_execution(**overrides):
    """Fresh minimal execution fixture with case-specific overrides."""
    return {**{"backend": "local", "devices": [0]}, **overrides}


def fixture_workload(**overrides):
    """Fresh minimal workload fixture with case-specific overrides."""
    return {
        **{
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [128],
            "batch_sizes": [1],
        },
        **overrides,
    }


def fixture_model(**overrides):
    """Fresh minimal model fixture with case-specific overrides."""
    return {**{"label": "test", "type": "registered", "checkpoint": "/m"}, **overrides}


def fixture_engine(**overrides):
    """Fresh minimal engine fixture with case-specific overrides."""
    return {
        **{"max_seq_len": 4096, "max_batch_tokens": 8192, "concurrency_limit": 8},
        **overrides,
    }


def fixture_config(**sections):
    """Fresh valid input; replace whole sections so missing-field tests stay explicit."""
    return {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(),
        "workload": fixture_workload(),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
        **sections,
    }


def fixture_envelope(mode, cases, **overrides):
    """Common raw result input; callers can replace any field to test corruption."""
    cases = list(cases)
    successful = sum(case["status"] == "ok" for case in cases)
    summary = {
        "planned_cases": len(cases),
        "successful_cases": successful,
        "failed_cases": len(cases) - successful,
    }
    if mode == "decode":
        summary["profiling_failed_cases"] = 0
    return {
        "schema_version": 2,
        "run_id": "run-001",
        "mode": mode,
        "backend": "local",
        "status": "completed",
        "config_fingerprint": "a" * 64,
        "workload_fingerprint": "b" * 64,
        "model": fixture_model(
            label="test-model",
            type="registered_model",
            checkpoint="/models/test",
            topology={"tp": 1, "dp": 1, "pp": 1, "ep": 1},
        ),
        "summary": summary,
        "cases": cases,
        **overrides,
    }


PERF_TEST_ROOT = Path(__file__).resolve().parents[2]


def config_resolution_base_config():
    return fixture_config(
        model=fixture_model(
            label="test-model", type="registered_model", checkpoint="/models/test"
        ),
        engine=fixture_engine(args={"act_type": "BF16", "reuse_cache": 1}),
        execution={"backend": "local"},
    )
