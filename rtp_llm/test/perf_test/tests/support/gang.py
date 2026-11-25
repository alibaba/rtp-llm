"""Performance framework behavior and failure regressions."""

import json

from rtp_llm.test.perf_test.aone.config_schema import parse_execution
from rtp_llm.test.perf_test.aone.plan import plan_resources
from rtp_llm.test.perf_test.capabilities import validate_aone_capability
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_config,
    fixture_engine,
    fixture_workload,
)


def _gang_source():
    source = prefill_source(
        model={"label": "gang", "tokenizer": "/m", "topology": dict(TOPOLOGY)},
        engine={
            "args": {"reuse_cache": 1},
            "max_seq_len": 4097,
            "concurrency_limit": 1,
        },
        execution={"max_parallel": MACHINE_COUNT, "gpus_per_machine": LOCAL_WORLD_SIZE},
    )
    source["execution"].pop("shard_count")
    return source


def _cases():
    return [
        {
            "input_len": input_len,
            "cache_len": cache_len,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        }
        for input_len, cache_len in ((512, 0), (1024, 512))
    ]


def _setup_run(root, source=None, *, run_id=None):
    source = _gang_source() if source is None else source
    root.parent.mkdir(parents=True, exist_ok=True)
    config_path = root.parent / "submit-host-only.json"
    config_path.write_text(json.dumps(source))
    plan = build_execution_plan(
        resource_planner=plan_resources,
        capability_validator=validate_aone_capability,
        mode="prefill",
        action="run",
        resolved=load_resolved_config(config_path, execution_parser=parse_execution),
        cases=_cases(),
        result_directory=str(root),
    )
    initialize_run_directory(
        root,
        source_config=source,
        plan=plan,
        run_id=RUN_ID if run_id is None else run_id,
        code_version="rev",
    )
    config_path.unlink()
    return plan


def _case_result(case, status="ok"):
    if isinstance(case, dict):
        case_id = case["case_id"]
        identity = {key: value for key, value in case.items() if key != "case_id"}
    else:
        case_id, identity = case.case_id, thaw(case.identity)
    ok = status == "ok"
    return {
        "case_id": case_id,
        "identity": identity,
        "status": status,
        "runs": 1,
        "errors": [] if ok else [{"code": status, "message": "remote case failed"}],
        "metrics": (
            {
                "input_length": identity["input_len"],
                "expected_reuse_length": identity["cache_len"],
                "observed_reuse_length": identity["cache_len"],
                "compute_length": identity["compute_len"],
                "latency_ms": 10.0,
                "throughput_tokens_per_second": 1000.0,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 1,
                "prefix_policy": {"kind": "shared_seed"},
                "measurements": [{"run": 0, "latency_ms": 10.0}],
            }
            if ok
            else None
        ),
    }


def _envelope(plan, cases, status):
    successful = sum((case["status"] == "ok" for case in cases))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "mode": "prefill",
        "backend": "aone",
        "status": status,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "model": {
            "label": "gang",
            "type": "registered",
            "checkpoint": "/m",
            "topology": dict(TOPOLOGY),
        },
        "summary": {
            "planned_cases": len(cases),
            "successful_cases": successful,
            "failed_cases": len(cases) - successful,
        },
        "cases": cases,
    }


def _write_master_run_tree(outputs, plan, status="completed"):
    cases = [_case_result(case) for case in plan.cases]
    return _write_master_run_tree_with_cases(outputs, plan, cases, status)


def _write_master_run_tree_with_cases(outputs, plan, cases, status):
    run = outputs / "run"
    (run / "results").mkdir(parents=True)
    (run / "results" / "cases.jsonl").write_text(
        "".join((json.dumps(case) + "\n" for case in cases)), encoding="utf-8"
    )
    (run / "results" / "result.json").write_text(
        json.dumps(_envelope(plan, cases, status)), encoding="utf-8"
    )
    (run / "results" / "summary.json").write_text(
        json.dumps({"run_id": RUN_ID, "status": status}), encoding="utf-8"
    )
    (run / "reports").mkdir()
    (run / "reports" / "master_report.json").write_text("{}", encoding="utf-8")
    (run / "traces").mkdir()
    (run / "traces" / "master_trace.json").write_text("{}", encoding="utf-8")
    (run / "profiles").mkdir()
    (run / "profiles" / "master_profile.json").write_text("{}", encoding="utf-8")
    return run


def _write_testlogs_tree(base, plan, *, status="completed", master=True):
    for index in range(MACHINE_COUNT):
        shard = base / f"shard_{index + 1}_of_{MACHINE_COUNT}"
        outputs = shard / "test.outputs"
        outputs.mkdir(parents=True)
        (shard / "test.log").write_text(f"shard {index} log\n", encoding="utf-8")
        (outputs / "gang_worker.json").write_text(
            json.dumps({"shard_index": index, "state": "completed"}), encoding="utf-8"
        )
        if index == 0 and master:
            _write_master_run_tree(outputs, plan, status=status)


TOPOLOGY = {"tp": 16, "dp": 1, "pp": 1, "ep": 16}
MACHINE_COUNT = 2
LOCAL_WORLD_SIZE = 8
RUN_ID = "run-gang"


def prefill_source(
    *, model=None, engine=None, workload=None, execution=None, output=None
):
    """Fresh minimal remote prefill config; section overrides are explicit per case."""
    source = fixture_config(
        engine=fixture_engine(concurrency_limit=2),
        workload=fixture_workload(compute_steps=[256]),
        execution=aone_execution(shard_count=2),
    )
    for section, overrides in (
        ("model", model),
        ("engine", engine),
        ("workload", workload),
        ("execution", execution),
        ("output", output),
    ):
        if overrides is not None:
            source[section].update(overrides)
    return source


def aone_execution(**fields):
    execution = {
        "backend": "aone",
        "resource": "daily",
        "gpu_type": "H20",
        "max_parallel": 2,
    }
    execution.update(fields)
    return execution


def aone_gang_config(execution, topology=None, mode="prefill"):
    if mode == "decode":
        workload = {
            "kind": "grid",
            "input_lengths": [128],
            "output_lengths": [16],
            "batch": {"strategy": "fixed", "sizes": [2]},
        }
    else:
        workload = fixture_workload()
    source = prefill_source(
        model={
            "label": "test-model",
            "type": "registered_model",
            "checkpoint": "/models/test",
            "topology": topology or {"tp": 2, "dp": 1, "pp": 1, "ep": 1},
        },
        engine={"concurrency_limit": 8},
    )
    source.update(workload=workload, execution=execution)
    return source
