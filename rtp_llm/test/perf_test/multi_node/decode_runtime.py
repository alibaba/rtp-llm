"""Unified decode planning and execution for an externally owned service."""

import json
import os
from pathlib import Path

from rtp_llm.test.perf_test.capabilities import validate_capability
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import EngineConfig, parse_config
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCaseExecutor,
)
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.planner import plan_decode_cases
from rtp_llm.test.perf_test.decode.profiling import DecodeProfiler
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.entry import _source_config
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.modes import DecodeAdapter, ModeExecutionContext
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


def build_external_decode_plan(
    config_path,
    result_directory,
    actual_topology,
    *,
    execution_parser=None,
    resource_planner=None,
    capability_validator=validate_capability,
):
    """Build the local decode case identities without creating run artifacts."""
    source = _source_config(str(config_path))
    resolved = load_resolved_config(config_path, execution_parser=execution_parser)
    config = parse_config(source, execution_parser=execution_parser)
    configured = config.model.topology.as_tuple()
    actual = tuple(actual_topology.get(name) for name in ("tp", "dp", "pp", "ep"))
    if any(type(value) is not int or value <= 0 for value in actual):
        raise PlanningError(
            "runtime topology must supply positive integer TP, DP, PP, and EP sizes"
        )
    if configured != actual:
        raise PlanningError(
            f"runtime topology {actual} does not match configured topology {configured}"
        )
    cases = plan_decode_cases(
        config.workload,
        config.engine,
        dp_size=config.model.topology.dp,
        resolved_workload=resolved.resolved_config["workload"],
    )
    plan = build_execution_plan(
        resource_planner=resource_planner,
        capability_validator=capability_validator,
        mode="decode",
        action="run",
        resolved=resolved,
        cases=cases,
        result_directory=str(result_directory),
    )
    return plan, source


def run_external_rank(
    *, world_rank, wait_for_master, execute_master, execute_worker=None
):
    """Only rank zero writes results; workers may drive synchronized local load."""
    if world_rank == 0:
        return execute_master()
    if execute_worker is None:
        wait_for_master()
    else:
        execute_worker()
    return None


def execute_master_decode(
    plan,
    *,
    source_config,
    endpoint,
    prompt_for_length,
    server_log_path,
    profiler_directory,
    transport=None,
):
    """Execute and finalize a run while the caller owns server lifetime."""
    root = Path(plan.result_directory)
    run_id = os.environ.get("PERF_TEST_RUN_ID", root.name)
    initialize_run_directory(
        root,
        source_config=source_config,
        plan=plan,
        run_id=run_id,
        code_version=os.environ.get("BUILD_SCM_REVISION", "unknown"),
    )
    store = RunStore(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["state"] = "running"
    store.write_manifest(manifest)
    try:
        resolved = thaw(plan.resolved_config)
        topology = resolved["model"]["topology"]
        if transport is None:
            transport = DecodeHttpTransport(
                endpoint=endpoint, prompt_for_length=prompt_for_length
            )
        batch_executor = DecodeBatchExecutor(
            transport,
            engine=EngineConfig(**resolved["engine"]),
            dp_size=int(topology["dp"]),
        )
        profiler = DecodeProfiler(
            batch_executor,
            run_root=root,
            log_path=server_log_path,
            trace_directory=Path(profiler_directory),
        )
        case_executor = DecodeCaseExecutor(
            batch_executor, topology=topology, profiler=profiler
        )
        result = DecodeAdapter(lambda _context: case_executor)(
            ModeExecutionContext(plan=plan, session=None, store=store)
        )
        finalize_decode_run(
            root, plan=plan, run_id=run_id, case_results=result.case_results
        )
    except BaseException:
        manifest["state"] = "failed"
        store.write_manifest(manifest)
        raise
    manifest["state"] = result.state.value
    store.write_manifest(manifest)
    return result
