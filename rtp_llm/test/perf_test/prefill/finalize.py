"""Finalize prefill measurements into the common result envelope."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION, parse_result_envelope
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore


def finalize_prefill_run(
    root: Union[str, Path],
    *,
    plan: ExecutionPlan,
    run_id: str,
    case_results: Sequence[Mapping[str, Any]],
    terminal_state: Optional[RunState] = None,
    missing_error: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    if (
        plan.mode != "prefill"
        or plan.action not in {"run", "resume"}
        or plan.backend not in {"local", "aone"}
    ):
        raise PlanningError("prefill finalization requires a prefill run plan")
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("prefill run_id must be a non-empty string")

    provided: Dict[str, Dict[str, Any]] = {}
    for result in case_results:
        if not isinstance(result, Mapping) or not isinstance(
            result.get("case_id"), str
        ):
            raise PlanningError("prefill case result must contain case_id")
        case_id = result["case_id"]
        if case_id in provided:
            raise PlanningError(f"duplicate prefill case result: {case_id}")
        provided[case_id] = copy.deepcopy(dict(result))
    planned_ids = {case.case_id for case in plan.cases}
    if not set(provided).issubset(planned_ids):
        unknown = sorted(set(provided) - planned_ids)[0]
        raise PlanningError(f"unknown prefill case result: {unknown}")
    if terminal_state is None and set(provided) != planned_ids:
        raise PlanningError(
            "prefill results must cover every planned case exactly once"
        )
    if terminal_state is not None and terminal_state not in {
        RunState.COMPLETED,
        RunState.PARTIAL,
        RunState.FAILED,
        RunState.INTERRUPTED,
    }:
        raise PlanningError("prefill finalization requires a terminal run state")
    error = dict(
        missing_error
        or {
            "code": "missing_result",
            "message": "case did not produce an acceptable result",
        }
    )
    if set(error) != {"code", "message"} or any(
        not isinstance(error[name], str) or not error[name]
        for name in ("code", "message")
    ):
        raise PlanningError("missing_error requires non-empty code and message")

    ordered = []
    for case in plan.cases:
        result = provided.get(case.case_id)
        if result is None:
            result = {
                "case_id": case.case_id,
                "identity": thaw(case.identity),
                "status": "request_failed",
                "runs": 0,
                "errors": [copy.deepcopy(error)],
                "metrics": None,
            }
        if result.get("identity") != thaw(case.identity):
            raise PlanningError(f"prefill case identity mismatch: {case.case_id}")
        ordered.append(result)
    successful = sum(result.get("status") == "ok" for result in ordered)
    derived_state = (
        RunState.COMPLETED if successful == len(ordered) else RunState.PARTIAL
    )
    state = terminal_state or derived_state
    if state == RunState.COMPLETED and derived_state != RunState.COMPLETED:
        raise PlanningError("completed prefill result requires every case to succeed")
    summary = {
        "planned_cases": len(ordered),
        "successful_cases": successful,
        "failed_cases": len(ordered) - successful,
    }
    resolved_model = plan.to_dict()["resolved_config"]["model"]
    envelope = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "prefill",
        "backend": plan.backend,
        "status": state.value,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "model": {
            name: copy.deepcopy(resolved_model[name])
            for name in ("label", "type", "checkpoint", "topology")
        },
        "summary": summary,
        "cases": ordered,
    }
    parse_result_envelope(envelope)

    store = RunStore(root)
    committed = store.read_prefill_cases(plan)
    for result in ordered:
        if committed.get(result["case_id"]) != result:
            store.write_prefill_case(result["case_id"], result)
    store.write_final_result(envelope)
    store.write_summary(
        {
            "schema_version": 1,
            "run_id": run_id,
            "mode": "prefill",
            "backend": plan.backend,
            "status": state.value,
            **summary,
        }
    )
    return envelope
