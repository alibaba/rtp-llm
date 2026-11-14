"""Strict collection and merge of untrusted AOne shard result artifacts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rtp_llm.test.perf_test.aone.sharding import AOneShardPlan
from rtp_llm.test.perf_test.errors import (
    CollectionError,
    PartialRunError,
    PerfTestError,
)
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.state import RunOutcome, RunState


@dataclass(frozen=True)
class CollectionOutcome:
    state: RunState
    cases: Tuple[Dict[str, Any], ...]
    missing_shard_ids: Tuple[str, ...] = ()
    failure: Optional[PerfTestError] = None
    diagnostic_cases: Tuple[Dict[str, Any], ...] = ()
    rejected_shards: Mapping[str, str] = field(default_factory=dict)
    shard_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return RunOutcome(state=self.state, failure=self.failure).exit_code


def _integrity_failure(message: str) -> CollectionOutcome:
    return CollectionOutcome(
        state=RunState.FAILED,
        cases=(),
        failure=CollectionError(message),
    )


def _string_list(path: str, value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise CollectionError(f"{path} must be an array of non-empty strings")
    parsed = tuple(value)
    if len(set(parsed)) != len(parsed):
        raise CollectionError(f"{path} contains duplicates")
    return parsed


def _validate_shard_plan(plan: ExecutionPlan, shard_plan: AOneShardPlan) -> None:
    if shard_plan.config_fingerprint != plan.config_fingerprint:
        raise CollectionError("shard plan config fingerprint mismatch")
    if shard_plan.workload_fingerprint != plan.workload_fingerprint:
        raise CollectionError("shard plan workload fingerprint mismatch")
    shard_ids = [shard.shard_id for shard in shard_plan.shards]
    if len(set(shard_ids)) != len(shard_ids):
        raise CollectionError("shard plan contains duplicate shard IDs")
    owned = [case_id for shard in shard_plan.shards for case_id in shard.case_ids]
    if len(set(owned)) != len(owned):
        raise CollectionError("shard plan case ownership overlaps")
    planned = {case.case_id for case in plan.cases}
    if set(owned) != planned:
        raise CollectionError("shard plan does not cover all planned cases")


def _validate_result(
    *,
    result: Mapping[str, Any],
    expected_run_id: str,
    plan: ExecutionPlan,
    expected_owned: Tuple[str, ...],
    planned_cases: Mapping[str, Any],
) -> Tuple[Tuple[Dict[str, Any], ...], bool]:
    if result.get("schema_version") != 1 or result.get("mode") != "prefill":
        raise CollectionError("unsupported AOne shard result schema or mode")
    if result.get("run_id") != expected_run_id:
        raise CollectionError("AOne shard result run identity mismatch")
    if result.get("config_fingerprint") != plan.config_fingerprint:
        raise CollectionError("AOne shard result config fingerprint mismatch")
    if result.get("workload_fingerprint") != plan.workload_fingerprint:
        raise CollectionError("AOne shard result workload fingerprint mismatch")
    owned = _string_list("owned_case_ids", result.get("owned_case_ids"))
    if owned != expected_owned:
        raise CollectionError("AOne shard result ownership mismatch")
    completed = _string_list("completed_case_ids", result.get("completed_case_ids"))
    if not set(completed).issubset(expected_owned):
        raise CollectionError("AOne shard completed an out-of-scope case")
    raw_cases = result.get("cases")
    if not isinstance(raw_cases, list):
        raise CollectionError("AOne shard cases must be an array")
    case_ids = []
    cases = []
    for index, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            raise CollectionError(f"AOne shard cases[{index}] must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise CollectionError(f"AOne shard cases[{index}] has invalid case_id")
        if case_id not in expected_owned:
            raise CollectionError(f"AOne shard returned out-of-scope case {case_id}")
        expected = planned_cases.get(case_id)
        if expected is None:
            raise CollectionError(f"AOne shard returned unknown case {case_id}")
        if case.get("identity") != thaw(expected.identity):
            raise CollectionError(f"AOne shard case identity mismatch: {case_id}")
        case_ids.append(case_id)
        cases.append(dict(case))
    if len(set(case_ids)) != len(case_ids):
        raise CollectionError("AOne shard result contains duplicate cases")
    if tuple(case_ids) != completed:
        raise CollectionError("AOne shard completion claims do not match case results")

    status = result.get("status")
    if not isinstance(status, str) or status not in {
        "completed",
        "partial",
        "failed",
        "interrupted",
    }:
        raise CollectionError("AOne shard result has invalid status")
    if status == "completed":
        if completed != expected_owned:
            raise CollectionError("completed AOne shard is missing owned cases")
        if any(case.get("status") != "ok" for case in cases):
            raise CollectionError("completed AOne shard contains failed cases")
        return tuple(cases), False
    return tuple(cases), True


def _validated_diagnostics(value: Any) -> Dict[str, Any]:
    """Keep optional diagnostics from invalidating otherwise verified case data."""
    try:
        if not isinstance(value, Mapping):
            raise ValueError("expected an object")
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        return {"collection_errors": [f"invalid engine_diagnostics: {error}"]}
    return copy.deepcopy(dict(value))


def collect_aone_results(
    plan: ExecutionPlan,
    *,
    run_id: str,
    shard_plan: AOneShardPlan,
    results: Sequence[Mapping[str, Any]],
) -> CollectionOutcome:
    """Validate all remote claims before accepting any merged formal result."""
    try:
        _validate_shard_plan(plan, shard_plan)
    except CollectionError as error:
        return _integrity_failure(str(error))

    expected = {shard.shard_id: shard.case_ids for shard in shard_plan.shards}
    provided: Dict[str, Mapping[str, Any]] = {}
    rejected: Dict[str, str] = {}
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            rejected[f"result[{index}]"] = "AOne result must be an object"
            continue
        shard_id = result.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id:
            rejected[f"result[{index}]"] = "AOne result has invalid shard_id"
            continue
        if shard_id not in expected:
            rejected[shard_id] = f"unknown AOne shard result: {shard_id}"
            continue
        if shard_id in provided or shard_id in rejected:
            # A duplicated identity makes every candidate for this shard ambiguous.
            provided.pop(shard_id, None)
            rejected[shard_id] = f"duplicate AOne shard result: {shard_id}"
            continue
        provided[shard_id] = result

    missing = tuple(
        shard_id
        for shard_id in expected
        if shard_id not in provided and shard_id not in rejected
    )
    planned_cases = {case.case_id: case for case in plan.cases}
    merged: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {}
    incomplete = bool(missing)
    for shard in shard_plan.shards:
        result = provided.get(shard.shard_id)
        if result is None:
            continue
        try:
            cases, shard_incomplete = _validate_result(
                result=result,
                expected_run_id=run_id,
                plan=plan,
                expected_owned=shard.case_ids,
                planned_cases=planned_cases,
            )
        except CollectionError as error:
            rejected[shard.shard_id] = str(error)
            continue
        incomplete = incomplete or shard_incomplete
        merged.extend(copy.deepcopy(cases))
        diagnostics[shard.shard_id] = _validated_diagnostics(
            result.get("engine_diagnostics", {})
        )

    order = {case.case_id: index for index, case in enumerate(plan.cases)}
    merged.sort(key=lambda case: order[case["case_id"]])
    if rejected:
        return CollectionOutcome(
            state=RunState.FAILED,
            cases=(),
            missing_shard_ids=missing,
            failure=CollectionError(
                "; ".join(f"{shard}: {reason}" for shard, reason in rejected.items())
            ),
            diagnostic_cases=tuple(merged),
            rejected_shards=rejected,
            shard_diagnostics=diagnostics,
        )
    if incomplete:
        if not merged:
            return CollectionOutcome(
                state=RunState.FAILED,
                cases=(),
                missing_shard_ids=missing,
                failure=CollectionError(
                    "AOne collection produced no acceptable case results"
                ),
            )
        return CollectionOutcome(
            state=RunState.PARTIAL,
            cases=tuple(merged),
            missing_shard_ids=missing,
            failure=PartialRunError("AOne shards did not complete the full plan"),
        )
    return CollectionOutcome(state=RunState.COMPLETED, cases=tuple(merged))
