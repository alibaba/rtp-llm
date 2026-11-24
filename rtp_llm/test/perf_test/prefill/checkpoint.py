"""Strict local-prefill resume identity and selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.modes import ModeExecutionContext, ModeExecutionResult
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.prefill.fingerprints import (
    RUNNER_CONTRACT,
    resume_fingerprints,
)
from rtp_llm.test.perf_test.prefill.planner import require_concurrent_prefill_plan
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock


@dataclass(frozen=True)
class ResumeSelection:
    rerun_case_ids: Tuple[str, ...]
    preserved_results: Tuple[Dict[str, Any], ...]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanningError(
            f"required resume {label} is missing or invalid: {path}"
        ) from error


def validate_prefill_resume(plan: ExecutionPlan, store: Any) -> ResumeSelection:
    if plan.mode != "prefill" or plan.backend != "local" or plan.action != "resume":
        raise PlanningError("prefill resume requires a local prefill resume plan")
    require_concurrent_prefill_plan(plan)
    root = Path(plan.result_directory)
    persisted = ExecutionPlan.from_dict(
        _read_json(root / "execution_plan.json", "execution plan")
    )
    if persisted.mode != "prefill" or persisted.backend != "local":
        raise PlanningError("persisted execution plan is not local prefill")
    require_concurrent_prefill_plan(persisted)
    expected = resume_fingerprints(plan)
    persisted_fingerprints = resume_fingerprints(persisted)
    for name in (
        "config_fingerprint",
        "workload_fingerprint",
        "model_fingerprint",
        "runner_fingerprint",
    ):
        if persisted_fingerprints[name] != expected[name]:
            raise PlanningError(f"resume {name} mismatch")
    if [case.to_dict() for case in persisted.cases] != [
        case.to_dict() for case in plan.cases
    ]:
        raise PlanningError("resume planned case identity mismatch")
    checkpoint = _read_json(
        root / "checkpoints" / "prefill_progress.json", "checkpoint"
    )
    if not isinstance(checkpoint, dict):
        raise PlanningError("resume checkpoint must be an object")
    if "engine_fingerprint" not in checkpoint:
        raise PlanningError(
            "resume checkpoint has no engine build identity; start a new run"
        )
    for name, value in expected.items():
        if checkpoint.get(name) != value:
            raise PlanningError(f"resume checkpoint {name} mismatch")
    if (root / "results" / "cases").exists():
        raise PlanningError(
            "legacy prefill results/cases layout is unsupported; start a new run"
        )
    committed = store.read_prefill_cases(plan)
    planned_ids = {case.case_id for case in plan.cases}
    unknown = set(committed) - planned_ids
    if unknown:
        raise PlanningError(f"resume found unknown case: {sorted(unknown)[0]}")
    preserved = []
    rerun = []
    for case in plan.cases:
        result = committed.get(case.case_id)
        if result is not None and result.get("status") == "ok":
            preserved.append(result)
        else:
            rerun.append(case.case_id)
    return ResumeSelection(tuple(rerun), tuple(preserved))


class PrefillResume:
    def __init__(self, adapter: Any):
        self._adapter = adapter

    def __call__(self, context: ModeExecutionContext) -> ModeExecutionResult:
        with RunDirectoryLock(context.plan.result_directory, action="resume"):
            return self.run_locked(context)

    def run_locked(
        self, context: ModeExecutionContext, *, selection: ResumeSelection | None = None
    ) -> ModeExecutionResult:
        """Resume while the caller owns the run directory's writer lock."""
        if selection is None:
            selection = validate_prefill_resume(context.plan, context.store)
        return self._adapter.run(
            context,
            selected_case_ids=selection.rerun_case_ids,
            existing_results=selection.preserved_results,
        )
