"""Isolated profiling of cases reconstructed from a persisted prefill plan."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Union

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.prefill.planner import require_concurrent_prefill_plan
from rtp_llm.test.perf_test.prefill.trace import capture_scope
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock
from rtp_llm.test.perf_test.storage import atomic_write_json

_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class PrefillProfiler:
    def __init__(self, runner_factory: Callable[[Path], Any]):
        self._runner_factory = runner_factory

    @staticmethod
    def _load_plan(root: Path) -> ExecutionPlan:
        try:
            value = json.loads(
                (root / "execution_plan.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise PlanningError(
                "profile requires a valid persisted execution plan"
            ) from error
        plan = ExecutionPlan.from_dict(value)
        if plan.mode != "prefill" or plan.backend != "local":
            raise PlanningError("profile requires a local prefill run")
        require_concurrent_prefill_plan(plan)
        return plan

    def run(
        self,
        root: Union[str, Path],
        case_ids: Sequence[str],
        *,
        session_id: str,
    ) -> Dict[str, Any]:
        root = Path(root)
        if not _SESSION_ID.fullmatch(session_id):
            raise PlanningError(f"invalid profile session_id: {session_id!r}")
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise PlanningError("profile case IDs must be non-empty and unique")
        plan = self._load_plan(root)
        by_id = {case.case_id: case for case in plan.cases}
        for case_id in case_ids:
            if case_id not in by_id:
                raise PlanningError(f"unknown profile case: {case_id}")

        with RunDirectoryLock(root, action="profile"):
            profile_dir = root / "profiles" / session_id
            try:
                profile_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError as error:
                raise PlanningError(
                    f"profile session already exists: {session_id}"
                ) from error
            cases_dir = profile_dir / "cases"
            cases_dir.mkdir()
            results = []
            manifest = {
                "schema_version": 1,
                "session_id": session_id,
                "diagnostic_only": True,
                "case_ids": list(case_ids),
                "profiling": thaw(plan.resolved_config)["workload"].get(
                    "profiling", {}
                ),
                "status": "running",
                "results": results,
            }
            manifest["capture_scope"] = capture_scope(
                manifest["profiling"].get("backend", "kineto")
            )
            atomic_write_json(profile_dir / "manifest.json", manifest)
            try:
                runner = self._runner_factory(profile_dir)
                for case_id in case_ids:
                    case = thaw(by_id[case_id].identity)
                    case["case_id"] = case_id
                    try:
                        result = runner.run_case(case)
                        if result.get("status") == "ok":
                            artifacts = result.get("trace_files") or []
                            if not artifacts:
                                raise RuntimeError(
                                    "profile produced no trace artifacts"
                                )
                            for artifact in artifacts:
                                path = (profile_dir / artifact).resolve()
                                if (
                                    not path.is_relative_to(profile_dir.resolve())
                                    or not path.is_file()
                                    or path.stat().st_size == 0
                                ):
                                    raise RuntimeError(
                                        f"invalid profile trace artifact: {artifact}"
                                    )
                    except Exception as error:
                        result = {
                            "case_id": case_id,
                            "status": "request_failed",
                            "metrics": None,
                            "errors": [
                                {"code": "profile_failed", "message": str(error)}
                            ],
                        }
                    atomic_write_json(cases_dir / f"{case_id}.json", result)
                    results.append(result)
                outcome = {
                    "schema_version": 1,
                    "session_id": session_id,
                    "run_id": plan.result_directory,
                    "case_ids": list(case_ids),
                    "status": (
                        "completed"
                        if all(result.get("status") == "ok" for result in results)
                        else "failed"
                    ),
                    "results": results,
                }
                manifest.update(status=outcome["status"], results=results)
                atomic_write_json(profile_dir / "manifest.json", manifest)
                atomic_write_json(profile_dir / "profile_result.json", outcome)
                return outcome
            except BaseException as error:
                manifest.update(
                    status=(
                        "interrupted"
                        if isinstance(error, (KeyboardInterrupt, SystemExit))
                        else "failed"
                    ),
                    error=str(error),
                )
                atomic_write_json(profile_dir / "manifest.json", manifest)
                raise
