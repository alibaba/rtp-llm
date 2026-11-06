"""Narrow contracts shared by prefill and decode mode adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    EngineLifecycleError,
    PerfTestError,
)
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.state import RunState


@runtime_checkable
class _EngineHealth(Protocol):
    def assert_healthy(self) -> None: ...


def _assert_engine_healthy(session: Any) -> None:
    if isinstance(session, _EngineHealth):
        session.assert_healthy()


@dataclass(frozen=True)
class ModeExecutionContext:
    plan: ExecutionPlan
    session: Any
    store: Any


@dataclass(frozen=True)
class ModeExecutionResult:
    state: RunState
    case_results: Tuple[Mapping[str, Any], ...]
    failure: Optional[PerfTestError] = None

    def __post_init__(self) -> None:
        if self.state not in {
            RunState.COMPLETED,
            RunState.PARTIAL,
            RunState.FAILED,
        }:
            raise ValueError("mode execution result requires a terminal state")


class DecodeAdapter:
    def __init__(self, runner_factory: Callable[[ModeExecutionContext], Any]):
        self._runner_factory = runner_factory

    @staticmethod
    def _request_failure(case: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "identity": {key: value for key, value in case.items() if key != "case_id"},
            "status": "request_failed",
            "runs": int(case.get("measure_runs", 0)),
            "errors": [{"code": "request_failed", "message": str(error)}],
            "metrics": None,
        }

    def __call__(self, context: ModeExecutionContext) -> ModeExecutionResult:
        if context.plan.mode != "decode":
            raise ValueError("DecodeAdapter requires a decode execution plan")
        executor = self._runner_factory(context)
        results: List[Dict[str, Any]] = []
        statuses: Dict[str, str] = {}
        session_reusable = True
        for planned_case in context.plan.cases:
            case = thaw(planned_case.identity)
            case["case_id"] = planned_case.case_id
            try:
                if not session_reusable:
                    raise RuntimeError(
                        "decode case not executed: profiling session requires engine "
                        "shutdown after incomplete capture"
                    )
                _assert_engine_healthy(context.session)
                result = executor.execute(case)
                _assert_engine_healthy(context.session)
            except EngineLifecycleError:
                raise
            except Exception as error:
                _assert_engine_healthy(context.session)
                result = self._request_failure(case, error)
            session_reusable = (
                session_reusable
                and getattr(executor, "session_reusable", True) is not False
            )
            context.store.write_case(planned_case.case_id, result)
            results.append(result)
            statuses[planned_case.case_id] = result["status"]
            context.store.write_checkpoint(
                "decode_progress",
                {
                    "schema_version": 1,
                    "config_fingerprint": context.plan.config_fingerprint,
                    "workload_fingerprint": context.plan.workload_fingerprint,
                    "completed_case_ids": [item["case_id"] for item in results],
                    "case_statuses": statuses,
                },
            )
        failed = [result for result in results if result["status"] != "ok"]
        if failed or not session_reusable:
            message = f"{len(failed)} of {len(results)} decode cases failed"
            if not session_reusable:
                message += (
                    "; decode profiling capture did not complete; "
                    "the engine session requires shutdown"
                )
            return ModeExecutionResult(
                state=RunState.PARTIAL,
                case_results=tuple(results),
                failure=CaseExecutionError(message),
            )
        return ModeExecutionResult(
            state=RunState.COMPLETED, case_results=tuple(results)
        )


class PrefillAdapter:
    def __init__(
        self, runner_factory: Callable[[ModeExecutionContext], Any], *, probe=None
    ):
        self._runner_factory = runner_factory
        self._probe = probe

    @staticmethod
    def _request_failure(case: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "identity": {key: value for key, value in case.items() if key != "case_id"},
            "status": "request_failed",
            "runs": int(case.get("measure_runs", 0)),
            "errors": [{"code": "request_failed", "message": str(error)}],
            "metrics": None,
        }

    def run(
        self,
        context: ModeExecutionContext,
        *,
        selected_case_ids=None,
        existing_results=(),
    ) -> ModeExecutionResult:
        if context.plan.mode != "prefill":
            raise ValueError("PrefillAdapter requires a prefill execution plan")
        runner = self._runner_factory(context)
        results: List[Dict[str, Any]] = list(existing_results)
        selected = (
            {case.case_id for case in context.plan.cases}
            if selected_case_ids is None
            else set(selected_case_ids)
        )
        if selected and hasattr(runner, "preflight"):
            _assert_engine_healthy(context.session)
            cases = []
            for planned in context.plan.cases:
                if planned.case_id in selected:
                    case = thaw(planned.identity)
                    case["case_id"] = planned.case_id
                    cases.append(case)
            try:
                runner.preflight(cases)
            except Exception:
                _assert_engine_healthy(context.session)
                raise
            _assert_engine_healthy(context.session)
        if selected and self._probe is not None:
            try:
                probe_result = self._probe(context)
            except Exception as error:
                context.store.write_checkpoint(
                    "prefill_cache_probe",
                    {"schema_version": 1, "status": "failed", "error": str(error)},
                )
                _assert_engine_healthy(context.session)
                raise
            _assert_engine_healthy(context.session)
            context.store.write_checkpoint("prefill_cache_probe", probe_result)
        for planned_case in context.plan.cases:
            if planned_case.case_id not in selected:
                continue
            case = thaw(planned_case.identity)
            case["case_id"] = planned_case.case_id
            try:
                _assert_engine_healthy(context.session)
                result = runner.run_case(case)
                _assert_engine_healthy(context.session)
            except EngineLifecycleError:
                raise
            except Exception as error:
                _assert_engine_healthy(context.session)
                result = self._request_failure(case, error)
            context.store.write_prefill_case(planned_case.case_id, result)
            results.append(result)
        failed = [result for result in results if result["status"] != "ok"]
        if failed:
            return ModeExecutionResult(
                state=RunState.PARTIAL,
                case_results=tuple(results),
                failure=CaseExecutionError(
                    f"{len(failed)} of {len(results)} prefill cases failed validation"
                ),
            )
        return ModeExecutionResult(
            state=RunState.COMPLETED,
            case_results=tuple(results),
        )

    def __call__(self, context: ModeExecutionContext) -> ModeExecutionResult:
        return self.run(context)
