"""Local execution of one immutable plan through one engine session."""

from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCaseExecutor,
)
from rtp_llm.test.perf_test.decode.finalize import finalize_decode_run
from rtp_llm.test.perf_test.decode.profiling import DecodeProfiler
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.errors import (
    EngineLifecycleError,
    PerfTestError,
    PlanningError,
)
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineDiagnostics,
    EngineInterrupted,
    EngineSession,
)
from rtp_llm.test.perf_test.execution.runtime import create_maga_handle
from rtp_llm.test.perf_test.modes import (
    DecodeAdapter,
    ModeExecutionContext,
    ModeExecutionResult,
    PrefillAdapter,
    create_prefill_adapter,
)
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.prefill.checkpoint import (
    PrefillResume,
    ResumeSelection,
    validate_prefill_resume,
)
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.prefill.profile import PrefillProfiler, ProfiledRunner
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock, inspect_run_directory
from rtp_llm.test.perf_test.state import RunOutcome, RunState
from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json


@dataclass(frozen=True)
class ExecutionOutcome:
    state: RunState
    case_results: Tuple[Mapping[str, Any], ...] = ()
    failure: Optional[PerfTestError] = None
    signal_number: Optional[int] = None
    engine_diagnostics: EngineDiagnostics = field(default_factory=EngineDiagnostics)

    @property
    def exit_code(self) -> int:
        return RunOutcome(
            state=self.state,
            failure=self.failure,
            signal_number=self.signal_number,
        ).exit_code


class LocalExecutor:
    def __init__(
        self,
        *,
        session_factory: Callable[[ExecutionPlan], Any],
        store_factory: Callable[[str], Any],
        adapters: Mapping[str, Callable[[ModeExecutionContext], ModeExecutionResult]],
    ):
        self._session_factory = session_factory
        self._store_factory = store_factory
        self._adapters = dict(adapters)

    def execute(self, plan: ExecutionPlan) -> ExecutionOutcome:
        if plan.backend != "local":
            raise PlanningError("LocalExecutor requires a local execution plan")
        adapter = self._adapters.get(plan.mode)
        if adapter is None:
            raise PlanningError(f"no local adapter registered for mode {plan.mode!r}")
        session = self._session_factory(plan)
        store = self._store_factory(plan.result_directory)
        try:
            with session as active_session:
                result = adapter(
                    ModeExecutionContext(
                        plan=plan,
                        session=active_session,
                        store=store,
                    )
                )
        except EngineInterrupted as error:
            return ExecutionOutcome(
                state=RunState.INTERRUPTED,
                signal_number=error.signal_number,
                engine_diagnostics=session.diagnostics,
            )
        except EngineLifecycleError as error:
            diagnostics = getattr(error, "diagnostics", session.diagnostics)
            return ExecutionOutcome(
                state=RunState.FAILED,
                failure=error,
                engine_diagnostics=diagnostics,
            )
        except PerfTestError as error:
            return ExecutionOutcome(
                state=RunState.FAILED,
                failure=error,
                engine_diagnostics=session.diagnostics,
            )
        return ExecutionOutcome(
            state=result.state,
            case_results=result.case_results,
            failure=result.failure,
            engine_diagnostics=session.diagnostics,
        )


SessionFactory = Callable[[ExecutionPlan], Any]
TokenizerLoader = Callable[[ExecutionPlan], Any]
QueryBuilder = Callable[[ExecutionPlan, Sequence[int]], Mapping[int, str]]
DecodeBatchExecutorFactory = Callable[..., DecodeBatchExecutor]


def _default_session_factory(plan: ExecutionPlan, **launch_options) -> EngineSession:
    return EngineSession(
        launcher=lambda: create_maga_handle(plan, **launch_options),
        readiness_timeout_seconds=3600,
        timeline_collector=lambda: (),
        coredump_collector=lambda: {},
    )


def _default_tokenizer_loader(plan: ExecutionPlan) -> Any:
    from rtp_llm.test.perf_test.test_util import _load_tokenizer

    model = thaw(plan.resolved_config)["model"]
    return _load_tokenizer(
        model["tokenizer"],
        checkpoint_path=model["checkpoint"],
        model_type=model["type"],
    )


def _default_query_builder(
    plan: ExecutionPlan, input_lengths: Sequence[int]
) -> Mapping[int, str]:
    from rtp_llm.test.perf_test.test_util import create_query

    model = thaw(plan.resolved_config)["model"]
    return create_query(model["tokenizer"], sorted(set(input_lengths)))


def _write_manifest_state(
    root: Path,
    state: str,
    *,
    error: Optional[BaseException] = None,
    signal_number: Optional[int] = None,
) -> Dict[str, Any]:
    snapshot = inspect_run_directory(root)
    manifest = copy.deepcopy(snapshot.manifest)
    manifest["state"] = state
    if error is not None:
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    elif state == "running":
        manifest.pop("error", None)
    if signal_number is not None:
        manifest["signal_number"] = signal_number
    elif state == "running":
        manifest.pop("signal_number", None)
    RunStore(root).write_manifest(manifest)
    return manifest


class UnifiedLocalExecutor:
    """Compose local actions without leaking config parsing into mode runners."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = _default_session_factory,
        tokenizer_loader: TokenizerLoader = _default_tokenizer_loader,
        query_builder: QueryBuilder = _default_query_builder,
        decode_batch_executor_factory: DecodeBatchExecutorFactory = DecodeBatchExecutor,
        selected_case_ids: Sequence[str] = (),
    ):
        self._session_factory = session_factory
        self._tokenizer_loader = tokenizer_loader
        self._query_builder = query_builder
        self._decode_batch_executor_factory = decode_batch_executor_factory
        self._selected_case_ids = tuple(selected_case_ids)

    def _prefill_adapter(self, plan: ExecutionPlan) -> PrefillAdapter:
        return create_prefill_adapter(
            thaw(plan.resolved_config), self._tokenizer_loader(plan)
        )

    def _decode_adapter(self, plan: ExecutionPlan, stack: ExitStack) -> DecodeAdapter:
        config = thaw(plan.resolved_config)
        topology = config["model"]["topology"]
        input_lengths = sorted(
            {
                int(length)
                for planned_case in plan.cases
                for length in thaw(planned_case.identity)["input_lengths"]
            }
        )
        queries = self._query_builder(plan, input_lengths)

        def executor_factory(context):
            session = context.session
            transport = DecodeHttpTransport(
                endpoint=f"http://127.0.0.1:{session.client.port}",
                prompt_for_length=lambda length: queries[length],
            )
            stack.callback(transport.close)
            batch_executor = self._decode_batch_executor_factory(
                transport,
                engine=EngineConfig(**config["engine"]),
                dp_size=int(topology["dp"]),
            )
            profiler = DecodeProfiler(
                batch_executor,
                run_root=Path(plan.result_directory),
                log_path=session.log_path,
                trace_directory=Path(plan.result_directory).resolve() / "engine_traces",
                world_size=int(topology["tp"]),
            )
            return DecodeCaseExecutor(
                batch_executor,
                topology=topology,
                profiler=profiler,
            )

        return DecodeAdapter(executor_factory)

    def _pressure_epoch(self, plan, *, profiling):
        from rtp_llm.test.continuous_pressure.config import validate_profile_steps
        from rtp_llm.test.perf_test.prefill.continuous_pressure import (
            run_continuous_pressure,
        )

        validate_profile_steps(
            plan.resolved_config["workload"]["pressure"]["profile_steps"]
        )
        root = Path(plan.result_directory)
        epoch = uuid4().hex
        directory = root / ("profiles" if profiling else "epochs") / epoch
        capture = profiling or plan.resolved_config["workload"]["pressure"]["capture"]
        if capture:
            (directory / "traces").mkdir(parents=True)
        if not profiling:
            atomic_write_json(
                root / "results" / "pressure_epoch.json",
                {"epoch": f"epochs/{epoch}/pressure"},
            )
        options = {}
        if capture and self._session_factory is _default_session_factory:
            options.update(
                profile_directory=(directory / "traces").resolve(), nsys_session=epoch
            )
        outcome = LocalExecutor(
            session_factory=partial(self._session_factory, **options),
            store_factory=RunStore,
            adapters={
                "prefill": partial(
                    run_continuous_pressure,
                    tokenizer=self._tokenizer_loader(plan),
                    directory=directory,
                    capture=capture,
                    nsys_session=epoch,
                )
            },
        ).execute(plan)
        return directory, outcome

    def _execute_measurement(
        self, plan: ExecutionPlan, *, resume_selection: Optional[ResumeSelection] = None
    ):
        if plan.resolved_config["workload"].get("pressure"):
            return self._pressure_epoch(plan, profiling=False)[1]
        with ExitStack() as stack:
            if plan.mode == "prefill":
                adapter = self._prefill_adapter(plan)
                if plan.action == "resume":
                    adapter = partial(
                        PrefillResume(adapter).run_locked, selection=resume_selection
                    )
            else:
                adapter = self._decode_adapter(plan, stack)
            return LocalExecutor(
                session_factory=self._session_factory,
                store_factory=RunStore,
                adapters={plan.mode: adapter},
            ).execute(plan)

    def __call__(self, plan: ExecutionPlan):
        root = Path(plan.result_directory)
        if plan.action == "report":
            return generate_prefill_report(root)
        pressure = plan.resolved_config["workload"].get("pressure")
        if plan.action == "profile" and pressure:
            with RunDirectoryLock(root, action="pressure-profile"):
                directory, outcome = self._pressure_epoch(plan, profiling=True)
                result = {
                    "status": "ok" if outcome.state == RunState.COMPLETED else "failed",
                    "epoch": str(directory.relative_to(root)),
                    "cases": list(outcome.case_results),
                    "error": str(outcome.failure) if outcome.failure else None,
                    "exit_code": outcome.exit_code,
                }
                atomic_write_json(directory / "profile.json", result)
                return result
        if plan.action == "profile":
            tokenizer = self._tokenizer_loader(plan)
            selected = self._selected_case_ids or tuple(
                case.case_id for case in plan.cases
            )
            session_id = datetime.now(timezone.utc).strftime("profile-%Y%m%dT%H%M%S%fZ")
            with ExitStack() as stack:

                def runner_factory(trace_dir):
                    options = {}
                    if self._session_factory is _default_session_factory:
                        options = {
                            "profile_directory": trace_dir.resolve(),
                            "nsys_session": session_id,
                        }
                    session = stack.enter_context(
                        self._session_factory(plan, **options)
                    )
                    return ProfiledRunner(
                        session.client,
                        tokenizer,
                        trace_dir,
                        thaw(plan.resolved_config),
                        session_id,
                    )

                return PrefillProfiler(runner_factory).run(
                    root, selected, session_id=session_id
                )

        with RunDirectoryLock(root, action=f"{plan.mode}-{plan.action}"):
            snapshot = inspect_run_directory(root)
            selection = (
                validate_prefill_resume(plan, RunStore(root))
                if plan.action == "resume" and not pressure
                else None
            )
            _write_manifest_state(root, "running")
            try:
                outcome = self._execute_measurement(plan, resume_selection=selection)
                if outcome.state in {RunState.FAILED, RunState.INTERRUPTED}:
                    _write_manifest_state(
                        root,
                        outcome.state.value,
                        error=outcome.failure,
                        signal_number=outcome.signal_number,
                    )
                    return outcome
                if plan.mode == "prefill":
                    finalize_prefill_run(
                        root,
                        plan=plan,
                        run_id=snapshot.manifest["run_id"],
                        case_results=outcome.case_results,
                    )
                    generate_prefill_report(root, acquire_lock=False)
                else:
                    finalize_decode_run(
                        root,
                        plan=plan,
                        run_id=snapshot.manifest["run_id"],
                        case_results=outcome.case_results,
                    )
                _write_manifest_state(root, outcome.state.value)
                return outcome
            except BaseException as error:
                state = (
                    "interrupted"
                    if isinstance(error, (KeyboardInterrupt, SystemExit))
                    else "failed"
                )
                _write_manifest_state(root, state, error=error)
                raise


def create_default_local_executor(
    *, selected_case_ids: Sequence[str] = ()
) -> UnifiedLocalExecutor:
    return UnifiedLocalExecutor(selected_case_ids=selected_case_ids)
