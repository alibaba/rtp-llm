"""AOne build, submit, collect, and offline-report orchestration."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from rtp_llm.test.perf_test.aone.collector import (
    CollectionOutcome,
    collect_aone_results,
)
from rtp_llm.test.perf_test.aone.package import (
    GeneratedAOnePackage,
    generate_aone_package,
)
from rtp_llm.test.perf_test.aone.sharding import plan_shards
from rtp_llm.test.perf_test.errors import PerfTestError, PostExecutionError, ReportError
from rtp_llm.test.perf_test.plan import AOneResourcePlan, ExecutionPlan
from rtp_llm.test.perf_test.prefill.finalize import finalize_prefill_run
from rtp_llm.test.perf_test.prefill.report import generate_prefill_report
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock, inspect_run_directory
from rtp_llm.test.perf_test.state import RunOutcome, RunState
from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class AOneExecutionOutcome:
    state: RunState
    failure: Optional[PerfTestError] = None
    collection: Optional[CollectionOutcome] = None
    report: Optional[Mapping[str, Any]] = None
    package: Optional[GeneratedAOnePackage] = None

    @property
    def exit_code(self) -> int:
        return RunOutcome(state=self.state, failure=self.failure).exit_code


class AOneExecutor:
    def __init__(
        self,
        *,
        workspace_root: Union[str, Path],
        builder: Callable[[GeneratedAOnePackage], Any],
        submitter: Callable[[str, str], Any],
        result_collector: Callable[[Any, str], Optional[Mapping[str, Any]]],
        reporter: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    ):
        self._workspace_root = Path(workspace_root).resolve()
        self._builder = builder
        self._submitter = submitter
        self._result_collector = result_collector
        self._reporter = reporter or (
            lambda root: generate_prefill_report(root, acquire_lock=False)
        )

    @staticmethod
    def _write_manifest(
        store: RunStore,
        manifest: Dict[str, Any],
        *,
        state: Optional[RunState] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if state is not None:
            manifest["state"] = state.value
        if stage is not None:
            stages = manifest.setdefault("stages", {})
            stage_value = {"status": status, "updated_at": _timestamp()}
            if details:
                stage_value.update(copy.deepcopy(dict(details)))
            stages[stage] = stage_value
        manifest["updated_at"] = _timestamp()
        store.write_manifest(manifest)

    @staticmethod
    def _write_collection_diagnostics(
        root: Path,
        run_id: str,
        collection: CollectionOutcome,
        errors: Dict[str, str],
    ) -> None:
        if not collection.rejected_shards:
            return
        # This optional artifact is best effort; formal result writes and execution
        # errors remain outside this guard so terminal finalization still runs.
        try:
            atomic_write_json(
                root / "results" / "aone_diagnostics.json",
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "mode": "prefill",
                    "status": "failed",
                    "cases": list(collection.diagnostic_cases),
                    "rejected_shards": dict(collection.rejected_shards),
                    "shard_diagnostics": dict(collection.shard_diagnostics),
                },
            )
        except Exception as error:
            errors["diagnostics"] = (
                f"AOne diagnostic artifact write failed: {type(error).__name__}: {error}"
            )

    def _package_location(self, root: Path) -> tuple[Path, str]:
        package_root = root / "workload" / "aone_package"
        try:
            relative = package_root.resolve().relative_to(self._workspace_root)
        except ValueError as error:
            raise PostExecutionError(
                f"AOne run directory must be inside workspace {self._workspace_root}"
            ) from error
        return package_root, relative.as_posix()

    def _stage_failure(
        self,
        *,
        store: RunStore,
        manifest: Dict[str, Any],
        stage: str,
        error: Exception,
        package: Optional[GeneratedAOnePackage] = None,
    ) -> AOneExecutionOutcome:
        failure = PostExecutionError(f"AOne {stage} failed: {error}")
        self._write_manifest(
            store,
            manifest,
            state=RunState.FAILED,
            stage=stage,
            status="failed",
            details={"error": str(error)},
        )
        return AOneExecutionOutcome(
            state=RunState.FAILED, failure=failure, package=package
        )

    def execute(self, plan: ExecutionPlan) -> AOneExecutionOutcome:
        if (
            plan.mode != "prefill"
            or plan.action != "run"
            or not isinstance(plan.resource_plan, AOneResourcePlan)
            or plan.resource_plan.machine_count != 1
        ):
            raise ValueError("AOneExecutor requires an AOne prefill run plan")
        root = Path(plan.result_directory)
        with RunDirectoryLock(root, action="aone-run"):
            snapshot = inspect_run_directory(root)
            if snapshot.execution_plan != plan:
                raise ValueError("persisted execution plan does not match AOne plan")
            manifest = snapshot.manifest
            run_id = manifest["run_id"]
            store = RunStore(root)
            shard_plan = plan_shards(plan)
            self._write_manifest(
                store,
                manifest,
                state=RunState.RUNNING,
                stage="build",
                status="running",
            )
            package: Optional[GeneratedAOnePackage] = None
            try:
                package_root, package_name = self._package_location(root)
                package = generate_aone_package(
                    package_root,
                    plan,
                    run_id=run_id,
                    ownership=shard_plan.ownership,
                    package=package_name,
                )
                self._builder(package)
            except Exception as error:
                return self._stage_failure(
                    store=store,
                    manifest=manifest,
                    stage="build",
                    error=error,
                    package=package,
                )
            self._write_manifest(
                store,
                manifest,
                stage="build",
                status="completed",
                details={"targets": list(package.targets)},
            )

            self._write_manifest(store, manifest, stage="submit", status="running")
            jobs: Dict[str, Any] = {}
            submission_errors: Dict[str, str] = {}
            for target, shard in zip(package.targets, shard_plan.shards):
                try:
                    jobs[shard.shard_id] = self._submitter(target, shard.shard_id)
                except Exception as error:
                    submission_errors[shard.shard_id] = str(error)
            self._write_manifest(
                store,
                manifest,
                stage="submit",
                status="partial" if submission_errors else "completed",
                details={
                    "submitted_shards": sorted(jobs),
                    "errors": submission_errors,
                },
            )

            self._write_manifest(
                store,
                manifest,
                state=RunState.COLLECTING,
                stage="collect",
                status="running",
            )
            remote_results = []
            collection_errors: Dict[str, str] = {}
            for shard_id, job in jobs.items():
                try:
                    result = self._result_collector(job, shard_id)
                    if result is not None:
                        remote_results.append(result)
                        remote_failure = result.get("failure")
                        if (
                            result.get("status") != "completed"
                            and isinstance(remote_failure, Mapping)
                            and isinstance(remote_failure.get("message"), str)
                        ):
                            collection_errors[shard_id] = remote_failure["message"]
                except KeyboardInterrupt:
                    interrupted = collect_aone_results(
                        plan,
                        run_id=run_id,
                        shard_plan=shard_plan,
                        results=remote_results,
                    )
                    self._write_collection_diagnostics(
                        root, run_id, interrupted, collection_errors
                    )
                    collection_errors.update(interrupted.rejected_shards)
                    self._write_manifest(
                        store,
                        manifest,
                        state=RunState.INTERRUPTED,
                        stage="collect",
                        status="interrupted",
                        details={
                            "missing_shards": list(interrupted.missing_shard_ids),
                            "errors": collection_errors,
                        },
                    )
                    finalize_prefill_run(
                        root,
                        plan=plan,
                        run_id=run_id,
                        case_results=interrupted.cases,
                        terminal_state=RunState.INTERRUPTED,
                        missing_error={
                            "code": "interrupted",
                            "message": "AOne collection was interrupted",
                        },
                    )
                    store.write_summary(
                        self._summary(
                            run_id,
                            plan,
                            interrupted,
                            RunState.INTERRUPTED,
                            None,
                        )
                    )
                    raise
                except Exception as error:
                    collection_errors[shard_id] = str(error)
            collection = collect_aone_results(
                plan,
                run_id=run_id,
                shard_plan=shard_plan,
                results=remote_results,
            )
            self._write_collection_diagnostics(
                root, run_id, collection, collection_errors
            )
            collection_errors.update(collection.rejected_shards)
            for case in collection.cases:
                store.write_prefill_case(case["case_id"], case)
            self._write_manifest(
                store,
                manifest,
                stage="collect",
                status=collection.state.value,
                details={
                    "exit_code": collection.exit_code,
                    "missing_shards": list(collection.missing_shard_ids),
                    "errors": collection_errors,
                },
            )

            self._write_manifest(
                store,
                manifest,
                state=RunState.REPORTING,
                stage="report",
                status="running",
            )
            report: Optional[Mapping[str, Any]] = None
            final_state = collection.state
            final_failure = collection.failure
            try:
                report = self._reporter(root)
            except Exception as error:
                final_failure = ReportError(f"AOne offline report failed: {error}")
                final_state = RunState.FAILED
                self._write_manifest(
                    store,
                    manifest,
                    stage="report",
                    status="failed",
                    details={"error": str(error)},
                )
            else:
                self._write_manifest(
                    store,
                    manifest,
                    stage="report",
                    status="completed",
                    details={"report_kind": report.get("report_kind")},
                )

            try:
                finalize_prefill_run(
                    root,
                    plan=plan,
                    run_id=run_id,
                    case_results=collection.cases,
                    terminal_state=final_state,
                    missing_error={
                        "code": "collection_incomplete",
                        "message": str(
                            collection.failure
                            or "AOne collection did not return the planned case"
                        ),
                    },
                )
            except Exception as error:
                return self._stage_failure(
                    store=store,
                    manifest=manifest,
                    stage="report",
                    error=error,
                    package=package,
                )

            self._write_manifest(store, manifest, state=final_state)
            store.write_summary(
                self._summary(run_id, plan, collection, final_state, report)
            )
            return AOneExecutionOutcome(
                state=final_state,
                failure=final_failure,
                collection=collection,
                report=report,
                package=package,
            )

    @staticmethod
    def _summary(
        run_id: str,
        plan: ExecutionPlan,
        collection: CollectionOutcome,
        state: RunState,
        report: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        successful = sum(case.get("status") == "ok" for case in collection.cases)
        return {
            "schema_version": 1,
            "run_id": run_id,
            "mode": "prefill",
            "backend": "aone",
            "status": state.value,
            "planned_cases": len(plan.cases),
            "successful_cases": successful,
            "failed_cases": len(plan.cases) - successful,
            "collection_exit_code": collection.exit_code,
            "report_kind": None if report is None else report.get("report_kind"),
        }
