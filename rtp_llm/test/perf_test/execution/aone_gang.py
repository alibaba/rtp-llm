"""Gang-aware AOne execution: a staged broker lifecycle across machines.

The defined submit environment is a CI control host reachable by workers.
Local submission requires separate worker-to-broker connectivity validation.

The gang executor owns the submit side of a multi-machine AOne run. It
starts the rendezvous broker on a routable host, generates the gang Bazel
package, and drives the remote build/test invocations. The broker URL and run
ID are declared in the gang plan; only the token reaches remote shards through
the generated rule's ``env_inherit``. It collects the master shard's full
``run/`` tree plus non-master logs from the testlogs tree, and produces the
standard envelope report.

Stage order is rendezvous -> build -> submit -> collect -> report, every
transition persists to the manifest, partial collection always runs, and
the original test failure exit code wins over collect/report errors. The
broker token is only ever present in the bazel test invocation's
environment; it never reaches a command line (where ps and command logs
could capture it), nor the manifest, the run directory, or any log line.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from rtp_llm.test.perf_test.aone.broker import GangBroker, run_gang_broker
from rtp_llm.test.perf_test.aone.collector import CollectionOutcome
from rtp_llm.test.perf_test.aone.package import (
    GeneratedAOnePackage,
    generate_gang_package,
    package_location,
)
from rtp_llm.test.perf_test.aone.plan import (
    AONE_GPU_HOSTS,
    AONE_GPU_PLATFORMS,
    AONE_RESOURCES,
)
from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    CollectionError,
    PerfTestError,
    PostExecutionError,
    ReportError,
)
from rtp_llm.test.perf_test.execution.aone import AOneExecutionOutcome, bazel_argv
from rtp_llm.test.perf_test.execution.aone_bazel import (
    _MAX_ARCHIVE_MEMBERS,
    _positive_integer,
    _safe_zip_member,
    _workspace_root,
)
from rtp_llm.test.perf_test.execution.artifacts import (
    collect_master_artifacts,
    copy_member_traces,
)
from rtp_llm.test.perf_test.execution.gang import (
    reconcile_terminal_state,
    record_stage,
    write_manifest,
)
from rtp_llm.test.perf_test.plan import AOneResourcePlan, ExecutionPlan
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock, inspect_run_directory
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json

BROKER_TOKEN_ENV = "PERF_TEST_BROKER_TOKEN"


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "<redacted>") if secret else text


def _package_path(target: str) -> Path:
    """Map ``//package:name`` to its testlogs directory (``package/name``)."""
    invalid = f"invalid generated gang target: {target!r}"
    if not target.startswith("//") or ":" not in target:
        raise ValueError(invalid)
    package, _, name = target[2:].partition(":")
    if not package or not name or "/" in name:
        raise ValueError(invalid)
    return Path(package) / name


class GangAOneExecutor:
    """Execute one multi-machine AOne plan through a broker-coordinated gang."""

    def __init__(
        self,
        *,
        workspace_root: Union[str, Path],
        resource: str,
        gpu_type: str,
        max_parallel: int,
        build_jobs: Optional[int] = None,
        startup_args: Sequence[str] = (),
        common_args: Sequence[str] = (),
        testlogs_root: Optional[Union[str, Path]] = None,
        process_env: Optional[Mapping[str, str]] = None,
        run_command: Callable[..., Any] = subprocess.run,
        broker_host: Optional[str] = None,
        reporter: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    ):
        if resource not in AONE_RESOURCES:
            supported = ", ".join(AONE_RESOURCES)
            raise ValueError(
                f"unsupported AOne resource {resource!r}; "
                f"supported resources: {supported}"
            )
        if gpu_type not in AONE_GPU_HOSTS:
            supported = ", ".join(sorted(AONE_GPU_HOSTS))
            raise ValueError(
                f"unsupported gang GPU type {gpu_type!r}; gang shards request "
                f"full hosts, supported GPU types: {supported}"
            )
        self._workspace_root = Path(workspace_root).resolve()
        self._resource = resource
        self._platform = AONE_GPU_PLATFORMS[gpu_type]
        self._max_parallel = _positive_integer("max_parallel", max_parallel)
        self._bazel = "bazelisk"
        self._startup_args = tuple(startup_args)
        self._build_jobs = _positive_integer(
            "build_jobs", build_jobs if build_jobs is not None else os.cpu_count() or 1
        )
        self._common_args = tuple(common_args)
        self._testlogs_root = Path(
            testlogs_root or self._workspace_root / "bazel-testlogs"
        )
        self._process_env = dict(os.environ if process_env is None else process_env)
        self._run_command = run_command
        self._broker_host = broker_host
        self._reporter_override = reporter

    _write_manifest = staticmethod(write_manifest)

    def _package_location(self, root: Path) -> tuple[Path, str]:
        return package_location(self._workspace_root, root)

    # Public entry ---------------------------------------------------------

    def _write_stage(
        self,
        store: RunStore,
        manifest: Dict[str, Any],
        *,
        stage: str,
        status: str,
        state: Optional[RunState] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        record_stage(
            store, manifest, stage=stage, status=status, state=state, details=details
        )

    def _stage_failure(
        self,
        *,
        store: RunStore,
        manifest: Dict[str, Any],
        stage: str,
        error: Exception,
        package: Optional[GeneratedAOnePackage] = None,
    ) -> AOneExecutionOutcome:
        failure = PostExecutionError(f"AOne gang {stage} failed: {error}")
        self._write_stage(
            store, manifest, stage=stage, status="failed", details={"error": str(error)}
        )
        self._write_manifest(store, manifest, state=RunState.FAILED)
        return AOneExecutionOutcome(
            state=RunState.FAILED, failure=failure, package=package
        )

    def execute(self, plan: ExecutionPlan) -> AOneExecutionOutcome:
        if (
            plan.action != "run"
            or not isinstance(plan.resource_plan, AOneResourcePlan)
            or plan.resource_plan.machine_count <= 1
        ):
            raise ValueError("GangAOneExecutor requires a multi-machine AOne run plan")
        root = Path(plan.result_directory)
        with RunDirectoryLock(root, action="aone-gang-run"):
            snapshot = inspect_run_directory(root)
            if snapshot.execution_plan != plan:
                raise ValueError("persisted execution plan does not match gang plan")
            manifest = snapshot.manifest
            run_id = manifest["run_id"]
            store = RunStore(root)
            resource = plan.resource_plan
            manifest["gang"] = {
                "machine_count": resource.machine_count,
                "local_world_size": resource.local_world_size,
                "gpu_type": resource.gpu_type,
            }
            self._write_stage(
                store,
                manifest,
                state=RunState.RUNNING,
                stage="rendezvous",
                status="running",
            )
            broker = None
            try:
                broker = run_gang_broker(
                    resource.machine_count,
                    state_dir=root / "workload" / "gang_broker",
                    host=self._broker_host or socket.getfqdn(),
                )
                broker.start()
            except Exception as error:
                if broker is not None:
                    self._quiet(broker.stop)
                return self._stage_failure(
                    store=store, manifest=manifest, stage="rendezvous", error=error
                )
            try:
                manifest["gang"]["broker_url"] = broker.url
                manifest["gang"]["broker_run_id"] = broker.run_id
                self._write_stage(
                    store,
                    manifest,
                    stage="rendezvous",
                    status="completed",
                    details={
                        "broker_url": broker.url,
                        "broker_run_id": broker.run_id,
                    },
                )
                try:
                    return self._execute_with_broker(
                        plan, root, store, manifest, run_id, broker
                    )
                except KeyboardInterrupt:
                    self._write_manifest(store, manifest, state=RunState.INTERRUPTED)
                    raise
            finally:
                self._quiet(broker.cleanup)
                self._quiet(broker.stop)

    # Stages ---------------------------------------------------------------

    def _execute_with_broker(
        self,
        plan: ExecutionPlan,
        root: Path,
        store: RunStore,
        manifest: Dict[str, Any],
        run_id: str,
        broker: GangBroker,
    ) -> AOneExecutionOutcome:
        built = self._run_build_stage(plan, root, store, manifest, run_id, broker)
        if isinstance(built, AOneExecutionOutcome):
            return built
        test_failure = self._run_submit_stage(plan, store, manifest, built, broker)
        collection, collect_error = self._run_collect_stage(
            root, store, manifest, broker, built, plan.resource_plan.machine_count
        )
        return self._run_report_stage(
            plan, root, store, manifest, built, test_failure, collection, collect_error
        )

    def _run_build_stage(
        self,
        plan: ExecutionPlan,
        root: Path,
        store: RunStore,
        manifest: Dict[str, Any],
        run_id: str,
        broker: GangBroker,
    ):
        self._write_stage(store, manifest, stage="build", status="running")
        package: Optional[GeneratedAOnePackage] = None
        try:
            package_root, package_name = self._package_location(root)
            package = generate_gang_package(
                package_root, plan, run_id=run_id, package=package_name
            )
            gang_plan = json.loads(package.shard_plan.read_text(encoding="utf-8"))
            gang_plan["broker_run_id"] = broker.run_id
            gang_plan["broker_url"] = broker.url
            gang_plan["repository_revisions"] = manifest.get("repository_revisions", {})
            atomic_write_json(package.shard_plan, gang_plan)
            command = self._bazel_command(
                "build",
                package.targets,
                f"--config={self._resource}_aone_bazel_cache",
                f"--jobs={self._build_jobs}",
                "--remote_download_outputs=minimal",
                "--keep_going",
            )
            completed = self._run_command(
                command, cwd=self._workspace_root, env=self._process_env, check=False
            )
            if completed.returncode:
                raise RuntimeError(f"Bazel gang build exited {completed.returncode}")
        except Exception as error:
            return self._stage_failure(
                store=store,
                manifest=manifest,
                stage="build",
                error=error,
                package=package,
            )
        self._write_stage(
            store,
            manifest,
            stage="build",
            status="completed",
            details={"targets": list(package.targets)},
        )
        return package

    def _run_submit_stage(
        self,
        plan: ExecutionPlan,
        store: RunStore,
        manifest: Dict[str, Any],
        package: GeneratedAOnePackage,
        broker: GangBroker,
    ) -> Optional[PerfTestError]:
        """Run the remote test; a nonzero exit is recorded, never raised."""
        resource = plan.resource_plan
        self._write_stage(store, manifest, stage="submit", status="running")
        test_failure: Optional[PerfTestError] = None
        test_exit: Optional[int] = None
        details: Dict[str, Any] = {
            "targets": list(package.targets),
            # Allow all members to run concurrently; this is not atomic allocation.
            "jobs": max(self._max_parallel, resource.machine_count),
            # The generated gang rule declares env_inherit for these names,
            # so exporting them into the submit environment is what carries
            # them to every remote shard.
            "env_inherit": [BROKER_TOKEN_ENV],
        }
        submit_env, saved_env = self._export_broker_env(broker)
        try:
            command = self._bazel_command(
                "test",
                package.targets,
                f"--config={self._resource}_aone_bazel_remote",
                "--strategy=TestRunner=remote",
                f"--jobs={details['jobs']}",
                "--test_output=errors",
                "--keep_going",
                "--remote_download_outputs=all",
                "--nocache_test_results",
            )
            command.append(f"--test_timeout={package.test_timeout_seconds}")
            completed = self._run_command(
                command, cwd=self._workspace_root, env=submit_env, check=False
            )
            test_exit = int(completed.returncode)
        except Exception as error:
            test_failure = PostExecutionError(
                f"Bazel gang test invocation failed: {error}"
            )
        finally:
            self._restore_broker_env(saved_env)
        if test_failure is None and test_exit:
            test_failure = PostExecutionError(f"Bazel gang test exited {test_exit}")
        if test_failure is None and broker.gang_state != "completed":
            test_failure = PostExecutionError(
                f"AOne gang did not complete: {broker.failure_reason or broker.gang_state}"
            )
        if test_exit is not None:
            details["bazel_exit_code"] = test_exit
        if test_failure is not None:
            details["error"] = _redact(str(test_failure), broker.token)
        self._write_stage(
            store,
            manifest,
            stage="submit",
            status="completed" if test_failure is None else "failed",
            details=details,
        )
        return test_failure

    def _export_broker_env(
        self, broker: GangBroker
    ) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
        """Export the broker token into the submit environment.

        The generated rule inherits this token from the Bazel invocation's
        environment. It is never placed on the command line, where it would
        be visible to process listings and command logs.
        """
        saved = {BROKER_TOKEN_ENV: os.environ.get(BROKER_TOKEN_ENV)}
        submit_env = dict(self._process_env)
        submit_env[BROKER_TOKEN_ENV] = broker.token
        os.environ[BROKER_TOKEN_ENV] = broker.token
        return submit_env, saved

    @staticmethod
    def _restore_broker_env(saved: Mapping[str, Optional[str]]) -> None:
        """Drop the exported token so later stages never see it."""
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _run_collect_stage(
        self,
        root: Path,
        store: RunStore,
        manifest: Dict[str, Any],
        broker: GangBroker,
        package: GeneratedAOnePackage,
        machine_count: int,
    ) -> Tuple[Optional[CollectionOutcome], Optional[str]]:
        """Collect master artifacts and non-master logs; failures never raise."""
        self._write_stage(
            store,
            manifest,
            state=RunState.COLLECTING,
            stage="collect",
            status="running",
        )
        manifest["gang"]["hosts"] = [
            {"shard_index": member.shard_index, "hostname": member.hostname}
            for member in broker.members
        ]
        try:
            collection = self._collect(root, package, machine_count)
        except Exception as error:
            message = _redact(f"{type(error).__name__}: {error}", broker.token)
            self._write_stage(
                store,
                manifest,
                stage="collect",
                status="failed",
                details={"error": message},
            )
            return None, message
        self._write_stage(
            store,
            manifest,
            stage="collect",
            status=collection.state.value,
            details={"master_state": collection.state.value},
        )
        return collection, None

    def _run_report_stage(
        self,
        plan: ExecutionPlan,
        root: Path,
        store: RunStore,
        manifest: Dict[str, Any],
        package: GeneratedAOnePackage,
        test_failure: Optional[PerfTestError],
        collection: Optional[CollectionOutcome],
        collect_error: Optional[str],
    ) -> AOneExecutionOutcome:
        self._write_stage(
            store,
            manifest,
            state=RunState.REPORTING,
            stage="report",
            status="running",
        )
        if collection is not None:
            final_state = reconcile_terminal_state(collection.state, test_failure)
            final_failure = test_failure or collection.failure
            if test_failure is not None and collection.state == RunState.COMPLETED:
                # A completed envelope contradicts the observed nonzero bazel
                # exit; the test failure is authoritative.
                final_state = RunState.FAILED
                final_failure = CaseExecutionError(str(test_failure))
        else:
            final_state = RunState.FAILED
            # The original test failure wins over collection errors.
            final_failure = (
                test_failure
                if test_failure is not None
                else CollectionError(
                    collect_error or "gang collection produced no master run tree"
                )
            )
        # A worker's completed envelope cannot override a submit-side failure.
        # Preserve case measurements, but reconcile the persisted terminal status.
        self._persist_terminal_status(root, final_state)
        reporter = self._reporter_override or self._default_reporter(plan.mode)
        report: Optional[Mapping[str, Any]] = None
        try:
            report = reporter(root)
        except Exception as error:
            if final_state == RunState.COMPLETED:
                # Only a fully successful run is failed by its report errors.
                final_state = RunState.FAILED
                final_failure = ReportError(f"Gang offline report failed: {error}")
            self._write_stage(
                store,
                manifest,
                stage="report",
                status="failed",
                details={"error": str(error)},
            )
        else:
            self._write_stage(
                store,
                manifest,
                stage="report",
                status="completed",
                details={"report_kind": report.get("report_kind")},
            )
        self._persist_terminal_status(root, final_state)
        self._write_manifest(store, manifest, state=final_state)
        return AOneExecutionOutcome(
            state=final_state,
            failure=final_failure,
            collection=collection,
            report=report,
            package=package,
        )

    # Collection -----------------------------------------------------------

    def _collect(
        self,
        root: Path,
        package: GeneratedAOnePackage,
        machine_count: int,
    ) -> CollectionOutcome:
        """Copy the master run tree and non-master logs from the testlogs."""
        target = self._testlogs_root / _package_path(package.targets[0])
        collection: Optional[CollectionOutcome] = None
        errors: List[str] = []
        for shard_index in range(machine_count):
            shard_dir = target / f"shard_{shard_index + 1}_of_{machine_count}"
            logs_home = root / "workload" / "gang_shards" / f"shard_{shard_index + 1}"
            try:
                # Even failed actions without undeclared outputs can have a log.
                if (shard_dir / "test.log").is_file():
                    logs_home.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(shard_dir / "test.log", logs_home / "test.log")
                outputs = self._shard_outputs(shard_dir)
                self._copy_shard_logs(shard_dir, outputs, logs_home)
                if shard_index == 0:
                    collection = self._collect_master_tree(root, outputs)
                else:
                    self._copy_worker_traces(outputs, root / "traces", shard_index)
            except Exception as error:
                # Preserve the failure, but finish visiting the other machines.
                errors.append(f"shard {shard_index}: {type(error).__name__}: {error}")
        if errors:
            raise CollectionError("; ".join(errors))
        if collection is None:
            raise CollectionError(
                f"gang master shard produced no test outputs under {target}"
            )
        return collection

    @staticmethod
    def _persist_terminal_status(root: Path, state: RunState) -> None:
        for relative in ("results/result.json", "results/summary.json"):
            path = root / relative
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != state.value:
                    payload["status"] = state.value
                    atomic_write_json(path, payload)

    def _shard_outputs(self, shard_dir: Path) -> Path:
        """Return the shard's undeclared outputs dir, extracting remote zips."""
        outputs = shard_dir / "test.outputs"
        archive = outputs / "outputs.zip"
        if archive.is_file():
            destination = shard_dir / "test.outputs_extracted"
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            with zipfile.ZipFile(archive) as stream:
                members = stream.infolist()
                if len(members) > _MAX_ARCHIVE_MEMBERS:
                    raise ValueError(
                        f"gang outputs archive exceeds member limit: {archive}"
                    )
                for info in members:
                    _safe_zip_member(info, archive)
                    stream.extract(info, destination)
            return destination
        if outputs.is_dir():
            return outputs
        raise CollectionError(f"gang shard test outputs are missing: {shard_dir}")

    @staticmethod
    def _copy_shard_logs(shard_dir: Path, outputs: Path, logs_home: Path) -> None:
        logs_home.mkdir(parents=True, exist_ok=True)
        test_log = shard_dir / "test.log"
        if test_log.is_file():
            shutil.copy2(test_log, logs_home / "test.log")
        for path in sorted(outputs.iterdir()):
            if path.is_file():
                shutil.copy2(path, logs_home / path.name)
        # Engine server logs live under the worker's TEST_OUTPUT_* run dirs;
        # cap each file so a verbose engine cannot balloon the run directory.
        _MAX_ENGINE_LOG_BYTES = 2 * 1024 * 1024
        for engine_dir in sorted(outputs.glob("TEST_OUTPUT_*")):
            destination = logs_home / engine_dir.name
            for source in sorted(engine_dir.rglob("*.log")):
                if not source.is_file():
                    continue
                relative = source.relative_to(engine_dir)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.stat().st_size > _MAX_ENGINE_LOG_BYTES:
                    with source.open("rb") as handle:
                        handle.seek(-_MAX_ENGINE_LOG_BYTES, os.SEEK_END)
                        tail = handle.read(_MAX_ENGINE_LOG_BYTES)
                    target.write_bytes(tail)
                else:
                    shutil.copy2(source, target)

    @staticmethod
    def _copy_worker_traces(outputs: Path, trace_root: Path, shard_index: int) -> None:
        for engine_dir in sorted(outputs.glob("TEST_OUTPUT_*")):
            copy_member_traces(engine_dir, trace_root, f"shard_{shard_index + 1}")

    def _collect_master_tree(self, root: Path, outputs: Path) -> CollectionOutcome:
        snapshot = inspect_run_directory(root)
        collected = collect_master_artifacts(
            root,
            outputs / "run",
            plan=snapshot.execution_plan,
            run_id=snapshot.manifest["run_id"],
        )
        return CollectionOutcome(
            collected.state, collected.cases, failure=collected.failure
        )

    # Helpers --------------------------------------------------------------

    def _bazel_command(
        self, verb: str, targets: Sequence[str], *extra: str
    ) -> List[str]:
        return bazel_argv(
            self._bazel,
            self._startup_args,
            verb,
            targets,
            self._platform,
            self._common_args,
            extra,
        )

    def _default_reporter(self, mode: str) -> Callable[[Path], Mapping[str, Any]]:
        if mode == "prefill":
            from rtp_llm.test.perf_test.prefill.report import generate_prefill_report

            return lambda root: generate_prefill_report(root, acquire_lock=False)

        def _decode_report(root: Path) -> Mapping[str, Any]:
            # The decode master finalizes its report next to the envelope, so
            # the collected artifact is the report.
            path = Path(root) / "reports" / "decode_report.json"
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ReportError(
                    f"collected decode report is unreadable: {error}"
                ) from error

        return _decode_report

    @staticmethod
    def _quiet(action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception:  # pragma: no cover - shutdown must never mask results
            pass


def create_default_gang_executor(
    plan: ExecutionPlan,
    *,
    startup_args: Sequence[str] = (),
    common_args: Sequence[str] = (),
    broker_host: Optional[str] = None,
):
    """Build and run the production gang executor from the AOne environment."""
    resource = plan.resource_plan
    if not isinstance(resource, AOneResourcePlan):  # pragma: no cover - upstream
        raise ValueError("gang execution requires an AOne resource plan")
    return GangAOneExecutor(
        workspace_root=_workspace_root(),
        resource=resource.resource,
        gpu_type=resource.gpu_type,
        max_parallel=resource.max_parallel,
        startup_args=startup_args,
        common_args=common_args,
        broker_host=broker_host,
    ).execute(plan)
