"""Self-describing run-directory layout and exclusive writer ownership."""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.storage import atomic_write_json

RUN_MANIFEST_SCHEMA_VERSION = 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise PlanningError(f"cannot read run artifact {path}: {error}") from error


@dataclass(frozen=True)
class RunDirectoryPaths:
    root: Path
    source_config: Path
    resolved_config: Path
    execution_plan: Path
    manifest: Path
    workload: Path
    checkpoints: Path
    case_results: Path
    final_result: Path
    summary: Path
    reports: Path
    profiles: Path
    traces: Path
    writer_lock: Path

    @classmethod
    def for_root(cls, root: Union[str, Path]) -> "RunDirectoryPaths":
        root_path = Path(root)
        return cls(
            root=root_path,
            source_config=root_path / "source_config.json",
            resolved_config=root_path / "resolved_config.json",
            execution_plan=root_path / "execution_plan.json",
            manifest=root_path / "manifest.json",
            workload=root_path / "workload",
            checkpoints=root_path / "checkpoints",
            case_results=root_path / "results" / "cases",
            final_result=root_path / "results" / "result.json",
            summary=root_path / "results" / "summary.json",
            reports=root_path / "reports",
            profiles=root_path / "profiles",
            traces=root_path / "traces",
            writer_lock=root_path / ".writer.lock",
        )


@dataclass(frozen=True)
class RunDirectorySnapshot:
    paths: RunDirectoryPaths
    source_config: Dict[str, Any]
    resolved_config: Dict[str, Any]
    execution_plan: ExecutionPlan
    manifest: Dict[str, Any]


def initialize_run_directory(
    root: Union[str, Path],
    *,
    source_config: Mapping[str, Any],
    plan: ExecutionPlan,
    run_id: str,
    code_version: str,
    repository_revisions: Optional[Mapping[str, str]] = None,
) -> RunDirectoryPaths:
    """Create a new run layout and immutable planning snapshots exactly once."""
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("run_id must be a non-empty string")
    if not isinstance(code_version, str) or not code_version:
        raise PlanningError("code_version must be a non-empty string")
    if repository_revisions is not None and (
        not isinstance(repository_revisions, Mapping)
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None
            for name, revision in repository_revisions.items()
        )
    ):
        raise PlanningError(
            "repository_revisions must map repository names to full Git commit IDs"
        )
    paths = RunDirectoryPaths.for_root(root)
    if str(paths.root) != plan.result_directory:
        raise PlanningError(
            "execution plan result_directory does not match run directory"
        )
    try:
        paths.root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise PlanningError(f"run directory already exists: {paths.root}") from error
    except OSError as error:
        raise PlanningError(
            f"cannot create run directory {paths.root}: {error}"
        ) from error

    for directory in (
        paths.workload,
        paths.checkpoints,
        paths.case_results if plan.mode != "prefill" else paths.case_results.parent,
        paths.reports,
        paths.profiles,
        paths.traces,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    plan_dict = plan.to_dict()
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "state": "planned",
        "mode": plan.mode,
        "action": plan.action,
        "backend": plan.backend,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "case_ids": [case.case_id for case in plan.cases],
        "resource_plan": plan.resource_plan.to_dict(),
        "code_version": code_version,
        "repository_revisions": dict(repository_revisions or {}),
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "artifacts": {
            "source_config": "source_config.json",
            "resolved_config": "resolved_config.json",
            "execution_plan": "execution_plan.json",
            "case_results": (
                "results/cases.jsonl" if plan.mode == "prefill" else "results/cases"
            ),
            "result": "results/result.json",
            "summary": "results/summary.json",
            "reports": "reports",
            "profiles": "profiles",
            "traces": "traces",
        },
    }
    _write_json(paths.source_config, dict(source_config))
    _write_json(paths.resolved_config, plan_dict["resolved_config"])
    _write_json(paths.execution_plan, plan_dict)
    _write_json(paths.manifest, manifest)
    if plan.mode == "prefill":
        from rtp_llm.test.perf_test.prefill.fingerprints import resume_fingerprints

        (paths.case_results.parent / "cases.jsonl").touch()
        _write_json(
            paths.checkpoints / "prefill_progress.json",
            {"schema_version": 1, **resume_fingerprints(plan)},
        )
    return paths


def inspect_run_directory(root: Union[str, Path]) -> RunDirectorySnapshot:
    """Load the planning snapshots required to understand a run in isolation."""
    paths = RunDirectoryPaths.for_root(root)
    source_config = _read_json(paths.source_config)
    resolved_config = _read_json(paths.resolved_config)
    execution_plan = ExecutionPlan.from_dict(_read_json(paths.execution_plan))
    if paths.root.resolve() != Path(execution_plan.result_directory).resolve():
        raise PlanningError(
            f"run directory {paths.root.resolve()} does not match persisted "
            f"result_directory {execution_plan.result_directory}; "
            "moving or copying a run directory is not supported"
        )
    manifest = _read_json(paths.manifest)
    for name, value in (
        ("source_config", source_config),
        ("resolved_config", resolved_config),
        ("manifest", manifest),
    ):
        if not isinstance(value, dict):
            raise PlanningError(f"run {name} must be a JSON object")
    return RunDirectorySnapshot(
        paths=paths,
        source_config=source_config,
        resolved_config=resolved_config,
        execution_plan=execution_plan,
        manifest=manifest,
    )


class RunDirectoryLock:
    """Kernel-owned writer lock; the metadata file persists after release."""

    def __init__(self, root: Union[str, Path], *, action: str):
        self.path = RunDirectoryPaths.for_root(root).writer_lock
        self.action = action
        self._owned = False
        self._descriptor: Optional[int] = None

    def _owner_description(self) -> str:
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(owner, dict):
                return ", ".join(
                    f"{key}={owner.get(key, 'unknown')}"
                    for key in ("hostname", "pid", "action", "acquired_at")
                )
        except (OSError, UnicodeError, ValueError):
            pass
        return "owner metadata is unavailable"

    def __enter__(self) -> "RunDirectoryLock":
        metadata = {
            "schema_version": 2,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "action": self.action,
            "acquired_at": _timestamp(),
        }
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        created = False
        descriptor: Optional[int] = None
        try:
            try:
                try:
                    descriptor = os.open(self.path, flags, 0o600)
                    created = True
                except FileExistsError:
                    descriptor = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
            except OSError as error:
                raise ResourceError(
                    f"cannot acquire writer lock {self.path}: {error}"
                ) from error
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ResourceError(
                    f"run directory already has a writer: {self.path}; "
                    f"{self._owner_description()}"
                ) from error
            if not created:
                try:
                    owner = json.loads(os.read(descriptor, 4096))
                except (UnicodeError, ValueError):
                    owner = None
                # Older versions used file existence alone. Do not steal an
                # older writer's lock just because it does not hold flock.
                if not isinstance(owner, dict) or owner.get("schema_version") != 2:
                    raise ResourceError(
                        f"legacy or incomplete writer lock {self.path}; "
                        f"{self._owner_description()}; confirm the owner has "
                        "stopped before removing this file (see docs/reference.md)"
                    )
            payload = (
                json.dumps(metadata, sort_keys=True, ensure_ascii=True) + "\n"
            ).encode("utf-8")
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, payload)
            os.fsync(descriptor)
            self._descriptor = descriptor
            self._owned = True
            return self
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            self._descriptor = None
            self._owned = False
            raise

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> Optional[bool]:
        if self._owned:
            # Do not unlink: another opener must lock the same inode.
            assert self._descriptor is not None
            os.close(self._descriptor)
            self._descriptor = None
            self._owned = False
        return None
