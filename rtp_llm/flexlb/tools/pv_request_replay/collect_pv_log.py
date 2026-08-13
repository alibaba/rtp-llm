#!/usr/bin/env python3
"""Collect bounded FlexLB ``pv.log`` snapshots from one or more instances."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")
LOG_TIME_RE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d{3,6})"
)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CommandRunner = Callable[[Sequence[str]], str]


class CollectionError(RuntimeError):
    """Raised after a strict collection writes its failure manifest."""

    def __init__(self, message: str, manifest: dict[str, Any]):
        super().__init__(message)
        self.manifest = manifest


class _SourceCollectionError(RuntimeError):
    def __init__(self, message: str, manifest: dict[str, Any]):
        super().__init__(message)
        self.manifest = manifest


def _format_time(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ", timespec="milliseconds") if value else None


def _as_log_time(value: datetime | str) -> datetime:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = f"{candidate[:-1]}+00:00"
        value = datetime.fromisoformat(candidate)
    if value.tzinfo is not None:
        value = value.astimezone(SHANGHAI).replace(tzinfo=None)
    return value


def _parse_log_time(line: str) -> datetime | None:
    match = LOG_TIME_RE.match(line)
    if not match:
        return None
    value = match.group("time").replace(",", ".")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _safe_name(value: str) -> str:
    safe = SAFE_NAME_RE.sub("_", value).strip("._")
    return safe or "unknown-instance"


def _instance_name(value: str) -> str:
    return value.removeprefix("inst/")


def _default_runner(command: Sequence[str]) -> str:
    for attempt in range(3):
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout
        if attempt < 2:
            time.sleep(attempt + 1)
            continue
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        )
    raise AssertionError("unreachable")


def resolve_running_instances(
    workspace: str,
    deployment: str,
    command_runner: CommandRunner | None = None,
) -> list[str]:
    """Resolve all currently RUNNING instances for a deployment."""

    runner = command_runner or _default_runner
    output = runner(
        [
            "dashctl",
            "-w",
            workspace,
            "get",
            "inst",
            "--dep",
            deployment,
            "-o",
            "json",
        ]
    )
    payload = json.loads(output)
    if isinstance(payload, dict):
        payload = payload.get("objects", payload.get("items", []))
    if not isinstance(payload, list):
        raise RuntimeError("dashctl instance response is not a JSON list")

    resolved: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        status_name = status.get("name") if isinstance(status, dict) else status
        if str(status_name).lower() != "running":
            continue
        metadata = item.get("metadata")
        name = metadata.get("name") if isinstance(metadata, dict) else item.get("name")
        if isinstance(name, str) and name:
            resolved.append(_instance_name(name))
    return sorted(set(resolved))


def _remote_command(
    workspace: str,
    instance: str,
    command_runner: CommandRunner,
    *remote_command: str,
) -> str:
    return command_runner(
        [
            "dashctl",
            "-w",
            workspace,
            "exec",
            f"inst/{instance}",
            "--",
            *remote_command,
        ]
    )


def _log_file_pattern(log_name: str) -> re.Pattern[str]:
    escaped = re.escape(log_name)
    return re.compile(rf"^{escaped}(?:\.\d{{4}}-\d{{2}}-\d{{2}}\.\d+\.log)?$")


def _rotation_sort_key(path: str, log_name: str) -> tuple[int, str, int]:
    basename = Path(path).name
    if basename == log_name:
        return (1, "", 0)
    match = re.search(r"\.(\d{4}-\d{2}-\d{2})\.(\d+)\.log$", basename)
    if not match:
        return (0, basename, 0)
    return (0, match.group(1), int(match.group(2)))


def _rotation_index_gap_errors(paths: Sequence[str | Path], log_name: str) -> list[str]:
    escaped = re.escape(log_name)
    pattern = re.compile(rf"^{escaped}\.(\d{{4}}-\d{{2}}-\d{{2}})\.(\d+)\.log$")
    by_date: dict[str, set[int]] = {}
    for path in paths:
        match = pattern.fullmatch(Path(path).name)
        if match:
            by_date.setdefault(match.group(1), set()).add(int(match.group(2)))

    errors: list[str] = []
    for date, indices in sorted(by_date.items()):
        missing = sorted(set(range(0, max(indices) + 1)) - indices)
        if missing:
            errors.append(
                f"rotation index gap for {log_name} on {date}: missing "
                + ", ".join(str(index) for index in missing)
            )
    return errors


def list_remote_log_files(
    workspace: str,
    instance: str,
    log_dir: str,
    log_name: str,
    command_runner: CommandRunner | None = None,
) -> list[str]:
    runner = command_runner or _default_runner
    output = _remote_command(workspace, instance, runner, "ls", "-1", log_dir)
    pattern = _log_file_pattern(log_name)
    files = [
        f"{log_dir.rstrip('/')}/{name.strip()}"
        for name in output.splitlines()
        if pattern.fullmatch(name.strip())
    ]
    return sorted(files, key=lambda path: _rotation_sort_key(path, log_name))


def _remote_line_count(
    workspace: str,
    instance: str,
    remote_path: str,
    command_runner: CommandRunner,
) -> int:
    output = _remote_command(
        workspace, instance, command_runner, "wc", "-l", remote_path
    )
    match = re.search(r"(?:^|\s)(\d+)(?:\s|$)", output)
    if not match:
        raise RuntimeError(f"could not parse line count for {remote_path}: {output!r}")
    return int(match.group(1))


def _remote_file_stat(
    workspace: str,
    instance: str,
    remote_path: str,
    command_runner: CommandRunner,
) -> tuple[int, int]:
    output = _remote_command(
        workspace,
        instance,
        command_runner,
        "stat",
        "-c",
        "%i %s",
        remote_path,
    )
    match = re.search(r"(?:^|\s)(\d+)\s+(\d+)(?:\s|$)", output)
    if not match:
        raise RuntimeError(f"could not parse inode/size for {remote_path}: {output!r}")
    return int(match.group(1)), int(match.group(2))


def _remote_time_bounds(
    workspace: str,
    instance: str,
    remote_path: str,
    command_runner: CommandRunner,
) -> tuple[datetime | None, datetime | None]:
    first_output = _remote_command(
        workspace, instance, command_runner, "head", "-n", "1", remote_path
    )
    last_output = _remote_command(
        workspace, instance, command_runner, "tail", "-n", "1", remote_path
    )
    first = next(
        (timestamp for line in first_output.splitlines() if (timestamp := _parse_log_time(line))),
        None,
    )
    last = next(
        (timestamp for line in last_output.splitlines() if (timestamp := _parse_log_time(line))),
        None,
    )
    return first, last


def _outside_window_manifest(
    remote_path: str,
    first: datetime,
    last: datetime,
    collection_start: datetime,
    collection_end: datetime,
) -> dict[str, Any]:
    before = last < collection_start
    return {
        "path": remote_path,
        "source": "remote",
        "status": "skipped_outside_window",
        "reported_line_count": None,
        "fetched_line_count": 0,
        "snapshot_truncated": False,
        "pages": [],
        "errors": [],
        "parsed_line_count": 2 if first != last else 1,
        "unparsable_line_count": 0,
        "before_collection_window_line_count": None,
        "after_collection_window_line_count": None,
        "selected_line_count": 0,
        "first_parsed_log_time": _format_time(first),
        "last_parsed_log_time": _format_time(last),
        "first_selected_log_time": None,
        "last_selected_log_time": None,
        "skip_reason": (
            "file ends before collection window"
            if before
            else "file starts after collection window"
        ),
        "probed_collection_window": {
            "start": _format_time(collection_start),
            "end": _format_time(collection_end),
        },
    }


def _validate_active_file_state(
    remote_path: str,
    before: tuple[int, int],
    after: tuple[int, int],
    source_manifest: dict[str, Any],
) -> None:
    source_manifest["active_file_stat"] = {
        "before": {"inode": before[0], "size": before[1]},
        "after": {"inode": after[0], "size": after[1]},
    }
    if before[0] == after[0] and after[1] >= before[1]:
        return
    if before[0] != after[0]:
        detail = f"inode changed from {before[0]} to {after[0]}"
    else:
        detail = f"size shrank from {before[1]} to {after[1]}"
    message = f"active log changed during collection for {remote_path}: {detail}"
    source_manifest["status"] = "failed"
    source_manifest["snapshot_truncated"] = True
    source_manifest["errors"] = [message]
    raise _SourceCollectionError(message, source_manifest)


def _complete_lines(output: str) -> list[str]:
    lines = output.splitlines(keepends=True)
    return [line for line in lines if line.endswith(("\n", "\r"))]


def _fetch_remote_file(
    workspace: str,
    instance: str,
    remote_path: str,
    page_lines: int,
    command_runner: CommandRunner,
) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    # Kept in the private call contract for CLI/API compatibility.  A response
    # now consumes every complete line returned by dashctl rather than imposing
    # this old client-side page cap.
    del page_lines
    source_manifest: dict[str, Any] = {
        "path": remote_path,
        "source": "remote",
        "status": "collecting",
        "reported_line_count": None,
        "fetched_line_count": 0,
        "snapshot_truncated": True,
        "pages": [],
        "errors": [],
    }
    try:
        line_count = _remote_line_count(
            workspace, instance, remote_path, command_runner
        )
    except Exception as error:
        source_manifest["status"] = "failed"
        source_manifest["errors"] = [str(error)]
        raise _SourceCollectionError(str(error), source_manifest) from error
    source_manifest["reported_line_count"] = line_count
    fetched: list[tuple[int, str]] = []
    next_line = 1
    pending_boundary_line: str | None = None

    while next_line <= line_count:
        try:
            output = _remote_command(
                workspace,
                instance,
                command_runner,
                "tail",
                "-n",
                f"+{next_line}",
                remote_path,
            )
        except Exception as error:
            source_manifest["status"] = "failed"
            source_manifest["fetched_line_count"] = len(fetched)
            source_manifest["errors"] = [str(error)]
            raise _SourceCollectionError(str(error), source_manifest) from error
        remaining = line_count - next_line + 1
        complete = _complete_lines(output)
        if not complete:
            message = (
                f"tail response for {remote_path}:{next_line} contained no complete line; "
                "the command response was truncated inside the first line"
            )
            source_manifest["status"] = "failed"
            source_manifest["fetched_line_count"] = len(fetched)
            source_manifest["errors"] = [message]
            raise _SourceCollectionError(message, source_manifest)
        # ``dashctl exec`` can append a newline while truncating an output
        # response in the middle of a source log line.  A trailing newline
        # alone is therefore insufficient evidence that the final returned
        # line is complete.  Leave that boundary line pending and re-read it
        # as the first line of the next ``tail -n +N`` response.  A match
        # verifies it; a mismatch drops the synthetic partial line and starts
        # again from the real source line at the same offset.  Even when an
        # apparent response has as many newline-delimited chunks as the rest
        # of the file, defer its final line: the transport can synthesize that
        # final newline too.
        request_start = next_line
        boundary_verified = pending_boundary_line is None
        accepted: list[tuple[int, str]] = []
        candidate_start = next_line
        candidates = complete[:remaining]
        if pending_boundary_line is not None:
            if complete[0] == pending_boundary_line:
                accepted.append((next_line, pending_boundary_line))
                candidates = complete[1:]
                candidate_start = next_line + 1
                boundary_verified = True
            else:
                boundary_verified = False
        candidates = candidates[: line_count - candidate_start + 1]

        if request_start == line_count:
            # This response starts at the physical final source line.  It is
            # safe to finish after the overlap check above; all observed PV
            # lines are well below the command transport limit.
            if not accepted:
                accepted.append((line_count, candidates[0]))
            pending_boundary_line = None
            next_line = line_count + 1
        elif candidates:
            safe_candidates = candidates[:-1]
            accepted.extend(
                (source_line, line)
                for source_line, line in zip(
                    range(candidate_start, candidate_start + len(safe_candidates)),
                    safe_candidates,
                )
            )
            pending_boundary_line = candidates[-1]
            next_line = candidate_start + len(candidates) - 1
        else:
            # The verified overlap was the only complete source line in this
            # response.  Move past it and re-read the next source line.
            pending_boundary_line = None
            next_line = candidate_start

        fetched.extend(accepted)
        endpoint_verified = next_line > line_count
        raw_response_covers_remaining = len(complete) >= remaining
        page_start = accepted[0][0] if accepted else request_start
        page_end = accepted[-1][0] if accepted else request_start - 1
        source_manifest["pages"].append(
            {
                "start_line": request_start,
                "end_line": page_end,
                "line_count": len(accepted),
                "tail_returned_complete_lines": len(complete),
                "tail_response_complete": endpoint_verified,
                "transport_response_truncated": not raw_response_covers_remaining,
                "raw_response_covers_remaining": raw_response_covers_remaining,
                "boundary_verified": boundary_verified,
                "endpoint_verified": endpoint_verified,
            }
        )

    if len(fetched) != line_count:
        message = (
            f"snapshot truncation for {remote_path}: fetched {len(fetched)} of "
            f"{line_count} lines"
        )
        source_manifest["status"] = "failed"
        source_manifest["fetched_line_count"] = len(fetched)
        source_manifest["errors"] = [message]
        raise _SourceCollectionError(message, source_manifest)
    source_manifest.update(
        {
            "status": "complete",
            "fetched_line_count": len(fetched),
            "snapshot_truncated": False,
        }
    )
    return fetched, source_manifest


def _local_files(
    value: str | Path | Iterable[str | Path], log_name: str
) -> list[Path]:
    if isinstance(value, (str, Path)):
        candidates = [Path(value)]
    else:
        candidates = [Path(path) for path in value]
    files: list[Path] = []
    pattern = _log_file_pattern(log_name)
    for candidate in candidates:
        if candidate.is_dir():
            files.extend(
                path
                for path in candidate.iterdir()
                if path.is_file() and pattern.fullmatch(path.name)
            )
        else:
            files.append(candidate)
    return sorted(files, key=lambda path: _rotation_sort_key(str(path), log_name))


def _fetch_local_file(path: Path) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        lines = handle.readlines()
    return list(enumerate(lines, start=1)), {
        "path": str(path.resolve()),
        "source": "local",
        "status": "complete",
        "reported_line_count": len(lines),
        "fetched_line_count": len(lines),
        "snapshot_truncated": False,
        "pages": [],
        "errors": [],
    }


def _filter_file_lines(
    lines: list[tuple[int, str]],
    source_manifest: dict[str, Any],
    collection_start: datetime,
    collection_end: datetime,
) -> list[tuple[datetime, int, str]]:
    selected: list[tuple[datetime, int, str]] = []
    parsed_times: list[datetime] = []
    unparsable = 0
    before = 0
    after = 0
    for source_line, line in lines:
        timestamp = _parse_log_time(line)
        if timestamp is None:
            unparsable += 1
            continue
        parsed_times.append(timestamp)
        if timestamp < collection_start:
            before += 1
        elif timestamp > collection_end:
            after += 1
        else:
            selected.append((timestamp, source_line, line))

    selected_times = [item[0] for item in selected]
    source_manifest.update(
        {
            "parsed_line_count": len(parsed_times),
            "unparsable_line_count": unparsable,
            "before_collection_window_line_count": before,
            "after_collection_window_line_count": after,
            "selected_line_count": len(selected),
            "first_parsed_log_time": _format_time(min(parsed_times, default=None)),
            "last_parsed_log_time": _format_time(max(parsed_times, default=None)),
            "first_selected_log_time": _format_time(min(selected_times, default=None)),
            "last_selected_log_time": _format_time(max(selected_times, default=None)),
        }
    )
    return selected


def _collect_instance(
    workspace: str,
    instance: str,
    output_dir: Path,
    source_paths: list[str | Path],
    source_mode: str,
    collection_start: datetime,
    collection_end: datetime,
    page_lines: int,
    command_runner: CommandRunner,
    log_name: str,
    preflight_errors: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    safe_instance = _safe_name(instance)
    instance_dir = output_dir / "raw" / safe_instance
    instance_dir.mkdir(parents=True, exist_ok=True)
    file_manifests: list[dict[str, Any]] = []
    selected: list[tuple[datetime, int, int, str]] = []
    parsed_first: list[datetime] = []
    parsed_last: list[datetime] = []

    try:
        for file_index, source_path in enumerate(source_paths):
            try:
                if source_mode == "remote":
                    remote_path = str(source_path)
                    active_file = Path(remote_path).name == log_name
                    active_before = (
                        _remote_file_stat(
                            workspace, instance, remote_path, command_runner
                        )
                        if active_file
                        else None
                    )
                    first_bound, last_bound = _remote_time_bounds(
                        workspace, instance, remote_path, command_runner
                    )
                    if (
                        first_bound is not None
                        and last_bound is not None
                        and (
                            last_bound < collection_start
                            or first_bound > collection_end
                        )
                    ):
                        lines = []
                        source_manifest = _outside_window_manifest(
                            remote_path,
                            first_bound,
                            last_bound,
                            collection_start,
                            collection_end,
                        )
                    else:
                        lines, source_manifest = _fetch_remote_file(
                            workspace,
                            instance,
                            remote_path,
                            page_lines,
                            command_runner,
                        )
                    if active_before is not None:
                        active_after = _remote_file_stat(
                            workspace, instance, remote_path, command_runner
                        )
                        _validate_active_file_state(
                            remote_path,
                            active_before,
                            active_after,
                            source_manifest,
                        )
                else:
                    lines, source_manifest = _fetch_local_file(Path(source_path))
            except _SourceCollectionError as error:
                error.manifest["order"] = file_index
                file_manifests.append(error.manifest)
                raise
            except Exception as error:
                file_manifests.append(
                    {
                        "path": str(source_path),
                        "source": source_mode,
                        "status": "failed",
                        "reported_line_count": None,
                        "fetched_line_count": 0,
                        "snapshot_truncated": True,
                        "pages": [],
                        "errors": [str(error)],
                        "order": file_index,
                    }
                )
                raise
            if source_manifest["status"] == "skipped_outside_window":
                filtered = []
            else:
                filtered = _filter_file_lines(
                    lines, source_manifest, collection_start, collection_end
                )
            source_manifest["order"] = file_index
            file_manifests.append(source_manifest)
            if source_manifest["first_parsed_log_time"]:
                parsed_first.append(
                    datetime.fromisoformat(source_manifest["first_parsed_log_time"])
                )
                parsed_last.append(
                    datetime.fromisoformat(source_manifest["last_parsed_log_time"])
                )
            selected.extend(
                (timestamp, file_index, source_line, line)
                for timestamp, source_line, line in filtered
            )

        selected.sort(key=lambda item: (item[0], item[1], item[2]))
        snapshot_path = instance_dir / "pv.log.snapshot"
        with snapshot_path.open("w", encoding="utf-8", newline="") as handle:
            for _, _, _, line in selected:
                handle.write(line)

        first_observed = min(parsed_first, default=None)
        last_observed = max(parsed_last, default=None)
        first_selected = selected[0][0] if selected else None
        last_selected = selected[-1][0] if selected else None
        covers_start = first_observed is not None and first_observed <= collection_start
        covers_end = last_observed is not None and last_observed >= collection_end
        coverage_complete = covers_start and covers_end and not preflight_errors
        snapshot = {
            "instance": instance,
            "path": str(snapshot_path.relative_to(output_dir)),
            "first_log_time": _format_time(first_selected),
            "last_log_time": _format_time(last_selected),
            "line_count": len(selected),
        }
        instance_manifest = {
            "instance": instance,
            "safe_instance": safe_instance,
            "status": "complete" if coverage_complete else "partial",
            "source_files": file_manifests,
            "snapshot": snapshot,
            "coverage": {
                "first_observed_log_time": _format_time(first_observed),
                "last_observed_log_time": _format_time(last_observed),
                "covers_collection_start": covers_start,
                "covers_collection_end": covers_end,
                "complete": coverage_complete,
            },
            "snapshot_truncated": any(
                item["snapshot_truncated"] for item in file_manifests
            ),
            "errors": list(preflight_errors),
        }
        return instance_manifest, snapshot
    except Exception as error:  # keep other instances collectable in non-strict mode
        instance_manifest = {
            "instance": instance,
            "safe_instance": safe_instance,
            "status": "failed",
            "source_files": file_manifests,
            "snapshot": None,
            "coverage": {
                "first_observed_log_time": None,
                "last_observed_log_time": None,
                "covers_collection_start": False,
                "covers_collection_end": False,
                "complete": False,
            },
            "snapshot_truncated": True,
            "errors": [str(error)],
        }
        return instance_manifest, None


def _failed_instance_manifest(instance: str, error: str) -> dict[str, Any]:
    return {
        "instance": instance,
        "safe_instance": _safe_name(instance),
        "status": "failed",
        "source_files": [],
        "snapshot": None,
        "coverage": {
            "first_observed_log_time": None,
            "last_observed_log_time": None,
            "covers_collection_start": False,
            "covers_collection_end": False,
            "complete": False,
        },
        "snapshot_truncated": True,
        "errors": [error],
    }


def collect_logs(
    workspace: str,
    deployment: str | None,
    instances: Sequence[str] | None,
    start: datetime | str,
    end: datetime | str,
    output_dir: str | Path,
    log_dir: str = "/home/admin/logs",
    log_name: str = "pv.log",
    lead_grace: timedelta = timedelta(minutes=5),
    tail_grace: timedelta = timedelta(minutes=10),
    strict: bool = False,
    page_lines: int = 50,
    workers: int = 4,
    local_inputs: Mapping[str, str | Path | Iterable[str | Path]] | None = None,
    command_runner: CommandRunner | None = None,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Collect per-instance snapshots and return a complete collection manifest.

    Naive request and log timestamps are interpreted as Asia/Shanghai wall-clock
    time. Aware request timestamps are converted to that timezone before the
    local log comparison.
    """

    if page_lines <= 0:
        raise ValueError("page_lines must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    start_time = _as_log_time(start)
    end_time = _as_log_time(end)
    if start_time > end_time:
        raise ValueError("start must not be later than end")
    if lead_grace < timedelta(0) or tail_grace < timedelta(0):
        raise ValueError("grace periods must not be negative")

    collection_start = start_time - lead_grace
    collection_end = end_time + tail_grace
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runner = command_runner or _default_runner
    source_mode = "local" if local_inputs is not None else "remote"
    requested_instances = list(
        dict.fromkeys(_instance_name(instance) for instance in (instances or []))
    )
    top_errors: list[str] = []

    if local_inputs is not None:
        local_map = {
            _instance_name(name): _local_files(paths, log_name)
            for name, paths in local_inputs.items()
        }
        resolved_instances = requested_instances or sorted(local_map)
        missing = [name for name in resolved_instances if name not in local_map]
        if missing:
            top_errors.append(
                f"local input is missing for instances: {', '.join(missing)}"
            )
    else:
        local_map = {}
        if requested_instances:
            resolved_instances = requested_instances
        elif deployment:
            try:
                resolved_instances = resolve_running_instances(
                    workspace, deployment, runner
                )
            except Exception as error:
                resolved_instances = []
                top_errors.append(f"could not resolve deployment instances: {error}")
        else:
            resolved_instances = []
            top_errors.append("deployment or explicit instances is required")

    if not resolved_instances and not top_errors:
        top_errors.append("no RUNNING instances were resolved")

    def collect_target(
        instance: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if source_mode == "local":
            source_paths: list[str | Path] = list(local_map.get(instance, []))
            if not source_paths:
                return (
                    _failed_instance_manifest(
                        instance, "no local log files were supplied"
                    ),
                    None,
                )
        else:
            try:
                source_paths = list_remote_log_files(
                    workspace, instance, log_dir, log_name, runner
                )
            except Exception as error:
                return (
                    _failed_instance_manifest(
                        instance, f"could not list log files: {error}"
                    ),
                    None,
                )
            if not source_paths:
                return (
                    _failed_instance_manifest(
                        instance,
                        f"no {log_name} or {log_name}.YYYY-MM-DD.i.log files found",
                    ),
                    None,
                )
            preflight_errors = _rotation_index_gap_errors(source_paths, log_name)
        if source_mode == "local":
            preflight_errors = []

        return _collect_instance(
            workspace=workspace,
            instance=instance,
            output_dir=destination,
            source_paths=source_paths,
            source_mode=source_mode,
            collection_start=collection_start,
            collection_end=collection_end,
            page_lines=page_lines,
            command_runner=runner,
            log_name=log_name,
            preflight_errors=preflight_errors,
        )

    if resolved_instances:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(resolved_instances))
        ) as executor:
            collected = list(executor.map(collect_target, resolved_instances))
    else:
        collected = []
    instance_manifests = [item[0] for item in collected]
    snapshots: list[dict[str, Any]] = []
    for _, snapshot in collected:
        if snapshot:
            snapshots.append(snapshot)

    all_errors = list(top_errors)
    for item in instance_manifests:
        all_errors.extend(f"{item['instance']}: {error}" for error in item["errors"])
    if all_errors or any(item["status"] == "failed" for item in instance_manifests):
        status = "failed" if not snapshots else "partial"
    elif any(item["status"] == "partial" for item in instance_manifests):
        status = "partial"
    else:
        status = "complete"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "collector": "flexlb-pv-request-replay",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai",
        "status": status,
        "requested_window": {
            "start": _format_time(start_time),
            "end": _format_time(end_time),
        },
        "collection_window": {
            "start": _format_time(collection_start),
            "end": _format_time(collection_end),
            "lead_grace_seconds": lead_grace.total_seconds(),
            "tail_grace_seconds": tail_grace.total_seconds(),
        },
        "source": {
            "mode": source_mode,
            "workspace": workspace if source_mode == "remote" else None,
            "deployment": deployment if source_mode == "remote" else None,
            "requested_instances": requested_instances,
            "resolved_instances": resolved_instances,
            "log_dir": log_dir if source_mode == "remote" else None,
            "log_name": log_name,
            "workers": workers,
        },
        "instances": instance_manifests,
        "snapshots": snapshots,
        "snapshot_truncated": any(
            item["snapshot_truncated"] for item in instance_manifests
        ),
        "errors": all_errors,
    }
    if write_manifest:
        manifest_path = destination / "collect_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest["manifest_path"] = str(manifest_path.resolve())

    if strict and status != "complete":
        raise CollectionError(
            f"PV log collection was {status}; see collect_manifest.json", manifest
        )
    return manifest


def _parse_duration(value: str) -> timedelta:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("duration must look like 30s, 5m, or 1h")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    multiplier = {"s": 1, "m": 60, "h": 3600}[unit]
    return timedelta(seconds=amount * multiplier)


def _parse_local_inputs(values: Sequence[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for index, value in enumerate(values, start=1):
        if "=" in value:
            instance, path = value.split("=", 1)
        else:
            path = value
            instance = Path(path).stem or f"local-{index}"
        if not instance or not path:
            raise ValueError(f"invalid local input: {value!r}")
        result.setdefault(instance, []).append(Path(path))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect time-bounded FlexLB pv.log snapshots"
    )
    parser.add_argument("--workspace", default="ai-lab-test")
    parser.add_argument("--deployment")
    parser.add_argument("--instance", action="append", dest="instances")
    parser.add_argument(
        "--local-input",
        action="append",
        default=[],
        metavar="[INSTANCE=]PATH",
        help="read a local log file/directory instead of dashctl (repeatable)",
    )
    parser.add_argument("--start", required=True, help="Asia/Shanghai log time")
    parser.add_argument("--end", required=True, help="Asia/Shanghai log time")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", default="/home/admin/logs")
    parser.add_argument("--log-name", default="pv.log")
    parser.add_argument("--lead-grace", type=_parse_duration, default=timedelta(minutes=5))
    parser.add_argument("--tail-grace", type=_parse_duration, default=timedelta(minutes=10))
    parser.add_argument("--page-lines", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    local_inputs = _parse_local_inputs(args.local_input) if args.local_input else None
    try:
        manifest = collect_logs(
            workspace=args.workspace,
            deployment=args.deployment,
            instances=args.instances,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            log_name=args.log_name,
            lead_grace=args.lead_grace,
            tail_grace=args.tail_grace,
            strict=args.strict,
            page_lines=args.page_lines,
            workers=args.workers,
            local_inputs=local_inputs,
        )
    except CollectionError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
