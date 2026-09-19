#!/usr/bin/env python3
"""Read bounded FlexLB ``pv.log`` snapshots from a Whale role.

Whale deployments are whale-on-spectrum: the Spectrum ``dep/<name>`` object that
``dashctl`` sees is a shell whose zones map to carbon roles, so ``dashctl get inst``
reports no instances for a role that is serving. This module resolves the role's pods
through ``whale`` and reads their logs through ``asicli`` instead.

Two platform constraints shape the implementation:

* ``asicli console exec`` truncates stdout at exactly 1048576 bytes and appends a
  newline to the truncated tail, so a whole-file ``cat`` silently loses data. Every
  read is therefore split into line-range chunks that are md5-verified against a
  remote digest, and a chunk that reaches the cap is reported instead of trusted.
* ``whale deployment describe`` is rate-limited locally to roughly one call per ten
  seconds per deployment and answers with a non-JSON message inside that window. It is
  called once per run and the cooldown answer is surfaced as such.

Remote commands are read-only (``ls``, ``head``, ``tail``, ``awk``, ``sed``, ``md5sum``
and ``wc``); nothing is written inside the container. ``asicli`` passes argv through
verbatim, so ``sh -c`` composes the pipelines and only the remote shell parses them.

The window end must not be in the future: ``pv.log`` keeps growing, and lines appended
after the count step would otherwise change the remote digest mid-collection.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from collect_pv_log import (
    CommandRunner,
    _as_log_time,
    _log_file_pattern,
    _parse_log_time,
    _rotation_sort_key,
)


POD_NAME_LABEL_SUFFIX = "/pod-name"
NAMESPACE_LABEL_SUFFIX = "namespace"
EXEC_STDOUT_LIMIT = 1048576
DEFAULT_CONTAINER = "load-balancer"
DEFAULT_LOG_DIR = "/home/admin/ai-whale/logs"
DEFAULT_ROLE = "master"
DEFAULT_PAGE_LINES = 300
COOLDOWN_HINT = "whale deployment describe is rate-limited locally"


class WhaleSourceError(RuntimeError):
    """Raised when a Whale role cannot be resolved or its log cannot be read intact."""


@dataclass(frozen=True)
class WhalePod:
    """One replica of a Whale role, addressed by ASI coordinates."""

    role: str
    pod_name: str
    namespace: str
    cluster: str
    ip: str | None
    health_status: str | None
    service_status: str | None

    @property
    def instance(self) -> str:
        return self.pod_name


def default_runner(command: Sequence[str]) -> str:
    """Run a CLI command and return stdout, retrying transient failures."""
    for attempt in range(3):
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout
        detail = result.stderr.strip() or result.stdout.strip()
        if attempt < 2 and "请等待" not in detail:
            time.sleep(attempt + 1)
            continue
        raise WhaleSourceError(
            f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        )
    raise AssertionError("unreachable")


def _whale_payload(
    runner: CommandRunner, args: Sequence[str], context: str | None
) -> Any:
    command = ["whale"]
    if context:
        command += ["--context", context]
    command += [*args, "-o", "json"]
    output = runner(command)
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        snippet = output.strip().splitlines()[0] if output.strip() else ""
        if "请等待" in output:
            raise WhaleSourceError(
                f"{COOLDOWN_HINT}; wait a few seconds and retry ({snippet})"
            ) from error
        raise WhaleSourceError(
            f"whale returned non-JSON output for {' '.join(args)}: {snippet}"
        ) from error


def resolve_deployment_id(
    runner: CommandRunner,
    service: str,
    deployment: str,
    context: str | None = None,
) -> str:
    """Resolve one deployment id from a service, matching name or logic name."""
    payload = _whale_payload(runner, ["deployment", "list", service], context)
    entries = (payload or {}).get("data", {}).get("deployment_status") or []
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and deployment
        in (entry.get("deployment_name"), entry.get("logic_deployment_name"))
    ]
    if not matches:
        known = ", ".join(
            str(entry.get("deployment_name"))
            for entry in entries
            if isinstance(entry, dict)
        )
        raise WhaleSourceError(
            f"no deployment named {deployment!r} on service {service!r}; "
            f"known: {known or 'none'}"
        )
    if len(matches) > 1:
        raise WhaleSourceError(
            f"deployment name {deployment!r} matched {len(matches)} deployments "
            f"on {service!r}"
        )
    deployment_id = matches[0].get("deployment_id")
    if not isinstance(deployment_id, str) or not deployment_id:
        raise WhaleSourceError(f"deployment {deployment!r} has no deployment_id")
    return deployment_id


def _label_value(labels: dict[str, Any], suffix: str) -> str | None:
    """Read a carbon label by key suffix.

    Carbon escapes ``.`` in Kubernetes label keys as ``#dot#``, so
    ``app.c2.io/pod-name`` arrives as ``app#dot#c2#dot#io/pod-name``. Matching on the
    suffix keeps that escaping convention from being load-bearing.
    """
    for key in sorted(labels):
        if isinstance(key, str) and key.endswith(suffix):
            value = labels[key]
            if isinstance(value, str) and value:
                return value
    return None


def resolve_role_pods(
    runner: CommandRunner,
    deployment_id: str,
    role: str,
    context: str | None = None,
) -> list[WhalePod]:
    """Resolve every pod of one carbon role, e.g. ``master`` → ``master_part``."""
    payload = _whale_payload(runner, ["deployment", "describe", deployment_id], context)
    carbon = (payload or {}).get("carbon_status") or {}
    cluster = carbon.get("hippo_id")
    if not isinstance(cluster, str) or not cluster:
        raise WhaleSourceError(
            f"deployment {deployment_id} has no carbon_status.hippo_id; "
            "cannot address ASI"
        )
    roles = (carbon.get("carbon_status") or {}).get("roles") or {}
    pattern = re.compile(rf"^{re.escape(role)}_part\d*$")
    matched = sorted(
        key for key in roles if isinstance(key, str) and pattern.fullmatch(key)
    )
    if not matched:
        raise WhaleSourceError(
            f"deployment {deployment_id} has no role matching {role!r}; "
            f"known roles: {', '.join(sorted(roles)) or 'none'}"
        )

    pods: list[WhalePod] = []
    for role_id in matched:
        for node in (roles[role_id] or {}).get("nodes") or []:
            status = (node or {}).get("cur_worker_node_status") or {}
            labels = status.get("labels") or {}
            pod_name = _label_value(labels, POD_NAME_LABEL_SUFFIX)
            if pod_name is None:
                continue
            namespace = _label_value(labels, NAMESPACE_LABEL_SUFFIX)
            if namespace is None:
                raise WhaleSourceError(
                    f"pod {pod_name} has no namespace label; cannot address ASI"
                )
            pods.append(
                WhalePod(
                    role=role_id,
                    pod_name=pod_name,
                    namespace=namespace,
                    cluster=cluster,
                    ip=status.get("ip"),
                    health_status=(status.get("health_info") or {}).get("health_status"),
                    service_status=(status.get("service_info") or {}).get("status"),
                )
            )
    if not pods:
        raise WhaleSourceError(
            f"role {role!r} of deployment {deployment_id} has no pods"
        )
    return pods


def _exec(
    runner: CommandRunner, pod: WhalePod, container: str, script: str
) -> str:
    return runner(
        [
            "asicli",
            "console",
            "exec",
            "-p",
            pod.pod_name,
            "-c",
            pod.cluster,
            "-N",
            pod.namespace,
            "--container",
            container,
            "--",
            "sh",
            "-c",
            script,
        ]
    )


def _window_filter(start: datetime, end: datetime, path: str) -> str:
    """Build an awk program selecting the lines inside ``[start, end]``.

    ``pv.log`` prefixes every line with a fixed-width ``YYYY-MM-DD HH:MM:SS.mmm``
    timestamp, so comparing the first 19 characters lexicographically is a
    chronological comparison that stays correct across day and month boundaries.
    """
    bounds = (
        f"-v s='{start.strftime('%Y-%m-%d %H:%M:%S')}' "
        f"-v e='{end.strftime('%Y-%m-%d %H:%M:%S')}'"
    )
    return f"awk {bounds} 'substr($0,1,19)>=s && substr($0,1,19)<=e' '{path}'"


def _list_log_files(
    runner: CommandRunner,
    pod: WhalePod,
    container: str,
    log_dir: str,
    log_name: str,
) -> list[str]:
    output = _exec(runner, pod, container, f"ls -1 '{log_dir}'")
    pattern = _log_file_pattern(log_name)
    names = [
        line.strip()
        for line in output.splitlines()
        if line.strip() and pattern.fullmatch(line.strip())
    ]
    return sorted(set(names), key=lambda name: _rotation_sort_key(name, log_name))


def _file_bounds(
    runner: CommandRunner, pod: WhalePod, container: str, path: str
) -> tuple[datetime | None, datetime | None]:
    output = _exec(
        runner, pod, container, f"head -n 1 '{path}'; tail -n 1 '{path}'"
    )
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return None, None
    first = _parse_log_time(lines[0])
    last = _parse_log_time(lines[-1]) if len(lines) > 1 else first
    return first, last


def _count_window_lines(
    runner: CommandRunner,
    pod: WhalePod,
    container: str,
    path: str,
    start: datetime,
    end: datetime,
) -> int:
    script = f"{_window_filter(start, end, path)} | wc -l"
    output = _exec(runner, pod, container, script).strip()
    try:
        return int(output.split()[0])
    except (IndexError, ValueError) as error:
        raise WhaleSourceError(
            f"could not count window lines in {path} on {pod.pod_name}: {output!r}"
        ) from error


def _chunk_count(total: int, page_lines: int) -> int:
    return (total + page_lines - 1) // page_lines if total > 0 else 0


def _chunk_digests(
    runner: CommandRunner,
    pod: WhalePod,
    container: str,
    path: str,
    start: datetime,
    end: datetime,
    total: int,
    page_lines: int,
) -> dict[int, str]:
    selector = _window_filter(start, end, path)
    script = (
        f"a=1; i=1; while [ $a -le {total} ]; do "
        f"b=$((a+{page_lines - 1})); if [ $b -gt {total} ]; then b={total}; fi; "
        f'printf "%d %s\\n" $i '
        f'"$({selector} | sed -n "${{a}},${{b}}p" | md5sum | cut -d\' \' -f1)"; '
        f"i=$((i+1)); a=$((b+1)); done"
    )
    output = _exec(runner, pod, container, script)
    digests: dict[int, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            digests[int(parts[0])] = parts[1]
    expected = _chunk_count(total, page_lines)
    if len(digests) != expected:
        raise WhaleSourceError(
            f"expected {expected} chunk digests for {path} on {pod.pod_name}, "
            f"got {len(digests)}"
        )
    return digests


def _download_chunk(
    runner: CommandRunner,
    pod: WhalePod,
    container: str,
    path: str,
    start: datetime,
    end: datetime,
    first_line: int,
    last_line: int,
    page_lines: int,
    expected_digest: str,
) -> list[str]:
    selector = _window_filter(start, end, path)
    output = _exec(
        runner, pod, container, f'{selector} | sed -n "{first_line},{last_line}p"'
    )
    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) >= EXEC_STDOUT_LIMIT:
        raise WhaleSourceError(
            f"chunk {first_line}-{last_line} of {path} on {pod.pod_name} reached the "
            f"{EXEC_STDOUT_LIMIT}-byte asicli stdout cap; reduce --page-lines "
            f"(currently {page_lines})"
        )
    if hashlib.md5(encoded).hexdigest() != expected_digest:
        raise WhaleSourceError(
            f"chunk {first_line}-{last_line} of {path} on {pod.pod_name} failed md5 "
            "verification; the read was lossy"
        )
    return output.splitlines(keepends=True)


def fetch_pod_window(
    runner: CommandRunner,
    pod: WhalePod,
    container: str,
    log_dir: str,
    log_name: str,
    start: datetime,
    end: datetime,
    page_lines: int = DEFAULT_PAGE_LINES,
) -> tuple[list[str], dict[str, Any]]:
    """Read every in-window log line of one pod, oldest rotation first."""
    if page_lines <= 0:
        raise ValueError("page_lines must be positive")
    window_start = _as_log_time(start)
    window_end = _as_log_time(end)
    names = _list_log_files(runner, pod, container, log_dir, log_name)
    files: list[dict[str, Any]] = []
    lines: list[str] = []
    for name in names:
        path = f"{log_dir.rstrip('/')}/{name}"
        first, last = _file_bounds(runner, pod, container, path)
        entry: dict[str, Any] = {
            "path": path,
            "first_log_time": _stamp(first),
            "last_log_time": _stamp(last),
        }
        if first is None:
            files.append({**entry, "status": "empty", "selected_line_count": 0})
            continue
        if (last or first) < window_start or first > window_end:
            files.append(
                {**entry, "status": "outside_window", "selected_line_count": 0}
            )
            continue
        total = _count_window_lines(
            runner, pod, container, path, window_start, window_end
        )
        entry["window_line_count"] = total
        if total == 0:
            files.append(
                {**entry, "status": "no_matching_lines", "selected_line_count": 0}
            )
            continue
        digests = _chunk_digests(
            runner, pod, container, path, window_start, window_end, total, page_lines
        )
        fetched: list[str] = []
        for index in range(_chunk_count(total, page_lines)):
            first_line = index * page_lines + 1
            last_line = min(first_line + page_lines - 1, total)
            fetched.extend(
                _download_chunk(
                    runner,
                    pod,
                    container,
                    path,
                    window_start,
                    window_end,
                    first_line,
                    last_line,
                    page_lines,
                    digests[index + 1],
                )
            )
        if len(fetched) != total:
            raise WhaleSourceError(
                f"read {len(fetched)} lines from {path} on {pod.pod_name} but the "
                f"remote window count was {total}"
            )
        files.append(
            {**entry, "status": "complete", "selected_line_count": len(fetched)}
        )
        lines.extend(fetched)
    return lines, {"pod": pod.pod_name, "files": files}


def _stamp(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ", timespec="milliseconds") if value else None


def write_pod_snapshot(output_dir: Path, pod: WhalePod, lines: Iterable[str]) -> Path:
    """Stage one pod's raw lines for the collector's local-input mode."""
    directory = output_dir / "whale_raw" / pod.instance
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pv.log"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)
    return path


def collect_whale_snapshots(
    pods: Sequence[WhalePod],
    output_dir: str | Path,
    start: datetime,
    end: datetime,
    container: str = DEFAULT_CONTAINER,
    log_dir: str = DEFAULT_LOG_DIR,
    log_name: str = "pv.log",
    page_lines: int = DEFAULT_PAGE_LINES,
    runner: CommandRunner | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Fetch every pod's window and return collector ``local_inputs`` plus a manifest."""
    active = runner or default_runner
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    local_inputs: dict[str, Path] = {}
    pods_manifest: list[dict[str, Any]] = []
    for pod in pods:
        lines, detail = fetch_pod_window(
            active, pod, container, log_dir, log_name, start, end, page_lines
        )
        path = write_pod_snapshot(destination, pod, lines)
        local_inputs[pod.instance] = path
        pods_manifest.append(
            {
                "instance": pod.instance,
                "role": pod.role,
                "namespace": pod.namespace,
                "cluster": pod.cluster,
                "ip": pod.ip,
                "health_status": pod.health_status,
                "service_status": pod.service_status,
                "container": container,
                "snapshot": str(path),
                "line_count": len(lines),
                "files": detail["files"],
            }
        )
    return local_inputs, {
        "source": "whale+asicli",
        "container": container,
        "log_dir": log_dir,
        "log_name": log_name,
        "page_lines": page_lines,
        "exec_stdout_limit": EXEC_STDOUT_LIMIT,
        "pods": pods_manifest,
    }
