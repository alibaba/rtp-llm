"""Bounded, read-only Linux resource evidence for slow smoke startup."""

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Final, Iterator, TypedDict

_FILE_LIMIT: Final = 65536
_PID_LIMIT: Final = 64
_THREAD_LIMIT: Final = 128
_THREAD_DETAIL_LIMIT: Final = 16
_SAMPLE_LIMIT: Final = 180
_INTERVAL_SECONDS: Final = 30
_CGROUP_FILES: Final = (
    "memory.current",
    "memory.max",
    "memory.high",
    "memory.stat",
    "memory.events",
    "memory.events.local",
    "memory.pressure",
    "cpu.stat",
    "cpu.max",
    "cpu.pressure",
    "io.stat",
    "io.pressure",
    "memory.usage_in_bytes",
    "memory.limit_in_bytes",
    "memory.failcnt",
    "memory.oom_control",
    "cpu.cfs_quota_us",
    "cpu.cfs_period_us",
)


class ResourceSample(TypedDict):
    monotonic_s: float
    unix_s: float
    root_pid: int
    processes: dict[str, dict[str, str]]
    cgroups: dict[str, dict[str, str]]
    host: dict[str, str]
    pid_limit_reached: bool


def _read(path: Path, limit: int = _FILE_LIMIT) -> str:
    try:
        with path.open(errors="replace") as source:
            text = source.read(limit + 1)
        return text[:limit] + ("\nTRUNCATED" if len(text) > limit else "")
    except OSError as error:
        return f"ERROR errno={error.errno}: {error.strerror}"


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_directories(membership: str, mountinfo: str) -> list[Path]:
    """Resolve v1/v2 membership against mount roots, including subtree mounts."""
    directories: list[Path] = []
    for member in membership.splitlines():
        parts = member.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = set(parts[1].split(",")) if parts[1] else set()
        group = PurePosixPath(parts[2])
        for mount in mountinfo.splitlines():
            fields = mount.split()
            if "-" not in fields:
                continue
            separator = fields.index("-")
            if separator < 6 or len(fields) <= separator + 3:
                continue
            kind = fields[separator + 1]
            if (controllers and kind != "cgroup") or (
                not controllers and kind != "cgroup2"
            ):
                continue
            if controllers and not controllers.issubset(
                set(fields[separator + 3].split(","))
            ):
                continue
            root = PurePosixPath(_unescape_mount(fields[3]))
            if not group.is_relative_to(root):
                continue
            mountpoint = Path(_unescape_mount(fields[4]))
            directory = mountpoint / group.relative_to(root)
            while True:
                if directory not in directories:
                    directories.append(directory)
                if directory == mountpoint:
                    break
                directory = directory.parent
    return directories


def _thread_snapshot(root: Path) -> str:
    threads: dict[str, dict[str, str]] = {}
    thread_limit_reached = False
    detail_limit_reached = False
    detail_count = 0
    for task in (root / "task").glob("[0-9]*"):
        if len(threads) == _THREAD_LIMIT:
            thread_limit_reached = True
            break
        record = {
            name: _read(task / name, 4096)
            for name in ("stat", "wchan", "schedstat", "io")
        }
        state = record["stat"].rpartition(")")[2].split()
        if state and state[0] in ("R", "D"):
            if detail_count < _THREAD_DETAIL_LIMIT:
                for name in ("stack", "syscall"):
                    record[name] = _read(task / name, 4096)
                detail_count += 1
            else:
                detail_limit_reached = True
        record["stat_after"] = _read(task / "stat", 4096)
        threads[task.name] = record
    return json.dumps(
        {
            "threads": threads,
            "thread_limit_reached": thread_limit_reached,
            "detail_limit_reached": detail_limit_reached,
        }
    )


def _snapshot(root_pid: int) -> ResourceSample:
    processes: dict[str, dict[str, str]] = {}
    cgroups: dict[str, dict[str, str]] = {}
    memberships: set[str] = set()
    pending = [root_pid]
    mountinfo = _read(Path("/proc/self/mountinfo"))
    mount_namespace = os.readlink("/proc/self/ns/mnt")
    while pending and len(processes) < _PID_LIMIT:
        pid = pending.pop()
        if str(pid) in processes:
            continue
        root = Path("/proc") / str(pid)
        files = {
            name: _read(root / name)
            for name in ("stat", "status", "io", "schedstat", "wchan", "cgroup")
        }
        try:
            with (root / "maps").open(errors="replace") as source:
                maps = source.read(1024 * 1024 + 1)
            files["maps"] = "\n".join(
                line
                for line in maps[: 1024 * 1024].splitlines()
                if any(
                    marker in line
                    for marker in ("safetensors", "rtp_llm_engram", "/mnt/")
                )
            )[:_FILE_LIMIT]
            files["maps_truncated"] = str(
                len(maps) > 1024 * 1024 or len(files["maps"]) >= _FILE_LIMIT
            )
        except OSError as error:
            files["maps"] = f"ERROR errno={error.errno}: {error.strerror}"
        if any(marker in files["maps"] for marker in ("safetensors", "rtp_llm_engram")):
            files["threads"] = _thread_snapshot(root)
        # The second stat distinguishes an exit/PID-reuse race from a stable sample.
        files["stat_after"] = _read(root / "stat")
        processes[str(pid)] = files
        try:
            files["mount_namespace"] = os.readlink(root / "ns/mnt")
        except OSError as error:
            files["mount_namespace"] = f"ERROR errno={error.errno}: {error.strerror}"
        children = {
            str(path.parent.name): _read(path)
            for path in (root / "task").glob("*/children")
        }
        files["children"] = json.dumps(children)
        for child_list in children.values():
            pending.extend(
                int(child) for child in child_list.split() if child.isdecimal()
            )
        membership = files["cgroup"]
        if files["mount_namespace"] != mount_namespace:
            cgroups["UNRESOLVED mount namespace pid=" + str(pid)] = {
                "target": files["mount_namespace"],
                "sampler": mount_namespace,
            }
            continue
        if membership in memberships:
            continue
        memberships.add(membership)
        directories = _cgroup_directories(membership, mountinfo)
        if not directories:
            cgroups["UNRESOLVED " + membership] = {"mountinfo": mountinfo}
        for directory in directories:
            if str(directory) not in cgroups:
                cgroups[str(directory)] = {
                    name: _read(directory / name) for name in _CGROUP_FILES
                }
    host = {
        name: _read(Path("/proc") / name)
        for name in (
            "meminfo",
            "vmstat",
            "pressure/memory",
            "pressure/cpu",
            "pressure/io",
            "uptime",
            "sys/kernel/hostname",
            "sys/kernel/random/boot_id",
        )
    }
    host["mountinfo"] = mountinfo
    host["page_size_bytes"] = str(os.sysconf("SC_PAGE_SIZE"))
    host["clock_ticks_per_second"] = str(os.sysconf("SC_CLK_TCK"))
    try:
        space = os.statvfs("/dev/shm")
        host["tmpfs_free_bytes"] = str(space.f_bavail * space.f_frsize)
    except OSError as error:
        host["tmpfs_free_bytes"] = f"ERROR errno={error.errno}: {error.strerror}"
    return ResourceSample(
        monotonic_s=time.monotonic(),
        unix_s=time.time(),
        root_pid=root_pid,
        processes=processes,
        cgroups=cgroups,
        host=host,
        pid_limit_reached=bool(pending),
    )


@contextmanager
def monitor_startup_resources(output: Path) -> Iterator[None]:
    """Sample independently of blocked copy chunks; stop before closing the artifact."""
    stopped = threading.Event()
    root_pid = os.getpid()
    with output.open("w") as stream:

        def collect() -> None:
            for _ in range(_SAMPLE_LIMIT):
                stream.write(json.dumps(_snapshot(root_pid)) + "\n")
                stream.flush()
                if stopped.wait(_INTERVAL_SECONDS):
                    return
            stream.write(json.dumps({"sample_limit_reached": _SAMPLE_LIMIT}) + "\n")
            stream.flush()

        worker = threading.Thread(
            target=collect, name="startup-resource-monitor", daemon=True
        )
        worker.start()
        try:
            yield
        finally:
            stopped.set()
            worker.join()
