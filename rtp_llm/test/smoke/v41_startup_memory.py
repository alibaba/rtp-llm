"""Best-effort OOM event evidence for the four-rank V4.1 startup smoke only."""

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path


def _snapshot(proc=Path("/proc/self")):
    """Resolve the test's own memory cgroup, not a host-wide default mount."""
    try:
        membership = {}
        for line in (proc / "cgroup").read_text().splitlines():
            _, controllers, path = line.split(":", 2)
            if controllers == "" or "memory" in controllers.split(","):
                membership[2 if not controllers else 1] = Path(path)
        for line in (proc / "mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            fs, _, options = right.split()[:3]
            ver = (
                2
                if fs == "cgroup2"
                else 1 if fs == "cgroup" and "memory" in options.split(",") else 0
            )
            group = membership.get(ver)
            if group is None or not group.is_absolute() or ".." in group.parts:
                continue
            root, mount = map(Path, left.split()[3:5])
            if group.is_relative_to(root):
                group_dir = mount / group.relative_to(root)
            elif group == Path("/"):
                # A container can present its cgroup namespace root as "/"
                # while mountinfo still describes the delegated subtree.
                group_dir = mount
                try:
                    with (group_dir / "cgroup.procs").open() as members:
                        if not any(row.strip() == str(os.getpid()) for row in members):
                            continue
                except OSError:
                    continue
            else:
                continue
            events_file = group_dir / (
                "memory.events" if ver == 2 else "memory.oom_control"
            )
            try:
                stat = events_file.stat()
                lines = events_file.read_text().splitlines()
            except OSError:
                continue
            allowed = (
                {"oom_kill", "oom_group_kill", "oom", "max"}
                if ver == 2
                else {"oom_kill"}
            )
            counters = {}
            for row in lines:
                name, value = row.split()
                if name in allowed:
                    counters[name] = int(value)
            if "oom_kill" in counters:
                return (ver, stat.st_dev, stat.st_ino), counters
    except (OSError, ValueError):
        pass
    return None


def _event_delta(before, after):
    if before is None or after is None or before[0] != after[0]:
        return {"status": "unavailable"}
    keys = before[1].keys() & after[1].keys()
    diff = {key: after[1][key] - before[1][key] for key in sorted(keys)}
    if any(value < 0 for value in diff.values()):
        return {"status": "unavailable"}
    return {"status": "observed", "version": before[0][0], "delta": diff}


@contextmanager
def record_v41_startup_events(output_dir):
    before = _snapshot()
    # Leave a breadcrumb even if the smoke Python process itself is killed.
    # Don't log absolute cgroup counters, paths, container IDs, PIDs or peers.
    logging.info(
        "V4.1 startup cgroup oom_kill counter available: %s", before is not None
    )
    try:
        yield
    finally:
        report = _event_delta(before, _snapshot())
        report["scope"] = "test cgroup during P/D startup; not per-rank attribution"
        logging.info(
            "V4.1 startup memory events: %s", json.dumps(report, sort_keys=True)
        )
        try:
            (output_dir / "v41_startup_memory.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        except OSError:
            # Diagnostics must never replace a startup error or fail a smoke.
            logging.warning("Could not record V4.1 startup memory artifact")
