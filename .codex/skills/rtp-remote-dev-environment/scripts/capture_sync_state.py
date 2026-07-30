#!/usr/bin/env python3
"""Capture read-only Mac/origin/b300 Git state for an operation ledger."""

import argparse
import datetime as dt
import json
import pathlib
import re
import shlex
import subprocess
from typing import Dict, List, Optional


def run(command: List[str], cwd: Optional[pathlib.Path] = None) -> Dict[str, object]:
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "command": command,
        "exit_code": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def output(command: List[str], cwd: pathlib.Path) -> Optional[str]:
    result = run(command, cwd)
    if result["exit_code"] != 0:
        return None
    value = str(result["stdout"])
    return value or None


def redact_url(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", value)


def local_state(repo: pathlib.Path, branch: str) -> Dict[str, object]:
    status = output(["git", "status", "--porcelain=v1"], repo) or ""
    tracked_status = output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"], repo
    ) or ""
    origin_url = output(["git", "remote", "get-url", "origin"], repo)
    ls_remote = run(
        ["git", "ls-remote", "--heads", "origin", branch],
        repo,
    )
    remote_sha = None
    if ls_remote["exit_code"] == 0 and ls_remote["stdout"]:
        remote_sha = str(ls_remote["stdout"]).split()[0]
    return {
        "repo": str(repo),
        "branch": output(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo),
        "head": output(["git", "rev-parse", "HEAD"], repo),
        "tree": output(["git", "rev-parse", "HEAD^{tree}"], repo),
        "origin_tracking": output(
            ["git", "rev-parse", "--verify", "origin/" + branch], repo
        ),
        "origin_remote": remote_sha,
        "origin_url": redact_url(origin_url),
        "status": status.splitlines(),
        "tracked_status": tracked_status.splitlines(),
        "ls_remote_exit_code": ls_remote["exit_code"],
        "ls_remote_stderr": str(ls_remote["stderr"]),
    }


def remote_state(host: str, repo: str) -> Dict[str, object]:
    probes = {
        "head": ["git", "-C", repo, "rev-parse", "HEAD"],
        "tree": ["git", "-C", repo, "rev-parse", "HEAD^{tree}"],
        "branch": ["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
        "status": ["git", "-C", repo, "status", "--porcelain=v1"],
    }
    state: Dict[str, object] = {"host": host, "repo": repo}
    for key, probe in probes.items():
        remote_command = " ".join(shlex.quote(part) for part in probe)
        result = run(["ssh", "-o", "BatchMode=yes", host, remote_command])
        state[key] = result["stdout"] if result["exit_code"] == 0 else None
        state[key + "_exit_code"] = result["exit_code"]
        if result["exit_code"] != 0:
            state[key + "_stderr"] = result["stderr"]
    status = str(state.get("status") or "")
    state["status"] = status.splitlines()
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-repo")
    parser.add_argument("--log-root", type=pathlib.Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    log_root = args.log_root or repo / ".tmp" / "rtp-remote-dev-operations"
    operation_dir = log_root / args.operation_id
    operation_dir.mkdir(parents=True, exist_ok=True)

    record: Dict[str, object] = {
        "schema_version": 1,
        "operation_id": args.operation_id,
        "phase": args.phase,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local": local_state(repo, args.branch),
    }
    if bool(args.remote_host) != bool(args.remote_repo):
        parser.error("--remote-host and --remote-repo must be provided together")
    if args.remote_host and args.remote_repo:
        record["b300"] = remote_state(args.remote_host, args.remote_repo)

    local = record["local"]
    assert isinstance(local, dict)
    remote = record.get("b300")
    synchronized = (
        local.get("head") is not None
        and local.get("head") == local.get("origin_tracking")
        and local.get("head") == local.get("origin_remote")
        and not local.get("tracked_status")
    )
    if isinstance(remote, dict):
        synchronized = (
            synchronized
            and remote.get("head") == local.get("head")
            and remote.get("tree") == local.get("tree")
            and not remote.get("status")
        )
    record["synchronized"] = synchronized

    destination = operation_dir / (args.phase + ".json")
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (operation_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(destination)
    print("synchronized=" + str(synchronized).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
