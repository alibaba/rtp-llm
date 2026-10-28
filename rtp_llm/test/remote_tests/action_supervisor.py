"""Supervise a remote pytest action and clean up escaped session children."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

TIMEOUT_EXIT_CODE = 124
HEARTBEAT_STALL_EXIT_CODE = 125


def _read_environ(pid: int) -> bytes:
    try:
        return Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return b""


def _pid_has_session(pid: int, session_id: str) -> bool:
    needle = f"RTP_REMOTE_SESSION_ID={session_id}".encode()
    return needle in _read_environ(pid).split(b"\0")


def _session_pids(session_id: str) -> List[int]:
    current = os.getpid()
    pids: List[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current:
            continue
        if _pid_has_session(pid, session_id):
            pids.append(pid)
    return pids


def _signal_process_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            os.kill(pid, signum)
        except OSError:
            pass


def _signal_pids(pids: Iterable[int], signum: int) -> None:
    for pid in sorted(set(pids)):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signum)
        except OSError:
            pass


def _dump_diagnostics(reason: str, child_pid: Optional[int], session_id: str) -> None:
    sys.stderr.write(
        f">>>RTP_REMOTE_INFRA_STALL reason={reason} child_pid={child_pid}\n"
    )
    sys.stderr.write(f">>>RTP_REMOTE_SESSION_ID {session_id}\n")
    pids = set(_session_pids(session_id))
    if child_pid:
        pids.add(child_pid)
    try:
        sys.stderr.write(">>>RTP_REMOTE_PROCESS_SNAPSHOT_START\n")
        if pids:
            proc = subprocess.run(
                [
                    "ps",
                    "-o",
                    "pid,ppid,pgid,stat,etime,cmd",
                    "-p",
                    ",".join(str(pid) for pid in sorted(pids)),
                ],
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5,
                check=False,
            )
            sys.stderr.write(proc.stdout)
            if proc.stdout and not proc.stdout.endswith("\n"):
                sys.stderr.write("\n")
        else:
            sys.stderr.write("(no live session pids)\n")
        sys.stderr.write(">>>RTP_REMOTE_PROCESS_SNAPSHOT_END\n")
    except Exception as exc:
        sys.stderr.write(f"[action_supervisor] failed to collect ps snapshot: {exc}\n")
    sys.stderr.flush()


def _terminate_child(
    child: subprocess.Popen,
    *,
    session_id: str,
    reason: str,
    grace_seconds: int,
) -> None:
    _dump_diagnostics(reason, child.pid if child else None, session_id)
    if child and child.poll() is None:
        _signal_process_group(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_process_group(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    escaped = _session_pids(session_id)
    if escaped:
        sys.stderr.write(
            "[action_supervisor] terminating escaped session pids: "
            + ",".join(str(p) for p in escaped)
            + "\n"
        )
        sys.stderr.flush()
        _signal_pids(escaped, signal.SIGTERM)
        time.sleep(min(2, grace_seconds))
        _signal_pids(_session_pids(session_id), signal.SIGKILL)


def _heartbeat_age(path: Path) -> Optional[float]:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _touch_heartbeat(path: Optional[Path], label: str) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{int(time.time())} {label}\n")
    except OSError as exc:
        sys.stderr.write(f"[action_supervisor] heartbeat write failed: {exc}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--heartbeat-file", default=None)
    parser.add_argument("--heartbeat-timeout", type=int, default=0)
    parser.add_argument("--grace-seconds", type=int, default=15)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    session_id = os.environ.get("RTP_REMOTE_SESSION_ID") or f"supervisor-{os.getpid()}"
    os.environ["RTP_REMOTE_SESSION_ID"] = session_id
    heartbeat_file = Path(args.heartbeat_file) if args.heartbeat_file else None
    if heartbeat_file is not None:
        os.environ["RTP_REMOTE_HEARTBEAT_FILE"] = str(heartbeat_file)
    _touch_heartbeat(heartbeat_file, "supervisor_start")

    child = subprocess.Popen(command, start_new_session=True)
    terminating = False

    def _handle_signal(signum, frame):
        nonlocal terminating
        terminating = True
        _signal_process_group(child.pid, signum)

    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    started = time.time()
    exit_code = 0
    reason = ""
    try:
        while True:
            rc = child.poll()
            if rc is not None:
                exit_code = rc
                break
            now = time.time()
            if args.timeout > 0 and now - started >= args.timeout:
                reason = "supervisor_timeout"
                exit_code = TIMEOUT_EXIT_CODE
                _terminate_child(
                    child,
                    session_id=session_id,
                    reason=reason,
                    grace_seconds=args.grace_seconds,
                )
                break
            if heartbeat_file is not None and args.heartbeat_timeout > 0:
                age = _heartbeat_age(heartbeat_file)
                if age is not None and age >= args.heartbeat_timeout:
                    reason = "heartbeat_stall"
                    exit_code = HEARTBEAT_STALL_EXIT_CODE
                    _terminate_child(
                        child,
                        session_id=session_id,
                        reason=reason,
                        grace_seconds=args.grace_seconds,
                    )
                    break
            if terminating:
                try:
                    child.wait(timeout=args.grace_seconds)
                except subprocess.TimeoutExpired:
                    _terminate_child(
                        child,
                        session_id=session_id,
                        reason="signal",
                        grace_seconds=args.grace_seconds,
                    )
                exit_code = child.returncode if child.returncode is not None else 143
                break
            time.sleep(1)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        _touch_heartbeat(heartbeat_file, "supervisor_end")

    if reason:
        sys.stderr.write(f"[action_supervisor] {reason} exit_code={exit_code}\n")
        sys.stderr.flush()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
