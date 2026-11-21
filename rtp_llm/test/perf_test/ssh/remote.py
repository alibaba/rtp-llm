"""System SSH transport with data-only argv, bounded processes and file copies."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Sequence


class RemoteHost:
    def __init__(
        self,
        target: str,
        *,
        user=None,
        port=None,
        python_bin="/opt/conda310/bin/python",
        command_log=None,
    ):
        if (
            not isinstance(target, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:@\[\]-]+", target)
            or target.startswith("-")
        ):
            raise ValueError("invalid SSH target")
        if user is not None and (
            not isinstance(user, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", user)
            or user.startswith("-")
        ):
            raise ValueError("invalid SSH user")
        self.target = f"{user}@{target}" if user else target
        self._processes = set()
        self._process_lock = threading.Lock()
        self._cancelled = threading.Event()
        self.port = port
        self.python_bin = python_bin
        self.command_log = Path(command_log) if command_log else None

    def ssh_argv(self):
        args = [
            "ssh",
            "-T",
            "-oBatchMode=yes",
            "-oConnectTimeout=15",
            "-oServerAliveInterval=15",
            "-oServerAliveCountMax=3",
        ]
        if self.port is not None:
            args += ["-p", str(self.port)]
        return args

    def command(self, argv: Sequence[str]):
        return [
            *self.ssh_argv(),
            "--",
            self.target,
            shlex.join([str(arg) for arg in argv]),
        ]

    def _log(self, argv, returncode=None):
        if self.command_log:
            self.command_log.parent.mkdir(parents=True, exist_ok=True)
            with self.command_log.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "host": self.target,
                            "argv": list(argv),
                            "returncode": returncode,
                        }
                    )
                    + "\n"
                )

    def cancel_pending(self):
        """Abort local launches before the controller reconciles remote owners."""
        with self._process_lock:
            self._cancelled.set()
            for process in tuple(self._processes):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def reset_cancelled(self):
        with self._process_lock:
            if self._processes:
                raise RuntimeError("cannot reset transport with active operations")
            self._cancelled.clear()

    def _run(self, argv, *, data=None, timeout=120, output=None):
        self._log(argv)
        with tempfile.TemporaryFile() as errors:
            with self._process_lock:
                if self._cancelled.is_set():
                    raise RuntimeError("SSH transport launch was cancelled")
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                    stdout=output or subprocess.PIPE,
                    stderr=errors,
                    start_new_session=True,
                )
                self._processes.add(process)
            try:
                stdout, _ = process.communicate(input=data, timeout=timeout)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
            finally:
                with self._process_lock:
                    self._processes.discard(process)
            errors.seek(0)
            stderr = errors.read().decode(errors="replace")
            self._log(argv, process.returncode)
            if process.returncode:
                raise RuntimeError(
                    f"SSH operation failed on {self.target} (exit {process.returncode}): {stderr[-8192:]}"
                )
            return (stdout or b"").decode(errors="replace")

    def run(self, argv, *, data=None, timeout=120):
        return self._run(self.command(argv), data=data, timeout=timeout)

    def python(self, script: str, payload=None, *, timeout=120):
        data = None if payload is None else json.dumps(payload).encode()
        return self.run([self.python_bin, "-c", script], data=data, timeout=timeout)

    def write_json(self, path: str, payload):
        return self.python(
            "import json,pathlib,sys; p=json.load(sys.stdin); "
            'q=pathlib.Path(p["path"]); q.parent.mkdir(parents=True,exist_ok=True); '
            'q.write_text(json.dumps(p["value"],sort_keys=True,separators=(",",":"),ensure_ascii=True))',
            {"path": path, "value": payload},
        )

    def worker(self, worker_path, descriptor_path, operation, *, timeout=120):
        output = self.run(
            [
                self.python_bin,
                worker_path,
                "--descriptor",
                descriptor_path,
                "--operation",
                operation,
            ],
            timeout=timeout,
        )
        try:
            return json.loads(output)
        except ValueError as error:
            raise RuntimeError(
                f"worker on {self.target} returned invalid JSON: {output[-4096:]}"
            ) from error

    def push_git(self, worktree: str, revision: str, mirror: str, run_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("invalid Git run ref")
        remote = f"{self.target}:{mirror}"
        argv = [
            "git",
            "-C",
            worktree,
            "-c",
            f"core.sshCommand={shlex.join(self.ssh_argv())}",
            "push",
            "--",
            remote,
            f"{revision}:refs/perf/runs/{run_id}",
        ]
        self._run(argv, timeout=600)

    def fetch_tree(self, remote_path: str, destination: Path):
        """Stream an archive without capturing traces in memory; reject links/traversal."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        path = Path(remote_path)
        with tempfile.TemporaryFile() as stream:
            self._run(
                self.command(["tar", "-C", str(path), "-cf", "-", "."]),
                timeout=600,
                output=stream,
            )
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                for member in archive.getmembers():
                    target = (destination / member.name).resolve()
                    if not target.is_relative_to(destination.resolve()) or not (
                        member.isdir() or member.isfile()
                    ):
                        raise ValueError(f"unsafe remote artifact path: {member.name}")
                archive.extractall(destination)
