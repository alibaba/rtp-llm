"""Owned subprocess handles and directed process operations."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional


class ManagedProcess:
    """A subprocess started by the harness, restartable from its original argv."""

    def __init__(
        self, proc: subprocess.Popen, argv: list[str], env: dict, log_file: Path
    ):
        self.proc = proc
        self.argv = argv
        self.env = env
        self.log_file = log_file
        self.start_epoch = int(time.time())

    @property
    def pid(self) -> int:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait(self, timeout_s: float = 15.0) -> bool:
        try:
            self.proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            return False

    def terminate(self, timeout_s: float = 10.0) -> None:
        """SIGTERM → wait → SIGKILL fallback."""
        if not self.alive():
            return
        try:
            self.proc.terminate()
        except OSError:
            pass
        if not self.wait(timeout_s):
            self.kill9()
        # Drain zombies.
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def kill9(self) -> None:
        try:
            os.kill(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    # -- Mode 2 (freeze) primitives, per-master directed -------------------

    def freeze(self) -> None:
        """SIGSTOP this process (frozen: port up, state retained)."""
        ProcessOps.sigstop(self.proc.pid)

    def unfreeze(self) -> None:
        """SIGCONT this process (hot recovery: state intact)."""
        ProcessOps.sigcont(self.proc.pid)

    def tail_log(self, lines: int = 40) -> str:
        try:
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
            return "\n".join(text.splitlines()[-lines:])
        except Exception:
            return "<no log>"


class ProcessOps:
    """Static process utilities (kill by pid / pgrep pattern sweep)."""

    @staticmethod
    def start(
        argv: list[str], env: dict, log_file: Path, cwd: Optional[Path] = None
    ) -> ManagedProcess:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        out = open(log_file, "wb")
        try:
            proc = subprocess.Popen(
                argv,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(cwd) if cwd else None,
            )
        finally:
            out.close()
        return ManagedProcess(proc, argv, env, log_file)

    @staticmethod
    def kill9(pid: int) -> None:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    # ------------------------------------------------------------------
    # Mode 2 fault primitives (HA case test, brief p3: "SIGSTOP / SIGCONT
    # 原语（ProcessOps 一行级扩展）").  Directed at a specific master PID
    # through the masters registry (ManagedProcess.freeze / unfreeze).
    # ------------------------------------------------------------------

    @staticmethod
    def sigstop(pid: int) -> None:
        """Mode 2 (freeze) injection: process frozen, port stays up,
        application stops responding, in-memory state retained."""
        os.kill(pid, signal.SIGSTOP)

    @staticmethod
    def sigcont(pid: int) -> None:
        """Mode 2 recovery: thaw a SIGSTOP-frozen process (hot recovery —
        memory state intact, as opposed to Mode 1 kill -9 cold start)."""
        os.kill(pid, signal.SIGCONT)

    @staticmethod
    def restart(
        mp: ManagedProcess, new_log: Optional[Path] = None, timeout_s: float = 0.0
    ) -> ManagedProcess:
        """Start a fresh process with the same argv/env (old one must be dead)."""
        log = new_log or mp.log_file
        return ProcessOps.start(mp.argv, mp.env, log)
