"""Launch and stop JavaLoadClient with isolated request configuration."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from runtime.java_runtime import resolve_java21
from runtime.load_client import LOAD_CLIENT_ENV_VARS
from runtime.paths import MOCK_JAR
from runtime.process import ManagedProcess, ProcessOps

if TYPE_CHECKING:
    from runtime.environment import EnvManager


class LoadClientResult:
    """Raw client output handle.

    Phase B removed summary.json (the client records raw rows only), so the
    derived total/ok/errors summary fields are gone with it — per_request()
    rows are the sole client-side source (no-backward-compat). The underlying
    file is client_events.jsonl (renamed from per_request.jsonl together with
    the multi-component JSONL event streams).
    """

    def __init__(self, output_dir: Path, returncode: int):
        self.output_dir = output_dir
        self.returncode = returncode

    def per_request(self) -> list[dict]:
        path = self.output_dir / "client_events.jsonl"
        rows = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows


class ClientOps:
    """Drives JavaLoadClient as a subprocess with fully explicit env."""

    def __init__(
        self, env_manager: EnvManager, jvm_xms: str = "4g", jvm_xmx: str = "4g"
    ):
        self.env_manager = env_manager
        self.jvm_xms = jvm_xms
        self.jvm_xmx = jvm_xmx

    def _base_env(self, overrides: dict) -> dict:
        menv = dict(os.environ)
        for var in LOAD_CLIENT_ENV_VARS:
            menv[var] = ""
        for key, value in overrides.items():
            menv[key] = str(value)
        return menv

    def _argv(self) -> list[str]:
        return [
            resolve_java21(),
            f"-Xms{self.jvm_xms}",
            f"-Xmx{self.jvm_xmx}",
            "-cp",
            str(MOCK_JAR),
            "org.flexlb.mockengine.JavaLoadClient",
        ]

    def run(
        self,
        overrides: dict,
        output_dir: Path,
        log_file: Path,
        timeout_s: Optional[float] = None,
        label: str = "load_client",
    ) -> LoadClientResult:
        """Run JavaLoadClient synchronously until it exits."""
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = self._argv()
        menv = self._base_env({**overrides, "OUTPUT_DIR": str(output_dir)})
        proc = ProcessOps.start(argv, menv, log_file)
        try:
            proc.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill9()
        rc = proc.proc.returncode if proc.proc.returncode is not None else -1
        return LoadClientResult(output_dir, rc)

    def run_async(
        self,
        overrides: dict,
        output_dir: Path,
        log_file: Path,
        label: str = "load_client_async",
    ) -> tuple[ManagedProcess, Path]:
        """Start JavaLoadClient WITHOUT waiting for it (HA background flow).

        HA cases keep a steady client running ACROSS fault injections
        (that is the whole point — observe the in-flight behaviour).  The
        process is registered on env.load_clients so a mid-case failure
        still gets cleaned up by _stop_env_processes().
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = self._argv()
        menv = self._base_env({**overrides, "OUTPUT_DIR": str(output_dir)})
        proc = ProcessOps.start(argv, menv, log_file)
        env = self.env_manager.current
        if env is not None:
            env.load_clients.append(proc)
        return proc, output_dir

    def stop_async(
        self,
        proc: ManagedProcess,
        output_dir: Path,
        timeout_s: float = 15.0,
    ) -> LoadClientResult:
        """Terminate a run_async client and return its result handle.

        SIGTERM lets the JVM run its shutdown path (jsonl writers flush);
        rows still buffered at the exact kill instant may be lost, which is
        why HA assertions compare pre/post-injection WINDOWS instead of
        exact request totals.
        """
        proc.terminate(timeout_s=timeout_s)
        rc = proc.proc.returncode if proc.proc.returncode is not None else -1
        return LoadClientResult(output_dir, rc)
