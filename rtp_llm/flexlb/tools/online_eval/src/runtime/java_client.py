"""Launch JavaLoadClient with isolated request configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from runtime.java_runtime import resolve_java21
from runtime.load_client import LOAD_CLIENT_ENV_VARS, validate_environment, validate_flow_environment
from runtime.paths import MOCK_JAR
from runtime.process import ManagedProcess, ProcessOps

if TYPE_CHECKING:
    from runtime.environment import EnvManager


class ClientOps:
    """Drives JavaLoadClient as a subprocess with fully explicit env."""

    def __init__(
        self, env_manager: EnvManager, jvm_xms: str = "4g", jvm_xmx: str = "4g"
    ):
        self.env_manager = env_manager
        self.jvm_xms = jvm_xms
        self.jvm_xmx = jvm_xmx

    def _base_env(self, overrides: dict) -> dict:
        overrides = validate_environment(overrides)
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
        overrides = validate_flow_environment(overrides)
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = self._argv()
        menv = self._base_env({**overrides, "OUTPUT_DIR": str(output_dir)})
        proc = ProcessOps.start(argv, menv, log_file)
        env = self.env_manager.current
        if env is not None:
            env.load_clients.append(proc)
        return proc, output_dir
