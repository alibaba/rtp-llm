"""Build and launch the local ZooKeeper helper using its readiness protocol."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from runtime.paths import FLEXLB_DIR


ZK_LAUNCHER_CLASS = "org.flexlb.consistency.ZkTestingServerLauncher"

ZK_READY_PREFIX = "ZK_READY"

ZK_LAUNCH_CMD_ENV = "FLEXLB_FT_ZK_LAUNCH_CMD"

ZK_READY_TIMEOUT_S = float(os.environ.get("FLEXLB_FT_ZK_READY_TIMEOUT_S", "120"))


class ZkHelperOps:
    """Lifecycle glue for the ZK helper JVM.

    Default launch: resolve the flexlb-sync TEST classpath once per run-dir
    via ``mvnw -pl flexlb-sync -am dependency:build-classpath`` (cached in
    <run_dir>/zk_classpath.txt), then
    ``java -cp test-classes:classes:<deps> org.flexlb.consistency.ZkTestingServerLauncher``.
    Override the whole command through FLEXLB_FT_ZK_LAUNCH_CMD if the
    flexlb-sync assembly differs.
    """

    @staticmethod
    def find_ready_connect_string(log_file: Path) -> Optional[str]:
        """Scan the redirected stdout for the 'ZK_READY <connectString>' line."""
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] == ZK_READY_PREFIX and len(parts) >= 2:
                return parts[1]
        return None

    @staticmethod
    def _mvnw(args: list, env: dict, timeout_s: int = 900, what: str = "mvnw") -> None:
        mvnw = FLEXLB_DIR / "mvnw"
        if not mvnw.is_file():
            raise RuntimeError(f"mvnw not found: {mvnw}")
        cmd = [str(mvnw), "-q", *args]
        # First run may compile reactor modules — allow a long window.
        proc = subprocess.run(
            cmd,
            cwd=str(FLEXLB_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{what} failed (rc={proc.returncode}):\n"
                f"{(proc.stderr or proc.stdout)[-2000:]}"
            )

    @staticmethod
    def _ensure_test_classpath(java_bin: str, run_dir: Path) -> str:
        """flexlb-sync TEST-scope dependency classpath via mvnw (cached).

        Field-tested two-step boot (flexlb-sync owner contract):
          1. ``-pl flexlb-sync -am test-compile`` — compiles the launcher
             itself (src/test) plus reactor deps;
          2. ``-pl flexlb-sync dependency:build-classpath`` (NO -am:
             with it every reactor module writes its own outputFile)
             with an ABSOLUTE -Dmdep.outputFile and includeScope=test
             (curator-test is test-scoped in flexlb-sync/pom.xml).
        Cached per run-dir so repeated env builds reuse one resolution.
        """
        cp_file = run_dir / "zk_classpath.txt"
        if cp_file.is_file() and cp_file.stat().st_size > 0:
            return cp_file.read_text(encoding="utf-8").strip()
        env = dict(os.environ)
        # mvnw honours JAVA_HOME; derive it from the resolved java binary.
        java_home = Path(java_bin).resolve().parents[1]
        if (java_home / "bin" / "java").exists():
            env.setdefault("JAVA_HOME", str(java_home))
        ZkHelperOps._mvnw(
            ["-pl", "flexlb-sync", "-am", "test-compile"],
            env,
            timeout_s=900,
            what="mvnw test-compile (flexlb-sync)",
        )
        ZkHelperOps._mvnw(
            [
                "-pl",
                "flexlb-sync",
                "dependency:build-classpath",
                f"-Dmdep.outputFile={cp_file}",  # absolute (run_dir is abs)
                "-Dmdep.includeScope=test",
            ],
            env,
            timeout_s=600,
            what="mvnw dependency:build-classpath (flexlb-sync)",
        )
        if not cp_file.is_file() or cp_file.stat().st_size == 0:
            raise RuntimeError(
                f"mvnw build-classpath produced no classpath file: {cp_file}"
            )
        return cp_file.read_text(encoding="utf-8").strip()

    @staticmethod
    def default_launch_argv(java_bin: str, run_dir: Path) -> list[str]:
        deps = ZkHelperOps._ensure_test_classpath(java_bin, run_dir)
        sync_target = FLEXLB_DIR / "flexlb-sync" / "target"
        cp = os.pathsep.join(
            [
                str(sync_target / "test-classes"),
                str(sync_target / "classes"),
                deps,
            ]
        )
        # --port 0: ZK auto-allocates a free port; the ZK_READY line
        # advertises the actual connectString (no port-collision risk
        # against sibling harness processes).
        return [java_bin, "-cp", cp, ZK_LAUNCHER_CLASS, "--port", "0"]
