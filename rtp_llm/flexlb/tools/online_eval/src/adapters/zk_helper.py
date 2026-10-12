"""Prepare ZooKeeper outside runs; launch only verified compiled artifacts."""

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
    """Explicit preparation owns Maven; runtime never compiles shared targets."""

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
    def prepare(java_bin: str) -> Path:
        import fcntl
        from artifacts.json_io import write_json
        directory = FLEXLB_DIR / 'flexlb-sync' / 'target' / 'online-eval-runtime'
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'prepare.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cp_file = directory / 'classpath.txt'
            env = dict(os.environ, JAVA_HOME=str(Path(java_bin).resolve().parents[1]))
            profile = ['-Popensource,!internal']
            # The selected terminal module is flexlb-sync. Resolve its dependencies
            # in the same reactor that builds them; no installed snapshot needed.
            ZkHelperOps._mvnw(profile + ['-pl', 'flexlb-sync', '-am', 'package',
                'dependency:build-classpath', '-DskipTests', '-Dmdep.regenerateFile=true',
                f'-Dmdep.outputFile={cp_file}', '-Dmdep.includeScope=test'], env,
                what='ZooKeeper reactor runtime preparation')
            if not cp_file.is_file() or not cp_file.read_text().strip():
                raise RuntimeError('ZooKeeper preparation produced no classpath')
            paths = ZkHelperOps._classpath_paths(cp_file)
            write_json(directory / 'manifest.json', {'files': ZkHelperOps._fingerprints(paths)})
        return directory

    @staticmethod
    def _classpath_paths(cp_file):
        target = FLEXLB_DIR / 'flexlb-sync' / 'target'
        return [target / 'test-classes', target / 'classes',
                *[Path(path) for path in cp_file.read_text().strip().split(os.pathsep)]]

    @staticmethod
    def _fingerprints(paths):
        from artifacts.archive import file_digest
        files = {}
        for path in paths:
            if not path.exists():
                raise RuntimeError('ZooKeeper classpath entry missing: ' + str(path))
            entries = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
            if not entries:
                raise RuntimeError('ZooKeeper classpath entry is empty: ' + str(path))
            for entry in entries:
                files[str(entry.resolve())] = file_digest(entry)
        return files

    @staticmethod
    def default_launch_argv(java_bin: str) -> list[str]:
        import json
        directory = FLEXLB_DIR / 'flexlb-sync' / 'target' / 'online-eval-runtime'
        cp_file, manifest = directory / 'classpath.txt', directory / 'manifest.json'
        if not cp_file.is_file() or not manifest.is_file():
            raise RuntimeError('ZooKeeper runtime is not prepared; run tools/online_eval/scripts/commands/prepare_runtime.py')
        paths = ZkHelperOps._classpath_paths(cp_file)
        if ZkHelperOps._fingerprints(paths) != json.loads(manifest.read_text())['files']:
            raise RuntimeError('ZooKeeper runtime changed after preparation; run prepare_runtime.py again')
        launcher = paths[0] / (ZK_LAUNCHER_CLASS.replace('.', '/') + '.class')
        if not launcher.is_file():
            raise RuntimeError('ZooKeeper launcher class missing: ' + str(launcher))
        cp = os.pathsep.join(map(str, paths))
        # --port 0: ZK auto-allocates a free port; the ZK_READY line
        # advertises the actual connectString (no port-collision risk
        # against sibling harness processes).
        return [java_bin, "-cp", cp, ZK_LAUNCHER_CLASS, "--port", "0"]
