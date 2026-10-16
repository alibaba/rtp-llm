"""Resolve Java 21 and define common JVM module options."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


JAVA_MODULE_OPTS = [
    "--add-modules",
    "ALL-SYSTEM",
    "--add-opens",
    "java.base/java.lang=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.util=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.nio=ALL-UNNAMED",
    "--add-opens",
    "java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens",
    "java.instrument/sun.instrument=ALL-UNNAMED",
]

_java21_cache: Optional[str] = None


def _java_major(java_bin: str) -> int:
    try:
        out = subprocess.run(
            [java_bin, "-version"], capture_output=True, text=True, timeout=30
        ).stderr
    except Exception:
        return 0
    import re as _re

    m = _re.search(r'version "(\d+)', out)
    return int(m.group(1)) if m else 0


def resolve_java21() -> str:
    """Resolve a JDK >= 21 binary path (JAVA_HOME → JAVA21_HOME → ~/java21 → homebrew)."""
    global _java21_cache
    if _java21_cache:
        return _java21_cache
    candidates = []
    for var in ("JAVA_HOME", "JAVA21_HOME"):
        home = os.environ.get(var)
        if home:
            candidates.append(f"{home}/bin/java")
    candidates.append(f"{Path.home()}/java21/bin/java")
    candidates.append("/opt/homebrew/opt/openjdk@21/bin/java")
    path_java = shutil.which("java")
    if path_java:
        candidates.append(path_java)
    for cand in candidates:
        if (
            os.path.isfile(cand)
            and os.access(cand, os.X_OK)
            and _java_major(cand) >= 21
        ):
            _java21_cache = cand
            return cand
    raise RuntimeError(
        "Java 21+ is required (set JAVA_HOME/JAVA21_HOME). Tried: "
        + ", ".join(candidates)
    )
