"""Read-only source freezing and verified native-build identities."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from rtp_llm.test.perf_test.errors import ConfigurationError


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ConfigurationError(f"cannot resolve SSH source: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_source(source: Mapping[str, Any], *, source_dir: Path) -> dict:
    """Freeze local Git input before any SSH side effect; never fetch a branch."""
    result = dict(source)
    configured = result.get("worktree")
    root = Path(configured) if configured else Path(__file__).resolve().parents[4]
    if not root.is_absolute():
        root = Path(source_dir) / root
    root = Path(_git(root.resolve(), "rev-parse", "--show-toplevel")).resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ConfigurationError(
            "SSH source contains uncommitted tracked changes; commit them before running"
        )
    ref = result.get("ref", "auto")
    ref = "HEAD" if ref == "auto" else ref
    if not isinstance(ref, str) or not ref or ref.startswith("-"):
        raise ConfigurationError("SSH source.ref must be a local Git reference")
    result["ref"] = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    result["worktree"] = str(root)
    return result


def build_identity(source: Mapping[str, Any]) -> str:
    fields = {
        key: source.get(key)
        for key in ("ref", "bazel_config", "bazel_targets", "bazel_args")
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
