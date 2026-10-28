"""Helpers for resolving ``--rtp-ci-profile`` paths from pyproject.toml.

No **pytest** dependency — safe for ``scripts/verify_smoke_paths.py`` and minimal envs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_toml(path: Path) -> Dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[import-not-found]

    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def find_pyproject_with_ci_profiles(start: Path) -> Optional[Path]:
    """Resolve pyproject.toml that contains ``[tool.rtp_llm.pytest_ci].profiles``."""
    cur = start.resolve()
    for base in (cur, *cur.parents):
        candidates = [
            base / "pyproject.toml",
            base / "github-opensource" / "pyproject.toml",
            base / "internal_source" / "pyproject.toml",
            base / "open_source" / "pyproject.toml",
        ]
        for pp in candidates:
            if not pp.is_file():
                continue
            try:
                data = load_toml(pp)
            except Exception:
                continue
            profiles = (
                data.get("tool", {})
                .get("rtp_llm", {})
                .get("pytest_ci", {})
                .get("profiles")
            )
            if isinstance(profiles, dict):
                return pp
    return None


def load_pytest_ci_config(start: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Load pytest_ci config from pyproject.toml plus local monorepo overlay."""
    pyproject = find_pyproject_with_ci_profiles(start)
    if pyproject is None:
        return None, {}
    data = load_toml(pyproject)
    base_dir = pyproject.parent
    for overlay in (
        base_dir / "internal_source" / "pyproject_internal.toml",
        base_dir.parent / "internal_source" / "pyproject_internal.toml",
    ):
        if not overlay.is_file():
            continue
        _deep_merge(data, load_toml(overlay))
        break
    return pyproject, data.get("tool", {}).get("rtp_llm", {}).get("pytest_ci", {}) or {}


def resolve_profile_paths(repo_root: Path, paths: list[str]) -> list[str]:
    """Resolve profile ``paths`` entries relative to the pyproject.toml parent directory.

    CI runs with ``cwd == github-opensource/``; internal smoke uses
    ``../internal_source/...`` without requiring a symlink under ``github-opensource/``.
    """
    pyproject = find_pyproject_with_ci_profiles(repo_root)
    base = pyproject.parent if pyproject is not None else repo_root
    return [str((base / p).resolve()) for p in paths]
