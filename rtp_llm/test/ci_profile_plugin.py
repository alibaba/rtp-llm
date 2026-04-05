"""pytest plugin: apply CI marker/path presets from pyproject.toml.

Reads [tool.rtp_llm.pytest_ci].profiles.<name> from the nearest pyproject.toml
(github-opensource root or open_source/pyproject.toml).

Use --rtp-ci-profile=NAME.
Can be combined with -m; final markexpr is:
    (<profile markexpr>) and (<user -m markexpr>)

When a profile is active, [tool.rtp_llm.pytest_ci].default_pytest_cli is applied:
sets PYTEST_ARGS for --remote-session workers and updates local pytest options
unless PYTEST_ARGS is already set in the environment.
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


def _load_toml(path: Path) -> Dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[import-not-found]
    with path.open("rb") as f:
        return tomllib.load(f)


def _find_pyproject(start: Path) -> Optional[Path]:
    """Resolve pyproject.toml that contains CI profiles."""
    cur = start.resolve()
    for base in (cur, *cur.parents):
        candidates = [
            base / "pyproject.toml",
            base / "internal_source" / "pyproject.toml",
            base / "open_source" / "pyproject.toml",
        ]
        for pp in candidates:
            if not pp.is_file():
                continue
            try:
                data = _load_toml(pp)
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


def _get_profile(root: Path, name: str) -> Dict[str, Any]:
    pyproject = _find_pyproject(root)
    if pyproject is None:
        raise pytest.UsageError(
            f"--rtp-ci-profile: no pyproject.toml found (searched from {root})"
        )
    data = _load_toml(pyproject)
    try:
        profiles = data["tool"]["rtp_llm"]["pytest_ci"]["profiles"]
    except KeyError as exc:
        raise pytest.UsageError(
            f"--rtp-ci-profile: missing [tool.rtp_llm.pytest_ci].profiles in {pyproject}"
        ) from exc
    if name not in profiles:
        known = ", ".join(sorted(profiles.keys()))
        raise pytest.UsageError(
            f"--rtp-ci-profile: unknown profile {name!r}. Known: {known}"
        )
    prof = profiles[name]
    if not isinstance(prof, dict):
        raise pytest.UsageError(f"--rtp-ci-profile: profile {name!r} must be a table")
    if "markexpr" not in prof:
        raise pytest.UsageError(f"--rtp-ci-profile: profile {name!r} missing 'markexpr'")
    return prof


def _get_pytest_ci_section(root: Path) -> Dict[str, Any]:
    pyproject = _find_pyproject(root)
    if pyproject is None:
        return {}
    data = _load_toml(pyproject)
    return data.get("tool", {}).get("rtp_llm", {}).get("pytest_ci", {}) or {}


def _apply_default_cli(config: pytest.Config, cli: str) -> None:
    """Apply -v / --tb= / --timeout= to config (best-effort for pytest-timeout)."""
    for tok in shlex.split(cli):
        if tok == "-v":
            config.option.verbose = max(getattr(config.option, "verbose", 0), 1)
        elif tok == "-vv":
            config.option.verbose = max(getattr(config.option, "verbose", 0), 2)
        elif tok.startswith("--tb="):
            config.option.tbstyle = tok.split("=", 1)[1]
        elif tok.startswith("--timeout="):
            val = int(tok.split("=", 1)[1])
            if hasattr(config.option, "timeout"):
                setattr(config.option, "timeout", val)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rtp-ci-profile", "RTP-LLM CI profiles (pyproject.toml)")
    group.addoption(
        "--rtp-ci-profile",
        action="store",
        default=None,
        metavar="NAME",
        help=(
            "Load markexpr/paths from [tool.rtp_llm.pytest_ci].profiles.NAME "
            "in pyproject.toml"
        ),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    name = config.getoption("--rtp-ci-profile")
    if not name:
        return

    root = Path(config.rootpath)
    pytest_ci = _get_pytest_ci_section(root)
    default_cli = (pytest_ci.get("default_pytest_cli") or "-v --tb=short --timeout=300").strip()
    if default_cli:
        _apply_default_cli(config, default_cli)

    prof = _get_profile(root, name)
    markexpr = prof["markexpr"]
    if not isinstance(markexpr, str) or not markexpr.strip():
        raise pytest.UsageError(f"--rtp-ci-profile: profile {name!r} has empty markexpr")
    profile_markexpr = markexpr.strip()
    user_markexpr = (getattr(config.option, "markexpr", None) or "").strip()
    if user_markexpr:
        config.option.markexpr = f"({profile_markexpr}) and ({user_markexpr})"
    else:
        config.option.markexpr = profile_markexpr

    paths = prof.get("paths")
    if paths is not None:
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise pytest.UsageError(
                f"--rtp-ci-profile: profile {name!r} 'paths' must be a list of strings"
            )
        # Restrict collection roots (e.g. smoke file or frontend dirs only)
        config.args[:] = list(paths)
