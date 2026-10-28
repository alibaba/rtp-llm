#!/usr/bin/env python3
"""Validate smoke path wiring without Bazel-built .so, pytest, or pydantic.

Run with ``cwd == github-opensource/``. Uses only stdlib + **tomli** (Python 3.10)
for ``ci_profile_support`` when reading ``pyproject.toml``.

Exit 0 if internal/OSS smoke layout and CI profile path resolution are consistent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _warn_incomplete_rules_pkg_cache() -> None:
    """Detect corrupted @rules_pkg checkouts (breaks havenask bundle.bzl analysis)."""
    home = Path.home()
    if not home.is_dir():
        return
    for cache_root in home.glob(".cache/bazel*_cache"):
        if not cache_root.is_dir():
            continue
        for rules_pkg in cache_root.glob("*/external/rules_pkg"):
            if not rules_pkg.is_dir():
                continue
            if (rules_pkg / "providers.bzl").is_file():
                continue
            print(
                "WARNING: incomplete Bazel @rules_pkg (missing providers.bzl): "
                f"{rules_pkg}\n"
                "  Fix: rm -rf that directory, or run `bazelisk fetch @rules_pkg//:providers.bzl` "
                "after updating WORKSPACE (mirror.bazel.build URL first).",
                file=sys.stderr,
            )


def main() -> int:
    gho = Path(__file__).resolve().parents[1]
    repo = gho.parent
    internal_smoke = repo / "internal_source" / "rtp_llm" / "test" / "smoke"
    gho_smoke = gho / "rtp_llm" / "test" / "smoke"

    # Suite split: each suite is its own test_<suite>.py + <suite>_cases.py
    # under suites/. The monolith test_smoke_oss.py / test_smoke_internal.py
    # files are gone — pyproject `paths` now points at the suites/ directory.
    assert internal_smoke.is_dir(), f"missing {internal_smoke}"
    assert (internal_smoke / "data").is_dir()
    assert (internal_smoke / "suites").is_dir(), "internal suites/ dir missing"
    assert (internal_smoke / "suites" / "conftest.py").is_file()
    assert (gho_smoke / "case_runner.py").is_file()
    assert (gho_smoke / "suites").is_dir(), "OSS suites/ dir missing"
    assert (gho_smoke / "suites" / "conftest.py").is_file()

    sys.path.insert(0, str(gho / "rtp_llm" / "test"))
    from ci_profile_support import resolve_profile_paths  # noqa: E402

    resolved = resolve_profile_paths(
        gho, ["../internal_source/rtp_llm/test/smoke/suites/"]
    )
    assert len(resolved) == 1
    assert Path(resolved[0]).resolve() == (internal_smoke / "suites").resolve()

    from smoke.rel_path_config import compute_smoke_rel_path  # noqa: E402

    rel = Path(compute_smoke_rel_path(str(gho_smoke), prefer="internal")).resolve()
    assert rel == internal_smoke.resolve(), (rel, internal_smoke)

    rel2 = Path(compute_smoke_rel_path(str(gho_smoke), prefer="oss")).resolve()
    assert rel2 == gho_smoke.resolve(), (rel2, gho_smoke)

    _warn_incomplete_rules_pkg_cache()

    print("verify_smoke_paths: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
