#!/usr/bin/env python3
"""Validate smoke path wiring without Bazel-built .so or full pytest (no root conftest).

Run from repo with ``cwd == github-opensource/`` (needs **pydantic** — use CI venv or
``/opt/conda310/bin/python`` after ``pip install pydantic``).

Exit 0 if internal/OSS smoke layout and CI profile path resolution are consistent.
"""
from __future__ import annotations

import importlib.util
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

    assert internal_smoke.is_dir(), f"missing {internal_smoke}"
    assert (internal_smoke / "test_smoke_internal.py").is_file()
    assert (internal_smoke / "data").is_dir()
    assert (gho_smoke / "case_runner.py").is_file()
    assert (gho_smoke / "test_smoke_oss.py").is_file()

    sys.path.insert(0, str(gho / "rtp_llm" / "test"))
    from ci_profile_plugin import _resolve_profile_paths  # noqa: E402

    resolved = _resolve_profile_paths(
        gho, ["../internal_source/rtp_llm/test/smoke/test_smoke_internal.py"]
    )
    assert len(resolved) == 1
    assert Path(resolved[0]).resolve() == (
        internal_smoke / "test_smoke_internal.py"
    ).resolve()

    os.environ["SMOKE_REL_PATH_PREFER"] = "internal"
    cd_path = gho_smoke / "common_def.py"
    spec = importlib.util.spec_from_file_location("_smoke_common_def", cd_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rel = Path(mod.REL_PATH).resolve()
    assert rel == internal_smoke.resolve(), (rel, internal_smoke)

    os.environ["SMOKE_REL_PATH_PREFER"] = "oss"
    spec2 = importlib.util.spec_from_file_location("_smoke_common_def_oss", cd_path)
    mod2 = importlib.util.module_from_spec(spec2)
    assert spec2 and spec2.loader
    spec2.loader.exec_module(mod2)
    rel2 = Path(mod2.REL_PATH).resolve()
    assert rel2 == gho_smoke.resolve(), (rel2, gho_smoke)

    _warn_incomplete_rules_pkg_cache()

    print("verify_smoke_paths: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
