#!/usr/bin/env python3
"""Validate smoke test manifests after per-suite split.

Each suite is its own `<suite>_cases.py` file under `suites/`. This
script discovers them, loads their SMOKE_CASES dict, and runs the same checks
verify_smoke_suites used to apply to the monolith SMOKE_TESTS dict:

- gpu_count vs smoke_args world_size mismatch
- Unknown markers
- Missing task_info JSON files

stdlib-only — runs in CI prepare-source before any .so build.

Exit 0 on success, 1 on errors.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _load_cases(path: Path) -> Mapping[str, Any]:
    """Import a `<suite>_cases.py` file and return its SMOKE_CASES dict."""
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = getattr(mod, "SMOKE_CASES", None)
    if cases is None:
        raise AttributeError(f"{path} missing SMOKE_CASES dict")
    return cases


def _validate_dir(suites_dir: Path, data_root_dir: str) -> List[str]:
    """Find all test_smoke_*.py files under suites_dir and validate them."""
    sys.path.insert(0, str(suites_dir.parents[2] / "smoke_framework"))
    import validation  # type: ignore[import-not-found]

    errors: List[str] = []
    smoke_tests: Dict[str, Mapping[str, Any]] = {}
    case_files = sorted(suites_dir.glob("test_smoke_*.py"))
    if not case_files:
        errors.append(f"no test_smoke_*.py files found in {suites_dir}")
        return errors

    for cases_path in case_files:
        # test_smoke_<suite>.py -> <suite>
        suite_name = cases_path.stem.removeprefix("test_smoke_")
        try:
            smoke_tests[suite_name] = _load_cases(cases_path)
        except Exception as e:
            errors.append(f"[{suite_name}] failed to load: {e}")

    composite_suites = {
        "maga_model_smoke_full": list(smoke_tests.keys()),
        "maga_model_smoke_light": [],
    }
    errors.extend(
        validation.validate_manifest(smoke_tests, composite_suites, data_root_dir)
    )
    return errors


def main() -> int:
    gho = Path(__file__).resolve().parents[1]
    repo = gho.parent
    internal = repo / "internal_source"

    sys.path.insert(0, str(gho / "rtp_llm" / "test" / "smoke_framework"))

    all_errors: List[str] = []

    # OSS suites
    oss_suites = gho / "rtp_llm" / "test" / "smoke" / "suites"
    if oss_suites.is_dir():
        oss_data_root = str(gho / "rtp_llm" / "test" / "smoke")
        oss_errors = _validate_dir(oss_suites, oss_data_root)
        if oss_errors:
            print(
                f"=== OSS smoke suites: {len(oss_errors)} error(s) ===", file=sys.stderr
            )
            for e in oss_errors:
                print(f"  {e}", file=sys.stderr)
            all_errors.extend(oss_errors)
    else:
        all_errors.append(f"missing OSS suites dir: {oss_suites}")

    # Internal suites (only if internal_source is present)
    internal_suites = internal / "rtp_llm" / "test" / "smoke" / "suites"
    if internal_suites.is_dir():
        internal_data_root = str(internal / "rtp_llm" / "test" / "smoke")
        internal_errors = _validate_dir(internal_suites, internal_data_root)
        if internal_errors:
            print(
                f"=== internal smoke suites: {len(internal_errors)} error(s) ===",
                file=sys.stderr,
            )
            for e in internal_errors:
                print(f"  {e}", file=sys.stderr)
            all_errors.extend(internal_errors)

    if all_errors:
        print(
            f"verify_smoke_suites: FAILED ({len(all_errors)} error(s))",
            file=sys.stderr,
        )
        return 1

    print("verify_smoke_suites: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
