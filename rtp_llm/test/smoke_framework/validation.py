"""Smoke manifest validators — stdlib-only so verify_smoke_suites can run pre-build.

Checks:
- Each case's `gpu_count` (derived from smoke_args world_size) is internally consistent.
- `markers` only contains known marker names.
- `light` and `full` composite suites are disjoint where appropriate.
- `task_info` files exist under the case's expected data tree.

Errors are accumulated and returned as a list — callers print + exit non-zero.
"""

from __future__ import annotations

import os
from typing import Any, List, Mapping

# Import via flat module name when this file is imported by verify_smoke_suites
# (which mounts smoke_framework/ on sys.path directly). When imported from a
# normal pytest run, the rtp_llm.test.smoke_framework path also works.
try:
    from manifest import _parse_world_size  # type: ignore[import-not-found]
except ImportError:
    from rtp_llm.test.smoke_framework.manifest import _parse_world_size

# Closed set of common markers emitted by smoke_defs_*.
_KNOWN_MARKERS = {
    # General smoke / function flavors
    "smoke",
    "manual",
    "light",
    "full",
    "remote_cache",
    "perf",
    "next",
    "eagle",
    "moe",
    "mla",
    "dense",
    "vit",
    "embedding",
    "PD",
    "eval",
    # Platform families
    "cuda",
    "rocm",
    # GPU SKU markers (drive REAPI platform.properties via pytest.mark.gpu(type=...))
    "H20",
    "H100",
    "A100",
    "L20",
    "SM100_ARM",
    "MI300X",
    "MI308X",
    "MI308X_ROCM7",
}


def _is_platform_marker(marker: str) -> bool:
    return marker.isidentifier() and marker.upper() == marker and any(
        ch.isdigit() for ch in marker
    )


def validate_case(
    suite_name: str,
    case_name: str,
    config: Mapping[str, Any],
    data_root_dir: str,
) -> List[str]:
    """Return list of validation error strings for one case (empty = OK)."""
    errors: List[str] = []
    prefix = f"[{suite_name}/{case_name}]"

    smoke_args = config.get("smoke_args", "")
    envs = config.get("envs", [])
    # Empty smoke_args is allowed iff envs declares WORLD_SIZE/TP_SIZE
    # (some cases use env-driven config instead of CLI flags — e.g.,
    # LOAD_PYTHON_MODEL=1 paths that read TP_SIZE from env).
    if not smoke_args:
        env_strs = envs if isinstance(envs, list) else []
        for d in envs.values() if isinstance(envs, dict) else ():
            env_strs.extend(d)
        joined = " ".join(env_strs)
        if "WORLD_SIZE" not in joined and "TP_SIZE" not in joined:
            errors.append(
                f"{prefix} smoke_args empty AND envs lacks WORLD_SIZE/TP_SIZE"
            )

    if isinstance(smoke_args, dict):
        derived = sum(_parse_world_size(s) for s in smoke_args.values())
    else:
        derived = _parse_world_size(smoke_args)

    declared = config.get("gpu_count")
    if declared is not None and int(declared) != derived:
        errors.append(
            f"{prefix} gpu_count={declared} disagrees with parsed world_size={derived}"
        )

    for marker in config.get("markers", []):
        if marker not in _KNOWN_MARKERS and not _is_platform_marker(str(marker)):
            errors.append(f"{prefix} unknown marker {marker!r} (not in _KNOWN_MARKERS)")

    task_info_rel = config.get("task_info", "")
    if task_info_rel:
        full = os.path.join(data_root_dir, task_info_rel)
        if not os.path.isfile(full):
            errors.append(f"{prefix} task_info missing on disk: {full}")

    return errors


def validate_manifest(
    smoke_tests: Mapping[str, Mapping[str, Mapping[str, Any]]],
    composite_suites: Mapping[str, List[str]],
    data_root_dir: str,
) -> List[str]:
    """Validate full manifest. Returns combined error list (empty = OK)."""
    errors: List[str] = []

    light = set(composite_suites.get("maga_model_smoke_light", []))
    full = set(composite_suites.get("maga_model_smoke_full", []))
    overlap = light & full
    if overlap:
        errors.append(f"light ∩ full = {sorted(overlap)} (composite suites overlap)")

    declared_suites = set(smoke_tests.keys())
    referenced_suites = light | full
    missing = referenced_suites - declared_suites
    if missing:
        errors.append(
            f"composite suites reference unknown suite names: {sorted(missing)}"
        )

    for suite_name, suite in smoke_tests.items():
        if not isinstance(suite, Mapping):
            errors.append(
                f"[{suite_name}] suite must be dict-of-cases, got {type(suite).__name__}"
            )
            continue
        for case_name, config in suite.items():
            errors.extend(validate_case(suite_name, case_name, config, data_root_dir))

    return errors
