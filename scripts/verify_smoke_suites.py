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

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_FRAMEWORK_DIR = REPO_ROOT / "rtp_llm" / "test" / "smoke_framework"


EXPECTED_OSS_SUITE_COUNTS = {
    "cuda_remote_cache": 9,
    "h20_dense": 10,
    "h20_eagle": 6,
    "h20_grammar_heavy": 2,
    "h20_kimi_linear": 6,
    "h20_mla": 15,
    "h20_moe": 12,
    "h20_next": 17,
    "h20_vl": 5,
    "h20_jit_cache": 1,
    "rocm_basic": 4,
    "rocm_dense": 5,
    "rocm_eagle": 2,
    "rocm_jit_cache": 1,
    "rocm_qwen35_mrope": 2,
    "rocm_embedding": 9,
    "rocm_moe": 2,
    "rocm_pd": 1,
    "rocm_qwen35_mtp": 1,
    "sm100_dense": 3,
    "sm100_eval": 1,
    "sm100_moe": 9,
    "sm120_basic": 8,
    "sm8x_basic": 9,
}

EXPECTED_INTERNAL_SUITE_COUNTS = {
    "cuda13_arm": 5,
    "cuda13_x86": 2,
    "cuda13_flexlb": 2,
    "h20_dense_internal": 1,
    "ppu_basic": 8,
    "ppu_pd": 8,
    "ppu_qwen35": 6,
    "ppu_qwen35_w8a8_manual": 4,
    "rocm_embedding_internal": 2,
    "rocm_visionbert_internal": 1,
    "sm120_basic_internal": 1,
}

EXPECTED_OSS_PROFILE_COUNTS = {
    "smoke_h20_light_oss": 16,
    "smoke_h20_full_oss": 58,
    "smoke_sm8x_light_oss": 9,
    "smoke_sm8x_full_oss": 9,
    "smoke_rocm_oss": 26,
    "smoke_rocm_qwen35_mtp_manual": 1,
    "smoke_sm100_oss": 12,
    "smoke_sm100_eval_oss": 1,
    "smoke_sm120_oss": 8,
}

LIGHT_SUITES = {
    "h20_dense",
    "h20_eagle",
    "sm8x_basic",
    "sm120_basic",
}


def _oss_profile_owners(
    suite_name: str, case_name: str, config: Mapping[str, Any]
) -> List[str]:
    markers = set(config.get("markers", []))
    is_light = suite_name in LIGHT_SUITES
    owners: List[str] = []
    if "H20" in markers:
        owners.append("smoke_h20_light_oss" if is_light else "smoke_h20_full_oss")
    if "L20" in markers:
        if is_light:
            owners.append("smoke_sm8x_light_oss")
        if suite_name == "cuda_remote_cache":
            owners.append("smoke_sm8x_full_oss")
    if "MI308X_ROCM7" in markers:
        owners.append(
            "smoke_rocm_qwen35_mtp_manual"
            if "dedicated" in markers
            else "smoke_rocm_oss"
        )
    if "SM100_ARM" in markers:
        owners.append(
            "smoke_sm100_eval_oss"
            if "eval" in markers
            else "smoke_sm100_oss"
        )
    if "RTX_5000_PRO" in markers:
        owners.append("smoke_sm120_oss")
    return owners


def _validate_oss_profile_coverage(
    smoke_tests: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> List[str]:
    errors: List[str] = []
    profile_counts = {name: 0 for name in EXPECTED_OSS_PROFILE_COUNTS}
    for suite_name, cases in smoke_tests.items():
        for case_name, config in cases.items():
            owners = _oss_profile_owners(suite_name, case_name, config)
            if len(owners) != 1:
                errors.append(
                    f"[{suite_name}/{case_name}] expected exactly one automatic OSS "
                    f"Smoke profile owner, got {owners}"
                )
                continue
            profile_counts[owners[0]] += 1
    if profile_counts != EXPECTED_OSS_PROFILE_COUNTS:
        errors.append(
            "OSS Smoke profile counts changed without updating the parity baseline: "
            f"expected={EXPECTED_OSS_PROFILE_COUNTS}, actual={profile_counts}"
        )
    return errors


def _load_cases(path: Path) -> Mapping[str, Any]:
    """Parse a ``test_smoke_*.py`` file and return its SMOKE_CASES dict.

    Uses AST parsing so the script stays stdlib-only and does not execute
    the suite's top-level imports (pytest, torch, etc.).
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    constants: Dict[str, Any] = {}

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            if node.id not in constants:
                raise ValueError(f"unknown manifest constant {node.id!r}")
            return constants[node.id]
        if isinstance(node, ast.Dict):
            result: Dict[Any, Any] = {}
            for key_node, value_node in zip(node.keys, node.values):
                value = evaluate(value_node)
                if key_node is None:
                    if not isinstance(value, Mapping):
                        raise ValueError("manifest ** expansion must be a mapping")
                    result.update(value)
                else:
                    result[evaluate(key_node)] = value
            return result
        if isinstance(node, ast.List):
            return [evaluate(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(item) for item in node.elts)
        if isinstance(node, ast.Set):
            return {evaluate(item) for item in node.elts}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(left, (str, list, tuple)) and type(left) is type(right):
                return left + right
            raise ValueError(
                "manifest + operands must be matching strings, lists, or tuples"
            )
        if isinstance(node, ast.JoinedStr):
            parts: List[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                    continue
                if isinstance(value, ast.FormattedValue):
                    if value.conversion != -1 or value.format_spec is not None:
                        raise ValueError(
                            "manifest f-strings cannot use conversions or format specs"
                        )
                    formatted = evaluate(value.value)
                    if not isinstance(formatted, str):
                        raise ValueError(
                            "manifest f-string values must resolve to strings"
                        )
                    parts.append(formatted)
                    continue
                raise ValueError(
                    f"unsupported manifest f-string component {type(value).__name__}"
                )
            return "".join(parts)
        return ast.literal_eval(node)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "SMOKE_CASES":
                    return evaluate(node.value)
                if target.id.startswith("_"):
                    constants[target.id] = evaluate(node.value)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "SMOKE_CASES"
                and node.value is not None
            ):
                return evaluate(node.value)
    raise AttributeError(f"{path} missing SMOKE_CASES dict")


def _validate_dir(
    suites_dir: Path,
    data_root_dir: str,
    expected_suite_counts: Mapping[str, int] | None = None,
    validate_oss_profiles: bool = False,
) -> List[str]:
    """Find all test_smoke_*.py files under suites_dir and validate them."""
    sys.path.insert(0, str(SMOKE_FRAMEWORK_DIR))
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

    if expected_suite_counts is not None:
        actual_counts = {name: len(cases) for name, cases in smoke_tests.items()}
        if actual_counts != dict(expected_suite_counts):
            errors.append(
                "Smoke inventory changed without updating the parity baseline: "
                f"expected={dict(expected_suite_counts)}, actual={actual_counts}"
            )
        if validate_oss_profiles:
            errors.extend(_validate_oss_profile_coverage(smoke_tests))

    owners: Dict[str, str] = {}
    for suite_name, cases in smoke_tests.items():
        for case_name in cases:
            previous = owners.setdefault(case_name, suite_name)
            if previous != suite_name:
                errors.append(
                    f"duplicate case name {case_name!r} in {previous!r} and {suite_name!r}"
                )

    composite_suites = {
        "maga_model_smoke_full": list(smoke_tests.keys()),
        "maga_model_smoke_light": [],
    }
    errors.extend(
        validation.validate_manifest(smoke_tests, composite_suites, data_root_dir)
    )
    return errors


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-root",
        default=os.environ.get("RTP_INTERNAL_SOURCE_ROOT"),
        help=(
            "Path to the matching internal_source directory. Defaults to the "
            "repository sibling when present."
        ),
    )
    parser.add_argument(
        "--oss-only",
        action="store_true",
        help="Validate only OSS manifests; combined validation remains the default.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    gho = REPO_ROOT
    repo = gho.parent
    internal = (
        Path(args.internal_root).expanduser().resolve()
        if args.internal_root
        else repo / "internal_source"
    )

    sys.path.insert(0, str(gho / "rtp_llm" / "test" / "smoke_framework"))

    all_errors: List[str] = []

    # OSS suites
    oss_suites = gho / "rtp_llm" / "test" / "smoke" / "suites"
    if oss_suites.is_dir():
        oss_data_root = str(gho / "rtp_llm" / "test" / "smoke")
        oss_errors = _validate_dir(
            oss_suites,
            oss_data_root,
            EXPECTED_OSS_SUITE_COUNTS,
            validate_oss_profiles=True,
        )
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
    if not args.oss_only:
        if internal_suites.is_dir():
            internal_data_root = str(internal / "rtp_llm" / "test" / "smoke")
            internal_errors = _validate_dir(
                internal_suites,
                internal_data_root,
                EXPECTED_INTERNAL_SUITE_COUNTS,
            )
            if internal_errors:
                print(
                    f"=== internal smoke suites: {len(internal_errors)} error(s) ===",
                    file=sys.stderr,
                )
                for e in internal_errors:
                    print(f"  {e}", file=sys.stderr)
                all_errors.extend(internal_errors)
        elif args.internal_root:
            all_errors.append(
                f"missing internal smoke suites dir: {internal_suites}"
            )

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
