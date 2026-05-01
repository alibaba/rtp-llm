"""Manifest helpers — pure-data parsers + pytest parametrize builder.

Replaces the duplicate copies of these helpers in `smoke_defs_oss.py` /
`smoke_defs_internal.py`. Data files now contain only `SMOKE_TESTS` and
`COMPOSITE_SUITES` dicts; they call into this module to build pytest params.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


def _parse_world_size(args_str: str) -> int:
    """Parse --tp_size, --dp_size, --pp_size, --world_size from an args string."""
    if not args_str:
        return 1
    parts = args_str.split()
    tp = pp = dp = 1
    world_size: Optional[int] = None
    i = 0
    while i < len(parts):
        if parts[i] == "--world_size" and i + 1 < len(parts):
            world_size = int(parts[i + 1])
            i += 2
            continue
        if parts[i] == "--tp_size" and i + 1 < len(parts):
            tp = int(parts[i + 1])
            i += 2
            continue
        if parts[i] == "--dp_size" and i + 1 < len(parts):
            dp = int(parts[i + 1])
            i += 2
            continue
        if parts[i] == "--pp_size" and i + 1 < len(parts):
            pp = int(parts[i + 1])
            i += 2
            continue
        i += 1
    return world_size if world_size is not None else tp * pp * dp


def get_gpu_count(config: Mapping[str, Any]) -> int:
    """Calculate GPU count from a case config's smoke_args.

    For multi-role dict smoke_args (PD-separation), sums world_size across roles.
    """
    smoke_args = config.get("smoke_args", "")
    if isinstance(smoke_args, dict):
        return sum(_parse_world_size(role_args) for role_args in smoke_args.values())
    return _parse_world_size(smoke_args)


def get_all_suites(smoke_tests: Mapping[str, Mapping[str, Any]]) -> List[str]:
    return list(smoke_tests.keys())


def get_tests_in_suite(
    suite_name: str,
    smoke_tests: Mapping[str, Mapping[str, Any]],
    composite_suites: Mapping[str, List[str]],
) -> List[str]:
    if suite_name in smoke_tests:
        return list(smoke_tests[suite_name].keys())
    if suite_name in composite_suites:
        tests: List[str] = []
        for s in composite_suites[suite_name]:
            if s in smoke_tests:
                tests.extend(smoke_tests[s].keys())
        return tests
    return []


def get_tests_for_platform(
    platform: str, smoke_tests: Mapping[str, Mapping[str, Any]]
) -> List[Tuple[str, Mapping[str, Any]]]:
    result: List[Tuple[str, Mapping[str, Any]]] = []
    for suite in smoke_tests.values():
        for name, config in suite.items():
            if config.get("platform") == platform:
                result.append((name, config))
    return result


# Test/suite names that should also receive the `remote_cache` pytest marker.
# Kept as a small set so the OSS smoke profile `smoke_remote_cache_oss`
# (markexpr: `remote_cache`) still picks them up after the helpers move here.
# Update this set when adding new remote-cache cases.
_REMOTE_CACHE_SUITES = frozenset({"smoke_cuda_remote_cache"})
_REMOTE_CACHE_TESTS = frozenset({"next_long_reuse_remote", "eagle_remote_cache_tp2"})


def build_smoke_params(
    pytest_module,
    smoke_tests: Mapping[str, Mapping[str, Mapping[str, Any]]],
    composite_suites: Mapping[str, List[str]],
):
    """Build a list of pytest.param(...) entries from SMOKE_TESTS / COMPOSITE_SUITES.

    Iteration order = data dict order, so case nodeids are stable across calls.
    """
    light_suites = composite_suites.get("maga_model_smoke_light", [])
    params = []
    for suite_name, suite in smoke_tests.items():
        for test_name, config in suite.items():
            marks = []
            for marker_name in config.get("markers", []):
                marks.append(getattr(pytest_module.mark, marker_name))
            marks.append(pytest_module.mark.manual)

            if suite_name in _REMOTE_CACHE_SUITES or test_name in _REMOTE_CACHE_TESTS:
                marks.append(pytest_module.mark.remote_cache)

            if suite_name in light_suites:
                marks.append(pytest_module.mark.light)

            gpu_type = config.get("gpu_type", "gpu_cuda12")
            gpu_count = get_gpu_count(config)
            marks.append(pytest_module.mark.gpu(type=gpu_type, count=gpu_count))

            params.append(
                pytest_module.param(test_name, config, id=test_name, marks=marks)
            )
    return params
