"""Mock launch inputs: effective engine behavior, calibration, process runtime."""

import json
from pathlib import Path


def reject_removed_environment(env):
    removed = {
        "MOCK_PERFORMANCE_CONFIG_JSON",
        "MOCK_BLOCK_SIZE",
        "MOCK_PREFILL_BLOCK_SIZE",
        "MOCK_DECODE_BLOCK_SIZE",
        "MOCK_PREFILL_KV_POOL_BLOCKS",
        "MOCK_DECODE_KV_POOL_BLOCKS",
        "MOCK_DECODE_MAX_CONCURRENCY",
        "MOCK_VIT_CACHE_BYTES",
        "VIT_MAX_CONCURRENT_REQUESTS",
        "MOCK_HASH_BLOCK_SIZE",
        "MOCK_VIT_ENABLED",
        "MOCK_VIT_PORT",
        "MOCK_VIT_HIDDEN_SIZE",
        "MOCK_VIT_CONCURRENCY",
        "MOCK_VIT_QUEUE_CAPACITY",
        "MOCK_VIT_EMBEDDING_CACHE_BYTES",
        "MOCK_VIT_HASH_CACHE_BYTES",
        "MOCK_VIT_COMPUTE_DELAY_MS",
    }
    conflicts = removed.intersection(env)
    if conflicts:
        raise ValueError(f"removed configuration inputs: {sorted(conflicts)}; use MOCK_ENGINE_CONFIG_JSON and MOCK_CALIBRATION_CONFIG_JSON")


def engine_input(env, default=None):
    reject_removed_environment(env)
    if "MOCK_ENGINE_CONFIG_JSON" in env:
        value = json.loads(env["MOCK_ENGINE_CONFIG_JSON"])
    elif default is not None:
        value = json.loads(Path(default).read_text())
    else:
        raise ValueError("MOCK_ENGINE_CONFIG_JSON is required")
    if not isinstance(value, dict):
        raise ValueError("engine config must be an object")
    return value


def calibration_input(env, default=None):
    import sys

    root = Path(__file__).resolve().parent
    for directory in (root.parent / "online_eval", root.parent / "online_eval/src"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from adapters.perf_presets import resolve_performance_calibration

    path = Path(default) if default is not None else root / "performance.json"
    encoded = env.get("MOCK_CALIBRATION_CONFIG_JSON")
    value = json.loads(path.read_text() if encoded is None else encoded)
    if not isinstance(value, dict):
        raise ValueError("MOCK_CALIBRATION_CONFIG_JSON must be an object")
    return resolve_performance_calibration(value, path)
