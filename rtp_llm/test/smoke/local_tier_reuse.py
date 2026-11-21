"""Local-tier variants share the existing real-model launch and numerical oracles."""

import os
from collections.abc import Mapping
from pathlib import Path
from shlex import quote
from typing import Literal

LocalCacheTier = Literal["device", "host", "disk"]


def local_cache_args(
    tier: LocalCacheTier, output_dir: Path, local_world_size: int = 1
) -> str:
    """Override only prefill cache policy, using an attempt-local disk path."""
    scratch_root = Path(os.environ.get("TEST_TMPDIR") or output_dir.parent)
    paths = [
        scratch_root / "precise_cache" / str(rank) for rank in range(local_world_size)
    ]
    if tier == "disk":
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
    options = {
        "device": "",
        "host": (
            " --enable_device_cache 0 --enable_memory_cache 1"
            " --enable_disk_cache 0 --enable_remote_cache 0 --memory_cache_size_mb 2048"
        ),
        "disk": (
            " --enable_device_cache 0 --enable_memory_cache 0"
            " --enable_disk_cache 1 --enable_remote_cache 0 --disk_cache_size_mb 2048"
            " --disk_cache_buffered_io 1 --disk_cache_paths "
            + quote(",".join(str(path) for path in paths))
        ),
    }
    return options[tier]


def assert_local_tier_reuse(
    tier: LocalCacheTier, aux: Mapping[str, int | bool]
) -> None:
    """Require the actual local medium to account for the fixture's exact reuse."""
    if tier == "device":
        return
    prefix = "prefill_" if aux["pd_sep"] else ""
    total_key = "prefill_total_reuse_len" if aux["pd_sep"] else "local_reuse_len"
    medium = {"host": "memory", "disk": "disk"}[tier]
    other = {"host": "disk", "disk": "memory"}[tier]
    assert aux[f"{prefix}{medium}_reuse_len"] == aux[total_key], aux
    assert aux[f"{prefix}{other}_reuse_len"] == 0, aux
