# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Reads `*.autotune.json` files produced by Triton autotune (one per kernel
# per benchmark run) and writes per-kernel `default_config` JSON to
# autotune_cache/configs/{GPU}/.

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import triton

from rtp_llm.models_py.triton_kernels.autotune_cache.cache import KernelConfigFile


def _timing_value_key(value: object) -> float:
    """Sort key for picking the fastest config.

    Triton's `Autotuner._bench` calls `do_bench(..., quantiles=(0.5, 0.2, 0.8))`,
    which returns `[p50, p20, p80]` — we pick by p50 (median). The other
    quantiles are tie-breakers in theory but floats are effectively never
    equal in practice, so they don't matter. On Triton's failure path
    (OOM / compile error) the timing is `[inf, inf, inf]`, also handled.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    return float("inf")


def _normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Canonical shape for comparison and writing. Strips run-specific /
    Triton-version-specific fields so two configs that differ only in
    incidental metadata compare equal.
    """
    kwargs = cfg.get("kwargs") or {}
    return {
        "kwargs": copy.deepcopy(kwargs),
        "num_warps": cfg.get("num_warps"),
        "num_ctas": cfg.get("num_ctas", 1),
        "num_stages": cfg.get("num_stages"),
    }


@dataclass(frozen=True)
class WinnerSample:
    """Winner config extracted from one `*.autotune.json` produced by one
    benchmark run.
    """

    kernel_name: str
    source_file: str
    config: dict[str, Any]
    timing: Any

    @classmethod
    def from_file(cls, autotune_file: Path) -> "WinnerSample | None":
        try:
            with open(autotune_file) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {autotune_file}: {e}")
            return None
        if not isinstance(data, dict) or "configs_timings" not in data:
            return None
        configs_timings = data["configs_timings"]
        if not configs_timings:
            return None
        parts = autotune_file.stem.split(".")
        kernel_name = parts[0] if parts else "unknown_kernel"
        best = min(configs_timings, key=lambda e: _timing_value_key(e[1]))
        return cls(
            kernel_name=kernel_name,
            source_file=str(autotune_file),
            config=best[0],
            timing=best[1],
        )


def collect_winners(triton_cache_dir: Path) -> list[WinnerSample]:
    """Scan one Triton cache dir for `*.autotune.json`, return winner per file."""
    winners: list[WinnerSample] = []
    for f in triton_cache_dir.rglob("*.autotune.json"):
        w = WinnerSample.from_file(f)
        if w is not None:
            winners.append(w)
    return winners


def write_default_config_json(
    output_file: Path,
    kernel_name: str,
    default_config: dict[str, Any],
) -> str:
    """Write a minimal `{kernel}.json` containing only default_config.

    Returns "created", "updated", or "unchanged" based on existing disk state.
    """
    existing = KernelConfigFile.from_file(output_file) if output_file.exists() else None
    payload = {
        "kernel_name": kernel_name,
        "triton_version": triton.__version__,
        "default_config": _normalize_config(default_config),
    }
    if existing is not None and existing.default_config == payload["default_config"]:
        return "unchanged"
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    if existing is None:
        return "created"
    return "updated"
