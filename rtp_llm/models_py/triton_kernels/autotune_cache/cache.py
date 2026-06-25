# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Generic Triton autotune cache infrastructure for rtp-llm. Replaces
# `triton.autotune` with a `CachedAutotuner` that consults checked-in
# per-kernel JSON configs under `autotune_cache/configs/{GPU}/` before
# falling back to Triton's normal benchmark-and-pick autotune.
#
# Each per-kernel JSON file holds a single `default_config` block that is
# installed as the winner.
#
# Environment variables:
#   TRITON_AUTOTUNE_CACHE_MODE     - "disabled" | "cached"; default "disabled"
#   TRITON_AUTOTUNE_CONFIG_DIR     - override JSON root; default <this_dir>/configs/{GPU}/
#   TRITON_AUTOTUNE_GPU_NAME       - override GPU model id used for path lookup

import dataclasses
import enum
import inspect
import json
import logging
import os
import re
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import torch
import triton
from packaging import version
from triton.runtime.autotuner import Autotuner

TRITON_ABOVE_3_5_1 = version.parse(triton.__version__) >= version.parse("3.5.1")


logger = logging.getLogger(__name__)


# Detect whether triton.autotune accepts a `cache_results` kwarg (added in
# Triton 3.4).
try:
    SUPPORTS_AUTOTUNE_CACHE = (
        "cache_results" in inspect.signature(triton.autotune).parameters
    )
except Exception:
    SUPPORTS_AUTOTUNE_CACHE = False


# Spread `**autotune_cache_kwargs` at the autotune call site to opt in to
# Triton's on-disk benchmark caching when available (Triton >= 3.4); no-op
# on older Triton. Hardcoded on — debug-time invalidation goes via
# `TRITON_CACHE_DIR` rather than a separate env knob.
autotune_cache_kwargs = {"cache_results": True} if SUPPORTS_AUTOTUNE_CACHE else {}


class CacheMode(enum.Enum):
    """How the cache loads kernel configs (TRITON_AUTOTUNE_CACHE_MODE env var).

    DISABLED - skip all cache lookups, always fall back to Triton autotune
               (default when the env var is unset).
    CACHED   - use the top-level `default_config` of the kernel JSON file.
               If the file is missing or has no default_config, fall back
               to Triton autotune. CI default.
    """

    DISABLED = "disabled"
    CACHED = "cached"

    @classmethod
    def from_env(cls) -> "CacheMode":
        mode_str = os.environ.get("TRITON_AUTOTUNE_CACHE_MODE", cls.DISABLED.value)
        try:
            return cls(mode_str)
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(
                f"Invalid TRITON_AUTOTUNE_CACHE_MODE={mode_str!r}. "
                f"Valid values: {valid}"
            ) from None


@lru_cache(maxsize=1)
def _get_cache_mode() -> CacheMode:
    return CacheMode.from_env()


def sanitize_gpu_name(gpu_name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", gpu_name)
    sanitized = sanitized.strip("_")
    return sanitized or "unknown_gpu"


@lru_cache(maxsize=1)
def get_gpu_info() -> str:
    """Get GPU model identifier (sanitized).

    Priority: TRITON_AUTOTUNE_GPU_NAME env var > torch.cuda. CUDA-only —
    other backends are not in scope for this cache. Falls back to "unknown"
    so the module still imports on a host without a usable GPU (tests, dev
    machines); the kernel-side decorator no-ops in that case.
    """
    gpu_name = os.environ.get("TRITON_AUTOTUNE_GPU_NAME")
    if gpu_name is None and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    if gpu_name:
        return sanitize_gpu_name(gpu_name)
    return "unknown"


def get_config_dir() -> Path:
    """Triton autotune config directory.

    Override with `TRITON_AUTOTUNE_CONFIG_DIR`. Default is the
    `configs/{GPU}/` subdirectory next to this file.
    """
    override = os.environ.get("TRITON_AUTOTUNE_CONFIG_DIR")
    if override is not None and override != "__builtin__":
        return Path(override)
    return Path(__file__).parent / "configs" / get_gpu_info()


@dataclasses.dataclass(frozen=True)
class KernelConfigFile:
    """Validated in-memory representation of a {kernel_name}.json config file."""

    kernel_name: str | None
    triton_version: str | None
    default_config: dict[str, Any] | None

    @classmethod
    def from_dict(cls, config_file: Path, data: Any) -> "KernelConfigFile | None":
        """Parse and validate a raw JSON dict. Returns None (with a warning)
        if malformed.
        """
        if not isinstance(data, dict):
            logger.warning(
                "Malformed config %s: root is %s, expected dict",
                config_file,
                type(data).__name__,
            )
            return None
        default_config = data.get("default_config")
        if default_config is not None and not isinstance(default_config, dict):
            logger.warning(
                "Malformed config %s: 'default_config' is %s, expected dict",
                config_file,
                type(default_config).__name__,
            )
            return None
        return cls(
            kernel_name=data.get("kernel_name"),
            triton_version=data.get("triton_version"),
            default_config=default_config,
        )

    @classmethod
    def from_file(cls, config_file: Path) -> "KernelConfigFile | None":
        config_data = load_config_file(config_file)
        if config_data is None:
            return None
        return cls.from_dict(config_file, config_data)


@cache
def load_config_file(config_file: Path) -> dict[str, Any] | None:
    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Error reading config file %s: %s", config_file, e)
        return None


def load_cached_default_config(kernel_name: str) -> dict[str, Any] | None:
    """Return the `default_config` dict for a kernel, or None.

    Returns None when: cache mode is disabled, file is missing/malformed,
    or the file has no `default_config` field.
    """
    if _get_cache_mode() is CacheMode.DISABLED:
        return None

    config_file = get_config_dir() / f"{kernel_name}.json"
    if not config_file.exists():
        return None

    config = KernelConfigFile.from_file(config_file)
    if config is None:
        return None
    return config.default_config


def _build_triton_config(cfg: dict[str, Any]) -> triton.Config:
    extra = (
        {
            "num_ctas": cfg.get("num_ctas", 1),
            "maxnreg": cfg.get("maxnreg"),
            "pre_hook": None,
            "ir_override": cfg.get("ir_override"),
        }
        if TRITON_ABOVE_3_5_1
        else {}
    )
    return triton.Config(
        cfg["kwargs"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **extra,
    )


class CachedAutotuner(Autotuner):
    """Triton Autotuner subclass that consults checked-in JSON configs first.

    On the first call (in CACHED mode), if a JSON `default_config` is
    available, replaces `self.configs` with `[default_cfg]`. Triton's own
    `Autotuner.run()` short-circuits when `len(self.configs) == 1`: no key
    building, no benchmark, just returns `self.configs[0]`. This keeps us
    decoupled from Triton's internal key-extraction logic.

    If CACHED mode is active but no JSON / no default_config is available,
    behavior degrades to plain triton.autotune (benchmark-and-pick over the
    original config list).
    """

    def __init__(
        self, fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs
    ):
        super().__init__(
            fn, arg_names, configs, key, reset_to_zero, restore_value, **kwargs
        )
        self.kernel_name = fn.fn.__name__ if hasattr(fn, "fn") else fn.__name__
        self._default_cfg_resolved = False

    def _resolve_default_cfg_once(self) -> None:
        if self._default_cfg_resolved:
            return
        self._default_cfg_resolved = True
        raw = load_cached_default_config(self.kernel_name)
        if raw is None:
            logger.info(
                "No cached config for %s; falling back to Triton autotune",
                self.kernel_name,
            )
            return
        try:
            self.configs = [_build_triton_config(raw)]
            logger.info(
                "Loaded cached default_config for %s: %s",
                self.kernel_name,
                raw,
            )
        except Exception as e:
            logger.warning(
                "Failed to materialize default_config for %s: %s; "
                "falling back to Triton autotune",
                self.kernel_name,
                e,
            )

    def run(self, *args, **kwargs):
        if _get_cache_mode() is CacheMode.CACHED:
            self._resolve_default_cfg_once()
        return super().run(*args, **kwargs)


def cached_autotune(
    configs,
    key=None,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
    cache_results=False,
):
    """Decorator for auto-tuning a `triton.jit`'d function with checked-in
    JSON config support.

    Extends `triton.autotune` to load `default_config` from per-kernel JSON
    under `autotune_cache/configs/{GPU}/` (overridable via
    `TRITON_AUTOTUNE_CONFIG_DIR`). Lookup is gated by the
    `TRITON_AUTOTUNE_CACHE_MODE` env var. When disabled or when no config
    is available, behavior is identical to plain `triton.autotune`.
    """
    if key is None:
        key = []

    def decorator(fn):
        kwargs = {}
        if SUPPORTS_AUTOTUNE_CACHE:
            kwargs = {"cache_results": cache_results}

        return CachedAutotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            do_bench=do_bench,
            **kwargs,
        )

    return decorator
