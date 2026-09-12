"""Opt-in configuration for the local DSV4 CSA offload experiment."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CsaOffloadConfig:
    gpu_cache_mib: int
    logical_blocks: int
    hot_entries: int = 2048
    fetch_ctas: int = 64

    @classmethod
    def from_env(cls):
        enabled = os.environ.get("DSV4_CSA_OFFLOAD", "0")
        if enabled == "0":
            return None
        if enabled != "1":
            raise ValueError("DSV4_CSA_OFFLOAD must be 0 or 1")
        config = cls(
            gpu_cache_mib=int(os.environ.get("DSV4_CSA_GPU_CACHE_MIB", "32768")),
            logical_blocks=int(os.environ.get("DSV4_CSA_LOGICAL_BLOCKS", "65537")),
            fetch_ctas=int(os.environ.get("DSV4_CSA_FETCH_CTAS", "64")),
        )
        if min(config.gpu_cache_mib, config.logical_blocks, config.fetch_ctas) <= 0:
            raise ValueError(
                "DSV4 CSA offload capacities and fetch CTAs must be positive"
            )
        return config
