"""Environment-backed configuration shared by generic MoE components."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TpMoeChunkConfig:
    """Opt-in, eager pure-TP prefill experiment; identical on every TP rank."""

    chunks: int = 0
    mode: str = "overlap"
    min_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "TpMoeChunkConfig":
        config = cls(
            chunks=int(os.environ.get("MOE_TP_CHUNKS", "0")),
            mode=os.environ.get("MOE_TP_CHUNK_MODE", "overlap").strip().lower(),
            min_tokens=int(os.environ.get("MOE_TP_CHUNK_MIN_TOKENS", "4096")),
        )
        if config.chunks not in (0, 2, 4):
            raise ValueError("MOE_TP_CHUNKS must be 0, 2 or 4")
        if config.mode not in ("serial", "overlap"):
            raise ValueError("MOE_TP_CHUNK_MODE must be serial or overlap")
        if config.min_tokens < 1:
            raise ValueError("MOE_TP_CHUNK_MIN_TOKENS must be positive")
        return config


def strict_fused_moe_enabled() -> bool:
    return os.environ.get("MOE_STRICT_FUSED", "1") != "0"


def shared_expert_mode() -> str:
    return os.environ.get("MOE_SHARED_EXPERT_MODE", "sequential").strip().lower()


def mega_moe_input_packer_mode() -> str:
    return os.environ.get("MEGA_MOE_INPUT_PACKER", "fused").strip().lower()
