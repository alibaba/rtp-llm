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


@dataclass(frozen=True)
class TpMoePrefillConfig:
    """Independent, opt-in SM12x pure-TP prefill implementations."""

    backend: str = "default"
    direct_output: bool = False
    min_tokens: int = 4096

    @property
    def enabled(self) -> bool:
        return self.backend != "default" or self.direct_output

    @classmethod
    def from_env(cls) -> "TpMoePrefillConfig":
        direct_output = os.environ.get("MOE_TP_DIRECT_OUTPUT", "0").strip()
        if direct_output not in ("0", "1"):
            raise ValueError("MOE_TP_DIRECT_OUTPUT must be 0 or 1")
        config = cls(
            backend=os.environ.get("MOE_TP_PREFILL_BACKEND", "default").strip().lower(),
            direct_output=direct_output == "1",
            min_tokens=int(os.environ.get("MOE_TP_FUSION_MIN_TOKENS", "4096")),
        )
        if config.backend not in ("default", "deepgemm_fused", "flashinfer_sm12x"):
            raise ValueError(
                "MOE_TP_PREFILL_BACKEND must be default, deepgemm_fused or flashinfer_sm12x"
            )
        if config.min_tokens < 1:
            raise ValueError("MOE_TP_FUSION_MIN_TOKENS must be positive")
        if config.backend == "deepgemm_fused" and os.environ.get(
            "DSV4_FP8_QUANT_KERNEL", "auto"
        ).strip().lower() not in ("auto", "v2"):
            raise ValueError("deepgemm_fused requires DSV4_FP8_QUANT_KERNEL=auto or v2")
        return config


def shared_expert_mode() -> str:
    return os.environ.get("MOE_SHARED_EXPERT_MODE", "sequential").strip().lower()


def mega_moe_input_packer_mode() -> str:
    return os.environ.get("MEGA_MOE_INPUT_PACKER", "fused").strip().lower()
