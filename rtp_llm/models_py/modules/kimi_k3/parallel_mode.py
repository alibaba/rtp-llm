"""Strict parallel execution modes for Kimi K3 modeling."""

from __future__ import annotations

from enum import Enum

from rtp_llm.ops import ParallelismConfig, RoleType


class KimiK3ParallelMode(str, Enum):
    """The only two layouts accepted by Kimi K3 operators."""

    TP_SP = "tp_sp"
    PROJECTION_KTP = "projection_ktp"


def resolve_kimi_k3_parallel_mode(
    parallelism_config: ParallelismConfig,
) -> KimiK3ParallelMode:
    """Resolve and validate the immutable modeling strategy."""

    tp_size = int(parallelism_config.get_attn_tp_size())
    ktp_size = int(getattr(parallelism_config, "ktp_size", 1))
    if tp_size <= 0 or ktp_size <= 0:
        raise ValueError(
            f"Kimi K3 parallel sizes must be positive, got TP={tp_size}, KTP={ktp_size}"
        )
    if ktp_size <= 1:
        return KimiK3ParallelMode.TP_SP
    if parallelism_config.role_type != RoleType.DECODE:
        raise RuntimeError("Projection KTP is supported only by Kimi K3 Decode")
    if tp_size != 1:
        raise RuntimeError(
            f"Projection KTP requires attention TP=1, got TP={tp_size}, KTP={ktp_size}"
        )
    return KimiK3ParallelMode.PROJECTION_KTP


__all__ = ["KimiK3ParallelMode", "resolve_kimi_k3_parallel_mode"]
