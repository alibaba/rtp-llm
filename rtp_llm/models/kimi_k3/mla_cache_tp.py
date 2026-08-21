"""K3 Prefill MLA cache dimension-sharding contracts.

The compute path remains ordinary MLA TP. Only resident cache components are
sharded: each rank stores a contiguous 512/TP latent slice followed by a
contiguous 64/TP suffix slice. Keeping the two component boundaries explicit
is required because a naive split of the packed 576 dimension would mix latent
and suffix offsets during P->D fan-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from rtp_llm.ops import RoleType


_MLA_CACHE_TP_ENV = "KIMI_K3_MLA_CACHE_TP"


def mla_cache_tp_enabled(parallelism_config: Any) -> bool:
    raw = os.environ.get(_MLA_CACHE_TP_ENV, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{_MLA_CACHE_TP_ENV} must be 0 or 1, got {raw!r}")
    if raw == "0" or parallelism_config is None:
        return False
    return (
        parallelism_config.role_type == RoleType.PREFILL
        and int(parallelism_config.get_attn_tp_size()) > 1
    )


@dataclass(frozen=True)
class MlaCacheShardLayout:
    full_latent: int
    full_suffix: int
    tp_size: int
    tp_rank: int

    @classmethod
    def fixed(
        cls,
        full_latent: int,
        full_suffix: int,
        tp_size: int,
        tp_rank: int,
    ) -> "MlaCacheShardLayout":
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid TP placement size={tp_size} rank={tp_rank}")
        if full_latent <= 0 or full_suffix <= 0:
            raise ValueError(
                f"MLA cache dimensions must be positive, got {full_latent}+{full_suffix}"
            )
        if full_latent % tp_size or full_suffix % tp_size:
            raise ValueError(
                "MLA cache TP requires TP to divide latent and suffix independently; "
                f"got latent={full_latent}, suffix={full_suffix}, TP={tp_size}"
            )
        return cls(full_latent, full_suffix, tp_size, tp_rank)

    @property
    def local_latent(self) -> int:
        return self.full_latent // self.tp_size

    @property
    def local_suffix(self) -> int:
        return self.full_suffix // self.tp_size

    @property
    def local_width(self) -> int:
        return self.local_latent + self.local_suffix

    @property
    def latent_start(self) -> int:
        return self.tp_rank * self.local_latent

    @property
    def suffix_start(self) -> int:
        return self.tp_rank * self.local_suffix

    def shard_components(
        self, compressed_kv: torch.Tensor, suffix: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if compressed_kv.shape[-1] != self.full_latent:
            raise ValueError(
                f"compressed KV width must be {self.full_latent}, "
                f"got {compressed_kv.shape[-1]}"
            )
        if suffix.shape[-1] != self.full_suffix:
            raise ValueError(
                f"MLA suffix width must be {self.full_suffix}, got {suffix.shape[-1]}"
            )
        local_latent = compressed_kv.narrow(
            -1, self.latent_start, self.local_latent
        ).contiguous()
        local_suffix = suffix.narrow(
            -1, self.suffix_start, self.local_suffix
        ).contiguous()
        return local_latent, local_suffix

    def reconstruct_rank_major(
        self, rank_major_shards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full components from ``[TP,tokens,local_width]`` shards."""

        if rank_major_shards.ndim != 3:
            raise ValueError(
                "rank-major MLA shards must be [TP,tokens,width], got "
                f"{tuple(rank_major_shards.shape)}"
            )
        if (
            rank_major_shards.shape[0] != self.tp_size
            or rank_major_shards.shape[-1] != self.local_width
        ):
            raise ValueError(
                "rank-major MLA shard shape mismatch: "
                f"shape={tuple(rank_major_shards.shape)} TP={self.tp_size} "
                f"local_width={self.local_width}"
            )
        latent = torch.cat(
            [
                rank_major_shards[rank, :, : self.local_latent]
                for rank in range(self.tp_size)
            ],
            dim=-1,
        )
        suffix = torch.cat(
            [
                rank_major_shards[rank, :, self.local_latent :]
                for rank in range(self.tp_size)
            ],
            dim=-1,
        )
        return latent, suffix


def kimi_k3_mla_cache_layout(parallelism_config: Any) -> MlaCacheShardLayout:
    return MlaCacheShardLayout.fixed(
        512,
        64,
        int(parallelism_config.get_attn_tp_size()),
        int(parallelism_config.get_attn_tp_rank()),
    )


__all__ = [
    "MlaCacheShardLayout",
    "kimi_k3_mla_cache_layout",
    "mla_cache_tp_enabled",
]
