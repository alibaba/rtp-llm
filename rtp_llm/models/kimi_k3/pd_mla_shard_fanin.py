"""K3 Prefill-to-Decode MLA cache shard fan-in contracts.

KDA cache routing stays rank-to-rank.  MLA cache routing is different: all
Prefill TP shards for one request land on the request's explicitly selected
Decode owner.  This module deliberately does not derive ownership from a
request id; scheduling must bind an owner before transfer starts and keep it
stable for the lifetime of the request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from rtp_llm.models.kimi_k3.mla_cache_tp import MlaCacheShardLayout


KIMI_K3_MLA_LAYOUT_VERSION = 1
KIMI_K3_MLA_CACHE_GROUP = "mla"
KIMI_K3_MLA_LATENT_WIDTH = 512
KIMI_K3_MLA_SUFFIX_WIDTH = 64


@dataclass(frozen=True)
class MlaShardMetadata:
    request_id: int
    target_owner_rank: int
    cache_group: str
    layer_id: int
    shard_rank: int
    shard_count: int
    latent_offset: int
    latent_width: int
    suffix_offset: int
    suffix_width: int
    token_count: int
    layout_version: int = KIMI_K3_MLA_LAYOUT_VERSION

    @classmethod
    def for_kimi_k3(
        cls,
        *,
        request_id: int,
        target_owner_rank: int,
        layer_id: int,
        shard_rank: int,
        shard_count: int,
        token_count: int,
    ) -> "MlaShardMetadata":
        layout = MlaCacheShardLayout.fixed(
            KIMI_K3_MLA_LATENT_WIDTH,
            KIMI_K3_MLA_SUFFIX_WIDTH,
            shard_count,
            shard_rank,
        )
        return cls(
            request_id=request_id,
            target_owner_rank=target_owner_rank,
            cache_group=KIMI_K3_MLA_CACHE_GROUP,
            layer_id=layer_id,
            shard_rank=shard_rank,
            shard_count=shard_count,
            latent_offset=layout.latent_start,
            latent_width=layout.local_latent,
            suffix_offset=layout.suffix_start,
            suffix_width=layout.local_suffix,
            token_count=token_count,
        )

    def validate(self, *, decode_tp_size: int) -> None:
        if self.layout_version != KIMI_K3_MLA_LAYOUT_VERSION:
            raise ValueError(
                f"unsupported MLA cache layout version {self.layout_version}; "
                f"expected {KIMI_K3_MLA_LAYOUT_VERSION}"
            )
        if self.cache_group != KIMI_K3_MLA_CACHE_GROUP:
            raise ValueError(f"invalid MLA cache group {self.cache_group!r}")
        if self.request_id < 0 or self.layer_id < 0 or self.token_count <= 0:
            raise ValueError(
                "request_id/layer_id must be non-negative and token_count positive"
            )
        if not 0 <= self.target_owner_rank < decode_tp_size:
            raise ValueError(
                f"owner rank {self.target_owner_rank} outside Decode TP{decode_tp_size}"
            )
        layout = MlaCacheShardLayout.fixed(
            KIMI_K3_MLA_LATENT_WIDTH,
            KIMI_K3_MLA_SUFFIX_WIDTH,
            self.shard_count,
            self.shard_rank,
        )
        actual = (
            self.latent_offset,
            self.latent_width,
            self.suffix_offset,
            self.suffix_width,
        )
        expected = (
            layout.latent_start,
            layout.local_latent,
            layout.suffix_start,
            layout.local_suffix,
        )
        if actual != expected:
            raise ValueError(f"MLA shard component offsets mismatch: {actual} != {expected}")

    def to_wire_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_wire_dict(cls, payload: dict[str, Any]) -> "MlaShardMetadata":
        expected_fields = set(cls.__dataclass_fields__)
        actual_fields = set(payload)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            raise ValueError(
                f"invalid MLA shard metadata fields missing={missing} extra={extra}"
            )
        return cls(**payload)


class FixedMlaOwnerRegistry:
    """Stable owner bindings for the pre-DPLB implementation.

    The caller must make an explicit scheduling decision. Rebinding a live
    request is rejected because it would require moving resident MLA cache and
    rebuilding the Decode-side block table.
    """

    def __init__(self, decode_tp_size: int) -> None:
        if decode_tp_size <= 0:
            raise ValueError("decode_tp_size must be positive")
        self._decode_tp_size = decode_tp_size
        self._owners: dict[int, int] = {}

    def bind(self, request_id: int, owner_rank: int) -> None:
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        if not 0 <= owner_rank < self._decode_tp_size:
            raise ValueError(
                f"owner rank {owner_rank} outside Decode TP{self._decode_tp_size}"
            )
        previous = self._owners.get(request_id)
        if previous is not None and previous != owner_rank:
            raise ValueError(
                f"request {request_id} is sticky on owner {previous}, cannot rebind to {owner_rank}"
            )
        self._owners[request_id] = owner_rank

    def owner(self, request_id: int) -> int:
        try:
            return self._owners[request_id]
        except KeyError as exc:
            raise KeyError(
                f"request {request_id} has no explicit MLA owner binding"
            ) from exc

    def release(self, request_id: int) -> None:
        self._owners.pop(request_id, None)


@dataclass(frozen=True)
class PDMlaTransferPlan:
    request_id: int
    target_owner_rank: int
    tp_size: int

    def __post_init__(self) -> None:
        if self.request_id < 0 or self.tp_size <= 0:
            raise ValueError("invalid request id or TP size")
        if not 0 <= self.target_owner_rank < self.tp_size:
            raise ValueError("target owner outside TP group")

    def kda_destination(self, shard_rank: int) -> int:
        if not 0 <= shard_rank < self.tp_size:
            raise ValueError("KDA shard rank outside TP group")
        return shard_rank

    def mla_destination(self, shard_rank: int) -> int:
        if not 0 <= shard_rank < self.tp_size:
            raise ValueError("MLA shard rank outside TP group")
        return self.target_owner_rank


class MlaShardFanIn:
    """Directly assemble packed rank shards into token-major full-576 cache."""

    def __init__(
        self,
        *,
        request_id: int,
        target_owner_rank: int,
        layer_id: int,
        shard_count: int,
        token_count: int,
        decode_tp_size: int,
    ) -> None:
        self._request_id = request_id
        self._target_owner_rank = target_owner_rank
        self._layer_id = layer_id
        self._shard_count = shard_count
        self._token_count = token_count
        self._decode_tp_size = decode_tp_size
        self._received: set[int] = set()
        self._output: torch.Tensor | None = None

        # Validate the invariant once even before the first shard arrives.
        MlaShardMetadata.for_kimi_k3(
            request_id=request_id,
            target_owner_rank=target_owner_rank,
            layer_id=layer_id,
            shard_rank=0,
            shard_count=shard_count,
            token_count=token_count,
        ).validate(decode_tp_size=decode_tp_size)

    def add(self, metadata: MlaShardMetadata, packed_shard: torch.Tensor) -> None:
        metadata.validate(decode_tp_size=self._decode_tp_size)
        identity = (
            metadata.request_id,
            metadata.target_owner_rank,
            metadata.layer_id,
            metadata.shard_count,
            metadata.token_count,
        )
        expected_identity = (
            self._request_id,
            self._target_owner_rank,
            self._layer_id,
            self._shard_count,
            self._token_count,
        )
        if identity != expected_identity:
            raise ValueError(
                f"MLA shard does not belong to this fan-in: {identity} != {expected_identity}"
            )
        expected_shape = (
            metadata.token_count,
            metadata.latent_width + metadata.suffix_width,
        )
        if packed_shard.ndim != 2 or tuple(packed_shard.shape) != expected_shape:
            raise ValueError(
                f"packed MLA shard shape {tuple(packed_shard.shape)} != {expected_shape}"
            )
        if metadata.shard_rank in self._received:
            raise ValueError(f"duplicate MLA shard rank {metadata.shard_rank}")

        if self._output is None:
            self._output = torch.empty(
                (self._token_count, KIMI_K3_MLA_LATENT_WIDTH + KIMI_K3_MLA_SUFFIX_WIDTH),
                dtype=packed_shard.dtype,
                device=packed_shard.device,
            )
        elif (
            self._output.dtype != packed_shard.dtype
            or self._output.device != packed_shard.device
        ):
            raise ValueError("all MLA shards must use the same dtype and device")

        local_latent = packed_shard[:, : metadata.latent_width]
        local_suffix = packed_shard[:, metadata.latent_width :]
        self._output[
            :, metadata.latent_offset : metadata.latent_offset + metadata.latent_width
        ].copy_(local_latent)
        suffix_start = KIMI_K3_MLA_LATENT_WIDTH + metadata.suffix_offset
        self._output[:, suffix_start : suffix_start + metadata.suffix_width].copy_(
            local_suffix
        )
        self._received.add(metadata.shard_rank)

    @property
    def complete(self) -> bool:
        return len(self._received) == self._shard_count

    def missing_shards(self) -> tuple[int, ...]:
        return tuple(sorted(set(range(self._shard_count)) - self._received))

    def finalize(self) -> torch.Tensor:
        if not self.complete or self._output is None:
            raise ValueError(f"incomplete MLA fan-in; missing shards {self.missing_shards()}")
        return self._output


__all__ = [
    "FixedMlaOwnerRegistry",
    "KIMI_K3_MLA_CACHE_GROUP",
    "KIMI_K3_MLA_LAYOUT_VERSION",
    "MlaShardFanIn",
    "MlaShardMetadata",
    "PDMlaTransferPlan",
]
