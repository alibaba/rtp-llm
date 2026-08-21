"""Memory-faithful cache allocation helpers for K3 Decode KTP experiments.

The serving scheduler currently creates the same MLA block pool on every TP
rank.  That pool cannot demonstrate request-owner memory savings even when the
forward pass only consumes owner-local rows.  This module is intentionally a
modeling harness: it allocates the KDA state for the global batch (TP sharded)
and the MLA history for either the global batch (baseline) or the rank-local
owner batch (KTP/MLA-DP).

It does not implement scheduling or cache migration.  The caller must keep the
fixed contiguous ownership contract from :mod:`decode_ktp` for the lifetime of
the allocation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from rtp_llm.models.kimi_k3.decode_ktp import DecodeOwnerLayout


@dataclass(frozen=True)
class DecodeKtpCacheSpec:
    global_batch: int
    tp_size: int
    tp_rank: int
    kv_length: int
    tokens_per_block: int
    mla_layer_num: int
    kda_layer_num: int
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    mla_width: int = 576
    cache_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        DecodeOwnerLayout.fixed(self.global_batch, self.tp_size, self.tp_rank)
        positive = {
            "kv_length": self.kv_length,
            "tokens_per_block": self.tokens_per_block,
            "mla_layer_num": self.mla_layer_num,
            "kda_layer_num": self.kda_layer_num,
            "linear_num_heads": self.linear_num_heads,
            "linear_head_dim": self.linear_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "mla_width": self.mla_width,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.linear_num_heads % self.tp_size:
            raise ValueError(
                "KDA heads must be divisible by TP; "
                f"got heads={self.linear_num_heads}, TP={self.tp_size}"
            )
        if self.linear_conv_kernel_dim <= 1:
            raise ValueError("linear_conv_kernel_dim must be greater than one")

    @property
    def owner_layout(self) -> DecodeOwnerLayout:
        return DecodeOwnerLayout.fixed(
            self.global_batch, self.tp_size, self.tp_rank
        )

    @property
    def blocks_per_request(self) -> int:
        return math.ceil(self.kv_length / self.tokens_per_block)

    @property
    def local_linear_heads(self) -> int:
        return self.linear_num_heads // self.tp_size

    @property
    def kda_state_elems_per_request_layer(self) -> int:
        # Matches LinearKVCacheSpec: [SSM state][short-conv state].
        projection = self.local_linear_heads * self.linear_head_dim
        ssm = self.local_linear_heads * self.linear_head_dim**2
        conv = (self.linear_conv_kernel_dim - 1) * 3 * projection
        return ssm + conv

    def mla_block_count(self, *, owner_local: bool) -> int:
        batch = self.owner_layout.local_batch if owner_local else self.global_batch
        return batch * self.blocks_per_request

    def mla_bytes(self, *, owner_local: bool) -> int:
        return (
            self.mla_layer_num
            * self.mla_block_count(owner_local=owner_local)
            * self.tokens_per_block
            * self.mla_width
            * torch.empty((), dtype=self.cache_dtype).element_size()
        )

    @property
    def kda_bytes(self) -> int:
        return (
            self.kda_layer_num
            * self.global_batch
            * self.kda_state_elems_per_request_layer
            * torch.empty((), dtype=self.cache_dtype).element_size()
        )


@dataclass
class DecodeKtpCacheAllocation:
    spec: DecodeKtpCacheSpec
    owner_local_mla: bool
    mla_layers: tuple[torch.Tensor, ...]
    kda_state: torch.Tensor

    @property
    def mla_allocated_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.mla_layers)

    @property
    def kda_allocated_bytes(self) -> int:
        return self.kda_state.numel() * self.kda_state.element_size()

    @property
    def total_allocated_bytes(self) -> int:
        return self.mla_allocated_bytes + self.kda_allocated_bytes

    def assert_matches_spec(self) -> None:
        expected_mla = self.spec.mla_bytes(owner_local=self.owner_local_mla)
        if self.mla_allocated_bytes != expected_mla:
            raise AssertionError(
                f"MLA allocation mismatch: actual={self.mla_allocated_bytes}, "
                f"expected={expected_mla}"
            )
        if self.kda_allocated_bytes != self.spec.kda_bytes:
            raise AssertionError(
                f"KDA allocation mismatch: actual={self.kda_allocated_bytes}, "
                f"expected={self.spec.kda_bytes}"
            )


def allocate_decode_ktp_cache(
    spec: DecodeKtpCacheSpec,
    *,
    owner_local_mla: bool,
    device: torch.device,
) -> DecodeKtpCacheAllocation:
    """Allocate cache tensors with the same byte formulas as RTP cache specs."""

    block_count = spec.mla_block_count(owner_local=owner_local_mla)
    mla_layers = tuple(
        torch.empty(
            (block_count, spec.tokens_per_block, spec.mla_width),
            dtype=spec.cache_dtype,
            device=device,
        )
        for _ in range(spec.mla_layer_num)
    )
    kda_state = torch.empty(
        (
            spec.kda_layer_num,
            spec.global_batch,
            spec.kda_state_elems_per_request_layer,
        ),
        dtype=spec.cache_dtype,
        device=device,
    )
    allocation = DecodeKtpCacheAllocation(
        spec=spec,
        owner_local_mla=owner_local_mla,
        mla_layers=mla_layers,
        kda_state=kda_state,
    )
    allocation.assert_matches_spec()
    return allocation


def build_owner_local_mla_block_table(
    spec: DecodeKtpCacheSpec,
    *,
    device: Optional[torch.device] = None,
    first_block_id: int = 1,
) -> torch.Tensor:
    """Build a dense, non-overlapping block table for this rank's requests."""

    if first_block_id < 0:
        raise ValueError(f"first_block_id must be non-negative, got {first_block_id}")
    count = spec.mla_block_count(owner_local=True)
    return torch.arange(
        first_block_id,
        first_block_id + count,
        dtype=torch.int32,
        device=device,
    ).reshape(spec.owner_layout.local_batch, spec.blocks_per_request)


__all__ = [
    "DecodeKtpCacheAllocation",
    "DecodeKtpCacheSpec",
    "allocate_decode_ktp_cache",
    "build_owner_local_mla_block_table",
]
