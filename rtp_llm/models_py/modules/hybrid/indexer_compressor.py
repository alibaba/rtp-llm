"""GLM-5.3-Flash KPool reference math and typed-cache storage views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def raw_pool_block_rows(base: torch.Tensor, pool_name: str) -> torch.Tensor:
    if base.dim() < 2:
        raise RuntimeError(
            f"{pool_name} must have block and storage dimensions, "
            f"got shape={tuple(base.shape)} dtype={base.dtype}"
        )
    if not base.is_contiguous():
        raise RuntimeError(
            f"{pool_name} must be contiguous, got shape={tuple(base.shape)} "
            f"stride={tuple(base.stride())} dtype={base.dtype}"
        )
    return base.view(torch.uint8).view(int(base.shape[0]), -1)


def fp8_pool_view(base: torch.Tensor, entry_bytes: int) -> tuple[torch.Tensor, int]:
    """Return the nominal DeepGEMM view over block-planar FP8 storage.

    The last dimension describes the total bytes owned by one logical entry,
    but it is not a physically interleaved ``[K | scale]`` row.  Within each
    block, all FP8 K vectors are contiguous first and all FP32 scales follow.
    Consumers must either pass the raw view to the existing DSV4/DeepGEMM
    kernels or calculate offsets from the block base; indexing ``view[b, e]``
    as a self-contained entry is invalid.
    """
    raw = raw_pool_block_rows(base, "compressed indexer KV pool")
    stride_bytes = int(raw.shape[1])
    if entry_bytes <= 0 or stride_bytes % entry_bytes != 0:
        raise RuntimeError(
            "compressed indexer KV pool block size is not an exact multiple "
            f"of one entry: stride_bytes={stride_bytes} entry_bytes={entry_bytes}"
        )
    entries_per_block = stride_bytes // entry_bytes
    return (
        raw.view(int(raw.shape[0]), entries_per_block, entry_bytes),
        entries_per_block,
    )


def fp32_state_pool_view(
    base: torch.Tensor, state_width: int
) -> tuple[torch.Tensor, int]:
    raw = raw_pool_block_rows(base, "compressed indexer state pool")
    entry_bytes = state_width * torch.float32.itemsize
    stride_bytes = int(raw.shape[1])
    if state_width <= 0 or stride_bytes % entry_bytes != 0:
        raise RuntimeError(
            "compressed indexer state pool block size is not an exact multiple "
            f"of one entry: stride_bytes={stride_bytes} entry_bytes={entry_bytes}"
        )
    entries_per_block = stride_bytes // entry_bytes
    return raw.view(torch.float32).view(-1, state_width), entries_per_block


def _hadamard_rotate_reference(x: torch.Tensor) -> torch.Tensor:
    width = int(x.size(-1))
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    result = x
    stride = 1
    while stride < width:
        pairs = result.reshape(*result.shape[:-1], -1, 2, stride)
        left, right = pairs.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        stride *= 2
    return result * (width**-0.5)


def compressor_state_ring_entries(
    compress_ratio: int, overlap: int, gen_num_per_cycle: int = 0
) -> int:
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if overlap not in (0, 1):
        raise ValueError(f"overlap must be 0 or 1, got {overlap}")
    if gen_num_per_cycle < 0:
        raise ValueError(
            f"gen_num_per_cycle must be non-negative, got {gen_num_per_cycle}"
        )
    raw_entries = (1 + overlap) * compress_ratio + gen_num_per_cycle
    return (raw_entries + 1) & ~1


@dataclass(frozen=True)
class IndexerCompressorCacheLayout:
    head_dim: int = 128
    compress_ratio: int = 4
    overlap: int = 0
    fp8: bool = True
    gen_num_per_cycle: int = 0

    @property
    def kv_entry_bytes(self) -> int:
        return self.head_dim + 4 if self.fp8 else self.head_dim * 2

    @property
    def state_width(self) -> int:
        return 2 * (1 + self.overlap) * self.head_dim

    @property
    def state_ring_entries(self) -> int:
        return compressor_state_ring_entries(
            self.compress_ratio, self.overlap, self.gen_num_per_cycle
        )

    def entries_per_kernel_block(self, kernel_tokens_per_block: int) -> int:
        if kernel_tokens_per_block <= 0:
            raise ValueError("kernel_tokens_per_block must be positive")
        if kernel_tokens_per_block % self.compress_ratio != 0:
            raise ValueError(
                f"kernel block {kernel_tokens_per_block} is not divisible by "
                f"compress ratio {self.compress_ratio}"
            )
        return kernel_tokens_per_block // self.compress_ratio


def compress_indexer_projection_reference(
    kv_projection: torch.Tensor,
    score_projection: torch.Tensor,
    ape: torch.Tensor,
    *,
    compress_ratio: int = 4,
    overlap: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference: per-channel four-way softmax, pool, normalized Hadamard."""

    if kv_projection.dim() != 2 or score_projection.shape != kv_projection.shape:
        raise ValueError("kv_projection and score_projection must share shape [T, C]")
    if ape.dim() != 2 or tuple(ape.shape) != (
        compress_ratio,
        kv_projection.size(1),
    ):
        raise ValueError(
            f"ape must have shape [{compress_ratio}, {kv_projection.size(1)}]"
        )
    if overlap != 0:
        raise ValueError("GLM-5.3-Flash KPool uses non-overlapping groups")

    width = int(kv_projection.size(1))
    if width <= 0 or width & (width - 1):
        raise ValueError("head_dim must be a power of two")
    boundaries = torch.arange(
        compress_ratio - 1,
        int(kv_projection.size(0)),
        compress_ratio,
        dtype=torch.int64,
        device=kv_projection.device,
    )
    if boundaries.numel() == 0:
        return kv_projection.new_empty((0, width)), boundaries

    groups = []
    for boundary_tensor in boundaries:
        boundary = int(boundary_tensor.item())
        key = kv_projection[boundary - compress_ratio + 1 : boundary + 1]
        score = score_projection[boundary - compress_ratio + 1 : boundary + 1]
        weights = torch.softmax((score + ape).float(), dim=0).to(key.dtype)
        pooled = torch.sum(key * weights, dim=0).to(torch.bfloat16).float()
        groups.append(_hadamard_rotate_reference(pooled).to(torch.bfloat16).float())
    return torch.stack(groups, dim=0), boundaries


class IndexerCompressorReference(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int = 128) -> None:
        super().__init__()
        self.wgate = nn.Linear(hidden_dim, head_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(4, head_dim))

    def forward(
        self, hidden_states: torch.Tensor, normalized_key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return compress_indexer_projection_reference(
            normalized_key, F.linear(hidden_states, self.wgate.weight), self.ape
        )
