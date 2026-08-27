"""Reference GLM-5.3 KPool indexer compressor and cache geometry.

This is deliberately a correctness/reference implementation. It mirrors the
released checkpoint: ratio=4, non-overlapping groups, per-channel softmax over
the four-token group, then a normalized Hadamard rotation. Only complete
groups are written to the FP8 paged cache; the raw incomplete tail stays in
the state ring and is always selected by sparse attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def raw_pool_block_rows(base: torch.Tensor, pool_name: str) -> torch.Tensor:
    """Expose any contiguous framework pool shape as byte rows per block.

    Typed cache regions may arrive as either ``[blocks, stride]`` or an MLA
    view such as ``[blocks, tokens, width]``.  The dimensions after the block
    axis are only a storage view for the compressed indexer, so flatten them
    without copying and preserve the physical per-block byte stride.
    """

    if base.dim() < 2:
        raise RuntimeError(
            f"{pool_name} must have a block axis and storage dimensions, "
            f"got shape={tuple(base.shape)} dtype={base.dtype}"
        )
    if not base.is_contiguous():
        raise RuntimeError(
            f"{pool_name} must be contiguous, got shape={tuple(base.shape)} "
            f"stride={tuple(base.stride())} dtype={base.dtype}"
        )
    return base.view(torch.uint8).view(int(base.shape[0]), -1)


def fp8_pool_view(base: torch.Tensor, entry_bytes: int) -> tuple[torch.Tensor, int]:
    raw = raw_pool_block_rows(base, "compressed indexer KV pool")
    stride_bytes = int(raw.shape[1])
    if entry_bytes <= 0 or stride_bytes % entry_bytes != 0:
        raise RuntimeError(
            "compressed indexer KV pool block size is not an exact multiple "
            f"of one entry: shape={tuple(base.shape)} dtype={base.dtype} "
            f"stride_bytes={stride_bytes} entry_bytes={entry_bytes}"
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
    bytes_per_entry = state_width * torch.float32.itemsize
    stride_bytes = int(raw.shape[1])
    if state_width <= 0 or stride_bytes % bytes_per_entry != 0:
        raise RuntimeError(
            "compressed indexer state pool block size is not an exact multiple "
            f"of one entry: shape={tuple(base.shape)} dtype={base.dtype} "
            f"stride_bytes={stride_bytes} entry_bytes={bytes_per_entry}"
        )
    entries_per_block = stride_bytes // bytes_per_entry
    return raw.view(torch.float32).view(-1, state_width), entries_per_block


def _hadamard_rotate_reference(x: torch.Tensor) -> torch.Tensor:
    """Framework-only normalized Walsh-Hadamard transform for tests."""

    width = int(x.size(-1))
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    result = x
    stride = 1
    while stride < width:
        shape = (*result.shape[:-1], -1, 2, stride)
        pairs = result.reshape(shape)
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
        # Normalized indexer K plus gate logits, both retained for the tail.
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
    """Compress a single sequence and return ``(keys, boundary_positions)``.

    ``kv_projection`` is the already LayerNorm-normalized ``wk(x)`` tensor and
    ``score_projection`` is ``index_kpool_compress_gate(x)``. Each channel gets
    an independent four-way softmax after adding the learned APE. The weighted
    key is Hadamard-rotated after pooling. Only complete groups produce a key.
    """

    if kv_projection.dim() != 2 or score_projection.shape != kv_projection.shape:
        raise ValueError("kv_projection and score_projection must share shape [T, C]")
    if ape.dim() != 2 or ape.size(0) != compress_ratio:
        raise ValueError(
            f"ape must have shape [{compress_ratio}, C], got {tuple(ape.shape)}"
        )
    if ape.size(1) != kv_projection.size(1):
        raise ValueError("ape width must match projection width")
    if overlap != 0:
        raise ValueError("GLM-5.3 KPool uses non-overlapping four-token groups")
    head_dim = kv_projection.size(1)
    if head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two for Hadamard rotation")

    token_count = int(kv_projection.size(0))
    boundaries = torch.arange(
        compress_ratio - 1,
        token_count,
        compress_ratio,
        dtype=torch.int64,
        device=kv_projection.device,
    )
    if boundaries.numel() == 0:
        return kv_projection.new_empty((0, head_dim)), boundaries

    outputs = []
    for boundary_tensor in boundaries:
        boundary = int(boundary_tensor.item())
        positions = torch.arange(
            boundary - compress_ratio + 1,
            boundary + 1,
            dtype=torch.int64,
            device=kv_projection.device,
        )
        kv = kv_projection[positions]
        score = score_projection[positions] + ape
        weights = torch.softmax(score.float(), dim=0).to(kv.dtype)
        compressed = torch.sum(kv * weights, dim=0)
        outputs.append(_hadamard_rotate_reference(compressed))

    return torch.stack(outputs, dim=0), boundaries


class IndexerCompressorReference(nn.Module):
    """Learned projection wrapper around the pure-Torch compressor reference."""

    def __init__(
        self,
        hidden_dim: int,
        head_dim: int = 128,
        compress_ratio: int = 4,
        overlap: int = 0,
    ) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        if overlap != 0:
            raise ValueError("GLM-5.3 KPool does not use overlap")
        projection_dim = (1 + overlap) * head_dim
        self.wgate = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(compress_ratio, projection_dim))

    def forward(
        self, hidden_states: torch.Tensor, normalized_key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        score = F.linear(hidden_states, self.wgate.weight)
        return compress_indexer_projection_reference(
            normalized_key,
            score,
            self.ape,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
        )
