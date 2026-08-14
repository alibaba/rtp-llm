"""Reference GLM-5.4 indexer compressor and cache geometry.

This is deliberately a correctness/reference implementation. It mirrors the
DSV4 CSA compressor assumption used by the first cache layout: ratio=4,
one-group overlap, two projection branches, per-channel softmax over the
8-token window, then RMSNorm. The production FP8 writer will reuse the DSV4
CompressorFP8 kernels after checkpoint names and state semantics are verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


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
    overlap: int = 1
    fp8: bool = True
    gen_num_per_cycle: int = 0

    @property
    def kv_entry_bytes(self) -> int:
        return self.head_dim + 4 if self.fp8 else self.head_dim * 2

    @property
    def state_width(self) -> int:
        # Projected KV plus score+APE, each with (1+overlap) branches.
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
    norm_weight: torch.Tensor,
    *,
    compress_ratio: int = 4,
    overlap: int = 1,
    norm_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress a single sequence and return ``(keys, boundary_positions)``.

    The projection width is ``(1+overlap) * head_dim``. With ratio=4 and
    overlap=1, tokens from the previous group read branch 0 while tokens from
    the current group read branch 1, matching DSV4's CSA fused writer.
    Only complete groups produce a key.
    """

    if kv_projection.dim() != 2 or score_projection.shape != kv_projection.shape:
        raise ValueError("kv_projection and score_projection must share shape [T, C]")
    if ape.dim() != 2 or ape.size(0) != compress_ratio:
        raise ValueError(
            f"ape must have shape [{compress_ratio}, C], got {tuple(ape.shape)}"
        )
    if ape.size(1) != kv_projection.size(1):
        raise ValueError("ape width must match projection width")
    branches = 1 + overlap
    if overlap not in (0, 1):
        raise ValueError(f"overlap must be 0 or 1, got {overlap}")
    if kv_projection.size(1) % branches != 0:
        raise ValueError("projection width must be divisible by branch count")
    head_dim = kv_projection.size(1) // branches
    if tuple(norm_weight.shape) != (head_dim,):
        raise ValueError(
            f"norm_weight must have shape ({head_dim},), got {tuple(norm_weight.shape)}"
        )

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
    window = branches * compress_ratio
    for boundary_tensor in boundaries:
        boundary = int(boundary_tensor.item())
        window_start = boundary - window + 1
        positions = torch.arange(
            max(0, window_start),
            boundary + 1,
            dtype=torch.int64,
            device=kv_projection.device,
        )
        # Logical window slots determine the projection branch. This remains
        # correct for the first group where negative positions are masked.
        logical_slots = positions - window_start
        branch = torch.div(logical_slots, compress_ratio, rounding_mode="floor")
        offsets = branch * head_dim
        dims = torch.arange(head_dim, device=kv_projection.device)
        gather_dims = offsets.unsqueeze(1) + dims.unsqueeze(0)

        kv = kv_projection[positions.unsqueeze(1), gather_dims]
        score = score_projection[positions.unsqueeze(1), gather_dims]
        score = score + ape[
            (positions % compress_ratio).unsqueeze(1), gather_dims
        ]
        weights = torch.softmax(score.float(), dim=0).to(kv.dtype)
        compressed = torch.sum(kv * weights, dim=0)
        variance = compressed.float().square().mean()
        normalized = compressed.float() * torch.rsqrt(variance + norm_eps)
        outputs.append((normalized * norm_weight.float()).to(kv_projection.dtype))

    return torch.stack(outputs, dim=0), boundaries


class IndexerCompressorReference(nn.Module):
    """Learned projection wrapper around the pure-Torch compressor reference."""

    def __init__(
        self,
        hidden_dim: int,
        head_dim: int = 128,
        compress_ratio: int = 4,
        overlap: int = 1,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        self.norm_eps = norm_eps
        projection_dim = (1 + overlap) * head_dim
        self.wkv = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.wgate = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.ape = nn.Parameter(torch.zeros(compress_ratio, projection_dim))
        self.norm_weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kv = F.linear(hidden_states, self.wkv.weight)
        score = F.linear(hidden_states, self.wgate.weight)
        return compress_indexer_projection_reference(
            kv,
            score,
            self.ape,
            self.norm_weight,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            norm_eps=self.norm_eps,
        )
