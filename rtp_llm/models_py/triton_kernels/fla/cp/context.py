# -*- coding: utf-8 -*-
# CP context — Plan A: chunk-aligned padding for the zigzag GDN forward.
#
# Mirrors fla_cp/context.py in shape: a small dataclass holding precomputed
# tensors / Python lists describing the per-forward batch layout, plus a
# `build` classmethod that constructs everything from CPU-side seq lengths.
# Sharing this object across all CP linear layers in one forward avoids
# rebuilding indices per layer.

from dataclasses import dataclass
from typing import List, Optional

import torch

from rtp_llm.models_py.triton_kernels.fla.cp.utils import zigzag_causal_order


@dataclass
class CpChunkAlignInfo:
    """Precomputed indices for chunk-aligned padding/reorder in CP linear attention.

    Plan A: align padded sequence to (cp_size * 2 * chunk_size) so each rank's
    half-segment is a chunk_size multiple — segment-internal chunk boundaries
    then coincide with full-sequence chunk boundaries, making h states reusable
    for SSM cache writes. Padding tokens must be made identity-preserving in the
    GDN recurrence (see Qwen3NextGatedDeltaNet._forward_cp_prefill mask use).
    """

    # ---- CPU-side index tables (shared across layers, no GPU sync) ----
    local_cu_cpu: List[int]
    half_lengths_cpu: List[int]
    orig_full_lengths_cpu: List[int]

    # ---- Device tensors (uploaded once, reused per layer) ----
    orig_full_cu: torch.Tensor  # int64, [batch+1]
    local_cu: torch.Tensor  # int64, [batch+1]
    seg_cu: torch.Tensor  # int64, [2*batch+1] — half-seg cu
    causal_order: torch.Tensor  # int64, [2*cp_size]
    h_causal_indices: torch.Tensor  # int64, full→reordered chunk indices
    padded_full_cu_d: torch.Tensor  # int32, [batch+1] — caller-provided
    orig_full_lengths_d: torch.Tensor  # int32, [batch]

    # ---- Scalars / optional padding mask ----
    local_total: int
    chunk_size: int
    local_padding_mask: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.orig_full_lengths_cpu)

    @classmethod
    def build(
        cls,
        cp_size: int,
        cp_rank: int,
        device: torch.device,
        orig_lengths_cpu: list,
        padded_full_cu: torch.Tensor,
        chunk_size: int = 64,
        local_padding_mask: Optional[torch.Tensor] = None,
    ) -> "CpChunkAlignInfo":
        """Build CP metadata from CPU-side sequence lengths.

        Args:
            orig_lengths_cpu: per-sequence original lengths (Python list of ints).
            padded_full_cu: cumulative padded lengths from C++, int32 on `device`.
                Stored as-is; no re-cumsum, no GPU sync.
            local_padding_mask: optional pre-built mask. Pass `cp_local_valid_mask`
                when any sequence was padded; None when caller verified no padding.
        """
        batch_size = len(orig_lengths_cpu)
        align = cp_size * 2 * chunk_size

        padded_lengths_cpu = [
            ((L + align - 1) // align) * align for L in orig_lengths_cpu
        ]
        local_lengths_cpu = [L // cp_size for L in padded_lengths_cpu]
        half_lengths_cpu = [L // 2 for L in local_lengths_cpu]

        local_cu_cpu = [0]
        for L in local_lengths_cpu:
            local_cu_cpu.append(local_cu_cpu[-1] + L)
        local_total = local_cu_cpu[-1]

        orig_full_cu_cpu = [0]
        for L in orig_lengths_cpu:
            orig_full_cu_cpu.append(orig_full_cu_cpu[-1] + L)

        # seg_cu: treats each seq's two halves as separate sequences
        seg_cu_cpu = [0]
        for h in half_lengths_cpu:
            seg_cu_cpu.append(seg_cu_cpu[-1] + h)
            seg_cu_cpu.append(seg_cu_cpu[-1] + h)

        local_chunks_per_seq = [L // chunk_size for L in local_lengths_cpu]
        local_NT = sum(local_chunks_per_seq)
        half_chunks = [n // 2 for n in local_chunks_per_seq]
        causal_order = zigzag_causal_order(cp_size)

        seg_chunk_offsets = [0]
        for hc in half_chunks:
            seg_chunk_offsets.append(seg_chunk_offsets[-1] + 2 * hc)
        orig_chunks_per_seq = [
            (L + chunk_size - 1) // chunk_size for L in orig_lengths_cpu
        ]

        indices = []
        for b in range(batch_size):
            hc = half_chunks[b]
            seg_base = seg_chunk_offsets[b]
            remaining = orig_chunks_per_seq[b]
            for pos in range(2 * cp_size):
                ag_idx = causal_order[pos]
                r = ag_idx // 2
                seg = ag_idx % 2
                src_start = r * local_NT + seg_base + seg * hc
                n = min(hc, remaining)
                if n <= 0:
                    break
                indices.extend(range(src_start, src_start + n))
                remaining -= n

        return cls(
            local_cu_cpu=local_cu_cpu,
            half_lengths_cpu=half_lengths_cpu,
            orig_full_lengths_cpu=orig_lengths_cpu,
            orig_full_cu=torch.tensor(
                orig_full_cu_cpu, dtype=torch.long, device=device
            ),
            local_cu=torch.tensor(local_cu_cpu, dtype=torch.long, device=device),
            seg_cu=torch.tensor(seg_cu_cpu, dtype=torch.long, device=device),
            causal_order=torch.tensor(causal_order, dtype=torch.long, device=device),
            h_causal_indices=torch.tensor(indices, dtype=torch.long, device=device),
            padded_full_cu_d=padded_full_cu,  # reuse caller's tensor as-is
            orig_full_lengths_d=torch.tensor(
                orig_lengths_cpu, dtype=torch.int32, device=device
            ),
            local_total=local_total,
            chunk_size=chunk_size,
            local_padding_mask=local_padding_mask,
        )
