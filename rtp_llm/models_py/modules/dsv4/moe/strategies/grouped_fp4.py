"""GroupedFP4Strategy: single-card DeepGEMM ``m_grouped_fp8_fp4_gemm_nt_contiguous``.

EP == 1 + factory_mode + DeepGEMM ≥ 2.4 + SM100. Opt-in via
``DSV4_USE_GROUPED_FP4=1`` (legacy toggle, mapped to ``forced='grouped_fp4'``
in ``select_strategy``).

Wired into ``MoE`` via ``select_strategy`` when ep_size == 1 and the
DeepGEMM kernel is available + opted in.

Phase 2 (per ``.claude/plans/optimized-riding-mist.md::Phase 2``) will add
4 prefill optimizations to ``forward``:
  - quant-first reorder (input quantized once before scatter, not M_padded times)
  - Triton ep_scatter (replaces argsort/bincount/cumsum/index_copy chain)
  - fused silu+mul+fp8_quant kernel (replaces clamp/silu/mul/cast/quant chain)
  - Triton ep_gather (replaces index_select+sum, no fp32 [N, topk, D] materialization)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    m_grouped_fp8_fp4_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from ..quant_layouts import FP4_BLOCK, FP8_BLOCK


def _has_fp8_fp4_grouped_kernel() -> bool:
    """True iff the grouped FP4 routed-expert path should be used.

    Opt-in via ``DSV4_USE_GROUPED_FP4=1`` because the path historically
    produced garbage output across V4-Flash's 60 layers when the kernel's
    per-group row alignment (``get_mk_alignment_for_contiguous_layout()``,
    default 128 on SM100) was not honored — single-pass synthetic tests
    were inside 5% rel-diff but per-layer error compounded catastrophically.
    The current ``_grouped_routed_experts`` zeros padding rows + tags
    ``-1`` in ``grouped_layout`` per the kernel contract (mirrors
    ``deepgemm/tests/generators.py::generate_m_grouped_contiguous``),
    but smoke validation on every new SM100 variant is required before
    flipping this default ON.

    Requires deep_gemm ≥ 2.4 (ships ``m_grouped_fp8_fp4_gemm_nt_contiguous``)
    and an SM100+ device.
    """
    if os.environ.get("DSV4_USE_GROUPED_FP4", "0") != "1":
        return False
    try:
        import deep_gemm
    except Exception:
        return False
    if not hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        return False
    if not hasattr(deep_gemm, "get_mk_alignment_for_contiguous_layout"):
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


@register_strategy
class GroupedFP4Strategy(RoutedExpertsStrategy):
    name = "grouped_fp4"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return cfg.ep_size == 1 and cfg.factory_mode and _has_fp8_fp4_grouped_kernel()

    def setup_weights(
        self,
        weights: Optional[Dict[str, torch.Tensor]],
        prefix: str,
    ) -> None:
        """Stack all routed experts into ``[E, ...]`` int8 + UE8M0 SF tensors.

        Pops keys (factory_mode):
          ``{prefix}.experts.{i}.{w1,w2,w3}.{weight,scale}`` for i in [0, n_routed_experts).

        Memory: stacked tensors are alloc'd/copied via a per-expert mini-buffer
        pattern to avoid holding both the loader's dict entries AND the stacked
        layout simultaneously: ``weights.pop`` detaches each tensor from the
        loader BEFORE the next expert's allocation runs, and
        ``torch.cuda.empty_cache()`` after the copy loop returns the freed FP4
        blocks to the CUDA driver so they don't sit in the caching allocator
        while KV-pool sizing measures available HBM.
        """
        if weights is None:
            raise RuntimeError(
                "GroupedFP4Strategy requires factory_mode (weights dict). "
                "Eager init is not supported."
            )

        cfg = self.cfg
        E, D, inter = cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim
        device = weights[f"{prefix}.experts.0.w1.weight"].device

        self._w13 = torch.empty(
            (E, 2 * inter, D // 2), dtype=torch.int8, device=device
        )
        self._s13 = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        self._w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        self._s2 = torch.empty(
            (E, D, inter // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        for i in range(E):
            p = f"{prefix}.experts.{i}"
            # w1 → [:inter], w3 → [inter:2*inter]
            self._w13[i, :inter].copy_(weights.pop(f"{p}.w1.weight"))
            self._s13[i, :inter].copy_(weights.pop(f"{p}.w1.scale"))
            self._w13[i, inter:].copy_(weights.pop(f"{p}.w3.weight"))
            self._s13[i, inter:].copy_(weights.pop(f"{p}.w3.scale"))
            self._w2[i].copy_(weights.pop(f"{p}.w2.weight"))
            self._s2[i].copy_(weights.pop(f"{p}.w2.scale"))

        # Return loader's freed FP4 blocks to CUDA so the KV-cache
        # planner sees the real residual HBM rather than what's
        # cached-but-unused inside PyTorch's allocator.
        torch.cuda.empty_cache()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Single-GPU grouped FP4 expert compute.

        Args:
          x: [N, D] BF16 flattened tokens (post-MoE-gate activation).
          weights: [N, topk] FP32 router weights.
          indices: [N, topk] int64 expert IDs.

        Returns:
          y: [N, D] float32 sum over (token, top-k) of
             ``weight * expert[idx](x)``.

        Flow:
          1. Expand each token into topk copies, sort by expert id.
          2. Compute per-expert counts and pad each group to
             ``get_mk_alignment_for_contiguous_layout()`` (128 on SM100).
             The kernel REQUIRES this alignment — without it, internal
             block scheduling produces wrong outputs even on the valid
             rows; the error compounds catastrophically across the 60
             V4-Flash layers.
          3. Build a padded activation buffer (zero on padding rows) and
             a per-row ``grouped_layout`` tensor (expert id on real rows,
             ``-1`` on padding rows so the kernel skips them).
          4. Quantize, run grouped FP8×FP4 GEMM for gate+up.
          5. Clamped SwiGLU + per-slot router weight (zero on padding rows
             ⇒ they contribute exactly zero to the down result).
          6. Quantize, run grouped FP8×FP4 GEMM for down.
          7. Single-gather un-permute back to token order, reduce top-k.

        Layout reference: ``deepgemm/tests/generators.py::generate_m_grouped_contiguous``.
        """
        import deep_gemm

        cfg = self.cfg
        N, D = x.shape
        topk = indices.size(-1)
        E = cfg.n_routed_experts
        inter = cfg.moe_inter_dim

        if N == 0:
            return torch.zeros(N, D, dtype=torch.float32, device=x.device)

        M_ALIGN = deep_gemm.get_mk_alignment_for_contiguous_layout()

        # (1) flatten + sort by expert
        expanded_x = x.unsqueeze(1).expand(N, topk, D).reshape(N * topk, D)
        flat_indices = indices.flatten().contiguous()  # [NK] int64
        flat_weights = weights.flatten().contiguous()  # [NK] fp32
        perm = torch.argsort(flat_indices, stable=True)  # [NK]
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N * topk, device=x.device, dtype=perm.dtype)
        expert_per_slot = flat_indices.index_select(0, perm)  # [NK] sorted expert ids

        # (2) per-expert counts → exclusive-scan offsets (real + aligned).
        # NOTE the EXCLUSIVE scan: the first expert's slot starts at 0, so
        # ``sorted_offsets[0] == 0``.  An inclusive ``cumsum`` would give the
        # END offset and break ``within_expert`` for the first expert.
        counts = torch.bincount(flat_indices.to(torch.int64), minlength=E)  # [E]
        aligned_counts = ((counts + M_ALIGN - 1) // M_ALIGN) * M_ALIGN  # [E]
        sorted_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])  # [E]
        aligned_offsets = torch.cat(
            [aligned_counts.new_zeros(1), aligned_counts.cumsum(0)[:-1]]
        )  # [E]
        # CPU sync — required to allocate the padded tensor with a Python
        # int shape.  Eager-only by construction; cuda-graph capture takes
        # the legacy local path (see MoE.forward()).
        M_padded = int(aligned_counts.sum().item())

        # (3) destination row per slot: aligned_offsets[expert] + within_expert
        slot_idx = torch.arange(N * topk, device=x.device, dtype=torch.int64)
        within_expert = slot_idx - sorted_offsets[expert_per_slot]
        dest = aligned_offsets[expert_per_slot] + within_expert  # [NK]

        # (4) padded activation + grouped_layout (zero-init for padding rows)
        perm_x = torch.zeros(M_padded, D, dtype=x.dtype, device=x.device)
        perm_weights = torch.zeros(M_padded, dtype=flat_weights.dtype, device=x.device)
        grouped_layout = torch.full((M_padded,), -1, dtype=torch.int32, device=x.device)
        perm_x.index_copy_(0, dest, expanded_x.index_select(0, perm))
        perm_weights.index_copy_(0, dest, flat_weights.index_select(0, perm))
        grouped_layout.index_copy_(0, dest, expert_per_slot.to(torch.int32))

        # (5) gate+up GEMM via grouped FP8×FP4
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            perm_x,
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        s13_fp32 = self._s13.float()
        gate_up = torch.empty(
            M_padded, 2 * inter, device=x.device, dtype=torch.bfloat16
        )
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a_fp8, a_scale),
            (self._w13, s13_fp32),
            gate_up,
            grouped_layout,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        del a_fp8, a_scale, s13_fp32

        # (6) clamped SwiGLU + per-slot router weight (padding rows: weight=0
        # so their contribution to down GEMM input is exactly 0)
        gate = gate_up[:, :inter].float()
        up = gate_up[:, inter:].float()
        if cfg.swiglu_limit > 0:
            up = torch.clamp(up, min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
            gate = torch.clamp(gate, max=cfg.swiglu_limit)
        hidden = F.silu(gate) * up
        hidden = hidden * perm_weights.unsqueeze(-1)
        hidden = hidden.to(torch.bfloat16)
        del gate, up, gate_up

        # (7) down GEMM via grouped FP8×FP4
        h_fp8, h_scale = sgl_per_token_group_quant_fp8(
            hidden,
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        s2_fp32 = self._s2.float()
        down_out = torch.empty(M_padded, D, device=x.device, dtype=torch.bfloat16)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, s2_fp32),
            down_out,
            grouped_layout,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        del h_fp8, h_scale, s2_fp32

        # (8) un-permute (single fused gather) + reduce top-k.
        # ``dest[inv_perm[t]]`` = padded position holding token ``t``'s
        # expert output.  ``inv_perm`` undoes the argsort; the ``dest``
        # indirection then jumps to the right padded row.
        gather_idx = dest.index_select(0, inv_perm)  # [NK]
        unperm = down_out.index_select(0, gather_idx).view(N, topk, D)
        return unperm.float().sum(dim=1)  # [N, D] fp32
