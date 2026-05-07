"""GroupedFP4Strategy: single-card DeepGEMM ``m_grouped_fp8_fp4_gemm_nt_contiguous``.

EP == 1 + DeepGEMM ≥ 2.4 + SM100. Opt-in via
``DSV4_USE_GROUPED_FP4=1`` (legacy toggle, mapped to ``forced='grouped_fp4'``
in ``select_strategy``).

Wired into ``MoE`` via ``select_strategy`` when ep_size == 1 and the
DeepGEMM kernel is available + opted in.

Forward is the 4-opt prefill path:
  (1) quant input ONCE pre-permute (vs. ×topk on padded buffer)
  (2) Triton ep_scatter (vs. argsort + bincount + cumsum + index_copy chain)
  (3) Fused silu+mul+fp8 quant kernel (vs. clamp + silu + mul + cast + quant)
  (4) Triton ep_gather with router-weight reduce (vs. index_select +
      fp32 [N,topk,D] materialize + sum)
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    m_grouped_fp8_fp4_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.utils.math import align, ceil_div

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .._silu_mul_fp8_quant_triton import silu_mul_fp8_quant_packed
from ...quant_layouts import FP4_BLOCK, FP8_BLOCK, prepare_fp4_weight_scale_for_deepgemm


# ep_scatter requires m_indices.shape[0] % BLOCK_E == 0 (BLOCK_E=128); also
# DeepGEMM contiguous requires per-expert M to be a multiple of the kernel's
# alignment (128 on SM100). We use the same constant.
_GROUPED_ALIGNMENT = 128


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
        return cfg.ep_size == 1 and _has_fp8_fp4_grouped_kernel()

    def setup_weights(self, layer_weights: Dict) -> None:
        """Stack EP-sliced routed-expert tensors into ``[E, ...]`` int8 +
        UE8M0 SF buffers in the layout DeepGEMM's contiguous kernel reads.

        Pops keys: ``W.v4_routed_w{1,2,3}_{w,s}`` from ``layer_weights``
        (each shaped ``[E_local, ...]``).

        Memory: pop the framework's stacked tensors so the only references
        kept alive are the repacked grouped buffers below, then bulk-copy
        in one `[:, :inter].copy_(stacked)` shot per slice (vs the legacy
        per-expert loop) — same allocation footprint, simpler code path.
        ``torch.cuda.empty_cache()`` after the copies returns the freed
        FP4 blocks to the CUDA driver so they don't sit in PyTorch's
        caching allocator while KV-pool sizing measures available HBM.
        """
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        E, D, inter = cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim
        stacked_w1_w = layer_weights.pop(W.v4_routed_w1_w)
        stacked_w1_s = layer_weights.pop(W.v4_routed_w1_s)
        stacked_w2_w = layer_weights.pop(W.v4_routed_w2_w)
        stacked_w2_s = layer_weights.pop(W.v4_routed_w2_s)
        stacked_w3_w = layer_weights.pop(W.v4_routed_w3_w)
        stacked_w3_s = layer_weights.pop(W.v4_routed_w3_s)
        device = stacked_w1_w.device

        self._w13 = torch.empty(
            (E, 2 * inter, D // 2), dtype=torch.int8, device=device
        )
        s13_raw = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        self._w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2_raw = torch.empty(
            (E, D, inter // FP4_BLOCK),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        # Bulk copy from stacked → repacked layout (one slice per dim,
        # no per-expert iteration).
        self._w13[:, :inter].copy_(stacked_w1_w)
        s13_raw[:, :inter].copy_(stacked_w1_s)
        self._w13[:, inter:].copy_(stacked_w3_w)
        s13_raw[:, inter:].copy_(stacked_w3_s)
        self._w2.copy_(stacked_w2_w)
        s2_raw.copy_(stacked_w2_s)
        del stacked_w1_w, stacked_w1_s, stacked_w2_w, stacked_w2_s
        del stacked_w3_w, stacked_w3_s

        self._s13 = prepare_fp4_weight_scale_for_deepgemm(
            s13_raw, 2 * inter, D, E
        )
        self._s2 = prepare_fp4_weight_scale_for_deepgemm(s2_raw, D, inter, E)
        del s13_raw, s2_raw

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
        """4-opt prefill path; returns ``[N, D] fp32``.

        Args:
          x: ``[N, D]`` BF16 flattened tokens (post-MoE-gate activation).
          weights: ``[N, topk]`` FP32 router weights.
          indices: ``[N, topk]`` int64 expert IDs.

        Returns:
          y: ``[N, D]`` float32 sum over (token, top-k) of
             ``weight * expert[idx](x)``.
        """
        cfg = self.cfg
        N, D = x.shape
        E = cfg.n_routed_experts
        inter = cfg.moe_inter_dim
        device = x.device

        if N == 0:
            return torch.zeros(N, D, dtype=torch.float32, device=device)

        # (1) Quant input ONCE — column-major TMA-aligned UE8M0 packed scale,
        # shape compatible with both ep_scatter input and DeepGEMM contiguous.
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            x.contiguous(),
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )

        # Per-expert counts in local index space (== global since ep_size==1).
        adjusted_topk_ids, num_recv = recompute_topk_ids_sum_expert_count(
            indices,
            current_expert_start_id=0,
            num_local_experts=E,
        )

        # Sum of aligned counts → all_tokens (CPU sync, ~E ints; same kind of
        # sync the framework's contiguous executor does at deepgemm_hybrid_executor.py:445).
        num_recv_cpu = num_recv.cpu().tolist()
        aligned_counts_list = [align(c, _GROUPED_ALIGNMENT) for c in num_recv_cpu]
        all_tokens = sum(aligned_counts_list)
        if all_tokens == 0:
            return torch.zeros((N, D), dtype=torch.float32, device=device)

        # ep_scatter's kernel_1 builds expert_start_loc as the EXCLUSIVE cumsum
        # of the per-expert counts it receives. For per-expert padded layout we
        # therefore must pass the ALIGNED counts (not the raw ``num_recv``) —
        # otherwise consecutive experts overlap each other's padding rows and
        # the GEMM reads garbage. Mirrors framework
        # ``deepgemm_hybrid_executor.py::execute_contiguous`` which builds a
        # GPU tensor of aligned counts before calling ep_scatter.
        aligned_counts = torch.tensor(
            aligned_counts_list,
            dtype=torch.int32, pin_memory=True, device="cpu",
        ).to(device, non_blocking=True)

        # (2) Triton ep_scatter: per-expert padded layout in 1 kernel pair.
        # Output scale is column-major TMA-aligned int32 (matches DeepGEMM
        # contiguous expectation when scale_ue8m0=True) — see framework's
        # deepgemm_hybrid_executor.py:427-432 for the same allocation pattern.
        scatter_out = torch.empty(
            (all_tokens, D), dtype=torch.float8_e4m3fn, device=device
        )
        scatter_out_scale = torch.zeros(
            [ceil_div(D // FP8_BLOCK, 4), all_tokens],
            device=device, dtype=torch.int,
        ).transpose(0, 1)
        # m_indices is fully overwritten by ep_scatter's kernel_1 (one expert_id
        # per row across the aligned region). Padding rows therefore tag a real
        # expert and DeepGEMM does (wasted) compute against it; ep_gather only
        # fetches the valid rows tracked in ``output_index`` so the wasted
        # output is discarded. Matches framework pattern.
        m_indices = torch.empty(all_tokens, dtype=torch.int32, device=device)
        output_index = torch.empty_like(adjusted_topk_ids)
        expert_start_loc = torch.empty_like(aligned_counts)
        ep_scatter(
            a_fp8,
            a_scale,
            adjusted_topk_ids,
            aligned_counts,
            expert_start_loc,
            scatter_out,
            scatter_out_scale,
            m_indices,
            output_index,
            scale_ue8m0=True,
        )
        # Defensive clamp against any -1 leakage (e.g. if num_local_experts is
        # later split for EP > 1); matches framework safety guard.
        m_indices.clamp_(min=0, max=E - 1)
        del a_fp8, a_scale

        # GEMM 1: gate+up
        gate_up = torch.empty(
            all_tokens, 2 * inter, device=device, dtype=torch.bfloat16
        )
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (scatter_out, scatter_out_scale),
            (self._w13, self._s13),
            gate_up,
            m_indices,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        del scatter_out, scatter_out_scale

        # (3) Fused SiLU+clamp+mul + per-token-group FP8 quant + UE8M0 packed scale.
        # Router weight is NOT applied here — the ep_gather below folds it
        # into the topk-reduce.
        h_fp8, h_scale = silu_mul_fp8_quant_packed(
            gate_up,
            clamp_limit=cfg.swiglu_limit,
            group_size=FP8_BLOCK,
        )
        del gate_up

        # GEMM 2: down
        down_out = torch.empty(
            all_tokens, D, device=device, dtype=torch.bfloat16
        )
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, self._s2),
            down_out,
            m_indices,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        del h_fp8, h_scale

        # (4) Triton ep_gather: per output token accumulates topk source rows
        # × router weight in fp32 register, single BF16 store. No
        # [N, topk, D] fp32 intermediate (legacy materialised ~700 MB at
        # N=4k, topk=6, D=7168).
        gather_out = torch.empty((N, D), dtype=torch.bfloat16, device=device)
        ep_gather(down_out, adjusted_topk_ids, weights, output_index, gather_out)
        return gather_out.float()
