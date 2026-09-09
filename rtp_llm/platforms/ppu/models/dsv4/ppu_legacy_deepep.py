"""Legacy PPU grouped DeepEP execution, composed with common dispatch/combine."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies.deepep import DeepEPStrategy


def _select_ppu_grouped_fp4_capacity(
    configured_capacity: int,
    num_recv_tokens_per_expert: Sequence[int],
    n_local_experts: int,
    *,
    fixed_shape: bool,
) -> int:
    """Choose an aligned grouped-GEMM capacity without truncating eager Prefill.

    Decode graph capture/replay must retain the configured fixed shape. Eager
    Prefill already receives exact per-expert counts from DeepEP's dynamic
    dispatch CPU result, so grow the capacity to cover the busiest local
    expert and round it to the kernel's ``E * capacity`` alignment.
    """
    if configured_capacity <= 0 or n_local_experts <= 0:
        raise ValueError("grouped-FP4 capacity and local expert count must be positive")
    alignment = 128 // math.gcd(128, n_local_experts)
    if configured_capacity % alignment:
        raise ValueError(
            "DSV4_PPU_GROUPED_FP4_CAPACITY must make "
            "n_local_experts*capacity divisible by 128"
        )
    if fixed_shape:
        return configured_capacity
    if not num_recv_tokens_per_expert:
        raise ValueError("eager grouped-FP4 requires DeepEP CPU expert counts")
    required = max(int(count) for count in num_recv_tokens_per_expert)
    if required < 0:
        raise ValueError("DeepEP expert token counts must be non-negative")
    selected = max(configured_capacity, required)
    return ((selected + alignment - 1) // alignment) * alignment


class PpuLegacyDeepEPStrategy(DeepEPStrategy):
    name = "ppu_legacy_deepep_fp4"

    def __init__(self, cfg, *, options):
        super().__init__(cfg)
        self._grouped_capacity = int(
            options.get("DSV4_PPU_GROUPED_FP4_CAPACITY", "128")
        )

    def setup_weights(self, layer_weights):
        from .ppu_deepep_fp4 import prepare_routed_mxfp4_weights

        weights = prepare_routed_mxfp4_weights(self.cfg, layer_weights)
        for name, value in zip(("_ppu_w13", "_ppu_s13", "_ppu_w2", "_ppu_s2"), weights):
            self.register_buffer(name, value, persistent=False)

    def _forward_low_latency(self, x, weights, indices, wrapper):
        return self._forward_ppu_grouped_fp4_low_latency(x, weights, indices, wrapper)

    def _compute_local(
        self, recv_x, recv_weights, recv_indices, counts, *, fixed_shape
    ):
        if recv_x.size(0) == 0:
            return super()._compute_local(
                recv_x, recv_weights, recv_indices, counts, fixed_shape=fixed_shape
            )
        capacity = _select_ppu_grouped_fp4_capacity(
            self._grouped_capacity,
            counts,
            self.cfg.n_local_experts,
            fixed_shape=fixed_shape,
        )
        return self._forward_ppu_grouped_fp4(
            recv_x.contiguous(),
            recv_weights.contiguous(),
            recv_indices.contiguous(),
            capacity,
        )

    def _forward_ppu_grouped_fp4(
        self,
        recv_x: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """Fixed-capacity, CUDA-graph-safe grouped MXFP4 local expert compute."""
        import deep_gemm

        from rtp_llm.models_py.modules.dsv4.moe.expert import require_silu_mul_split
        from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
            ep_gather,
            ep_scatter_v2,
            recompute_topk_ids_sum_expert_count,
        )
        from rtp_llm.platforms.ppu.kernels.ppu_mxfp4 import downcast_to_mxfp4

        cfg = self.cfg
        M, D = recv_x.shape
        E = cfg.n_local_experts
        inter = cfg.moe_inter_dim
        if capacity <= 0 or (E * capacity) % 128:
            raise ValueError(
                "DSV4_PPU_GROUPED_FP4_CAPACITY must be positive and "
                "n_local_experts*capacity must be divisible by 128"
            )

        local_idx = recv_topk_idx.to(torch.int64).contiguous()
        adjusted_idx, actual_counts = recompute_topk_ids_sum_expert_count(
            local_idx, current_expert_start_id=0, num_local_experts=E
        )

        total = E * capacity
        safe_counts = actual_counts.clamp(max=capacity).to(torch.int32).contiguous()
        expert_start = torch.empty(E, dtype=torch.int32, device=recv_x.device)
        output_index = torch.full_like(adjusted_idx, -1, dtype=torch.int64)
        scatter_x = torch.zeros((total, D), dtype=recv_x.dtype, device=recv_x.device)
        # ep_scatter is dtype-generic. Dummy scales let us reuse its fixed
        # expert-layout/index kernel for BF16; MXFP4 quantization follows.
        dummy_in_scale = torch.zeros(
            (M, D // 128), dtype=torch.float32, device=recv_x.device
        )
        dummy_out_scale = torch.zeros(
            (E, capacity, D // 128), dtype=torch.float32, device=recv_x.device
        )
        ep_scatter_v2(
            recv_x,
            dummy_in_scale,
            adjusted_idx,
            capacity,
            expert_start,
            scatter_x,
            dummy_out_scale,
            output_index,
            scale_ue8m0=False,
        )

        scatter_fp4, scatter_scale = downcast_to_mxfp4(scatter_x.contiguous())
        scatter_fp4_grouped = scatter_fp4.view(E, capacity, -1)
        scatter_scale_grouped = scatter_scale.as_strided(
            (E, capacity, scatter_scale.size(1)),
            (capacity, 1, total),
        )
        # GroupedMasked assumes each expert owns a compact [K/64, M]
        # scale block; the flat packer instead owns one global [K/64, E*M]
        # block. Materialize the expert-major physical layout while retaining
        # the logical [E, M, K/64] mn-major view required by DeepGEMM.
        scatter_scale_grouped = (
            scatter_scale_grouped.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        )
        gate_up_grouped = torch.empty(
            (E, capacity, 2 * inter),
            dtype=torch.bfloat16,
            device=recv_x.device,
        )
        expected_m = max(
            1,
            (
                cfg.max_tokens_per_rank * cfg.ep_size * cfg.n_activated_experts
                + cfg.n_routed_experts
                - 1
            )
            // cfg.n_routed_experts,
        )
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (scatter_fp4_grouped, scatter_scale_grouped),
            (self._ppu_w13, self._ppu_s13),
            None,
            gate_up_grouped,
            safe_counts,
            expected_m,
        )
        gate_up = gate_up_grouped.view(total, 2 * inter)
        hidden = (
            require_silu_mul_split()(
                gate_up[:, :inter].float().contiguous(),
                gate_up[:, inter:].float().contiguous(),
                clamp_limit=cfg.swiglu_limit,
            )
            .to(torch.bfloat16)
            .contiguous()
        )
        hidden_fp4, hidden_scale = downcast_to_mxfp4(hidden)
        hidden_fp4_grouped = hidden_fp4.view(E, capacity, -1)
        hidden_scale_grouped = hidden_scale.as_strided(
            (E, capacity, hidden_scale.size(1)),
            (capacity, 1, total),
        )
        hidden_scale_grouped = (
            hidden_scale_grouped.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        )
        down_grouped = torch.empty(
            (E, capacity, D), dtype=torch.bfloat16, device=recv_x.device
        )
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (hidden_fp4_grouped, hidden_scale_grouped),
            (self._ppu_w2, self._ppu_s2),
            None,
            down_grouped,
            safe_counts,
            expected_m,
        )
        down = down_grouped.view(total, D)
        gathered = torch.empty((M, D), dtype=torch.bfloat16, device=recv_x.device)
        ep_gather(
            down,
            adjusted_idx,
            recv_topk_weights.contiguous(),
            output_index,
            gathered,
        )
        return gathered.float()

    def _compute_ppu_grouped_fp4_packed(
        self,
        expert_x,
        expert_num_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compatibility entry point using full-slot PPU masked execution."""
        from rtp_llm.platforms.ppu.kernels.ppu_mxfp4_masked import mxfp4_experts_masked

        cfg = self.cfg
        expected_m = max(
            1,
            (
                cfg.max_tokens_per_rank * cfg.ep_size * cfg.n_activated_experts
                + cfg.n_routed_experts
                - 1
            )
            // cfg.n_routed_experts,
        )
        return mxfp4_experts_masked(
            expert_x,
            (self._ppu_w13, self._ppu_s13),
            (self._ppu_w2, self._ppu_s2),
            expert_num_tokens,
            expected_m=expected_m,
            swiglu_limit=cfg.swiglu_limit if cfg.swiglu_limit > 0 else None,
        )

    def _forward_ppu_grouped_fp4_low_latency(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        wrapper,
    ) -> torch.Tensor:
        """Legacy opt-in delegates to the public platform's full-slot adapter."""
        from rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency import (
            low_latency_mxfp4_moe,
        )

        cfg = self.cfg
        expected_m = max(
            1,
            (
                cfg.max_tokens_per_rank * cfg.ep_size * cfg.n_activated_experts
                + cfg.n_routed_experts
                - 1
            )
            // cfg.n_routed_experts,
        )
        return low_latency_mxfp4_moe(
            wrapper.buffer,
            x,
            weights,
            indices,
            (self._ppu_w13, self._ppu_s13),
            (self._ppu_w2, self._ppu_s2),
            num_experts=cfg.n_routed_experts,
            max_dispatch_tokens=wrapper.ll_num_max_token_per_rank,
            expected_m=expected_m,
            swiglu_limit=cfg.swiglu_limit if cfg.swiglu_limit > 0 else None,
        )
