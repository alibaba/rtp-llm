"""Legacy PPU grouped DeepEP execution, composed with common dispatch/combine."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.warmup_sync import (
    cuda_graph_warmup_forward_enabled,
    sync_cuda_graph_warmup_ranks,
)

_DEEPEP_SUPPORTED_TOPK = (2, 4, 8, 16)


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


class PpuLegacyDeepEPStrategy(torch.nn.Module):
    name = "ppu_legacy_deepep_fp4"

    def __init__(self, cfg, *, options):
        super().__init__()
        self.cfg = cfg
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
            return torch.zeros(
                0, self.cfg.dim, dtype=torch.float32, device=recv_x.device
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

        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.expert import (
            require_silu_mul_split,
        )
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

    @staticmethod
    def _pad_topk_for_deepep(
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad ``(indices, weights)`` to the nearest supported topk width.

        See ``_DEEPEP_SUPPORTED_TOPK`` docstring above.
        """
        n_act = indices.size(-1)
        if n_act in _DEEPEP_SUPPORTED_TOPK:
            return indices, weights
        pad_to = next((k for k in _DEEPEP_SUPPORTED_TOPK if k > n_act), None)
        if pad_to is None:
            raise RuntimeError(
                f"n_activated_experts={n_act} exceeds largest DeepEP-supported "
                f"topk ({max(_DEEPEP_SUPPORTED_TOPK)})"
            )
        N = indices.size(0)
        pad_n = pad_to - n_act
        pad_idx = torch.full((N, pad_n), -1, dtype=indices.dtype, device=indices.device)
        pad_w = torch.zeros((N, pad_n), dtype=weights.dtype, device=weights.device)
        return (
            torch.cat([indices, pad_idx], dim=-1),
            torch.cat([weights, pad_w], dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,  # [N, D] local rank's tokens (BF16)
        weights: torch.Tensor,  # [N, k] fp32
        indices: torch.Tensor,  # [N, k] int64 global expert IDs
    ) -> torch.Tensor:
        """DP+EP path: DeepEP normal dispatch → local per-expert compute
        → DeepEP combine. Requires ``init_deepep_wrapper`` to have been
        called by the engine (``backend_manager.py``).
        """
        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        if DeepEPWrapper._instance is None:
            raise RuntimeError(
                "DeepEPWrapper not initialised; ep_size>1 requires "
                "init_deepep_wrapper() at engine startup (enable via "
                "--use_deepep_moe 1)."
            )
        wrapper = DeepEPWrapper._instance
        buf = wrapper.buffer
        cfg = self.cfg

        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        graph_warmup = cuda_graph_warmup_forward_enabled()
        if graph_warmup:
            sync_cuda_graph_warmup_ranks("deepep_before_dispatch", x.device)

        # Pad topk to nearest supported value (V4's 6 → 8).
        indices_p, weights_p = self._pad_topk_for_deepep(indices, weights)

        if wrapper.mode == DeepEPMode.LOW_LATENCY:
            y_combined = self._forward_low_latency(x, weights_p, indices_p, wrapper)
            if graph_warmup:
                sync_cuda_graph_warmup_ranks("deepep_after_combine", x.device)
            return y_combined
        if wrapper.mode != DeepEPMode.NORMAL:
            raise RuntimeError(f"unsupported DeepEP mode for DSV4: {wrapper.mode}")

        # 1. Dispatch layout. indices cast to int64 already.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buf.get_dispatch_layout(indices_p, cfg.n_routed_experts)

        # 2. Dispatch the BF16 tokens + topk scaffolding.
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            _,
        ) = buf.dispatch(
            x,
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            indices_p,
            weights_p,
            expert_alignment=1,
            # DeepEP's normal dispatch otherwise synchronizes the dynamic
            # receive count through the CPU. Fixed worst-case capacity keeps
            # warmup, capture, and replay shapes identical and uses the
            # graph-safe no-CPU-sync kernel path. DP ranks can have different
            # local batches (real batch 2 beside fake batch 1), so capacity
            # must use the common startup budget rather than this rank's x.
            # Otherwise a rank captured at batch 1 reserves 8 rows although
            # the EP group can send it 9, which deadlocks ACCL-EP replay.
            num_worst_tokens=(
                (int(cfg.max_tokens_per_rank) * cfg.ep_size)
                if (graph_warmup or capturing)
                else 0
            ),
        )

        # 3. Local per-expert compute. ACCL-EP's dispatch returns
        # ``recv_topk_idx`` in the LOCAL index space ``[0, n_local_experts)``
        # (with -1 for tokens not destined for any local expert), NOT the
        # global expert id. The PPU grouped compute consumes this local
        # index space directly.
        y_local = self._compute_local(
            recv_x,
            recv_topk_weights,
            recv_topk_idx,
            num_recv_tokens_per_expert_list,
            fixed_shape=graph_warmup or capturing,
        )

        # 4. Combine back to source ranks. combine expects the tensor
        # dtype to match x (BF16) — cast the fp32 accumulator.
        y_combined, _, _ = buf.combine(
            y_local.to(x.dtype),
            handle,
        )
        if graph_warmup:
            sync_cuda_graph_warmup_ranks("deepep_after_combine", x.device)
        return y_combined.float()
