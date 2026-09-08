"""DeepEPStrategy: ACCL-EP normal-mode dispatch + per-expert local compute + combine.

EP > 1 DeepEP implementation. DSV4 automatic strategy selection no longer
falls back here when Mega is unavailable; EP>1 requires Mega and fails fast.
This class is kept as an explicit implementation for targeted tests or
experiments. Composes ``LocalLoopStrategy`` for the local per-expert compute
on the dispatched recv tokens.

Direct port of the pre-refactor ``_routed_experts_deepep`` +
``_pad_topk_for_deepep`` + the ``_DEEPEP_SUPPORTED_TOPK`` constant.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence, Tuple

import torch

from ..warmup_sync import (
    cuda_graph_warmup_forward_enabled,
    sync_cuda_graph_warmup_ranks,
)
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .local_loop import LocalLoopStrategy


def _ppu_grouped_fp4_enabled() -> bool:
    return os.environ.get("DSV4_PPU_GROUPED_FP4", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


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


# ACCL-EP's intranode dispatch kernel has a compile-time switch over
# ``num_topk`` that only covers {2, 4, 8, 16} (asserts false on others —
# intranode.cu:2237 "Unsupported num_topk"). V4-Flash uses
# ``n_activated_experts = 6``; we pad both ``indices`` and ``weights``
# up to 8 slots with ``-1`` and ``0.0`` so the dispatch accepts them,
# and the padding slots are silently dropped by the per-expert loop
# (``torch.where(idx == -1)`` never matches a real expert index).
_DEEPEP_SUPPORTED_TOPK = (2, 4, 8, 16)


@register_strategy
class DeepEPStrategy(RoutedExpertsStrategy):
    name = "deepep"

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        # Composition: hold a LocalLoopStrategy instance for the per-expert
        # local compute on dispatched recv tokens. Registered as a child
        # nn.Module so its ``experts`` ModuleList propagates through
        # ``MoE.to(device)`` / state_dict.
        self._local = LocalLoopStrategy(cfg)
        self._ppu_grouped_fp4 = False

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # ep_size > 1. Mega-vs-DeepEP priority is enforced by registry order
        # (Mega registered first).
        return cfg.ep_size > 1

    def setup_weights(self, layer_weights: Dict) -> None:
        """Delegates to ``LocalLoopStrategy.setup_weights`` — DeepEP has no
        weights of its own; it dispatches recv tokens to the per-expert loop
        owned by the inner ``LocalLoopStrategy``.
        """
        if not _ppu_grouped_fp4_enabled():
            self._local.setup_weights(layer_weights)
            return

        # M890P checkpoints keep routed experts as packed MXFP4. Preserve the
        # native payloads and prepare E8M0 checkpoint scales once, then execute
        # all local experts with two grouped GEMMs instead of 3*E small GEMMs.
        # The opt-in is deliberately strict: other storage geometries continue
        # to use LocalLoopStrategy and cannot silently enter this platform path.
        from rtp_llm.utils.model_weight import W

        w1 = layer_weights.pop(W.v4_routed_w1_w)
        s1 = layer_weights.pop(W.v4_routed_w1_s)
        w2 = layer_weights.pop(W.v4_routed_w2_w)
        s2 = layer_weights.pop(W.v4_routed_w2_s)
        w3 = layer_weights.pop(W.v4_routed_w3_w)
        s3 = layer_weights.pop(W.v4_routed_w3_s)
        cfg = self.cfg
        expected_w1 = (cfg.n_local_experts, cfg.moe_inter_dim, cfg.dim // 2)
        expected_w2 = (cfg.n_local_experts, cfg.dim, cfg.moe_inter_dim // 2)
        if tuple(w1.shape) != expected_w1 or tuple(w3.shape) != expected_w1:
            raise ValueError(
                f"PPU grouped-FP4 w1/w3 must have shape {expected_w1}, got "
                f"{tuple(w1.shape)}/{tuple(w3.shape)}"
            )
        if tuple(w2.shape) != expected_w2:
            raise ValueError(
                f"PPU grouped-FP4 w2 must have shape {expected_w2}, got {tuple(w2.shape)}"
            )
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if any(w.dtype not in (torch.int8, torch.uint8) for w in (w1, w2, w3)):
            raise TypeError("PPU grouped-FP4 requires packed int8/uint8 routed weights")
        if e8m0_dtype is None or any(s.dtype != e8m0_dtype for s in (s1, s2, s3)):
            raise TypeError("PPU grouped-FP4 requires float8_e8m0fnu checkpoint scales")
        if not w1.is_cuda or torch.cuda.get_device_name(w1.device) != "ZW-M890P":
            raise RuntimeError("DSV4_PPU_GROUPED_FP4=1 requires ZW-M890P weights")

        from rtp_llm.platforms.ppu.kernels.ppu_mxfp4 import (
            prepare_fp4_weight_scale_mxfp4,
        )

        self.register_buffer(
            "_ppu_w13",
            torch.cat((w1, w3), dim=1).view(torch.uint8).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_ppu_s13",
            prepare_fp4_weight_scale_mxfp4(torch.cat((s1, s3), dim=1).contiguous()),
            persistent=False,
        )
        self.register_buffer(
            "_ppu_w2", w2.view(torch.uint8).contiguous(), persistent=False
        )
        self.register_buffer(
            "_ppu_s2",
            prepare_fp4_weight_scale_mxfp4(s2.contiguous()),
            persistent=False,
        )
        self._ppu_grouped_fp4 = True

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
        """Run grouped MXFP4 experts on DeepEP LL's compact payload."""
        import deep_gemm
        from rtp_llm.models_py.modules.dsv4.moe.expert import require_silu_mul_split
        from rtp_llm.platforms.ppu.kernels.ppu_mxfp4 import downcast_to_mxfp4

        cfg = self.cfg
        packed_dispatch = isinstance(expert_x, tuple)
        if packed_dispatch:
            if len(expert_x) != 2:
                raise ValueError("DeepEP LL MXFP4 dispatch must return data and scales")
            packed_x, packed_scale = expert_x
            if packed_x.dim() != 3 or packed_scale.dim() != 3:
                raise ValueError("DeepEP LL MXFP4 tensors must both be rank 3")
            E, ll_capacity, packed_D = packed_x.shape
            D = packed_D * 2
            device = packed_x.device
        else:
            if not isinstance(expert_x, torch.Tensor) or expert_x.dim() != 3:
                raise ValueError(
                    "DeepEP LL input must be BF16 [E, M, D] or an MXFP4 tuple"
                )
            E, ll_capacity, D = expert_x.shape
            device = expert_x.device
        if (E, D) != (cfg.n_local_experts, cfg.dim):
            raise ValueError(
                f"grouped-FP4 packed input must be "
                f"[{cfg.n_local_experts}, M, {cfg.dim}], got E={E}, M={ll_capacity}, D={D}"
            )
        compute_capacity = int(os.environ.get("DSV4_PPU_GROUPED_FP4_CAPACITY", "128"))
        if (
            compute_capacity <= 0
            or compute_capacity > ll_capacity
            or (E * compute_capacity) % 128
        ):
            raise ValueError(
                "grouped-FP4 compute capacity must be positive, no larger than "
                "the DeepEP LL slot, and E*capacity divisible by 128"
            )
        inter = cfg.moe_inter_dim
        total = E * compute_capacity
        if expert_num_tokens.numel() != E:
            raise ValueError(
                f"grouped-FP4 expert counts must have {E} elements, "
                f"got {expert_num_tokens.numel()}"
            )
        safe_counts = (
            expert_num_tokens.clamp(min=0, max=compute_capacity)
            .to(torch.int32)
            .contiguous()
        )
        # DeepEP LL reserves ``max_tokens_per_rank * ep_size`` rows per expert
        # for a theoretical all-to-one route.  Quantizing that entire slot
        # would erase the LL communication win.  The production grouped-MXFP4
        # path already uses a conservative fixed expert capacity (128 by
        # default, >8x the measured B80 mean); copy just that prefix into a
        # compact graph-stable tensor and keep the original LL-shaped output
        # for combine.
        if packed_dispatch:
            x_fp4_grouped = packed_x[:, :compute_capacity, :].contiguous()
            # DeepEP exposes logical [E, LL_M, K/64] scales with mn-major
            # stride. Compact the physical [E, K/64, M] storage, then restore
            # the same logical view with the smaller M stride.
            x_scale_grouped = (
                packed_scale.permute(0, 2, 1)[:, :, :compute_capacity]
                .contiguous()
                .permute(0, 2, 1)
            )
        else:
            compact_x = expert_x[:, :compute_capacity, :].contiguous()
            flat_x = compact_x.view(total, D)
            x_fp4, x_scale = downcast_to_mxfp4(flat_x)
            x_fp4_grouped = x_fp4.view(E, compute_capacity, -1)
            x_scale_grouped = x_scale.as_strided(
                (E, compute_capacity, x_scale.size(1)),
                (compute_capacity, 1, total),
            )
            x_scale_grouped = (
                x_scale_grouped.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            )
        gate_up_grouped = torch.empty(
            (E, compute_capacity, 2 * inter),
            dtype=torch.bfloat16,
            device=device,
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
            (x_fp4_grouped, x_scale_grouped),
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
        hidden_fp4_grouped = hidden_fp4.view(E, compute_capacity, -1)
        hidden_scale_grouped = hidden_scale.as_strided(
            (E, compute_capacity, hidden_scale.size(1)),
            (compute_capacity, 1, total),
        )
        hidden_scale_grouped = (
            hidden_scale_grouped.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        )
        compact_down = torch.empty(
            (E, compute_capacity, D),
            dtype=torch.bfloat16,
            device=device,
        )
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (hidden_fp4_grouped, hidden_scale_grouped),
            (self._ppu_w2, self._ppu_s2),
            None,
            compact_down,
            safe_counts,
            expected_m,
        )
        return compact_down

    def _forward_ppu_grouped_fp4_low_latency(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        wrapper,
    ) -> torch.Tensor:
        """DeepEP LL packed dispatch → grouped MXFP4 experts → combine."""
        dispatch_args = {
            "x": x.contiguous(),
            "topk_idx": indices.to(torch.int64).contiguous(),
            "num_max_dispatch_tokens_per_rank": wrapper.ll_num_max_token_per_rank,
            "num_experts": self.cfg.n_routed_experts,
            "use_fp8": False,
            "use_mxfp4": True,
            "mxfp4_scale_row_major": False,
            "quant_size": 32,
            "async_finish": False,
            "return_recv_hook": False,
        }
        expert_x, expert_num_tokens, handle, _, _ = wrapper.buffer.low_latency_dispatch(
            **dispatch_args
        )
        if not isinstance(expert_x, tuple) or len(expert_x) != 2:
            raise RuntimeError("DeepEP LL MXFP4 dispatch must return (data, scale)")
        compact_expert_y = self._compute_ppu_grouped_fp4_packed(
            expert_x, expert_num_tokens
        )
        # The LL RDMA buffer already owns the required full slot geometry.  Do
        # not allocate another [E, ll_capacity, D] tensor (768 MiB for V4 at
        # gamma3); copy only the compact valid prefix into the next combine
        # buffer and let the handle's receive counts delimit the rows consumed.
        expert_y = wrapper.buffer.get_next_low_latency_combine_buffer(handle)
        expert_y[:, : compact_expert_y.size(1), :].copy_(compact_expert_y)
        combine_args = {
            "x": expert_y,
            "topk_idx": dispatch_args["topk_idx"],
            "topk_weights": weights.contiguous(),
            "handle": handle,
            "zero_copy": True,
            "async_finish": False,
            "return_recv_hook": False,
        }
        combined_x, _, _ = wrapper.buffer.low_latency_combine(**combine_args)
        return combined_x.float()

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
            if not getattr(self, "_ppu_grouped_fp4", False):
                raise RuntimeError(
                    "DSV4 DeepEP low-latency requires the M890P grouped-FP4 "
                    "executor (set DSV4_PPU_GROUPED_FP4=1)"
                )
            y_combined = self._forward_ppu_grouped_fp4_low_latency(
                x, weights_p, indices_p, wrapper
            )
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
        # global expert id. Shift to global so the per-expert loop in
        # ``LocalLoopStrategy`` indexes ``self._local.experts[global_i]``
        # correctly. Also force int64 and contiguous — the ACCL tensor
        # sometimes comes back with a non-standard dtype that triggers
        # ``torch.where(idx == i)`` with "unknown parameter type".
        M = recv_x.size(0)
        if M > 0 and getattr(self, "_ppu_grouped_fp4", False):
            configured_capacity = int(
                os.environ.get("DSV4_PPU_GROUPED_FP4_CAPACITY", "128")
            )
            grouped_capacity = _select_ppu_grouped_fp4_capacity(
                configured_capacity,
                num_recv_tokens_per_expert_list,
                cfg.n_local_experts,
                fixed_shape=(graph_warmup or capturing),
            )
            y_local = self._forward_ppu_grouped_fp4(
                recv_x.contiguous(),
                recv_topk_weights.contiguous(),
                recv_topk_idx.contiguous(),
                grouped_capacity,
            )
        elif M > 0:
            global_topk_idx = recv_topk_idx.to(torch.int64).contiguous()
            # Shift local→global; keep -1 as -1 (won't match any expert id).
            global_topk_idx = torch.where(
                global_topk_idx == -1,
                global_topk_idx,
                global_topk_idx + cfg.local_expert_start,
            )
            # _local.forward() allocates its own y_local buffer (its
            # _local_y_buf), runs the [local_start, local_end) loop, and
            # returns the fp32 accumulator. We pass through.
            y_local = self._local._forward_into_buf(
                recv_x.contiguous(),
                recv_topk_weights.contiguous(),
                global_topk_idx,
                local_start=cfg.local_expert_start,
                local_end=cfg.local_expert_end,
            )
        else:
            # M == 0: no recv tokens this rank — produce a fresh empty
            # fp32 accumulator so combine still has a valid tensor to send.
            y_local = torch.zeros(M, cfg.dim, dtype=torch.float32, device=recv_x.device)

        # 4. Combine back to source ranks. combine expects the tensor
        # dtype to match x (BF16) — cast the fp32 accumulator.
        y_combined, _, _ = buf.combine(
            y_local.to(x.dtype),
            handle,
        )
        if graph_warmup:
            sync_cuda_graph_warmup_ranks("deepep_after_combine", x.device)
        return y_combined.float()
