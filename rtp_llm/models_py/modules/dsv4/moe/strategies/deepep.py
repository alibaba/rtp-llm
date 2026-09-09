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

from typing import Dict, Tuple

import torch

from ..warmup_sync import (
    cuda_graph_warmup_forward_enabled,
    sync_cuda_graph_warmup_ranks,
)
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .local_loop import LocalLoopStrategy

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

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # ep_size > 1. Mega-vs-DeepEP priority is enforced by registry order
        # (Mega registered first).
        return cfg.ep_size > 1

    def setup_weights(self, layer_weights: Dict) -> None:
        self._local.setup_weights(layer_weights)

    def _forward_low_latency(self, x, weights, indices, wrapper):
        raise RuntimeError("This DeepEP strategy requires normal-mode dispatch")

    def _compute_local(
        self, recv_x, recv_weights, recv_indices, counts, *, fixed_shape
    ):
        cfg = self.cfg
        if recv_x.size(0) == 0:
            return torch.zeros(0, cfg.dim, dtype=torch.float32, device=recv_x.device)
        global_indices = recv_indices.to(torch.int64).contiguous()
        global_indices = torch.where(
            global_indices == -1,
            global_indices,
            global_indices + cfg.local_expert_start,
        )
        return self._local._forward_into_buf(
            recv_x.contiguous(),
            recv_weights.contiguous(),
            global_indices,
            local_start=cfg.local_expert_start,
            local_end=cfg.local_expert_end,
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
        # global expert id. Shift to global so the per-expert loop in
        # ``LocalLoopStrategy`` indexes ``self._local.experts[global_i]``
        # correctly. Also force int64 and contiguous — the ACCL tensor
        # sometimes comes back with a non-standard dtype that triggers
        # ``torch.where(idx == i)`` with "unknown parameter type".
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
