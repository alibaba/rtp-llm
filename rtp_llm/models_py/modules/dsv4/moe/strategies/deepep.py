"""DeepEPStrategy: ACCL-EP normal-mode dispatch + per-expert local compute + combine.

EP > 1 DeepEP implementation. DSV4 automatic strategy selection never falls
back here when Mega is unavailable; this path requires the explicit
``DSV4_MOE_STRATEGY=deepep`` opt-in. Composes ``LocalLoopStrategy`` for the
local per-expert compute on the dispatched recv tokens.

Direct port of the pre-refactor ``_routed_experts_deepep`` +
``_pad_topk_for_deepep`` + the ``_DEEPEP_SUPPORTED_TOPK`` constant.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

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

    @staticmethod
    def _require_normal_wrapper(cfg: MoeCfg):
        if cfg.ep_size <= 1:
            raise RuntimeError(
                "DeepEPStrategy requires ep_size > 1; "
                f"got ep_size={cfg.ep_size} (layer_id={cfg.layer_id})."
            )

        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        if not DeepEPWrapper.is_initialized():
            raise RuntimeError(
                "DeepEPStrategy requires an initialized DeepEPWrapper, but the "
                "wrapper is not initialized. Call init_deepep_wrapper() at engine "
                f"startup (layer_id={cfg.layer_id}, ep_size={cfg.ep_size})."
            )
        if DeepEPWrapper._instance is None:
            raise RuntimeError(
                "DeepEPStrategy requires an initialized DeepEPWrapper, but the "
                "wrapper instance is missing despite initialized state "
                f"(layer_id={cfg.layer_id}, ep_size={cfg.ep_size})."
            )

        wrapper = DeepEPWrapper._instance
        if wrapper.mode != DeepEPMode.NORMAL:
            mode_name = getattr(wrapper.mode, "name", str(wrapper.mode))
            raise RuntimeError(
                "DeepEPStrategy requires DeepEPMode.NORMAL, but the initialized "
                f"wrapper uses {mode_name} (layer_id={cfg.layer_id}, "
                f"ep_size={cfg.ep_size})."
            )
        return wrapper

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        self._require_normal_wrapper(cfg)
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
        """Delegates to ``LocalLoopStrategy.setup_weights`` — DeepEP has no
        weights of its own; it dispatches recv tokens to the per-expert loop
        owned by the inner ``LocalLoopStrategy``.
        """
        self._local.setup_weights(layer_weights)

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
        wrapper = self._require_normal_wrapper(self.cfg)
        buf = wrapper.buffer
        cfg = self.cfg

        # Pad topk to nearest supported value (V4's 6 → 8).
        indices_p, weights_p = self._pad_topk_for_deepep(indices, weights)

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
        if M > 0:
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
        return y_combined.float()
