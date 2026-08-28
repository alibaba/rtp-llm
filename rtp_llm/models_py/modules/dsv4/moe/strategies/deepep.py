"""DeepEP normal-mode dispatch, local expert compute, and combine.

This module contains only the real DeepEP implementation. SM120 is handled
by ``Sm120FusedMoeStrategy`` because the installed CUDA 13 DeepEP binary has
no SM120-compatible cubin.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .local_loop import LocalLoopStrategy
from rtp_llm.models_py.utils.arch import is_sm120


# ACCL-EP's intranode dispatch kernel has a compile-time switch over
# ``num_topk`` that only covers {2, 4, 8, 16}. V4-Flash uses topk=6, so pad
# ids/weights to 8; -1 ids and zero weights are discarded by local compute.
_DEEPEP_SUPPORTED_TOPK = (2, 4, 8, 16)


@register_strategy
class DeepEPStrategy(RoutedExpertsStrategy):
    name = "deepep"

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        # DeepEP owns dispatch/combine; the rank-local expert weights and
        # compute live in this child module.
        self._local = LocalLoopStrategy(cfg)

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # The installed DeepEP binary has no SM120 cubin. Exclude it during
        # selection so an explicit ``deepep`` request fails before model load.
        return cfg.ep_size > 1 and not is_sm120()

    def setup_weights(self, layer_weights: Dict) -> None:
        self._local.setup_weights(layer_weights)

    @staticmethod
    def _pad_topk_for_deepep(
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_act = indices.size(-1)
        if n_act in _DEEPEP_SUPPORTED_TOPK:
            return indices, weights
        pad_to = next((k for k in _DEEPEP_SUPPORTED_TOPK if k > n_act), None)
        if pad_to is None:
            raise RuntimeError(
                f"n_activated_experts={n_act} exceeds largest DeepEP-supported "
                f"topk ({max(_DEEPEP_SUPPORTED_TOPK)})"
            )
        num_tokens = indices.size(0)
        pad_n = pad_to - n_act
        pad_idx = torch.full(
            (num_tokens, pad_n),
            -1,
            dtype=indices.dtype,
            device=indices.device,
        )
        pad_w = torch.zeros(
            (num_tokens, pad_n),
            dtype=weights.dtype,
            device=weights.device,
        )
        return (
            torch.cat([indices, pad_idx], dim=-1),
            torch.cat([weights, pad_w], dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """DeepEP dispatch, local expert compute, and DeepEP combine."""

        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        if DeepEPWrapper._instance is None:
            raise RuntimeError(
                "DeepEPWrapper not initialised; enable --use_deepep_moe 1 "
                "on a device supported by the installed DeepEP binary."
            )
        wrapper = DeepEPWrapper._instance
        if wrapper.mode != DeepEPMode.NORMAL:
            raise RuntimeError(f"expected NORMAL DeepEP mode, got {wrapper.mode}")
        buf = wrapper.buffer
        cfg = self.cfg

        indices_p, weights_p = self._pad_topk_for_deepep(indices, weights)
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buf.get_dispatch_layout(indices_p, cfg.n_routed_experts)

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            _,
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

        # DeepEP returns local expert ids. Shift them into the global id space
        # expected by LocalLoopStrategy, while preserving -1 sentinels.
        num_recv = recv_x.size(0)
        if num_recv > 0:
            global_topk_idx = recv_topk_idx.to(torch.int64).contiguous()
            global_topk_idx = torch.where(
                global_topk_idx == -1,
                global_topk_idx,
                global_topk_idx + cfg.local_expert_start,
            )
            y_local = self._local._forward_into_buf(
                recv_x.contiguous(),
                recv_topk_weights.contiguous(),
                global_topk_idx,
                local_start=cfg.local_expert_start,
                local_end=cfg.local_expert_end,
            )
        else:
            y_local = torch.zeros(
                (0, cfg.dim),
                dtype=torch.float32,
                device=recv_x.device,
            )

        y_combined, _, _ = buf.combine(y_local.to(x.dtype), handle)
        return y_combined.float()


__all__ = ["DeepEPStrategy"]
