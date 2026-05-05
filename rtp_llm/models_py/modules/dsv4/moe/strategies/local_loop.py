"""LocalLoopStrategy: Python-level for-loop over per-expert ``Expert`` modules.

Universal fallback path:
  - ep_size == 1 without the grouped FP4 kernel (also composed inside
    ``DeepEPStrategy`` for the local compute on dispatched recv tokens)
  - cuda-graph capture (forward dispatches to graph-safe variant internally)

Owns the ``self._local_y_buf`` fp32 accumulator (lazy-allocated, reused
across forward calls). Same buffer is reused under cuda-graph capture.

Wired into ``MoE`` via ``select_strategy`` as the universal fallback (and
also composed inside ``DeepEPStrategy`` for local recv-token compute).
Direct port of the pre-refactor ``_routed_experts_local{,_graph_safe}``
methods + matching ModuleList construction + forward buffer-management.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..expert import Expert
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy


@register_strategy
class LocalLoopStrategy(RoutedExpertsStrategy):
    name = "local_loop"

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # Universal fallback — accepts every cfg. Strategy registry order
        # ensures higher-priority paths get picked first.
        return True

    def setup_weights(self, layer_weights: Dict) -> None:
        """Build per-expert ``Expert`` ModuleList from EP-sliced stacks.

        Pops keys: ``W.v4_routed_w{1,2,3}_{w,s}`` from ``layer_weights``
        (each shaped ``[E_local, ...]``). Slots for non-local experts stay
        ``None``; preserves V4-official indexing convention
        (``self.experts[global_idx]``) so forward loops stay identical
        across ranks.
        """
        from rtp_llm.utils.model_weight import W

        cfg = self.cfg
        stacked_routed = {
            "w1_w": layer_weights.pop(W.v4_routed_w1_w),
            "w1_s": layer_weights.pop(W.v4_routed_w1_s),
            "w2_w": layer_weights.pop(W.v4_routed_w2_w),
            "w2_s": layer_weights.pop(W.v4_routed_w2_s),
            "w3_w": layer_weights.pop(W.v4_routed_w3_w),
            "w3_s": layer_weights.pop(W.v4_routed_w3_s),
        }

        def _expert_at(global_idx: int) -> Optional[Expert]:
            if not (cfg.local_expert_start <= global_idx < cfg.local_expert_end):
                return None
            local_idx = global_idx - cfg.local_expert_start
            ew = {
                "w1_w": stacked_routed["w1_w"][local_idx],
                "w1_s": stacked_routed["w1_s"][local_idx],
                "w2_w": stacked_routed["w2_w"][local_idx],
                "w2_s": stacked_routed["w2_s"][local_idx],
                "w3_w": stacked_routed["w3_w"][local_idx],
                "w3_s": stacked_routed["w3_s"][local_idx],
            }
            return Expert(
                cfg.dim,
                cfg.moe_inter_dim,
                swiglu_limit=cfg.swiglu_limit,
                storage="fp4",
                expert_weights=ew,
            )

        self.experts = nn.ModuleList(
            [_expert_at(i) for i in range(cfg.n_routed_experts)]
        )

        # Lazy fp32 accumulator buffer — replaces a per-forward
        # ``torch.zeros_like(x, fp32)``. Sized to max_tokens_per_rank on first
        # use; subsequent calls slice ``[:T]`` and zero only the live prefix.
        # Eliminates one ``FillFunctor<float>`` per MoE layer per forward
        # (kernel #7 cluster in V4 prefill timeline). Also makes this strategy
        # safe under cuda-graph capture (no fresh allocation per replay).
        self._local_y_buf: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert compute over [local_expert_start, local_expert_end).

        Internally dispatches:
          - cuda-graph capture: ``_forward_graph_safe`` (fixed-shape mask compute
            avoids data-dependent ``torch.where``)
          - eager: ``_forward_eager`` (only iterates routed tokens per expert)
        """
        return self._forward_into_buf(
            x, weights, indices,
            local_start=self.cfg.local_expert_start,
            local_end=self.cfg.local_expert_end,
        )

    def _forward_into_buf(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        local_start: int,
        local_end: int,
    ) -> torch.Tensor:
        """Allocate / reuse the fp32 accumulator buffer + dispatch to
        eager / graph-safe variant. Public to ``DeepEPStrategy`` so it can
        run local compute on dispatched recv tokens with custom local range.
        """
        T = x.size(0)
        buf = self._local_y_buf
        if buf is None or buf.size(0) < T or buf.device != x.device:
            self._local_y_buf = torch.empty(
                (max(T, self.cfg.max_tokens_per_rank), self.cfg.dim),
                dtype=torch.float32,
                device=x.device,
            )
            buf = self._local_y_buf
        y = buf[:T]
        y.zero_()
        if torch.cuda.is_current_stream_capturing():
            self._forward_graph_safe(x, weights, indices, y, local_start, local_end)
        else:
            self._forward_eager(x, weights, indices, y, local_start, local_end)
        return y

    def _forward_eager(
        self,
        x: torch.Tensor,        # [N, D]
        weights: torch.Tensor,  # [N, k] fp32
        indices: torch.Tensor,  # [N, k] int64 — GLOBAL expert IDs
        y: torch.Tensor,        # [N, D] fp32, accumulator
        local_start: int,
        local_end: int,
    ) -> torch.Tensor:
        """Per-expert compute restricted to ``[local_start, local_end)``.

        Accumulates into ``y`` in-place; returns ``y`` for chaining.
        """
        for i in range(local_start, local_end):
            expert = self.experts[i]
            if expert is None:
                continue
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            y[idx] = y[idx] + expert(x[idx], weights[idx, top, None]).float()
        return y

    def _forward_graph_safe(
        self,
        x: torch.Tensor,        # [N, D]
        weights: torch.Tensor,  # [N, k] fp32
        indices: torch.Tensor,  # [N, k] int64 — GLOBAL expert IDs
        y: torch.Tensor,        # [N, D] fp32, accumulator
        local_start: int,
        local_end: int,
    ) -> torch.Tensor:
        """Graph-safe variant of :meth:`_forward_eager`.

        Uses fixed-shape per-expert mask compute instead of
        ``torch.where(indices == i)`` — the latter returns a
        data-dependent-shape result that triggers a CPU sync during
        ``cudaStreamCapture``.

        Per expert i:
          * ``mask[N]``      = True iff any topk slot of token routes to i
          * ``per_token_w[N, 1]`` = sum of router weights on slots == i
            (zero for tokens not routed to i)
          * ``Expert.forward(x, per_token_w)`` applies ``per_token_w *
            (silu(gate)*up)`` BEFORE the down projection — so unrouted
            tokens contribute exactly zero without explicit masking.

        Inefficiency: every expert sees every token (vs. only routed
        tokens in the eager path). For decode (N ≤ max_bs ~32) the
        per-call overhead dominates anyway, so the wasted FP4 GEMM cost
        is small. The Python ``for i in range(...)`` loop unrolls during
        graph capture into a static sequence of kernel launches.
        """
        for i in range(local_start, local_end):
            expert = self.experts[i]
            if expert is None:
                continue
            mask = (indices == i).to(weights.dtype)  # [N, k] fp32 0/1
            per_token_w = (weights * mask).sum(dim=-1, keepdim=True)  # [N, 1]
            # In-place accumulation: caller's y aliases the accumulator.
            # ``y = y + ...`` would rebind the local name and silently drop
            # all routed-expert contributions.
            y.add_(expert(x, per_token_w).float())
        return y
