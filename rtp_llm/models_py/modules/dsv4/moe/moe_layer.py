"""DeepSeek-V4 MoE block: routed top-k experts + 1 shared expert.

This orchestrator owns:
  - ``self.gate``: ``Gate`` module (routing scores → topk)
  - ``self.shared_experts``: ``Expert`` (FP8 shared expert)
  - ``self._strategy``: a ``RoutedExpertsStrategy`` chosen by ``select_strategy``

Forward: gate → strategy.forward → shared_y add. The 4-way if/elif from
legacy is gone; per-strategy weight loading lives inside
``strategy.setup_weights``. Debug instrumentation (``_record_tensor``) is
preserved verbatim.

Public-surface contract preserved:
  - ``MoE`` constructor accepts the same 16 positional+keyword args used by
    ``dsv4/block.py:167`` (no rename, no removal). Plus a new optional
    ``strategy`` kwarg for tests / forced strategy selection.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .expert import Expert
from .gate import Gate
from .strategies.base import MoeCfg, _resolve_forced, select_strategy


class MoE(nn.Module):
    """V4 MoE block: routed top-k experts + 1 shared expert.

    For TP=1 / EP=1 we instantiate ALL experts locally (LocalLoopStrategy)
    or the grouped FP4 stacked tensors (GroupedFP4Strategy). For sharded
    setups only ``[start, end)`` of routed experts live on this rank —
    DeepEPStrategy or MegaMoEStrategy handles the cross-rank dispatch.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: str,
        route_scale: float,
        swiglu_limit: float,
        n_hash_layers: int,
        vocab_size: int,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        ep_size: int = 1,
        ep_rank: int = 0,
        max_tokens_per_rank: int = 8192,
        strategy: Optional[str] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.moe_inter_dim = moe_inter_dim
        self.swiglu_limit = swiglu_limit
        self.max_tokens_per_rank = max_tokens_per_rank
        factory_mode = weights is not None
        self._factory_mode = factory_mode

        assert (
            n_routed_experts % max(ep_size, 1) == 0
        ), f"n_routed_experts={n_routed_experts} must divide ep_size={ep_size}"
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.n_local_experts = n_routed_experts // max(ep_size, 1)
        self.local_expert_start = ep_rank * self.n_local_experts
        self.local_expert_end = self.local_expert_start + self.n_local_experts

        self.gate = Gate(
            layer_id,
            dim,
            n_routed_experts,
            n_activated_experts,
            score_func,
            route_scale,
            n_hash_layers,
            vocab_size,
            weights=weights,
            prefix=f"{prefix}.gate" if factory_mode else "",
        )
        assert n_shared_experts == 1, "V4 always has exactly 1 shared expert"
        self.shared_experts = Expert(
            dim,
            moe_inter_dim,
            swiglu_limit=0.0,
            storage="fp8",
            weights=weights,
            prefix=f"{prefix}.shared_experts" if factory_mode else "",
        )

        # --- Strategy selection + weight setup ---
        cfg = MoeCfg(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            swiglu_limit=swiglu_limit,
            ep_size=ep_size,
            ep_rank=ep_rank,
            n_local_experts=self.n_local_experts,
            local_expert_start=self.local_expert_start,
            local_expert_end=self.local_expert_end,
            max_tokens_per_rank=max_tokens_per_rank,
            factory_mode=factory_mode,
        )
        forced = _resolve_forced(strategy)
        strategy_cls = select_strategy(cfg, forced=forced)
        # Register strategy as a child nn.Module so its child weights
        # (e.g. LocalLoopStrategy.experts ModuleList) propagate through
        # ``MoE.to(device)``.
        self._strategy = strategy_cls(cfg)
        self._strategy.setup_weights(weights, prefix)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        # Master switch: when MOEDBG=0 the AND short-circuits so neither the
        # layer_id compare nor any record_if_level call site below runs.
        # Instruments layers 0..2 (first CSA layer is L2) when enabled.
        _dbg = _rt.should_record_layer(self.layer_id)
        shape = x.size()
        x = x.view(-1, self.dim)
        dbg_pos = getattr(_rt, "_DBG_GLOBAL_POS", -1)
        dbg_pos_mask = None
        dbg_pos_name = None
        dbg_positions = getattr(self, "_dbg_positions", None)
        if _dbg and dbg_pos >= 0 and dbg_positions is not None:
            dbg_positions = dbg_positions.to(device=x.device, dtype=torch.long).view(-1)
            if dbg_positions.numel() == x.size(0):
                dbg_pos_mask = dbg_positions == int(dbg_pos)
                dbg_pos_name = f"pos{dbg_pos}"
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_x_in", x)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_moe_x_in_{dbg_pos_name}",
                    x[dbg_pos_mask].contiguous(),
                )
            self.gate._dbg_prefix = f"L{self.layer_id:02d}_moe_gate"
        weights, indices = self.gate(x, input_ids.flatten())
        if _dbg:
            self.gate._dbg_prefix = None
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_topk_weights", weights)
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_topk_indices", indices)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_moe_topk_weights_{dbg_pos_name}",
                    weights[dbg_pos_mask].contiguous(),
                )
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_moe_topk_indices_{dbg_pos_name}",
                    indices[dbg_pos_mask].contiguous(),
                )

        # GroupedFP4 path uses bincount/cumsum/argsort + .item() — these
        # abort cuda-stream capture. The legacy code asserted this at the
        # call site; we keep the assert here for the exact same dev-error
        # behavior. (Future: Phase 2 removes the .item() and lets grouped
        # fall through cleanly under capture.)
        if self._strategy.name == "grouped_fp4" and torch.cuda.is_current_stream_capturing():
            raise AssertionError(
                "grouped FP4 path uses bincount/cumsum/argsort which abort "
                "cuda-stream capture; do not enable cuda_graph + "
                "DSV4_USE_GROUPED_FP4=1 together (grouped is a prefill "
                "optimisation, decode-under-graph should keep the env off "
                "to fall through to LocalLoopStrategy)."
            )

        y = self._strategy(x, weights, indices)

        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_routed_y", y)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_moe_routed_y_{dbg_pos_name}",
                    y[dbg_pos_mask].contiguous(),
                )
        shared_y = self.shared_experts(x).float()
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_shared_y", shared_y)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_moe_shared_y_{dbg_pos_name}",
                    shared_y[dbg_pos_mask].contiguous(),
                )
        y = y + shared_y
        if _dbg and dbg_pos_mask is not None:
            _rt.record_if_level(
                2,
                f"L{self.layer_id:02d}_moe_y_{dbg_pos_name}",
                y[dbg_pos_mask].contiguous(),
            )
        return y.type_as(x).view(shape)
