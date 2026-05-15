"""DeepSeek-V4 MoE block: routed top-k experts + 1 shared expert.

This orchestrator owns:
  - ``self.gate``: ``Gate`` module (routing scores → topk)
  - ``self.shared_experts``: ``W13SharedExpert`` (FP8 shared expert)
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

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

from .gate import Gate
from .shared_expert import (
    W13SharedExpert,
    combine_routed_and_shared,
    get_shared_expert_executor,
)
from .strategies.base import MoeCfg, _resolve_forced, select_strategy

_FINAL_OUT_CACHE: dict[tuple, torch.Tensor] = {}
_CHUNKED_MOE_LOGGED = False

# Default per-rank MoE prefill chunk size for DeepSeek-V4-Flash long-context
# serving.  With 1M context and CP=4, a rank can see up to 262144 local tokens.
# Keeping MegaMoE/shared-expert workspaces sized for all of them is the OOM
# source this feature addresses. 65536 bounds the persistent MegaMoE symmetric
# buffer while keeping chunks large enough to avoid excessive
# kernel/dispatch overhead. Override with
# DSV4_MOE_CHUNK_TOKENS for smoke tests or tighter HBM budgets.
DEFAULT_MOE_CHUNK_TOKENS = 65536


def chunked_moe_enabled() -> bool:
    return os.environ.get("DSV4_MOE_CHUNK_PREFILL", "1") != "0"


def moe_chunk_tokens_from_env(default: int = DEFAULT_MOE_CHUNK_TOKENS) -> int:
    raw_value = os.environ.get("DSV4_MOE_CHUNK_TOKENS", str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logging.warning(
            "[DSV4 MoE] invalid DSV4_MOE_CHUNK_TOKENS=%r; using default=%d",
            raw_value,
            default,
        )
        value = default
    return max(value, 1)


def cp_padded_tokens_per_rank_bound(max_seq_len: int, cp_size: int) -> int:
    """Rank-local CP token upper bound, including ZigZag padding.

    The C++ ZigZagProcessor pads each request to a multiple of
    ``cp_size * 2`` so every rank gets an even-length local chunk.  A plain
    ``max_seq_len // cp_size`` underestimates lengths such as 200002 with
    CP=4: the local padded chunk is 50002, not 50000.
    """
    cp_size = max(int(cp_size), 1)
    max_seq_len = max(int(max_seq_len), 0)
    if cp_size <= 1:
        return max_seq_len
    if max_seq_len == 0:
        return 0
    global_alignment = cp_size * 2
    padded_seq_len = (
        (max_seq_len + global_alignment - 1) // global_alignment
    ) * global_alignment
    return padded_seq_len // cp_size


def resolve_moe_max_tokens_per_rank(
    max_seq_len: int,
    current_max_tokens_per_rank: int,
    cp_size: int,
    max_generate_batch_size: int,
    role_type: str | None = None,
) -> int:
    budget = int(current_max_tokens_per_rank)
    cp_size = max(int(cp_size), 1)
    if cp_size > 1:
        cp_bound = max(cp_padded_tokens_per_rank_bound(max_seq_len, cp_size), 4096)
        budget = min(budget, cp_bound)

    role = (
        role_type if role_type is not None else os.environ.get("ROLE_TYPE", "")
    ).upper()
    if role == "DECODE":
        return min(budget, max(int(max_generate_batch_size or 1), 1))
    if chunked_moe_enabled():
        return min(budget, moe_chunk_tokens_from_env())
    return budget


def _get_or_create_final_out(
    capacity: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (device, dim, dtype)
    cached = _FINAL_OUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), dim), dtype=dtype, device=device)
    _FINAL_OUT_CACHE[key] = cached
    return cached


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
        layer_weights: Optional[Dict] = None,
        ep_size: int = 1,
        ep_rank: int = 0,
        max_tokens_per_rank: int = 8192,
        strategy: Optional[str] = None,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``) keyed by ``W.v4_*`` enum.
        Forwards router/shared sub-dicts to ``Gate``/``Expert``; reads the
        stacked routed tensors ``W.v4_routed_w{1,2,3}_{w,s}`` (shape
        ``[E_local, ...]``) directly into mega-moe / grouped-fp4 / per-expert
        paths via the chosen strategy."""
        super().__init__()
        assert layer_weights is not None, (
            "MoE requires layer_weights (descriptor path); legacy "
            "weights/prefix dict path was removed."
        )
        self.layer_id = layer_id
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.moe_inter_dim = moe_inter_dim
        self.swiglu_limit = swiglu_limit
        self.max_tokens_per_rank = max_tokens_per_rank

        assert (
            n_routed_experts % max(ep_size, 1) == 0
        ), f"n_routed_experts={n_routed_experts} must divide ep_size={ep_size}"
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.n_local_experts = n_routed_experts // max(ep_size, 1)
        self.local_expert_start = ep_rank * self.n_local_experts
        self.local_expert_end = self.local_expert_start + self.n_local_experts

        from rtp_llm.utils.model_weight import W

        self.gate = Gate(
            layer_id,
            dim,
            n_routed_experts,
            n_activated_experts,
            score_func,
            route_scale,
            n_hash_layers,
            vocab_size,
            layer_weights=layer_weights,
        )
        assert n_shared_experts == 1, "V4 always has exactly 1 shared expert"
        shared_w = {
            "w13_w": layer_weights[W.v4_shared_w13_w],
            "w13_s": layer_weights[W.v4_shared_w13_s],
            "w2_w": layer_weights[W.v4_shared_w2_w],
            "w2_s": layer_weights[W.v4_shared_w2_s],
        }
        self.shared_experts = W13SharedExpert(
            dim,
            moe_inter_dim,
            expert_weights=shared_w,
            swiglu_limit=swiglu_limit,
        )
        self._shared_executor = get_shared_expert_executor(
            max_tokens_per_rank=max_tokens_per_rank,
            dim=dim,
            inter_dim=moe_inter_dim,
            swiglu_limit=swiglu_limit,
        )
        self._shared_executor.prepare(self.shared_experts)
        self._final_out: torch.Tensor | None = None

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
        )
        forced, strict = _resolve_forced(strategy)
        strategy_cls = select_strategy(cfg, forced=forced, strict=strict)
        # Register strategy as a child nn.Module so its child weights
        # (e.g. LocalLoopStrategy.experts ModuleList) propagate through
        # ``MoE.to(device)``.
        self._strategy = strategy_cls(cfg)
        self._strategy.setup_weights(layer_weights)

    def _should_chunk(self, tokens: int) -> bool:
        if not chunked_moe_enabled():
            return False
        return tokens > max(int(self.max_tokens_per_rank), 0)

    def _run_chunk(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        with record_function_range("dsv4.moe.gate"):
            weights, indices = self.gate(x, input_ids)

        with record_function_range("dsv4.moe.shared_expert_start"):
            self._shared_executor.start(self.shared_experts, x)
        try:
            with record_function_range("dsv4.moe.routed_experts"):
                routed = self._strategy(x, weights, indices)
        except Exception:
            with record_function_range("dsv4.moe.shared_expert_finish"):
                self._shared_executor.finish()
            raise

        with record_function_range("dsv4.moe.shared_expert_finish"):
            shared = self._shared_executor.finish()
        with record_function_range("dsv4.moe.add_shared"):
            combined = combine_routed_and_shared(routed, shared, x.dtype, out=out)
            if combined.data_ptr() != out.data_ptr():
                out.copy_(combined)

    def _forward_chunked(
        self,
        x: torch.Tensor,
        input_ids_flat: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        global _CHUNKED_MOE_LOGGED
        T = x.size(0)
        chunk_tokens = max(int(self.max_tokens_per_rank), 1)
        if not _CHUNKED_MOE_LOGGED:
            _CHUNKED_MOE_LOGGED = True
            logging.info(
                "[DSV4 MoE] chunked forward enabled: layer=%d tokens=%d "
                "chunk_tokens=%d chunks=%d dim=%d device=%s",
                self.layer_id,
                T,
                chunk_tokens,
                (T + chunk_tokens - 1) // chunk_tokens,
                self.dim,
                x.device,
            )
        out = _get_or_create_final_out(T, self.dim, x.dtype, x.device)[:T]
        for token_start in range(0, T, chunk_tokens):
            token_end = min(token_start + chunk_tokens, T)
            self._run_chunk(
                x[token_start:token_end],
                input_ids_flat[token_start:token_end],
                out[token_start:token_end],
            )
        return out.view(shape)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        # Master switch: when MOEDBG=0 the AND short-circuits so neither the
        # layer_id compare nor any record_if_level call site below runs.
        # Instruments layers 0..2 (first CSA layer is L2) when enabled.
        _dbg = _rt.should_record_layer(self.layer_id)
        shape = x.size()
        x = x.view(-1, self.dim)
        input_ids_flat = input_ids.flatten()
        if input_ids_flat.numel() != x.size(0):
            raise RuntimeError(
                "MoE input_ids/token mismatch: "
                f"input_ids={input_ids_flat.numel()} tokens={x.size(0)}"
            )
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
        if self._should_chunk(x.size(0)):
            return self._forward_chunked(x, input_ids_flat, shape)

        with record_function_range("dsv4.moe.gate"):
            if _dbg:
                self.gate._dbg_prefix = f"L{self.layer_id:02d}_moe_gate"
            try:
                weights, indices = self.gate(x, input_ids_flat)
            finally:
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

        with record_function_range("dsv4.moe.shared_expert_start"):
            self._shared_executor.start(self.shared_experts, x)
        try:
            with record_function_range("dsv4.moe.routed_experts"):
                y = self._strategy(x, weights, indices)
        except Exception:
            with record_function_range("dsv4.moe.shared_expert_finish"):
                self._shared_executor.finish()
            raise

        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_routed_y", y)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_moe_routed_y_{dbg_pos_name}",
                    y[dbg_pos_mask].contiguous(),
                )
        with record_function_range("dsv4.moe.shared_expert_finish"):
            shared_y = self._shared_executor.finish()
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_shared_y", shared_y)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_moe_shared_y_{dbg_pos_name}",
                    shared_y[dbg_pos_mask].contiguous(),
                )
        if _dbg:
            with record_function_range("dsv4.moe.add_shared"):
                y = y + shared_y
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_moe_y_{dbg_pos_name}",
                    y[dbg_pos_mask].contiguous(),
                )
            return y.type_as(x).view(shape)
        with record_function_range("dsv4.moe.add_shared"):
            T = x.size(0)
            out = _get_or_create_final_out(
                max(T, self.max_tokens_per_rank, 1),
                self.dim,
                x.dtype,
                x.device,
            )
            y = combine_routed_and_shared(y, shared_y, x.dtype, out=out[:T])
            return y.view(shape)
