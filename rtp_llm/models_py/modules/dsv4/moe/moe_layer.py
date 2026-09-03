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
_MOE_MARK = [0]
from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.chunk_env import (
    DEFAULT_DSV4_CHUNK_TOKENS,
    dsv4_chunk_tokens_from_env,
    dsv4_global_chunk_tokens_configured,
)

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
# source this feature addresses. 16384 bounds the persistent MegaMoE symmetric
# buffer while keeping chunks large enough to avoid excessive
# kernel/dispatch overhead. Override with DSV4_CHUNK_TOKENS or
# DSV4_MOE_CHUNK_TOKENS for smoke tests or tighter HBM budgets.
DEFAULT_MOE_CHUNK_TOKENS = DEFAULT_DSV4_CHUNK_TOKENS


def chunked_moe_enabled() -> bool:
    if dsv4_global_chunk_tokens_configured():
        return moe_chunk_tokens_from_env() > 0
    return os.environ.get("DSV4_MOE_CHUNK_PREFILL", "1") != "0"


def moe_chunk_tokens_from_env(default: int = DEFAULT_MOE_CHUNK_TOKENS) -> int:
    min_value = 0 if dsv4_global_chunk_tokens_configured() else 1
    return dsv4_chunk_tokens_from_env(
        "DSV4_MOE_CHUNK_TOKENS",
        default,
        min_value=min_value,
    )


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
    *,
    is_decode_role: bool = False,
    is_speculative: bool = False,
    gen_num_per_cycle: int = 0,
) -> int:
    max_generate_batch_size = int(max_generate_batch_size)
    assert (
        max_generate_batch_size > 0
    ), f"max_generate_batch_size must be positive, got {max_generate_batch_size}"
    if is_decode_role:
        tokens_per_batch = 1
        if is_speculative:
            tokens_per_batch = max(int(gen_num_per_cycle or 0) + 1, 1)
        return max_generate_batch_size * tokens_per_batch

    budget = int(current_max_tokens_per_rank)
    cp_size = max(int(cp_size), 1)
    if cp_size > 1:
        cp_bound = max(cp_padded_tokens_per_rank_bound(max_seq_len, cp_size), 4096)
        budget = min(budget, cp_bound)

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


class MoeDeferredHalves:
    """X1 (DSV4_XLAYER_C1): half-0 of the MoE output is combined and ready;
    half-1 is deferred behind the caller's half-0 epilogue (the cover for
    c1's flight). ``finish()`` waits the strategy's combine event, dequants
    and combines half-1, and returns ``y1`` — row-identical to the stock
    combine.
    """

    def __init__(self, y0, deferred, shared_y, th, out):
        self.y0 = y0
        self.th = th
        self._deferred = deferred
        self._shared_y = shared_y
        self._out = out

    def finish(self) -> torch.Tensor:
        full = self._deferred.finish()
        y1 = combine_routed_and_shared(
            full[self.th :],
            self._shared_y[self.th :],
            self.y0.dtype,
            out=self._out[self.th :] if self._out is not None else None,
        )
        return y1


def _a2_deferred_cls():
    from .strategies.deepep import _A2DeferredHalves

    return _A2DeferredHalves


_X1_DEF_CT = [0]  # X1 engagement proof (DSV4_DIAG, first 3 fires)


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
        is_decode_role: bool = False,
        strategy: Optional[str] = None,
        tp_size: int = 1,
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
        self._is_decode_role = bool(is_decode_role)

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

        # --- Strategy selection ---
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
            tp_size=tp_size,
        )
        forced, strict = _resolve_forced(strategy)
        strategy_cls = select_strategy(cfg, forced=forced, strict=strict)
        # Strategies that fold the shared expert into their routed kernel
        # (MegaMoEFusedStrategy / MegaMoEStrategySE) own the shared-expert
        # weights themselves and produce ``routed + shared`` directly; the
        # MoE layer then skips its standalone shared-expert executor and the
        # combine add.
        self._routed_includes_shared = bool(
            getattr(strategy_cls, "routed_includes_shared", False)
        )

        if self._routed_includes_shared:
            # The fused strategy pops + owns W.v4_shared_* in setup_weights.
            self.shared_experts = None
            self._shared_executor = None
        else:
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

        # Register strategy as a child nn.Module so its child weights
        # (e.g. LocalLoopStrategy.experts ModuleList) propagate through
        # ``MoE.to(device)``.
        self._strategy = strategy_cls(cfg)
        self._gate_pack_static = os.environ.get(
            "MOEDBG", "0"
        ) == "0" and self._strategy.can_use_gate_pack_static(self.gate)
        self._strategy._gate_pack_warmup_enabled = self._gate_pack_static
        self._strategy._gate_pack_route_scale = float(self.gate.route_scale)
        self._strategy.setup_weights(layer_weights)

    def _should_chunk(self, tokens: int) -> bool:
        max_tokens = int(self.max_tokens_per_rank)
        assert max_tokens > 0, f"max_tokens_per_rank must be positive, got {max_tokens}"
        cuda_graph_capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if self._is_decode_role or cuda_graph_capturing:
            if int(tokens) > max_tokens:
                mode = "decode" if self._is_decode_role else "CUDA graph capture"
                raise AssertionError(
                    f"{mode} MoE input tokens="
                    f"{int(tokens)} exceeds max_tokens_per_rank={max_tokens}; "
                    f"{mode} must not use chunked MoE. Increase the startup "
                    "MoE token budget."
                )
            return False
        if not chunked_moe_enabled():
            return False
        return tokens > max_tokens

    def _run_chunk(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if self._gate_pack_static:
            if self._routed_includes_shared:
                with record_function_range("dsv4.moe.routed_experts"):
                    routed = self._strategy.forward_with_gate_pack(
                        x, self.gate, input_ids
                    )
                out.copy_(routed)
                return

            with record_function_range("dsv4.moe.shared_expert_start"):
                self._shared_executor.start(self.shared_experts, x)
            try:
                with record_function_range("dsv4.moe.routed_experts"):
                    routed = self._strategy.forward_with_gate_pack(
                        x, self.gate, input_ids
                    )
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
            return

        with record_function_range("dsv4.moe.gate"):
            weights, indices = self.gate(x, input_ids)

        if self._routed_includes_shared:
            # Fused strategy returns ``routed + shared`` directly.
            with record_function_range("dsv4.moe.routed_experts"):
                routed = self._strategy(x, weights, indices)
            out.copy_(routed)
            return

        # C2 prepared-dispatch split — see the comment at the forward()
        # call site; identical contract, chunked-tile flavor.
        prepared = self._strategy.prepare_dispatch(x, weights, indices)
        with record_function_range("dsv4.moe.shared_expert_start"):
            self._shared_executor.start(self.shared_experts, x)
        try:
            with record_function_range("dsv4.moe.routed_experts"):
                if prepared is not None:
                    routed = self._strategy.run_dispatch_prepared(prepared)
                    if isinstance(routed, _a2_deferred_cls()):
                        # Chunked path: no split epilogue here — materialize
                        # (the deferral only pays off on the non-chunked path).
                        routed = routed.finish()
                else:
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
        T = int(x.size(0))
        if os.environ.get("DSV4_MEMDUMP") and _MOE_MARK[0] == 1:
            import sys as _s
            print("[MEMDUMP] after-capture allocator state:", file=_s.stderr, flush=True)
            print(torch.cuda.memory_summary(0, abbreviated=False), file=_s.stderr, flush=True)
            # NOTE: torch.cuda.memory_snapshot() is NOT usable here — this build
            # runs expandable_segments:True and the snapshot call raises, killing
            # the rank. Itemize via memory_summary totals + config arithmetic.
        if os.environ.get("DSV4_DIAG") and _MOE_MARK[0] < 60:
            _MOE_MARK[0] += 1
            import sys as _s
            print("[MOE] fwd layer=%s T=%d" % (getattr(self, "layer_id", "?"), T), file=_s.stderr, flush=True)
        if not _CHUNKED_MOE_LOGGED:
            _CHUNKED_MOE_LOGGED = True
            logging.info(
                "[DSV4 MoE] single-round EP forward: layer=%d tokens=%d "
                "dim=%d device=%s (one count-AG + dispatch/combine a2a per "
                "layer on every rank; per-rank GEMM tiled at %d)",
                self.layer_id,
                T,
                self.dim,
                x.device,
                self.max_tokens_per_rank,
            )
        out = _get_or_create_final_out(T, self.dim, x.dtype, x.device)[:T]
        # Single collective round over ALL local tokens. The token-dim chunk
        # loop this replaced made the per-layer EP collective count depend on
        # the rank's token count (owner ran ceil(T/chunk) rounds while
        # 1-token peers ran one) and deadlocked fused >= 8K ISL prefills:
        # the owner's later-round count-AllGather had no matching peer
        # collectives. One round per rank keeps the NCCL op sequence
        # identical on every rank for any token distribution (including
        # T == 0 and the rotating per-request owner); DeepEPStrategy tiles
        # its GEMM internally, so workspace stays bounded.
        if (
            T > 8192
            and self._shared_executor is not None
            and not self._routed_includes_shared
        ):
            # Big-T memory sequencing: routed experts FIRST, shared expert
            # after. Holding the shared expert's fp32 output (~512 MiB at 32K
            # tokens) across the routed path's a2a transients (~1 GiB) OOM'd
            # 32K prefills at layer 0; sequencing drops the MoE-stage peak to
            # max(routed, shared) plus the small surviving routed result.
            _free_b, _ = torch.cuda.mem_get_info()
            if _free_b < 1_500_000_000 and os.environ.get("DSV4_EMPTY_CACHE_MODE", "all") == "all":
                # Attention just freed its transients as allocator-cached
                # fragments; the a2a needs large contiguous buffers — flush
                # the cache rather than OOM with free memory held hostage.
                torch.cuda.empty_cache()
            if self._gate_pack_static:
                routed = self._strategy.forward_with_gate_pack(
                    x, self.gate, input_ids_flat
                )
            else:
                _w, _idx = self.gate(x, input_ids_flat)
                routed = self._strategy(x, _w, _idx)
            self._shared_executor.start(self.shared_experts, x)
            shared = self._shared_executor.finish()
            combined = combine_routed_and_shared(routed, shared, x.dtype, out=out)
            if combined.data_ptr() != out.data_ptr():
                out.copy_(combined)
            return out.view(shape)
        self._run_chunk(x, input_ids_flat, out)
        return out.view(shape)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
        if os.environ.get("DSV4_MEMDUMP") and _MOE_MARK[0] == 0:
            import sys as _s
            _MOE_MARK[0] = 1
            print("[MEMDUMP] after-capture allocator state:", file=_s.stderr, flush=True)
            print(torch.cuda.memory_summary(0, abbreviated=False), file=_s.stderr, flush=True)
            # NOTE: torch.cuda.memory_snapshot() is NOT usable here — this build
            # runs expandable_segments:True and the snapshot call raises, killing
            # the rank. Itemize via memory_summary totals + config arithmetic.

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

        if self._gate_pack_static:
            if self._routed_includes_shared:
                with record_function_range("dsv4.moe.routed_experts"):
                    y = self._strategy.forward_with_gate_pack(
                        x, self.gate, input_ids_flat
                    )
                with record_function_range("dsv4.moe.add_shared"):
                    T = x.size(0)
                    out = _get_or_create_final_out(
                        max(T, self.max_tokens_per_rank, 1),
                        self.dim,
                        x.dtype,
                        x.device,
                    )
                    out[:T].copy_(y)
                    return out[:T].view(shape)

            with record_function_range("dsv4.moe.shared_expert_start"):
                self._shared_executor.start(self.shared_experts, x)
            try:
                with record_function_range("dsv4.moe.routed_experts"):
                    y = self._strategy.forward_with_gate_pack(
                        x, self.gate, input_ids_flat
                    )
            except Exception:
                with record_function_range("dsv4.moe.shared_expert_finish"):
                    self._shared_executor.finish()
                raise

            with record_function_range("dsv4.moe.shared_expert_finish"):
                shared_y = self._shared_executor.finish()
            with record_function_range("dsv4.moe.add_shared"):
                T = x.size(0)
                out = _get_or_create_final_out(
                    max(T, self.max_tokens_per_rank, 1),
                    self.dim,
                    x.dtype,
                    x.device,
                )
                # NOTE: the X1 deferred contract is intentionally NOT wired
                # into the gate-pack branch — it is a deepep-halves concept
                # and Mega strategies never return deferreds.
                y = combine_routed_and_shared(y, shared_y, x.dtype, out=out[:T])
                return y.view(shape)

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

        if self._routed_includes_shared:
            # Fused strategy: the kernel already produced ``routed + shared``.
            with record_function_range("dsv4.moe.routed_experts"):
                y = self._strategy(x, weights, indices)
            if _dbg:
                _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_routed_y", y)
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
                out[:T].copy_(y)
                return out[:T].view(shape)

        # C2 (bench/results_20260902_q4c/VERDICT.md): run the strategy's
        # host-serial dispatch front half (quantize + count-AG + .cpu() sync
        # + pack) BEFORE the shared expert starts, so the dispatch a2a
        # launches while the aux-stream shared expert is still running.
        # prepare_dispatch returns None for every strategy/path it does not
        # own (base default None) — the stock order then runs unchanged.
        prepared = self._strategy.prepare_dispatch(x, weights, indices)
        with record_function_range("dsv4.moe.shared_expert_start"):
            self._shared_executor.start(self.shared_experts, x)
        try:
            with record_function_range("dsv4.moe.routed_experts"):
                if prepared is not None:
                    y = self._strategy.run_dispatch_prepared(prepared)
                else:
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
                if isinstance(y, _a2_deferred_cls()):
                    y = y.finish()  # dbg path needs the full tensor eagerly
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
            if isinstance(y, _a2_deferred_cls()):
                # X1 (DSV4_XLAYER_C1): half-0 rows are materialized — combine
                # them NOW and defer half-1 behind the caller's half-0
                # epilogue (the cover for c1's flight). The deferred carries
                # the out-buffer slices so the h1 combine lands in the same
                # buffer the stock path would have written.
                y0 = combine_routed_and_shared(
                    y.result[: y.th], shared_y[: y.th], x.dtype, out=out[: y.th]
                )
                if os.environ.get("DSV4_DIAG") and _X1_DEF_CT[0] < 3:
                    _X1_DEF_CT[0] += 1
                    import sys as _sys
                    print(
                        "[X1-DEF] rank=%d th=%d T=%d" % (
                            torch.distributed.get_rank()
                            if torch.distributed.is_initialized()
                            else -1,
                            y.th,
                            T,
                        ),
                        file=_sys.stderr,
                        flush=True,
                    )
                return MoeDeferredHalves(
                    y0=y0,
                    deferred=y,
                    shared_y=shared_y,
                    th=y.th,
                    out=out,
                )
            y = combine_routed_and_shared(y, shared_y, x.dtype, out=out[:T])
            return y.view(shape)
