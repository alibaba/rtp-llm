"""LocalLoopStrategy: Python-level for-loop over per-expert ``Expert`` modules.

Universal fallback path:
  - ep_size == 1 without the grouped FP4 kernel (also composed inside
    ``DeepEPStrategy`` for the local compute on dispatched recv tokens)
  - cuda-graph capture (forward dispatches to graph-safe variant internally)

Three forward variants:
  - ``_forward_eager``       — eager (non-capturing) — only iterates routed
    tokens per expert via ``torch.where``. Already optimal in eager mode.
  - ``_forward_topk_bs1`` /
    ``_forward_topk_bsN``    — graph-capture top-K dispatch: iterates only
    the K = ``cfg.n_activated_experts`` active slots per token (via
    ``torch.index_select`` on stacked routed weights), instead of all
    ``cfg.n_routed_experts``. Used when ``ep_size==1`` and the captured
    batch size N ≤ ``DSV4_LOCALLOOP_TOPK_MAX_N`` (default 32). Replaces a
    256-expert × 3-GEMM unconditional loop with N×K×3 GEMMs per layer
    (e.g. bs=1: 768→18, bs=8: 768→144).
  - ``_forward_graph_safe``  — graph-capture fallback for batches the top-K
    dispatch doesn't cover (large N, ep>1, or path disabled). Uses
    fixed-shape per-expert mask compute to avoid data-dependent
    ``torch.where`` shapes.

Owns the ``self._local_y_buf`` fp32 accumulator (lazy-allocated, reused
across forward calls). Same buffer is reused under cuda-graph capture.

Wired into ``MoE`` via ``select_strategy`` as the universal fallback (and
also composed inside ``DeepEPStrategy`` for local recv-token compute).
Direct port of the pre-refactor ``_routed_experts_local{,_graph_safe}``
methods + matching ModuleList construction + forward buffer-management.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..expert import Expert
from ...quant_layouts import prepare_fp4_weight_scale_for_deepgemm
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy

# Block sizes for FP8 / FP4 (mirror values in qlinear.py)
_FP8_BLOCK = 128
_FP4_BLOCK = 32

# Toggle for the bs=1 fast path. Default ON. Set DSV4_LOCALLOOP_BS1_FAST=0 to disable.
def _bs1_fast_enabled() -> bool:
    return os.environ.get("DSV4_LOCALLOOP_BS1_FAST", "1") != "0"


# Toggle for the bs>1 fast path (per-token K-slot dispatch).
# Default ON when bs <= this threshold; set DSV4_LOCALLOOP_TOPK_MAX_N=0 to disable.
def _topk_dispatch_max_n() -> int:
    """Max N (batch tokens) for which the per-token K-slot dispatch is used.

    For N×K = N×6 < E = 256 to be a win, need N < 42. Default cap is 32 to
    match common decode batch graph captures (1, 8, 16, 24, 32). For larger
    captures (48, 64, 80, 96, 112, 128) the per-token loop overhead and
    Python unroll size start to outweigh savings; fall through to slow path.
    """
    return int(os.environ.get("DSV4_LOCALLOOP_TOPK_MAX_N", "32"))


_LOCAL_Y_CACHE: dict[tuple, torch.Tensor] = {}


def _get_or_create_local_y(
    capacity: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (device, dim, dtype)
    cached = _LOCAL_Y_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), dim), dtype=dtype, device=device)
    _LOCAL_Y_CACHE[key] = cached
    return cached


def _select_mn_major_scale_for_index(
    scale_gemm_t: torch.Tensor,
    expert_idx: torch.Tensor,
) -> torch.Tensor:
    """Select one expert scale while preserving DeepGEMM's MN-major stride."""
    return torch.index_select(scale_gemm_t, 0, expert_idx).squeeze(0).transpose(0, 1)


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

        PATCH: also stash the stacked tensors on ``self._W{1,2,3}_{w,s}``
        for the bs=1 fast path. Same underlying storage as the per-Expert
        slices — no extra memory.
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

        # PATCH: keep references to stacked tensors for fast top-K dispatch.
        # These share storage with the per-expert slices held by Expert objects
        # below — zero memory overhead. Stored as plain attributes (not nn.Parameter
        # / register_buffer) since they're already managed by the framework loader
        # and we don't want to double-track them in module state.
        self._W1_w = stacked_routed["w1_w"]
        self._W1_s = stacked_routed["w1_s"]
        self._W2_w = stacked_routed["w2_w"]
        self._W2_s = stacked_routed["w2_s"]
        self._W3_w = stacked_routed["w3_w"]
        self._W3_s = stacked_routed["w3_s"]
        self._W1_s_gemm = prepare_fp4_weight_scale_for_deepgemm(
            self._W1_s, cfg.moe_inter_dim, cfg.dim, self._W1_s.shape[0]
        )
        self._W2_s_gemm = prepare_fp4_weight_scale_for_deepgemm(
            self._W2_s, cfg.dim, cfg.moe_inter_dim, self._W2_s.shape[0]
        )
        self._W3_s_gemm = prepare_fp4_weight_scale_for_deepgemm(
            self._W3_s, cfg.moe_inter_dim, cfg.dim, self._W3_s.shape[0]
        )
        # Per-expert DeepGEMM scales are MN-major: a direct
        # self._W*_s_gemm[i] view has stride (1, mn).  torch.index_select on
        # the grouped tensor returns a row-major copy, which fails
        # DeepGEMM's layout check during CUDA graph top-k dispatch.  Select
        # from the transposed view and transpose the selected copy back.
        self._W1_s_gemm_t = self._W1_s_gemm.transpose(-1, -2)
        self._W2_s_gemm_t = self._W2_s_gemm.transpose(-1, -2)
        self._W3_s_gemm_t = self._W3_s_gemm.transpose(-1, -2)

        def _expert_at(global_idx: int) -> Optional[Expert]:
            if not (cfg.local_expert_start <= global_idx < cfg.local_expert_end):
                return None
            local_idx = global_idx - cfg.local_expert_start
            ew = {
                "w1_w": stacked_routed["w1_w"][local_idx],
                "w1_s": stacked_routed["w1_s"][local_idx],
                "w1_s_gemm": self._W1_s_gemm[local_idx],
                "w2_w": stacked_routed["w2_w"][local_idx],
                "w2_s": stacked_routed["w2_s"][local_idx],
                "w2_s_gemm": self._W2_s_gemm[local_idx],
                "w3_w": stacked_routed["w3_w"][local_idx],
                "w3_s": stacked_routed["w3_s"][local_idx],
                "w3_s_gemm": self._W3_s_gemm[local_idx],
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
        self._local_y_buf = _get_or_create_local_y(
            max(T, self.cfg.max_tokens_per_rank),
            self.cfg.dim,
            torch.float32,
            x.device,
        )
        buf = self._local_y_buf
        y = buf[:T]
        y.zero_()
        if torch.cuda.is_current_stream_capturing():
            # PATCH: top-K fast path. Only N×K Python iterations instead of
            # `local_end - local_start` (=n_routed_experts; 256 for V4-Flash).
            # Conditioned on ep_size==1 (so all experts are local — no per-rank
            # filter needed) and T (=N) small enough that N×K < E.
            topk_max_n = _topk_dispatch_max_n()
            if (_bs1_fast_enabled()
                and self.cfg.ep_size == 1
                and topk_max_n > 0
                and T <= topk_max_n):
                if T == 1:
                    self._forward_topk_bs1(x, weights, indices, y)
                else:
                    self._forward_topk_bsN(x, weights, indices, y)
            else:
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

    # ------------------------------------------------------------------
    # PATCH: bs=1 fast path — top-K dispatch via `index_select` on stacked
    # routed weights. Avoids the 256-expert loop in `_forward_graph_safe`
    # while remaining CUDA-graph-safe (Python loop is over `K` = compile-time
    # `n_activated_experts`, all index ops produce fixed-shape outputs).
    # ------------------------------------------------------------------
    def _forward_topk_bs1(
        self,
        x: torch.Tensor,        # [1, D] bf16
        weights: torch.Tensor,  # [1, K] fp32
        indices: torch.Tensor,  # [1, K] int64 — GLOBAL expert IDs (== local for ep=1)
        y: torch.Tensor,        # [1, D] fp32, accumulator
    ) -> torch.Tensor:
        """bs=1 hot path: K (≈top-6) expert calls instead of E (=256).

        Math equivalence vs. ``_forward_graph_safe``:
            both compute ``y[0] = Σ_k weights[0,k] * expert_{indices[0,k]}(x[0])``.
            graph_safe walks all E experts, multiplying by 0 for inactive ones.
            this path walks only the K active slots.

        CUDA-graph safety:
            * `K` is `cfg.n_activated_experts` (compile-time, =6 for V4-Flash)
              → Python ``for k in range(K)`` unrolls into a fixed kernel sequence
            * `indices[0, k:k+1]` slice has compile-time shape [1] (no
              data-dependent shape)
            * `torch.index_select(stacked, 0, eid_t)` returns shape
              ``[1, *stacked.shape[1:]]`` regardless of `eid_t` value
            * each expert kernel sees fixed (M=1, K=D, N=inter) shapes

        Performance vs. ``_forward_graph_safe``:
            * E×3 = 768 FP8-FP4 GEMMs → K×3 = 18 (≈40× fewer launches +
              ≈40× less wasted compute on inactive experts)
            * input quant runs ONCE per layer (vs. once per expert call → 256x)
        """
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

        cfg = self.cfg
        D = cfg.dim
        inter = cfg.moe_inter_dim
        swiglu_limit = cfg.swiglu_limit
        device = x.device
        K = indices.size(1)

        # Quant input ONCE for all K slots (gate / up share the same a tensor).
        x_2d = x.reshape(-1, D).contiguous()
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)
        x_fp8, x_scale = sgl_per_token_group_quant_fp8(
            x_2d,
            group_size=_FP8_BLOCK,
            eps=1e-10,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )

        # Lazy import the fused SiLU+clamp+mul (matches Expert.forward path)
        try:
            from rtp_llm.models_py.modules.dsv4._silu_mul_split_triton import silu_mul_split
            _have_silu_mul_split = True
        except Exception:  # pragma: no cover
            silu_mul_split = None
            _have_silu_mul_split = False

        for k in range(K):  # K=6, fully unrolled by graph capture
            eid_t = indices[0, k : k + 1]  # [1] long, on device
            router_w = weights[0, k : k + 1, None]  # [1, 1] fp32

            # Gather expert eid's weight slices (graph-safe, fixed-shape output).
            # Use squeeze(0) on dim 0 to drop the [1, ...] from index_select.
            w1_w = torch.index_select(self._W1_w, 0, eid_t).squeeze(0)  # [inter, D/2] int8
            w1_s = _select_mn_major_scale_for_index(self._W1_s_gemm_t, eid_t)
            w3_w = torch.index_select(self._W3_w, 0, eid_t).squeeze(0)
            w3_s = _select_mn_major_scale_for_index(self._W3_s_gemm_t, eid_t)
            w2_w = torch.index_select(self._W2_w, 0, eid_t).squeeze(0)  # [D, inter/2]
            w2_s = _select_mn_major_scale_for_index(self._W2_s_gemm_t, eid_t)

            # gate = w1 @ x
            gate = torch.empty(1, inter, dtype=torch.bfloat16, device=device)
            fp8_fp4_gemm_nt(
                (x_fp8, x_scale),
                (w1_w, w1_s),
                gate,
                recipe_a=(1, _FP8_BLOCK), recipe_b=(1, _FP4_BLOCK),
            )
            # up = w3 @ x
            up = torch.empty(1, inter, dtype=torch.bfloat16, device=device)
            fp8_fp4_gemm_nt(
                (x_fp8, x_scale),
                (w3_w, w3_s),
                up,
                recipe_a=(1, _FP8_BLOCK), recipe_b=(1, _FP4_BLOCK),
            )

            # SiLU + (optional clamp) + mul, in fp32 (matches Expert.forward).
            gate_f = gate.float()
            up_f = up.float()
            if _have_silu_mul_split:
                sm_fp32 = silu_mul_split(
                    gate_f.contiguous(),
                    up_f.contiguous(),
                    clamp_limit=swiglu_limit,
                )
            else:
                if swiglu_limit > 0:
                    up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
                    gate_f = torch.clamp(gate_f, max=swiglu_limit)
                sm_fp32 = torch.nn.functional.silu(gate_f) * up_f

            # Apply router weight BEFORE w2 (matches Expert.forward semantics).
            sm_fp32 = sm_fp32 * router_w  # [1, inter]
            sm_bf16 = sm_fp32.to(torch.bfloat16)

            # Quant for w2 input
            sm_fp8, sm_scale = sgl_per_token_group_quant_fp8(
                sm_bf16.contiguous(),
                group_size=_FP8_BLOCK,
                eps=1e-10,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            # delta = w2 @ sm
            delta = torch.empty(1, D, dtype=torch.bfloat16, device=device)
            fp8_fp4_gemm_nt(
                (sm_fp8, sm_scale),
                (w2_w, w2_s),
                delta,
                recipe_a=(1, _FP8_BLOCK), recipe_b=(1, _FP4_BLOCK),
            )
            # Accumulate (router_w already folded into sm above).
            y.add_(delta.float())

        return y

    # ------------------------------------------------------------------
    # PATCH v2: bs > 1 fast path — per-token K-slot dispatch.
    # ------------------------------------------------------------------
    def _forward_topk_bsN(
        self,
        x: torch.Tensor,        # [N, D] bf16
        weights: torch.Tensor,  # [N, K] fp32
        indices: torch.Tensor,  # [N, K] int64 — GLOBAL expert IDs
        y: torch.Tensor,        # [N, D] fp32, accumulator
    ) -> torch.Tensor:
        """N>1 hot path: N*K (≈N*6) expert calls instead of E (=256).

        Math equivalence vs. `_forward_graph_safe`:
            both compute y[n] = Σ_k weights[n,k] * expert_{indices[n,k]}(x[n]).
            graph_safe walks all E experts with mask=0 for non-routed tokens.
            this path walks only the K active slots per token.

        CUDA-graph safety:
            * `N` is the captured batch size (compile-time known)
            * `K` is `cfg.n_activated_experts` (compile-time, =6 for V4-Flash)
            * Python `for n in range(N): for k in range(K)` unrolls during
              capture into a fixed kernel sequence
            * `indices[n, k:k+1]` slice has compile-time shape [1]
            * `torch.index_select(stacked, 0, eid_t)` returns shape
              `[1, *stacked.shape[1:]]` regardless of `eid_t` value
            * each expert kernel sees fixed (M=1, K=D, N=inter) shapes

        Win vs. `_forward_graph_safe`:
            * E*3 = 768 GEMMs per layer → N*K*3 GEMMs (e.g. bs=8: 144, 5.3×↓)
            * Per-token quant cost is unchanged (N quants per layer either way)

        Limit:
            * Python loop unroll size = N*K. At bs=32 that's 192 GEMM calls per
              layer per slot k = 576 per layer total. Capture overhead grows
              proportionally. The dispatch caps at `_topk_dispatch_max_n()`.
        """
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_fp4_gemm_nt
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

        cfg = self.cfg
        D = cfg.dim
        inter = cfg.moe_inter_dim
        swiglu_limit = cfg.swiglu_limit
        device = x.device
        N = x.size(0)
        K = indices.size(1)

        # Cast input to bf16 if needed (we quant per-token below)
        x_2d = x.reshape(N, D).contiguous()
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)

        try:
            from rtp_llm.models_py.modules.dsv4._silu_mul_split_triton import silu_mul_split
            _have_silu_mul_split = True
        except Exception:  # pragma: no cover
            silu_mul_split = None
            _have_silu_mul_split = False

        # Per-token, per-slot dispatch. Both N and K are compile-time, so the
        # nested loops fully unroll under graph capture into N*K kernel sequences.
        # NOTE: we quant each row separately rather than slicing a batched
        # `sgl_per_token_group_quant_fp8(x_2d, ...)` output. Slicing the column-
        # major TMA-aligned scale tensor breaks DeepGEMM's TMA alignment check
        # (`sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size())`),
        # causing capture to fail. Per-row quant produces a freshly aligned
        # [1, D/128] scale per token — same path as `_forward_topk_bs1`.
        for n in range(N):
            x_n = x_2d[n : n + 1].contiguous()  # [1, D]
            x_fp8_n, x_scale_n = sgl_per_token_group_quant_fp8(
                x_n,
                group_size=_FP8_BLOCK,
                eps=1e-10,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            for k in range(K):
                eid_t = indices[n, k : k + 1]            # [1] long
                router_w = weights[n, k : k + 1, None]   # [1, 1] fp32

                # Gather expert eid's weight slices (graph-safe).
                w1_w = torch.index_select(self._W1_w, 0, eid_t).squeeze(0)
                w1_s = _select_mn_major_scale_for_index(self._W1_s_gemm_t, eid_t)
                w3_w = torch.index_select(self._W3_w, 0, eid_t).squeeze(0)
                w3_s = _select_mn_major_scale_for_index(self._W3_s_gemm_t, eid_t)
                w2_w = torch.index_select(self._W2_w, 0, eid_t).squeeze(0)
                w2_s = _select_mn_major_scale_for_index(self._W2_s_gemm_t, eid_t)

                # gate = w1 @ x_n
                gate = torch.empty(1, inter, dtype=torch.bfloat16, device=device)
                fp8_fp4_gemm_nt(
                    (x_fp8_n, x_scale_n),
                    (w1_w, w1_s),
                    gate,
                    recipe_a=(1, _FP8_BLOCK), recipe_b=(1, _FP4_BLOCK),
                )
                # up = w3 @ x_n
                up = torch.empty(1, inter, dtype=torch.bfloat16, device=device)
                fp8_fp4_gemm_nt(
                    (x_fp8_n, x_scale_n),
                    (w3_w, w3_s),
                    up,
                    recipe_a=(1, _FP8_BLOCK), recipe_b=(1, _FP4_BLOCK),
                )

                gate_f = gate.float()
                up_f = up.float()
                if _have_silu_mul_split:
                    sm_fp32 = silu_mul_split(
                        gate_f.contiguous(),
                        up_f.contiguous(),
                        clamp_limit=swiglu_limit,
                    )
                else:
                    if swiglu_limit > 0:
                        up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
                        gate_f = torch.clamp(gate_f, max=swiglu_limit)
                    sm_fp32 = torch.nn.functional.silu(gate_f) * up_f

                sm_fp32 = sm_fp32 * router_w
                sm_bf16 = sm_fp32.to(torch.bfloat16)

                sm_fp8, sm_scale = sgl_per_token_group_quant_fp8(
                    sm_bf16.contiguous(),
                    group_size=_FP8_BLOCK,
                    eps=1e-10,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                delta = torch.empty(1, D, dtype=torch.bfloat16, device=device)
                fp8_fp4_gemm_nt(
                    (sm_fp8, sm_scale),
                    (w2_w, w2_s),
                    delta,
                    recipe_a=(1, _FP8_BLOCK), recipe_b=(1, _FP4_BLOCK),
                )
                # Accumulate into y[n]
                y[n : n + 1].add_(delta.float())

        return y
