"""DeepSeek-V4 MoE: Gate (sqrt(softplus) + hash routing) + Expert (clamped SwiGLU) + MoE.

Direct port of `inference/model.py:Gate / Expert / MoE` (BF16-only).
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    _m_grouped_fp8_fp4_gemm_nt_contiguous_impl,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear

# P2 (plan_0427.md): single-Triton-kernel router-gate epilogue for
# score_func='sqrtsoftplus'.  Replaces ~7 elementwise/reduce/topk launches
# (softplus → sqrt → bias-add → topk → gather → sum → div → mul) with one
# fused kernel.  ~4× per-call speedup, identical top-k indices vs eager.
try:
    from rtp_llm.models_py.modules.dsv4._gate_fused_triton import (
        fused_sqrtsoftplus_gate,
    )

    _GATE_FUSED_OK = True
except Exception:  # pragma: no cover
    fused_sqrtsoftplus_gate = None
    _GATE_FUSED_OK = False


def _use_fused_gate(score_func: str, x_size_0: int) -> bool:
    """Gate for the fused router-gate kernel.

    Defaults to OFF: the kernel is bit-equivalent to the eager FP32 epilogue
    in microbench (max abs diff 4.5e-8 at rel ~2e-7, top-k strict-equal 100%
    across 5 random seeds), but in practice the slight fp32 reduction-order
    drift in `weights / weights.sum()` is enough to flip greedy decode on
    tied/near-tied logits across ~60 layers — re-capturing all 6 V4-Flash
    smoke goldens is required when this is on.  The per-call win (~0.18 ms)
    over 43 layers/forward is small (~8 ms / ~0.25% of the 3090 ms prefill),
    so we keep this opt-in for users willing to pay the golden churn.

    Set ``DSV4_GATE_FUSED=1`` to enable.
    """
    if not _GATE_FUSED_OK or fused_sqrtsoftplus_gate is None:
        return False
    if os.environ.get("DSV4_GATE_FUSED", "0") != "1":
        return False
    if score_func != "sqrtsoftplus":
        return False
    if x_size_0 == 0:
        return False
    return True


FP4_BLOCK = 32
FP8_BLOCK = 128


def _per_token_cast_to_fp8_packed_ue8m0(
    x: torch.Tensor,
    gran_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inline ``deep_gemm.utils.per_token_cast_to_fp8(use_ue8m0=True,
    use_packed_ue8m0=True)`` without the ``pack_ue8m0_to_int`` ``.all()``
    debug assertion — that assertion does a CUDA→CPU sync which is illegal
    during ``cudaStreamCapture``.

    Math is bit-identical to the upstream helper.
    """
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    m, n = x.shape
    padded_n = ((n + gran_k - 1) // gran_k) * gran_k
    if padded_n != n:
        x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
        x_padded[:, :n] = x
    else:
        x_padded = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    bits = sf.abs().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    sf_u = (exp.clamp(1, 254) << 23).view(torch.float)
    x_fp8 = (
        (x_view * (1.0 / sf_u.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    sf_packed = (sf_u.view(torch.int) >> 23).to(torch.uint8).view(torch.int)
    return x_fp8, sf_packed


# Module-level cache for the Mega MoE symm-mem dispatch buffer.
# Each MoE layer's symm buffer holds only single-layer staging
# (per-token x/sf, topk, l1_acts/sf, l2_acts/sf) — the previous layer's
# data is no longer needed once the next layer's MoE starts, so a single
# buffer can be reused across all layers.  Without sharing, V4-Flash's
# 64+ MoE-layer instances each allocate ~3.4 GiB at CP=4 → ~218 GiB symm
# memory per rank, OOMing the GB200's 188 GiB after dozens of allocs.
# Keyed by the shape parameters so different model configs in the same
# process don't collide; in practice there's only ever one entry.
_MEGA_BUF_CACHE = {}


def _get_or_create_mega_buf(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    use_fp8_dispatch,
    activation,
):
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        bool(use_fp8_dispatch),
        activation,
    )
    buf = _MEGA_BUF_CACHE.get(key)
    if buf is None:
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group=group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            use_fp8_dispatch=use_fp8_dispatch,
            activation=activation,
        )
        _MEGA_BUF_CACHE[key] = buf
    return buf


# ACCL-EP's intranode dispatch kernel has a compile-time switch over
# ``num_topk`` that only covers {2, 4, 8, 16} (asserts false on others —
# intranode.cu:2237 "Unsupported num_topk").  V4-Flash uses
# ``n_activated_experts = 6``; we pad both ``indices`` and ``weights``
# up to 8 slots with ``-1`` and ``0.0`` so the dispatch accepts them,
# and the padding slots are silently dropped by the per-expert loop
# (``torch.where(idx == -1)`` never matches a real expert index).
_DEEPEP_SUPPORTED_TOPK = (2, 4, 8, 16)


def _mega_moe_available() -> bool:
    """Whether DeepGEMM's ``fp8_fp4_mega_moe`` (symm-mem fused dispatch +
    L1 GEMM + SwiGLU + L2 GEMM + combine, SM100-only) is usable here.

    Requires: deep_gemm ≥ 2.5 (commit 891d57b introduced it), torch ≥ 2.9
    for ``torch.distributed._symmetric_memory``, CUDA device SM100+, and
    an initialised world-size process group of size > 1."""
    try:
        import deep_gemm

        if not hasattr(deep_gemm, "fp8_fp4_mega_moe"):
            return False
    except Exception:
        return False
    try:
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return False
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


def _mega_moe_enabled() -> bool:
    """Default on when ``_mega_moe_available()`` holds; set
    ``DSV4_USE_MEGA_MOE=0`` to force the pre-mega per-expert path."""
    if os.environ.get("DSV4_USE_MEGA_MOE", "1") == "0":
        return False
    return _mega_moe_available()


def _has_fp8_fp4_grouped_kernel() -> bool:
    """True iff the grouped FP4 routed-expert path should be used.

    Opt-in via ``DSV4_USE_GROUPED_FP4=1`` because the path historically
    produced garbage output across V4-Flash's 60 layers when the kernel's
    per-group row alignment (``get_mk_alignment_for_contiguous_layout()``,
    default 128 on SM100) was not honored — single-pass synthetic tests
    were inside 5% rel-diff but per-layer error compounded catastrophically.
    The current ``_grouped_routed_experts`` zeros padding rows + tags
    ``-1`` in ``grouped_layout`` per the kernel contract (mirrors
    ``deepgemm/tests/generators.py::generate_m_grouped_contiguous``),
    but smoke validation on every new SM100 variant is required before
    flipping this default ON.

    Requires deep_gemm ≥ 2.4 (ships ``m_grouped_fp8_fp4_gemm_nt_contiguous``)
    and an SM100+ device.
    """
    if os.environ.get("DSV4_USE_GROUPED_FP4", "0") != "1":
        return False
    try:
        import deep_gemm
    except Exception:
        return False
    if not hasattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous"):
        return False
    if not hasattr(deep_gemm, "get_mk_alignment_for_contiguous_layout"):
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10


class Gate(nn.Module):
    """Per-token routing scores + top-k expert selection.

    Score functions:
      - "softmax"      -> scores.softmax(-1)
      - "sigmoid"      -> scores.sigmoid()
      - "sqrtsoftplus" -> sqrt(softplus(scores))   # V4 default

    For first `n_hash_layers`, routing is deterministic via `tid2eid` lookup
    (token id -> [n_activated_experts] expert ids), and `bias` is None.
    Otherwise, top-k from biased scores; weights pulled from un-biased scores.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        score_func: str = "sqrtsoftplus",
        route_scale: float = 1.0,
        n_hash_layers: int = 0,
        vocab_size: int = 0,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.hash = layer_id < n_hash_layers
        self._factory_mode = weights is not None
        self._dbg_prefix: Optional[str] = None

        if self._factory_mode:
            self.weight = nn.Parameter(weights[f"{prefix}.weight"], requires_grad=False)
            if self.hash:
                assert vocab_size > 0
                self.tid2eid = nn.Parameter(
                    weights[f"{prefix}.tid2eid"].to(torch.int32),
                    requires_grad=False,
                )
                self.bias = None
            else:
                self.bias = nn.Parameter(
                    weights[f"{prefix}.bias"].float(),
                    requires_grad=False,
                )
        else:
            self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
            if self.hash:
                assert vocab_size > 0
                self.tid2eid = nn.Parameter(
                    torch.empty(vocab_size, n_activated_experts, dtype=torch.int32),
                    requires_grad=False,
                )
                self.bias = None
            else:
                self.bias = nn.Parameter(
                    torch.empty(n_routed_experts, dtype=torch.float32)
                )

    def _weight_bf16(self) -> torch.Tensor:
        """Lazy-cached BF16 view of ``self.weight``.

        V4 checkpoint ships gate weights in BF16 or FP32 (loader.py:88); when
        FP32, the previous forward upcast both x and weight to FP32, hitting
        the SIMT sgemm 128x128 path (~80 TFLOPS, no tensor cores).  Caching
        a BF16 view + matmul-ing in BF16 gets us tensor-core throughput.
        See plan_0427.md P1.
        """
        if self.weight.dtype == torch.bfloat16:
            return self.weight
        cached = getattr(self, "_w_bf16", None)
        if (
            cached is None
            or cached.shape != self.weight.shape
            or cached.device != self.weight.device
        ):
            cached = self.weight.to(torch.bfloat16)
            self._w_bf16 = cached
        return cached

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        _dbg = self._dbg_prefix if _rt.ENABLED else None
        # x: [N, dim] flat.  Empty-batch safe — some paths (DP rank with
        # zero local tokens; F.softplus on certain degenerate shapes)
        # blow up with "unknown parameter type" on empty ``scores``, so
        # short-circuit with correctly-shaped empty outputs.
        if x.size(0) == 0:
            return (
                torch.zeros((0, self.topk), dtype=torch.float32, device=x.device),
                torch.zeros((0, self.topk), dtype=torch.long, device=x.device),
            )
        # P1 (plan_0427.md): BF16 GEMM with FP32 epilogue replaces the
        # FP32-everywhere path that previously emitted SIMT sgemm 128x128
        # (127× × 1.15 ms = 145 ms in the 64k+CP=4 trace).  Score numerics
        # then run in FP32 through softplus/sqrt/topk, same as before.
        if os.environ.get("DSV4_GATE_FP32", "0") == "1":
            scores = F.linear(x.float(), self.weight.float())
        else:
            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            scores = F.linear(x_bf16, self._weight_bf16()).float()
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_linear_scores", scores)

        # P2 fast path: fuse softplus+sqrt+bias+topk+normalize for the
        # default V4 score_func='sqrtsoftplus' + non-hash routing.
        if (
            not self.hash
            and self.bias is not None
            and _use_fused_gate(self.score_func, x.size(0))
        ):
            return fused_sqrtsoftplus_gate(
                scores.contiguous(),
                self.bias.contiguous(),
                topk=self.topk,
                route_scale=float(self.route_scale),
                norm_eps=1e-12,
            )

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # "sqrtsoftplus"
            scores = F.softplus(scores).sqrt()
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_activated_scores", scores)

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if _dbg is not None:
            _rt.record_if_level(2, f"{_dbg}_biased_scores", scores)

        if self.hash:
            assert input_ids is not None
            indices = self.tid2eid[input_ids].long()  # [N, topk]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]  # [N, topk]

        weights = original_scores.gather(1, indices)  # [N, topk]
        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """SwiGLU MLP with optional clamping.

    V4-Flash layout:
      - routed experts: storage="fp4" (packed int8 + UE8M0 32-block scale)
      - shared expert:  storage="fp8"

    Factory mode (shared expert, ``storage="fp8"``): each of w1/w2/w3 is
    built through ``LinearFactory`` → ``CudaFp8DeepGEMMLinear``.  Forward
    flattens 3D inputs to 2D for the strategy's GEMM.

    Factory mode (routed expert, ``storage="fp4"``): for now the expert
    keeps ``QuantizedLinear`` and its forward still dequants per call;
    S4 replaces the routed-expert loop with a single grouped
    ``m_grouped_fp8_fp4_gemm_nt_*`` call via ``MoeStrategy``.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 0.0,
        storage: str = "fp8",
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
    ):
        super().__init__()
        self._factory_mode_fp8 = weights is not None and storage == "fp8"

        if self._factory_mode_fp8:
            from rtp_llm.models_py.modules.dsv4.attention import (
                _v4_fp8_linear_from_dict,
            )

            self.w1 = _v4_fp8_linear_from_dict(
                weights, f"{prefix}.w1.weight", f"{prefix}.w1.scale"
            )
            self.w2 = _v4_fp8_linear_from_dict(
                weights, f"{prefix}.w2.weight", f"{prefix}.w2.scale"
            )
            self.w3 = _v4_fp8_linear_from_dict(
                weights, f"{prefix}.w3.weight", f"{prefix}.w3.scale"
            )
        else:
            self.w1 = QuantizedLinear(dim, inter_dim, storage=storage)  # gate
            self.w2 = QuantizedLinear(inter_dim, dim, storage=storage)  # down
            self.w3 = QuantizedLinear(dim, inter_dim, storage=storage)  # up
            if weights is not None:
                # Legacy storage="fp4" — copy weight + scale into Parameters;
                # forward still dequants on the fly (until S4 swaps to grouped GEMM).
                for name in ("w1", "w2", "w3"):
                    lin = getattr(self, name)
                    lin.weight = nn.Parameter(
                        weights[f"{prefix}.{name}.weight"], requires_grad=False
                    )
                    lin.scale = nn.Parameter(
                        weights[f"{prefix}.{name}.scale"], requires_grad=False
                    )
        self.swiglu_limit = swiglu_limit

    def _apply_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Route through a factory LinearBase (expects 2D input) or legacy
        QuantizedLinear (accepts N-D).

        NB: do **not** name this ``_apply`` — that shadows
        ``nn.Module._apply``, breaking ``.to(device, dtype)`` for anything
        containing an ``Expert``.
        """
        if self._factory_mode_fp8 and x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self._apply_layer(self.w1, x).float()
        up = self._apply_layer(self.w3, x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self._apply_layer(self.w2, x.to(dtype))


class MoE(nn.Module):
    """V4 MoE block: routed top-k experts + 1 shared expert.

    For TP=1 / EP=1 we instantiate ALL experts locally. For sharded setups
    only `[start, end)` are non-None — preserving the official ckpt name layout.
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
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self._factory_mode = weights is not None
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
            prefix=f"{prefix}.gate" if self._factory_mode else "",
        )
        assert n_shared_experts == 1, "V4 always has exactly 1 shared expert"
        self.shared_experts = Expert(
            dim,
            moe_inter_dim,
            swiglu_limit=0.0,
            storage="fp8",
            weights=weights,
            prefix=f"{prefix}.shared_experts" if self._factory_mode else "",
        )

        # Pick routed-expert path: DeepGEMM grouped FP4 if kernel is
        # available (deep_gemm ≥ 2.4 ships fp8_fp4_* on SM100); otherwise
        # fall back to the Python per-expert loop with QuantizedLinear
        # (works on any DeepGEMM but keeps FP4 dequant-per-call).
        self._use_grouped_fp4 = self._factory_mode and _has_fp8_fp4_grouped_kernel()

        # Mega MoE: single DeepGEMM kernel fuses dispatch + L1 + SwiGLU +
        # L2 + combine, replacing the ACCL/DeepEP round-trip + per-expert
        # FP4 GEMM loop (``_routed_experts_deepep`` + ``_routed_experts_local``).
        # Applies only when EP > 1 and the kernel is available — env gated.
        self._use_mega_moe = (
            self._factory_mode
            and ep_size > 1
            and not self._use_grouped_fp4
            and _mega_moe_enabled()
        )

        if self._use_mega_moe:
            # Mega MoE: stack EP-local experts, repack SFs to the int32
            # layout required by ``fp8_fp4_mega_moe``, and allocate the
            # symmetric-memory dispatch/combine buffer.  Per-expert
            # ``ModuleList`` is dropped (Mega MoE owns the per-expert
            # compute internally).
            self._setup_mega_moe(weights, prefix)
            self.experts = None
        elif self._use_grouped_fp4:
            # Grouped-GEMM path: stack routed expert weights into 3-D tensors
            # along the expert dim so a single
            # ``m_grouped_fp8_fp4_gemm_nt_contiguous`` produces gate+up in one
            # call (``_w13`` packs w1 over [:inter] + w3 over [inter:2*inter]).
            # No per-expert ``self.experts`` ModuleList: the cuda-graph
            # fallback in ``forward()`` would need it but grouped + cuda
            # graph + ep_size==1 doesn't co-occur in real workloads (grouped
            # is a prefill optimisation; decode under cuda graph leaves the
            # eager ``ep_size==1`` branch via ``_use_grouped_fp4=False``).
            # An assert in ``forward()`` blocks the unsupported combination.
            #
            # Memory: stacked tensors are alloc'd/copied via a per-expert
            # mini-buffer pattern to avoid holding both the loader's dict
            # entries AND the stacked layout simultaneously: ``weights.pop``
            # detaches each tensor from the loader BEFORE the next expert's
            # allocation runs, and ``torch.cuda.empty_cache()`` after the
            # copy loop returns the freed FP4 blocks to the CUDA driver so
            # they don't sit in the caching allocator while KV-pool sizing
            # measures available HBM.
            E, D, inter = n_routed_experts, dim, moe_inter_dim
            device = weights[f"{prefix}.experts.0.w1.weight"].device

            self._w13 = torch.empty(
                (E, 2 * inter, D // 2), dtype=torch.int8, device=device
            )
            self._s13 = torch.empty(
                (E, 2 * inter, D // FP4_BLOCK),
                dtype=torch.float8_e8m0fnu,
                device=device,
            )
            self._w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
            self._s2 = torch.empty(
                (E, D, inter // FP4_BLOCK),
                dtype=torch.float8_e8m0fnu,
                device=device,
            )
            for i in range(E):
                p = f"{prefix}.experts.{i}"
                # w1 → [:inter], w3 → [inter:2*inter]
                self._w13[i, :inter].copy_(weights.pop(f"{p}.w1.weight"))
                self._s13[i, :inter].copy_(weights.pop(f"{p}.w1.scale"))
                self._w13[i, inter:].copy_(weights.pop(f"{p}.w3.weight"))
                self._s13[i, inter:].copy_(weights.pop(f"{p}.w3.scale"))
                self._w2[i].copy_(weights.pop(f"{p}.w2.weight"))
                self._s2[i].copy_(weights.pop(f"{p}.w2.scale"))

            # Return loader's freed FP4 blocks to CUDA so the KV-cache
            # planner sees the real residual HBM rather than what's
            # cached-but-unused inside PyTorch's allocator.
            torch.cuda.empty_cache()

            self.experts = None  # grouped path does not use per-expert dispatch
        else:
            # Legacy/per-expert path. EP sharding: only build the
            # ``n_local_experts`` Experts that live on this rank; slots
            # for non-local experts are None.  Preserves V4-official
            # indexing convention (``self.experts[global_idx]``) so
            # forward loops stay identical across ranks.
            self.experts = nn.ModuleList(
                [
                    (
                        Expert(
                            dim,
                            moe_inter_dim,
                            swiglu_limit=swiglu_limit,
                            storage="fp4",
                            weights=weights,
                            prefix=(
                                f"{prefix}.experts.{i}" if self._factory_mode else ""
                            ),
                        )
                        if self.local_expert_start <= i < self.local_expert_end
                        else None
                    )
                    for i in range(n_routed_experts)
                ]
            )
            self._w13 = self._s13 = self._w2 = self._s2 = None

    # --- Mega MoE ---------------------------------------------------------

    def _setup_mega_moe(
        self,
        weights: Dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        """Stack EP-local routed expert weights, convert SFs to the
        int32 UTCCP-transposed layout required by ``fp8_fp4_mega_moe``,
        and register the symmetric-memory dispatch buffer.

        V4 ckpt stores, per expert i:
          w1.weight [inter, dim//2] int8  (FP4 gate)
          w3.weight [inter, dim//2] int8  (FP4 up)
          w2.weight [dim, inter//2] int8  (FP4 down)
          ... .scale [inter, dim//FP4_BLOCK] / [dim, inter//FP4_BLOCK] UE8M0

        Mega MoE expects, per expert:
          L1 w [2*inter, dim//2] int8 (gate | up rows concatenated)
          L1 sf [2*inter, ...] int32  (post-``transform_sf_into_required_layout``
            + ``transform_weights_for_mega_moe``: gate/up interleaved gran=8
            along N, SF UTCCP-transposed)
          L2 w [dim, inter//2] int8
          L2 sf [dim, ...] int32

        The kernel runs a full dispatch + two grouped FP4 GEMMs + SwiGLU
        + combine; we only need to feed it the token activations and the
        topk routing decisions at forward time.
        """
        import deep_gemm
        import torch.distributed as dist

        E = self.n_local_experts
        D = self.dim
        inter = self.moe_inter_dim
        start = self.local_expert_start
        device = weights[f"{prefix}.experts.{start}.w1.weight"].device

        # (1) Stack EP-local experts into [E_local, ...] tensors.  SF is
        # kept as fp32 initially — ``transform_sf_into_required_layout``
        # is the only entry point that produces the int32 layout the
        # kernel consumes.
        w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
        s13 = torch.empty(
            (E, 2 * inter, D // FP4_BLOCK), dtype=torch.float32, device=device
        )
        w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
        s2 = torch.empty((E, D, inter // FP4_BLOCK), dtype=torch.float32, device=device)
        for local_i in range(E):
            p = f"{prefix}.experts.{start + local_i}"
            w13[local_i, :inter].copy_(weights.pop(f"{p}.w1.weight"))
            s13[local_i, :inter].copy_(weights.pop(f"{p}.w1.scale").float())
            w13[local_i, inter:].copy_(weights.pop(f"{p}.w3.weight"))
            s13[local_i, inter:].copy_(weights.pop(f"{p}.w3.scale").float())
            w2[local_i].copy_(weights.pop(f"{p}.w2.weight"))
            s2[local_i].copy_(weights.pop(f"{p}.w2.scale").float())

        # (2) Repack SFs to the kernel's required int32 layout (MN-major,
        # TMA-aligned, packed 4-UE8M0-per-int32).
        s13_int = deep_gemm.transform_sf_into_required_layout(
            s13, 2 * inter, D, (1, FP4_BLOCK), E
        )
        s2_int = deep_gemm.transform_sf_into_required_layout(
            s2, D, inter, (1, FP4_BLOCK), E
        )

        # (3) Apply Mega MoE-specific transforms: L1 gate/up interleave
        # (gran=8 along N) + both SFs UTCCP-transposed.  Returns the
        # final weights that the kernel reads directly each forward.
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13_int),
            (w2, s2_int),
        )

        # Stash as plain attributes (not Parameters — the kernel reads
        # raw int8/int32 buffers with no autograd).  Original stacked
        # fp32 SFs are dropped now that the int layout has been derived.
        self._mega_l1_w = l1_w
        self._mega_l1_sf = l1_sf
        self._mega_l2_w = l2_w
        self._mega_l2_sf = l2_sf

        # (4) Allocate the symmetric-memory buffer.  Uses
        # ``torch.distributed.group.WORLD`` because our DP+EP layout has
        # ``ep_size == world_size`` — every rank holds a distinct 64/256
        # slice.  ``num_max_tokens_per_rank`` caps per-rank token count
        # fed into the MoE; bounded from ``max_tokens_per_rank`` (plumbed
        # from ``V4Args.max_seq_len`` upstream).  The library aligns this
        # up to ``get_token_alignment_for_mega_moe()`` internally (384 on
        # SM100).
        assert dist.is_initialized(), (
            "Mega MoE requires torch.distributed initialised; "
            "_mega_moe_available() should have gated this earlier"
        )
        group = dist.group.WORLD
        # Symm buffer is single-layer staging — share one across all
        # MoE layers via the module-level cache (see _get_or_create_mega_buf).
        self._mega_buf = _get_or_create_mega_buf(
            group=group,
            num_experts=self.n_routed_experts,
            num_max_tokens_per_rank=max(self.max_tokens_per_rank, 1),
            num_topk=self.n_activated_experts,
            hidden=D,
            intermediate_hidden=inter,
            use_fp8_dispatch=True,
            activation="swiglu",
        )
        # Pre-allocate static output buffer — avoids torch.empty((T, D)) inside the
        # forward, which reallocates on every step and blocks CUDA graph capture.
        # Sized to max_tokens_per_rank; forward slices [:T] for the live batch.
        self._mega_y = torch.empty(
            (max(self.max_tokens_per_rank, 1), D),
            dtype=torch.bfloat16,
            device=device,
        )

    def _routed_experts_mega_moe(
        self,
        x: torch.Tensor,  # [T, D] BF16 local-rank tokens
        weights: torch.Tensor,  # [T, topk] FP32 router weights
        indices: torch.Tensor,  # [T, topk] int64 GLOBAL expert IDs
    ) -> torch.Tensor:
        """Run the fused DeepGEMM Mega MoE kernel: dispatch + L1 GEMM +
        SwiGLU + L2 GEMM + combine — all fused, symm-mem backed.

        Returns the combined routed-expert output in FP32 (to match the
        contract of ``_routed_experts_deepep`` / ``_routed_experts_local``).
        """
        import deep_gemm

        T = x.size(0)
        buf = self._mega_buf
        if T > buf.num_max_tokens_per_rank:
            raise RuntimeError(
                f"Mega MoE input tokens={T} exceeds num_max_tokens_per_rank="
                f"{buf.num_max_tokens_per_rank} (derived from max_seq_len / "
                f"max_tokens_per_rank). Raise the budget at startup."
            )
        if T == 0:
            return torch.zeros_like(x, dtype=torch.float32)

        # Per-token FP8 cast with packed UE8M0 group-32 scale — the
        # dispatch side of Mega MoE reads this layout directly.
        # Inline impl avoids deep_gemm's pack_ue8m0_to_int .all() assertion
        # which does a CUDA→CPU sync illegal during stream capture.
        x_fp8, x_sf = _per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)
        # Fill the symm-mem buffer slots.  Only the first T rows are
        # meaningful; the remainder was zero-initialised at buffer
        # alloc (0 is expert 0, but tokens past T aren't read because
        # the kernel uses y.size(0) as the effective token count).
        buf.x[:T].copy_(x_fp8)
        buf.x_sf[:T].copy_(x_sf)
        buf.topk_idx[:T].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:T].copy_(weights.to(torch.float32).contiguous())

        y = self._mega_y[:T]
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=self.swiglu_limit if self.swiglu_limit > 0 else None,
            fast_math=True,
        )
        return y.float()

    # --- Grouped FP4 (single-rank ep_size==1 fast path) ---

    def _grouped_routed_experts(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Single-GPU grouped FP4 expert compute.

        Args:
          x: [N, D] BF16 flattened tokens (post-MoE-gate activation).
          weights: [N, topk] FP32 router weights.
          indices: [N, topk] int64 expert IDs.

        Returns:
          y: [N, D] float32 sum over (token, top-k) of
             ``weight * expert[idx](x)``.

        Flow:
          1. Expand each token into topk copies, sort by expert id.
          2. Compute per-expert counts and pad each group to
             ``get_mk_alignment_for_contiguous_layout()`` (128 on SM100).
             The kernel REQUIRES this alignment — without it, internal
             block scheduling produces wrong outputs even on the valid
             rows; the error compounds catastrophically across the 60
             V4-Flash layers.
          3. Build a padded activation buffer (zero on padding rows) and
             a per-row ``grouped_layout`` tensor (expert id on real rows,
             ``-1`` on padding rows so the kernel skips them).
          4. Quantize, run grouped FP8×FP4 GEMM for gate+up.
          5. Clamped SwiGLU + per-slot router weight (zero on padding rows
             ⇒ they contribute exactly zero to the down result).
          6. Quantize, run grouped FP8×FP4 GEMM for down.
          7. Single-gather un-permute back to token order, reduce top-k.

        Layout reference: ``deepgemm/tests/generators.py::generate_m_grouped_contiguous``.
        """
        import deep_gemm

        N, D = x.shape
        topk = indices.size(-1)
        E = self.n_routed_experts
        inter = self.moe_inter_dim

        if N == 0:
            return torch.zeros(N, D, dtype=torch.float32, device=x.device)

        M_ALIGN = deep_gemm.get_mk_alignment_for_contiguous_layout()

        # (1) flatten + sort by expert
        expanded_x = x.unsqueeze(1).expand(N, topk, D).reshape(N * topk, D)
        flat_indices = indices.flatten().contiguous()  # [NK] int64
        flat_weights = weights.flatten().contiguous()  # [NK] fp32
        perm = torch.argsort(flat_indices, stable=True)  # [NK]
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N * topk, device=x.device, dtype=perm.dtype)
        expert_per_slot = flat_indices.index_select(0, perm)  # [NK] sorted expert ids

        # (2) per-expert counts → exclusive-scan offsets (real + aligned).
        # NOTE the EXCLUSIVE scan: the first expert's slot starts at 0, so
        # ``sorted_offsets[0] == 0``.  An inclusive ``cumsum`` would give the
        # END offset and break ``within_expert`` for the first expert.
        counts = torch.bincount(flat_indices.to(torch.int64), minlength=E)  # [E]
        aligned_counts = ((counts + M_ALIGN - 1) // M_ALIGN) * M_ALIGN  # [E]
        sorted_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])  # [E]
        aligned_offsets = torch.cat(
            [aligned_counts.new_zeros(1), aligned_counts.cumsum(0)[:-1]]
        )  # [E]
        # CPU sync — required to allocate the padded tensor with a Python
        # int shape.  Eager-only by construction; cuda-graph capture takes
        # the legacy local path (see forward()).
        M_padded = int(aligned_counts.sum().item())

        # (3) destination row per slot: aligned_offsets[expert] + within_expert
        slot_idx = torch.arange(N * topk, device=x.device, dtype=torch.int64)
        within_expert = slot_idx - sorted_offsets[expert_per_slot]
        dest = aligned_offsets[expert_per_slot] + within_expert  # [NK]

        # (4) padded activation + grouped_layout (zero-init for padding rows)
        perm_x = torch.zeros(M_padded, D, dtype=x.dtype, device=x.device)
        perm_weights = torch.zeros(M_padded, dtype=flat_weights.dtype, device=x.device)
        grouped_layout = torch.full((M_padded,), -1, dtype=torch.int32, device=x.device)
        perm_x.index_copy_(0, dest, expanded_x.index_select(0, perm))
        perm_weights.index_copy_(0, dest, flat_weights.index_select(0, perm))
        grouped_layout.index_copy_(0, dest, expert_per_slot.to(torch.int32))

        # (5) gate+up GEMM via grouped FP8×FP4
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            perm_x,
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        s13_fp32 = self._s13.float()
        gate_up = torch.empty(
            M_padded, 2 * inter, device=x.device, dtype=torch.bfloat16
        )
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a_fp8, a_scale),
            (self._w13, s13_fp32),
            gate_up,
            grouped_layout,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        del a_fp8, a_scale, s13_fp32

        # (6) clamped SwiGLU + per-slot router weight (padding rows: weight=0
        # so their contribution to down GEMM input is exactly 0)
        gate = gate_up[:, :inter].float()
        up = gate_up[:, inter:].float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        hidden = F.silu(gate) * up
        hidden = hidden * perm_weights.unsqueeze(-1)
        hidden = hidden.to(torch.bfloat16)
        del gate, up, gate_up

        # (7) down GEMM via grouped FP8×FP4
        h_fp8, h_scale = sgl_per_token_group_quant_fp8(
            hidden,
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        s2_fp32 = self._s2.float()
        down_out = torch.empty(M_padded, D, device=x.device, dtype=torch.bfloat16)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, s2_fp32),
            down_out,
            grouped_layout,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        del h_fp8, h_scale, s2_fp32

        # (8) un-permute (single fused gather) + reduce top-k.
        # ``dest[inv_perm[t]]`` = padded position holding token ``t``'s
        # expert output.  ``inv_perm`` undoes the argsort; the ``dest``
        # indirection then jumps to the right padded row.
        gather_idx = dest.index_select(0, inv_perm)  # [NK]
        unperm = down_out.index_select(0, gather_idx).view(N, topk, D)
        return unperm.float().sum(dim=1)  # [N, D] fp32

    def _routed_experts_local(
        self,
        x: torch.Tensor,  # [N, D]
        weights: torch.Tensor,  # [N, k] fp32
        indices: torch.Tensor,  # [N, k] int64 — GLOBAL expert IDs
        y: torch.Tensor,  # [N, D] fp32, accumulator
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

    def _routed_experts_local_graph_safe(
        self,
        x: torch.Tensor,  # [N, D]
        weights: torch.Tensor,  # [N, k] fp32
        indices: torch.Tensor,  # [N, k] int64 — GLOBAL expert IDs
        y: torch.Tensor,  # [N, D] fp32, accumulator
        local_start: int,
        local_end: int,
    ) -> torch.Tensor:
        """Graph-safe variant of :meth:`_routed_experts_local`.

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

    @staticmethod
    def _pad_topk_for_deepep(
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad ``(indices, weights)`` to the nearest supported topk width.

        See ``_DEEPEP_SUPPORTED_TOPK`` docstring at module top.
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

    def _routed_experts_deepep(
        self,
        x: torch.Tensor,  # [N, D] local rank's tokens (BF16)
        weights: torch.Tensor,  # [N, k] fp32
        indices: torch.Tensor,  # [N, k] int64 global expert IDs
    ) -> torch.Tensor:
        """DP+EP path: DeepEP normal dispatch → local per-expert compute
        → DeepEP combine.  Requires ``init_deepep_wrapper`` to have been
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
        assert (
            wrapper.mode == DeepEPMode.NORMAL
        ), f"expected NORMAL DeepEP mode, got {wrapper.mode}"
        buf = wrapper.buffer

        # Pad topk to nearest supported value (V4's 6 → 8).
        indices_p, weights_p = self._pad_topk_for_deepep(indices, weights)

        # 1. Dispatch layout.  indices cast to int64 already.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buf.get_dispatch_layout(indices_p, self.n_routed_experts)

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

        # 3. Local per-expert compute.  ACCL-EP's dispatch returns
        # ``recv_topk_idx`` in the LOCAL index space ``[0, n_local_experts)``
        # (with -1 for tokens not destined for any local expert), NOT the
        # global expert id.  Shift to global so the per-expert loop in
        # ``_routed_experts_local`` indexes ``self.experts[global_i]``
        # correctly.  Also force int64 and contiguous — the ACCL tensor
        # sometimes comes back with a non-standard dtype that triggers
        # ``torch.where(idx == i)`` with "unknown parameter type".
        M = recv_x.size(0)
        y_local = torch.zeros(M, self.dim, dtype=torch.float32, device=recv_x.device)
        if M > 0:
            global_topk_idx = recv_topk_idx.to(torch.int64).contiguous()
            # Shift local→global; keep -1 as -1 (won't match any expert id).
            global_topk_idx = torch.where(
                global_topk_idx == -1,
                global_topk_idx,
                global_topk_idx + self.local_expert_start,
            )
            self._routed_experts_local(
                recv_x.contiguous(),
                recv_topk_weights.contiguous(),
                global_topk_idx,
                y_local,
                self.local_expert_start,
                self.local_expert_end,
            )

        # 4. Combine back to source ranks.  combine expects the tensor
        # dtype to match x (BF16) — cast the fp32 accumulator.
        y_combined, _, _ = buf.combine(
            y_local.to(x.dtype),
            handle,
        )
        return y_combined.float()

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        # Master switch: when MOEDBG=0 the AND short-circuits so neither the
        # layer_id compare nor any record_if_level call site below runs.
        # Instruments layers 0..2 (first CSA layer is L2) and 17..20
        # (cp2_ep1 vs tp1 first divergence window) when enabled.
        _dbg = _rt.ENABLED and (self.layer_id <= 2 or 17 <= self.layer_id <= 20)
        shape = x.size()
        x = x.view(-1, self.dim)
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_x_in", x)
            self.gate._dbg_prefix = f"L{self.layer_id:02d}_moe_gate"
        weights, indices = self.gate(x, input_ids.flatten())
        if _dbg:
            self.gate._dbg_prefix = None
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_topk_weights", weights)
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_topk_indices", indices)

        # DSV4_DEBUG_INDICES=1 prints router output stats per layer per
        # forward, used to diff legacy vs framework load paths to localize
        # numerical drift to a specific layer/op.
        import os as _os
        if _os.environ.get("DSV4_DEBUG_INDICES", "0") == "1":
            _imin = int(indices.min().item()) if indices.numel() else -1
            _imax = int(indices.max().item()) if indices.numel() else -1
            _wmin = float(weights.min().item()) if weights.numel() else 0.0
            _wmax = float(weights.max().item()) if weights.numel() else 0.0
            _wmean = float(weights.float().mean().item()) if weights.numel() else 0.0
            _xmean = float(x.float().mean().item()) if x.numel() else 0.0
            _xstd = float(x.float().std().item()) if x.numel() else 0.0
            print(
                f"[dsv4 indices] L{self.layer_id:02d} N={x.size(0)} "
                f"x.mean={_xmean:+.4e} x.std={_xstd:.4e} "
                f"idx[{_imin},{_imax}] (n_routed={self.n_routed_experts}) "
                f"w[{_wmin:+.4e},{_wmax:+.4e}] w.mean={_wmean:+.4e}",
                flush=True,
            )

        if self._use_mega_moe:
            y = self._routed_experts_mega_moe(x, weights, indices)
        elif self._use_grouped_fp4:
            assert (
                self.ep_size == 1
            ), "grouped FP4 path + ep_size>1 not supported; gated off anyway"
            assert not torch.cuda.is_current_stream_capturing(), (
                "grouped FP4 path uses bincount/cumsum/argsort which abort "
                "cuda-stream capture; do not enable cuda_graph + "
                "DSV4_USE_GROUPED_FP4=1 together (grouped is a prefill "
                "optimisation, decode-under-graph should keep the env off "
                "to fall through to _routed_experts_local)."
            )
            y = self._grouped_routed_experts(x, weights, indices)
        elif self.ep_size == 1:
            # Full 256 experts on this rank.  Under CUDA-graph capture
            # the eager ``torch.where(indices == i)`` per-expert loop is
            # illegal (data-dependent shape ⇒ CPU sync); switch to the
            # graph-safe fixed-shape mask variant.  Eager keeps the fast
            # path for prefill performance.
            y = torch.zeros_like(x, dtype=torch.float32)
            if torch.cuda.is_current_stream_capturing():
                self._routed_experts_local_graph_safe(
                    x,
                    weights,
                    indices,
                    y,
                    0,
                    self.n_routed_experts,
                )
            else:
                self._routed_experts_local(
                    x,
                    weights,
                    indices,
                    y,
                    0,
                    self.n_routed_experts,
                )
        else:
            y = self._routed_experts_deepep(x, weights, indices)

        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_routed_y", y)
        shared_y = self.shared_experts(x).float()
        if _dbg:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_moe_shared_y", shared_y)
        y = y + shared_y
        return y.type_as(x).view(shape)
