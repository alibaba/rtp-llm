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

FP4_BLOCK = 32
FP8_BLOCK = 128

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


def _get_or_create_mega_buf(group, num_experts, num_max_tokens_per_rank,
                            num_topk, hidden, intermediate_hidden,
                            use_fp8_dispatch, activation):
    import deep_gemm
    key = (id(group), num_experts, num_max_tokens_per_rank, num_topk,
           hidden, intermediate_hidden, bool(use_fp8_dispatch), activation)
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
    """True iff the installed DeepGEMM exposes ``m_grouped_fp8_fp4_gemm_nt_*``.

    Currently HARD-WIRED to False — the ``_grouped_routed_experts`` path
    below doesn't honor ``get_mk_alignment_for_contiguous_layout()``'s
    per-group padding requirement (128-row default on SM100): the kernel
    expects each group's rows padded up to alignment with ``-1`` entries
    in ``grouped_layout`` and zeroed activation rows (see
    ``deepgemm/tests/generators.py::generate_m_grouped_contiguous``).

    On synthetic test data the violation is absorbed by the 5% BF16
    rel-diff tolerance (``grouped_moe_equivalence_test``), but in
    real-model flow (V4-Flash, E=256, 60 layers) the per-layer error
    compounds catastrophically and produces garbage output (verified on
    the SM100_ARM smoke after bumping deep_gemm to 2.5.0).

    V4-official ``inference/model.py`` itself uses a per-expert loop
    (``torch.where(indices == i); expert(x[idx])``) — no grouped kernel.
    The K4 perf win instead comes from replacing ``QuantizedLinear``'s
    dequant-to-BF16 forward with ``deep_gemm.fp8_fp4_gemm_nt`` (single-
    group, no alignment constraint) inside each expert's ``w1``/``w2``/
    ``w3``. See ``qlinear.py:QuantizedLinear.forward``.
    """
    return False


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

        if self._factory_mode:
            self.weight = nn.Parameter(weights[f"{prefix}.weight"], requires_grad=False)
            if self.hash:
                assert vocab_size > 0
                self.tid2eid = nn.Parameter(
                    weights[f"{prefix}.tid2eid"].to(torch.int32), requires_grad=False,
                )
                self.bias = None
            else:
                self.bias = nn.Parameter(
                    weights[f"{prefix}.bias"].float(), requires_grad=False,
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
                self.bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [N, dim] flat.  Empty-batch safe — some paths (DP rank with
        # zero local tokens; F.softplus on certain degenerate shapes)
        # blow up with "unknown parameter type" on empty ``scores``, so
        # short-circuit with correctly-shaped empty outputs.
        if x.size(0) == 0:
            return (
                torch.zeros((0, self.topk), dtype=torch.float32, device=x.device),
                torch.zeros((0, self.topk), dtype=torch.long, device=x.device),
            )
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # "sqrtsoftplus"
            scores = F.softplus(scores).sqrt()

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        if self.hash:
            assert input_ids is not None
            indices = self.tid2eid[input_ids].long()        # [N, topk]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]     # [N, topk]

        weights = original_scores.gather(1, indices)        # [N, topk]
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

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0,
                 storage: str = "fp8",
                 weights: Optional[Dict[str, torch.Tensor]] = None,
                 prefix: str = ""):
        super().__init__()
        self._factory_mode_fp8 = weights is not None and storage == "fp8"

        if self._factory_mode_fp8:
            from rtp_llm.models_py.modules.dsv4.attention import _v4_fp8_linear_from_dict
            self.w1 = _v4_fp8_linear_from_dict(weights, f"{prefix}.w1.weight", f"{prefix}.w1.scale")
            self.w2 = _v4_fp8_linear_from_dict(weights, f"{prefix}.w2.weight", f"{prefix}.w2.scale")
            self.w3 = _v4_fp8_linear_from_dict(weights, f"{prefix}.w3.weight", f"{prefix}.w3.scale")
        else:
            self.w1 = QuantizedLinear(dim, inter_dim, storage=storage)   # gate
            self.w2 = QuantizedLinear(inter_dim, dim, storage=storage)   # down
            self.w3 = QuantizedLinear(dim, inter_dim, storage=storage)   # up
            if weights is not None:
                # Legacy storage="fp4" — copy weight + scale into Parameters;
                # forward still dequants on the fly (until S4 swaps to grouped GEMM).
                for name in ("w1", "w2", "w3"):
                    lin = getattr(self, name)
                    lin.weight = nn.Parameter(weights[f"{prefix}.{name}.weight"], requires_grad=False)
                    lin.scale = nn.Parameter(weights[f"{prefix}.{name}.scale"], requires_grad=False)
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

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        assert n_routed_experts % max(ep_size, 1) == 0, (
            f"n_routed_experts={n_routed_experts} must divide ep_size={ep_size}")
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.n_local_experts = n_routed_experts // max(ep_size, 1)
        self.local_expert_start = ep_rank * self.n_local_experts
        self.local_expert_end = self.local_expert_start + self.n_local_experts

        self.gate = Gate(
            layer_id, dim, n_routed_experts, n_activated_experts,
            score_func, route_scale, n_hash_layers, vocab_size,
            weights=weights,
            prefix=f"{prefix}.gate" if self._factory_mode else "",
        )
        assert n_shared_experts == 1, "V4 always has exactly 1 shared expert"
        self.shared_experts = Expert(
            dim, moe_inter_dim, swiglu_limit=0.0, storage="fp8",
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
            # along the expert dim and drop the per-expert ModuleList entirely.
            # `_w13` holds w1 (gate) and w3 (up) concatenated along the N-dim so
            # a single m_grouped_fp8_fp4_gemm_nt_contiguous produces both at
            # once; SwiGLU then splits the output into halves.
            E, D, inter = n_routed_experts, dim, moe_inter_dim
            device = weights[f"{prefix}.experts.0.w1.weight"].device
            self._w13 = torch.empty((E, 2 * inter, D // 2), dtype=torch.int8, device=device)
            self._s13 = torch.empty(
                (E, 2 * inter, D // FP4_BLOCK),
                dtype=torch.float8_e8m0fnu, device=device,
            )
            self._w2 = torch.empty((E, D, inter // 2), dtype=torch.int8, device=device)
            self._s2 = torch.empty(
                (E, D, inter // FP4_BLOCK),
                dtype=torch.float8_e8m0fnu, device=device,
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
            # No per-expert submodule.
            self.experts = None
        else:
            # Legacy/per-expert path. EP sharding: only build the
            # ``n_local_experts`` Experts that live on this rank; slots
            # for non-local experts are None.  Preserves V4-official
            # indexing convention (``self.experts[global_idx]``) so
            # forward loops stay identical across ranks.
            self.experts = nn.ModuleList([
                Expert(
                    dim, moe_inter_dim, swiglu_limit=swiglu_limit, storage="fp4",
                    weights=weights,
                    prefix=f"{prefix}.experts.{i}" if self._factory_mode else "",
                )
                if self.local_expert_start <= i < self.local_expert_end else None
                for i in range(n_routed_experts)
            ])
            self._w13 = self._s13 = self._w2 = self._s2 = None

    # --- Mega MoE ---------------------------------------------------------

    def _setup_mega_moe(
        self, weights: Dict[str, torch.Tensor], prefix: str,
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
        import torch.distributed as dist
        import deep_gemm

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
        s13 = torch.empty((E, 2 * inter, D // FP4_BLOCK), dtype=torch.float32, device=device)
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
            s13, 2 * inter, D, (1, FP4_BLOCK), E)
        s2_int = deep_gemm.transform_sf_into_required_layout(
            s2, D, inter, (1, FP4_BLOCK), E)

        # (3) Apply Mega MoE-specific transforms: L1 gate/up interleave
        # (gran=8 along N) + both SFs UTCCP-transposed.  Returns the
        # final weights that the kernel reads directly each forward.
        (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
            (w13, s13_int), (w2, s2_int),
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

    def _routed_experts_mega_moe(
        self,
        x: torch.Tensor,                  # [T, D] BF16 local-rank tokens
        weights: torch.Tensor,            # [T, topk] FP32 router weights
        indices: torch.Tensor,            # [T, topk] int64 GLOBAL expert IDs
    ) -> torch.Tensor:
        """Run the fused DeepGEMM Mega MoE kernel: dispatch + L1 GEMM +
        SwiGLU + L2 GEMM + combine — all fused, symm-mem backed.

        Returns the combined routed-expert output in FP32 (to match the
        contract of ``_routed_experts_deepep`` / ``_routed_experts_local``).
        """
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp8

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
        x_fp8, x_sf = per_token_cast_to_fp8(
            x.contiguous(), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True,
        )
        # Fill the symm-mem buffer slots.  Only the first T rows are
        # meaningful; the remainder was zero-initialised at buffer
        # alloc (0 is expert 0, but tokens past T aren't read because
        # the kernel uses y.size(0) as the effective token count).
        buf.x[:T].copy_(x_fp8)
        buf.x_sf[:T].copy_(x_sf)
        buf.topk_idx[:T].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:T].copy_(weights.to(torch.float32).contiguous())

        y = torch.empty((T, self.dim), dtype=torch.bfloat16, device=x.device)
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

    # --- Grouped FP4 (dormant; retained for future enabling) ---

    def _grouped_routed_experts(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
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
          1. Expand each token into topk copies and permute so each expert's
             tokens are contiguous.
          2. Quantize the permuted activation to FP8-e4m3fn with per-token
             group-128 UE8M0 scale.
          3. Call ``m_grouped_fp8_fp4_gemm_nt_contiguous`` with the stacked
             ``_w13`` (FP4) to produce gate || up in a single GEMM.
          4. Apply clamped SwiGLU, multiply by the per-slot router weight.
          5. Quantize again, call grouped FP4 GEMM with ``_w2`` for down.
          6. Un-permute and reduce top-k slots back to token rows.
        """
        N, D = x.shape
        topk = indices.size(-1)
        E = self.n_routed_experts
        inter = self.moe_inter_dim

        # (1) expand + permute
        expanded_x = x.unsqueeze(1).expand(N, topk, D).reshape(N * topk, D)
        flat_indices = indices.flatten()                     # [N*topk] int64
        flat_weights = weights.flatten()                     # [N*topk] fp32
        perm = torch.argsort(flat_indices, stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N * topk, device=perm.device)
        perm_x = expanded_x.index_select(0, perm).contiguous()
        perm_experts = flat_indices.index_select(0, perm).to(torch.int32)
        perm_weights = flat_weights.index_select(0, perm)

        # (2) act quant → FP8 e4m3fn + UE8M0 scale (column-major, TMA-aligned)
        a_fp8, a_scale = sgl_per_token_group_quant_fp8(
            perm_x, group_size=FP8_BLOCK, eps=1e-4,
            column_major_scales=True, scale_tma_aligned=True,
            scale_ue8m0=True,
        )

        # (3) grouped gate+up GEMM
        s13_fp32 = self._s13.float()
        gate_up = torch.empty(N * topk, 2 * inter, device=x.device, dtype=torch.bfloat16)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a_fp8, a_scale),
            (self._w13, s13_fp32),
            gate_up, perm_experts,
            recipe_a=(1, FP8_BLOCK), recipe_b=(1, FP4_BLOCK),
        )
        del a_fp8, a_scale, s13_fp32

        # (4) clamped SwiGLU + router weight multiplication
        gate = gate_up[:, :inter].float()
        up = gate_up[:, inter:].float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        hidden = F.silu(gate) * up
        hidden = hidden * perm_weights.unsqueeze(-1)
        hidden = hidden.to(torch.bfloat16)
        del gate, up, gate_up

        # (5) down GEMM
        h_fp8, h_scale = sgl_per_token_group_quant_fp8(
            hidden, group_size=FP8_BLOCK, eps=1e-4,
            column_major_scales=True, scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        s2_fp32 = self._s2.float()
        down_out = torch.empty(N * topk, D, device=x.device, dtype=torch.bfloat16)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (h_fp8, h_scale),
            (self._w2, s2_fp32),
            down_out, perm_experts,
            recipe_a=(1, FP8_BLOCK), recipe_b=(1, FP4_BLOCK),
        )
        del h_fp8, h_scale, s2_fp32

        # (6) un-permute and reduce top-k
        unperm = down_out.index_select(0, inv_perm).view(N, topk, D)
        return unperm.float().sum(dim=1)                    # [N, D] fp32

    def _routed_experts_local(
        self,
        x: torch.Tensor,                  # [N, D]
        weights: torch.Tensor,            # [N, k] fp32
        indices: torch.Tensor,            # [N, k] int64 — GLOBAL expert IDs
        y: torch.Tensor,                  # [N, D] fp32, accumulator
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

    @staticmethod
    def _pad_topk_for_deepep(
        indices: torch.Tensor, weights: torch.Tensor,
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
        return (torch.cat([indices, pad_idx], dim=-1),
                torch.cat([weights, pad_w], dim=-1))

    def _routed_experts_deepep(
        self,
        x: torch.Tensor,                  # [N, D] local rank's tokens (BF16)
        weights: torch.Tensor,            # [N, k] fp32
        indices: torch.Tensor,            # [N, k] int64 global expert IDs
    ) -> torch.Tensor:
        """DP+EP path: DeepEP normal dispatch → local per-expert compute
        → DeepEP combine.  Requires ``init_deepep_wrapper`` to have been
        called by the engine (``backend_manager.py``).
        """
        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPWrapper, DeepEPMode,
        )
        if DeepEPWrapper._instance is None:
            raise RuntimeError(
                "DeepEPWrapper not initialised; ep_size>1 requires "
                "init_deepep_wrapper() at engine startup (enable via "
                "--use_deepep_moe 1)."
            )
        wrapper = DeepEPWrapper._instance
        assert wrapper.mode == DeepEPMode.NORMAL, (
            f"expected NORMAL DeepEP mode, got {wrapper.mode}")
        buf = wrapper.buffer

        # Pad topk to nearest supported value (V4's 6 → 8).
        indices_p, weights_p = self._pad_topk_for_deepep(indices, weights)

        # 1. Dispatch layout.  indices cast to int64 already.
        (num_tokens_per_rank,
         num_tokens_per_rdma_rank,
         num_tokens_per_expert,
         is_token_in_rank,
         _,
         ) = buf.get_dispatch_layout(indices_p, self.n_routed_experts)

        # 2. Dispatch the BF16 tokens + topk scaffolding.
        (recv_x,
         recv_topk_idx,
         recv_topk_weights,
         num_recv_tokens_per_expert_list,
         handle,
         _,
         ) = buf.dispatch(
            x, None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            indices_p, weights_p,
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
                self.local_expert_start, self.local_expert_end,
            )

        # 4. Combine back to source ranks.  combine expects the tensor
        # dtype to match x (BF16) — cast the fp32 accumulator.
        y_combined, _, _ = buf.combine(
            y_local.to(x.dtype), handle,
        )
        return y_combined.float()

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())

        if self._use_mega_moe:
            y = self._routed_experts_mega_moe(x, weights, indices)
        elif self._use_grouped_fp4:
            assert self.ep_size == 1, (
                "grouped FP4 path + ep_size>1 not supported; gated off anyway")
            y = self._grouped_routed_experts(x, weights, indices)
        elif self.ep_size == 1:
            # Fast path: full 256 experts on this rank, plain per-expert loop.
            y = torch.zeros_like(x, dtype=torch.float32)
            self._routed_experts_local(
                x, weights, indices, y, 0, self.n_routed_experts,
            )
        else:
            y = self._routed_experts_deepep(x, weights, indices)

        y = y + self.shared_experts(x).float()
        return y.type_as(x).view(shape)
