"""DeepSeek-V4 MoE: Gate (sqrt(softplus) + hash routing) + Expert (clamped SwiGLU) + MoE.

Direct port of `inference/model.py:Gate / Expert / MoE` (BF16-only).
"""

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
        # x: [N, dim] flat
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
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self._factory_mode = weights is not None
        self.moe_inter_dim = moe_inter_dim
        self.swiglu_limit = swiglu_limit
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

        if self._use_grouped_fp4:
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
            # Legacy/per-expert path. Build 256 Experts and (in factory
            # mode) copy their FP4 weight + scale into each QuantizedLinear.
            self.experts = nn.ModuleList([
                Expert(
                    dim, moe_inter_dim, swiglu_limit=swiglu_limit, storage="fp4",
                    weights=weights,
                    prefix=f"{prefix}.experts.{i}" if self._factory_mode else "",
                )
                for i in range(n_routed_experts)
            ])
            self._w13 = self._s13 = self._w2 = self._s2 = None

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

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        if self._use_grouped_fp4:
            y = self._grouped_routed_experts(x, weights, indices)
        else:
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
            for i in range(self.n_routed_experts):
                if counts[i] == 0:
                    continue
                idx, top = torch.where(indices == i)
                y[idx] = y[idx] + self.experts[i](x[idx], weights[idx, top, None]).float()
        y = y + self.shared_experts(x).float()
        return y.type_as(x).view(shape)
