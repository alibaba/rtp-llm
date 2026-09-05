"""Routing gate with optional hash-based expert selection.

Per-token routing scores + top-k expert selection. Three score functions:
  - "softmax"      -> scores.softmax(-1)
  - "sigmoid"      -> scores.sigmoid()
  - "sqrtsoftplus" -> sqrt(softplus(scores))

For first ``n_hash_layers`` of the network, routing is deterministic via a
``tid2eid`` lookup (token id -> [n_activated_experts] expert ids); ``bias`` is
None. Otherwise routing picks top-k from biased scores; weights are pulled
from un-biased scores.

Optional fused-Triton fast path for ``score_func='sqrtsoftplus' + non-hash``
(see ``_use_fused_gate``); env-gated by ``MOE_GATE_FUSED`` (default ON).
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertGatePayload

# Single-Triton-kernel router-gate epilogue for
# score_func='sqrtsoftplus'.  Replaces ~7 elementwise/reduce/topk launches
# (softplus → sqrt → bias-add → topk → gather → sum → div → mul) with one
# fused kernel.  ~4× per-call speedup, identical top-k indices vs eager.
try:
    from rtp_llm.models_py.triton_kernels.moe.gate_fused import (
        fused_sqrtsoftplus_gate,
        fused_sqrtsoftplus_hash_gate,
    )

    _GATE_FUSED_OK = True
except Exception:  # pragma: no cover
    fused_sqrtsoftplus_gate = None
    fused_sqrtsoftplus_hash_gate = None
    _GATE_FUSED_OK = False


def _use_fused_gate(score_func: str, x_size_0: int) -> bool:
    """Gate for the fused router-gate kernel.

    Defaults to ON (2026-05-04): the kernel is bit-equivalent to the eager
    FP32 epilogue in microbench (max abs diff 4.5e-8 at rel ~2e-7, top-k
    strict-equal 100% across 5 random seeds).  ULP-scale fp32 reduction-order
    drift in ``weights / weights.sum()`` can flip greedy decode on tied or
    near-tied logits across many layers, so model smoke goldens may need to be
    re-captured. The broader value is collapsing 7-10
    elementwise + topk launches per layer into one kernel, which compounds
    nicely with launch-overhead-bound regimes (small prefill, decode).

    Set ``MOE_GATE_FUSED=0`` to revert to the eager epilogue for debugging.
    """
    if os.environ.get("MOE_GATE_FUSED", "1") == "0":
        return False
    if score_func != "sqrtsoftplus":
        return False
    if x_size_0 == 0:
        return False
    if not _GATE_FUSED_OK or fused_sqrtsoftplus_gate is None:
        raise RuntimeError("The fused MoE gate is enabled by default but unavailable")
    return True


def _select_routes_with_nonfinite_fallback(
    original_scores: torch.Tensor,
    ranking_scores: torch.Tensor,
    topk: int,
    route_scale: float,
    normalize: bool,
    indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select routes without exposing non-finite router outputs downstream."""
    # ranking_scores contains original_scores plus the optional bias, so every
    # non-finite original score remains non-finite here as well.
    row_is_finite = torch.isfinite(ranking_scores).all(dim=-1)

    if indices is None:
        safe_ranking_scores = torch.nan_to_num(
            ranking_scores,
            nan=-float("inf"),
            posinf=-float("inf"),
            neginf=-float("inf"),
        )
        indices = safe_ranking_scores.topk(topk, dim=-1)[1]

    weights = original_scores.gather(1, indices)
    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
    weights = weights * route_scale

    # A bad router row must not reach MegaMoE. PyTorch topk still returns valid
    # indices after the replacement above; use uniform finite weights for the
    # whole row so later quantization/dispatch kernels never consume NaN/Inf.
    fallback_weight = float(route_scale) / float(topk)
    weights = torch.where(
        row_is_finite.unsqueeze(-1),
        weights,
        fallback_weight,
    )
    return weights, indices


class Gate(nn.Module):
    """Per-token routing scores + top-k expert selection.

    Score functions:
      - "softmax"      -> scores.softmax(-1)
      - "sigmoid"      -> scores.sigmoid()
      - "sqrtsoftplus" -> sqrt(softplus(scores))

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
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``) keyed by ``W``. Reads
        ``W.moe_gate`` and either ``W.moe_gate_tid2eid`` (hash layers) or
        ``W.moe_gate_bias`` (non-hash)."""
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.hash = layer_id < n_hash_layers
        self.fuse_hash_gate = False
        from rtp_llm.utils.model_weight import W

        assert (
            layer_weights is not None
        ), "Gate requires layer_weights (descriptor path)"
        self.weight = layer_weights[W.moe_gate]
        if self.hash:
            assert vocab_size > 0
            self.tid2eid = layer_weights[W.moe_gate_tid2eid]
            self.bias = None
        else:
            self.bias = layer_weights[W.moe_gate_bias]

    def _weight_bf16(self) -> torch.Tensor:
        """Lazy-cached BF16 view of ``self.weight``.

        Checkpoints may ship gate weights in BF16 or FP32; when
        FP32, the previous forward upcast both x and weight to FP32, hitting
        the SIMT sgemm 128x128 path (~80 TFLOPS, no tensor cores).  Caching
        a BF16 view + matmul-ing in BF16 gets tensor-core throughput.
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

    def _bf16_scores(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        return F.linear(x_bf16, self._weight_bf16())

    def can_prepare_gate_payload(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> bool:
        """Return whether this call can use the fused route-and-pack path."""

        if (
            x.size(0) == 0
            or os.environ.get("MOE_GATE_FP32", "0") == "1"
            or not _use_fused_gate(self.score_func, x.size(0))
        ):
            return False
        if self.hash and input_ids is None:
            raise ValueError("hash-routed MoE requires input_ids")
        return True

    def prepare_gate_payload(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Optional[ExpertGatePayload]:
        """Materialize raw gate scores for a route-and-pack capable backend.

        Returning ``None`` keeps the ordinary gate + executor path available
        for unsupported score functions, explicit FP32/debug execution, and
        empty local batches.
        """

        if not self.can_prepare_gate_payload(x, input_ids):
            return None
        scores = self._bf16_scores(x).contiguous()
        return ExpertGatePayload(
            scores=scores,
            topk=self.topk,
            score_func=self.score_func,
            route_scale=float(self.route_scale),
            bias=None if self.bias is None else self.bias.contiguous(),
            input_ids=(
                input_ids.reshape(-1).contiguous()
                if self.hash and input_ids is not None
                else None
            ),
            tid2eid=(self.tid2eid.contiguous() if self.hash else None),
        )

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [N, dim] flat.  Empty-batch safe — some paths (DP rank with
        # zero local tokens; F.softplus on certain degenerate shapes)
        # blow up with "unknown parameter type" on empty ``scores``, so
        # short-circuit with correctly-shaped empty outputs.
        if x.size(0) == 0:
            return (
                torch.zeros((0, self.topk), dtype=torch.float32, device=x.device),
                torch.zeros((0, self.topk), dtype=torch.long, device=x.device),
            )
        # BF16 GEMM with FP32 epilogue replaces the
        # FP32-everywhere path that previously emitted SIMT sgemm 128x128
        # (127× × 1.15 ms = 145 ms in the 64k+CP=4 trace).  Score numerics
        # then run in FP32 through softplus/sqrt/topk, same as before.
        if os.environ.get("MOE_GATE_FP32", "0") == "1":
            scores = F.linear(x.float(), self.weight.float())
        else:
            scores = self._bf16_scores(x)
            if (
                self.hash
                and self.fuse_hash_gate
                and _use_fused_gate(self.score_func, x.size(0))
            ):
                assert input_ids is not None
                return fused_sqrtsoftplus_hash_gate(
                    scores.contiguous(),
                    input_ids.reshape(-1).contiguous(),
                    self.tid2eid.contiguous(),
                    route_scale=float(self.route_scale),
                    norm_eps=1e-12,
                )
            scores = scores.float()

        # Fuse softplus+sqrt+bias+topk+normalize for sqrtsoftplus + non-hash
        # routing.
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

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        if self.hash:
            assert input_ids is not None
            indices = self.tid2eid[input_ids].long()  # [N, topk]
        else:
            indices = None

        return _select_routes_with_nonfinite_fallback(
            original_scores,
            scores,
            self.topk,
            float(self.route_scale),
            self.score_func != "softmax",
            indices,
        )
