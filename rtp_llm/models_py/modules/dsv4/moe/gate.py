"""DeepSeek-V4 routing gate.

Per-token routing scores + top-k expert selection. Three score functions:
  - "softmax"      -> scores.softmax(-1)
  - "sigmoid"      -> scores.sigmoid()
  - "sqrtsoftplus" -> sqrt(softplus(scores))   # V4 default

For first ``n_hash_layers`` of the network, routing is deterministic via a
``tid2eid`` lookup (token id -> [n_activated_experts] expert ids); ``bias`` is
None. Otherwise routing picks top-k from biased scores; weights are pulled
from un-biased scores.

Optional fused-Triton fast path for ``score_func='sqrtsoftplus' + non-hash``
(see ``_use_fused_gate``); env-gated by ``DSV4_GATE_FUSED`` (default ON).
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    Defaults to ON (2026-05-04): the kernel is bit-equivalent to the eager
    FP32 epilogue in microbench (max abs diff 4.5e-8 at rel ~2e-7, top-k
    strict-equal 100% across 5 random seeds).  ULP-scale fp32 reduction-order
    drift in ``weights / weights.sum()`` can flip greedy decode on tied /
    near-tied logits across ~60 layers, so V4-Flash smoke goldens must be
    re-captured.  Per-call win is ~0.18 ms × 43 layers ≈ ~8 ms / forward
    (~0.25% of 3090 ms prefill); the broader value is collapsing 7-10
    elementwise + topk launches per layer into one kernel, which compounds
    nicely with launch-overhead-bound regimes (small prefill, decode).

    Set ``DSV4_GATE_FUSED=0`` to revert to the eager epilogue for debugging.
    """
    if os.environ.get("DSV4_GATE_FUSED", "1") == "0":
        return False
    if score_func != "sqrtsoftplus":
        return False
    if x_size_0 == 0:
        return False
    if not _GATE_FUSED_OK or fused_sqrtsoftplus_gate is None:
        raise RuntimeError("DSV4 fused gate is enabled by default but unavailable")
    return True


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
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``) keyed by ``W.v4_*`` enum.
        Reads ``W.v4_router_w`` and either ``W.v4_router_tid2eid`` (hash
        layers) or ``W.v4_router_bias`` (non-hash)."""
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.hash = layer_id < n_hash_layers
        self._dbg_prefix: Optional[str] = None

        from rtp_llm.utils.model_weight import W

        assert (
            layer_weights is not None
        ), "Gate requires layer_weights (descriptor path)"
        self.weight = layer_weights[W.v4_router_w]
        if self.hash:
            assert vocab_size > 0
            self.tid2eid = layer_weights[W.v4_router_tid2eid]
            self.bias = None
        else:
            self.bias = layer_weights[W.v4_router_bias]

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
