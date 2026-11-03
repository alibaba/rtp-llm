"""PPU routing arithmetic over the shared gate's weight and payload contract."""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import (
    Gate as BaseGate, _GATE_FUSED_OK, _select_routes_with_nonfinite_fallback,
    fused_sqrtsoftplus_gate, fused_sqrtsoftplus_hash_gate,
)


def _use_fused_gate(score_func: str, x_size_0: int, topk: int, *, enabled=None) -> bool:
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
    if enabled is None:
        enabled = os.environ.get("MOE_GATE_FUSED", "1") != "0"
    if not enabled:
        return False
    if score_func != "sqrtsoftplus":
        return False
    if x_size_0 == 0 or not 1 <= topk <= 32:
        return False
    if not _GATE_FUSED_OK or fused_sqrtsoftplus_gate is None:
        raise RuntimeError("The fused MoE gate is enabled by default but unavailable")
    return True


class Gate(BaseGate):
    def __init__(self, *args, linear=None, fp32_gemm=None, fused_gate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bf16_fp32_linear = F.linear if linear is None else linear
        self._fp32_gemm = fp32_gemm
        self._fused_gate = fused_gate

    def _fp32_gemm_enabled(self) -> bool:
        return (
            os.environ.get("MOE_GATE_FP32", "0") == "1"
            if self._fp32_gemm is None else self._fp32_gemm
        )

    def can_prepare_gate_payload(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> bool:
        """Return whether this call can use the fused route-and-pack path."""

        if (
            x.size(0) == 0
            or self._fp32_gemm_enabled()
            or not _use_fused_gate(
                self.score_func, x.size(0), self.topk, enabled=self._fused_gate
            )
        ):
            return False
        if self.hash and input_ids is None:
            raise ValueError("hash-routed MoE requires input_ids")
        return True

    def forward(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        *,
        include_route_scale: bool = True,
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
        effective_route_scale = float(self.route_scale) if include_route_scale else 1.0
        if self._fp32_gemm_enabled():
            scores = F.linear(x.float(), self.weight.float())
        else:
            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            scores = self._bf16_fp32_linear(x_bf16, self._weight_bf16())
            if (
                self.hash
                and self.fuse_hash_gate
                and _use_fused_gate(
                    self.score_func, x.size(0), self.topk, enabled=self._fused_gate
                )
            ):
                assert input_ids is not None
                return fused_sqrtsoftplus_hash_gate(
                    scores.contiguous(),
                    input_ids.reshape(-1).contiguous(),
                    self.tid2eid.contiguous(),
                    route_scale=effective_route_scale,
                    norm_eps=1e-12,
                )
            scores = scores.float()
        if (
            not self.hash
            and self.bias is not None
            and _use_fused_gate(
                self.score_func, x.size(0), self.topk, enabled=self._fused_gate
            )
        ):
            return fused_sqrtsoftplus_gate(
                scores.contiguous(),
                self.bias.contiguous(),
                topk=self.topk,
                route_scale=effective_route_scale,
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
            effective_route_scale,
            self.score_func != "softmax",
            indices,
        )
