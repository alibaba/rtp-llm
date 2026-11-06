"""PPU routing arithmetic over the shared gate's weight and payload contract."""

import os
from functools import partial
from rtp_llm.utils.model_weight import W
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import (
    Gate as BaseGate, _GATE_FUSED_OK, _select_routes_with_nonfinite_fallback,
    fused_sqrtsoftplus_hash_gate,
)
from rtp_llm.models_py.kernels.ppu.moe.ppu_gate_padding import (
    stable_sqrtsoftplus_gate as fused_sqrtsoftplus_gate,
)


def _use_fused_gate(score_func: str, x_size_0: int, topk: int, *, enabled=None) -> bool:
    """Select the PPU router with stable softplus and FP32 weights."""
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
    def __init__(self, *args, linear=None, fp32_gemm=None, fused_gate=None, fp32_rowwise=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._bf16_fp32_linear = F.linear if linear is None else linear
        self._fp32_gemm = fp32_gemm
        self._fused_gate = fused_gate
        self._fp32_rowwise = fp32_rowwise
        if fp32_rowwise and not self._fp32_gemm_enabled():
            raise ValueError("Rowwise PPU gate requires FP32 projection")

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
            if self._fp32_rowwise:
                scores = torch.cat([F.linear(row[None].float(), self.weight.float()) for row in x])
            else:
                from rtp_llm.models_py.modules.dsv4.ppu.prefill.request_rows import request_row_lengths

                lengths = request_row_lengths(len(x))
                weight = self.weight.float()
                if lengths:
                    scores = torch.cat([F.linear(part.float(), weight) for part in x.split(lengths)])
                else:
                    scores = F.linear(x.float(), weight)
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


class PpuGate(Gate):
    def __init__(self, *args, layer_weights, platform_provider, **kwargs):
        if platform_provider is None:
            raise ValueError("PPU Gate requires an instance-owned operator adapter")
        keys = {
            W.v4_router_w: W.moe_gate,
            W.v4_router_bias: W.moe_gate_bias,
            W.v4_router_tid2eid: W.moe_gate_tid2eid,
        }
        weights = {
            dst: layer_weights[src] for src, dst in keys.items() if src in layer_weights
        }
        options = platform_provider.execution_options
        super().__init__(
            *args,
            layer_weights=weights,
            linear=partial(platform_provider.run_bf16_fp32_linear, F.linear),
            fp32_gemm=options.get("DSV4_GATE_FP32", "1" if options.get("DSV4_PPU_EXPERT_PRECISION") == "a8" else "0") == "1",
            fp32_rowwise=getattr(platform_provider, "_gate_rowwise", False),
            fused_gate=options.get("DSV4_GATE_FUSED", "1") != "0",
            **kwargs,
        )
        self._fuse_padding = getattr(platform_provider, "_decode_fusions", False)

    def forward(self, x, input_ids=None, *, include_route_scale=True):
        result = self._fused_forward(x, input_ids, include_route_scale, None)
        if result is not None:
            return result
        return super().forward(x, input_ids, include_route_scale=include_route_scale)

    def forward_with_mask(self, x, input_ids, active_token_mask):
        """Return the mask still required by the expert dispatch implementation."""
        result = self._fused_forward(x, input_ids, True, active_token_mask)
        if result is not None:
            return *result, None
        weights, indices = super().forward(x, input_ids)
        return weights, indices, active_token_mask

    def _fused_forward(self, x, input_ids, include_route_scale, active_token_mask):
        if (
            self._fuse_padding
            and not (self.hash and self.fuse_hash_gate)
            and not self._fp32_gemm
            and self._fused_gate
            and self.score_func == "sqrtsoftplus"
            and (self.hash or self.bias is not None)
            and 1 <= self.topk <= 16
            and x.is_cuda
            and x.shape[0]
        ):
            from rtp_llm.models_py.kernels.ppu.moe.ppu_gate_padding import gate_with_padding

            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            scores = self._bf16_fp32_linear(x_bf16, self._weight_bf16()).float()
            scale = float(self.route_scale) if include_route_scale else 1.0
            if self.hash:
                from rtp_llm.models_py.kernels.ppu.moe.ppu_hash_gate import (
                    hash_gate_with_padding,
                    is_supported,
                )

                if is_supported(scores, input_ids, self.tid2eid):
                    return hash_gate_with_padding(
                        scores,
                        input_ids,
                        self.tid2eid,
                        scale,
                        active_token_mask=active_token_mask,
                    )
                return None
            return gate_with_padding(
                scores.contiguous(),
                self.bias.contiguous(),
                self.topk,
                scale,
                active_token_mask=active_token_mask,
            )
        return None
