"""Bind PPU routing arithmetic to the common MoE gate without copying weights."""

from functools import partial

import torch
import torch.nn.functional as F
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.utils.model_weight import W


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
            fp32_gemm=options.get("DSV4_GATE_FP32", "0") == "1",
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
            from ...kernels.ppu_gate_padding import gate_with_padding

            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            scores = self._bf16_fp32_linear(x_bf16, self._weight_bf16()).float()
            scale = float(self.route_scale) if include_route_scale else 1.0
            if self.hash:
                from ...kernels.ppu_hash_gate import (
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
