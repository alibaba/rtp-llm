"""FP8/FP4 single-expert module.

A SwiGLU MLP with optional clamping, used both for *shared* experts
(``storage="fp8"``, factory-mode FP8 path) and for *routed* experts
(``storage="fp4"``, packed int8 + UE8M0 32-block scale, kept for the
LocalLoopExecutor fallback path).

The fused SiLU+clamp+mul Triton path is mandatory for this executor.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    record_function_range,
)

# Fused SiLU + (optional clamp) + element-wise mul replacement for the
# Expert.forward chain. See silu_mul_split.py for the kernel contract.
try:
    from rtp_llm.models_py.triton_kernels.moe.silu_mul_split import silu_mul_split

    _SILU_MUL_SPLIT_OK = True
except Exception:  # pragma: no cover — keep the module importable without Triton
    silu_mul_split = None
    _SILU_MUL_SPLIT_OK = False


def require_silu_mul_split():
    if not _SILU_MUL_SPLIT_OK:
        raise RuntimeError("The fused expert SiLU path is required but unavailable")
    return silu_mul_split


from .quantized_linear import QuantizedLinear, create_fp8_linear


class Expert(nn.Module):
    """SwiGLU MLP with optional clamping.

    Supported layout:
      - routed experts: storage="fp4" (packed int8 + UE8M0 32-block scale)
      - shared expert:  storage="fp8"

    Factory mode (shared expert, ``storage="fp8"``): each of w1/w2/w3 is
    built through ``LinearFactory`` → ``CudaFp8DeepGEMMLinear``.  Forward
    flattens 3D inputs to 2D for the strategy's GEMM.

    Factory mode (routed expert, ``storage="fp4"``): the expert keeps
    ``QuantizedLinear``, whose forward uses native FP8 x FP4 GEMM.
    Higher-throughput strategies replace the routed-expert loop with a grouped
    ``m_grouped_fp8_fp4_gemm_nt_*`` call via ``MoeStrategy``.
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 0.0,
        storage: str = "fp8",
        expert_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``expert_weights`` is a 6-key dict ``{"w1_w","w1_s","w2_w","w2_s",
        "w3_w","w3_s"}`` extracted by the caller from the layer's W tags
        using the canonical fused-MoE weight mapping."""
        super().__init__()
        # storage="fp8" → CudaFp8DeepGEMMLinear (2D input only).
        # storage="fp4" → QuantizedLinear with bound weight/scale (accepts N-D).
        self._uses_fp8_linear = storage == "fp8"

        assert (
            expert_weights is not None
        ), "Expert requires expert_weights (descriptor path)"

        if self._uses_fp8_linear:
            self.w1 = create_fp8_linear(expert_weights["w1_w"], expert_weights["w1_s"])
            self.w2 = create_fp8_linear(expert_weights["w2_w"], expert_weights["w2_s"])
            self.w3 = create_fp8_linear(expert_weights["w3_w"], expert_weights["w3_s"])
        else:
            # Legacy storage="fp4": bind weight and scale directly from the
            # framework tensors; QuantizedLinear executes native DeepGEMM.
            self.w1 = QuantizedLinear(dim, inter_dim, storage=storage)  # gate
            self.w2 = QuantizedLinear(inter_dim, dim, storage=storage)  # down
            self.w3 = QuantizedLinear(dim, inter_dim, storage=storage)  # up
            self.w1.bind_fp4_weight(
                expert_weights["w1_w"],
                expert_weights["w1_s"],
                expert_weights.get("w1_s_gemm"),
            )
            self.w2.bind_fp4_weight(
                expert_weights["w2_w"],
                expert_weights["w2_s"],
                expert_weights.get("w2_s_gemm"),
            )
            self.w3.bind_fp4_weight(
                expert_weights["w3_w"],
                expert_weights["w3_s"],
                expert_weights.get("w3_s_gemm"),
            )
        self.swiglu_limit = swiglu_limit

    def _apply_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Route through CudaFp8DeepGEMMLinear (expects 2D input) or
        QuantizedLinear (accepts N-D).

        NB: do **not** name this ``_apply`` — that shadows
        ``nn.Module._apply``, breaking ``.to(device, dtype)`` for anything
        containing an ``Expert``.
        """
        if self._uses_fp8_linear and x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        with record_function_range("moe.expert.w1_w3"):
            gate = self._apply_layer(self.w1, x).float()
            up = self._apply_layer(self.w3, x).float()
        with record_function_range("moe.expert.silu_mul"):
            # Fused SiLU + optional SwiGLU clamp + multiply (1 launch).
            # Replaces 2 clamp launches (when swiglu_limit>0) + silu + mul.
            x = require_silu_mul_split()(
                gate.contiguous(),
                up.contiguous(),
                clamp_limit=self.swiglu_limit,
            )
        if weights is not None:
            x = weights * x
        with record_function_range("moe.expert.w2"):
            return self._apply_layer(self.w2, x.to(dtype))
