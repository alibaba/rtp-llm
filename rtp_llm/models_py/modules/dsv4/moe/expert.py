"""DeepSeek-V4 single Expert module.

A SwiGLU MLP with optional clamping, used both for V4's *shared* expert
(``storage="fp8"``, factory-mode FP8 path) and for legacy *routed* experts
(``storage="fp4"``, packed int8 + UE8M0 32-block scale, kept for the
LocalLoopStrategy / DeepEPStrategy fallback paths).

The ``_use_silu_mul_split`` helper controls the fused SiLU+clamp+mul Triton
fast path (env var ``DSV4_EXPERT_SILU_FUSED``, default ON).
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range

# Fused SiLU + (optional clamp) + element-wise mul replacement for the
# Expert.forward chain.  Default ON via DSV4_EXPERT_SILU_FUSED=1.  See
# _silu_mul_split_triton.py module docstring.
try:
    from rtp_llm.models_py.modules.dsv4._silu_mul_split_triton import silu_mul_split
    _SILU_MUL_SPLIT_OK = True
except Exception:  # pragma: no cover — keep V4 importable without Triton
    silu_mul_split = None
    _SILU_MUL_SPLIT_OK = False


def _use_silu_mul_split() -> bool:
    enabled = os.environ.get("DSV4_EXPERT_SILU_FUSED", "1") != "0"
    if enabled and not _SILU_MUL_SPLIT_OK:
        raise RuntimeError("DSV4 fused Expert SiLU path is enabled by default but unavailable")
    return enabled


from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear


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
        expert_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``expert_weights`` is a 6-key dict ``{"w1_w","w1_s","w2_w","w2_s",
        "w3_w","w3_s"}`` extracted by the caller from the layer's W tags
        (shared: ``W.v4_shared_w{1,2,3}_{w,s}``; routed: per-expert slice
        of ``W.v4_routed_w{1,2,3}_{w,s}``)."""
        super().__init__()
        # storage="fp8" → CudaFp8DeepGEMMLinear (2D input only).
        # storage="fp4" → QuantizedLinear with bound weight/scale (accepts N-D).
        self._uses_fp8_linear = storage == "fp8"

        assert expert_weights is not None, "Expert requires expert_weights (descriptor path)"

        if self._uses_fp8_linear:
            from rtp_llm.models_py.modules.dsv4.attention import _v4_fp8_linear

            self.w1 = _v4_fp8_linear(expert_weights["w1_w"], expert_weights["w1_s"])
            self.w2 = _v4_fp8_linear(expert_weights["w2_w"], expert_weights["w2_s"])
            self.w3 = _v4_fp8_linear(expert_weights["w3_w"], expert_weights["w3_s"])
        else:
            # Legacy storage="fp4" — bind weight + scale directly from
            # the framework tensors; forward still dequants on the fly
            # (until S4 swaps to grouped GEMM).
            self.w1 = QuantizedLinear(dim, inter_dim, storage=storage)  # gate
            self.w2 = QuantizedLinear(inter_dim, dim, storage=storage)  # down
            self.w3 = QuantizedLinear(dim, inter_dim, storage=storage)  # up
            self.w1.weight = expert_weights["w1_w"]
            self.w1.scale = expert_weights["w1_s"]
            self.w2.weight = expert_weights["w2_w"]
            self.w2.scale = expert_weights["w2_s"]
            self.w3.weight = expert_weights["w3_w"]
            self.w3.scale = expert_weights["w3_s"]
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
        with record_function_range("dsv4.expert.w1_w3"):
            gate = self._apply_layer(self.w1, x).float()
            up = self._apply_layer(self.w3, x).float()
        with record_function_range("dsv4.expert.silu_mul"):
            if _use_silu_mul_split():
                # Fused SiLU + optional SwiGLU clamp + multiply (1 launch).
                # Replaces 2 clamp launches (when swiglu_limit>0) + silu + mul.
                x = silu_mul_split(
                    gate.contiguous(),
                    up.contiguous(),
                    clamp_limit=self.swiglu_limit,
                )
            else:
                if self.swiglu_limit > 0:
                    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                    gate = torch.clamp(gate, max=self.swiglu_limit)
                x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        with record_function_range("dsv4.expert.w2"):
            return self._apply_layer(self.w2, x.to(dtype))
