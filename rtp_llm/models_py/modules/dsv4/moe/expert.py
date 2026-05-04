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
    if not _SILU_MUL_SPLIT_OK:
        return False
    return os.environ.get("DSV4_EXPERT_SILU_FUSED", "1") != "0"


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
        return self._apply_layer(self.w2, x.to(dtype))
