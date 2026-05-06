"""MegaMoE input packer abstraction.

MegaMoE consumes a symmetric-memory dispatch buffer.  The original path builds
temporary FP8 activation and UE8M0 scale tensors, then copies four tensors into
that buffer.  This module centralizes the implementation choice so Triton/CUDA
packers can coexist with the exact torch fallback.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch

from .quant_layouts import _per_token_cast_to_fp8_packed_ue8m0


class MegaMoeInputPacker(ABC):
    """Pack routed MegaMoE inputs into the DeepGEMM symm-mem buffer.

    The ``fused`` implementation follows the same math as DeepGEMM's
    ``per_token_cast_to_fp8(use_ue8m0=True, use_packed_ue8m0=True)`` but writes
    the final buffer directly.  Keeping this behind an abstraction lets us
    disable it for shape/debug/capture issues without touching the MegaMoE
    strategy.
    """

    name: str

    @abstractmethod
    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        raise NotImplementedError


class TorchMegaMoeInputPacker(MegaMoeInputPacker):
    name = "torch"

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        x_fp8, x_sf = _per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())


class FusedMegaMoeInputPacker(MegaMoeInputPacker):
    name = "fused"

    def __init__(self) -> None:
        self._fallback = TorchMegaMoeInputPacker()

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 128 == 0):
            return self._fallback.pack(x, weights, indices, buf, tokens)
        try:
            from ._mega_input_pack_triton import fused_pack_mega_moe_inputs

            fused_pack_mega_moe_inputs(
                x,
                weights,
                indices,
                buf.x[:tokens],
                buf.x_sf[:tokens],
                buf.topk_idx[:tokens],
                buf.topk_weights[:tokens],
            )
        except Exception:
            if _mode() == "fused":
                raise
            self._fallback.pack(x, weights, indices, buf, tokens)


def _mode() -> str:
    return os.environ.get("DSV4_MEGA_MOE_INPUT_PACKER", "auto").strip().lower()


def get_mega_moe_input_packer() -> MegaMoeInputPacker:
    mode = _mode()
    if mode == "torch":
        return TorchMegaMoeInputPacker()
    if mode in ("auto", "fused"):
        return FusedMegaMoeInputPacker()
    raise ValueError(
        f"invalid DSV4_MEGA_MOE_INPUT_PACKER={mode!r}; expected auto|torch|fused"
    )
