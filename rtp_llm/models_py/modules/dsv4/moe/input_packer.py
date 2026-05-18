"""MegaMoE input packer abstraction.

MegaMoE consumes a symmetric-memory dispatch buffer.  The original path builds
temporary FP8 activation and UE8M0 scale tensors, then copies four tensors into
that buffer.  This module centralizes the implementation choice so Triton/CUDA
packers can coexist with the exact torch implementation for explicit debug use.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch

from .quant_layouts import _per_token_cast_to_fp8_packed_ue8m0
from .shared_expert import strict_fused_moe_enabled


class MegaMoeInputPacker(ABC):
    """Pack routed MegaMoE inputs into the DeepGEMM symm-mem buffer.

    The ``fused`` implementation follows the same math as DeepGEMM's
    ``per_token_cast_to_fp8(use_ue8m0=True, use_packed_ue8m0=True)`` but writes
    the final buffer directly.
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
        if strict_fused_moe_enabled():
            raise RuntimeError(
                "DSV4_MOE_STRICT_FUSED=1 forbids TorchMegaMoeInputPacker"
            )
        _clear_mega_moe_tail(buf, tokens)
        x_fp8, x_sf = _per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())


class FusedMegaMoeInputPacker(MegaMoeInputPacker):
    name = "fused"

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 128 == 0):
            raise RuntimeError(
                "DSV4 fused MegaMoE input packer requires CUDA bf16 input with "
                f"hidden dim divisible by 128; got device={x.device}, "
                f"dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        _clear_mega_moe_tail(buf, tokens)
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


def _clear_mega_moe_tail(buf, tokens: int) -> None:
    if os.environ.get("DSV4_MEGA_MOE_CLEAR_TAIL", "0") != "1":
        return
    capacity = int(buf.topk_idx.shape[0])
    if tokens >= capacity:
        return
    buf.x[tokens:].zero_()
    buf.x_sf[tokens:].zero_()
    buf.topk_idx[tokens:].fill_(-1)
    buf.topk_weights[tokens:].zero_()


def _mode() -> str:
    return os.environ.get("DSV4_MEGA_MOE_INPUT_PACKER", "fused").strip().lower()


def get_mega_moe_input_packer() -> MegaMoeInputPacker:
    mode = _mode()
    if mode == "torch":
        if strict_fused_moe_enabled():
            raise RuntimeError(
                "DSV4_MOE_STRICT_FUSED=1 forbids DSV4_MEGA_MOE_INPUT_PACKER=torch"
            )
        return TorchMegaMoeInputPacker()
    if mode in ("auto", "fused"):
        return FusedMegaMoeInputPacker()
    raise ValueError(
        f"invalid DSV4_MEGA_MOE_INPUT_PACKER={mode!r}; expected auto|torch|fused"
    )
