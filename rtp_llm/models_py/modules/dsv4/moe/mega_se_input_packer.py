"""Dedicated input packers for DeepGEMM Mega MoE fused-SE execution."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch

from .quant_layouts import _per_token_cast_to_fp8_packed_ue8m0
from .shared_expert import strict_fused_moe_enabled


class MegaMoeSEInputPacker(ABC):
    name: str

    @abstractmethod
    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
        block_m: int,
    ) -> None:
        raise NotImplementedError


class TorchMegaMoeSEInputPacker(MegaMoeSEInputPacker):
    name = "torch"

    def pack(self, x, weights, indices, buf, tokens, block_m) -> None:
        if strict_fused_moe_enabled():
            raise RuntimeError(
                "DSV4_MOE_STRICT_FUSED=1 forbids TorchMegaMoeSEInputPacker"
            )
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        x_fp8, x_sf = _per_token_cast_to_fp8_packed_ue8m0(safe_x, gran_k=32)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())
        from ._mega_se_input_pack_triton import stage_mega_moe_se_shared_l1_scales

        stage_mega_moe_se_shared_l1_scales(
            buf.x_sf[:tokens], buf.shared_l1_acts_sf, tokens, block_m
        )


class FusedMegaMoeSEInputPacker(MegaMoeSEInputPacker):
    name = "fused"

    def pack(self, x, weights, indices, buf, tokens, block_m) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 128 == 0):
            raise RuntimeError(
                "DSV4 fused MegaMoE-SE input packer requires CUDA bf16 input "
                f"with hidden dim divisible by 128; got device={x.device}, "
                f"dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from ._mega_se_input_pack_triton import fused_pack_mega_moe_se_inputs

        fused_pack_mega_moe_se_inputs(
            x,
            weights,
            indices,
            buf.x[:tokens],
            buf.x_sf[:tokens],
            buf.shared_l1_acts_sf,
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
            block_m,
        )


def get_mega_moe_se_input_packer() -> MegaMoeSEInputPacker:
    mode = os.environ.get("DSV4_MEGA_MOE_INPUT_PACKER", "fused").strip().lower()
    if mode == "torch":
        if strict_fused_moe_enabled():
            raise RuntimeError(
                "DSV4_MOE_STRICT_FUSED=1 forbids DSV4_MEGA_MOE_INPUT_PACKER=torch"
            )
        return TorchMegaMoeSEInputPacker()
    if mode in ("auto", "fused"):
        return FusedMegaMoeSEInputPacker()
    raise ValueError(
        f"invalid DSV4_MEGA_MOE_INPUT_PACKER={mode!r}; expected auto|torch|fused"
    )


__all__ = [
    "FusedMegaMoeSEInputPacker",
    "MegaMoeSEInputPacker",
    "TorchMegaMoeSEInputPacker",
    "get_mega_moe_se_input_packer",
]
