"""Input packers for FP8xFP4 MegaMoE with a fused shared expert."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch

from .quant_layouts import per_token_cast_to_fp8_packed_ue8m0
from .shared_fp8_scale import stage_shared_fp8_input_scales


class MegaMoeSeInputPacker(ABC):
    """Pack routed metadata and the extra shared-L1 activation scales."""

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


class TorchMegaMoeSeInputPacker(MegaMoeSeInputPacker):
    name = "torch"

    def pack(self, x, weights, indices, buf, tokens: int, block_m: int) -> None:
        x_fp8, x_sf = per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())
        stage_shared_fp8_input_scales(
            buf.x_sf,
            buf.shared_l1_acts_sf,
            tokens,
            block_m,
        )


class FusedMegaMoeSeInputPacker(MegaMoeSeInputPacker):
    name = "fused"

    def pack(self, x, weights, indices, buf, tokens: int, block_m: int) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 128 == 0):
            raise RuntimeError(
                "GLM5 fused MegaMoE SE input packer requires CUDA bf16 input "
                "with hidden dim divisible by 128; got "
                f"device={x.device}, dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from .input_packer_se_triton import fused_pack_mega_moe_se_inputs

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


def get_mega_moe_se_input_packer() -> MegaMoeSeInputPacker:
    """Return the independently configured SE input packer."""
    default = os.environ.get("GLM5_MEGA_MOE_INPUT_PACKER", "fused")
    mode = os.environ.get("GLM5_MEGA_MOE_SE_INPUT_PACKER", default).strip().lower()
    if mode == "torch":
        return TorchMegaMoeSeInputPacker()
    if mode in ("auto", "fused"):
        return FusedMegaMoeSeInputPacker()
    raise ValueError(
        f"invalid GLM5_MEGA_MOE_SE_INPUT_PACKER={mode!r}; " "expected auto|torch|fused"
    )


__all__ = [
    "FusedMegaMoeSeInputPacker",
    "MegaMoeSeInputPacker",
    "TorchMegaMoeSeInputPacker",
    "get_mega_moe_se_input_packer",
]
