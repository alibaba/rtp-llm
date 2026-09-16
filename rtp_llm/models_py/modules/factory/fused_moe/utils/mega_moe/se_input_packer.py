"""Dedicated input packers for DeepGEMM Mega MoE fused-SE execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    per_token_cast_to_fp8_packed_ue8m0,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config import (
    mega_moe_input_packer_mode,
    strict_fused_moe_enabled,
)


class MegaMoESEInputPacker(ABC):
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


class TorchMegaMoESEInputPacker(MegaMoESEInputPacker):
    name = "torch"

    def pack(self, x, weights, indices, buf, tokens, block_m) -> None:
        if strict_fused_moe_enabled():
            raise RuntimeError("MOE_STRICT_FUSED=1 forbids TorchMegaMoESEInputPacker")
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        x_fp8, x_sf = per_token_cast_to_fp8_packed_ue8m0(safe_x, gran_k=32)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_se_input_pack import (
            stage_mega_moe_se_shared_l1_scales,
        )

        stage_mega_moe_se_shared_l1_scales(
            buf.x_sf[:tokens], buf.shared_l1_acts_sf, tokens, block_m
        )


class FusedMegaMoESEInputPacker(MegaMoESEInputPacker):
    name = "fused"

    def pack(self, x, weights, indices, buf, tokens, block_m) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 128 == 0):
            raise RuntimeError(
                "MegaMoESE input packer requires CUDA bf16 input "
                f"with hidden dim divisible by 128; got device={x.device}, "
                f"dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_se_input_pack import (
            fused_pack_mega_moe_se_inputs,
        )

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


def get_mega_moe_se_input_packer() -> MegaMoESEInputPacker:
    mode = mega_moe_input_packer_mode()
    if mode == "torch":
        if strict_fused_moe_enabled():
            raise RuntimeError("MOE_STRICT_FUSED=1 forbids MEGA_MOE_INPUT_PACKER=torch")
        return TorchMegaMoESEInputPacker()
    if mode in ("auto", "fused"):
        return FusedMegaMoESEInputPacker()
    raise ValueError(
        f"invalid MEGA_MOE_INPUT_PACKER={mode!r}; expected auto|torch|fused"
    )


__all__ = [
    "FusedMegaMoESEInputPacker",
    "MegaMoESEInputPacker",
    "TorchMegaMoESEInputPacker",
    "get_mega_moe_se_input_packer",
]
