"""Input packers for DeepGEMM ``nvfp4_nvfp4_mega_moe``."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch


class MegaNVFP4InputPacker(ABC):
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


class TorchMegaNVFP4InputPacker(MegaNVFP4InputPacker):
    """DeepGEMM reference packer, useful for correctness diagnosis."""

    name = "torch"

    def pack(self, x, weights, indices, buf, tokens: int) -> None:
        from deep_gemm.utils import per_token_cast_to_nvfp4

        x_fp4, x_sf, x_gsf = per_token_cast_to_nvfp4(
            x.contiguous(), gran_k=16, use_packed_ue4m3=True
        )
        buf.x[:tokens].copy_(x_fp4)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.x_gsf[:tokens].copy_(x_gsf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())


class FusedMegaNVFP4InputPacker(MegaNVFP4InputPacker):
    name = "fused"

    def pack(self, x, weights, indices, buf, tokens: int) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 128 == 0):
            raise RuntimeError(
                "fused NVFP4 MegaMoE input packer requires CUDA bf16 input "
                f"with hidden dim divisible by 128; got device={x.device}, "
                f"dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from .mega_nvfp4_input_packer_triton import fused_pack_mega_nvfp4_inputs

        fused_pack_mega_nvfp4_inputs(
            x,
            weights,
            indices,
            buf.x[:tokens],
            buf.x_sf[:tokens],
            buf.x_gsf[:tokens],
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
        )


def get_mega_nvfp4_input_packer() -> MegaNVFP4InputPacker:
    mode = os.environ.get("GLM5_MEGA_MOE_NVFP4_INPUT_PACKER", "fused").strip().lower()
    if mode == "torch":
        return TorchMegaNVFP4InputPacker()
    if mode in ("auto", "fused"):
        return FusedMegaNVFP4InputPacker()
    raise ValueError(
        f"invalid GLM5_MEGA_MOE_NVFP4_INPUT_PACKER={mode!r}; "
        "expected auto|torch|fused"
    )
