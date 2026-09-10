"""MegaMoE input packer abstraction.

MegaMoE consumes a symmetric-memory dispatch buffer.  The original path builds
temporary FP8 activation and UE8M0 scale tensors, then copies four tensors into
that buffer.  This module centralizes the implementation choice so Triton/CUDA
packers can coexist with the exact torch implementation for explicit debug use.
"""

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


class MegaMoEInputPacker(ABC):
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


class TorchMegaMoEInputPacker(MegaMoEInputPacker):
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
            raise RuntimeError("MOE_STRICT_FUSED=1 forbids TorchMegaMoEInputPacker")
        safe_x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        x_fp8, x_sf = per_token_cast_to_fp8_packed_ue8m0(safe_x, gran_k=32)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())


class FusedMegaMoEInputPacker(MegaMoEInputPacker):
    name = "fused"

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        if not (
            x.dim() == 2
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.size(1) % 128 == 0
        ):
            raise RuntimeError(
                "MegaMoE input packer requires CUDA bf16 input with "
                f"hidden dim divisible by 128; got device={x.device}, "
                f"dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
            fused_pack_mega_moe_inputs,
        )

        fused_pack_mega_moe_inputs(
            x,
            weights,
            indices,
            buf.x[:tokens],
            buf.x_sf[:tokens],
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
        )


def get_mega_moe_input_packer() -> MegaMoEInputPacker:
    mode = mega_moe_input_packer_mode()
    if mode == "torch":
        if strict_fused_moe_enabled():
            raise RuntimeError("MOE_STRICT_FUSED=1 forbids MEGA_MOE_INPUT_PACKER=torch")
        return TorchMegaMoEInputPacker()
    if mode in ("auto", "fused"):
        return FusedMegaMoEInputPacker()
    raise ValueError(
        f"invalid MEGA_MOE_INPUT_PACKER={mode!r}; expected auto|torch|fused"
    )
