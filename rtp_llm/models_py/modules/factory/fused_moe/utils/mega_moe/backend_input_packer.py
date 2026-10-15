"""MegaMoE input packer abstraction for GLM-5.

Packs BF16 activations + router metadata into DeepGEMM's symmetric-memory
dispatch buffer. Supports torch (debug) and fused (Triton) implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from rtp_llm.models_py.kernels.cuda.mega_moe_quant_layouts import (
    per_token_cast_to_fp8_packed_ue8m0,
)


class MegaMoeInputPacker(ABC):
    """Pack routed MegaMoE inputs into the DeepGEMM symm-mem buffer."""

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
    """Reference implementation using PyTorch ops (for debugging)."""

    name = "torch"

    def pack(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        buf,
        tokens: int,
    ) -> None:
        x_fp8, x_sf = per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())


class FusedMegaMoeInputPacker(MegaMoeInputPacker):
    """Fused Triton implementation (BF16→FP8+UE8M0 + router copy in 1 kernel)."""

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
                "GLM5 fused MegaMoE input packer requires CUDA bf16 input with "
                f"hidden dim divisible by 128; got device={x.device}, "
                f"dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from rtp_llm.models_py.triton_kernels.moe.mega_moe_backend_input_pack import (
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


def get_mega_moe_input_packer():
    """Use the fused production packer; reference packers remain explicit test APIs."""
    return FusedMegaMoeInputPacker()
