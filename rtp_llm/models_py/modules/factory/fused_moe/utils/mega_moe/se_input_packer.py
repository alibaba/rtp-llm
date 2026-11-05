"""Dedicated input packers for DeepGEMM Mega MoE fused-SE execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


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
    return FusedMegaMoESEInputPacker()


__all__ = [
    "FusedMegaMoESEInputPacker",
    "MegaMoESEInputPacker",
    "get_mega_moe_se_input_packer",
]
