"""Dedicated routed-input packers for Kimi K3 MegaMoE with fused SE.

The fused shared-expert specialization consumes the same routed FP8 input
layout as routed-only MegaMoE, but keeps an independent implementation choice
and name so the two strategies cannot silently change one another's behavior.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import torch

from rtp_llm.models_py.modules.dsv4.moe.quant_layouts import (
    _per_token_cast_to_fp8_packed_ue8m0,
)


class KimiK3MegaMoeSeInputPacker(ABC):
    """Pack the routed half of a K3 fused-SE MegaMoE invocation."""

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


class TorchKimiK3MegaMoeSeInputPacker(KimiK3MegaMoeSeInputPacker):
    """Reference packer that materializes FP8 activation and scale tensors."""

    name = "torch"

    def pack(self, x, weights, indices, buf, tokens: int) -> None:
        x_fp8, x_sf = _per_token_cast_to_fp8_packed_ue8m0(x.contiguous(), gran_k=32)
        buf.x[:tokens].copy_(x_fp8)
        buf.x_sf[:tokens].copy_(x_sf)
        buf.topk_idx[:tokens].copy_(indices.to(torch.int64).contiguous())
        buf.topk_weights[:tokens].copy_(weights.to(torch.float32).contiguous())


class FusedKimiK3MegaMoeSeInputPacker(KimiK3MegaMoeSeInputPacker):
    """Triton packer that writes the final symmetric-buffer views directly."""

    name = "fused"

    def pack(self, x, weights, indices, buf, tokens: int) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 128 == 0):
            raise RuntimeError(
                "Kimi K3 fused-SE MegaMoE input packer requires CUDA BF16 "
                "input with hidden dim divisible by 128; got "
                f"device={x.device}, dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from rtp_llm.models_py.modules.dsv4.moe._mega_input_pack_triton import (
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


def get_kimi_k3_mega_moe_se_input_packer() -> KimiK3MegaMoeSeInputPacker:
    """Return the independently configured K3 fused-SE input packer."""

    mode = (
        os.environ.get(
            "KIMI_K3_MEGA_MOE_SE_INPUT_PACKER",
            os.environ.get("DSV4_MEGA_MOE_INPUT_PACKER", "fused"),
        )
        .strip()
        .lower()
    )
    if mode == "torch":
        return TorchKimiK3MegaMoeSeInputPacker()
    if mode in ("auto", "fused"):
        return FusedKimiK3MegaMoeSeInputPacker()
    raise ValueError(
        f"invalid KIMI_K3_MEGA_MOE_SE_INPUT_PACKER={mode!r}; "
        "expected auto|torch|fused"
    )


__all__ = [
    "FusedKimiK3MegaMoeSeInputPacker",
    "KimiK3MegaMoeSeInputPacker",
    "TorchKimiK3MegaMoeSeInputPacker",
    "get_kimi_k3_mega_moe_se_input_packer",
]
