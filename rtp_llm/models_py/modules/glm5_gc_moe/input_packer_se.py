"""GC-specific fused-SE packing; Mega's default packing remains independent."""

from __future__ import annotations

import os

import torch

from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_se import (
    MegaMoeSeInputPacker,
    TorchMegaMoeSeInputPacker,
)


class FusedGCMoeSeInputPacker(MegaMoeSeInputPacker):
    name = "fused_gc"

    def pack(self, x, weights, indices, buf, tokens: int, block_m: int) -> None:
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[1] % 128 == 0):
            raise RuntimeError(
                "GLM5 fused GC MoE SE input packer requires CUDA bf16 input "
                "with hidden dim divisible by 128; got "
                f"device={x.device}, dtype={x.dtype}, shape={tuple(x.shape)}"
            )
        from .input_packer_triton import fused_pack_gc_moe_inputs

        fused_pack_gc_moe_inputs(
            x,
            weights,
            indices,
            buf.x[:tokens],
            buf.x_sf[:tokens],
            buf.topk_idx[:tokens],
            buf.topk_weights[:tokens],
            out_shared_sf=buf.shared_l1_acts_sf,
            shared_block_m=block_m,
        )


def get_gc_moe_se_input_packer() -> MegaMoeSeInputPacker:
    # Preserve explicit legacy torch/fused selection while allowing GC-specific
    # selection. No process-wide environment is mutated during a forward call.
    default = os.environ.get("GLM5_MEGA_MOE_INPUT_PACKER", "fused")
    default = os.environ.get("GLM5_MEGA_MOE_SE_INPUT_PACKER", default)
    mode = os.environ.get("GLM5_GC_MOE_SE_INPUT_PACKER", default).strip().lower()
    if mode == "torch":
        return TorchMegaMoeSeInputPacker()
    if mode in ("auto", "fused"):
        return FusedGCMoeSeInputPacker()
    raise ValueError(
        f"invalid GLM5_GC_MOE_SE_INPUT_PACKER={mode!r}; expected auto|torch|fused"
    )
