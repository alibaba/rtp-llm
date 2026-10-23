"""Explicit V4.1 EP return bytes, independent of the FP32 accumulation dtype."""

from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import quantize_group32


@dataclass(frozen=True)
class V41ReturnLayout:
    hidden: int
    mode: str = "fp32"

    def __post_init__(self):
        if (
            type(self.hidden) is not int
            or self.hidden <= 0
            or self.hidden % 32
            or self.mode not in ("fp32", "mxfp8")
        ):
            raise ValueError("Unsupported V4.1 return geometry or format")

    @property
    def payload_bytes(self):
        return (
            self.hidden * 4 if self.mode == "fp32" else self.hidden + self.hidden // 32
        )

    @property
    def identity(self):
        return {
            "mode": self.mode,
            "hidden": self.hidden,
            "row_bytes": self.payload_bytes,
            "recipe": (
                "fp32-identity-v1"
                if self.mode == "fp32"
                else "fp32-bf16-group32-amax1e-4-linear-ue8m0-v1"
            ),
        }


def pack_return(partial, layout, *, backend="triton", fused=False):
    """Preserve the historical return recipe, including BF16 rounding for MXFP8."""
    if (
        type(layout) is not V41ReturnLayout
        or partial.ndim != 2
        or partial.shape[1] != layout.hidden
        or partial.dtype != torch.float32
        or (torch.is_grad_enabled() and partial.requires_grad)
    ):
        raise ValueError("V4.1 return requires inference FP32 [N,D] partials")
    if not partial.shape[0]:
        return torch.empty(
            (0, layout.payload_bytes), dtype=torch.uint8, device=partial.device
        )
    if layout.mode == "fp32":
        return partial.contiguous().view(torch.uint8)
    if fused and backend == "triton":
        from ._v41_packet_triton import is_supported, write_return

        if is_supported(partial):
            return write_return(partial, layout)
    values, scales = quantize_group32(
        partial.to(torch.bfloat16).contiguous(), backend=backend
    )
    packet = torch.empty(
        (partial.shape[0], layout.payload_bytes),
        dtype=torch.uint8,
        device=partial.device,
    )
    packet[:, : layout.hidden].copy_(values.view(torch.uint8))
    packet[:, layout.hidden :].copy_(scales)
    return packet
