"""Load-time rotated INT4 copies; the owning DenseMLP retains BF16 fallbacks."""

import logging

import torch
from torch import nn

from rtp_llm.models_py.kernels.cuda.w4a16_sm120 import gemm, transform

_BLOCK = 128


def rotate(inputs: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    signed = inputs * signs
    return rtp_llm_ops.w4a16_sm120_hadamard(
        signed.reshape(-1, _BLOCK), _BLOCK**-0.5
    ).reshape(inputs.shape)


class RotatedW4A16Linear(nn.Module):
    @torch.inference_mode()
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, seed: int):
        super().__init__()
        self.output_size, self.input_size = weight.shape
        generator = torch.Generator(device="cpu").manual_seed(seed)
        signs = torch.randint(0, 2, (self.input_size,), generator=generator)
        self.register_buffer(
            "signs", (signs * 2 - 1).to(device=weight.device, dtype=weight.dtype)
        )
        self.register_buffer("bias", bias)
        self.register_buffer(
            "packed",
            torch.empty(
                (self.input_size // 16, self.output_size // 2, 4),
                dtype=torch.int32,
                device=weight.device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(
                (self.input_size // 32, self.output_size, 4),
                dtype=torch.uint8,
                device=weight.device,
            ),
        )
        for start in range(0, self.output_size, 256):
            end = min(start + 256, self.output_size)
            rotated = rotate(weight[start:end].float(), self.signs).to(weight.dtype)
            packed, scales = transform(rotated.contiguous())
            self.packed[:, start // 2 : end // 2].copy_(packed)
            self.scales[:, start:end].copy_(scales)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = gemm(rotate(inputs, self.signs), self.packed, self.scales)
        return output if self.bias is None else output + self.bias


class W4A16DenseMLP(nn.Module):
    def __init__(self, up_proj: nn.Module, down_proj: nn.Module):
        super().__init__()
        self.up_proj = RotatedW4A16Linear(up_proj.weight, up_proj.bias, seed=0)
        self.down_proj = RotatedW4A16Linear(down_proj.weight, down_proj.bias, seed=1)

    @classmethod
    def create(cls, up_proj: nn.Module, down_proj: nn.Module):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        required_ops = (
            "w4a16_sm120_transform",
            "w4a16_sm120_gemm",
            "w4a16_sm120_hadamard",
        )
        missing_ops = [name for name in required_ops if not hasattr(rtp_llm_ops, name)]
        if missing_ops:
            raise RuntimeError(f"Rebuild RTP compute ops: missing {missing_ops}")
        if not all(isinstance(proj, CudaF16Linear) for proj in (up_proj, down_proj)):
            raise ValueError("W4A16 FFN requires unquantized CUDA BF16 linears")
        for proj in (up_proj, down_proj):
            weight = proj.weight
            if (
                not weight.is_cuda
                or torch.version.hip is not None
                or weight.dtype != torch.bfloat16
                or torch.cuda.get_device_capability(weight.device) != (12, 0)
            ):
                raise ValueError("W4A16 FFN requires BF16 weights on SM120")
            if weight.shape[0] % 256 or weight.shape[1] % _BLOCK:
                logging.warning(
                    "W4A16 FFN falls back to BF16 for shape %s: unmodified kernel "
                    "requires aligned tiles; Hadamard requires K %% 128 == 0",
                    tuple(weight.shape),
                )
                return None
        backend = cls(up_proj, down_proj)
        logging.info(
            "W4A16 FFN initialized: up=%s down=%s, block=128, 0<M<64",
            tuple(up_proj.weight.shape),
            tuple(down_proj.weight.shape),
        )
        return backend

    def forward(self, inputs: torch.Tensor, activation: nn.Module) -> torch.Tensor:
        return self.down_proj(activation(self.up_proj(inputs)))
