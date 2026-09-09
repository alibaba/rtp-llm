"""Producer implementations selected once when a K3 layer is initialized."""

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.triton_kernels.kimi_kda.attn_res_fp8 import kimi_k3_attn_res_fp8
from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import (
    rmsnorm_fp8,
    sigmoid_gate_fp8,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.rms_norm_gate import (
    kimi_kda_rms_norm_sigmoid_gate,
)


class Fp8AttentionResidual(KimiK3AttentionResidual):
    def forward(self, prefix_sum, block_residual, **kwargs):
        return kimi_k3_attn_res_fp8(
            prefix_sum,
            block_residual,
            self.norm_weight,
            self.projection_weight,
            self.eps,
            **kwargs,
        )


class Fp8RMSNorm(nn.Module):
    def __init__(self, weight, eps, *, retain_bf16=False):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps
        self.retain_bf16 = retain_bf16

    def forward(self, x):
        return rmsnorm_fp8(
            x, self.weight, self.variance_epsilon, retain_bf16=self.retain_bf16
        )


class SigmoidGate(nn.Module):
    def forward(self, x, gate):
        return x * torch.sigmoid(gate.reshape_as(x))


class Fp8SigmoidGate(nn.Module):
    def forward(self, x, gate):
        return sigmoid_gate_fp8(x, gate)


class KdaOutputNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight, self.eps = weight, eps

    def forward(self, output, output_gate, mode):
        if mode == "decode":
            output_dtype = output.dtype
            output_float = output.float()
            rms = torch.rsqrt(
                output_float.square().mean(dim=-1, keepdim=True) + self.eps
            )
            output = output_float * rms
            output = output * self.weight.float()
            output = output * torch.sigmoid(output_gate.float())
            return output.to(dtype=output_dtype)
        return kimi_kda_rms_norm_sigmoid_gate(
            output, output_gate, self.weight, self.eps
        )


class Fp8KdaOutputNorm(KdaOutputNorm):
    def forward(self, output, output_gate, mode):
        from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import (
            kda_output_fp8,
        )

        return kda_output_fp8(output, output_gate, self.weight, self.eps, mode=mode)
