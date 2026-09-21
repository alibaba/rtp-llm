"""Producer implementations selected once when a K3 layer is initialized."""

from torch import nn

from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.triton_kernels.kimi_kda import decode_bf16_producers as bf16
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


class Bf16RMSNorm(nn.Module):
    """BF16 latent norm that consumes projection slices directly."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight, self.variance_epsilon = weight, eps

    def forward(self, x):
        return bf16.latent_rmsnorm(x, self.weight, self.variance_epsilon)


class SigmoidGate(nn.Module):
    def accepts_strided(self, x, gate):
        return gate is not None and bf16.supports_sigmoid_gate(x, gate)

    def forward(self, x, gate):
        return bf16.sigmoid_gate(x, gate)


class Fp8SigmoidGate(nn.Module):
    def forward(self, x, gate):
        return sigmoid_gate_fp8(x, gate)


class KdaOutputNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight, self.eps = weight, eps

    def forward(self, output, output_gate, mode):
        # Prefill already has a fused producer with a distinct reduction and
        # sigmoid expression. Keep that arithmetic and only cache its launch.
        if mode in ("decode", "target_verify"):
            return bf16.kda_norm_gate(output, output_gate, self.weight, self.eps)
        return kimi_kda_rms_norm_sigmoid_gate(
            output,
            output_gate,
            self.weight,
            self.eps,
            use_cached_launch=bf16.supports_kda_prefill_norm_gate(
                output, output_gate, self.weight
            ),
        )


class Fp8KdaOutputNorm(KdaOutputNorm):
    def forward(self, output, output_gate, mode):
        from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import (
            kda_output_fp8,
        )

        return kda_output_fp8(output, output_gate, self.weight, self.eps, mode=mode)
