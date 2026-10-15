"""Unified dense MLP implementation supporting multiple activation types."""

from typing import Dict, Optional, Type

import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ActivationType, HWKernelConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

_ACTIVATION_FUNC_MAP: Dict[ActivationType, Type[nn.Module]] = {
    ActivationType.Swiglu: FusedSiluAndMul,
    ActivationType.Gelu: nn.GELU,
}

_GATED_ACTIVATION_TYPE_LIST = [ActivationType.Swiglu]


class DenseMLP(nn.Module):
    """
    Unified DenseMLP implementation supporting both SiGLU and GELU activations.

    - For SiGLU (Swiglu): Uses gate_up_proj + fused silu_and_mul + down_proj
    - For GELU (Gelu): Uses intermediate_proj + GELU activation + output_proj
    """

    def __init__(
        self,
        activation_type: ActivationType,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()

        self.activation_type = activation_type
        self.parallelism_config = parallelism_config
        if self.activation_type not in _ACTIVATION_FUNC_MAP:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        self.act_fn = _ACTIVATION_FUNC_MAP[activation_type]()
        self.is_gated = activation_type in _GATED_ACTIVATION_TYPE_LIST

        if self.is_gated:
            if W.ffn_w13 not in weights:
                self.up_proj = LinearFactory.create_merged_linear(
                    weights,
                    weight_keys=[W.ffn_w1, W.ffn_w3],
                    scale_keys=[W.ffn_s1, W.ffn_s3],
                    bias_keys=[W.ffn_b1, W.ffn_b3],
                    quant_config=quant_config,
                    dim=-1,
                    hw_kernel_config=hw_kernel_config,
                    scale2_keys=[W.ffn_w1_s2, W.ffn_w3_s2],
                    input_scale_keys=[W.ffn_w1_i_s, W.ffn_w3_i_s],
                )
            else:
                self.up_proj = LinearFactory.create_linear_from_weights(
                    weights,
                    W.ffn_w13,
                    W.ffn_s13,
                    W.ffn_b13,
                    quant_config=quant_config,
                    hw_kernel_config=hw_kernel_config,
                    weight_scale_2_key=W.ffn_w13_s2,
                    input_scale_key=W.ffn_w13_i_s,
                )

        else:
            self.up_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.ffn_w3,
                W.ffn_s3,
                W.ffn_b3,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
                weight_scale_2_key=W.ffn_w3_s2,
                input_scale_key=W.ffn_w3_i_s,
            )

        self.down_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.ffn_w2,
            W.ffn_s2,
            W.ffn_b2,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.ffn_w2_s2,
            input_scale_key=W.ffn_w2_i_s,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_allreduce: bool = False,
        quantized_input: Optional[tuple] = None,
    ) -> torch.Tensor:
        if quantized_input is not None and self.can_consume_quantized_input(
            quantized_input[0]
        ):
            return self.forward_quantized(
                quantized_input[0], quantized_input[1], skip_allreduce=skip_allreduce
            )
        # A provided-but-unusable quantized input falls back to the plain
        # path; x is the bf16 normed tensor either way.
        if not self.is_gated and self.activation_type == ActivationType.Gelu:
            activated = self.up_proj.forward_with_bias_gelu(x)
        else:
            up = self.up_proj(x)
            activated = self.act_fn(up)

        ffn_tp_size = self.parallelism_config.get_ffn_tp_size()
        output = self.down_proj(activated)
        if not skip_allreduce and ffn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    def can_consume_quantized_input(self, input_fp8: torch.Tensor) -> bool:
        """True when ``forward_quantized`` accepts this pre-quantized input.

        Only the gated (SiGLU) fp8-per-block ue8m0 DeepGEMM up projection
        qualifies: its ``quantize_input`` produces exactly the packed-scale
        layout the fused norm+quant kernel emits. Wrappers that dispatch
        small-M inputs to FlashInfer only qualify for m >= 32.
        """
        from rtp_llm.models_py.modules.hybrid.fused_norm_quant import (
            linear_supports_quantized_input,
        )

        if not self.is_gated:
            return False
        if input_fp8.dim() != 2 or not input_fp8.is_cuda:
            return False
        return linear_supports_quantized_input(
            self.up_proj, input_fp8.shape[1], m_rows=input_fp8.shape[0]
        )

    def forward_quantized(
        self,
        input_fp8: torch.Tensor,
        input_scales: torch.Tensor,
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        """SiGLU forward with a caller-provided quantized up-proj input.

        Mirrors ``forward`` for the gated path; the down projection keeps its
        own activation-side quantization (its input is the post-activation
        tensor, not this input).
        """
        from rtp_llm.models_py.modules.hybrid.fused_norm_quant import (
            deepgemm_forward_quantized,
        )

        up = deepgemm_forward_quantized(self.up_proj, input_fp8, input_scales)
        activated = self.act_fn(up)

        ffn_tp_size = self.parallelism_config.get_ffn_tp_size()
        output = self.down_proj(activated)
        if not skip_allreduce and ffn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output
