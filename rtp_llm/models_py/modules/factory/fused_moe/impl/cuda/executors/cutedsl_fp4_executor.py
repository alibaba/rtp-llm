from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.kernels.cuda.fp4_kernel import (
    flashinfer_cutedsl_moe_masked,
    scaled_fp4_grouped_quant
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.utils.model_weight import W

from flashinfer import scaled_fp4_grouped_quantize
from rtp_llm.device.device_impl import CudaImpl


class CutedslFp4Executor(FusedMoeExpertExecutor):
    """FP4 MoE executor using FlashInfer's CuteDSL kernels with masked computation."""

    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTEDSL_FP4

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if CutedslFp4Executor can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        # Check if quantization is enabled and uses FP4 (uint8 dtype)
        # FP4 weights are packed as uint8, so we check for quant_config with uint8 dtype
        checker.check(resolver.has_quantization(config) and resolver.get_quant_method(config) == "modelopt_fp4")        

    def __init__(
        self,
        config: MoEConfigAdapter,
        weights: Dict[str, torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize the CutedslFp4Executor.

        Args:
            config: Model configuration.
            weights: Dictionary containing model weights.
            quant_config: Quantization configuration.
        """
        super().__init__(quant_config=quant_config)
        self._config = config
        self._weights = weights

        # Initialize weights
        # For FP4, weights are stored as uint8 (packed FP4)
        # blockscale and alpha are additional quantization parameters
        self._w1 = self._weights.get(W.moe_w1, None)
        self._w2 = self._weights.get(W.moe_w2, None)
        # swizzle w13 and w2 scale
        w1_blockscale = self._weights.get(W.moe_s1, None)
        w2_blockscale = self._weights.get(W.moe_s2, None)
        # self._w1_blockscale = w1_blockscale
        # self._w2_blockscale = w2_blockscale
        if w1_blockscale.dim() == 3:
            self._w1_blockscale = CudaImpl.swizzle_blockscale(w1_blockscale)
            self._w2_blockscale = CudaImpl.swizzle_blockscale(w2_blockscale)
        else:
            self._w1_blockscale = w1_blockscale
            self._w2_blockscale = w2_blockscale
        # For FP4, alpha weights may be stored with specific keys
        # These should be mapped by the weight loader to standard keys
        _w1_alpha = self._weights.get(W.moe_w1_s2, None)
        _w2_alpha = self._weights.get(W.moe_w2_s2, None)

        input_global_scale = self._weights.get(W.moe_w1_i_s, None)
        a2_global_scale = self._weights.get(W.moe_w2_i_s, None)
        # weights_dict = {
        #     "w13": self._w1,
        #     "w2": self._w2,
        #     "w13_scale": w1_blockscale,
        #     "w2_scale": w2_blockscale,
        #     "w13_scale_2": _w1_alpha,
        #     "w2_scale_2": _w2_alpha,
        #     "w13_input_scale": input_global_scale,
        #     "w2_input_scale": a2_global_scale
        # }
        # torch.save(weights_dict, "cutedsl_data/layer0_moe_weights.pt")
        assert self._w1 is not None and self._w2 is not None, "FP4 MoE weights w1 and w2 must be provided"
        assert self._w1_blockscale is not None and self._w2_blockscale is not None, "FP4 MoE blockscale weights must be provided"
        assert _w1_alpha is not None and _w2_alpha is not None, "FP4 MoE alpha weights must be provided"
        assert input_global_scale is not None and a2_global_scale is not None, "FP4 MoE input scale must be provided"
        
        self._w1_alpha = input_global_scale * _w1_alpha
        self._w2_alpha = a2_global_scale * _w2_alpha
        self.input_global_scale = 1 / input_global_scale
        self.a2_global_scale = 1 / a2_global_scale

        # Check FP4 quantization
        assert self.quant_config.is_quantized

    @property
    def local_num_experts(self) -> int:
        assert self._w1 is not None
        return self._w1.size(0)

    def _create_masked_m(
        self, expert_num_tokens: torch.Tensor, max_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Create masked_m tensor from expert_num_tokens.

        Args:
            expert_num_tokens: Number of tokens per expert, shape (num_experts,)
            max_tokens: Maximum number of tokens (M dimension)
            device: Device to create tensor on

        Returns:
            masked_m: Boolean mask tensor, shape (num_experts, max_tokens)
        """
        num_experts = expert_num_tokens.size(0)
        masked_m = torch.zeros(
            (num_experts, max_tokens), dtype=torch.int, device=device
        )
        for expert_id in range(num_experts):
            num_tokens = expert_num_tokens[expert_id].item()
            if num_tokens > 0:
                masked_m[expert_id, :num_tokens] = True
        return masked_m

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        """Execute FP4 MoE computation using CuteDSL masked kernel.

        Args:
            payload: Expert forward payload containing inputs
            activation: Activation function (should be "SiGLU")
            expert_map: Optional expert mapping
            a2_scale: Optional scale for second activation
            apply_router_weight_on_input: Whether to apply router weight on input
            extra_expert_args: Extra arguments for expert execution

        Returns:
            Output tensor, shape (num_experts, max_tokens, hidden_size)
        """
        assert self._w1 is not None and self._w2 is not None
        assert payload.expert_x is not None
        assert payload.expert_tokens_meta is not None

        expert_x = payload.expert_x
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
        assert expert_num_tokens is not None

        assert expert_x.ndim == 3
        E, M, K = expert_x.size()

        # For FP4, weights are packed: w1 is [E, 2*N, K//2], w2 is [E, K, N//2]
        # where N is intermediate_size
        assert self._w1.size(0) == E
        assert self._w1.size(1) % 2 == 0  # w1 should be 2*N
        N = self._w1.size(1) // 2  # intermediate_size
        assert self._w1.size(2) == K // 2, f"w1 last dim should be K//2={K//2}, got {self._w1.size(2)}"
        assert self._w2.size(0) == E
        assert self._w2.size(1) == K, f"w2 second dim should be K={K}, got {self._w2.size(1)}"
        assert self._w2.size(2) == N // 2, f"w2 last dim should be N//2={N//2}, got {self._w2.size(2)}"

        if payload.expert_x.dtype is torch.bfloat16:
            hidden_states, hidden_states_scale = scaled_fp4_grouped_quantize(
                payload.expert_x,
                expert_num_tokens,
                self.input_global_scale
            )
            hidden_states = (hidden_states, hidden_states_scale)
        else:
            assert payload.expert_x.dtype is torch.uint8
            assert payload.expert_x_scale is not None
            hidden_states = (payload.expert_x, payload.expert_x_scale)
        
        # Call the CuteDSL FP4 MoE kernel
        output = flashinfer_cutedsl_moe_masked(
            hidden_states=hidden_states,
            input_global_scale=self.input_global_scale,
            w1=self._w1,
            w1_blockscale=self._w1_blockscale,
            w1_alpha=self._w1_alpha,
            w2=self._w2,
            a2_global_scale=self.a2_global_scale,
            w2_blockscale=self._w2_blockscale,
            w2_alpha=self._w2_alpha,
            masked_m=expert_num_tokens,
        )

        return output

