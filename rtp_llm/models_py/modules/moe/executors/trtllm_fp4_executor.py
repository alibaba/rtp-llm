from typing import Any, Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
# from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig
# from rtp_llm.models_py.modules.factory.fused_moe.type import ExecutorType
from rtp_llm.async_decoder_engine.engine_creator import ExecutorType
from rtp_llm.utils.model_weight import W

# Try to import trtllm_fp4_block_scale_routed_moe from flashinfer
try:
    from flashinfer.fused_moe import (
        GatedActType,
        trtllm_fp4_block_scale_routed_moe,
    )
    from flashinfer.utils import device_support_pdl
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False
    trtllm_fp4_block_scale_routed_moe = None
    GatedActType = None
    device_support_pdl = None

# NVFP4 constants
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
NVFP4_BLOCK_SIZE = 16


class TrtllmFp4Executor(FusedMoeExpertExecutor):
    """
    FP4 MoE executor using TensorRT-LLM's trtllm_fp4_block_scale_routed_moe.
    
    This executor supports NVFP4 quantization with block-wise scaling.
    """

    @classmethod
    def executor_type(cls):
        return ExecutorType.CUTLASS_FP8  # Reuse existing type or add new one

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if TrtllmFp4Executor can handle the configuration"""
        if not FLASHINFER_AVAILABLE:
            checker.fail("flashinfer is required for TrtllmFp4Executor")
        
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        quant_method = resolver.get_quant_method(config)
        # Check for NVFP4 quantization (uint8 dtype)
        checker.check(
            quant_method is None
            or (hasattr(quant_method, "quant_dtype") and quant_method.quant_dtype == torch.uint8)
        )

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize the TrtllmFp4Executor.
        
        Args:
            config: Model configuration.
            weights: Dictionary containing model weights.
            quant_config: Quantization configuration.
        """
        super().__init__(quant_config=quant_config)
        self._config = config
        self._weights = weights
        
        # Initialize weights
        self._w1 = self._weights.get(W.moe_w1, None)
        self._w2 = self._weights.get(W.moe_w2, None)
        self._w1_scale = self._weights.get(W.moe_s1, None)
        self._w2_scale = self._weights.get(W.moe_s2, None)
        
        assert self._w1 is not None and self._w2 is not None
        assert self._w1_scale is not None and self._w2_scale is not None
        
        # Check NVFP4 quantization
        if not self.quant_config.is_quantized:
            raise NotImplementedError(
                "TrtllmFp4Executor only supports NVFP4 quantization"
            )
        
        if self.quant_config.quant_dtype != torch.uint8:
            raise NotImplementedError(
                "TrtllmFp4Executor only supports NVFP4 quantization (uint8 dtype)"
            )
        
        if not self.quant_config.is_block_quantized:
            raise NotImplementedError(
                "TrtllmFp4Executor requires block quantization"
            )
        
        # Check block shape
        if self.quant_config.block_shape != [NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE]:
            raise NotImplementedError(
                f"TrtllmFp4Executor only supports block shape {[NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE]}"
            )
        
        # Compute global scales for weights (if not already computed)
        # These are used for dequantization scale computation
        self._w1_global_scale = None
        self._w2_global_scale = None
        self._compute_global_scales()
        
        # Get device capability
        self._device = next(iter(self._weights.values())).device
        self._enable_pdl = device_support_pdl(self._device) if device_support_pdl else None

    def _compute_global_scales(self):
        """Compute global scales for weights based on their amax values."""
        # For NVFP4, we typically use a fixed global scale or compute from amax
        # Here we use a simplified approach - in practice, these should be pre-computed
        # during weight quantization
        w1_amax = torch.abs(self._w1.to(torch.float32)).max()
        w2_amax = torch.abs(self._w2.to(torch.float32)).max()
        
        self._w1_global_scale = torch.tensor(
            [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax],
            device=self._w1.device,
            dtype=torch.float32,
        )
        self._w2_global_scale = torch.tensor(
            [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax],
            device=self._w2.device,
            dtype=torch.float32,
        )

    @property
    def local_num_experts(self) -> int:
        assert self._w1 is not None
        return self._w1.size(0)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Execute MoE forward pass using trtllm_fp4_block_scale_routed_moe.
        
        Args:
            payload: Expert forward payload containing quantized inputs
            activation: Activation function name (e.g., "silu")
            expert_map: Optional expert mapping tensor
            a2_scale: Optional activation scale for second layer
            apply_router_weight_on_input: Whether to apply router weights on input
            extra_expert_args: Optional extra arguments
        
        Returns:
            Output tensor of shape [num_local_experts, M, K]
        """
        assert self._w1 is not None and self._w2 is not None
        assert payload.expert_x is not None
        assert payload.expert_tokens_meta is not None
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None

        expert_x = payload.expert_x  # [E, M, K//2] uint8
        expert_x_scale = payload.expert_x_scale  # [E, M, K//16] float8_e4m3fn
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens  # [E] int32
        expert_topk_ids = payload.expert_topk_ids  # [seq_len, top_k] int32
        expert_topk_weights = payload.expert_topk_weights  # [seq_len, top_k] bfloat16

        assert expert_x.ndim == 3
        E, M, K_half = expert_x.size()
        K = K_half * 2  # Unpacked hidden size

        # Get weight shapes
        _, N, _ = self._w1.size()  # [E, N, K//2]
        assert N % 2 == 0
        intermediate_size = N // 2
        assert self._w1.size(0) == E
        assert self._w1.size(2) == K_half
        assert self._w2.size(0) == E
        assert self._w2.size(1) == K_half, f"w2 shape: {self._w2.shape}, K_half: {K_half}"
        assert self._w2.size(2) == intermediate_size // 2

        # Compute total number of tokens
        total_tokens = expert_num_tokens.sum().item()
        top_k = expert_topk_ids.size(1)

        # Reconstruct 2D hidden_states from 3D expert_x
        # This is a simplified approach - in practice, we might need to track
        # the original token ordering
        hidden_states = torch.zeros(
            (total_tokens, K_half), device=expert_x.device, dtype=torch.uint8
        )
        hidden_states_scale = torch.zeros(
            (total_tokens, K // NVFP4_BLOCK_SIZE),
            device=expert_x_scale.device,
            dtype=torch.float8_e4m3fn,
        )
        
        token_idx = 0
        for expert_id in range(E):
            num_tokens = expert_num_tokens[expert_id].item()
            if num_tokens > 0:
                hidden_states[token_idx : token_idx + num_tokens] = expert_x[
                    expert_id, :num_tokens, :
                ]
                hidden_states_scale[token_idx : token_idx + num_tokens] = expert_x_scale[
                    expert_id, :num_tokens, :
                ]
                token_idx += num_tokens

        # Build packed tensor from topk_ids and topk_weights
        # Format: (topk_ids << 16) | expert_weights.view(int16)
        topk_ids_int32 = expert_topk_ids.to(torch.int32)
        expert_weights_bf16 = expert_topk_weights.to(torch.bfloat16)
        packed_tensor = (topk_ids_int32 << 16) | expert_weights_bf16.view(torch.int16)

        # Prepare weight tensors
        # w13 combines w1 and w3 (gate and value) for SwiGLU
        # In our case, w1 is already [E, N, K//2] where N = 2 * intermediate_size
        w13 = self._w1  # [E, 2*intermediate_size, K//2]
        w13_scale = self._w1_scale  # [E, 2*intermediate_size, K//16]
        w2 = self._w2  # [E, K//2, intermediate_size//2]
        w2_scale = self._w2_scale  # [E, K//16, intermediate_size//16]

        # Compute output scale scalars
        # These are used for dequantization in the MoE kernel
        input_global_scale = torch.tensor(
            [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX],
            device=self._device,
            dtype=torch.float32,
        )
        w13_global_scale = self._w1_global_scale
        w2_global_scale = self._w2_global_scale

        output1_scale_scalar = torch.tensor(
            [input_global_scale.item() * w13_global_scale.item()] * E,
            device=self._device,
            dtype=torch.float32,
        )
        output1_scale_gate_scalar = torch.tensor(
            [input_global_scale.item() * w13_global_scale.item()] * E,
            device=self._device,
            dtype=torch.float32,
        )
        output2_scale_scalar = torch.tensor(
            [input_global_scale.item() * w2_global_scale.item()] * E,
            device=self._device,
            dtype=torch.float32,
        )

        # Map activation string to GatedActType
        if activation.lower() == "silu" or activation.lower() == "swiglu":
            gated_act_type = GatedActType.SwiGlu.value
        elif activation.lower() == "geglu":
            gated_act_type = GatedActType.GeGlu.value
        else:
            raise NotImplementedError(f"Activation {activation} not supported")

        # Call trtllm_fp4_block_scale_routed_moe
        output = trtllm_fp4_block_scale_routed_moe(
            packed_tensor,
            None,  # routing_bias
            hidden_states,
            hidden_states_scale,
            w13,
            w13_scale,
            None,  # w13_bias
            None,  # gemm1_alpha
            None,  # gemm1_beta
            None,  # gemm1_clamp_limit
            w2,
            w2_scale,
            None,  # w2_bias
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            self._config.expert_num,
            top_k,
            None,  # n_group
            None,  # topk_group
            intermediate_size,
            0,  # local_expert_offset
            E,  # local_num_experts
            None,  # routed_scaling_factor
            None,  # tile_tokens_dim
            0,  # routing_method_type (default: Softmax -> TopK)
            True,  # do_finalize
            self._enable_pdl,
            gated_act_type,
            None,  # output (optional inplace)
        )[0]  # Returns list, get first element

        # Convert output from 2D [total_tokens, K] to 3D [E, M, K]
        # This is the reverse of the reconstruction above
        output_3d = torch.zeros(
            (E, M, K), device=output.device, dtype=output.dtype
        )
        
        token_idx = 0
        for expert_id in range(E):
            num_tokens = expert_num_tokens[expert_id].item()
            if num_tokens > 0:
                output_3d[expert_id, :num_tokens, :] = output[
                    token_idx : token_idx + num_tokens, :
                ]
                token_idx += num_tokens

        return output_3d

