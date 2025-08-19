from typing import Dict, Optional

import torch
from libth_transformer import rtp_llm_ops
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import Linear
from rtp_llm.utils.model_weight import W


try:
    from rtp_llm.models_py.modules.fp8_linear import Fp8Linear
    FP8_LINEAR_AVAILABLE = True
except ImportError as e:
    Fp8Linear = None
    FP8_LINEAR_AVAILABLE = False

class DenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        
        use_fp8 = self._should_use_fp8_linear(config, weights)
        
        if use_fp8 and FP8_LINEAR_AVAILABLE:
            self.gate_proj = self._create_fp8_linear(weights[W.ffn_w1], weights[W.ffn_s1], weights.get(W.ffn_b1, None), config)
            self.up_proj = self._create_fp8_linear(weights[W.ffn_w3], weights[W.ffn_s3], weights.get(W.ffn_b3, None), config)
            self.down_proj = self._create_fp8_linear(weights[W.ffn_w2], weights[W.ffn_s2], weights.get(W.ffn_b2, None), config)
        else:
            self.gate_proj = Linear(weights[W.ffn_w1], weights.get(W.ffn_b1, None))
            self.up_proj = Linear(weights[W.ffn_w3], weights.get(W.ffn_b3, None))
            self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))

        if config.activation_type == "SiGLU":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {config.activation_type}")
        
    def _should_use_fp8_linear(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]) -> bool:
        """Check if FP8 linear layers should be used."""
        if not hasattr(config, 'quant_config') or config.quant_config is None:
            return False
        gate_weight = weights.get(W.ffn_w1)
        if gate_weight is None:
            return False
        
        return gate_weight.dtype == torch.float8_e4m3fn
    
    def _create_fp8_linear(self, weight: torch.Tensor, weight_scales: torch.Tensor, 
                          bias: Optional[torch.Tensor], config: GptInitModelParameters) -> Fp8Linear:
        """Create FP8 linear layer."""
        return Fp8Linear(weight, weight_scales, bias, config)

    def forward(self, x: torch.Tensor):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        activated = self.act_fn(gate_output)
        product = activated * up_output
        down_output = self.down_proj(product)
        return down_output


class FusedSiluActDenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        assert config.activation_type == "SiGLU", "FusedSiluActDenseMLP only supports SiGLU activation"
        
        use_fp8 = self._should_use_fp8_linear(config, weights)
        
        if use_fp8 and FP8_LINEAR_AVAILABLE:
            # Use FP8 linear layers
            gate_proj_bias = weights.get(W.ffn_b1, None)
            up_proj_bias = weights.get(W.ffn_b3, None)
            gate_up_proj_bias = None
            if gate_proj_bias is not None and up_proj_bias is not None:
                gate_up_proj_bias = torch.cat([gate_proj_bias, up_proj_bias], dim=-1)
            
            gate_up_proj_weight = torch.cat([weights[W.ffn_w1], weights[W.ffn_w3]], dim=-1)
            gate_up_proj_scales = torch.cat([weights[W.ffn_s1], weights[W.ffn_s3]], dim=-1)
            
            self.gate_up_proj = Fp8Linear(gate_up_proj_weight, gate_up_proj_scales, gate_up_proj_bias, config)
            self.down_proj = Fp8Linear(weights[W.ffn_w2], weights[W.ffn_s2], weights.get(W.ffn_b2, None), config)
        else:
            # Use regular linear layers
            gate_proj_bias = weights.get(W.ffn_b1, None)
            up_proj_bias = weights.get(W.ffn_b3, None)
            gate_up_proj_bias = None
            if gate_proj_bias is not None and up_proj_bias is not None:
                gate_up_proj_bias = torch.cat([gate_proj_bias, up_proj_bias], dim=-1)
            gate_up_proj_weight = torch.cat([weights[W.ffn_w1], weights[W.ffn_w3]], dim=-1)
            self.gate_up_proj = Linear(gate_up_proj_weight, gate_up_proj_bias)
            self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))
    
    def _should_use_fp8_linear(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]) -> bool:
        """Check if FP8 linear layers should be used."""
        if not hasattr(config, 'quant_config') or config.quant_config is None:
            return False
        
        gate_weight = weights.get(W.ffn_w1)
        if gate_weight is None:
            return False
        
        return gate_weight.dtype == torch.float8_e4m3fn

    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)

        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        stream_id = torch.cuda.current_stream().cuda_stream
        rtp_llm_ops.silu_and_mul(output, gate_up, stream_id)
        down_proj = self.down_proj(output)
        return down_proj
