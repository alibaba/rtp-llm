from .add_rmsnorm_fp8_quant_pass import register_add_rmsnorm_fp8_quant_pass
from .indexer_logits_head_gate_pass import register_indexer_logits_head_gate_pass
from .moe_shared_expert_sigmoid_gate_add_pass import (
    register_moe_shared_expert_sigmoid_gate_add_pass,
)
from .rmsnorm_gated_fp8_quant_pass import register_rmsnorm_gated_fp8_quant_pass
from .sigmoid_mul_fp8_quant_pass import register_sigmoid_mul_fp8_quant_pass
from .silu_and_mul_fp8_quant_pass import register_silu_and_mul_fp8_quant_pass
from .strided_rmsnorm_fp8_quant_pass import register_strided_rmsnorm_fp8_quant_pass

__all__ = [
    "register_add_rmsnorm_fp8_quant_pass",
    "register_indexer_logits_head_gate_pass",
    "register_moe_shared_expert_sigmoid_gate_add_pass",
    "register_rmsnorm_gated_fp8_quant_pass",
    "register_sigmoid_mul_fp8_quant_pass",
    "register_silu_and_mul_fp8_quant_pass",
    "register_strided_rmsnorm_fp8_quant_pass",
]
