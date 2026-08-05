import math
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.ops import MoeConfig

B12X_ZEROED_ENERGY_LIMIT_DEFAULT = 0.001
B12X_ZEROED_ENERGY_LIMIT_ENV = "RTP_LLM_B12X_ZEROED_ENERGY_LIMIT"
B12X_DISABLE_CUDA12_9_COMPAT_ENV = "RTP_LLM_DISABLE_B12X_CUDA12_9_COMPAT"


class MoeStrategyName(str, Enum):
    AUTO = "auto"
    NO_QUANT_EP_LOW_LATENCY = "no_auant_ep_low_latency"
    NO_QUANT_CPP = "no_auant_cpp"
    NO_QUANT_DP_NORMAL = "no_auant_dp_normal"
    FP8_PER_BLOCK_NO_DP_MASKED = "fp8_per_block_no_dp_masked"
    FP8_PER_BLOCK_NO_DP = "fp8_per_block_no_dp"
    FP8_PER_BLOCK_EP_LOW_LATENCY = "fp8_per_block_ep_low_latency"
    FP8_PER_BLOCK_EP_NORMAL = "fp8_per_block_ep_normal"
    FP8_PER_BLOCK_PURE_CP = "fp8_per_block_pure_cp"
    FP8_PER_BLOCK_PURE_DP = "fp8_per_block_pure_dp"
    FP8_PER_TENSOR_NO_DP = "fp8_per_tensor_no_dp"
    FP8_PER_TENSOR_EP_LOW_LATENCY = "fp8_per_tensor_ep_low_latency"
    FP8_PER_TENSOR_EP_NORMAL = "fp8_per_tensor_ep_normal"
    W4A8_INT4_PER_CHANNEL_NO_DP = "w4a8_int4_per_channel_no_dp"
    W4A8_INT4_PER_CHANNEL_EP_LOW_LATENCY = "w4a8_int4_per_channel_ep_low_latency"
    W4A8_INT4_PER_CHANNEL_EP_NORMAL = "w4a8_int4_per_channel_ep_normal"
    FP4_EP_LOW_LATENCY = "fp4_ep_low_latency"
    FP4_EP_NORMAL = "fp4_ep_normal"
    FP4_NO_DP = "fp4_no_dp"
    FP4_B12X = "fp4_b12x"


class Fp4MoeOp(str, Enum):
    AUTO = "auto"
    TRTLLM = "trtllm"
    CUTEDSL = "cutedsl"
    B12X = "b12x"


def validate_b12x_zeroed_energy_limit(limit: float) -> float:
    if not math.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError(
            "b12x_zeroed_energy_limit must be a finite float in [0, 1], "
            f"got {limit!r}"
        )
    return limit


def resolve_fp4_moe_op(moe_config: "MoeConfig", *, is_sm12x: bool) -> str:
    """Resolve fp4_moe_op="auto" from config and the caller's architecture."""
    try:
        fp4_moe_op = Fp4MoeOp(moe_config.fp4_moe_op)
    except ValueError as error:
        allowed = ", ".join(op.value for op in Fp4MoeOp)
        raise ValueError(
            f"invalid fp4_moe_op {moe_config.fp4_moe_op!r}; expected one of: "
            f"{allowed}"
        ) from error

    if fp4_moe_op is not Fp4MoeOp.AUTO:
        return fp4_moe_op.value
    if moe_config.use_deepep_moe and moe_config.use_deepep_low_latency:
        return Fp4MoeOp.CUTEDSL.value
    if is_sm12x:
        return Fp4MoeOp.B12X.value
    return Fp4MoeOp.TRTLLM.value
