from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.ops import MoeConfig


class Fp4MoeOp(str, Enum):
    AUTO = "auto"
    TRTLLM = "trtllm"
    CUTEDSL = "cutedsl"
    B12X = "b12x"


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
