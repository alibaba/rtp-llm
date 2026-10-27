from rtp_llm.ops import SpeculativeType

MODEL_BASED_SPECULATIVE_TYPES = (
    SpeculativeType.VANILLA,
    SpeculativeType.MTP,
    SpeculativeType.EAGLE3,
    SpeculativeType.EAGLE,
    SpeculativeType.DSPARK,
)


def resolve_speculative_type(
    sp_type: SpeculativeType, model_type: str
) -> SpeculativeType:
    """Keep frontend cache keys aligned with the backend's draft model mode."""
    if sp_type not in MODEL_BASED_SPECULATIVE_TYPES:
        return sp_type
    if model_type in ("deepseek-v3-mtp", "mixtbstars-mtp"):
        return SpeculativeType.MTP
    if model_type == "qwen_3_moe-mtp":
        return SpeculativeType.EAGLE3
    return sp_type
