"""GLM5 policy adapter for the shared MegaMoE buffer implementation."""

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _get_or_create_mega_buf,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _get_or_create_mega_output as get_or_create_mega_output,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _mega_moe_available as mega_moe_available,
)


def get_or_create_mega_buf(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
):
    return _get_or_create_mega_buf(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch,
        activation,
    )


__all__ = [
    "get_or_create_mega_buf",
    "get_or_create_mega_output",
    "mega_moe_available",
]
