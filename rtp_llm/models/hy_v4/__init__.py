"""HY V4 model registration and compatible public exports."""

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.hy_v4.model import Hy4, Hy4Mtp
from rtp_llm.models.hy_v4.mtp_weight import Hy4MtpWeight
from rtp_llm.models.hy_v4.weight import (
    Hy4Weight,
    _move_indexer_rope_to_front,
    _transpose_stacked_gate_up,
)

__all__ = [
    "Hy4",
    "Hy4Mtp",
    "Hy4Weight",
    "Hy4MtpWeight",
    "_move_indexer_rope_to_front",
    "_transpose_stacked_gate_up",
]

register_model("hy_v4", Hy4, ["HYV4ForCausalLM"])
register_model("hy_v4_mtp", Hy4Mtp, ["HYV4ForCausalLMNextN"])
