# EP MoE module with router+executor pattern

from rtp_llm.models_py.modules.ep.ep_moe import (
    EPDataRouter,
    EPExpertExecutor,
    EPMoE,
    create_ep_moe_instance,
)

# Legacy implementations (for backward compatibility)
from rtp_llm.models_py.modules.ep.layers import GroupedGemmRunner, LegacyEPMoE

__all__ = [
    # New router+executor pattern
    "EPMoE",
    "EPDataRouter",
    "EPExpertExecutor",
    "create_ep_moe_instance",
    # Legacy implementations
    "LegacyEPMoE",
    "GroupedGemmRunner",
]
