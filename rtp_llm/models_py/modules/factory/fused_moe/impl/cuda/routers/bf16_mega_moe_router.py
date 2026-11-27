"""Router for the BF16 MegaMoE executor."""

from typing import Any

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    Fp8Fp4Router,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class Bf16MegaMoeRouter(Fp8Fp4Router):
    """Pass routing tensors through; the BF16 Mega kernel owns dispatch.

    Identical to :class:`MegaMoeRouter` except that it admits the unquantized
    configuration instead of FP8_FP4, and declines fused gate packing because
    that writes the per-token FP8 scale plane the BF16 buffer does not carry.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(MoeConfigResolver().get_quant_method(config) is None)
        checker.check(config.ep_size > 1)

    @property
    def supports_gate_pack(self) -> bool:
        return False
