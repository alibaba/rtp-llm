"""Routers for fused FP8-activation/FP8-weight (MXFP8) MoE executors."""

from typing import Any

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)

from .fp8_fp4_router import Fp8Fp4Router


class Fp8Fp8Router(Fp8Fp4Router):
    """Pass routing tensors to an MXFP8 executor that dispatches locally.

    ``prepare``/``finalize`` are quantization-agnostic, so only the accepted
    quant method differs from the FP8/FP4 router.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_quant_method == "MXFP8")


class MegaMoeFp8Router(Fp8Fp8Router):
    """Router marker for MXFP8 kernels that fuse EP dispatch/combine.

    Unlike the FP4 ``MegaMoeRouter`` this does not advertise
    ``supports_gate_pack``: ``fp8_fp8_mega_moe`` has no fused gate-packing
    entry point, so routing stays materialized in the layer.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        super().check_conditions(checker, config)
        checker.check(config.ep_size > 1)
