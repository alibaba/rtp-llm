"""Strategy priority attributes

Defines attributes for automatic priority calculation based on
Router and Executor implementation characteristics.
"""

from typing import Any, Optional, Type

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)

# Place-value base of the priority encoding. Must stay strictly greater than
# every ExecutorType value, otherwise two (router, executor) pairs can encode
# to the same priority (this actually happened when ExecutorType reached 10
# with a base of 10: PURE_TP(5)*10 + B12X_FP4(10) == MORI_EP_INTRANODE(6)*10
# + BATCHED_TRITON(0)).
EXECUTOR_PRIORITY_BASE = 100


def calculate_strategy_priority(
    router_type: RouterType, executor_type: ExecutorType
) -> int:
    """Encode (router, executor) into a single comparable priority.

    Router dominates; executor breaks ties within the same router level.
    """
    if executor_type.value >= EXECUTOR_PRIORITY_BASE:
        raise ValueError(
            f"ExecutorType.{executor_type.name}={executor_type.value} no longer "
            f"fits the priority encoding base {EXECUTOR_PRIORITY_BASE}; raise "
            "EXECUTOR_PRIORITY_BASE to keep priorities collision-free"
        )
    return router_type.value * EXECUTOR_PRIORITY_BASE + executor_type.value


class StrategyAttributes:
    """Strategy attributes for priority calculation

    Strategies define their Router and Executor types, and the priority
    is calculated automatically based on their performance characteristics.

    Formula: priority = router_type.value * EXECUTOR_PRIORITY_BASE
                        + executor_type.value

    This ensures:
    - Better router implementation gets higher priority
    - Better executor implementation gets higher priority within same router level

    Examples:
    - BATCHED_DATA + BATCHED_TRITON = 0*100 + 0 = 0 (lowest)
    - DEEPEP_LOW_LATENCY + CUTLASS_BATCHED_FP8 = 4*100 + 4 = 404
    - PURE_TP + B12X_FP4 = 5*100 + 10 = 510
    """

    def __init__(
        self,
        router_class: Type[FusedMoeDataRouter],
        executor_class: Type[FusedMoeExpertExecutor],
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize strategy attributes

        Args:
            router_class: Actual router class
            executor_class: Actual executor class
            quant_config: Quantization configuration
        """
        self.router_class: Type[FusedMoeDataRouter] = router_class
        self.executor_class: Type[FusedMoeExpertExecutor] = executor_class
        self.quant_config: FusedMoEQuantConfig = quant_config

    def calculate_priority(self) -> int:
        """Calculate priority based on Router and Executor types

        Returns:
            Calculated priority value
        """
        return calculate_strategy_priority(
            self.router_class.router_type(),
            self.executor_class.executor_type(),
        )

    def get_router_class(self) -> Optional[Any]:
        """Get the router class for condition checking

        Returns:
            Router class if specified, None otherwise
        """
        return self.router_class

    def get_executor_class(self) -> Optional[Any]:
        """Get the executor class for condition checking

        Returns:
            Executor class if specified, None otherwise
        """
        return self.executor_class

    def __repr__(self) -> str:
        """Return string representation"""
        return (
            f"StrategyAttributes("
            f"router={self.router_class}, "
            f"executor={self.executor_class}, "
            f"priority={self.calculate_priority()})"
        )
