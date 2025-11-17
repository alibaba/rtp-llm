"""Strategy priority attributes

Defines attributes for automatic priority calculation based on
Router and Executor implementation characteristics.
"""

from typing import Any, Optional, Type

from rtp_llm.models_py.modules.common.moe.fused_moe import (
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)


class StrategyAttributes:
    """Strategy attributes for priority calculation

    Strategies define their Router and Executor types, and the priority
    is calculated automatically based on their performance characteristics.

    Formula: priority = router_type.value * 10 + executor_type.value

    This ensures:
    - Better router implementation gets higher priority
    - Better executor implementation gets higher priority within same router level
    - Priority range: 0-49 (5 router types × 10 + 5 executor types)

    Examples:
    - BATCHED_DATA + BATCHED_TRITON = 0*10 + 0 = 0 (lowest)
    - DEEPEP_LOW_LATENCY + CUTLASS_BATCHED_FP8 = 4*10 + 4 = 44 (highest)
    - DEEPEP_NORMAL + CUTLASS_FP8 = 3*10 + 3 = 33 (mid-high)
    """

    def __init__(
        self,
        router_class: Type[FusedMoeDataRouter],
        executor_class: Type[FusedMoeExpertExecutor],
    ):
        """Initialize strategy attributes

        Args:
            router_type: Router implementation type (optional, will be obtained from router_class if not provided)
            executor_type: Executor implementation type (optional, will be obtained from executor_class if not provided)
            router_class: Actual router class (required if router_type is not provided)
            executor_class: Actual executor class (required if executor_type is not provided)
        """
        self.router_class = router_class
        self.executor_class = executor_class

    def calculate_priority(self) -> int:
        """Calculate priority based on Router and Executor types

        Returns:
            Calculated priority value
        """
        return (
            self.router_class.router_type().value * 10
            + self.executor_class.executor_type().value
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
