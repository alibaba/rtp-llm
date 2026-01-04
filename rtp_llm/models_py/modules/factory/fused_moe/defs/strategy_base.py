"""MOE strategy base class

Defines the unified interface for all MOE strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)

from ..utils.condition_checker import ConditionChecker
from .priority_attributes import StrategyAttributes


class MoeStrategy(ABC):
    """MOE strategy base class

    Each strategy is responsible for determining whether it can handle a specific
    configuration and creating the corresponding Router and Executor.

    Priority is automatically calculated based on Router and Executor types
    rather than manually set numbers. Subclasses should override `get_attributes()`
    to define which Router and Executor implementations they use.
    """

    def can_handle(self, config: MoEConfigAdapter) -> bool:
        """Determine whether this strategy can handle the given configuration

        This method creates a checker and calls the check_conditions methods
        of the Router and Executor classes to determine if they can handle
        the given configuration.

        Args:
            config: MOE configuration adapter

        Returns:
            Whether this configuration can be handled
        """
        checker = ConditionChecker(f"{self.__class__.__name__}.can_handle()")

        # Get Router and Executor types from strategy attributes
        attrs = self.get_attributes()
        router_cls = attrs.get_router_class()
        executor_cls = attrs.get_executor_class()

        # for CudaNoQuantEpLowLatencyStrategy/CudaFp8PerBlockEpLowLatencyStrategy has same router and executor,
        # so we need to check Strategy conditions here (like quant_method)
        self.check_conditions(checker, config)

        # Call check_conditions on Router and Executor classes
        if router_cls:
            router_cls.check_conditions(checker, config)
        if executor_cls:
            executor_cls.check_conditions(checker, config)

        return checker.all_passed()

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        pass

    def create_router(self, config: MoEConfigAdapter) -> FusedMoeDataRouter:
        """Create Router

        Args:
            config: MOE configuration adapter

        Returns:
            Router instance
        """
        attrs: StrategyAttributes = self.get_attributes()
        router_cls: Optional[Type[FusedMoeDataRouter]] = attrs.get_router_class()
        quant_config: FusedMoEQuantConfig = attrs.quant_config

        if router_cls is None:
            raise ValueError("Router class is not defined in strategy attributes")

        return router_cls(config, quant_config)  # type: ignore[call-arg]

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> FusedMoeExpertExecutor:
        """Create Executor

        Args:
            config: MOE configuration adapter
            weights: Weight dictionary

        Returns:
            Executor instance
        """
        attrs: StrategyAttributes = self.get_attributes()
        executor_cls: Optional[Type[FusedMoeExpertExecutor]] = (
            attrs.get_executor_class()
        )
        quant_config: FusedMoEQuantConfig = attrs.quant_config

        if executor_cls is None:
            raise ValueError("Executor class is not defined in strategy attributes")

        return executor_cls(config, quant_config, weights)  # type: ignore[call-arg]

    @abstractmethod
    def get_attributes(self) -> StrategyAttributes:
        """Get strategy attributes for priority calculation

        Subclasses should return a StrategyAttributes instance describing
        the Router and Executor types this strategy uses.

        Returns:
            Strategy attributes
        """
        pass

    @property
    def priority(self) -> int:
        """Strategy priority (automatically calculated from Router and Executor types)

        When multiple strategies can handle the same configuration, the strategy
        with higher priority will be selected. Higher numeric value means higher priority.

        Priority is calculated based on:
        - Router type (communication strategy performance)
        - Executor type (computation strategy performance)

        Returns:
            Calculated priority value
        """
        return self.get_attributes().calculate_priority()

    def __repr__(self) -> str:
        """Return string representation of the strategy"""
        return f"{self.__class__.__name__}({self.get_attributes()})"
