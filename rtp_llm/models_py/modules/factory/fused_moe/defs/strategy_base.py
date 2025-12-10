"""MOE strategy base class

Defines the unified interface for all MOE strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)

from ..utils.condition_checker import ConditionChecker
from .priority_attributes import StrategyAttributes


class PriorityCallable:
    """A callable that can be used both as a value and as a function"""

    def __init__(
        self, method: Callable[[bool], int], default_use_cuda_graph: bool = False
    ):
        self._method = method
        self._default = default_use_cuda_graph
        self._default_value = method(default_use_cuda_graph)

    def __call__(self, use_cuda_graph: Optional[bool] = None) -> int:
        if use_cuda_graph is None:
            return self._default_value
        return self._method(use_cuda_graph)

    def __int__(self) -> int:
        return self._default_value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self._default_value == other
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self._default_value < other
        if isinstance(other, PriorityCallable):
            return self._default_value < other._default_value
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self._default_value > other
        if isinstance(other, PriorityCallable):
            return self._default_value > other._default_value
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self._default_value <= other
        if isinstance(other, PriorityCallable):
            return self._default_value <= other._default_value
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self._default_value >= other
        if isinstance(other, PriorityCallable):
            return self._default_value >= other._default_value
        return NotImplemented

    def __str__(self) -> str:
        return str(self._default_value)

    def __repr__(self) -> str:
        return str(self._default_value)


class MoeStrategy(ABC):
    """MOE strategy base class

    Each strategy is responsible for determining whether it can handle a specific
    configuration and creating the corresponding Router and Executor.

    Priority is automatically calculated based on Router and Executor types
    rather than manually set numbers. Subclasses should override `get_attributes()`
    to define which Router and Executor implementations they use.
    """

    def can_handle(self, config: GptInitModelParameters) -> bool:
        """Determine whether this strategy can handle the given configuration

        This method creates a checker and calls the check_conditions methods
        of the Router and Executor classes to determine if they can handle
        the given configuration.

        Args:
            config: Model initialization parameters

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
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        pass

    @abstractmethod
    def create_router(self, config: GptInitModelParameters) -> FusedMoeDataRouter:
        """Create Router

        Args:
            config: Model initialization parameters

        Returns:
            Router instance
        """
        pass

    @abstractmethod
    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> FusedMoeExpertExecutor:
        """Create Executor

        Args:
            config: Model initialization parameters
            weights: Weight dictionary

        Returns:
            Executor instance
        """
        pass

    @abstractmethod
    def get_attributes(self) -> StrategyAttributes:
        """Get strategy attributes for priority calculation

        Subclasses should return a StrategyAttributes instance describing
        the Router and Executor types this strategy uses.

        Returns:
            Strategy attributes
        """
        pass

    def _priority_impl(self, use_cuda_graph: bool = False) -> int:
        """Strategy priority implementation (automatically calculated from Router and Executor types)

        When multiple strategies can handle the same configuration, the strategy
        with higher priority will be selected. Higher numeric value means higher priority.

        Priority is calculated based on:
        - Router type (communication strategy performance)
        - Executor type (computation strategy performance)

        Args:
            use_cuda_graph: Whether CUDA graph is enabled (default: False)

        Returns:
            Calculated priority value
        """
        return self.get_attributes().calculate_priority()

    @property
    def priority(self) -> PriorityCallable:
        """Strategy priority (automatically calculated from Router and Executor types)

        Can be accessed directly: .priority (returns int with default use_cuda_graph=False)
        Or called with parameter: .priority(use_cuda_graph=True) (returns int with provided value)

        Examples:
            >>> strategy.priority  # Returns PriorityCallable, can be used as int
            >>> strategy.priority == 10  # Comparison works
            >>> strategy.priority()  # Returns int with default use_cuda_graph=False
            >>> strategy.priority(True)  # Returns int with use_cuda_graph=True
        """
        return PriorityCallable(self._priority_impl, default_use_cuda_graph=False)

    def __repr__(self) -> str:
        """Return string representation of the strategy"""
        return f"{self.__class__.__name__}({self.get_attributes()})"
