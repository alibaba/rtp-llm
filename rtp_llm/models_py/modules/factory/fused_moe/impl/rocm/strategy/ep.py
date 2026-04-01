"""ROCm Expert Parallelism strategies"""

import logging
import os
from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy

logger = logging.getLogger(__name__)


class RocmEpNormalStrategy(MoeStrategy):
    """ROCm EP normal mode strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
            FusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsBf16,
        )
        quant_config = FusedMoEQuantConfig(quant_dtype=None)

        # Check if user wants to force use MoriEP
        use_mori_ep = os.environ.get("USE_MORI_EP", "0").lower() in ("1", "true", "on")

        if use_mori_ep:
            logger.info("USE_MORI_EP is set, forcing mori router selection")
            try:
                import mori
                from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.mori_ep_intranode_router import (
                    MoriEpIntranodeRouter,
                )
                logger.info(
                    "ROCm EP strategy selected mori router: %s, executor: %s",
                    MoriEpIntranodeRouter.__name__,
                    RocmExpertsBf16.__name__,
                )
                return StrategyAttributes(
                    router_class=MoriEpIntranodeRouter,
                    executor_class=RocmExpertsBf16,
                    quant_config=quant_config,
                )
            except ImportError as e:
                logger.warning("mori not available even though USE_MORI_EP is set. detail: %s", e)
                raise ValueError("USE_MORI_EP is set but mori is not available")

        logger.info("ROCm EP strategy selection start: try deep_ep first, then mori fallback")
        try:
            import deep_ep

            from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.deepep_normal_router import (
                DeepepNormalRouter,
            )
            logger.info(
                "ROCm EP strategy selected deep_ep router: %s, executor: %s",
                DeepepNormalRouter.__name__,
                FusedMoeExecutor.__name__,
            )
            return StrategyAttributes(
                router_class=DeepepNormalRouter,
                executor_class=FusedMoeExecutor,
                quant_config=quant_config,
            )
        except ImportError as e:
            logger.warning("deep_ep not available, fallback to mori. detail: %s", e)
        try:
            import mori
            from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.mori_ep_intranode_router import (
                MoriEpIntranodeRouter,
            )
            logger.info(
                "ROCm EP strategy selected mori router: %s, executor: %s",
                MoriEpIntranodeRouter.__name__,
                RocmExpertsBf16.__name__,
            )
            return StrategyAttributes(
                router_class=MoriEpIntranodeRouter,
                executor_class=RocmExpertsBf16,
                quant_config=quant_config,
            )
        except ImportError as e:
            logger.warning("mori not available after deep_ep fallback. detail: %s", e)
        raise ValueError("No EP router and executor found, please install deep_ep or mori")


class RocmEpLowLatencyStrategy(MoeStrategy):
    """ROCm EP low latency strategy (not supported)"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """ROCm EP low latency is not supported, always fail."""
        checker.check(False)

    def create_router(self, config: MoEConfigAdapter) -> Any:
        raise ValueError("deepep_low_latency for rocm moe is not yet supported")

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        raise ValueError("deepep_low_latency for rocm moe is not yet supported")

    def get_attributes(self) -> StrategyAttributes:
        # Not actually used, but needed for interface completeness
        # Don't set router_class and executor_class since this strategy is not supported
        # This will raise an error if called, which is expected since the strategy is not supported
        return StrategyAttributes(
            router_class=None,
            executor_class=None,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )
