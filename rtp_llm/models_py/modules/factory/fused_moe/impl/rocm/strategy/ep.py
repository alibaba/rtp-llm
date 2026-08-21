"""ROCm Expert Parallelism strategies"""

import logging
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

    _SUPPORTED_QUANT_METHODS = {
        None,
        "FP8_PER_CHANNEL_COMPRESSED",
        "FP8_PER_CHANNEL_QUARK",
        "FP8_PER_BLOCK",
        "modelopt_fp4",
    }

    @staticmethod
    def _get_quant_method(config: MoEConfigAdapter) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        return MoeConfigResolver().get_quant_method(config)

    def can_handle(self, config: MoEConfigAdapter) -> bool:
        """Filter unsupported quant methods before resolving EP dependencies."""
        self._config = config
        if self._get_quant_method(config) not in self._SUPPORTED_QUANT_METHODS:
            return False
        return super().can_handle(config)

    def _resolve_executor_and_quant(
        self,
    ) -> tuple[Any, FusedMoEQuantConfig]:
        """Select executor class and quant config based on quant_method."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
            get_rocm_fp8_dtype,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsBf16,
            RocmExpertsFp4PerGroup,
            RocmExpertsFp8PerBlock,
            RocmExpertsFp8PerChannel,
        )
        config = getattr(self, "_config", None)
        quant_method = self._get_quant_method(config) if config else None

        if quant_method == "modelopt_fp4":
            executor_class = RocmExpertsFp4PerGroup
            quant_config = FusedMoEQuantConfig(
                quant_dtype=torch.float4_e2m1fn_x2,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=None,
            )
        elif quant_method in (
            "FP8_PER_CHANNEL_COMPRESSED",
            "FP8_PER_CHANNEL_QUARK",
        ):
            executor_class = RocmExpertsFp8PerChannel
            quant_config = FusedMoEQuantConfig(
                quant_dtype=get_rocm_fp8_dtype(),
                per_act_token_quant=True,
                per_out_ch_quant=True,
                block_shape=None,
            )
        elif quant_method == "FP8_PER_BLOCK":
            executor_class = RocmExpertsFp8PerBlock
            quant_config = FusedMoEQuantConfig(
                quant_dtype=get_rocm_fp8_dtype(),
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[128, 128],
            )
        elif quant_method is None:
            executor_class = RocmExpertsBf16
            quant_config = FusedMoEQuantConfig(quant_dtype=None)
        else:
            raise ValueError(
                f"ROCm EP does not support quantization method {quant_method}"
            )

        return executor_class, quant_config

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
            FusedMoeExecutor,
        )

        config = getattr(self, "_config", None)

        # Mutual exclusion: cannot use both DeepEP and MoriEP routers
        if config and config.use_deepep_moe and config.use_mori_ep:
            raise ValueError(
                "use_deepep_moe and use_mori_ep are mutually exclusive; "
                "enable only one EP router"
            )

        executor_class, quant_config = self._resolve_executor_and_quant()

        if config and config.use_mori_ep:
            logger.info("use_mori_ep is set, forcing mori router selection")
            try:
                import mori

                from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.mori_ep_intranode_router import (
                    MoriEpIntranodeRouter,
                )

                logger.info(
                    "ROCm EP strategy selected mori router: %s, executor: %s",
                    MoriEpIntranodeRouter.__name__,
                    executor_class.__name__,
                )
                return StrategyAttributes(
                    router_class=MoriEpIntranodeRouter,
                    executor_class=executor_class,
                    quant_config=quant_config,
                )
            except ImportError as e:
                logger.warning(
                    "mori not available even though use_mori_ep is set. detail: %s", e
                )
                raise ValueError("use_mori_ep is set but mori is not available")

        logger.info("ROCm EP strategy selection: try deep_ep")
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
            logger.warning("deep_ep not available. detail: %s", e)
            raise ImportError(
                "No EP router and executor found, please install deep_ep or enable use_mori_ep"
            )


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
