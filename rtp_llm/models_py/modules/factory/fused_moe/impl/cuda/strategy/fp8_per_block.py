"""CUDA FP8 PerBlock quantization strategies"""

import os
from typing import Any

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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class CudaFp8PerBlockNoDPStrategy(MoeStrategy):
    """CUDA FP8 PerBlock single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(
            config.moe_strategy == "fp8_per_block_no_dp"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
            DeepGemmHybridExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlock,
            executor_class=DeepGemmHybridExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockNoDPMaskedStrategy(MoeStrategy):
    """CUDA FP8 PerBlock No DP Masked strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(config.moe_strategy == "fp8_per_block_no_dp_masked")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor_v2 import (
            DeepGemmMaskedExecutorV2,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlock,
            executor_class=DeepGemmMaskedExecutorV2,
            quant_config=quant_config,
        )


class CudaFp8PerBlockPureDPStrategy(MoeStrategy):
    """CUDA FP8 PerBlock pure DP+EP strategy using allgather+reduce_scatter.

    Pure DP attention + EP MoE only: tp_size == 1, dp_size > 1, ep_size == dp_size.
    Mixed tp>1+dp>1 deliberately falls through to DeepEP strategies.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        is_pure_dp_ep = (
            config.parallelism_config.tp_size == 1
            and config.dp_size > 1
            and config.ep_size == config.dp_size
        )
        checker.check(config.moe_strategy == "fp8_per_block_pure_dp")
        checker.check(is_pure_dp_ep)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor_v2 import (
            DeepGemmMaskedExecutorV2,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_dp_router import (
            PureDpRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureDpRouterFp8PerBlock,
            executor_class=DeepGemmMaskedExecutorV2,
            quant_config=quant_config,
        )


class CudaFp8PerBlockPureCPStrategy(MoeStrategy):
    """CUDA FP8 PerBlock pure CP+EP strategy using allgather+reduce_scatter.

    Pure CP attention + EP MoE only: tp_size > 1, dp_size == 1,
    ep_size == tp_size, prefill CP enabled. No padding needed since CP
    splits context evenly across ranks.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        is_pure_cp_ep = (
            config.dp_size == 1
            and resolver.is_cp_equal_ep(config)
            and config.ep_size > 1
            and config.parallelism_config.prefill_cp_config.is_enabled()
        )
        checker.check(config.moe_strategy == "fp8_per_block_pure_cp")
        checker.check(is_pure_cp_ep)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
            DeepGemmHybridExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router import (
            PureCpRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureCpRouterFp8PerBlock,
            executor_class=DeepGemmHybridExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(
            config.moe_strategy == "fp8_per_block_ep_low_latency"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=DeepGemmMaskedExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(
            config.moe_strategy == "fp8_per_block_ep_normal"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
            DeepGemmHybridExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8PerBlock,
            executor_class=DeepGemmHybridExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpElasticContiguousStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP elastic 2D Contiguous strategy.

    Selected when ``USE_DEEPEP_ELASTIC=1`` with the default
    ``DEEPEP_ELASTIC_DO_EXPAND=1, DEEPEP_ELASTIC_DO_CPU_SYNC=1`` —
    pairs the elastic router (tight ``[ΣN_e, hidden]`` layout) with
    ``DeepGemmHybridExecutor``.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        do_expand = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_EXPAND", "1")))
        do_cpu_sync = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_CPU_SYNC", "1")))
        checker.check(do_cpu_sync)
        checker.check(
            config.moe_strategy == "fp8_per_block_ep_elastic_contiguous"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
            DeepGemmHybridExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_elastic_router import (
            DeepEpElasticRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepEpElasticRouter,
            executor_class=DeepGemmHybridExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpElasticDecodeStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP elastic decode cudagraph strategy.

    Selected when ``USE_DEEPEP_ELASTIC=1`` with
    ``DEEPEP_ELASTIC_DO_EXPAND=0, DEEPEP_ELASTIC_DO_CPU_SYNC=0`` — pairs
    the elastic router decode-mode layout (``[worst_case_N, hidden]`` +
    per-row ``recv_topk_idx`` with ``-1`` sentinels) with
    ``DeepGemmMaskedExecutorV2``.

    ``DeepGemmMaskedExecutorV2.execute()`` directly returns
    ``execute_masked(...)``, bypassing the Hybrid ``execute_contiguous``
    path (which has D2H sanity asserts at L400-401 that would break CUDA
    Graph capture).  The underlying ``ep_scatter_v2`` /  ``ep_gather``
    Triton kernels already guard ``expert_id >= 0`` per row, so the
    ``-1`` sentinels are skipped without any host-side mask reorg.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        do_expand = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_EXPAND", "1")))
        do_cpu_sync = bool(int(os.environ.get("DEEPEP_ELASTIC_DO_CPU_SYNC", "1")))
        checker.check((not do_expand) and (not do_cpu_sync))
        checker.check(
            config.moe_strategy == "fp8_per_block_ep_elastic_decode"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor_v2 import (
            DeepGemmMaskedExecutorV2,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_elastic_router import (
            DeepEpElasticRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepEpElasticRouter,
            executor_class=DeepGemmMaskedExecutorV2,
            quant_config=quant_config,
        )
