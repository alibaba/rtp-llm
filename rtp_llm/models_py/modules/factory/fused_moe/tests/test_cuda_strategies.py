"""CUDA strategy tests"""

import contextlib
import unittest
from typing import Any, Iterator, Optional
from unittest.mock import patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import (
    CompressedW8A8Int8PerChannelQuantConfig,
    Fp8BlockWiseQuantConfig,
    Fp8DynamicPerTensorQuantConfig,
    Fp8PerTensorCompressedQuantConfig,
    MXFp4QuarkQuantConfig,
    ModelOptFp4Config,
    W4a8Int4PerChannelQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    EXECUTOR_PRIORITY_BASE,
    calculate_strategy_priority,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.strategy.batched_triton_strategy import (
    BatchedTritonStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
    CudaFp8PerBlockEpNormalStrategy,
    CudaFp8PerBlockNoDPMaskedStrategy,
    CudaFp8PerBlockNoDPStrategy,
    CudaFp8PerBlockPureCPStrategy,
    CudaFp8PerBlockPureDPStrategy,
    CudaFp8PerTensorNoDPStrategy,
    CudaNoQuantCppStrategy,
    CudaNoQuantDpNormalStrategy,
    CudaW4a8Int4PerChannelNoDPStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.strategy.ep import (
    RocmEpNormalStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.condition_checker import (
    ConditionChecker,
)
from rtp_llm.ops import CPRotateMethod, MoeConfig, ParallelismConfig

# Helper functions for creating configuration objects


def create_model_config_without_quant() -> ModelConfig:
    """Create ModelConfig without quantization"""
    model_config = ModelConfig()
    model_config.quant_config = None
    return model_config


def create_model_config_with_fp8_block_quant(
    dtype: Optional[str] = None,
) -> ModelConfig:
    """Create ModelConfig with FP8 block-wise quantization"""
    model_config = ModelConfig()
    model_config.quant_config = Fp8BlockWiseQuantConfig()
    model_config.data_type = dtype if dtype is not None else "bf16"
    return model_config


def create_model_config_with_fp8_dynamic_per_tensor_quant() -> ModelConfig:
    """Create ModelConfig with dynamic FP8 per-tensor quantization"""
    model_config = ModelConfig()
    model_config.quant_config = Fp8DynamicPerTensorQuantConfig()
    return model_config


def create_model_config_with_fp8_per_tensor_compressed_quant() -> ModelConfig:
    """Create ModelConfig with compressed-tensors FP8 quantization"""
    model_config = ModelConfig()
    model_config.quant_config = Fp8PerTensorCompressedQuantConfig()
    return model_config


def create_model_config_with_w4a8_int4_per_channel_quant() -> ModelConfig:
    """Create ModelConfig with W4A8 INT4 per-channel quantization"""
    model_config = ModelConfig()
    model_config.quant_config = W4a8Int4PerChannelQuantConfig()
    return model_config


def create_model_config_with_w8a8_int8_per_channel_quant() -> ModelConfig:
    """Create ModelConfig with compressed-tensors W8A8 INT8 quantization."""
    model_config = ModelConfig()
    model_config.quant_config = CompressedW8A8Int8PerChannelQuantConfig()
    return model_config


def create_model_config_with_fp4_quant() -> ModelConfig:
    """Create ModelConfig with ModelOpt NVFP4 quantization"""
    model_config = ModelConfig()
    model_config.quant_config = ModelOptFp4Config(
        bits=4, group_size=16, is_quanted=True
    )
    model_config.data_type = "bf16"
    return model_config


def create_parallelism_config(
    ep_size: int = 1,
    tp_size: int = 1,
    dp_size: int = 1,
    enable_cp: bool = False,
) -> ParallelismConfig:
    """Create ParallelismConfig with specified parallelism settings

    Args:
        ep_size: Expert parallelism size
        tp_size: Physical tensor parallelism size (raw parallelism_config.tp_size).
            When enable_cp=True this is the CP size; the adapter's tp_size view
            (get_attn_tp_size()) will be 1.
        dp_size: Data parallelism size
        enable_cp: If True, enable prefill CP (ALL_GATHER). This makes
            get_attn_tp_size() return 1 while parallelism_config.tp_size keeps
            the physical value — which is the configuration that PureCP
            strategies expect.
    """
    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = ep_size
    parallelism_config.tp_size = tp_size
    parallelism_config.dp_size = dp_size
    if enable_cp:
        parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
    return parallelism_config


def create_moe_config(
    use_deepep_low_latency: bool = False,
    use_all_gather: Optional[bool] = None,
    moe_strategy: Optional[str] = None,
) -> MoeConfig:
    """Create MoeConfig with specified settings

    Args:
        use_deepep_low_latency: Whether to use DeepEP low latency mode
        use_all_gather: Whether to use all_gather (None means not set)
    """
    moe_config = MoeConfig()
    if use_deepep_low_latency is not None:
        moe_config.use_deepep_low_latency = use_deepep_low_latency
    if use_all_gather is not None:
        moe_config.use_all_gather = use_all_gather
    if moe_strategy is not None:
        moe_config.moe_strategy = moe_strategy
    return moe_config


def create_moe_config_adapter(
    model_config: ModelConfig,
    parallelism_config: ParallelismConfig,
    moe_config: MoeConfig,
    max_generate_batch_size: int = 128,
    enable_cuda_graph: bool = False,
) -> MoEConfigAdapter:
    """Create MoEConfigAdapter with specified configurations

    Args:
        model_config: Model configuration
        parallelism_config: Parallelism configuration
        moe_config: MoE configuration
        max_generate_batch_size: Maximum generate batch size
        enable_cuda_graph: Whether to enable CUDA graph
    """
    moe_config.ll_num_max_token = max_generate_batch_size
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        enable_cuda_graph=enable_cuda_graph,
    )


class TestCudaNoQuantFallbackStrategies(unittest.TestCase):
    """No-quant strategy conditions must reject quantized checkpoints."""

    def _conditions_pass(
        self, strategy: type, model_config: ModelConfig, moe_strategy: str
    ) -> bool:
        config = create_moe_config_adapter(
            model_config=model_config,
            parallelism_config=create_parallelism_config(),
            moe_config=create_moe_config(moe_strategy=moe_strategy),
        )
        checker = ConditionChecker(f"{strategy.__name__}.check_conditions()")
        strategy.check_conditions(checker, config)
        return checker.all_passed()

    def test_cpp_accepts_unquantized_checkpoint(self) -> None:
        self.assertTrue(
            self._conditions_pass(
                CudaNoQuantCppStrategy,
                create_model_config_without_quant(),
                "no_auant_cpp",
            )
        )

    def test_cpp_rejects_quantized_checkpoint(self) -> None:
        self.assertFalse(
            self._conditions_pass(
                CudaNoQuantCppStrategy,
                create_model_config_with_w8a8_int8_per_channel_quant(),
                "no_auant_cpp",
            )
        )

    def test_dp_normal_accepts_unquantized_checkpoint(self) -> None:
        self.assertTrue(
            self._conditions_pass(
                CudaNoQuantDpNormalStrategy,
                create_model_config_without_quant(),
                "no_auant_dp_normal",
            )
        )

    def test_dp_normal_rejects_quantized_checkpoint(self) -> None:
        self.assertFalse(
            self._conditions_pass(
                CudaNoQuantDpNormalStrategy,
                create_model_config_with_w8a8_int8_per_channel_quant(),
                "no_auant_dp_normal",
            )
        )


class _Sm9xStrategyTestCase(unittest.TestCase):
    """Patch the architecture probes reached by the SM9x strategy tests."""

    def setUp(self) -> None:
        super().setUp()
        arch_stack = contextlib.ExitStack()
        self.addCleanup(arch_stack.close)
        arch_stack.enter_context(
            patch("rtp_llm.models_py.utils.arch.is_sm12x", return_value=False)
        )
        arch_stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor.get_sm",
                return_value=(9, 0),
            )
        )
        arch_stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor_v2.get_sm",
                return_value=(9, 0),
            )
        )


class TestRocmEpStrategyQuantFiltering(unittest.TestCase):
    """Unsupported quant methods must leave ROCm EP candidate probing cleanly."""

    def test_unsupported_quant_methods_return_false(self) -> None:
        for quant_config in (
            CompressedW8A8Int8PerChannelQuantConfig(),
            MXFp4QuarkQuantConfig(),
        ):
            with self.subTest(quant_method=quant_config.get_method()):
                model_config = ModelConfig()
                model_config.quant_config = quant_config
                config = create_moe_config_adapter(
                    model_config=model_config,
                    parallelism_config=create_parallelism_config(),
                    moe_config=create_moe_config(),
                )
                self.assertFalse(RocmEpNormalStrategy().can_handle(config))


class TestCudaNoQuantSingleGpuStrategy(unittest.TestCase):
    """Test CUDA single GPU without quantization strategy"""

    def test_can_handle_true(self) -> None:
        """Test case that can be handled"""
        config = create_moe_config_adapter(
            model_config=create_model_config_without_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(),
        )

        strategy = BatchedTritonStrategy()
        self.assertTrue(strategy.can_handle(config))

    def test_can_handle_false_has_quant(self) -> None:
        """Test case with quantization"""
        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_dynamic_per_tensor_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(),
        )

        strategy = BatchedTritonStrategy()
        self.assertFalse(strategy.can_handle(config))

    def test_can_handle_true_not_single_gpu(self) -> None:
        """Test multi-GPU case with TP==EP"""
        config = create_moe_config_adapter(
            model_config=create_model_config_without_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=2, dp_size=1
            ),
            moe_config=create_moe_config(),
        )

        strategy = BatchedTritonStrategy()
        self.assertTrue(strategy.can_handle(config))


class TestCudaFp8PerBlockNoDPStrategy(_Sm9xStrategyTestCase):
    """Test CUDA FP8 PerBlock single GPU strategy"""

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_single_gpu(self, mock_has_deep_gemm: Any) -> None:
        """Test single GPU case"""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_all_gather=True),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockNoDPStrategy()
        self.assertTrue(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_tp_equal_ep(self, mock_has_deep_gemm: Any) -> None:
        """Test TP equals EP case"""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=2, dp_size=1
            ),
            moe_config=create_moe_config(use_all_gather=True),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockNoDPStrategy()
        self.assertTrue(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_cuda_graph(self, mock_has_deep_gemm: Any) -> None:
        """Test case when CUDA graph is enabled (should fail)"""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_all_gather=True),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockNoDPStrategy()
        self.assertTrue(strategy.can_handle(config))
        config.enable_cuda_graph = True
        self.assertFalse(strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        strategy = CudaFp8PerBlockNoDPStrategy()
        router_type = RouterType.PURE_TP
        executor_type = ExecutorType.DEEPGEMM_CONTINUOUS
        expected_priority = calculate_strategy_priority(router_type, executor_type)

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerBlockNoDPMaskedStrategy(_Sm9xStrategyTestCase):
    """Test CUDA FP8 PerBlock No DP Masked strategy"""

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_single_gpu(self, mock_has_deep_gemm: Any) -> None:
        """Test single GPU case"""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(
                use_all_gather=True, moe_strategy="fp8_per_block_no_dp_masked"
            ),
        )

        strategy = CudaFp8PerBlockNoDPMaskedStrategy()
        self.assertTrue(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_tp_equal_ep(self, mock_has_deep_gemm: Any) -> None:
        """Test TP equals EP case"""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=2, dp_size=1
            ),
            moe_config=create_moe_config(
                use_all_gather=True, moe_strategy="fp8_per_block_no_dp_masked"
            ),
        )

        strategy = CudaFp8PerBlockNoDPMaskedStrategy()
        self.assertTrue(strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        strategy = CudaFp8PerBlockNoDPMaskedStrategy()
        router_type = RouterType.PURE_TP
        executor_type = ExecutorType.DEEPGEMM_MASKED
        expected_priority = calculate_strategy_priority(router_type, executor_type)

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerBlockEpNormalStrategy(_Sm9xStrategyTestCase):
    """Test CUDA FP8 PerBlock EP Normal strategy"""

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router.get_sm"
    )
    @patch("rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported")
    def test_can_handle_ep_enabled(
        self, mock_supported: Any, mock_get_sm: Any, mock_has_deep_gemm: Any
    ) -> None:
        """Test EP enabled case"""
        mock_has_deep_gemm.return_value = True
        mock_get_sm.return_value = (9, 0)  # SM 9.0 (Hopper)
        mock_supported.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_deepep_low_latency=False),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockEpNormalStrategy()
        self.assertTrue(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router.get_sm"
    )
    @patch("rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported")
    def test_can_handle_tp_dp_ep(
        self, mock_supported: Any, mock_get_sm: Any, mock_has_deep_gemm: Any
    ) -> None:
        """Test case with TP, DP, and EP"""
        mock_has_deep_gemm.return_value = True
        mock_get_sm.return_value = (9, 0)
        mock_supported.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=2, dp_size=2
            ),
            moe_config=create_moe_config(use_deepep_low_latency=False),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockEpNormalStrategy()
        self.assertTrue(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router.get_sm"
    )
    @patch("rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported")
    def test_can_handle_false_cuda_graph(
        self, mock_supported: Any, mock_get_sm: Any, mock_has_deep_gemm: Any
    ) -> None:
        """Test case when CUDA graph is enabled (should fail)"""
        mock_has_deep_gemm.return_value = True
        mock_get_sm.return_value = (9, 0)
        mock_supported.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_deepep_low_latency=False),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockEpNormalStrategy()
        self.assertTrue(strategy.can_handle(config))

        # Now enable CUDA graph - should fail
        config.enable_cuda_graph = True
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router.get_sm"
    )
    @patch("rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported")
    def test_can_handle_false_low_latency(
        self, mock_supported: Any, mock_get_sm: Any, mock_has_deep_gemm: Any
    ) -> None:
        """Test case when low latency is enabled (should fail for normal mode)"""
        mock_has_deep_gemm.return_value = True
        mock_get_sm.return_value = (9, 0)
        mock_supported.return_value = True

        moe_config = create_moe_config(use_deepep_low_latency=True)
        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=1
            ),
            moe_config=moe_config,
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockEpNormalStrategy()
        self.assertFalse(strategy.can_handle(config))
        moe_config.use_deepep_low_latency = False
        self.assertTrue(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router.get_sm"
    )
    def test_can_handle_false_ep_not_enabled(
        self, mock_get_sm: Any, mock_has_deep_gemm: Any
    ) -> None:
        """Test case when EP is not enabled (should fail)"""
        mock_has_deep_gemm.return_value = True
        mock_get_sm.return_value = (9, 0)

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_deepep_low_latency=False),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockEpNormalStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported")
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router.get_sm"
    )
    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_sm_below_90(
        self, mock_has_deep_gemm: Any, mock_get_sm: Any, mock_supported: Any
    ) -> None:
        """Test case when SM < 9.0 (should fail, requires Hopper or newer)"""
        mock_has_deep_gemm.return_value = True
        mock_supported.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_deepep_low_latency=False),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockEpNormalStrategy()
        mock_get_sm.return_value = (8, 9)  # SM 8.9 (Ampere/Ada)
        self.assertFalse(strategy.can_handle(config))

        # Verify it works with SM 9.0+
        mock_get_sm.return_value = (9, 0)
        self.assertTrue(strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        strategy = CudaFp8PerBlockEpNormalStrategy()
        router_type = RouterType.DEEPEP_NORMAL
        executor_type = ExecutorType.DEEPGEMM_CONTINUOUS
        expected_priority = calculate_strategy_priority(router_type, executor_type)

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerTensorNoDPStrategy(_Sm9xStrategyTestCase):
    """Test CUDA FP8 PerTensor single GPU strategy"""

    def test_can_handle_fp8_per_tensor_compressed(self) -> None:
        """Test FP8_PER_TENSOR_COMPRESSED case"""
        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_per_tensor_compressed_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerTensorNoDPStrategy()
        self.assertTrue(strategy.can_handle(config))

    def test_can_handle_fp8_dynamic_per_tensor(self) -> None:
        """Test FP8_DYNAMIC_PER_TENSOR case"""
        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_dynamic_per_tensor_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerTensorNoDPStrategy()
        self.assertTrue(strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        strategy = CudaFp8PerTensorNoDPStrategy()
        router_type = RouterType.PURE_TP
        executor_type = ExecutorType.CUTLASS_FP8
        expected_priority = calculate_strategy_priority(router_type, executor_type)

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaW4a8Int4PerChannelNoDPStrategy(unittest.TestCase):
    """Test CUDA W4A8 INT4 PerChannel single GPU strategy"""

    def test_can_handle_w4a8_int4_per_channel(self) -> None:
        """Test W4A8 INT4 per-channel case"""
        config = create_moe_config_adapter(
            model_config=create_model_config_with_w4a8_int4_per_channel_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaW4a8Int4PerChannelNoDPStrategy()
        self.assertTrue(strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        strategy = CudaW4a8Int4PerChannelNoDPStrategy()
        router_type = RouterType.PURE_TP
        executor_type = ExecutorType.CUTLASS_W4A8_INT4_PER_CHANNEL
        expected_priority = calculate_strategy_priority(router_type, executor_type)

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerBlockPureCPStrategy(_Sm9xStrategyTestCase):
    """Test CUDA FP8 PerBlock pure CP+EP strategy.

    Pure CP requires: dp_size == 1, physical tp == ep > 1, prefill CP enabled,
    use_all_gather. The strategy also gates on moe_strategy being either
    "fp8_per_block_pure_cp" (explicit) or "auto" with matching topology.
    """

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_pure_cp_ep_explicit(self, mock_has_deep_gemm: Any) -> None:
        """Explicit moe_strategy=fp8_per_block_pure_cp on a pure CP+EP topology."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=4, dp_size=1, enable_cp=True
            ),
            moe_config=create_moe_config(
                use_all_gather=True, moe_strategy="fp8_per_block_pure_cp"
            ),
        )

        strategy = CudaFp8PerBlockPureCPStrategy()
        self.assertTrue(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_auto_falls_back_to_deepep(
        self, mock_has_deep_gemm: Any
    ) -> None:
        """moe_strategy=auto + pure CP+EP topology should NOT auto-select PureCP (falls back to DeepEP)."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=4, dp_size=1, enable_cp=True
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerBlockPureCPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_dp_gt_1(self, mock_has_deep_gemm: Any) -> None:
        """dp_size > 1 disqualifies pure CP."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=4, dp_size=2, enable_cp=True
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerBlockPureCPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_tp_ne_ep(self, mock_has_deep_gemm: Any) -> None:
        """Physical tp != ep disqualifies pure CP."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=2, dp_size=1, enable_cp=True
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerBlockPureCPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_cp_disabled(self, mock_has_deep_gemm: Any) -> None:
        """tp==ep but CP not enabled — must not auto-select pure CP."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=4, dp_size=1, enable_cp=False
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerBlockPureCPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_no_all_gather(self, mock_has_deep_gemm: Any) -> None:
        """use_all_gather=False routes back to DeepEP, not pure CP."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=4, dp_size=1, enable_cp=True
            ),
            moe_config=create_moe_config(use_all_gather=False),
        )

        strategy = CudaFp8PerBlockPureCPStrategy()
        self.assertFalse(strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        strategy = CudaFp8PerBlockPureCPStrategy()
        router_type = RouterType.PURE_TP
        executor_type = ExecutorType.DEEPGEMM_CONTINUOUS
        expected_priority = calculate_strategy_priority(router_type, executor_type)

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerBlockPureDPStrategy(_Sm9xStrategyTestCase):
    """Test CUDA FP8 PerBlock pure DP+EP strategy.

    Pure DP requires: physical tp == 1, dp > 1, ep == dp, use_all_gather.
    The strategy also gates on moe_strategy being either
    "fp8_per_block_pure_dp" (explicit) or "auto" with matching topology.
    """

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_pure_dp_ep_explicit(self, mock_has_deep_gemm: Any) -> None:
        """Explicit moe_strategy=fp8_per_block_pure_dp on a pure DP+EP topology."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=2
            ),
            moe_config=create_moe_config(
                use_all_gather=True, moe_strategy="fp8_per_block_pure_dp"
            ),
        )

        strategy = CudaFp8PerBlockPureDPStrategy()
        self.assertTrue(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_auto_falls_back_to_deepep(
        self, mock_has_deep_gemm: Any
    ) -> None:
        """moe_strategy=auto + pure DP+EP topology should NOT auto-select PureDP (falls back to DeepEP)."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=2
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerBlockPureDPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_tp_gt_1(self, mock_has_deep_gemm: Any) -> None:
        """Physical tp > 1 (mixed tp+dp+ep) falls back to DeepEP, not pure DP."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=2, dp_size=2
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerBlockPureDPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_dp_eq_1(self, mock_has_deep_gemm: Any) -> None:
        """dp_size == 1 disqualifies pure DP."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=1
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerBlockPureDPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_ep_ne_dp(self, mock_has_deep_gemm: Any) -> None:
        """ep_size != dp_size disqualifies pure DP."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=4, tp_size=1, dp_size=2
            ),
            moe_config=create_moe_config(use_all_gather=True),
        )

        strategy = CudaFp8PerBlockPureDPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_no_all_gather(self, mock_has_deep_gemm: Any) -> None:
        """use_all_gather=False routes back to DeepEP, not pure DP."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=2
            ),
            moe_config=create_moe_config(use_all_gather=False),
        )

        strategy = CudaFp8PerBlockPureDPStrategy()
        self.assertFalse(strategy.can_handle(config))

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_cuda_graph(self, mock_has_deep_gemm: Any) -> None:
        """enable_cuda_graph=True must reject PureDP (graph-unsafe .item() in _pad_to_max)."""
        mock_has_deep_gemm.return_value = True

        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_block_quant(),
            parallelism_config=create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=2
            ),
            moe_config=create_moe_config(
                use_all_gather=True, moe_strategy="fp8_per_block_pure_dp"
            ),
            enable_cuda_graph=False,
        )

        strategy = CudaFp8PerBlockPureDPStrategy()
        self.assertTrue(strategy.can_handle(config))
        config.enable_cuda_graph = True
        self.assertFalse(strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        strategy = CudaFp8PerBlockPureDPStrategy()
        router_type = RouterType.PURE_TP
        executor_type = ExecutorType.DEEPGEMM_MASKED
        expected_priority = calculate_strategy_priority(router_type, executor_type)

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestPriorityEncoding(unittest.TestCase):
    """The priority encoding must stay collision-free and router-dominant.

    With the old base-10 encoding this actually broke once ExecutorType
    reached 10: PURE_TP(5) + B12X_FP4(10) collided with
    MORI_EP_INTRANODE(6) + BATCHED_TRITON(0)."""

    def test_priority_is_injective(self) -> None:
        seen: dict = {}
        for router in RouterType:
            for executor in ExecutorType:
                p = calculate_strategy_priority(router, executor)
                key = (router.value, executor.value)
                if p in seen and seen[p] != key:
                    self.fail(
                        f"priority collision: {router.name}+{executor.name} and "
                        f"{seen[p]} both encode to {p}"
                    )
                seen[p] = key

    def test_priority_is_router_dominant(self) -> None:
        """A better router must outrank any executor on a worse router."""
        best_executor = max(ExecutorType, key=lambda e: e.value)
        worst_executor = min(ExecutorType, key=lambda e: e.value)
        routers = sorted(RouterType, key=lambda r: r.value)
        for worse, better in zip(routers, routers[1:]):
            self.assertLess(
                calculate_strategy_priority(worse, best_executor),
                calculate_strategy_priority(better, worst_executor),
                f"{worse.name}+{best_executor.name} must not outrank "
                f"{better.name}+{worst_executor.name}",
            )

    def test_executor_values_fit_encoding_base(self) -> None:
        self.assertLess(
            max(executor.value for executor in ExecutorType),
            EXECUTOR_PRIORITY_BASE,
        )

    def test_rejects_executor_that_reaches_encoding_base(self) -> None:
        largest_executor = max(ExecutorType, key=lambda executor: executor.value)
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes.EXECUTOR_PRIORITY_BASE",
            largest_executor.value,
        ), self.assertRaisesRegex(ValueError, "no longer fits"):
            calculate_strategy_priority(RouterType.BATCHED_DATA, largest_executor)


class TestCudaFp4StrategySelection(unittest.TestCase):
    """FP4 strategy matrix: {sm12x, sm100} x {no_dp, ep_low_latency, ep_normal}
    x fp4_moe_op {auto, b12x, cutedsl, trtllm}."""

    @contextlib.contextmanager
    def _arch(self, sm12x: bool) -> Iterator[None]:
        """Patch architecture probes reached by FP4 strategy selection.

        FP4 strategies and the B12X executor import ``is_sm12x`` lazily; the
        two DeepEP routers bind ``get_sm`` when their modules are imported.
        """
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch("rtp_llm.models_py.utils.arch.is_sm12x", return_value=sm12x)
            )
            sm = (12, 0) if sm12x else (10, 0)
            stack.enter_context(
                patch("rtp_llm.models_py.utils.arch.get_sm", return_value=sm)
            )
            stack.enter_context(
                patch(
                    "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router.get_sm",
                    return_value=sm,
                )
            )
            stack.enter_context(
                patch(
                    "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router.get_sm",
                    return_value=sm,
                )
            )
            stack.enter_context(
                patch(
                    "rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported",
                    return_value=True,
                )
            )
            yield

    def _make_config(
        self,
        topology: str,
        fp4_moe_op: str,
        *,
        moe_strategy: str = "auto",
        enable_cuda_graph: bool = False,
    ) -> MoEConfigAdapter:
        if topology == "no_dp":
            moe_config = create_moe_config(use_all_gather=True)
            parallelism_config = create_parallelism_config(
                ep_size=1, tp_size=1, dp_size=1
            )
        elif topology == "ep_low_latency":
            moe_config = create_moe_config(use_deepep_low_latency=True)
            moe_config.use_deepep_moe = True
            parallelism_config = create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=1
            )
        elif topology == "ep_normal":
            moe_config = create_moe_config(use_deepep_low_latency=False)
            moe_config.use_deepep_moe = True
            parallelism_config = create_parallelism_config(
                ep_size=2, tp_size=1, dp_size=1
            )
        elif topology == "tp_eq_ep":
            moe_config = create_moe_config(use_all_gather=True)
            parallelism_config = create_parallelism_config(
                ep_size=2, tp_size=2, dp_size=1
            )
        else:
            raise ValueError(f"unknown topology {topology}")
        moe_config.fp4_moe_op = fp4_moe_op
        moe_config.moe_strategy = moe_strategy
        return create_moe_config_adapter(
            model_config=create_model_config_with_fp4_quant(),
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            enable_cuda_graph=enable_cuda_graph,
        )

    def _candidates(self, config: MoEConfigAdapter) -> dict:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
            CudaFp4B12xNoDPStrategy,
            CudaFp4EpLowLatencyStrategy,
            CudaFp4EpNormalStrategy,
            CudaFp4NoDPStrategy,
        )

        return {
            "b12x": CudaFp4B12xNoDPStrategy().can_handle(config),
            "no_dp": CudaFp4NoDPStrategy().can_handle(config),
            "ep_low_latency": CudaFp4EpLowLatencyStrategy().can_handle(config),
            "ep_normal": CudaFp4EpNormalStrategy().can_handle(config),
        }

    NONE_SELECTED = {
        "b12x": False,
        "no_dp": False,
        "ep_low_latency": False,
        "ep_normal": False,
    }

    # ---- sm12x ----

    def test_sm12x_no_dp_auto_selects_b12x(self) -> None:
        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "auto")
            self.assertEqual(
                self._candidates(config), {**self.NONE_SELECTED, "b12x": True}
            )

    def test_sm12x_no_dp_explicit_b12x_selects_b12x(self) -> None:
        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "b12x")
            self.assertEqual(
                self._candidates(config), {**self.NONE_SELECTED, "b12x": True}
            )

    def test_sm12x_no_dp_b12x_accepts_cuda_graph(self) -> None:
        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "b12x", enable_cuda_graph=True)
            self.assertGreater(config.ll_num_max_token, 0)
            self.assertEqual(
                self._candidates(config), {**self.NONE_SELECTED, "b12x": True}
            )

    def test_sm12x_no_dp_explicit_b12x_strategy_selects_b12x(self) -> None:
        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "b12x", moe_strategy="fp4_b12x")
            self.assertEqual(
                self._candidates(config), {**self.NONE_SELECTED, "b12x": True}
            )

    def test_sm12x_no_dp_b12x_strategy_rejects_trtllm_op(self) -> None:
        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "trtllm", moe_strategy="fp4_b12x")
            self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    def test_no_candidate_error_names_explicit_fp4_conflict(self) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
            CudaFp4B12xNoDPStrategy,
            CudaFp4EpLowLatencyStrategy,
            CudaFp4EpNormalStrategy,
            CudaFp4NoDPStrategy,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
            StrategyRegistry,
        )

        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "trtllm", moe_strategy="fp4_b12x")
            registry = StrategyRegistry()
            for strategy in (
                CudaFp4B12xNoDPStrategy(),
                CudaFp4EpLowLatencyStrategy(),
                CudaFp4EpNormalStrategy(),
                CudaFp4NoDPStrategy(),
            ):
                registry.register(strategy)
            with self.assertRaisesRegex(
                ValueError,
                "moe_strategy='fp4_b12x'.*fp4_moe_op='trtllm'.*"
                "resolved_fp4_moe_op='trtllm'.*gpu_arch='sm120'.*"
                "preserved across backend process serialization.*"
                "fp4_moe_op='auto'",
            ):
                registry.get_strategy(config)

    def test_sm12x_no_dp_explicit_trtllm_rejects_all(self) -> None:
        """trtllm-gen cubins are SM100-only"""
        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "trtllm")
            self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    def test_sm12x_no_dp_explicit_cutedsl_rejects_all(self) -> None:
        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "cutedsl")
            self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    def test_invalid_fp4_moe_op_is_rejected_at_resolution(self) -> None:
        from rtp_llm.config.moe_config import resolve_fp4_moe_op

        with self._arch(sm12x=True):
            config = self._make_config("no_dp", "not-a-kernel")
            with self.assertRaisesRegex(
                ValueError,
                "invalid fp4_moe_op 'not-a-kernel'; expected one of: auto",
            ):
                resolve_fp4_moe_op(config.moe_config, is_sm12x=True)

    def test_sm12x_tp_eq_ep_rejects_all(self) -> None:
        """tp==ep>1 shards experts across ranks, but the b12x kernel indexes
        weights with GLOBAL topk ids (no local-expert remapping)"""
        with self._arch(sm12x=True):
            for fp4_moe_op in ("auto", "b12x", "cutedsl", "trtllm"):
                with self.subTest(fp4_moe_op=fp4_moe_op):
                    config = self._make_config("tp_eq_ep", fp4_moe_op)
                    self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    def test_sm12x_ep_low_latency_rejects_all(self) -> None:
        """sm12x + DeepEP low latency FP4 has no working executor"""
        with self._arch(sm12x=True):
            for fp4_moe_op in ("auto", "b12x", "cutedsl", "trtllm"):
                with self.subTest(fp4_moe_op=fp4_moe_op):
                    config = self._make_config("ep_low_latency", fp4_moe_op)
                    self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    def test_sm12x_ep_normal_rejects_all(self) -> None:
        with self._arch(sm12x=True):
            for fp4_moe_op in ("auto", "b12x", "cutedsl", "trtllm"):
                with self.subTest(fp4_moe_op=fp4_moe_op):
                    config = self._make_config("ep_normal", fp4_moe_op)
                    self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    # ---- sm100 ----

    def test_sm100_no_dp_auto_selects_trtllm(self) -> None:
        with self._arch(sm12x=False):
            config = self._make_config("no_dp", "auto")
            self.assertEqual(
                self._candidates(config), {**self.NONE_SELECTED, "no_dp": True}
            )

    def test_sm100_no_dp_explicit_cutedsl_rejects_all(self) -> None:
        """Mutual exclusion: cutedsl-layout weights must never pair with the
        trtllm executor."""
        with self._arch(sm12x=False):
            config = self._make_config("no_dp", "cutedsl")
            self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    def test_sm100_no_dp_explicit_b12x_rejects_all(self) -> None:
        with self._arch(sm12x=False):
            config = self._make_config("no_dp", "b12x")
            self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    def test_sm100_ep_low_latency_auto_selects_cutedsl(self) -> None:
        with self._arch(sm12x=False):
            config = self._make_config("ep_low_latency", "auto")
            self.assertEqual(
                self._candidates(config),
                {**self.NONE_SELECTED, "ep_low_latency": True},
            )

    def test_sm100_ep_low_latency_explicit_trtllm_rejects_all(self) -> None:
        with self._arch(sm12x=False):
            config = self._make_config("ep_low_latency", "trtllm")
            self.assertEqual(self._candidates(config), self.NONE_SELECTED)

    def test_sm100_ep_normal_auto_selects_trtllm(self) -> None:
        with self._arch(sm12x=False):
            config = self._make_config("ep_normal", "auto")
            self.assertEqual(
                self._candidates(config), {**self.NONE_SELECTED, "ep_normal": True}
            )


if __name__ == "__main__":
    unittest.main()
