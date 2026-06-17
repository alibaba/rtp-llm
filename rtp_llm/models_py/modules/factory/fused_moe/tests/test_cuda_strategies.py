"""CUDA strategy tests"""

import unittest
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8DynamicPerTensorQuantConfig,
    W4a8Int4PerChannelQuantConfig,
)
from rtp_llm.device.device_type import DeviceType
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
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
    CudaNoQuantDpNormalDeepGemmStrategy,
    CudaW4a8Int4PerChannelNoDPStrategy,
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


def create_model_config_with_fp8_per_tensor_quant() -> ModelConfig:
    """Create ModelConfig with FP8 per-tensor quantization"""
    model_config = ModelConfig()
    model_config.quant_config = Fp8DynamicPerTensorQuantConfig()
    return model_config


def create_model_config_with_w4a8_int4_per_channel_quant() -> ModelConfig:
    """Create ModelConfig with W4A8 INT4 per-channel quantization"""
    model_config = ModelConfig()
    model_config.quant_config = W4a8Int4PerChannelQuantConfig()
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
            model_config=create_model_config_with_fp8_per_tensor_quant(),
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


class TestCudaFp8PerBlockNoDPStrategy(unittest.TestCase):
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
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerBlockNoDPMaskedStrategy(unittest.TestCase):
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
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerBlockEpNormalStrategy(unittest.TestCase):
    """Test CUDA FP8 PerBlock EP Normal strategy"""

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    @patch("rtp_llm.models_py.utils.arch.get_sm")
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
    @patch("rtp_llm.models_py.utils.arch.get_sm")
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
    @patch("rtp_llm.models_py.utils.arch.get_sm")
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
    @patch("rtp_llm.models_py.utils.arch.get_sm")
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
    @patch("rtp_llm.models_py.utils.arch.get_sm")
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
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerTensorNoDPStrategy(unittest.TestCase):
    """Test CUDA FP8 PerTensor single GPU strategy"""

    def test_can_handle_fp8_per_tensor_compressed(self) -> None:
        """Test FP8_PER_TENSOR_COMPRESSED case"""
        config = create_moe_config_adapter(
            model_config=create_model_config_with_fp8_per_tensor_quant(),
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
            model_config=create_model_config_with_fp8_per_tensor_quant(),
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
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaW4a8Int4PerChannelNoDPStrategy(unittest.TestCase):
    """Test CUDA W4A8 INT4 PerChannel single GPU strategy"""

    def test_can_handle_w4a8_int4_per_channel(self) -> None:
        """Test FP8_DYNAMIC_PER_TENSOR case"""
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
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerBlockPureCPStrategy(unittest.TestCase):
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
    def test_can_handle_false_auto_falls_back_to_deepep(self, mock_has_deep_gemm: Any) -> None:
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
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaFp8PerBlockPureDPStrategy(unittest.TestCase):
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
    def test_can_handle_false_auto_falls_back_to_deepep(self, mock_has_deep_gemm: Any) -> None:
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
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


class TestCudaNoQuantDpNormalDeepGemmStrategy(unittest.TestCase):
    """Test CUDA no-quant DeepEP-Normal bf16 DeepGEMM strategy (opt-in)."""

    def _make_config(
        self,
        *,
        data_type: str = "bf16",
        moe_strategy: str = "no_quant_dp_normal_deepgemm",
        ep_size: int = 2,
        tp_size: int = 1,
        dp_size: int = 1,
        enable_cuda_graph: bool = False,
    ) -> MoEConfigAdapter:
        model_config = create_model_config_without_quant()
        model_config.data_type = data_type
        return create_moe_config_adapter(
            model_config=model_config,
            parallelism_config=create_parallelism_config(
                ep_size=ep_size, tp_size=tp_size, dp_size=dp_size
            ),
            moe_config=create_moe_config(
                use_deepep_low_latency=False, moe_strategy=moe_strategy
            ),
            enable_cuda_graph=enable_cuda_graph,
        )

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    @patch(
        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm_bf16_grouped"
    )
    @patch("rtp_llm.models_py.utils.arch.get_sm")
    @patch("rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported")
    def test_can_handle_and_attributes(
        self,
        mock_supported: Any,
        mock_get_sm: Any,
        mock_has_grouped: Any,
        mock_has_deep_gemm: Any,
    ) -> None:
        """Positive: explicit opt-in, bf16, ep>1, sm9, no cuda graph -> selected,
        and routes to DeepEP-Normal router + bf16 hybrid executor."""
        mock_supported.return_value = True
        mock_get_sm.return_value = (9, 0)
        mock_has_grouped.return_value = True
        mock_has_deep_gemm.return_value = True

        strategy = CudaNoQuantDpNormalDeepGemmStrategy()
        self.assertTrue(strategy.can_handle(self._make_config()))

        attrs = strategy.get_attributes()
        self.assertEqual(attrs.router_class.__name__, "DeepepNormalRouterNoQuant")
        self.assertEqual(attrs.executor_class.__name__, "DeepGemmBf16HybridExecutor")

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    @patch(
        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm_bf16_grouped"
    )
    @patch("rtp_llm.models_py.utils.arch.get_sm")
    @patch("rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported")
    def test_negative_cases(
        self,
        mock_supported: Any,
        mock_get_sm: Any,
        mock_has_grouped: Any,
        mock_has_deep_gemm: Any,
    ) -> None:
        """Negative: fp16 dtype, auto (not opt-in), cuda graph, and missing bf16
        grouped symbols each disqualify the strategy."""
        mock_supported.return_value = True
        mock_get_sm.return_value = (9, 0)
        mock_has_grouped.return_value = True
        mock_has_deep_gemm.return_value = True

        strategy = CudaNoQuantDpNormalDeepGemmStrategy()

        # fp16 (not bf16)
        self.assertFalse(strategy.can_handle(self._make_config(data_type="fp16")))
        # not explicitly opted in
        self.assertFalse(strategy.can_handle(self._make_config(moe_strategy="auto")))
        # CUDA graph incompatible (runtime masked/contiguous dispatch)
        self.assertFalse(
            strategy.can_handle(self._make_config(enable_cuda_graph=True))
        )
        # bf16 grouped GEMM symbols unavailable -> fail fast at selection
        mock_has_grouped.return_value = False
        self.assertFalse(strategy.can_handle(self._make_config()))


class TestHasDeepGemmBf16Grouped(unittest.TestCase):
    """has_deep_gemm_bf16_grouped() must report unavailability, never raise, so it
    is safe to call during strategy enumeration for any config."""

    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_false_when_package_absent(self, mock_has_deep_gemm: Any) -> None:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            has_deep_gemm_bf16_grouped,
        )

        mock_has_deep_gemm.return_value = False
        self.assertFalse(has_deep_gemm_bf16_grouped())

    @patch(
        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper._ensure_bf16_initialized"
    )
    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_false_not_raise_on_symbol_resolution_error(
        self, mock_has_deep_gemm: Any, mock_ensure_bf16: Any
    ) -> None:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            has_deep_gemm_bf16_grouped,
        )

        mock_has_deep_gemm.return_value = True
        mock_ensure_bf16.side_effect = RuntimeError("symbol not found")
        # Must swallow the resolution error and return False, not propagate.
        self.assertFalse(has_deep_gemm_bf16_grouped())


class TestEnsureBf16Initialized(unittest.TestCase):
    """The bf16 symbol init must be decoupled from the fp8 path: missing bf16
    symbols leave the impls None and never raise, so _ensure_initialized() (fp8)
    is unaffected."""

    @patch(
        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper._bf16_symbols_initialized",
        False,
    )
    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper._lazy_init_deep_gemm")
    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_tolerates_missing_bf16_symbols(
        self, mock_has_deep_gemm: Any, mock_lazy_init: Any
    ) -> None:
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

        mock_has_deep_gemm.return_value = True
        mock_lazy_init.side_effect = RuntimeError("bf16 grouped symbol not found")
        # Must not propagate the resolution error (would otherwise break the fp8
        # path's _ensure_initialized, which is a separate call).
        deepgemm_wrapper._ensure_bf16_initialized()


if __name__ == "__main__":
    unittest.main()
