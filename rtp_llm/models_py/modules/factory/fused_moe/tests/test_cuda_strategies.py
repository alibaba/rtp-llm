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
    CudaFp8PerBlockNoDPStrategy,
    CudaFp8PerTensorNoDPStrategy,
    CudaW4a8Int4PerChannelNoDPStrategy,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import DeviceType


# Helper functions for creating configuration objects
def create_model_config_without_quant() -> ModelConfig:
    """Create ModelConfig without quantization"""
    model_config = ModelConfig()
    model_config.quant_config = None
    return model_config


def create_model_config_with_fp8_block_quant() -> ModelConfig:
    """Create ModelConfig with FP8 block-wise quantization"""
    model_config = ModelConfig()
    model_config.quant_config = Fp8BlockWiseQuantConfig()
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
    ep_size: int = 1, tp_size: int = 1, dp_size: int = 1
) -> ParallelismConfig:
    """Create ParallelismConfig with specified parallelism settings

    Args:
        ep_size: Expert parallelism size
        tp_size: Tensor parallelism size
        dp_size: Data parallelism size
    """
    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = ep_size
    parallelism_config.tp_size = tp_size
    parallelism_config.dp_size = dp_size
    return parallelism_config


def create_moe_config(
    use_deepep_low_latency: bool = False, use_all_gather: Optional[bool] = None
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
        executor_type = ExecutorType.CUTLASS_W4A8_INT4
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(strategy.priority, expected_priority)


if __name__ == "__main__":
    unittest.main()
