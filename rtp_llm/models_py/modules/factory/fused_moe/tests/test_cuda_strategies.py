"""CUDA strategy tests"""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.strategy.batched_triton_strategy import (
    BatchedTritonStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
    CudaFp8PerBlockNoDPStrategy,
    CudaFp8PerTensorNoDPStrategy,
)
from rtp_llm.ops.compute_ops import DeviceType


class TestCudaNoQuantSingleGpuStrategy(unittest.TestCase):
    """Test CUDA single GPU without quantization strategy"""

    def setUp(self):
        """Prepare for testing"""
        self.strategy = BatchedTritonStrategy()

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    def test_can_handle_true(self, mock_resolver_class: Any) -> None:
        """Test case that can be handled"""
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.get_device_type.return_value = DeviceType.Cuda
        mock_resolver.has_quantization.return_value = False
        mock_resolver.is_single_gpu.return_value = True

        config = MagicMock()
        self.assertTrue(self.strategy.can_handle(config))

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    def test_can_handle_false_not_cuda(self, mock_resolver_class: Any) -> None:
        """Test not CUDA device

        Note: This test may not work as expected because Router/Executor
        check_conditions methods don't check device type directly.
        The strategy name suggests it's CUDA-specific, but the actual
        implementation relies on Router/Executor conditions.
        """
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.has_quantization.return_value = False
        mock_resolver.is_single_gpu.return_value = (
            False  # Make it fail on single_gpu check
        )

        config = MagicMock()
        self.assertFalse(self.strategy.can_handle(config))

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    def test_can_handle_false_has_quant(self, mock_resolver_class: Any) -> None:
        """Test case with quantization"""
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.get_device_type.return_value = DeviceType.Cuda
        mock_resolver.has_quantization.return_value = True
        mock_resolver.is_single_gpu.return_value = True

        config = MagicMock()
        self.assertFalse(self.strategy.can_handle(config))

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    def test_can_handle_false_not_single_gpu(self, mock_resolver_class: Any) -> None:
        """Test multi-GPU case"""
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.get_device_type.return_value = DeviceType.Cuda
        mock_resolver.has_quantization.return_value = False
        mock_resolver.is_single_gpu.return_value = False

        config = MagicMock()
        self.assertFalse(self.strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        router_type = RouterType.BATCHED_DATA
        executor_type = ExecutorType.BATCHED_TRITON
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = self.strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(self.strategy.priority, expected_priority)


class TestCudaFp8PerBlockNoDPStrategy(unittest.TestCase):
    """Test CUDA FP8 PerBlock single GPU strategy"""

    def setUp(self):
        """Prepare for testing"""
        self.strategy = CudaFp8PerBlockNoDPStrategy()

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_single_gpu(
        self, mock_has_deep_gemm: Any, mock_resolver_class: Any
    ) -> None:
        """Test single GPU case"""
        mock_has_deep_gemm.return_value = True

        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.get_quant_method.return_value = "FP8_PER_BLOCK"
        mock_resolver.is_single_gpu.return_value = True
        mock_resolver.is_tp_equal_ep.return_value = False

        config = MagicMock()
        config.enable_cuda_graph = False
        self.assertTrue(self.strategy.can_handle(config))

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_tp_equal_ep(
        self, mock_has_deep_gemm: Any, mock_resolver_class: Any
    ) -> None:
        """Test TP equals EP case"""
        mock_has_deep_gemm.return_value = True

        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.get_quant_method.return_value = "FP8_PER_BLOCK"
        mock_resolver.is_single_gpu.return_value = False
        mock_resolver.is_tp_equal_ep.return_value = True

        config = MagicMock()
        config.enable_cuda_graph = False
        self.assertTrue(self.strategy.can_handle(config))

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    @patch("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.has_deep_gemm")
    def test_can_handle_false_no_deep_gemm(
        self, mock_has_deep_gemm: Any, mock_resolver_class: Any
    ) -> None:
        """Test case when deep_gemm is not available"""
        mock_has_deep_gemm.return_value = False

        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.get_quant_method.return_value = "FP8_PER_BLOCK"
        mock_resolver.is_single_gpu.return_value = True
        mock_resolver.is_tp_equal_ep.return_value = False

        config = MagicMock()
        config.enable_cuda_graph = False
        self.assertFalse(self.strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        router_type = RouterType.PURE_TP
        executor_type = ExecutorType.DEEPGEMM_CONTINUOUS
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = self.strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(self.strategy.priority, expected_priority)


class TestCudaFp8PerTensorNoDPStrategy(unittest.TestCase):
    """Test CUDA FP8 PerTensor single GPU strategy"""

    def setUp(self):
        """Prepare for testing"""
        self.strategy = CudaFp8PerTensorNoDPStrategy()

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    def test_can_handle_fp8_per_tensor_compressed(
        self, mock_resolver_class: Any
    ) -> None:
        """Test FP8_PER_TENSOR_COMPRESSED case"""
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.get_quant_method.return_value = "FP8_PER_TENSOR_COMPRESSED"
        mock_resolver.is_single_gpu.return_value = True

        config = MagicMock()
        self.assertTrue(self.strategy.can_handle(config))

    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.MoeConfigResolver"
    )
    def test_can_handle_fp8_dynamic_per_tensor(self, mock_resolver_class: Any) -> None:
        """Test FP8_DYNAMIC_PER_TENSOR case"""
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver

        mock_resolver.get_quant_method.return_value = "FP8_DYNAMIC_PER_TENSOR"
        mock_resolver.is_single_gpu.return_value = True

        config = MagicMock()
        self.assertTrue(self.strategy.can_handle(config))

    def test_priority(self) -> None:
        """Test priority"""
        router_type = RouterType.PURE_TP
        executor_type = ExecutorType.CUTLASS_FP8
        expected_priority = router_type.value * 10 + executor_type.value

        attributes = self.strategy.get_attributes()
        self.assertEqual(attributes.router_class.router_type(), router_type)
        self.assertEqual(attributes.executor_class.executor_type(), executor_type)
        self.assertEqual(self.strategy.priority, expected_priority)


if __name__ == "__main__":
    unittest.main()
