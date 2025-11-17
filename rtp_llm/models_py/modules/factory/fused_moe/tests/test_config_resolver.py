"""Configuration resolver tests"""

import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.ops.compute_ops import DeviceType


class TestMoeConfigResolver(unittest.TestCase):
    """Test MoeConfigResolver"""

    def setUp(self):
        """Prepare for testing"""
        self.resolver = MoeConfigResolver()

    def test_get_device_type(self):
        """Test getting device type"""
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.config_resolver.get_device"
        ) as mock_device:
            mock_device.return_value.get_device_type.return_value = DeviceType.Cuda
            device_type = self.resolver.get_device_type()
            self.assertEqual(device_type, DeviceType.Cuda)

    def test_has_quantization_false(self):
        """Test case without quantization"""
        config = MagicMock()
        config.quant_config = None
        self.assertFalse(self.resolver.has_quantization(config))

    def test_has_quantization_true(self):
        """Test case with quantization"""
        config = MagicMock()
        config.quant_config = MagicMock()
        self.assertTrue(self.resolver.has_quantization(config))

    def test_get_quant_method_none(self):
        """Test getting quantization method (no quantization)"""
        config = MagicMock()
        config.quant_config = None
        self.assertIsNone(self.resolver.get_quant_method(config))

    def test_get_quant_method(self):
        """Test getting quantization method"""
        config = MagicMock()
        config.quant_config = MagicMock()
        config.quant_config.get_method.return_value = "FP8_PER_BLOCK"
        self.assertEqual(self.resolver.get_quant_method(config), "FP8_PER_BLOCK")

    def test_is_ep_enabled_false(self):
        """Test EP not enabled"""
        config = MagicMock()
        config.ep_size = 1
        self.assertFalse(self.resolver.is_ep_enabled(config))

    def test_is_ep_enabled_true(self):
        """Test EP enabled"""
        config = MagicMock()
        config.ep_size = 4
        self.assertTrue(self.resolver.is_ep_enabled(config))

    def test_use_low_latency(self):
        """Test low latency mode"""
        config = MagicMock()
        config.moe_config.use_deepep_low_latency = True
        self.assertTrue(self.resolver.use_low_latency(config))

    def test_is_single_gpu(self):
        """Test single GPU mode"""
        config = MagicMock()
        config.ep_size = 1
        self.assertTrue(self.resolver.is_single_gpu(config))

    def test_is_tp_equal_ep(self):
        """Test TP equals EP"""
        config = MagicMock()
        config.tp_size = 4
        config.ep_size = 4
        self.assertTrue(self.resolver.is_tp_equal_ep(config))


if __name__ == "__main__":
    unittest.main()
