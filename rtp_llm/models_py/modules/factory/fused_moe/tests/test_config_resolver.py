"""Configuration resolver tests"""

import unittest
from unittest.mock import patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import DeviceType


def create_config_adapter(
    ep_size: int = 1,
    tp_size: int = 1,
    quant_config=None,
    use_deepep_low_latency: bool = False,
    data_type: str = "fp16",
) -> MoEConfigAdapter:
    """Helper function to create MoEConfigAdapter for testing"""
    model_config = ModelConfig()
    model_config.hidden_size = 1024
    model_config.expert_num = 8
    model_config.moe_k = 2
    model_config.data_type = data_type
    model_config.quant_config = quant_config
    
    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = ep_size
    parallelism_config.tp_size = tp_size
    parallelism_config.dp_size = 1
    parallelism_config.ep_rank = 0
    parallelism_config.tp_rank = 0
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = ep_size * tp_size
    parallelism_config.world_rank = 0
    parallelism_config.local_rank = 0
    parallelism_config.local_world_size = 1
    
    moe_config = MoeConfig()
    moe_config.use_deepep_low_latency = use_deepep_low_latency
    
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        max_generate_batch_size=128,
    )


class TestMoeConfigResolver(unittest.TestCase):
    """Test MoeConfigResolver"""

    def setUp(self):
        """Prepare for testing"""
        self.resolver = MoeConfigResolver()

    def test_get_device_type(self):
        """Test getting device type"""
        with patch(
            "rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver.get_device"
        ) as mock_device:
            mock_device.return_value.get_device_type.return_value = DeviceType.Cuda
            device_type = self.resolver.get_device_type()
            self.assertEqual(device_type, DeviceType.Cuda)

    def test_has_quantization_false(self):
        """Test case without quantization"""
        config = create_config_adapter(quant_config=None)
        self.assertFalse(self.resolver.has_quantization(config))

    def test_has_quantization_true(self):
        """Test case with quantization"""
        quant_config = Fp8BlockWiseQuantConfig()
        config = create_config_adapter(quant_config=quant_config)
        self.assertTrue(self.resolver.has_quantization(config))

    def test_get_quant_method_none(self):
        """Test getting quantization method (no quantization)"""
        config = create_config_adapter(quant_config=None)
        self.assertIsNone(self.resolver.get_quant_method(config))

    def test_get_quant_method(self):
        """Test getting quantization method"""
        quant_config = Fp8BlockWiseQuantConfig()
        config = create_config_adapter(quant_config=quant_config)
        self.assertEqual(self.resolver.get_quant_method(config), "FP8_PER_BLOCK")

    def test_is_bf16_false(self):
        """Test is_bf16 returns False for fp16"""
        config = create_config_adapter(data_type="fp16")
        self.assertFalse(self.resolver.is_bf16(config))

    def test_is_bf16_true(self):
        """Test is_bf16 returns True for bf16"""
        config = create_config_adapter(data_type="bf16")
        self.assertTrue(self.resolver.is_bf16(config))

    def test_is_ep_enabled_false(self):
        """Test EP not enabled"""
        config = create_config_adapter(ep_size=1)
        self.assertFalse(self.resolver.is_ep_enabled(config))

    def test_is_ep_enabled_true(self):
        """Test EP enabled"""
        config = create_config_adapter(ep_size=4)
        self.assertTrue(self.resolver.is_ep_enabled(config))

    def test_use_low_latency(self):
        """Test low latency mode"""
        config = create_config_adapter(use_deepep_low_latency=True)
        self.assertTrue(self.resolver.use_low_latency(config))

    def test_is_single_gpu(self):
        """Test single GPU mode"""
        config = create_config_adapter(ep_size=1)
        self.assertTrue(self.resolver.is_single_gpu(config))

    def test_is_tp_equal_ep(self):
        """Test TP equals EP"""
        config = create_config_adapter(ep_size=4, tp_size=4)
        self.assertTrue(self.resolver.is_tp_equal_ep(config))


if __name__ == "__main__":
    unittest.main()
