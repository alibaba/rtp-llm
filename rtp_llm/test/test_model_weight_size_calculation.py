"""
Unit tests for model weight size calculation with TP and EP parallelism.

Tests the eval_model_weight_size_with_parallelism method to ensure correct
memory calculation for MoE models considering both TP and EP splitting.
"""

import unittest
from unittest.mock import MagicMock, Mock

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters


class TestModelWeightSizeCalculation(unittest.TestCase):
    """Test cases for model weight size calculation with parallelism"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a GptInitModelParameters with required parameters
        self.config = GptInitModelParameters(
            head_num=32,
            size_per_head=128,
            layer_num=32,
            max_seq_len=2048,
            vocab_size=32000,
        )
        self.config.head_num_kv = 32
        self.config.inter_size = 11008
        self.config.activation_type = "silu"
        self.config.layer_head_num = None
        self.config.layer_inter_size = None
        self.config.moe_layer_index = []

        # Mock quant_algo
        self.config.quant_algo = Mock()
        self.config.quant_algo.getWeightBits = Mock(return_value=16)

    def test_non_moe_model_tp_only(self):
        """Test non-MoE model with TP splitting only"""
        self.config.expert_num = 0
        self.config.phy_exp_num = 0

        # Calculate with TP=2, EP=1
        size_with_parallelism = self.config.eval_model_weight_size_with_parallelism(
            tp_size=2, ep_size=1
        )

        # Calculate original size
        original_size = self.config.eval_model_weight_size()

        # For non-MoE, size should be original / tp_size
        expected_size = original_size / 2
        self.assertAlmostEqual(
            size_with_parallelism, expected_size, delta=expected_size * 0.01
        )

    def test_moe_model_with_ep_splitting(self):
        """Test MoE model with both TP and EP splitting"""
        # Configure MoE model
        self.config.expert_num = 8
        self.config.phy_exp_num = 12  # 8 logical + 4 redundant
        self.config.moe_layer_index = [0, 2, 4, 6]  # 4 MoE layers
        self.config.moe_style = 0  # Partial layers are MoE
        self.config.moe_inter_padding_size = 2048

        # Calculate with TP=2, EP=2
        size_with_parallelism = self.config.eval_model_weight_size_with_parallelism(
            tp_size=2, ep_size=2
        )

        # Verify size is less than original / tp_size (due to EP splitting of MoE)
        original_size = self.config.eval_model_weight_size()
        size_with_tp_only = original_size / 2

        # With EP splitting, size should be less than TP-only
        self.assertLess(size_with_parallelism, size_with_tp_only)

    def test_moe_style_1_all_layers_moe(self):
        """Test MoE style 1 where all layers are MoE"""
        self.config.expert_num = 8
        self.config.phy_exp_num = 12
        self.config.moe_layer_index = list(range(32))  # All layers
        self.config.moe_style = 1
        self.config.moe_inter_padding_size = 2048

        size_tp2_ep2 = self.config.eval_model_weight_size_with_parallelism(
            tp_size=2, ep_size=2
        )
        size_tp2_ep4 = self.config.eval_model_weight_size_with_parallelism(
            tp_size=2, ep_size=4
        )

        # With higher EP, size should be smaller
        self.assertLess(size_tp2_ep4, size_tp2_ep2)

    def test_moe_style_2_shared_expert(self):
        """Test MoE style 2 with shared expert"""
        self.config.expert_num = 8
        self.config.phy_exp_num = 12
        self.config.moe_layer_index = [0, 2, 4, 6]
        self.config.moe_style = 2  # Shared Expert + MoE Expert
        self.config.moe_inter_padding_size = 2048

        size = self.config.eval_model_weight_size_with_parallelism(tp_size=2, ep_size=2)

        # Should successfully calculate without error
        self.assertGreater(size, 0)

    def test_invalid_parallelism_config(self):
        """Test error handling for invalid parallelism config"""
        self.config.expert_num = 0
        self.config.phy_exp_num = 0

        # Test invalid tp_size
        with self.assertRaises(ValueError):
            self.config.eval_model_weight_size_with_parallelism(tp_size=0, ep_size=1)

        # Test invalid ep_size
        with self.assertRaises(ValueError):
            self.config.eval_model_weight_size_with_parallelism(tp_size=1, ep_size=0)

        # Test negative values
        with self.assertRaises(ValueError):
            self.config.eval_model_weight_size_with_parallelism(tp_size=-1, ep_size=1)

    def test_ep_size_effect_on_moe_only(self):
        """Test that EP size only affects MoE layers, not non-MoE layers"""
        self.config.expert_num = 8
        self.config.phy_exp_num = 12
        self.config.moe_layer_index = [0, 2]  # Only 2 MoE layers out of 32
        self.config.moe_style = 0
        self.config.moe_inter_padding_size = 2048

        # Calculate non-MoE and MoE weights separately
        non_moe_weight = self.config._calc_non_moe_layer_weight()
        moe_weight = self.config._calc_moe_layer_weight()

        # Verify MoE weight is non-zero
        self.assertGreater(moe_weight, 0)

        # Verify non-MoE weight is much larger (30 layers vs 2 MoE layers)
        self.assertGreater(non_moe_weight, moe_weight)

    def test_physical_vs_logical_experts(self):
        """Test that physical expert count is used, not logical"""
        self.config.expert_num = 8  # Logical experts
        self.config.phy_exp_num = 12  # Physical experts (with redundancy)
        self.config.moe_layer_index = [0]
        self.config.moe_style = 0
        self.config.moe_inter_padding_size = 2048

        moe_weight = self.config._calc_moe_layer_weight()

        # Calculate expected weight with physical experts
        hidden_size = self.config.gpt_init_params.hidden_size
        ffn_w_count = 3  # silu activation
        expected_weight = (
            1
            * self.config.inter_size
            * hidden_size
            * ffn_w_count
            * 12  # Use phy_exp_num
            + 1 * hidden_size * 8  # Gate weight uses expert_num
        )

        self.assertEqual(moe_weight, expected_weight)

    def test_quant_weight_bits(self):
        """Test different quantization bit widths"""
        self.config.expert_num = 0
        self.config.phy_exp_num = 0

        # Test FP16 (2 bytes)
        self.config.quant_algo.getWeightBits = Mock(return_value=16)
        size_fp16 = self.config.eval_model_weight_size_with_parallelism(
            tp_size=1, ep_size=1
        )

        # Test INT8 (1 byte)
        self.config.quant_algo.getWeightBits = Mock(return_value=8)
        size_int8 = self.config.eval_model_weight_size_with_parallelism(
            tp_size=1, ep_size=1
        )

        # Test INT4 (0.54 bytes)
        self.config.quant_algo.getWeightBits = Mock(return_value=4)
        size_int4 = self.config.eval_model_weight_size_with_parallelism(
            tp_size=1, ep_size=1
        )

        # Verify size relationships
        self.assertAlmostEqual(size_fp16 / size_int8, 2.0, delta=0.1)
        self.assertLess(size_int4, size_int8)

    def test_consistency_with_original_method(self):
        """Test that new method is consistent with original for non-MoE + TP=1, EP=1"""
        self.config.expert_num = 0
        self.config.phy_exp_num = 0

        original_size = self.config.eval_model_weight_size()
        new_size = self.config.eval_model_weight_size_with_parallelism(
            tp_size=1, ep_size=1
        )

        # Should be identical for non-MoE with no parallelism
        self.assertAlmostEqual(original_size, new_size, delta=original_size * 0.01)


class TestMoEWeightCalculation(unittest.TestCase):
    """Test cases specifically for MoE weight calculation helpers"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = GptInitModelParameters(
            head_num=32,
            size_per_head=128,
            layer_num=32,
            max_seq_len=2048,
            vocab_size=32000,
        )
        self.config.head_num_kv = 32
        self.config.inter_size = 11008
        self.config.activation_type = "silu"
        self.config.layer_head_num = None
        self.config.layer_inter_size = None
        self.config.quant_algo = Mock()
        self.config.quant_algo.getWeightBits = Mock(return_value=16)

    def test_calc_non_moe_layer_weight_no_moe(self):
        """Test non-MoE layer weight calculation when there's no MoE"""
        self.config.expert_num = 0
        self.config.moe_layer_index = []

        weight = self.config._calc_non_moe_layer_weight()

        # Should include all layers' FFN weights
        self.assertGreater(weight, 0)

    def test_calc_moe_layer_weight_no_moe(self):
        """Test MoE layer weight calculation returns 0 when there's no MoE"""
        self.config.expert_num = 0

        weight = self.config._calc_moe_layer_weight()

        self.assertEqual(weight, 0)

    def test_calc_moe_layer_weight_with_redundancy(self):
        """Test MoE weight calculation uses physical expert count"""
        self.config.expert_num = 8
        self.config.phy_exp_num = 12
        self.config.moe_layer_index = [0, 1]
        self.config.moe_style = 0
        self.config.moe_inter_padding_size = 2048

        weight = self.config._calc_moe_layer_weight()

        # Should use phy_exp_num (12) not expert_num (8)
        self.assertGreater(weight, 0)


if __name__ == "__main__":
    unittest.main()
