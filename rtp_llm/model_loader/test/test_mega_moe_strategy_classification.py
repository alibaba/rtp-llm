import os
import unittest
from unittest import mock

from rtp_llm.model_loader.model_weight_info import (
    ModelWeightInfo,
    _apply_mega_moe_fp4_wrappers,
)
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import is_mega_moe_strategy


class TestMegaMoeStrategyClassification(unittest.TestCase):
    def test_fp4_strategies_enable_load_time_fp4_wrappers(self):
        for strategy in ("mega_moe", "mega_moe_se", "mega_moe_fused"):
            with self.subTest(strategy=strategy), mock.patch.dict(
                os.environ, {"MOE_STRATEGY": strategy}, clear=False
            ):
                self.assertTrue(is_mega_moe_strategy())

    def test_fp8_strategies_bypass_load_time_fp4_wrappers(self):
        for strategy in ("mega_moe_fp8", "mega_moe_fp8_se"):
            with self.subTest(strategy=strategy), mock.patch.dict(
                os.environ, {"MOE_STRATEGY": strategy}, clear=False
            ):
                self.assertFalse(is_mega_moe_strategy())

    def test_fp8_strategies_leave_weight_tree_untouched(self):
        for strategy in ("mega_moe_fp8", "mega_moe_fp8_se"):
            original_layer = object()
            weight_info = ModelWeightInfo(weights=[], layer_weights=[original_layer])
            with self.subTest(strategy=strategy), mock.patch.dict(
                os.environ, {"MOE_STRATEGY": strategy}, clear=False
            ), mock.patch(
                "rtp_llm.model_loader.online_modelopt_fp4_quant_weight.wrap_moe_for_mega_moe"
            ) as wrap:
                result = _apply_mega_moe_fp4_wrappers(weight_info)

            self.assertIs(result, weight_info)
            self.assertIs(result.layer_weights[0], original_layer)
            wrap.assert_not_called()

    def test_fp4_strategies_reach_wrapper_callsite(self):
        for strategy in ("mega_moe", "mega_moe_se", "mega_moe_fused"):
            original_layer = object()
            wrapped_layer = object()
            weight_info = ModelWeightInfo(weights=[], layer_weights=[original_layer])
            with self.subTest(strategy=strategy), mock.patch.dict(
                os.environ, {"MOE_STRATEGY": strategy}, clear=False
            ), mock.patch(
                "rtp_llm.model_loader.offline_modelopt_fp4_quant_weight.is_offline_mega_moe_fp4_ckpt",
                return_value=False,
            ), mock.patch(
                "rtp_llm.model_loader.online_modelopt_fp4_quant_weight.wrap_moe_for_mega_moe",
                return_value=wrapped_layer,
            ) as wrap:
                result = _apply_mega_moe_fp4_wrappers(weight_info)

            self.assertIs(result, weight_info)
            self.assertIs(result.layer_weights[0], wrapped_layer)
            wrap.assert_called_once_with(original_layer)


if __name__ == "__main__":
    unittest.main()
