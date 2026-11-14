import runpy
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp4 import (
    CudaNvfp4MarlinStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig


def _config(model_type="qwen4_exp"):
    model = ModelConfig()
    model.model_type, model.data_type = model_type, "bf16"
    model.expert_num, model.moe_k = 8, 3
    model.hidden_size, model.moe_inter_size = 2560, 640
    model.quant_config = ModelOptFp4Config(
        bits=4, group_size=16, is_quanted=False, model_recipe=QWEN4_FP8_NVFP4_RECIPE
    )
    moe = MoeConfig()
    moe.moe_strategy, moe.fp4_moe_op = "nvfp4_marlin", "marlin"
    moe.use_all_gather, moe.use_deepep_moe = True, False
    return MoEConfigAdapter(model, ParallelismConfig(), moe)


def _histogram(input, num_bins, mask=None):
    pass


class MarlinAdmissionTest(unittest.TestCase):
    def test_incompatible_factory_paths_never_construct_or_load_marlin(self):
        registry = StrategyRegistry()
        strategy = CudaNvfp4MarlinStrategy()
        registry.register(strategy)
        fallback = MagicMock(strategy_name="fallback", supported_moe_quant_method=None)
        fallback.can_handle.return_value = True
        registry.register(fallback)
        cases = [
            {"torch.__version__": version, "torch.cuda.get_device_capability": cap}
            for version in ("2.6.0+cu124", "2.8.0+cu129", "2.9.1+cu130")
            for cap in ((9, 0), (12, 0))
        ] + [
            {"torch.cuda.is_available": False},
            {"torch.version.hip": "6.3"},
            {"torch.cuda.get_device_capability": (8, 0)},
            {"torch.cuda.get_device_capability": (10, 0)},
            {"triton.language.histogram": lambda input, num_bins: None},
            {"triton.language.histogram": None},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), ExitStack() as stack:
                values = {
                    "torch.__version__": "2.11.0+cu130",
                    "torch.version.hip": None,
                    "torch.cuda.is_available": True,
                    "torch.cuda.get_device_capability": (12, 0),
                    "triton.language.histogram": _histogram,
                    **overrides,
                }
                for target, value in values.items():
                    stack.enter_context(
                        patch(target, return_value=value)
                        if target.startswith("torch.cuda.")
                        else patch(target, value)
                    )
                construct = stack.enter_context(
                    patch(
                        "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                        "marlin_nvfp4_executor.Qwen4MarlinExecutor.__init__"
                    )
                )
                load = stack.enter_context(
                    patch(
                        "rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4.load_library"
                    )
                )
                stack.enter_context(
                    patch.object(FusedMoeFactory, "_registry", registry)
                )
                for model_type in ("qwen4_exp", "qwen4_exp_mtp"):
                    config = _config(model_type)
                    self.assertFalse(strategy.can_handle(config))
                    with self.assertRaisesRegex(
                        ValueError, "MOE_STRATEGY='nvfp4_marlin'"
                    ):
                        FusedMoeFactory().create_fused_moe(config, {})
                construct.assert_not_called()
                load.assert_not_called()
                fallback.can_handle.assert_not_called()

    def test_supported_runtime_selects_marlin(self):
        registry = StrategyRegistry()
        strategy = CudaNvfp4MarlinStrategy()
        registry.register(strategy)
        for version in ("2.10.0+cu130", "2.11.0+cu130"):
            for cap in ((9, 0), (12, 0)):
                with (
                    self.subTest(version=version, capability=cap),
                    patch("torch.__version__", version),
                    patch("torch.version.hip", None),
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.get_device_capability", return_value=cap),
                    patch("triton.language.histogram", _histogram),
                ):
                    self.assertIs(registry.get_strategy(_config()), strategy)

    def test_automatic_selection_never_imports_marlin_executor(self):
        config = _config()
        config.moe_strategy = "auto"
        strategy = CudaNvfp4MarlinStrategy()
        with patch.object(strategy, "get_attributes") as attributes:
            self.assertFalse(strategy.can_handle(config))
            attributes.assert_not_called()

    def test_direct_kernel_tests_skip_incompatible_runtime_before_library_load(self):
        for filename, class_name in (
            ("marlin_nvfp4_test.py", "MarlinNvfp4Test"),
            ("marlin_large_decode_test.py", "MarlinLargeDecodeTest"),
        ):
            with (
                self.subTest(filename=filename),
                patch("torch.__version__", "2.8.0+cu129"),
                patch(
                    "rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4.load_library"
                ) as load,
            ):
                module = runpy.run_module(
                    "rtp_llm.models_py.kernels.cuda.marlin_nvfp4.test." + filename[:-3]
                )
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(
                    module[class_name]
                )
                expected = suite.countTestCases()
                self.assertGreater(expected, 0)
                result = unittest.TestResult()
                suite.run(result)
                self.assertTrue(result.wasSuccessful())
                self.assertEqual(len(result.skipped), expected)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
