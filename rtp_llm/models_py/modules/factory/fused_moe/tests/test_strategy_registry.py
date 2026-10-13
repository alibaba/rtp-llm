"""Strategy-registry diagnostics for public MOE_STRATEGY values."""

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.impl.common.strategy.batched_triton_strategy import (
    BatchedTritonStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda import (
    strategy as cuda_strategies,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
    CudaMegaMoeStrategy,
    CudaNoQuantCppStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm import (
    strategy as rocm_strategies,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.strategy import (
    RocmBf16PureTPStrategy,
    RocmEpLowLatencyStrategy,
    RocmEpNormalStrategy,
    RocmFp8PerBlockPureTPStrategy,
    RocmFp8PerChannelPureTPStrategy,
    RocmMXFp4PureTPStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.condition_checker import (
    ConditionChecker,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.server.server_args.moe_group_args import MOE_STRATEGY_CHOICES


def _config(strategy: str, quant_config=None) -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.model_type = "test_model"
    model_config.quant_config = quant_config

    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = 1
    parallelism_config.tp_size = 1
    parallelism_config.dp_size = 1
    parallelism_config.world_size = 1

    moe_config = MoeConfig()
    moe_config.moe_strategy = strategy
    moe_config.use_deepep_low_latency = False
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


class StrategyRegistryDiagnosticsTest(unittest.TestCase):
    def test_no_quant_cpp_runtime_conditions_run_in_regular_ci(self):
        def conditions_pass(quant_config) -> bool:
            checker = ConditionChecker("CudaNoQuantCppStrategy.check_conditions()")
            CudaNoQuantCppStrategy.check_conditions(
                checker, _config("no_quant_cpp", quant_config)
            )
            return checker.all_passed()

        self.assertTrue(conditions_pass(None))
        self.assertFalse(conditions_pass(CompressedW8A8Int8PerChannelQuantConfig()))

    def test_request_names_value_and_current_model_scope(self):
        with self.assertRaises(ValueError) as cm:
            StrategyRegistry().get_strategy(_config("grouped_fp4"))

        message = str(cm.exception)
        self.assertIn("MOE_STRATEGY='grouped_fp4'", message)
        self.assertIn("model scope 'test_model'", message)
        self.assertIn("generic fused-MoE factory", message)

    def test_explicit_strategy_does_not_fall_back_to_another_backend(self):
        registry = StrategyRegistry()
        fallback = MagicMock()
        fallback.strategy_name = "fallback"
        fallback.supported_moe_quant_method = None
        fallback.can_handle.return_value = True
        registry.register(fallback)

        with self.assertRaises(ValueError) as cm:
            registry.get_strategy(_config("requested"))

        fallback.can_handle.assert_not_called()
        self.assertIn("MOE_STRATEGY='requested'", str(cm.exception))

    def test_explicit_strategy_selects_only_its_registered_backend(self):
        registry = StrategyRegistry()
        requested = MagicMock()
        requested.strategy_name = "requested"
        requested.supported_moe_quant_method = None
        requested.can_handle.return_value = True
        requested.get_attributes.return_value.calculate_priority.return_value = 1
        fallback = MagicMock()
        fallback.strategy_name = "fallback"
        fallback.supported_moe_quant_method = None
        fallback.can_handle.return_value = True
        registry.register(fallback)
        registry.register(requested)

        self.assertIs(registry.get_strategy(_config("requested")), requested)
        fallback.can_handle.assert_not_called()

    def test_legacy_external_strategy_can_handle_fp8_fp4_auto_config(self):
        class LegacyStrategy:
            def can_handle(self, config):
                return config.moe_quant_method == "FP8_FP4"

            def get_attributes(self):
                return SimpleNamespace(calculate_priority=lambda: 7)

        strategy = LegacyStrategy()
        registry = StrategyRegistry()
        registry.register(strategy)
        legacy_config = SimpleNamespace(moe_quant_method="FP8_FP4")

        self.assertIs(registry.get_strategy(legacy_config), strategy)

    def test_named_external_strategy_can_claim_explicit_request(self):
        class LegacyStrategy:
            strategy_name = "external_strategy"

            def can_handle(self, config):
                return config.moe_strategy == "external_strategy"

            def get_attributes(self):
                return SimpleNamespace(calculate_priority=lambda: 7)

        strategy = LegacyStrategy()
        registry = StrategyRegistry()
        registry.register(strategy)
        legacy_config = SimpleNamespace(
            moe_strategy="external_strategy",
            moe_quant_method=None,
        )

        self.assertIs(registry.get_strategy(legacy_config), strategy)

    def test_unnamed_moe_strategy_cannot_claim_explicit_request(self):
        class ExternalMoeStrategy(MoeStrategy):
            def can_handle(self, config):
                return config.moe_strategy == "external_strategy"

            def get_attributes(self):
                return SimpleNamespace(calculate_priority=lambda: 7)

        strategy = ExternalMoeStrategy()
        registry = StrategyRegistry()
        registry.register(strategy)
        config = SimpleNamespace(
            moe_strategy="external_strategy",
            moe_quant_method=None,
        )

        with self.assertRaisesRegex(ValueError, "MOE_STRATEGY='external_strategy'"):
            registry.get_strategy(config)

    def test_explicit_request_never_falls_back_to_unnamed_strategy(self):
        for rejection in ("quant_method", "can_handle", "unknown_name"):
            with self.subTest(rejection=rejection):
                registry = StrategyRegistry()
                unnamed = SimpleNamespace(
                    can_handle=MagicMock(return_value=True),
                    get_attributes=lambda: SimpleNamespace(
                        calculate_priority=lambda: 100
                    ),
                )
                requested = MagicMock()
                requested.strategy_name = "mega_moe"
                requested.supported_moe_quant_method = "FP8_FP4"
                requested.can_handle.return_value = rejection != "can_handle"
                registry.register(unnamed)
                registry.register(requested)
                config = SimpleNamespace(
                    moe_strategy=(
                        "unknown" if rejection == "unknown_name" else "mega_moe"
                    ),
                    moe_quant_method=None if rejection == "quant_method" else "FP8_FP4",
                )
                with self.assertRaisesRegex(ValueError, "Requested MOE_STRATEGY"):
                    registry.get_strategy(config)
                unnamed.can_handle.assert_not_called()

    def test_legacy_shared_expert_dimensions_are_inferred_strictly(self):
        model_config = ModelConfig()
        model_config.moe_style = 2
        model_config.n_shared_experts = 0
        model_config.moe_inter_size = 128
        model_config.inter_size = 384

        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=ParallelismConfig(),
            moe_config=MoeConfig(),
        )

        self.assertEqual(config.n_shared_experts, 3)

    def test_invalid_legacy_shared_expert_dimensions_fail_closed(self):
        model_config = ModelConfig()
        model_config.moe_style = 2
        model_config.n_shared_experts = 0
        model_config.moe_inter_size = 128
        model_config.inter_size = 320

        with self.assertRaisesRegex(ValueError, "positive, divisible dimensions"):
            MoEConfigAdapter(
                model_config=model_config,
                parallelism_config=ParallelismConfig(),
                moe_config=MoeConfig(),
            )

    def test_prefill_capacity_is_not_limited_by_decode_concurrency(self):
        model_config = ModelConfig()
        model_config.max_seq_len = 4096
        moe_config = MoeConfig()
        moe_config.ll_num_max_token = 32

        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=ParallelismConfig(),
            moe_config=moe_config,
        )

        self.assertEqual(config.decode_max_tokens_per_rank, 32)
        self.assertEqual(config.prefill_max_tokens_per_rank, 4096)
        self.assertEqual(config.max_tokens_per_rank, 4096)

    def test_eplb_physical_expert_count_reaches_executor_contract(self):
        model_config = ModelConfig()
        model_config.expert_num = 64
        model_config.eplb_config.redundant_expert = 8
        parallelism_config = ParallelismConfig()
        parallelism_config.ep_size = 8
        parallelism_config.ep_rank = 3

        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=MoeConfig(),
        )

        self.assertEqual(config.expert_num, 64)
        self.assertEqual(config.physical_expert_num, 72)
        self.assertTrue(config.has_redundant_experts)
        self.assertEqual(config.n_local_experts, 9)
        self.assertEqual(config.local_expert_start, 27)
        self.assertEqual(config.local_expert_end, 36)

    def test_fp8_fp4_strategy_rejects_redundant_experts(self):
        config = SimpleNamespace(
            moe_strategy="mega_moe",
            moe_quant_method="FP8_FP4",
            ep_size=8,
            has_redundant_experts=True,
        )
        checker = ConditionChecker("CudaMegaMoeStrategy.check_conditions()")

        CudaMegaMoeStrategy.check_conditions(checker, config)

        self.assertFalse(checker.all_passed())

    def test_fp8_fp4_eplb_failure_explains_expert_counts(self):
        config = SimpleNamespace(
            moe_strategy="mega_moe",
            moe_quant_method="FP8_FP4",
            expert_num=64,
            physical_expert_num=72,
            has_redundant_experts=True,
            ep_size=8,
            world_size=8,
            tp_size=1,
        )

        with self.assertRaises(ValueError) as cm:
            StrategyRegistry().get_strategy(config)

        message = str(cm.exception)
        self.assertIn("do not support EPLB redundant experts", message)
        self.assertIn("logical_experts=64", message)
        self.assertIn("physical_experts=72", message)

    def test_local_loop_rejects_non_sm100_device(self):
        config = SimpleNamespace(ep_size=1, moe_quant_method="FP8_FP4")
        checker = ConditionChecker("LocalLoopExecutor.check_conditions()")
        deep_gemm = SimpleNamespace(fp8_fp4_gemm_nt=lambda: None)

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            LocalLoopExecutor.check_conditions(checker, config)

        self.assertFalse(checker.all_passed())

    def test_local_loop_rejects_missing_fp8_fp4_symbol(self):
        config = SimpleNamespace(ep_size=1, moe_quant_method="FP8_FP4")
        checker = ConditionChecker("LocalLoopExecutor.check_conditions()")

        with patch.dict(sys.modules, {"deep_gemm": SimpleNamespace()}), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            LocalLoopExecutor.check_conditions(checker, config)

        self.assertFalse(checker.all_passed())

    def test_supported_quant_method_collection_matches_membership(self):
        registry = StrategyRegistry()
        strategy = MagicMock()
        strategy.strategy_name = "requested"
        strategy.supported_moe_quant_method = {"FP8_FP4", "FP8_PER_BLOCK"}
        strategy.can_handle.return_value = True
        strategy.get_attributes.return_value.calculate_priority.return_value = 1
        registry.register(strategy)
        config = SimpleNamespace(
            moe_strategy="requested",
            moe_quant_method="FP8_FP4",
        )

        self.assertIs(registry.get_strategy(config), strategy)

    def test_empty_registry_reports_value_error_for_legacy_config(self):
        config = SimpleNamespace(
            moe_strategy="requested",
            moe_quant_method=None,
            ep_size=1,
            world_size=1,
            tp_size=1,
        )

        with self.assertRaisesRegex(ValueError, "MOE_STRATEGY='requested'"):
            StrategyRegistry().get_strategy(config)

    def test_batched_triton_does_not_consume_explicit_strategy(self):
        self.assertFalse(BatchedTritonStrategy().can_handle(_config("mega_moe")))

    def test_rocm_strategies_have_stable_public_names(self):
        expected = {
            RocmBf16PureTPStrategy: "no_quant_cpp",
            RocmEpNormalStrategy: "rocm_ep_normal",
            RocmEpLowLatencyStrategy: "rocm_ep_low_latency",
            RocmFp8PerBlockPureTPStrategy: "fp8_per_block_no_dp",
            RocmFp8PerChannelPureTPStrategy: "rocm_fp8_per_channel_no_dp",
            RocmMXFp4PureTPStrategy: "rocm_mxfp4_no_dp",
        }
        actual = {strategy: strategy().strategy_name for strategy in expected}

        self.assertEqual(actual, expected)
        self.assertTrue(set(actual.values()).issubset(MOE_STRATEGY_CHOICES))

    def test_all_in_tree_public_strategy_names_are_parser_choices(self):
        public_names = {
            getattr(module, class_name)().strategy_name
            for module in (cuda_strategies, rocm_strategies)
            for class_name in module.__all__
        }

        self.assertNotIn(None, public_names)
        self.assertTrue(public_names.issubset(MOE_STRATEGY_CHOICES))

    def test_factory_uses_class_name_when_strategy_name_is_empty(self):
        class UnnamedStrategy:
            strategy_name = None

            def create_router(self, config):
                return "router"

            def create_executor(self, config, weights):
                return "executor"

        registry = MagicMock()
        registry.get_strategy.return_value = UnnamedStrategy()
        config = SimpleNamespace(expert_num=8)
        with patch.object(FusedMoeFactory, "_registry", registry), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.factory.FusedMoe"
        ) as fused_moe:
            FusedMoeFactory().create_fused_moe(config, {})

        fused_moe.assert_called_once_with(
            "router",
            "executor",
            expert_num=8,
            strategy_name="UnnamedStrategy",
        )

    def test_package_import_does_not_eagerly_import_deep_gemm(self):
        code = """
import sys

class DeepGemmImportBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "deep_gemm" or fullname.startswith("deep_gemm."):
            raise ImportError("deep_gemm deliberately unavailable")
        return None

sys.meta_path.insert(0, DeepGemmImportBlocker())
import rtp_llm.models_py.modules.factory.fused_moe
assert not any(
    name == "deep_gemm" or name.startswith("deep_gemm.") for name in sys.modules
)
"""
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
