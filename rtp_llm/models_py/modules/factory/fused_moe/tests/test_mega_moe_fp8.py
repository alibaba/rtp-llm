import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.mega_moe_fp8 import (
    CudaMegaMoeFp8Strategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.group import (
    get_validated_world_ep_group,
)


class Checks:
    def __init__(self):
        self.values = []

    def check(self, value):
        self.values.append(bool(value))

    @property
    def passed(self):
        return all(self.values)


class MegaMoeFp8SelectionTest(unittest.TestCase):
    def config(self, **overrides):
        values = dict(
            moe_strategy="mega_moe_fp8",
            moe_quant_method="FP8_PER_BLOCK",
            ep_size=4,
            ep_rank=0,
            tp_size=1,
            world_size=4,
            world_rank=0,
            has_redundant_experts=False,
            enable_cuda_graph=False,
            swiglu_limit=0.0,
            hidden_size=4096,
            moe_inter_dim=1024,
            expert_num=512,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def check_executor(self, **overrides):
        checker = Checks()
        with patch.object(
            MoeConfigResolver, "get_quant_method", return_value="FP8_PER_BLOCK"
        ), patch.object(MoeConfigResolver, "is_bf16", return_value=True), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8.mega_moe_fp8_available",
            return_value=True,
        ):
            MegaMoeFp8Executor.check_conditions(checker, self.config(**overrides))
        return checker.passed

    def test_opt_in_only(self):
        for value, expected in [
            ("mega_moe_fp8", True),
            ("auto", False),
            ("mega_moe", False),
        ]:
            with self.subTest(strategy=value), patch.object(
                MoeConfigResolver, "get_quant_method", return_value="FP8_PER_BLOCK"
            ):
                checker = Checks()
                CudaMegaMoeFp8Strategy.check_conditions(
                    checker, self.config(moe_strategy=value)
                )
                self.assertEqual(checker.passed, expected)

    def test_supported_parallelism(self):
        self.assertTrue(self.check_executor())
        for invalid in [
            dict(swiglu_limit=1.0),
            dict(tp_size=2),
            dict(ep_size=1),
            dict(world_size=8),
            dict(world_rank=1),
            dict(has_redundant_experts=True),
            dict(enable_cuda_graph=True),
            dict(hidden_size=4000),
            dict(moe_inter_dim=1000),
            dict(expert_num=513),
        ]:
            with self.subTest(invalid=invalid):
                self.assertFalse(self.check_executor(**invalid))

    def test_world_group_validation(self):
        world = object()
        dist = SimpleNamespace(
            is_initialized=lambda: True,
            group=SimpleNamespace(WORLD=world),
            get_world_size=lambda group: 4,
            get_rank=lambda group: 0,
        )
        self.assertIs(get_validated_world_ep_group(self.config(), dist), world)
        with self.assertRaises(RuntimeError):
            get_validated_world_ep_group(self.config(ep_rank=1), dist)

    def test_registry_selects_explicit_backend(self):
        registry = StrategyRegistry()
        registry.register(CudaMegaMoeFp8Strategy())
        with patch.object(
            MoeConfigResolver, "get_quant_method", return_value="FP8_PER_BLOCK"
        ), patch.object(MoeConfigResolver, "is_bf16", return_value=True), patch(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8.mega_moe_fp8_available",
            return_value=True,
        ):
            self.assertEqual(
                registry.get_strategy(self.config()).strategy_name, "mega_moe_fp8"
            )

    def test_capacity_allows_oversized_single_request(self):
        from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
            MoEConfigAdapter,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
            mega_moe_fp8_capacity,
        )

        config = object.__new__(MoEConfigAdapter)
        config.model_config = SimpleNamespace(max_seq_len=30720)
        config.max_tokens_per_rank = 20000
        self.assertFalse(hasattr(config, "max_seq_len"))
        self.assertEqual(mega_moe_fp8_capacity(config), 30720)
        self.assertGreaterEqual(mega_moe_fp8_capacity(config), 24601)
        config.max_tokens_per_rank = 40000
        self.assertGreaterEqual(mega_moe_fp8_capacity(config), 40000)
        self.assertEqual(mega_moe_fp8_capacity(config) % 256, 0)


if __name__ == "__main__":
    unittest.main()
