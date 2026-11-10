"""Select the PPU FP8 DeepEP paths used by CP prefill and DP decode."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.models_py.modules.factory.fused_moe.ppu_impl.strategy.fp8_per_block import (
    PpuFp8PerBlockDpNormalDeepGemmStrategy,
    PpuFp8PerBlockEpLowLatencyDeepGemmStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)


class Fp8PdStrategyTest(TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(
            patch(
                "rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported",
                return_value=True,
            )
        )
        for module, symbol in (
            ("deepgemm_hybrid_executor", "has_deep_gemm_fp8_grouped"),
            ("deepgemm_masked_executor", "has_deep_gemm_fp8_grouped_masked"),
        ):
            self.patches.enter_context(
                patch(
                    "rtp_llm.models_py.modules.factory.fused_moe.ppu_impl.executors."
                    + module
                    + "."
                    + symbol,
                    return_value=True,
                )
            )
        self.registry = StrategyRegistry()
        self.registry.register(PpuFp8PerBlockDpNormalDeepGemmStrategy())
        self.registry.register(PpuFp8PerBlockEpLowLatencyDeepGemmStrategy())

    def config(self, *, low_latency=False, graph=False, tp_size=1, ep_size=8):
        return SimpleNamespace(
            moe_strategy="auto",
            moe_quant_method="FP8_PER_BLOCK",
            model_config=SimpleNamespace(
                model_type="glm_5",
                data_type="bf16",
                quant_config=SimpleNamespace(get_method=lambda: "FP8_PER_BLOCK"),
            ),
            moe_config=SimpleNamespace(
                use_all_gather=False, use_deepep_low_latency=low_latency
            ),
            tp_size=tp_size,
            ep_size=ep_size,
            enable_cuda_graph=graph,
        )

    def test_cp_prefill_selects_normal_hybrid(self):
        strategy = self.registry.get_strategy(self.config())
        self.assertIsInstance(strategy, PpuFp8PerBlockDpNormalDeepGemmStrategy)

    def test_dp_decode_selects_low_latency_masked_with_or_without_graph(self):
        for graph in (False, True):
            with self.subTest(graph=graph):
                strategy = self.registry.get_strategy(
                    self.config(low_latency=True, graph=graph)
                )
                self.assertIsInstance(
                    strategy, PpuFp8PerBlockEpLowLatencyDeepGemmStrategy
                )

    def test_unsupported_tp_and_single_rank_topologies_have_no_fp8_fallback(self):
        for low_latency in (False, True):
            for topology in ({"tp_size": 8}, {"ep_size": 1}):
                with self.subTest(low_latency=low_latency, **topology):
                    with self.assertRaisesRegex(ValueError, "No suitable MOE strategy"):
                        self.registry.get_strategy(
                            self.config(low_latency=low_latency, **topology)
                        )

    def test_normal_graph_is_rejected_before_host_count_execution(self):
        with self.assertRaisesRegex(ValueError, "No suitable MOE strategy"):
            self.registry.get_strategy(self.config(graph=True))


if __name__ == "__main__":
    main()
