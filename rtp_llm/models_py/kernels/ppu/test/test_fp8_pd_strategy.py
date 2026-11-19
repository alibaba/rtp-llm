"""Select the PPU FP8 DeepEP paths used by CP prefill and DP decode."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.ppu_impl.routers.deepep_low_latency_router import (
    PpuDeepEpLowLatencyRouterFp8PerBlock,
)

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


class Fp8CombineSlotTest(TestCase):
    def test_only_disjoint_native_slot_can_skip_combine_copy(self):
        x = torch.empty(2, 2048, dtype=torch.bfloat16)
        dispatched = x.view(2, 1, 2048)
        slot = torch.empty_like(dispatched)
        ids = torch.zeros(2, 1, dtype=torch.int64)
        weights = torch.ones_like(ids, dtype=torch.float32)
        counts = torch.ones(2, dtype=torch.int32)
        alias_counts = slot.view(torch.int32).reshape(-1)[8:10]
        for label, offered, count, use_slot, expect_direct in (
            ("direct", slot, counts, True, True),
            ("executor_fallback", slot, counts, False, False),
            ("dispatch_alias", dispatched, counts, True, False),
            ("counts_alias", slot, alias_counts, True, False),
            ("wrong_shape", slot[:, :, :128], counts, True, False),
            ("unavailable", None, counts, True, False),
        ):
            with self.subTest(label=label):
                router = object.__new__(PpuDeepEpLowLatencyRouterFp8PerBlock)
                router.config = SimpleNamespace(tp_size=1, tp_rank=0, ep_size=2)
                router._buffer = Mock()
                router._num_topk, router._num_experts = 1, 4
                router._num_max_dispatch_tokens_per_rank = 64
                router._use_fp8_dispatch = router._async_finish = False
                router._return_recv_hook = router._zero_copy = False
                router._handle = router._combine_slot = None
                handle = object()
                router._buffer.low_latency_dispatch.return_value = (
                    dispatched,
                    count,
                    handle,
                    None,
                    None,
                )
                router._buffer.get_next_low_latency_combine_buffer.return_value = (
                    offered
                )
                router._buffer.low_latency_combine.return_value = (x, None, None)
                payload = router.prepare(x, None, None, weights, ids)
                dest = payload.expert_output_buffer if use_slot else None
                output = dest if dest is not None else torch.empty_like(dispatched)
                router.finalize(
                    CombineForwardPayload(output),
                    weights,
                    ids,
                    False,
                    {"original_num_tokens": 2},
                )
                kwargs = router._buffer.low_latency_combine.call_args.kwargs
                self.assertIs(kwargs["x"], output)
                self.assertIs(kwargs["zero_copy"], expect_direct)
                self.assertIsNone(router._combine_slot)


if __name__ == "__main__":
    main()
