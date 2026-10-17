import os
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch
import torch.nn as nn
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_fused_moe import (
    Sm120FusedMoeExecutor,
    is_pure_cp,
    local_expert_config,
    mask_to_local_experts,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_fp4 import (
    CudaGroupedFp4Strategy,
    CudaLocalLoopStrategy,
    CudaMegaMoeSEStrategy,
    CudaMegaMoeStrategy,
    CudaSm120DecodeStrategy,
    CudaSm120FusedMoeStrategy,
    CudaSm120GroupedFp4Strategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
    StrategyRegistry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
    ChunkedFp8Fp4MoeLayer,
    _requires_synchronized_chunks,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _mega_moe_unavailable_reason,
)

_EXECUTORS = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors"


def _config(ep_size=1, ep_rank=0, strategy="auto", **topology):
    return Fp8Fp4MoeRuntimeConfig(
        layer_id=0,
        hidden_size=512,
        moe_inter_dim=256,
        expert_num=16,
        moe_k=4,
        n_shared_experts=1,
        swiglu_limit=10.0,
        ep_size=ep_size,
        ep_rank=ep_rank,
        max_tokens_per_rank=128,
        moe_strategy=strategy,
        moe_w1_layout="gate_up",
        **topology,
    )


def _registry():
    registry = StrategyRegistry()
    for strategy in (
        CudaMegaMoeSEStrategy(),
        CudaMegaMoeStrategy(),
        CudaGroupedFp4Strategy(),
        CudaLocalLoopStrategy(),
        CudaSm120FusedMoeStrategy(),
        CudaSm120DecodeStrategy(),
        CudaSm120GroupedFp4Strategy(),
    ):
        registry.register(strategy)
    return registry


class _Sm120Runtime:
    """Patch the runtime probes as seen on an SM120 host with FlashInfer."""

    def __enter__(self):
        self._patches = [
            patch(f"{_EXECUTORS}.mega_moe._mega_moe_available", return_value=False),
            patch(
                f"{_EXECUTORS}.mega_moe_se._mega_moe_se_available", return_value=False
            ),
            patch(
                f"{_EXECUTORS}.grouped_fp4._has_fp8_fp4_grouped_kernel",
                return_value=False,
            ),
            patch(
                f"{_EXECUTORS}.local_loop._has_local_loop_kernel", return_value=False
            ),
            patch(
                f"{_EXECUTORS}.sm120_grouped_fp4.sm120_grouped_fp4_available",
                return_value=True,
            ),
            patch(
                f"{_EXECUTORS}.sm120_fused_moe.sm120_grouped_fp4_available",
                return_value=True,
            ),
            patch(f"{_EXECUTORS}.sm120_decode.is_sm120", return_value=True),
            patch.object(torch.cuda, "is_available", return_value=True),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


class Sm120StrategySelectionTest(unittest.TestCase):
    def test_single_rank_auto_selects_sm120_grouped_fp4(self):
        with _Sm120Runtime():
            strategy = _registry().get_strategy(_config())
        self.assertIsInstance(strategy, CudaSm120GroupedFp4Strategy)
        self.assertEqual(strategy.strategy_name, "grouped_fp4")

    def test_explicit_grouped_fp4_resolves_to_sm120_variant(self):
        with _Sm120Runtime():
            strategy = _registry().get_strategy(_config(strategy="grouped_fp4"))
        self.assertIsInstance(strategy, CudaSm120GroupedFp4Strategy)

    def test_ep_auto_selects_sm120_fused_moe(self):
        with _Sm120Runtime():
            strategy = _registry().get_strategy(_config(ep_size=4))
        self.assertIsInstance(strategy, CudaSm120FusedMoeStrategy)

    def test_auto_never_selects_sm120_decode(self):
        with _Sm120Runtime():
            self.assertFalse(CudaSm120DecodeStrategy().can_handle(_config(ep_size=4)))

    def test_explicit_sm120_decode_on_dp_ep(self):
        with _Sm120Runtime():
            strategy = _registry().get_strategy(
                _config(ep_size=4, strategy="sm120_decode")
            )
        self.assertIsInstance(strategy, CudaSm120DecodeStrategy)

    def test_sm120_decode_rejects_context_parallel(self):
        config = _config(
            ep_size=4,
            strategy="sm120_decode",
            moe_cp_enabled=True,
            moe_cp_size=4,
            moe_tp_size=4,
        )
        with _Sm120Runtime(), self.assertRaisesRegex(
            ValueError, "MOE_STRATEGY='sm120_decode'"
        ):
            _registry().get_strategy(config)

    def test_sm120_strategies_do_not_use_mega_gate_pack(self):
        for strategy in (
            CudaSm120GroupedFp4Strategy(),
            CudaSm120FusedMoeStrategy(),
            CudaSm120DecodeStrategy(),
        ):
            router = strategy.get_attributes().get_router_class()
            self.assertEqual(router.__name__, "Fp8Fp4Router")

    def test_mega_requires_datacenter_blackwell(self):
        fake_deep_gemm = SimpleNamespace(fp8_fp4_mega_moe=object())
        with mock.patch.dict(
            "sys.modules", {"deep_gemm": fake_deep_gemm}
        ), mock.patch.object(
            torch.distributed, "is_initialized", return_value=True
        ), mock.patch.object(
            torch.distributed, "get_world_size", return_value=4
        ), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(12, 0)
        ):
            reason = _mega_moe_unavailable_reason()
        self.assertIsNotNone(reason)
        self.assertIn("not SM10x", reason)


class Sm120ExecutorContractTest(unittest.TestCase):
    def test_local_expert_config_describes_only_local_experts(self):
        config = _config(ep_size=4, ep_rank=2)
        local = local_expert_config(config)
        self.assertEqual(local.n_routed_experts, 4)
        self.assertEqual(local.expert_num, 4)
        self.assertEqual(local.ep_size, 1)
        self.assertEqual(local.local_expert_start, 0)
        self.assertEqual(config.local_expert_start, 8)
        self.assertEqual(config.ep_size, 4)

    def test_mask_to_local_experts_zeroes_remote_slots(self):
        config = _config(ep_size=4, ep_rank=1)
        weights = torch.tensor([[0.5, 0.25, 0.25]])
        indices = torch.tensor([[4, 9, 7]])
        local_weights, local_ids = mask_to_local_experts(config, weights, indices)
        torch.testing.assert_close(local_weights, torch.tensor([[0.5, 0.0, 0.25]]))
        self.assertEqual(local_ids.tolist(), [[0, 3, 3]])

    def test_pure_cp_requires_matching_cp_tp_and_ep(self):
        self.assertTrue(
            is_pure_cp(
                _config(ep_size=8, moe_cp_enabled=True, moe_cp_size=8, moe_tp_size=8)
            )
        )
        self.assertFalse(is_pure_cp(_config(ep_size=8)))
        self.assertFalse(
            is_pure_cp(
                _config(ep_size=8, moe_cp_enabled=True, moe_cp_size=4, moe_tp_size=4)
            )
        )

    def test_world_collective_executor_requests_synchronized_chunks(self):
        executor = SimpleNamespace(requires_synchronized_chunk_schedule=True)
        fused_moe = SimpleNamespace(fused_experts=executor)
        layer = nn.Module()
        layer.strategy_name = "sm120_fused_moe"
        layer._moe = SimpleNamespace(fused_moe=fused_moe)
        self.assertTrue(_requires_synchronized_chunks(layer))
        executor.requires_synchronized_chunk_schedule = False
        self.assertFalse(_requires_synchronized_chunks(layer))
        layer.strategy_name = "mega_moe"
        self.assertTrue(_requires_synchronized_chunks(layer))

    def test_decode_executor_is_single_round(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_decode import (
            Sm120DecodeExecutor,
        )

        self.assertIs(Sm120DecodeExecutor.single_round_dispatch, True)
        layer = ChunkedFp8Fp4MoeLayer.__new__(ChunkedFp8Fp4MoeLayer)
        nn.Module.__init__(layer)
        layer.max_tokens_per_rank = 4
        layer.chunking_enabled = True
        layer._is_decode_role = False
        executor = SimpleNamespace(single_round_dispatch=False)
        layer._moe = SimpleNamespace(fused_moe=SimpleNamespace(fused_experts=executor))
        self.assertTrue(layer._should_chunk(9))
        # Per-rank chunk counts would differ across DP ranks and mispair the
        # executor's WORLD collectives; single-round executors never chunk.
        executor.single_round_dispatch = True
        self.assertFalse(layer._should_chunk(9))

    def test_dp_fused_capture_requires_synchronized_decode_strategy(self):
        executor = SimpleNamespace(cfg=SimpleNamespace(dp_size=4))
        with patch.object(
            torch.distributed, "is_initialized", return_value=True
        ), patch(
            f"{_EXECUTORS}.sm120_fused_moe.validate_world_ep_topology",
            return_value=(4, 0),
        ), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "moe_strategy=sm120_decode"):
                Sm120FusedMoeExecutor._forward_world_collective(
                    executor, None, None, None
                )

    def test_return_precision_preserves_fp32_partials_before_transport(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_decode import (
            Sm120DecodeExecutor,
            _return_mode,
        )

        rows, hidden, topk = 2, 128, 1
        x_end, scale_end, weight_end = (
            hidden,
            hidden + hidden // 32,
            hidden + hidden // 32 + 4 * topk,
        )
        payload = torch.zeros((rows, weight_end + 4 * topk), dtype=torch.uint8)
        x = torch.zeros((rows, hidden), dtype=torch.bfloat16)
        partial = torch.full((rows, hidden), 1.001, dtype=torch.float32)
        executor = SimpleNamespace(
            cfg=SimpleNamespace(local_expert_start=0, n_local_experts=1),
            _experts=lambda *args, **kwargs: partial,
        )
        for mode, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"DSV4_MOE_RETURN_DTYPE": mode}
            ):
                returned = Sm120DecodeExecutor._compute_received(
                    executor, payload, rows, x, x_end, scale_end, weight_end
                )
                self.assertEqual(returned.dtype, dtype)
                torch.testing.assert_close(returned, partial.to(dtype), rtol=0, atol=0)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_return_mode(), "mxfp8")
            self.assertEqual(_return_mode(fixed_ep=True), "bf16")
        with patch.dict(os.environ, {"DSV4_MOE_RETURN_DTYPE": "invalid"}):
            with self.assertRaises(ValueError):
                _return_mode()


if __name__ == "__main__":
    unittest.main()
