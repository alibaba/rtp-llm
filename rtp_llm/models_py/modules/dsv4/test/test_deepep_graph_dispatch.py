import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPMode, DeepEPWrapper
from rtp_llm.platforms.ppu.models.dsv4.ppu_moe_config import PpuMoeConfig as MoeCfg
from rtp_llm.platforms.ppu.models.dsv4.ppu_legacy_deepep import (
    PpuLegacyDeepEPStrategy,
    _select_ppu_grouped_fp4_capacity,
)


class _FakeBuffer:
    def __init__(self) -> None:
        self.num_worst_tokens = None

    def get_dispatch_layout(self, indices, _num_experts):
        rows = indices.size(0)
        return (
            torch.zeros(8, dtype=torch.int32),
            None,
            torch.zeros(256, dtype=torch.int32),
            torch.zeros(rows, 8, dtype=torch.bool),
            None,
        )

    def dispatch(self, x, *_args, num_worst_tokens=0, **_kwargs):
        self.num_worst_tokens = num_worst_tokens
        rows = num_worst_tokens or x.size(0)
        recv_x = torch.zeros(rows, x.size(1), dtype=x.dtype)
        recv_idx = torch.full((rows, 8), -1, dtype=torch.int64)
        recv_weights = torch.zeros(rows, 8, dtype=torch.float32)
        return recv_x, recv_idx, recv_weights, [], object(), None

    def combine(self, y_local, _handle):
        return y_local[:1].to(torch.bfloat16), None, None


class _FakeLowLatencyBuffer:
    def __init__(self) -> None:
        self.dispatch_args = None
        self.combine_args = None
        self.combine_buffer = torch.empty(32, 128, 4, dtype=torch.bfloat16)

    def low_latency_dispatch(self, **kwargs):
        self.dispatch_args = kwargs
        expert_x = torch.zeros(32, 128, 2, dtype=torch.uint8)
        expert_scale = torch.zeros(32, 128, 1, dtype=torch.uint16)
        expert_counts = torch.ones(32, dtype=torch.int32)
        return (expert_x, expert_scale), expert_counts, "handle", None, None

    def get_next_low_latency_combine_buffer(self, handle):
        if handle != "handle":
            raise ValueError("unexpected handle")
        return self.combine_buffer

    def low_latency_combine(
        self,
        x,
        topk_idx,
        topk_weights,
        handle,
        zero_copy=False,
        async_finish=False,
        return_recv_hook=False,
    ):
        self.combine_args = {
            "x": x,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "handle": handle,
            "zero_copy": zero_copy,
            "async_finish": async_finish,
            "return_recv_hook": return_recv_hook,
        }
        return torch.full((1, 4), 3, dtype=torch.bfloat16), None, None


class _FakeLocal:
    def _forward_into_buf(self, x, _weights, _indices, **_kwargs):
        return torch.zeros(x.size(0), x.size(1), dtype=torch.float32)


def _strategy(max_tokens_per_rank=1):
    strategy = PpuLegacyDeepEPStrategy.__new__(PpuLegacyDeepEPStrategy)
    torch.nn.Module.__init__(strategy)
    strategy.cfg = MoeCfg(
        layer_id=0,
        dim=4,
        moe_inter_dim=8,
        n_routed_experts=256,
        n_activated_experts=6,
        swiglu_limit=1.0,
        ep_size=8,
        ep_rank=0,
        n_local_experts=32,
        local_expert_start=0,
        local_expert_end=32,
        max_tokens_per_rank=max_tokens_per_rank,
    )
    strategy._compute_local = lambda recv_x, *_args, **_kwargs: torch.zeros_like(
        recv_x, dtype=torch.float32
    )
    return strategy


class DeepEPGraphDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = _FakeBuffer()
        self.saved_instance = DeepEPWrapper._instance
        DeepEPWrapper._instance = SimpleNamespace(
            mode=DeepEPMode.NORMAL, buffer=self.buffer
        )
        self.x = torch.zeros(1, 4, dtype=torch.bfloat16)
        self.weights = torch.ones(1, 6, dtype=torch.float32)
        self.indices = torch.arange(6, dtype=torch.int64).view(1, 6)

    def tearDown(self) -> None:
        DeepEPWrapper._instance = self.saved_instance
        os.environ.pop("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD", None)

    def test_capture_uses_fixed_worst_case_receive_capacity(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            out = _strategy()(self.x, self.weights, self.indices)

        self.assertEqual(self.buffer.num_worst_tokens, 8)
        self.assertEqual(tuple(out.shape), (1, 4))

    def test_capture_capacity_is_common_when_local_dp_batches_differ(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            _strategy(max_tokens_per_rank=2)(self.x, self.weights, self.indices)

        self.assertEqual(self.buffer.num_worst_tokens, 16)

    def test_graph_warmup_syncs_and_uses_capture_shape(self) -> None:
        os.environ["RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"] = "1"
        with patch("torch.cuda.is_available", return_value=False), patch(
            "rtp_llm.platforms.ppu.models.dsv4.ppu_legacy_deepep.sync_cuda_graph_warmup_ranks"
        ) as sync:
            _strategy()(self.x, self.weights, self.indices)

        self.assertEqual(self.buffer.num_worst_tokens, 8)
        self.assertEqual(
            [call.args[0] for call in sync.call_args_list],
            ["deepep_before_dispatch", "deepep_after_combine"],
        )

    def test_eager_keeps_dynamic_receive_path(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            _strategy()(self.x, self.weights, self.indices)

        self.assertEqual(self.buffer.num_worst_tokens, 0)

    def test_grouped_fp4_eager_capacity_grows_without_truncation(self) -> None:
        self.assertEqual(
            _select_ppu_grouped_fp4_capacity(128, [17, 129, 7], 32, fixed_shape=False),
            132,
        )
        self.assertEqual(
            _select_ppu_grouped_fp4_capacity(128, [512, 1], 32, fixed_shape=False),
            512,
        )

    def test_grouped_fp4_graph_capacity_stays_fixed(self) -> None:
        self.assertEqual(
            _select_ppu_grouped_fp4_capacity(128, [1024], 32, fixed_shape=True),
            128,
        )

    def test_grouped_fp4_eager_capacity_requires_cpu_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "CPU expert counts"):
            _select_ppu_grouped_fp4_capacity(128, [], 32, fixed_shape=False)

    def test_low_latency_delegates_to_public_full_slot_adapter(self) -> None:
        buffer = _FakeLowLatencyBuffer()
        DeepEPWrapper._instance = SimpleNamespace(
            mode=DeepEPMode.LOW_LATENCY,
            buffer=buffer,
            ll_num_max_token_per_rank=4,
            use_accl_ep=True,
        )
        strategy = _strategy(max_tokens_per_rank=4)
        strategy = PpuLegacyDeepEPStrategy(strategy.cfg, options={})
        for name in ("_ppu_w13", "_ppu_s13", "_ppu_w2", "_ppu_s2"):
            setattr(strategy, name, torch.empty(0))
        expected = torch.full((1, 4), 3, dtype=torch.float32)
        # The public platform owns dispatch/compute/combine and full-slot
        # alias handling, exercised by test_mxfp4_masked on a real PPU.
        # This CPU test verifies the legacy strategy's forwarding contract.
        with patch(
            "rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency.low_latency_mxfp4_moe",
            return_value=expected,
        ) as execute:
            out = strategy(self.x, self.weights, self.indices)

        self.assertIs(out, expected)
        execute.assert_called_once()
        args, kwargs = execute.call_args
        self.assertIs(args[0], buffer)
        self.assertIs(args[1], self.x)
        self.assertEqual(tuple(args[2].shape), (1, 8))
        self.assertEqual(tuple(args[3].shape), (1, 8))
        self.assertTrue(torch.equal(args[2][:, :6], self.weights))
        self.assertTrue(torch.equal(args[3][:, :6], self.indices))
        self.assertTrue(torch.equal(args[2][:, 6:], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(args[3][:, 6:], torch.full((1, 2), -1)))
        self.assertIs(args[4][0], strategy._ppu_w13)
        self.assertIs(args[4][1], strategy._ppu_s13)
        self.assertIs(args[5][0], strategy._ppu_w2)
        self.assertIs(args[5][1], strategy._ppu_s2)
        self.assertEqual(
            kwargs,
            {
                "num_experts": 256,
                "max_dispatch_tokens": 4,
                "expected_m": 1,
                "swiglu_limit": 1.0,
            },
        )

    def test_normal_strategy_rejects_low_latency_dispatch(self) -> None:
        DeepEPWrapper._instance = SimpleNamespace(
            mode=DeepEPMode.LOW_LATENCY,
            buffer=_FakeLowLatencyBuffer(),
            ll_num_max_token_per_rank=4,
            use_accl_ep=True,
        )
        with self.assertRaisesRegex(RuntimeError, "requires normal-mode dispatch"):
            _strategy()(self.x, self.weights, self.indices)


if __name__ == "__main__":
    unittest.main()
