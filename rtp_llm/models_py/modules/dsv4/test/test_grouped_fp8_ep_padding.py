"""Host-only tests for bounded eager EP exchanges; collectives/GEMMs are mocked."""

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv4.moe.strategies import grouped_fp8
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe import moe_layer


class GroupedFP8EPPaddingTest(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(mock.patch.dict(os.environ, {"DSV4_EP_CHECK_SIZES": "1"}))
        self.patches.enter_context(mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        self.group = object()
        self.patches.enter_context(mock.patch.object(grouped_fp8, "_ep_group", return_value=self.group))

    def strategy(self, budget=4, ep=4, rank=0, ll=False, decode=False):
        strategy = grouped_fp8.GroupedFP8Strategy.__new__(grouped_fp8.GroupedFP8Strategy)
        torch.nn.Module.__init__(strategy)
        strategy.cfg = MoeCfg(
            layer_id=0, dim=128, moe_inter_dim=128, n_routed_experts=8,
            n_activated_experts=2, swiglu_limit=10.0, ep_size=ep, ep_rank=rank,
            n_local_experts=8 // ep, local_expert_start=rank * (8 // ep),
            local_expert_end=(rank + 1) * (8 // ep), max_tokens_per_rank=budget,
            is_decode_role=decode,
        )
        strategy._ll_ok = ll
        strategy._captured_ns = set()
        return strategy

    @staticmethod
    def inputs(n, rank=0):
        x = torch.full((n, 128), rank + 1, dtype=torch.bfloat16)
        weights = torch.full((n, 2), 0.5, dtype=torch.float32)
        indices = torch.tensor([[0, 3]], dtype=torch.int64).expand(n, -1).contiguous()
        return x, weights, indices

    @staticmethod
    def padded(tensor, count, value):
        output = tensor.new_full((count, tensor.shape[1]), value)
        output[:tensor.shape[0]].copy_(tensor)
        return output

    def exchange(self, strategy, local_inputs, exchange_rows, peers=None):
        ep, rank = strategy.cfg.ep_size, strategy.cfg.ep_rank
        peers = peers or [local_inputs] * ep
        seen = []
        partials = []

        def gather(tensor, passed_ep, group):
            column = len(seen)
            self.assertEqual(passed_ep, ep)
            self.assertIs(group, self.group)
            self.assertEqual(tensor.shape[0], exchange_rows)
            value = -1 if column == 2 else 0
            expected = self.padded(local_inputs[column], exchange_rows, value)
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
            seen.append(tensor)
            return torch.cat([self.padded(peer[column], exchange_rows, value) for peer in peers])

        def local(x, weights, indices, expert_start):
            self.assertEqual(expert_start, strategy.cfg.local_expert_start)
            effective_weight = (weights * (indices >= 0)).sum(-1, keepdim=True)
            partial = (x.float() * effective_weight).to(torch.bfloat16)
            partials.append(partial)
            return partial

        def scatter(output, partial, group):
            self.assertIs(group, self.group)
            self.assertEqual(tuple(output.shape), (exchange_rows, 128))
            self.assertEqual(partial.shape[0], exchange_rows * ep)
            output.copy_(partial[rank * exchange_rows:(rank + 1) * exchange_rows] * ep)

        with mock.patch.object(grouped_fp8, "_all_gather_cat", side_effect=gather), \
                mock.patch.object(strategy, "_local_experts", side_effect=local), \
                mock.patch.object(strategy, "_assert_uniform_token_count") as check, \
                mock.patch.object(dist, "reduce_scatter_tensor", side_effect=scatter) as reduce, \
                mock.patch.object(grouped_fp8.F, "pad", wraps=grouped_fp8.F.pad) as pad:
            output = strategy(*local_inputs)
        check.assert_called_once_with(exchange_rows, self.group, local_inputs[0].device)
        reduce.assert_called_once()
        self.assertEqual(tuple(output.shape), tuple(local_inputs[0].shape))
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, local_inputs[0].float() * ep, rtol=0, atol=0)
        return seen, partials, pad.call_count

    def test_ragged_ranks_zero_one_two_and_budget_share_fixed_exchange(self):
        counts = (0, 1, 2, 4)
        peers = [self.inputs(n, rank) for rank, n in enumerate(counts)]
        for rank, n in enumerate(counts):
            with self.subTest(rank=rank, rows=n):
                seen, partials, padding_calls = self.exchange(
                    self.strategy(rank=rank), peers[rank], 4, peers)
                self.assertEqual(padding_calls, 3 if n < 4 else 0)
                for peer_rank, peer_n in enumerate(counts):
                    dummy = partials[0][peer_rank * 4 + peer_n:(peer_rank + 1) * 4]
                    self.assertEqual(torch.count_nonzero(dummy).item(), 0)
                if n == 4:
                    self.assertTrue(all(actual is original for actual, original in zip(seen, peers[rank])))

    def test_all_empty_ranks_still_exchange_budget_rows(self):
        _, partials, pads = self.exchange(self.strategy(), self.inputs(0), 4)
        self.assertEqual(pads, 3)
        self.assertEqual(tuple(partials[0].shape), (16, 128))
        self.assertEqual(torch.count_nonzero(partials[0]).item(), 0)

    def test_padding_gate_depends_on_config_not_runtime_rows(self):
        for budget in (0, -1, 257, 8192):
            for n in (0, 1, 2):
                with self.subTest(budget=budget, rows=n):
                    seen, _, pads = self.exchange(self.strategy(budget=budget), self.inputs(n), n)
                    self.assertEqual(pads, 0)
                    self.assertTrue(all(tensor.shape[0] == n for tensor in seen))

    def test_masked_limit_boundary_uses_total_exchange_rows(self):
        budget = grouped_fp8._MASKED_MAX_N // 4
        _, _, pads = self.exchange(self.strategy(budget=budget), self.inputs(1), budget)
        self.assertEqual(pads, 3)
        _, _, pads = self.exchange(self.strategy(budget=budget + 1), self.inputs(1), 1)
        self.assertEqual(pads, 0)

    def test_explicit_decode_large_budget_pads_small_ragged_batches(self):
        for budget in (257, 512):
            for n in (0, 1, 2, budget):
                with self.subTest(budget=budget, rows=n):
                    original = self.inputs(n)
                    seen, partials, pads = self.exchange(
                        self.strategy(budget=budget, decode=True), original, budget)
                    self.assertEqual(pads, 3 if n < budget else 0)
                    self.assertEqual(tuple(partials[0].shape), (budget * 4, 128))
                    if n == budget:
                        self.assertTrue(all(actual is source for actual, source in zip(seen, original)))

    def test_explicit_decode_nonpositive_budget_rejected_at_initialization(self):
        for budget in (0, -1):
            with self.subTest(budget=budget), \
                    mock.patch.object(torch.cuda, "is_available") as cuda_available, \
                    mock.patch.object(grouped_fp8, "_ll_buffer") as ll_buffer:
                cfg = self.strategy(budget=budget, decode=True).cfg
                with self.assertRaisesRegex(ValueError, "decode requires a positive.*max_tokens_per_rank"):
                    grouped_fp8.GroupedFP8Strategy(cfg)
                cuda_available.assert_not_called()
                ll_buffer.assert_not_called()

    def test_explicit_decode_large_budget_still_rejects_overflow(self):
        strategy = self.strategy(budget=257, decode=True)
        with mock.patch.object(grouped_fp8.F, "pad") as pad, \
                mock.patch.object(grouped_fp8, "_all_gather_cat") as gather:
            with self.assertRaisesRegex(RuntimeError, "258 tokens.*exchange capacity 257"):
                strategy(*self.inputs(258))
        pad.assert_not_called()
        gather.assert_not_called()

    def test_over_budget_rejected_before_padding_or_collectives(self):
        strategy = self.strategy()
        with mock.patch.object(grouped_fp8.F, "pad") as pad, \
                mock.patch.object(grouped_fp8, "_all_gather_cat") as gather, \
                mock.patch.object(strategy, "_assert_uniform_token_count") as check:
            with self.assertRaisesRegex(RuntimeError, "5 tokens.*exchange capacity 4"):
                strategy(*self.inputs(5))
        pad.assert_not_called()
        gather.assert_not_called()
        check.assert_not_called()

    def test_ll_inputs_size_diagnostic_and_return_are_unchanged(self):
        strategy = self.strategy(ll=True, decode=True)
        inputs = self.inputs(2)
        buffer = object()
        expected = torch.full((2, 128), 7, dtype=torch.bfloat16)
        with mock.patch.object(grouped_fp8, "_ll_buffer", return_value=(buffer, 16)), \
                mock.patch.object(strategy, "_local_experts_ll", return_value=expected) as local, \
                mock.patch.object(strategy, "_assert_uniform_token_count") as check, \
                mock.patch.object(grouped_fp8.F, "pad") as pad, \
                mock.patch.object(grouped_fp8, "_all_gather_cat") as gather:
            output = strategy(*inputs)
        local.assert_called_once_with(*inputs, buffer, 16)
        check.assert_called_once_with(2, self.group, inputs[0].device)
        torch.testing.assert_close(output, expected.float(), rtol=0, atol=0)
        pad.assert_not_called()
        gather.assert_not_called()

    def test_ll_zero_batch_fallback_is_not_changed(self):
        _, _, pads = self.exchange(self.strategy(ll=True, decode=True), self.inputs(0), 0)
        self.assertEqual(pads, 0)

    def test_ll_capacity_error_is_not_replaced_by_padding(self):
        strategy = self.strategy(ll=True)
        with mock.patch.object(grouped_fp8, "_ll_buffer", return_value=(object(), 16)), \
                mock.patch.object(strategy, "_assert_uniform_token_count"), \
                mock.patch.object(grouped_fp8.F, "pad") as pad:
            with self.assertRaisesRegex(RuntimeError, "low-latency buffer's 16"):
                strategy(*self.inputs(17))
        pad.assert_not_called()

    def test_capture_guard_sees_original_rows_before_padding_and_diagnostic(self):
        strategy = self.strategy(decode=True)
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True), \
                mock.patch.object(grouped_fp8.F, "pad") as pad, \
                mock.patch.object(grouped_fp8, "_all_gather_cat") as gather, \
                mock.patch.object(strategy, "_assert_uniform_token_count") as check:
            with self.assertRaisesRegex(RuntimeError, "capturing 1 rows.*up to 4"):
                strategy(*self.inputs(1))
        self.assertEqual(strategy._captured_ns, {1})
        pad.assert_not_called()
        gather.assert_not_called()
        check.assert_not_called()

    def test_capture_at_budget_does_not_allocate_padding(self):
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            strategy = self.strategy(decode=True)
            _, _, pads = self.exchange(strategy, self.inputs(4), 4)
        self.assertEqual(pads, 0)
        self.assertEqual(strategy._captured_ns, {4})

    def test_single_rank_path_is_not_padded_or_bounded(self):
        strategy = self.strategy(budget=1, ep=1)
        with mock.patch.object(strategy, "_local_experts", return_value=self.inputs(2)[0]) as local, \
                mock.patch.object(grouped_fp8.F, "pad") as pad:
            self.assertEqual(tuple(strategy(*self.inputs(0)).shape), (0, 128))
            self.assertEqual(tuple(strategy(*self.inputs(2)).shape), (2, 128))
        self.assertEqual(local.call_count, 1)
        pad.assert_not_called()

    def test_moe_passes_decode_role_to_selected_strategy_config(self):
        class CaptureStrategy(torch.nn.Module):
            routed_includes_shared = True

            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg

            def can_use_gate_pack_static(self, gate):
                return False

            def setup_weights(self, weights):
                pass

            def setup_runtime(self):
                pass

        for role_options, expected in (({}, False), ({"is_decode_role": False}, False),
                                       ({"is_decode_role": True}, True)):
            with self.subTest(role_options=role_options):
                gate = torch.nn.Module()
                gate.route_scale = 1.0
                with mock.patch.object(moe_layer, "Gate", return_value=gate), \
                        mock.patch.object(moe_layer, "_resolve_forced", return_value=(None, False)), \
                        mock.patch.object(moe_layer, "select_strategy", return_value=CaptureStrategy) as select:
                    layer = moe_layer.MoE(
                        layer_id=0, dim=128, moe_inter_dim=128, n_routed_experts=8,
                        n_activated_experts=2, n_shared_experts=1, score_func="sqrtsoftplus",
                        route_scale=1.0, swiglu_limit=10.0, n_hash_layers=0, vocab_size=16,
                        layer_weights={}, ep_size=4, max_tokens_per_rank=512, **role_options)
                cfg = select.call_args.args[0]
                self.assertIs(cfg, layer._strategy.cfg)
                self.assertIs(cfg.is_decode_role, expected)
                self.assertEqual(cfg.max_tokens_per_rank, 512)


if __name__ == "__main__":
    unittest.main()
