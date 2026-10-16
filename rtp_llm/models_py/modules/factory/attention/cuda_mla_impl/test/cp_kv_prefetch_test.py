"""Independent layout reference and cross-layer CUDA event protocol tests."""

import random
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    cp_kv_prefetch as kv,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.layered_prefill_cache import (
    ContiguousPrefillCache,
)


class PrefixPlanTest(unittest.TestCase):
    def check_layout(self, prefixes, lengths, size, owner):
        # Scalar oracle: walk logical tokens in order and append to each rank.
        expected, packed, fresh = (
            [],
            [[] for _ in range(size)],
            [[] for _ in range(size)],
        )
        next_id = 1
        for p, q in zip(prefixes, lengths):
            histories = [[] for _ in range(size)]
            for position in range(p + q):
                rank = position // owner % size
                expected.append(next_id)
                if position < p:
                    histories[rank].append(next_id)
                else:
                    fresh[rank].append(next_id)
                next_id += 1
            capacity = max(map(len, histories), default=0)
            capacity = (capacity + owner - 1) // owner * owner
            for rank in range(size):
                packed[rank].extend(
                    histories[rank] + [-99] * (capacity - len(histories[rank]))
                )
        plans = [kv.prefix_plan(prefixes, lengths, size, r, owner) for r in range(size)]
        max_new = max(map(len, fresh))
        all_history = torch.tensor(sum(packed, []), dtype=torch.int64)
        all_new = torch.tensor(
            sum([x + [-88] * (max_new - len(x)) for x in fresh], []), dtype=torch.int64
        )
        for rank, plan in enumerate(plans):
            if sum(prefixes):
                output = all_history[plan["history_restore"]]
            else:
                output = torch.zeros(sum(lengths), dtype=torch.int64)
            output[plan["new_dest"]] = all_new[plan["new_restore"]]
            self.assertEqual(output.tolist(), expected)
            actual = [x for x in packed[rank] if x != -99]
            self.assertEqual(plan["history_actual"], len(actual))
            self.assertEqual(
                all_history[
                    rank * plan["history_padded"] + plan["history_scatter"]
                ].tolist(),
                actual,
            )
            logical_new = []
            start = 0
            for p, q in zip(prefixes, lengths):
                logical_new.extend(expected[start + p : start + p + q])
                start += p + q
            self.assertEqual(
                [logical_new[i] for i in plan["new_local_rows"]], fresh[rank]
            )

    def test_partial_owners_empty_requests_and_zero_owned_ranks(self):
        cases = [
            ([1, 511, 4097, 0], [6, 1025, 13, 0], 8, 512),
            ([0, 64, 513], [257, 0, 1], 8, 512),
            ([0, 0], [1, 0], 8, 512),
            ([13, 0], [0, 0], 3, 7),
        ]
        for args in cases:
            with self.subTest(args=args):
                self.check_layout(*args)

    def test_random_layouts_against_scalar_reference(self):
        rng = random.Random(4721)
        for _ in range(100):
            batch, size, owner = (
                rng.randint(1, 5),
                rng.randint(1, 8),
                rng.randint(1, 17),
            )
            self.check_layout(
                [rng.randint(0, 200) for _ in range(batch)],
                [rng.randint(0, 70) for _ in range(batch)],
                size,
                owner,
            )

    def test_invalid_plan_rejected(self):
        for args in [
            ([], [], 1, 0, 1),
            ([1], [], 1, 0, 1),
            ([-1], [1], 1, 0, 1),
            ([1], [1], 0, 0, 1),
            ([1], [1], 1, 1, 1),
            ([1], [1], 1, 0, 0),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                kv.prefix_plan(*args)


class PipelineProtocolTest(unittest.TestCase):
    def setUp(self):
        self.compute = Mock(name="compute")
        self.producer = Mock(name="producer")
        for target, kwargs in [
            ("current_stream", dict(return_value=self.compute)),
            ("stream", dict(side_effect=lambda _: nullcontext())),
            ("Event", dict(side_effect=lambda: Mock(name="event"))),
        ]:
            p = patch.object(torch.cuda, target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(
            kv,
            "all_gather_kv_overlap",
            side_effect=lambda x, _g, *, output: output.zero_(),
        )
        self.gather = p.start()
        self.addCleanup(p.stop)
        p = patch.object(
            kv.rtp_llm_ops,
            "cp_gather_and_upconvert_fp8_kv_cache_v2",
            side_effect=lambda _s, out, *_a: out.zero_(),
        )
        p.start()
        self.addCleanup(p.stop)
        op = SimpleNamespace(
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            attn_inputs=SimpleNamespace(prefix_lengths=torch.tensor([1])),
            cp_info=SimpleNamespace(prefill_actual_input_lengths_cpu=torch.tensor([1])),
            _allocate_fused_kv=lambda: torch.empty(2, 576, dtype=torch.bfloat16),
        )
        self.cache = SimpleNamespace(
            original_slots=torch.tensor([1]),
            slots=torch.tensor([1]),
            backing=[None] * 4,
            host_layers={},
            prefetched_layer=Mock(
                return_value=(
                    torch.zeros(1, 64, 656, dtype=torch.uint8),
                    op.block_table,
                    None,
                )
            ),
        )
        self.fmha = SimpleNamespace(
            fmha_impl=op,
            _kv_overlap_stream=self.producer,
            _cp_size=1,
            _cp_rank=0,
            _kv_owner_tokens_per_block=64,
        )
        self.pipeline = kv.CrossLayerKvPrefetch(self.fmha, self.cache)
        self.fmha._cross_layer_prefetch = self.pipeline

    def test_independent_buffers_delayed_moe_fence_and_ring_reuse(self):
        p = self.pipeline
        self.assertNotEqual(p.buffers[0].data_ptr(), p.buffers[1].data_ptr())
        p.prefetch(1)
        self.assertEqual(self.producer.wait_stream.call_count, 2)
        self.producer.wait_stream.assert_called_with(self.compute)
        self.compute.wait_event.assert_not_called()
        self.compute.wait_stream.assert_not_called()
        first_ready = p.ready[1]
        p.wait_before_moe()
        self.compute.wait_event.assert_called_once_with(first_ready)
        p.release(1)
        p.prefetch(3)
        self.producer.wait_event.assert_called_once_with(p.free[1])
        self.assertIsNot(p.ready[1], first_ready)
        p.wait_before_moe()
        self.assertIs(self.compute.wait_event.call_args.args[0], p.ready[1])
        self.assertEqual(self.gather.call_count, 2)
        p.prefetch(4)
        self.assertEqual(self.gather.call_count, 2)
        self.assertEqual(p.owners[1], 3)
        p.close()
        self.compute.wait_stream.assert_called_once_with(self.producer)
        self.assertIsNone(self.fmha._cross_layer_prefetch)

    def test_host_source_ready_is_joined_on_producer_only(self):
        ready = Mock(name="h2d_ready")
        self.cache.prefetched_layer.return_value = (
            torch.zeros(1, 64, 656, dtype=torch.uint8),
            self.fmha.fmha_impl.block_table,
            ready,
        )
        self.pipeline.prefetch(1)
        self.producer.wait_event.assert_called_once_with(ready)
        self.compute.wait_event.assert_not_called()

    def test_next_history_is_submitted_before_current_q_and_consumption(self):
        p = self.pipeline
        op = self.fmha.fmha_impl
        op._pack_sharded_kv_cache = lambda _cache, out: out.zero_()
        op.sharded_kv_restore_indices = torch.arange(2)
        _, ready = p.prepare(0, None)
        self.assertIs(ready, p.ready[0])
        self.cache.prefetched_layer.assert_called_once_with(1, op.block_table)
        self.assertEqual(self.gather.call_count, 2)
        p.consume(0)
        self.assertEqual(self.gather.call_count, 2)
        self.compute.wait_event.assert_not_called()
        self.compute.wait_stream.assert_not_called()

    def test_consumer_requires_matching_owner(self):
        with self.assertRaisesRegex(RuntimeError, "owner mismatch"):
            self.pipeline.consume(1)
        with self.assertRaisesRegex(RuntimeError, "not prefetched"):
            self.pipeline.prepare(1, None)

    def test_layerwise_dma_policy_does_not_disable_moe_nccl_fence(self):
        cache = object.__new__(ContiguousPrefillCache)
        cache.active = cache.window_submitted = cache.layerwise = True
        cache.device = torch.device("cpu")
        cache.kv_prefetch = self.pipeline
        cache._timing_event = Mock()
        self.pipeline.prefetch(1)
        cache.wait_prefetch_complete()
        self.compute.wait_event.assert_called_once_with(self.pipeline.ready[1])
        self.compute.wait_stream.assert_not_called()
        self.assertFalse(cache.window_open)

    def test_future_cache_slot_owner_is_checked_without_mutation(self):
        cache = object.__new__(ContiguousPrefillCache)
        cache.host_layers = {1: 0}
        cache.layerwise = True
        cache.slot_owners = [1]
        cache.ready_layers = {1}
        cache.staging, cache.ready = [object()], [object()]
        cache.block_table = object()
        self.assertEqual(
            cache.prefetched_layer(1, None),
            (cache.staging[0], cache.block_table, cache.ready[0]),
        )
        self.assertEqual(cache.slot_owners, [1])
        cache.slot_owners = [3]
        with self.assertRaisesRegex(RuntimeError, "no ready staging slot"):
            cache.prefetched_layer(1, None)


if __name__ == "__main__":
    unittest.main()
