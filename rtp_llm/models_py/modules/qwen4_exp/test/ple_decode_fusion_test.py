"""Exact SM90 decode tail, eager validity and Graph replay contracts."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple import ple_conv_triton
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_decode_plan_triton import (
    try_plan_decode_cache,
)


class TailFixture:
    """Inject real-shaped post-gate BF16 tensors into production decode_step."""

    def __init__(self, batch):
        kw = {"device": "cuda", "dtype": torch.bfloat16}
        self.gated = torch.randn(batch, 1, 10240, **kw)
        self.normed = torch.randn(batch, 1, 10240, **kw)
        self.initial = torch.randn(batch, 9, 10272, **kw)[..., :10240]
        self.conv_weight = torch.randn(10240, 1, 4, **kw) * 0.1
        self.conv_dilation = 3
        self.hc_hidden_size = 10240

    def _gated_values(self, *_args, **_kwargs):
        return self.gated, self.normed

    def call(self):
        return Qwen4ExpPLELayer.decode_step(self, self.gated, None, self.initial)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PLEDecodeFusionTest(unittest.TestCase):
    def setUp(self):
        if torch.version.hip or torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 qualification")
        torch.manual_seed(31914)

    def assert_exact(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(
                torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
            )

    def test_tail_output_and_all_nine_state_rows(self):
        flag = "QWEN4_PLE_DECODE_TAIL"
        for batch in (1, 16, 32, 64, 65, 128, 256):
            with self.subTest(batch=batch):
                fixture = TailFixture(batch)
                before = fixture.initial.clone()
                with patch.dict(os.environ):
                    os.environ.pop(flag, None)
                    with patch.object(ple_conv_triton, "try_ple_conv", side_effect=AssertionError):
                        expected = fixture.call()
                with patch.dict(os.environ, {flag: "1"}), patch.object(
                    ple_conv_triton, "try_ple_conv", wraps=ple_conv_triton.try_ple_conv
                ) as launch:
                    actual = fixture.call()
                    self.assertEqual(launch.call_count, 1)
                # Independently confirm this geometry qualifies; a fallback
                # inside decode_step must not satisfy the parity assertion.
                direct = ple_conv_triton.try_ple_conv(
                    fixture.gated,
                    fixture.normed,
                    fixture.initial,
                    fixture.conv_weight,
                    3,
                    roll_state=True,
                )
                self.assertIsNotNone(direct)
                self.assert_exact(actual, expected)
                self.assert_exact(direct, expected)
                self.assert_exact((fixture.initial,), (before,))

    def test_tail_graph_reads_new_values(self):
        fixture = TailFixture(3)
        flag = "QWEN4_PLE_DECODE_TAIL"
        with patch.dict(os.environ, {flag: "1"}):
            fixture.call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = fixture.call()
        for _ in range(2):
            fixture.initial.normal_()
            fixture.normed.normal_()
            fixture.gated.normal_()
            with patch.dict(os.environ, {flag: "0"}):
                expected = fixture.call()
            graph.replay()
            self.assert_exact(actual, expected)

    def test_cache_plan_cross_page_independent_tables(self):
        lengths = torch.tensor([63, 64, 65, 128], device="cuda", dtype=torch.int32)
        ids = torch.tensor([19, 23, 29, 31], device="cuda", dtype=torch.int32)
        s = torch.tensor(
            [[5, 0, 8, 0, 9, 0], [2, 0, 3, 0, 4, 0],
             [1, 0, 6, 0, 7, 0], [8, 0, 5, 0, 1, 0]],
            device="cuda", dtype=torch.int32,
        )[:, ::2]
        c = torch.tensor(
            [[11, 17, 19], [13, 18, 20], [21, 7, 9], [6, 10, 3]],
            device="cuda", dtype=torch.int32,
        )
        state_pool = torch.randn(11, 9, 10240, device="cuda", dtype=torch.bfloat16)
        context_pool = torch.arange(44, device="cuda", dtype=torch.int64).view(22, 2)
        before = state_pool.clone(), context_pool.clone()

        def plan(sequence_lengths=lengths):
            return try_plan_decode_cache(
                sequence_lengths, ids, s, c, state_pool, context_pool, 64
            )

        actual = plan()
        self.assertIsNotNone(actual)
        self.assertEqual(actual.state_read.tolist(), [5, 2, 6, 5])
        self.assertEqual(actual.state_write.tolist(), [5, 3, 6, 1])
        self.assertEqual(actual.context_write.tolist(), [11, 18, 7, 3])
        self.assertEqual(actual.history.tolist(), [[22, 23, 19], [26, 27, 23],
                                                   [14, 15, 29], [20, 21, 31]])
        pinned_lengths = lengths.cpu().pin_memory()
        host_plan = plan(pinned_lengths)
        self.assertIsNotNone(host_plan)
        self.assert_exact(host_plan, actual)
        self.assertIsNone(plan(lengths.cpu()))
        # Row 1 reads page 0 but writes page 1. Reject invalid writes before
        # even the valid request can consume this mixed-batch plan.
        for table, pool_rows in ((s, 11), (c, 22)):
            saved = int(table[1, 1].item())
            for bad_id in (0, -1, pool_rows):
                table[1, 1] = bad_id
                self.assertIsNone(plan())
            table[1, 1] = saved
        for bad_length in (0, -1, -(1 << 63), (1 << 63) - 1):
            invalid = lengths.long()
            invalid[-1] = bad_length
            self.assertIsNone(plan(invalid))
            self.assertIsNone(plan(invalid.cpu().pin_memory()))
        self.assert_exact((state_pool, context_pool), before)

    def test_graph_cache_plan_replay_updates_pages_and_state(self):
        def ints(values, dtype=torch.int32):
            return torch.tensor(values, device="cuda", dtype=dtype)

        # The two tags own different page IDs, shapes and strides.
        s = torch.zeros((4, 1028), device="cuda", dtype=torch.int32)[:, ::2]
        c = torch.zeros((4, 515), device="cuda", dtype=torch.int32)
        for row, page in enumerate((0, 127, 511, 0)):
            s[row, page : page + 2] = ints([2 * row + 1, 2 * row + 2])
            c[row, page : page + 2] = ints([2 * row + 11, 2 * row + 12])
        lengths_plus_one = ints([64, 8193, 32769, 0])
        ids = ints([19, 23, 29, 31], torch.int64)
        state = torch.randn(11, 9, 10240, device="cuda", dtype=torch.bfloat16)
        context = torch.arange(200, 244, device="cuda", dtype=torch.int64).view(22, 2)
        reference_state, reference_context = state.clone(), context.clone()
        current = torch.randn(4, 1, 10240, device="cuda", dtype=torch.bfloat16)

        def resolve(table, page, pool):
            index = table.gather(1, page.clamp(0, table.shape[1] - 1)[:, None])[:, 0]
            valid = (page >= 0) & (page < table.shape[1]) & (index > 0)
            return torch.where(valid & (index < pool.shape[0]), index, 0).long()

        def reference():
            raw = lengths_plus_one.long() - 1
            read_page = torch.div(raw - 1, 64, rounding_mode="floor")
            write_page = torch.div(raw, 64, rounding_mode="floor")
            sr = resolve(s, read_page, reference_state)
            sw = resolve(s, write_page, reference_state)
            cr = resolve(c, read_page, reference_context)
            cw = resolve(c, write_page, reference_context)
            history = torch.cat((reference_context.index_select(0, cr), ids[:, None]), 1)
            return sr, sw, cw, history

        def fused(lengths=lengths_plus_one):
            return try_plan_decode_cache(
                lengths, ids, s, c, state, context, 64, graph_mode=True
            )

        def update(pools, plan):
            state_pool, context_pool = pools
            sr, sw, cw, history = plan
            # Exercise the production gather and same two state destinations.
            rolled = torch.cat((state_pool.index_select(0, sr)[:, 1:], current), 1)
            state_pool.index_copy_(0, sw, rolled)
            context_pool.index_copy_(0, cw, history[:, -2:])

        self.assertIsNone(fused(lengths_plus_one.cpu().pin_memory()))
        self.assertIsNone(fused(lengths_plus_one.long()))
        warmup = fused()
        self.assertIsNotNone(warmup)
        update((state, context), warmup)
        state.copy_(reference_state)
        context.copy_(reference_context)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fused()
            self.assertIsNotNone(actual)
            update((state, context), actual)
        state.copy_(reference_state)
        context.copy_(reference_context)

        replays = ([64, 8193, 32769, 0], [65, 8194, 32770, 1], [65, 8193, 32769, 2])
        for step, lengths in enumerate(replays):
            lengths_plus_one.copy_(ints(lengths))
            ids.add_(1)
            current.normal_()
            if step == 2:
                # Invalid destinations/source independently map to the dummy
                # row; the other tag and the other direction stay live.
                s[0, 1] = state.shape[0]
                s[1, 128] = 9
                c[2, 511] = -1
            expected = reference()
            update((reference_state, reference_context), expected)
            graph.replay()
            self.assert_exact(actual, expected)
            self.assert_exact((state, context), (reference_state, reference_context))


if __name__ == "__main__":
    unittest.main()
