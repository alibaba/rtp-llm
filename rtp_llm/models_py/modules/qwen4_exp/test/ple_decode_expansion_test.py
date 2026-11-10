"""Exercise PLE planner dispatch through the serving runtime, not just the leaf."""

import os
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_execution import PLEExecution
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_transaction import (
    PLETransactionCoordinator,
)
from rtp_llm.models_py.modules.qwen4_exp.ple import (
    ple_commit_triton,
    ple_conv_triton,
    ple_decode_plan_triton,
)
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer


_PAGE = 64
_WIDTH = 10240
_STATE = 9
_CONTEXT = 2
_PAGES = 3
_PLAN = "QWEN4_PLE_DECODE_PLAN"
_TAIL = "QWEN4_PLE_DECODE_TAIL"
_S1_BATCHES = (1, 17, 64, 65, 128, 256, 257, 1025)


class _FixturePLE(Qwen4ExpPLELayer):
    """Use real conv/state geometry without an embedding table or projection GEMMs."""

    def __init__(self):
        nn.Module.__init__(self)
        self.ple_embedding = SimpleNamespace(
            context_len=_CONTEXT, eos_token_id=0, lookup_context=lambda _: nullcontext()
        )
        self.hc_hidden_size = _WIDTH
        self.conv_dilation = 3
        self.short_conv_state_len = _STATE
        self.conv_weight = torch.full(
            (_WIDTH, 1, 4), 0.125, device="cuda", dtype=torch.bfloat16
        )

    def _gated_values(
        self, hyper_states, token_history, padding_mask=None, *, embeddings=None
    ):
        # Include each candidate's n-gram history in the conv input so a stale
        # prefix or swapped request affects the output as well as cache checks.
        code = (
            token_history[:, :-2] + 2 * token_history[:, 1:-1] + token_history[:, 2:]
        ) % 8
        return hyper_states, hyper_states + code[..., None].to(torch.bfloat16) * 0.125


def _bf16_sample(shape):
    # Binary-fraction values make the four FP32 conv products/sums exact. A
    # mismatch then exercises the planner/state path, not accumulation order.
    return torch.randint(-8, 9, shape, device="cuda", dtype=torch.int8).to(
        torch.bfloat16
    ) * (1 / 32)


class _RuntimeFixture:
    def __init__(self, batch, width, *, target, graph=False):
        self.batch, self.width = batch, width
        self.target, self.graph = target, graph
        self.layer = _FixturePLE()
        shape = (_PAGES * batch + 1, _STATE, _WIDTH)
        self.state = _bf16_sample(shape)
        self.context = torch.arange(2 * shape[0], device="cuda").view(-1, 2)
        page_ids = torch.arange(1, shape[0], device="cuda", dtype=torch.int32)
        # The state table uses a non-unit column stride and a different mapping.
        self.state_table = torch.empty(
            (batch, 2 * _PAGES), device="cuda", dtype=torch.int32
        )[:, ::2]
        self.state_table.copy_(page_ids.view(_PAGES, batch).T)
        self.context_table = page_ids.flip(0).view(batch, _PAGES)
        self.ids = torch.arange(batch * width, device="cuda").view(batch, width) + 31
        self.hidden = _bf16_sample((batch, width, _WIDTH))
        self.lengths = torch.empty(batch, dtype=torch.int32, pin_memory=True)
        self.device_lengths = torch.empty(batch, device="cuda", dtype=torch.int32)
        self.update_lengths(62)
        empty = torch.empty(0, dtype=torch.int32, pin_memory=True)
        common = dict(
            is_prefill=target,
            is_target_verify=target,
            is_cuda_graph=graph,
            is_exact_cuda_graph_batch=graph,
            is_s_padded=graph,
            input_lengths=torch.full(
                (batch,), width, dtype=torch.int32, pin_memory=True
            ),
            prefix_lengths=self.lengths if target else empty,
            prefix_lengths_device=self.device_lengths if target else None,
            sequence_lengths=empty if target else self.lengths,
            sequence_lengths_plus_1_device=None if target else self.device_lengths,
            context_parallel_info=None,
            cache_store_inputs=None,
        )
        self.inputs = {
            tag: SimpleNamespace(**common, kv_cache_block_id_device=table)
            for tag, table in (
                (PLE_STATE_TAG, self.state_table),
                (PLE_NGRAM_CTX_TAG, self.context_table),
            )
        }
        caches = {
            tag: SimpleNamespace(tag=tag, kv_cache_base=base, seq_size_per_block=_PAGE)
            for tag, base in (
                (PLE_STATE_TAG, self.state.flatten(1)),
                (PLE_NGRAM_CTX_TAG, self.context),
            )
        }
        layers = {"1": self.layer}
        self.coordinator = PLETransactionCoordinator(layers, lambda _: None)
        self.runtime = PLEExecution(
            layers,
            SimpleNamespace(get_layer_cache=lambda _, tag: caches[tag]),
            SimpleNamespace(
                prefill_cp_config=SimpleNamespace(
                    is_enabled=lambda: False, is_prefill_enabled=lambda: False
                )
            ),
            0,
            self.coordinator,
        )

    def update_lengths(self, start):
        self.lengths.copy_(torch.arange(self.batch, dtype=torch.int32) % 3 + start)
        self.device_lengths.copy_(self.lengths, non_blocking=True)
        if not self.target:
            self.device_lengths.add_(1)

    def forward(self):
        return self.runtime._apply_ple(
            1, self.hidden.flatten(0, 1), self.ids.flatten(), self.inputs
        )

    def snapshot(self):
        return self.state.clone(), self.context.clone()

    def restore(self, snapshot):
        self.state.copy_(snapshot[0])
        self.context.copy_(snapshot[1])

    def advance_replay_inputs(self, step, start):
        self.update_lengths(start)
        self.ids.add_(step + 7)
        self.state_table.copy_(self.state_table.roll(1, dims=0))
        self.context_table.copy_(self.context_table.roll(-1, dims=0))

    @staticmethod
    def blocks(table, pages):
        return table.gather(1, pages[:, None]).squeeze(1).long()

    def reference(self):
        read_pages = (self.lengths.to(device="cuda", dtype=torch.long) - 1) // _PAGE
        state = self.state.index_select(0, self.blocks(self.state_table, read_pages))
        context = self.context.index_select(
            0, self.blocks(self.context_table, read_pages)
        )
        history = torch.cat((context, self.ids), dim=1)
        output, candidate = self.layer.decode_chunk(self.hidden, history, state)
        return (
            (self.hidden + output).flatten(0, 1),
            torch.cat((state, candidate), 1),
            history,
        )

    def after_accept(self, original, state_timeline, history, accepted):
        state_pool, ctx_pool = (t.clone() for t in original)
        prefixes = self.lengths.to(device="cuda", dtype=torch.long)
        accepted = accepted.long()
        read_pages = (prefixes - 1) // _PAGE
        final_pages = (prefixes + accepted - 1) // _PAGE
        # S <= 32 can cross at most one 64-token page. A prefix-cache hit must
        # see the completed page, not the later final state or rejected suffix.
        crossed = final_pages != read_pages

        def write(rows, pages, take):
            state_offsets = take[:, None] + torch.arange(_STATE, device="cuda")
            ctx_offsets = take[:, None] + torch.arange(_CONTEXT, device="cuda")
            state_pool.index_copy_(
                0,
                self.blocks(self.state_table, pages)[rows],
                state_timeline[rows[:, None], state_offsets],
            )
            ctx_pool.index_copy_(
                0,
                self.blocks(self.context_table, pages)[rows],
                history[rows[:, None], ctx_offsets],
            )

        if self.target:
            rows = torch.where(crossed)[0]
            write(rows, read_pages, ((read_pages + 1) * _PAGE - prefixes)[rows])
        write(torch.arange(self.batch, device="cuda"), final_pages, accepted)
        return state_pool, ctx_pool


@contextmanager
def _record_dispatch(module, name):
    original = getattr(module, name)
    calls = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((dict(kwargs), result is not None))
        return result

    with patch.object(module, name, record):
        yield calls


@unittest.skipUnless(torch.cuda.is_available(), "NVIDIA CUDA required")
class PLEDecodeExpansionTest(unittest.TestCase):
    def setUp(self):
        if torch.version.hip is not None or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            self.skipTest("SM90 required for production planner/conv dispatch")
        torch.manual_seed(1324)

    def assert_exact(self, actual, expected):
        if isinstance(actual, tuple):
            for a, b in zip(actual, expected, strict=True):
                self.assert_exact(a, b)
        else:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @torch.inference_mode()
    def test_s1_runtime_batch_expansion(self):
        for batch in _S1_BATCHES:
            with self.subTest(batch=batch):
                fixture = _RuntimeFixture(batch, 1, target=False)
                before = fixture.snapshot()
                with patch.dict(os.environ, {_PLAN: "0", _TAIL: "0"}):
                    expected = fixture.forward()
                    expected_pools = fixture.snapshot()
                fixture.restore(before)
                del before
                with (
                    patch.dict(os.environ, {_PLAN: "1", _TAIL: "1"}),
                    _record_dispatch(
                        ple_decode_plan_triton, "try_plan_decode_cache"
                    ) as plan,
                    _record_dispatch(ple_conv_triton, "try_ple_conv") as conv,
                ):
                    actual = fixture.forward()
                self.assertEqual(plan, [({"graph_mode": False}, True)])
                self.assertEqual(conv, [({"roll_state": True}, True)])
                self.assert_exact(actual, expected)
                self.assert_exact((fixture.state, fixture.context), expected_pools)

    @torch.inference_mode()
    def test_s1_graph_batch_expansion_matches_baseline(self):
        for batch in _S1_BATCHES:
            with self.subTest(batch=batch):
                self._check_s1_graph(batch)

    def _check_s1_graph(self, batch):
        fixture = _RuntimeFixture(batch, 1, target=False, graph=True)
        before = fixture.snapshot()

        def capture(enabled, initial):
            value = str(int(enabled))
            with patch.dict(os.environ, {_PLAN: value, _TAIL: value}):
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    fixture.forward()
                warmup_stream.synchronize()
                fixture.restore(initial)
                graph = torch.cuda.CUDAGraph()
                with (
                    _record_dispatch(
                        ple_decode_plan_triton, "try_plan_decode_cache"
                    ) as plan,
                    _record_dispatch(ple_conv_triton, "try_ple_conv") as conv,
                ):
                    with torch.cuda.graph(graph):
                        output = fixture.forward()
                self.assertEqual(
                    plan, [({"graph_mode": True}, True)] if enabled else []
                )
                self.assertEqual(
                    conv, [({"roll_state": True}, True)] if enabled else []
                )
            return graph, output

        baseline, expected = capture(False, before)
        optimized, actual = capture(True, before)
        del before
        for step, start in enumerate((63, 127)):
            with self.subTest(step=step):
                fixture.advance_replay_inputs(step, start)
                previous = fixture.snapshot()
                baseline.replay()
                expected_pools = fixture.snapshot()
                fixture.restore(previous)
                del previous
                optimized.replay()
                self.assert_exact(actual, expected)
                self.assert_exact((fixture.state, fixture.context), expected_pools)

    @torch.inference_mode()
    def test_s1_metadata_only_multicta_verdict_and_independent_dummy_rows(self):
        for batch in (129, 257, 1025, 4097):
            with self.subTest(batch=batch):
                self._check_multicta_metadata(batch)

    def _check_multicta_metadata(self, batch):
        # This leaf is read-only. Reuse valid physical block ids across
        # requests to exercise many CTAs without allocating B full state rows.
        rows = torch.arange(batch, device="cuda", dtype=torch.int64)
        lengths = rows % 3 + 63
        lengths[-4:] = 64
        plus_one = (lengths + 1).to(torch.int32)
        ids = rows * 7 + 19
        state_pool = torch.zeros(
            (7, _STATE, _WIDTH), device="cuda", dtype=torch.bfloat16
        )
        ctx_pool = torch.arange(16, device="cuda").view(8, 2) + 200
        state_table = torch.empty(
            (batch, 6), device="cuda", dtype=torch.int32
        )[:, ::2]
        state_table.copy_((rows[:, None] + torch.arange(3, device="cuda")) % 6 + 1)
        ctx_table = ((3 * rows[:, None] + torch.arange(4, device="cuda")) % 7 + 1).int()

        def plan(graph_mode=False):
            return ple_decode_plan_triton.try_plan_decode_cache(
                plus_one if graph_mode else lengths,
                ids,
                state_table,
                ctx_table,
                state_pool,
                ctx_pool,
                _PAGE,
                graph_mode=graph_mode,
            )

        def reference():
            read_page = (lengths - 1) // _PAGE
            write_page = lengths // _PAGE

            def resolve(table, page, pool_rows):
                block = _RuntimeFixture.blocks(table, page)
                return torch.where((block > 0) & (block < pool_rows), block, 0)

            sr = resolve(state_table, read_page, state_pool.shape[0])
            sw = resolve(state_table, write_page, state_pool.shape[0])
            cr = resolve(ctx_table, read_page, ctx_pool.shape[0])
            cw = resolve(ctx_table, write_page, ctx_pool.shape[0])
            history = torch.cat((ctx_pool.index_select(0, cr), ids[:, None]), 1)
            return sr, sw, cw, history

        actual = plan()
        self.assertIsNotNone(actual)
        self.assert_exact(actual, reference())
        # Earlier CTAs see only valid rows; the last CTA alone rejects.
        saved = ctx_table[-1, 1].clone()
        ctx_table[-1, 1] = 0
        self.assertIsNone(plan())
        ctx_table[-1, 1] = saved

        self.assertIsNotNone(plan(graph_mode=True))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = plan(graph_mode=True)
        self.assertIsNotNone(captured)
        # At a page boundary, all four lookups must fail independently. The
        # dummy context row is nonzero so a zero-filled history would fail.
        state_table[-4, 0] = 0
        state_table[-3, 1] = state_pool.shape[0]
        ctx_table[-2, 0] = -1
        ctx_table[-1, 1] = 0
        graph.replay()
        self.assert_exact(captured, reference())

    @torch.inference_mode()
    def test_target_eager_partial_accept_and_rollback(self):
        for batch, width in (
            (65, 2),
            (128, 3),
            (256, 5),
            (65, 32),
            (257, 3),
            (1025, 2),
        ):
            with self.subTest(batch=batch, width=width):
                fixture = _RuntimeFixture(batch, width, target=True)
                before = fixture.snapshot()
                expected, timeline, history = fixture.reference()
                with (
                    patch.dict(os.environ, {_PLAN: "1"}),
                    _record_dispatch(
                        ple_commit_triton, "gather_target_read_state"
                    ) as gather,
                ):
                    actual = fixture.forward()
                self.assertEqual(gather, [({"strict": True}, True)])
                self.assert_exact(actual, expected)
                self.assert_exact((fixture.state, fixture.context), before)
                accepted = (
                    torch.arange(batch, device="cuda", dtype=torch.int32) % width + 1
                )
                coordinator = fixture.coordinator
                coordinator.prepare_speculative_target_commit(accepted)
                self.assert_exact((fixture.state, fixture.context), before)
                coordinator.finish_speculative_target_commit(True)
                self.assert_exact(
                    (fixture.state, fixture.context),
                    fixture.after_accept(before, timeline, history, accepted),
                )
                coordinator.finish_speculative_target_commit(False)
                self.assertIsNone(coordinator.active_transaction)
                self.assert_exact((fixture.state, fixture.context), before)

    @torch.inference_mode()
    def test_target_eager_invalid_context_page_is_not_dummy_state(self):
        fixture = _RuntimeFixture(65, 2, target=True)
        fixture.context_table[0, 0] = 0
        before = fixture.snapshot()
        with (
            patch.dict(os.environ, {_PLAN: "1"}),
            _record_dispatch(ple_commit_triton, "gather_target_read_state") as gather,
        ):
            with self.assertRaisesRegex(RuntimeError, "unallocated logical page"):
                fixture.forward()
        self.assertEqual(gather, [({"strict": True}, False)])
        self.assertIsNone(fixture.coordinator.active_transaction)
        self.assert_exact((fixture.state, fixture.context), before)

    @torch.inference_mode()
    def test_target_eager_two_bad_tags_preserve_state_first_error(self):
        fixture = _RuntimeFixture(65, 2, target=True)
        fixture.state_table[0, 0] = 0
        fixture.inputs[PLE_NGRAM_CTX_TAG].kv_cache_block_id_device = (
            fixture.context_table.long()
        )
        before = fixture.snapshot()
        errors = []
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {_PLAN: enabled}):
                with self.assertRaisesRegex(
                    RuntimeError, "unallocated logical page"
                ) as error:
                    fixture.forward()
            errors.append(str(error.exception))
            self.assertIsNone(fixture.coordinator.active_transaction)
            self.assert_exact((fixture.state, fixture.context), before)
        self.assertEqual(errors[0], errors[1])
        self.assertIn(PLE_STATE_TAG, errors[1])

    @torch.inference_mode()
    def test_target_gather_extreme_prefix_and_independent_dummy_rows(self):
        for batch in (4, 4097):
            with self.subTest(batch=batch):
                self._check_target_gather_dummy_rows(batch)

    def _check_target_gather_dummy_rows(self, batch):
        width = 3
        ids = torch.arange(batch * width, device="cuda") + 100
        prefixes = torch.full((batch,), 63, dtype=torch.int64, device="cuda")
        # Repeated valid reads keep the pool size constant even at B=4097.
        state_table = torch.tensor([[1, 2]] * batch, dtype=torch.int32, device="cuda")
        ctx_table = torch.tensor([[3, 4]] * batch, dtype=torch.int32, device="cuda")
        state_pool = torch.arange(30, device="cuda").to(torch.bfloat16).view(5, 2, 3)
        ctx_pool = torch.arange(10, device="cuda").view(5, 2) + 300
        before = state_pool.clone(), ctx_pool.clone()

        def gather(*, strict):
            return ple_commit_triton.gather_target_read_state(
                prefixes,
                ids,
                state_table,
                ctx_table,
                state_pool,
                ctx_pool,
                _PAGE,
                strict=strict,
            )

        self.assertIsNotNone(gather(strict=True))
        limits = torch.iinfo(torch.int64)
        # A sole error in the final request must reject the entire eager plan.
        for value in (limits.min, limits.max):
            with self.subTest(prefix=value):
                prefixes[-1] = value
                self.assertIsNone(gather(strict=True))
                prefixes[-1] = 63
        for table in (state_table, ctx_table):
            block = table[-1, 0].clone()
            table[-1, 0] = 0
            self.assertIsNone(gather(strict=True))
            table[-1, 0] = block
        self.assertIsNotNone(gather(strict=False))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gather(strict=False)
        self.assertIsNotNone(actual)
        prefixes[-4] = limits.min
        prefixes[-3] = limits.max
        state_table[-2, 0] = 0
        ctx_table[-1, 0] = 0
        graph.replay()
        state_indices = torch.ones(batch, dtype=torch.long, device="cuda")
        context_indices = torch.full((batch,), 3, dtype=torch.long, device="cuda")
        state_indices[-4:-1] = 0
        context_indices[-4:-2] = 0
        context_indices[-1] = 0
        expected_context = ctx_pool.index_select(0, context_indices)
        self.assert_exact(
            actual,
            (
                state_pool.index_select(0, state_indices),
                expected_context,
                torch.cat((expected_context, ids.view(batch, width)), 1),
            ),
        )
        self.assert_exact((state_pool, ctx_pool), before)

    @torch.inference_mode()
    def test_target_graph_replay_uses_new_prefix_ids_and_physical_pages(self):
        for batch, width in ((65, 3), (128, 5), (256, 3), (257, 3), (1025, 2)):
            with self.subTest(batch=batch, width=width):
                self._check_target_graph(batch, width)

    def _check_target_graph(self, batch, width):
        fixture = _RuntimeFixture(batch, width, target=True, graph=True)
        coordinator = fixture.coordinator
        with patch.dict(os.environ, {_PLAN: "1"}):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                fixture.forward()
            coordinator.clear_speculative_target_graph_capture()
            graph = torch.cuda.CUDAGraph()
            with _record_dispatch(
                ple_commit_triton, "gather_target_read_state"
            ) as gather:
                with torch.cuda.graph(graph):
                    output = fixture.forward()
            self.assertEqual(gather, [({"strict": False}, True)])
        coordinator.save_speculative_target_graph_capture(fixture.batch)
        accepted = torch.arange(batch, device="cuda", dtype=torch.int32) % width + 1
        for step, start in enumerate((63, 127)):
            with self.subTest(step=step):
                fixture.advance_replay_inputs(step, start)
                before = fixture.snapshot()
                expected, timeline, history = fixture.reference()
                graph.replay()
                self.assert_exact(output, expected)
                self.assert_exact((fixture.state, fixture.context), before)
                coordinator.activate_speculative_target_graph_replay(fixture.batch)
                coordinator.prepare_speculative_target_commit(accepted)
                coordinator.finish_speculative_target_commit(True)
                self.assert_exact(
                    (fixture.state, fixture.context),
                    fixture.after_accept(before, timeline, history, accepted),
                )
                if step == 0:
                    coordinator.finish_speculative_target_commit(False)
                    self.assert_exact((fixture.state, fixture.context), before)
                else:
                    coordinator.finalize_speculative_target_commit()
                self.assertIsNone(coordinator.active_transaction)


if __name__ == "__main__":
    unittest.main()
