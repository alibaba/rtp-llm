import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit_prepare import CommitValueBuffers
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_cache_views import PLECacheViews
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_commit_prepare import (
    prepare_commit_values,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_state import (
    _PLETargetLayerStage,
    _PLETargetTransaction,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_transaction import (
    PLETransactionCoordinator,
)


class PLECommitPrepareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU test must execute on CUDA")
        torch.manual_seed(20261019)

    def fixture(self, batch, length, width, context, query, strided=False):
        def state(rows, tokens):
            value = torch.randn(
                rows,
                tokens,
                width * (2 if strided else 1),
                device="cuda",
                dtype=torch.bfloat16,
            )
            return value[:, :, ::2] if strided else value

        history = torch.randint(
            0, 250000, (batch, 2 * (context + query)), device="cuda", dtype=torch.int64
        )
        initial_context = history[:, : 2 * context : 2]
        ids = history[:, 2 * context :: 2]
        stage = SimpleNamespace(
            live_rows=None,
            initial_state=state(batch, length),
            candidate_state_inputs=state(batch, query),
            initial_context=initial_context,
            candidate_ids=ids,
            state_pool=torch.randn(
                batch + 3, length, width, device="cuda", dtype=torch.bfloat16
            ),
            ctx_pool=torch.randint(
                0, 250000, (batch + 3, context), device="cuda", dtype=torch.int64
            ),
        )
        accepted = torch.arange(batch, device="cuda", dtype=torch.int64) % query + 1
        state_blocks = torch.arange(1, batch + 1, device="cuda", dtype=torch.int64)
        ctx_blocks = state_blocks.flip(0).contiguous() + 1
        return stage, accepted, state_blocks, ctx_blocks

    def compare(self, args):
        stage = args[0]
        before = stage.state_pool.clone(), stage.ctx_pool.clone()
        actual = prepare_commit_values(*args)
        self.assertIsNotNone(actual)
        expected = PLETransactionCoordinator._prepare_ple_values_reference(*args)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
        self.assertTrue(torch.equal(stage.state_pool, before[0]))
        self.assertTrue(torch.equal(stage.ctx_pool, before[1]))
        pointers = {a.data_ptr() for a in actual}
        self.assertEqual(len(pointers), 4)
        self.assertNotIn(stage.state_pool.data_ptr(), pointers)
        self.assertNotIn(stage.ctx_pool.data_ptr(), pointers)
        return actual, expected

    def test_all_accepted_windows_and_strided_history(self):
        for batch in (1, 3, 7, 32):
            for length, width, context, query in (
                (1, 17, 1, 1),
                (2, 127, 2, 4),
                (4, 2560, 8, 7),
                (2, 10240, 3, 4),
            ):
                for strided in (False, True):
                    args = self.fixture(batch, length, width, context, query, strided)
                    self.compare(args)
                    for accepted in range(1, query + 1):
                        args[1].fill_(accepted)
                        self.compare(args)

    def test_undo_is_immutable_after_tentative_pool_writes(self):
        args = self.fixture(3, 2, 2560, 2, 4)
        actual, expected = self.compare(args)
        stage, _, state_blocks, ctx_blocks = args
        stage.state_pool.index_copy_(0, state_blocks, actual[0])
        stage.ctx_pool.index_copy_(0, ctx_blocks, actual[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertTrue(torch.equal(actual[3], expected[3]))
        stage.state_pool.index_copy_(0, state_blocks, actual[2])
        stage.ctx_pool.index_copy_(0, ctx_blocks, actual[3])
        self.assertTrue(torch.equal(stage.state_pool[state_blocks], expected[2]))
        self.assertTrue(torch.equal(stage.ctx_pool[ctx_blocks], expected[3]))

    def test_workspace_refreshes_full_and_live_prefix_outputs(self):
        stage, accepted, state_blocks, ctx_blocks = self.fixture(
            8, 2, 10240, 2, 4, strided=True
        )
        workspace = CommitValueBuffers(stage.initial_state, stage.initial_context)
        addresses, previous = None, {}
        before = stage.state_pool.clone(), stage.ctx_pool.clone()
        for batch, count in ((8, 1), (3, 4), (1, 2), (3, 3), (8, 4)):
            with self.subTest(batch=batch, accepted=count):
                stage.initial_state.normal_()
                stage.candidate_state_inputs.normal_()
                stage.initial_context.add_(1)
                stage.candidate_ids.add_(1)
                accepted.fill_(count)
                live = SimpleNamespace(**vars(stage))
                for name in (
                    "initial_state",
                    "candidate_state_inputs",
                    "initial_context",
                    "candidate_ids",
                ):
                    setattr(live, name, getattr(stage, name)[:batch])
                args = live, accepted[:batch], state_blocks[:batch], ctx_blocks[:batch]
                actual = prepare_commit_values(*args, workspace=workspace)
                expected = PLETransactionCoordinator._prepare_ple_values_reference(
                    *args
                )
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertTrue(a.is_contiguous())
                self.assertIs(
                    actual, workspace.get(live.initial_state, live.initial_context)
                )
                if batch in previous:
                    self.assertIs(actual, previous[batch])
                previous[batch] = actual
                pointers = tuple(value.data_ptr() for value in actual)
                if addresses is None:
                    addresses = pointers
                self.assertEqual(pointers, addresses)
                self.assertEqual(len(set(pointers)), 4)
        torch.testing.assert_close(stage.state_pool, before[0], rtol=0, atol=0)
        torch.testing.assert_close(stage.ctx_pool, before[1], rtol=0, atol=0)

    def test_workspaces_keep_independent_undo_until_release(self):
        args = self.fixture(4, 2, 2560, 2, 4)
        stage, accepted, state_blocks, ctx_blocks = args
        before = stage.state_pool.clone(), stage.ctx_pool.clone()
        final = CommitValueBuffers(stage.initial_state, stage.initial_context)
        boundary = CommitValueBuffers(stage.initial_state, stage.initial_context)
        selected = prepare_commit_values(*args, workspace=final)
        saved = tuple(value.clone() for value in selected)
        accepted.fill_(1)
        stage.candidate_state_inputs.add_(1)
        staged_boundary = prepare_commit_values(*args, workspace=boundary)
        self.assertEqual(
            len({value.data_ptr() for value in selected + staged_boundary}), 8
        )
        for actual, expected in zip(selected, saved):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        stage.state_pool.index_copy_(0, state_blocks, selected[0])
        stage.ctx_pool.index_copy_(0, ctx_blocks, selected[1])
        for actual, expected in zip(selected[2:], saved[2:]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        stage.state_pool.index_copy_(0, state_blocks, selected[2])
        stage.ctx_pool.index_copy_(0, ctx_blocks, selected[3])
        torch.testing.assert_close(stage.state_pool, before[0], rtol=0, atol=0)
        torch.testing.assert_close(stage.ctx_pool, before[1], rtol=0, atol=0)

    def test_workspace_mismatch_falls_back_to_fresh_outputs(self):
        args = self.fixture(3, 2, 127, 2, 4)
        stage = args[0]
        initial, context = stage.initial_state, stage.initial_context
        layouts = (
            (initial[:1], context[:1]),
            (initial[:, :1], context),
            (initial[:, :, :17], context),
            (initial, context[:, :1]),
            (initial.float(), context),
            (initial, context.int()),
            (initial.cpu(), context.cpu()),
        )
        expected = PLETransactionCoordinator._prepare_ple_values_reference(*args)
        before = stage.state_pool.clone(), stage.ctx_pool.clone()
        for workspace_initial, workspace_context in layouts:
            workspace = CommitValueBuffers(workspace_initial, workspace_context)
            self.assertIsNone(workspace.get(initial, context))
            owned = workspace.get(workspace_initial, workspace_context)
            actual = prepare_commit_values(*args, workspace=workspace)
            self.assertFalse(
                {value.data_ptr() for value in actual}.intersection(
                    value.data_ptr() for value in owned
                )
            )
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        # A caller cannot supply pool-backed output tensors as a workspace.
        actual = prepare_commit_values(
            *args, workspace=(stage.state_pool, stage.ctx_pool)
        )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(stage.state_pool, before[0], rtol=0, atol=0)
        torch.testing.assert_close(stage.ctx_pool, before[1], rtol=0, atol=0)

    def test_unsupported_layout_does_not_use_workspace(self):
        args = self.fixture(33, 2, 17, 2, 4)
        workspace = CommitValueBuffers(args[0].initial_state, args[0].initial_context)
        with patch.object(
            workspace, "get", side_effect=AssertionError("ineligible prepare")
        ):
            self.assertIsNone(prepare_commit_values(*args, workspace=workspace))

    def test_workspace_graph_replay_refreshes_values_and_undo(self):
        args = self.fixture(7, 2, 2560, 2, 4, strided=True)
        stage = args[0]
        workspace = CommitValueBuffers(stage.initial_state, stage.initial_context)
        for _ in range(3):
            prepare_commit_values(*args, workspace=workspace)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = prepare_commit_values(*args, workspace=workspace)
        addresses = tuple(value.data_ptr() for value in actual)
        for accepted in (1, 4, 2):
            args[1].fill_(accepted)
            stage.candidate_state_inputs.normal_()
            stage.candidate_ids.add_(1)
            stage.state_pool.normal_()
            stage.ctx_pool.add_(1)
            graph.replay()
            expected = PLETransactionCoordinator._prepare_ple_values_reference(*args)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(addresses, tuple(value.data_ptr() for value in actual))

    def test_graph_replay_reads_updated_accept_and_source(self):
        args = self.fixture(7, 2, 2560, 3, 4, strided=True)
        for _ in range(3):
            prepare_commit_values(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = prepare_commit_values(*args)
        addresses = [x.data_ptr() for x in actual]
        for accepted in (1, 4, 2):
            args[1].fill_(accepted)
            args[0].candidate_state_inputs.normal_()
            args[0].state_pool.normal_()
            args[0].candidate_ids.add_(1)
            graph.replay()
            expected = PLETransactionCoordinator._prepare_ple_values_reference(*args)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(addresses, [x.data_ptr() for x in actual])

    def test_unsupported_layout_falls_back_without_writes(self):
        args = self.fixture(33, 2, 17, 2, 4)
        self.assertIsNone(prepare_commit_values(*args))
        args = self.fixture(3, 2, 17, 2, 4)
        args[0].state_pool = args[0].state_pool.to(torch.float32)
        self.assertIsNone(prepare_commit_values(*args))

    def transaction_fixture(self, batch=7):
        stage, accepted, state_blocks, ctx_blocks = self.fixture(batch, 2, 2560, 3, 4)
        stage.layer_idx = 1
        stage.page_size = 128
        stage.prefixes = torch.full((batch,), 127, device="cuda", dtype=torch.int64)
        stage.state_inputs = SimpleNamespace(
            kv_cache_block_id_device=state_blocks.int()[:, None].expand(-1, 3)
        )
        stage.ctx_inputs = SimpleNamespace(
            kv_cache_block_id_device=ctx_blocks.int()[:, None].expand(-1, 3)
        )
        coordinator = PLETransactionCoordinator({"1": object()}, lambda phase: None)
        coordinator.active_transaction = _PLETargetTransaction(
            batch, 4, stage.prefixes, frozenset({1}), {1: stage}
        )
        return coordinator, stage, accepted, state_blocks, ctx_blocks

    def captured_fixture(self, batch=7):
        owner, stage, accepted, sb, cb = self.transaction_fixture(batch)
        stage = _PLETargetLayerStage(**vars(stage))
        transaction = owner.active_transaction
        transaction.stages[stage.layer_idx] = stage
        owner.save_speculative_target_graph_capture(batch)
        return owner, transaction, stage, accepted, sb, cb

    @unittest.skipUnless(
        torch.cuda.get_device_capability() == (12, 0), "workspace requires SM120"
    )
    def test_captured_workspace_live_rows_buckets_and_retirement(self):
        owner, first, _, _, _, _ = self.captured_fixture()
        _, second, _, _, _, _ = self.captured_fixture(3)
        owner.captured_transactions[3] = second
        pointers = {}
        for key, batch in ((7, 7), (7, 1), (3, 3), (7, 5), (3, 1), (7, 7)):
            with self.subTest(bucket=key, live_batch=batch):
                owner.activate_speculative_target_graph_replay(key)
                transaction = owner.active_transaction
                stage = transaction.stages[1]
                stage.candidate_state_inputs.normal_()
                stage.candidate_ids.add_(1)
                stage.state_pool.normal_()
                stage.ctx_pool.add_(1)
                before = stage.state_pool.clone(), stage.ctx_pool.clone()
                accepted = torch.arange(batch, device="cuda", dtype=torch.int32) % 4 + 1
                owner.prepare_speculative_target_commit(accepted)
                workspace = transaction.prepare_workspace
                self.assertIs(transaction.workspace_lease, workspace)
                self.assertTrue(workspace.busy)
                write = transaction.prepared_writes[0]
                outputs = (
                    write.state_values,
                    write.ctx_values,
                    write.original_state,
                    write.original_context,
                )
                addresses = tuple(value.data_ptr() for value in outputs)
                if key in pointers:
                    self.assertEqual(addresses, pointers[key])
                pointers[key] = addresses
                live = replace(
                    stage,
                    prefixes=stage.prefixes[:batch],
                    state_inputs=SimpleNamespace(
                        kv_cache_block_id_device=stage.state_inputs.kv_cache_block_id_device[
                            :batch
                        ]
                    ),
                    ctx_inputs=SimpleNamespace(
                        kv_cache_block_id_device=stage.ctx_inputs.kv_cache_block_id_device[
                            :batch
                        ]
                    ),
                    initial_state=stage.initial_state[:batch],
                    initial_context=stage.initial_context[:batch],
                    candidate_state_inputs=stage.candidate_state_inputs[:batch],
                    candidate_ids=stage.candidate_ids[:batch],
                )
                expected = self._prepare_reference_and_pools(
                    live, accepted.long(), write, before
                )
                owner.finish_speculative_target_commit(True)
                torch.testing.assert_close(
                    stage.state_pool, expected[0], rtol=0, atol=0
                )
                torch.testing.assert_close(stage.ctx_pool, expected[1], rtol=0, atol=0)
                owner.finalize_speculative_target_commit()
                self.assertIsNone(transaction.workspace_lease)
                self.assertTrue(
                    workspace.busy, "local finalize precedes rank consensus"
                )
                with self.assertRaises(RuntimeError):
                    owner.activate_speculative_target_graph_replay(99)
                self.assertTrue(workspace.busy)
        self.assertFalse(first.prepare_workspace is second.prepare_workspace)
        self.assertFalse(set(pointers[3]).intersection(pointers[7]))

    def _prepare_reference_and_pools(self, stage, accepted, write, before):
        # These fixtures start at token 127: one accepted token completes the
        # first page, while later tokens select the next final page.
        self.assertTrue(torch.all(stage.prefixes == 127))
        expected = before[0].clone(), before[1].clone()
        pointers = []
        for slot, count in (
            *((slot, torch.ones_like(accepted)) for slot in write.boundary_slots),
            (write, accepted),
        ):
            pages = (stage.prefixes + count - 1) // stage.page_size
            rows = torch.arange(accepted.numel(), device=accepted.device)
            sb = stage.state_inputs.kv_cache_block_id_device[rows, pages].long()
            cb = stage.ctx_inputs.kv_cache_block_id_device[rows, pages].long()
            torch.testing.assert_close(slot.state_blocks, sb, rtol=0, atol=0)
            torch.testing.assert_close(slot.ctx_blocks, cb, rtol=0, atol=0)
            actual = (
                slot.state_values,
                slot.ctx_values,
                slot.original_state,
                slot.original_context,
            )
            reference = PLETransactionCoordinator._prepare_ple_values_reference(
                stage, count, sb, cb
            )
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            pointers.extend(value.data_ptr() for value in actual)
            expected[0].index_copy_(0, sb, reference[0])
            expected[1].index_copy_(0, cb, reference[1])
        self.assertEqual(len(pointers), len(set(pointers)))
        return expected

    @unittest.skipUnless(
        torch.cuda.get_device_capability() == (12, 0), "workspace requires SM120"
    )
    def test_workspace_dp_compaction_and_empty_batch(self):
        for mask in ((False,) * 7, (True, False, True, False, False, True, False)):
            owner, transaction, stage, accepted, _, _ = self.captured_fixture()
            stage.live_rows = torch.tensor(mask)
            before = stage.state_pool.clone(), stage.ctx_pool.clone()
            owner.activate_speculative_target_graph_replay(7)
            owner.prepare_speculative_target_commit(accepted.int())
            if not any(mask):
                self.assertEqual(transaction.prepared_writes, [])
                self.assertIsNone(transaction.prepare_workspace)
            else:
                rows = stage.live_rows.nonzero().reshape(-1).cuda()
                live = replace(
                    stage,
                    prefixes=stage.prefixes.index_select(0, rows),
                    state_inputs=SimpleNamespace(
                        kv_cache_block_id_device=stage.state_inputs.kv_cache_block_id_device.index_select(
                            0, rows
                        )
                    ),
                    ctx_inputs=SimpleNamespace(
                        kv_cache_block_id_device=stage.ctx_inputs.kv_cache_block_id_device.index_select(
                            0, rows
                        )
                    ),
                    initial_state=stage.initial_state.index_select(0, rows),
                    initial_context=stage.initial_context.index_select(0, rows),
                    candidate_state_inputs=stage.candidate_state_inputs.index_select(
                        0, rows
                    ),
                    candidate_ids=stage.candidate_ids.index_select(0, rows),
                )
                expected = self._prepare_reference_and_pools(
                    live,
                    accepted.index_select(0, rows),
                    transaction.prepared_writes[0],
                    before,
                )
                owner.finish_speculative_target_commit(True)
                torch.testing.assert_close(
                    stage.state_pool, expected[0], rtol=0, atol=0
                )
                torch.testing.assert_close(stage.ctx_pool, expected[1], rtol=0, atol=0)
            owner.finish_speculative_target_commit(False)
            torch.testing.assert_close(stage.state_pool, before[0], rtol=0, atol=0)
            torch.testing.assert_close(stage.ctx_pool, before[1], rtol=0, atol=0)

    @unittest.skipUnless(
        torch.cuda.get_device_capability() == (12, 0), "workspace requires SM120"
    )
    def test_busy_workspace_falls_back_without_overwriting_held_undo(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_commit_workspace import (
            PLECommitWorkspace,
        )

        owner, transaction, stage, accepted, _, _ = self.captured_fixture()
        workspace = transaction.prepare_workspace = PLECommitWorkspace(transaction)
        workspace.busy = True
        held = workspace.values(stage, "final").get(
            stage.initial_state, stage.initial_context
        )
        for value in held:
            value.fill_(17)
        owner.activate_speculative_target_graph_replay(7)
        owner.prepare_speculative_target_commit(accepted.int())
        self.assertIsNone(transaction.workspace_lease)
        write = transaction.prepared_writes[0]
        self.assertFalse(
            {
                write.state_values.data_ptr(),
                write.ctx_values.data_ptr(),
                write.original_state.data_ptr(),
                write.original_context.data_ptr(),
            }.intersection(value.data_ptr() for value in held)
        )
        owner.finish_speculative_target_commit(False)
        self.assertTrue(workspace.busy)
        for value in held:
            self.assertTrue(torch.all(value == 17))
        self.assertEqual(owner._retired_workspaces, [])

    @unittest.skipUnless(
        torch.cuda.get_device_capability() == (12, 0), "workspace requires SM120"
    )
    def test_failed_prepare_rollback_commit_and_finalize_keep_lease(self):
        owner, transaction, stage, accepted, _, _ = self.captured_fixture()
        before = stage.state_pool.clone(), stage.ctx_pool.clone()
        owner.activate_speculative_target_graph_replay(7)
        invalid = accepted.int()
        invalid[0] = 0
        with self.assertRaises(RuntimeError):
            owner.prepare_speculative_target_commit(invalid)
        workspace = transaction.prepare_workspace
        self.assertTrue(workspace.busy)
        self.assertIs(transaction.workspace_lease, workspace)
        with patch.object(
            owner,
            "_synchronize_ple_transaction",
            side_effect=RuntimeError("rollback completion"),
        ):
            with self.assertRaisesRegex(RuntimeError, "rollback completion"):
                owner.finish_speculative_target_commit(False)
        self.assertIs(owner.active_transaction, transaction)
        self.assertIs(transaction.workspace_lease, workspace)
        owner.finish_speculative_target_commit(False)
        owner.finish_speculative_target_commit(False)
        self.assertTrue(workspace.busy)
        owner.activate_speculative_target_graph_replay(7)
        self.assertFalse(workspace.busy)
        owner.prepare_speculative_target_commit(accepted.int())
        with patch.object(
            owner,
            "_synchronize_ple_transaction",
            side_effect=RuntimeError("commit completion"),
        ):
            with self.assertRaisesRegex(RuntimeError, "commit completion"):
                owner.finish_speculative_target_commit(True)
        self.assertTrue(transaction.commit_started)
        self.assertTrue(workspace.busy)
        owner.finish_speculative_target_commit(False)
        torch.testing.assert_close(stage.state_pool, before[0], rtol=0, atol=0)
        torch.testing.assert_close(stage.ctx_pool, before[1], rtol=0, atol=0)
        owner.activate_speculative_target_graph_replay(7)
        owner.prepare_speculative_target_commit(accepted.int())
        owner.finish_speculative_target_commit(True)
        with patch.object(
            owner, "fault_hook", side_effect=RuntimeError("peer finalize")
        ):
            with self.assertRaisesRegex(RuntimeError, "peer finalize"):
                owner.finalize_speculative_target_commit()
        self.assertTrue(workspace.busy)
        with self.assertRaises(RuntimeError):
            owner.activate_speculative_target_graph_replay(7)
        owner.finalize_speculative_target_commit()
        self.assertTrue(workspace.busy)

    @unittest.skipUnless(
        torch.cuda.get_device_capability() == (12, 0), "fused metadata requires SM120"
    )
    def test_prepare_combines_verdict_and_values_completion_wait(self):
        owner, _, accepted, _, _ = self.transaction_fixture()
        complete = torch.cuda.Event()

        def prepare(*args, **kwargs):
            values = prepare_commit_values(*args, **kwargs)
            complete.record()
            return values

        with (
            patch(
                "rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_commit_prepare.prepare_commit_values",
                side_effect=prepare,
            ),
            patch.object(
                owner,
                "_synchronize_ple_transaction",
                side_effect=AssertionError(
                    "prepare inserted a second GPU completion wait"
                ),
            ),
        ):
            owner.prepare_speculative_target_commit(accepted.int())
        # query() cannot wait for GPU work: the combined verdict wait must also
        # have completed the immutable prepared values and undo buffers.
        self.assertTrue(complete.query())
        self.assertIsNotNone(owner.active_transaction.prepared_writes)

    def test_full_transaction_commit_rollback_and_finalize(self):
        owner, stage, accepted, sb, cb = self.transaction_fixture()
        before = stage.state_pool.clone(), stage.ctx_pool.clone()
        expected = PLETransactionCoordinator._prepare_ple_values_reference(
            stage, accepted, sb, cb
        )
        owner.prepare_speculative_target_commit(accepted.int())
        self.assertTrue(torch.equal(stage.state_pool, before[0]))
        self.assertTrue(torch.equal(stage.ctx_pool, before[1]))
        saved = owner.active_transaction
        owner.finish_speculative_target_commit(True)
        self.assertTrue(torch.equal(stage.state_pool[sb], expected[0]))
        self.assertTrue(torch.equal(stage.ctx_pool[cb], expected[1]))
        owner.finish_speculative_target_commit(False)
        self.assertTrue(torch.equal(stage.state_pool, before[0]))
        self.assertTrue(torch.equal(stage.ctx_pool, before[1]))
        owner.finish_speculative_target_commit(False)
        saved.prepared_writes = None
        saved.commit_started = saved.tentative_committed = False
        owner.active_transaction = saved
        owner.prepare_speculative_target_commit(accepted.int())
        owner.finish_speculative_target_commit(True)
        owner.finalize_speculative_target_commit()
        self.assertIsNone(owner.active_transaction)
        self.assertTrue(torch.equal(stage.state_pool[sb], expected[0]))

    def test_pair_verdict_matches_sequential_with_independent_tables(self):
        owner, stage, accepted, _, _ = self.transaction_fixture()
        stage.state_inputs.kv_cache_block_id_device = (
            torch.arange(7 * 6, device="cuda", dtype=torch.int32).reshape(7, 6) % 9 + 1
        )[:, ::2]
        for prefixes in (0, 127, 128, 383, -1, 384, 2**40):
            stage.prefixes.fill_(prefixes)
            pages = (stage.prefixes + accepted - 1) // stage.page_size
            expected_error = None
            expected = []
            try:
                for inputs, tag, pool in (
                    (stage.state_inputs, "ple_state", stage.state_pool),
                    (stage.ctx_inputs, "ple_ngram_ctx", stage.ctx_pool),
                ):
                    expected.append(
                        PLECacheViews._commit_physical_blocks(
                            inputs, tag, pages, pool.shape[0], pool.device
                        )
                    )
            except RuntimeError as exc:
                expected_error = str(exc).split("'", 2)[-1]
            if expected_error is not None:
                with self.assertRaises(RuntimeError) as caught:
                    PLECacheViews._commit_physical_block_pair(stage, accepted)
                self.assertEqual(
                    str(caught.exception).split("'", 2)[-1], expected_error
                )
            else:
                actual = PLECacheViews._commit_physical_block_pair(stage, accepted)
                self.assertIsNotNone(actual)
                self.assertTrue(torch.equal(actual[0], expected[0][0]))
                self.assertEqual(actual[1], expected[0][1])
                self.assertTrue(torch.equal(actual[2], expected[1][0]))
                self.assertEqual(actual[3], expected[1][1])

    def test_gpu_invalid_accept_and_destinations_never_write_pools(self):
        for failure in ("accept", "missing", "overflow", "duplicate", "context"):
            owner, stage, accepted, _, _ = self.transaction_fixture()
            before = stage.state_pool.clone(), stage.ctx_pool.clone()
            if failure == "accept":
                accepted[0] = 0
            else:
                inputs = (
                    stage.ctx_inputs if failure == "context" else stage.state_inputs
                )
                table = inputs.kv_cache_block_id_device.clone()
                table[0].fill_(0 if failure in ("missing", "context") else 999)
                if failure == "duplicate":
                    table[0].copy_(table[1])
                inputs.kv_cache_block_id_device = table
            with self.assertRaises(RuntimeError):
                owner.prepare_speculative_target_commit(accepted.int())
            self.assertIsNone(owner.active_transaction.prepared_writes)
            self.assertTrue(torch.equal(stage.state_pool, before[0]))
            self.assertTrue(torch.equal(stage.ctx_pool, before[1]))
            owner.finish_speculative_target_commit(False)

    def test_invalid_table_layout_never_writes_pools(self):
        for fused in (True, False):
            for region in ("state", "ctx"):
                with self.subTest(fused=fused, region=region):
                    owner, stage, accepted, _, _ = self.transaction_fixture()
                    before = stage.state_pool.clone(), stage.ctx_pool.clone()
                    inputs = getattr(stage, region + "_inputs")
                    inputs.kv_cache_block_id_device = (
                        inputs.kv_cache_block_id_device.long()
                    )
                    resolution = (
                        nullcontext()
                        if fused
                        else patch(
                            "rtp_llm.models_py.model_desc.qwen4_exp_runtime."
                            "ple_commit_metadata.begin_accepted_pair_resolution",
                            return_value=None,
                        )
                    )
                    with (
                        resolution,
                        self.assertRaisesRegex(RuntimeError, "must be int32"),
                    ):
                        owner.prepare_speculative_target_commit(accepted.int())
                    self.assertIsNone(owner.active_transaction.prepared_writes)
                    torch.testing.assert_close(
                        stage.state_pool, before[0], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        stage.ctx_pool, before[1], atol=0, rtol=0
                    )
                    owner.finish_speculative_target_commit(False)

    def test_rollback_retry_refreshes_distinct_boundary_and_final_pages(self):
        owner, stage, _, _, _ = self.transaction_fixture(batch=3)
        stage.state_pool = stage.state_pool.repeat(2, 1, 1)
        stage.ctx_pool = stage.ctx_pool.repeat(2, 1)
        table = torch.arange(1, 10, device="cuda", dtype=torch.int32).reshape(3, 3)
        stage.state_inputs.kv_cache_block_id_device = table
        stage.ctx_inputs.kv_cache_block_id_device = table.flip(0).contiguous()
        before_state, before_context = stage.state_pool.clone(), stage.ctx_pool.clone()
        for attempt, lengths in enumerate(((1, 2, 4), (4, 1, 2), (2, 4, 1))):
            with self.subTest(accepted=lengths):
                owner.active_transaction = _PLETargetTransaction(
                    3, 4, stage.prefixes, frozenset({1}), {1: stage}
                )
                accepted = torch.tensor(lengths, device="cuda", dtype=torch.int32)
                owner.prepare_speculative_target_commit(accepted)
                self.assertTrue(torch.equal(stage.state_pool, before_state))
                self.assertTrue(torch.equal(stage.ctx_pool, before_context))
                owner.finish_speculative_target_commit(True)
                expected_state, expected_context = (
                    before_state.clone(),
                    before_context.clone(),
                )
                # Prefix 127 crosses the 128-token boundary after one token.
                # Check each page against independent history concatenation.
                for row, count in enumerate(lengths):
                    final_page = (127 + count - 1) // stage.page_size
                    for page, tokens in {0: 1, final_page: count}.items():
                        state_block = int(table[row, page].item())
                        context_block = int(
                            stage.ctx_inputs.kv_cache_block_id_device[row, page].item()
                        )
                        expected_state[state_block] = torch.cat(
                            (
                                stage.initial_state[row],
                                stage.candidate_state_inputs[row, :tokens],
                            )
                        )[-stage.initial_state.shape[1] :]
                        expected_context[context_block] = torch.cat(
                            (
                                stage.initial_context[row],
                                stage.candidate_ids[row, :tokens],
                            )
                        )[-stage.initial_context.shape[1] :]
                self.assertTrue(torch.equal(stage.state_pool, expected_state))
                self.assertTrue(torch.equal(stage.ctx_pool, expected_context))
                if attempt < 2:
                    owner.finish_speculative_target_commit(False)
                    owner.finish_speculative_target_commit(False)
                    self.assertTrue(torch.equal(stage.state_pool, before_state))
                    self.assertTrue(torch.equal(stage.ctx_pool, before_context))
                else:
                    owner.finalize_speculative_target_commit()
                self.assertIsNone(owner.active_transaction)


if __name__ == "__main__":
    unittest.main()
