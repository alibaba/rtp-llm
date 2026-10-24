import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_cache_views import PLECacheViews
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_state import (
    _PLETargetTransaction,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_transaction import (
    PLETransactionCoordinator,
)
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_prepare_triton import (
    prepare_commit_values,
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


if __name__ == "__main__":
    unittest.main()
