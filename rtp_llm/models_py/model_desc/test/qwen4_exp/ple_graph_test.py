from types import SimpleNamespace
from unittest import TestCase, main

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_cache_views import PLECacheViews
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_state import (
    _PLETargetLayerStage,
    _PLETargetTransaction,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import (
    _HC,
    _HIDDEN,
    PLERuntimeFixture,
)


class Qwen4ExpPLEGraphTest(PLERuntimeFixture, TestCase):
    def test_target_graph_transaction_survives_replay_commit_and_rollback(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        query_len = 5
        self._seed_target_history(prefixes)
        ids = torch.tensor([[4, 5, 6, 8, 9], [3, 4, 5, 6, 8]], dtype=torch.long)
        hyper = torch.randn(2, query_len, _HC * _HIDDEN, dtype=torch.bfloat16)
        inputs = self._target_inputs(prefixes, query_len)

        # Capture warmup must not leave an active transaction behind.
        self.model._apply_ple(1, hyper.flatten(0, 1), ids.flatten(), inputs)
        self.model.on_cuda_graph_warmup(SimpleNamespace(attention_inputs=inputs))
        self.assertIsNone(self.model._ple_target_transaction)

        self.model._apply_ple(1, hyper.flatten(0, 1), ids.flatten(), inputs)
        self.model.on_cuda_graph_capture(2, SimpleNamespace(attention_inputs=inputs))
        captured = self.model._ple_graph_target_transactions[2]
        self.assertIsNone(self.model._ple_target_transaction)
        with self.assertRaisesRegex(RuntimeError, "missing for replay batch"):
            self.model.on_cuda_graph_replay(1, SimpleNamespace(attention_inputs=inputs))

        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()
        self.model.on_cuda_graph_replay(2, SimpleNamespace(attention_inputs=inputs))
        with self.assertRaisesRegex(RuntimeError, "still active"):
            self.model.on_cuda_graph_replay(2, SimpleNamespace(attention_inputs=inputs))
        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 5], dtype=torch.int32)
        )
        self.model.finish_speculative_target_commit(True)
        self.model.finish_speculative_target_commit(False)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)
        self.assertIsNone(captured.prepared_writes)
        self.assertFalse(captured.commit_started)

        # A second replay selects the same graph-owned stage after the first
        # invocation has released its transient undo buffers.
        self.model.on_cuda_graph_replay(2, SimpleNamespace(attention_inputs=inputs))
        self.model.prepare_speculative_target_commit(
            torch.tensor([2, 3], dtype=torch.int32)
        )
        self.model.finish_speculative_target_commit(True)
        self.model.finalize_speculative_target_commit()
        self.assertIsNone(self.model._ple_target_transaction)
        self.assertIs(self.model._ple_graph_target_transactions[2], captured)
        self.assertIsNone(captured.prepared_writes)
        self.assertFalse(captured.tentative_committed)

    def test_decode_graph_masks_padding_after_batch_and_page_changes(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Graph replay")
        self._move_ple_to_cuda(torch.device("cuda"))
        self.state_base.normal_()
        self.ctx_base.random_(1, 7)
        initial_state, initial_ctx = self.state_base.clone(), self.ctx_base.clone()
        state_table, ctx_table = self.state_blocks.clone(), self.ctx_blocks.clone()
        ids = torch.tensor([5, 6], dtype=torch.long, device="cuda")
        hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16, device="cuda")
        lengths = torch.tensor([7, 8], dtype=torch.int32).pin_memory()
        device_lengths = (lengths + 1).cuda()
        inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=lengths,
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=lengths,
        )
        for value in inputs.values():
            value.is_cuda_graph = value.is_s_padded = True
            value.sequence_lengths_plus_1_device = device_lengths
        self.model._apply_ple(1, hyper, ids, inputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.model._apply_ple(1, hyper, ids, inputs)
        pool = self.state_base.view(12, self.ple.short_conv_state_len, -1)
        for live_batch, position in ((1, 7), (2, 8), (1, 15), (0, 0)):
            self.state_base.copy_(initial_state)
            self.ctx_base.copy_(initial_ctx)
            self.state_blocks.copy_(state_table)
            self.ctx_blocks.copy_(ctx_table)
            self.state_blocks[live_batch:].zero_()
            self.ctx_blocks[live_batch:].zero_()
            device_lengths.fill_(1)
            device_lengths[:live_batch].fill_(position + 1)
            expected_state, expected_ctx = initial_state.clone(), initial_ctx.clone()
            if live_batch:
                read_page, write_page = (
                    position - 1
                ) // self._PAGE, position // self._PAGE
                state_rows = state_table[:live_batch, read_page].long()
                ctx_rows = ctx_table[:live_batch, read_page].long()
                history = torch.cat(
                    (initial_ctx[ctx_rows], ids[:live_batch, None]), dim=1
                )
                output, state = self.ple.decode_step(
                    hyper[:live_batch, None],
                    history,
                    initial_state.view_as(pool)[state_rows],
                )
                expected_state[state_table[:live_batch, write_page].long()] = (
                    state.flatten(1)
                )
                expected_ctx[ctx_table[:live_batch, write_page].long()] = ids[
                    :live_batch, None
                ]
            graph.replay()
            if live_batch:
                torch.testing.assert_close(
                    actual[:live_batch], hyper[:live_batch] + output.squeeze(1)
                )
            torch.testing.assert_close(self.state_base, expected_state)
            torch.testing.assert_close(self.ctx_base, expected_ctx)

    def test_target_graph_replays_padding_and_commits_only_live_requests(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Graph replay")
        self._move_ple_to_cuda(torch.device("cuda"))
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        self.state_base[0].fill_(0.25)
        self.ctx_base[0].fill_(6)
        before_state, before_ctx = self.state_base.clone(), self.ctx_base.clone()
        state_table, ctx_table = self.state_blocks.clone(), self.ctx_blocks.clone()
        ids = torch.tensor([[4, 5, 6], [3, 4, 5]], dtype=torch.long, device="cuda")
        hyper = torch.randn(6, _HC * _HIDDEN, dtype=torch.bfloat16, device="cuda")
        inputs = self._target_inputs(prefixes, 3)
        for value in inputs.values():
            value.is_cuda_graph = value.is_s_padded = True
        self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.model.on_cuda_graph_warmup(SimpleNamespace(attention_inputs=inputs))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.model.on_cuda_graph_capture(2, SimpleNamespace(attention_inputs=inputs))
        for live_batch in (1, 2, 1):
            self.state_base.copy_(before_state)
            self.ctx_base.copy_(before_ctx)
            self.state_blocks.copy_(state_table)
            self.ctx_blocks.copy_(ctx_table)
            self.state_blocks[live_batch:].zero_()
            self.ctx_blocks[live_batch:].zero_()
            for value in inputs.values():
                value.prefix_lengths_device.copy_(torch.tensor(prefixes, device="cuda"))
                value.prefix_lengths_device[live_batch:].zero_()
            graph.replay()
            self.model.on_cuda_graph_replay(2, SimpleNamespace(attention_inputs=inputs))
            self.model.prepare_speculative_target_commit(
                torch.full((live_batch,), 3, dtype=torch.int32, device="cuda")
            )
            self.model.finish_speculative_target_commit(True)
            if live_batch == 1:
                torch.testing.assert_close(
                    self.state_base[[0, 2, 4, 8]], before_state[[0, 2, 4, 8]]
                )
                torch.testing.assert_close(
                    self.ctx_base[[0, 4, 2, 10]], before_ctx[[0, 4, 2, 10]]
                )
            self.model.finish_speculative_target_commit(False)
            torch.testing.assert_close(self.state_base, before_state)
            torch.testing.assert_close(self.ctx_base, before_ctx)

    def test_graph_physical_blocks_replay_uses_updated_table_and_pages(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Graph replay")
        device = torch.device("cuda")
        table = torch.tensor([[1, 3], [5, 7]], dtype=torch.int32, device=device)
        pages = torch.tensor([1, 0], dtype=torch.long, device=device)
        attention_inputs = SimpleNamespace(kv_cache_block_id_device=table)
        resolve = PLECacheViews._physical_blocks

        resolve(attention_inputs, PLE_STATE_TAG, pages, 12, device, graph_capture=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            blocks = resolve(
                attention_inputs,
                PLE_STATE_TAG,
                pages,
                12,
                device,
                graph_capture=True,
            )
        graph.replay()
        torch.testing.assert_close(blocks.cpu(), torch.tensor([3, 5]))

        table.copy_(torch.tensor([[0, 9], [6, 8]], dtype=torch.int32, device=device))
        pages.copy_(torch.tensor([0, 1], dtype=torch.long, device=device))
        graph.replay()
        torch.testing.assert_close(blocks.cpu(), torch.tensor([0, 8]))

        pages.copy_(torch.tensor([99, -1], dtype=torch.long, device=device))
        graph.replay()
        torch.testing.assert_close(blocks.cpu(), torch.tensor([0, 0]))

    def test_target_graph_commit_reads_latest_replayed_candidate(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Graph replay")
        device = torch.device("cuda")
        state_seed = torch.arange(6, device=device, dtype=torch.float32).view(2, 3, 1)
        ids_seed = torch.arange(6, device=device, dtype=torch.int64).view(2, 3)
        state_seed.add_(1)
        ids_seed.add_(1)
        state_pool = torch.zeros((5, 2, 1), device=device)
        ctx_pool = torch.zeros((5, 2), device=device, dtype=torch.int64)
        table = torch.tensor([[1, 2], [3, 4]], device=device, dtype=torch.int32)
        inputs = SimpleNamespace(kv_cache_block_id_device=table)
        prefixes = torch.tensor([8, 8], device=device, dtype=torch.long)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate_states = state_seed + 10
            candidate_ids = ids_seed + 100
        stage = _PLETargetLayerStage(
            layer_idx=1,
            page_size=8,
            prefixes=prefixes,
            state_pool=state_pool,
            ctx_pool=ctx_pool,
            state_inputs=inputs,
            ctx_inputs=inputs,
            initial_state=torch.zeros((2, 2, 1), device=device),
            candidate_state_inputs=candidate_states,
            initial_context=torch.zeros((2, 2), device=device, dtype=torch.int64),
            candidate_ids=candidate_ids,
        )
        self.model._ple_target_transaction = _PLETargetTransaction(
            batch_size=2,
            query_len=3,
            prefixes=prefixes,
            expected_layers=frozenset({1}),
            stages={1: stage},
        )
        self.model.on_cuda_graph_capture(
            2,
            SimpleNamespace(
                attention_inputs={PLE_STATE_TAG: SimpleNamespace(is_target_verify=True)}
            ),
        )

        graph.replay()
        self.model.on_cuda_graph_replay(
            2,
            SimpleNamespace(
                attention_inputs={PLE_STATE_TAG: SimpleNamespace(is_target_verify=True)}
            ),
        )
        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 1], device=device, dtype=torch.int32)
        )
        self.model.finish_speculative_target_commit(True)
        self.model.finish_speculative_target_commit(False)
        torch.testing.assert_close(state_pool, torch.zeros_like(state_pool))
        torch.testing.assert_close(ctx_pool, torch.zeros_like(ctx_pool))

        state_seed.add_(50)
        ids_seed.add_(50)
        graph.replay()
        self.model.on_cuda_graph_replay(
            2,
            SimpleNamespace(
                attention_inputs={PLE_STATE_TAG: SimpleNamespace(is_target_verify=True)}
            ),
        )
        self.model.prepare_speculative_target_commit(
            torch.tensor([3, 2], device=device, dtype=torch.int32)
        )
        self.model.finish_speculative_target_commit(True)
        self.model.finalize_speculative_target_commit()
        torch.testing.assert_close(state_pool[2], state_seed[0, 1:] + 10)
        torch.testing.assert_close(state_pool[4], state_seed[1, :2] + 10)
        torch.testing.assert_close(ctx_pool[2], ids_seed[0, 1:] + 100)
        torch.testing.assert_close(ctx_pool[4], ids_seed[1, :2] + 100)

    def test_async_cpu_graph_target_and_decode_repeated_replay(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("NVIDIA CUDA is required for asynchronous PLE graphs")
        for case in (
            self.test_target_cpu_ple_forward_captures_and_replays_captured_batch,
            self.test_decode_cpu_ple_forward_captures_and_replays_captured_batch,
        ):
            self.setUp()
            case()

    def test_target_ple_forward_captures_and_replays_captured_batch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Graph replay")
        device = torch.device("cuda")
        gpu_embedding = self._move_ple_to_cuda(device)

        prefixes = (self._PAGE - 1, self._PAGE)
        initial_state, initial_context = self._seed_target_history(prefixes)
        initial_state = initial_state.to(device)
        initial_context = initial_context.to(device)
        ids = torch.tensor([[4, 5, 6], [3, 4, 5]], dtype=torch.long, device=device)
        hyper = torch.randn(6, _HC * _HIDDEN, dtype=torch.bfloat16, device=device)
        inputs = self._target_inputs(prefixes, 3)
        for value in inputs.values():
            value.prefix_lengths = value.prefix_lengths.pin_memory()
            value.input_lengths = value.input_lengths.pin_memory()
            value.is_cuda_graph = True
            value.is_s_padded = True

        self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.assertIsNotNone(gpu_embedding._triton_shard_ptrs)
        self.model.on_cuda_graph_warmup(SimpleNamespace(attention_inputs=inputs))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.model.on_cuda_graph_capture(2, SimpleNamespace(attention_inputs=inputs))
        graph.replay()

        baseline, _ = self.ple.decode_chunk(
            hyper.view(2, 3, -1),
            torch.cat([initial_context, ids], dim=1),
            initial_state,
        )
        torch.testing.assert_close(got.view(2, 3, -1), hyper.view(2, 3, -1) + baseline)

        ids.add_(1)
        hyper.add_(0.125)
        graph.replay()
        baseline, _ = self.ple.decode_chunk(
            hyper.view(2, 3, -1),
            torch.cat([initial_context, ids], dim=1),
            initial_state,
        )
        torch.testing.assert_close(got.view(2, 3, -1), hyper.view(2, 3, -1) + baseline)

    def test_target_cpu_ple_forward_captures_and_replays_captured_batch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Graph replay")
        device = torch.device("cuda")
        gpu_embedding = self._move_ple_to_cuda(device, cpu_tables=True)

        prefixes = (self._PAGE - 1, self._PAGE)
        initial_state, initial_context = self._seed_target_history(prefixes)
        initial_state = initial_state.to(device)
        initial_context = initial_context.to(device)
        ids = torch.tensor([[4, 5, 6], [3, 4, 5]], dtype=torch.long, device=device)
        hyper = torch.randn(6, _HC * _HIDDEN, dtype=torch.bfloat16, device=device)
        inputs = self._target_inputs(prefixes, 3)
        for value in inputs.values():
            value.prefix_lengths = value.prefix_lengths.pin_memory()
            value.input_lengths = value.input_lengths.pin_memory()
            value.is_cuda_graph = True
            value.is_s_padded = True

        self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.assertTrue(
            all(shard.device.type == "cpu" for shard in gpu_embedding.shards)
        )
        self.model.on_cuda_graph_warmup(SimpleNamespace(attention_inputs=inputs))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.model.on_cuda_graph_capture(2, SimpleNamespace(attention_inputs=inputs))
        graph.replay()

        self.ple.ple_embedding._graph_gather_required = False
        baseline, _ = self.ple.decode_chunk(
            hyper.view(2, 3, -1),
            torch.cat([initial_context, ids], dim=1),
            initial_state,
        )
        torch.testing.assert_close(got.view(2, 3, -1), hyper.view(2, 3, -1) + baseline)

        ids.add_(1)
        hyper.add_(0.125)
        graph.replay()
        self.ple.ple_embedding._graph_gather_required = False
        baseline, _ = self.ple.decode_chunk(
            hyper.view(2, 3, -1),
            torch.cat([initial_context, ids], dim=1),
            initial_state,
        )
        torch.testing.assert_close(got.view(2, 3, -1), hyper.view(2, 3, -1) + baseline)

        self.assertTrue(
            all(shard.device.type == "cpu" for shard in self.ple.ple_embedding.shards)
        )
        self.assertEqual(self.ple.ple_embedding._cpu_graph_buffers, {})

    def test_decode_ple_forward_captures_and_replays_captured_batch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Graph replay")
        device = torch.device("cuda")
        self._move_ple_to_cuda(device)
        state_pool = PLECacheViews._fixed_state_pool(
            self.model.kv_cache.get_layer_cache(1, PLE_STATE_TAG),
            torch.bfloat16,
            self.ple.short_conv_state_len,
            self.ple.hc_hidden_size,
        )
        state_pool[1].copy_(torch.randn_like(state_pool[1]))
        state_pool[2].copy_(torch.randn_like(state_pool[2]))
        state_pool[4].copy_(torch.randn_like(state_pool[4]))
        self.ctx_base[3].fill_(2)
        self.ctx_base[4].fill_(3)
        self.ctx_base[2].fill_(4)
        initial_state = state_pool[[1, 2]].clone()
        initial_context = self.ctx_base[[3, 4]].clone()
        initial_state_pool = self.state_base.clone()
        initial_context_pool = self.ctx_base.clone()

        ids = torch.tensor([5, 6], dtype=torch.long, device=device)
        hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16, device=device)
        inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=torch.tensor([3, 8], dtype=torch.int32).pin_memory(),
            prefix_lengths=torch.empty(0, dtype=torch.int32).pin_memory(),
            sequence_lengths=torch.tensor([3, 8], dtype=torch.int32).pin_memory(),
        )
        sequence_plus_one = torch.tensor([4, 9], dtype=torch.int32, device=device)
        for value in inputs.values():
            value.is_cuda_graph = True
            value.is_s_padded = True
            value.sequence_lengths_plus_1_device = sequence_plus_one

        def check_replay(state_rows=(1, 4), context_rows=(3, 2)):
            baseline, next_state = self.ple.decode_step(
                hyper.view(2, 1, -1),
                torch.cat([initial_context, ids.view(2, 1)], dim=1),
                initial_state,
            )
            torch.testing.assert_close(got, hyper + baseline.squeeze(1))
            for row, expected in zip(state_rows, next_state):
                torch.testing.assert_close(state_pool[row], expected)
            for row, expected in zip(context_rows, ids):
                torch.testing.assert_close(self.ctx_base[row], expected.view(1))

        self.model._apply_ple(1, hyper, ids, inputs)
        self.state_base.copy_(initial_state_pool)
        self.ctx_base.copy_(initial_context_pool)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = self.model._apply_ple(1, hyper, ids, inputs)
        self.state_base.copy_(initial_state_pool)
        self.ctx_base.copy_(initial_context_pool)
        graph.replay()
        check_replay()

        self.state_base.copy_(initial_state_pool)
        self.ctx_base.copy_(initial_context_pool)
        ids.add_(1)
        hyper.add_(0.125)
        graph.replay()
        check_replay()

        self.state_base.copy_(initial_state_pool)
        self.ctx_base.copy_(initial_context_pool)
        sequence_plus_one.copy_(torch.tensor([9, 10], dtype=torch.int32, device=device))
        self.state_blocks[0, 1] = 7
        self.ctx_blocks[0, 1] = 9
        initial_state = state_pool[[1, 4]].clone()
        initial_context = self.ctx_base[[3, 2]].clone()
        graph.replay()
        check_replay(state_rows=(7, 4), context_rows=(9, 2))

    def test_decode_cpu_ple_forward_captures_and_replays_captured_batch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Graph replay")
        device = torch.device("cuda")
        self._move_ple_to_cuda(device, cpu_tables=True)
        state_pool = PLECacheViews._fixed_state_pool(
            self.model.kv_cache.get_layer_cache(1, PLE_STATE_TAG),
            torch.bfloat16,
            self.ple.short_conv_state_len,
            self.ple.hc_hidden_size,
        )
        state_pool[1].copy_(torch.randn_like(state_pool[1]))
        state_pool[2].copy_(torch.randn_like(state_pool[2]))
        state_pool[4].copy_(torch.randn_like(state_pool[4]))
        self.ctx_base[3].fill_(2)
        self.ctx_base[4].fill_(3)
        self.ctx_base[2].fill_(4)
        initial_state = state_pool[[1, 2]].clone()
        initial_context = self.ctx_base[[3, 4]].clone()
        initial_state_pool = self.state_base.clone()
        initial_context_pool = self.ctx_base.clone()

        ids = torch.tensor([5, 6], dtype=torch.long, device=device)
        hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16, device=device)
        inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=torch.tensor([3, 8], dtype=torch.int32).pin_memory(),
            prefix_lengths=torch.empty(0, dtype=torch.int32).pin_memory(),
            sequence_lengths=torch.tensor([3, 8], dtype=torch.int32).pin_memory(),
        )
        sequence_plus_one = torch.tensor([4, 9], dtype=torch.int32, device=device)
        for value in inputs.values():
            value.is_cuda_graph = True
            value.is_s_padded = True
            value.sequence_lengths_plus_1_device = sequence_plus_one

        def check_replay(state_rows=(1, 4), context_rows=(3, 2)):
            self.ple.ple_embedding._graph_gather_required = False
            baseline, next_state = self.ple.decode_step(
                hyper.view(2, 1, -1),
                torch.cat([initial_context, ids.view(2, 1)], dim=1),
                initial_state,
            )
            torch.testing.assert_close(got, hyper + baseline.squeeze(1))
            for row, expected in zip(state_rows, next_state):
                torch.testing.assert_close(state_pool[row], expected)
            for row, expected in zip(context_rows, ids):
                torch.testing.assert_close(self.ctx_base[row], expected.view(1))

        self.model._apply_ple(1, hyper, ids, inputs)
        self.state_base.copy_(initial_state_pool)
        self.ctx_base.copy_(initial_context_pool)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = self.model._apply_ple(1, hyper, ids, inputs)
        self.state_base.copy_(initial_state_pool)
        self.ctx_base.copy_(initial_context_pool)
        graph.replay()
        check_replay()

        self.state_base.copy_(initial_state_pool)
        self.ctx_base.copy_(initial_context_pool)
        ids.add_(1)
        hyper.add_(0.125)
        graph.replay()
        check_replay()

        self.state_base.copy_(initial_state_pool)
        self.ctx_base.copy_(initial_context_pool)
        sequence_plus_one.copy_(torch.tensor([9, 10], dtype=torch.int32, device=device))
        self.state_blocks[0, 1] = 7
        self.ctx_blocks[0, 1] = 9
        initial_state = state_pool[[1, 4]].clone()
        initial_context = self.ctx_base[[3, 2]].clone()
        graph.replay()
        check_replay(state_rows=(7, 4), context_rows=(9, 2))

        self.assertTrue(
            all(shard.device.type == "cpu" for shard in self.ple.ple_embedding.shards)
        )
        self.assertEqual(self.ple.ple_embedding._cpu_graph_buffers, {})

    def test_unsupported_prefix_cp_and_cuda_graph_fail_fast(self):
        for field, value, error in (
            ("prefix_lengths", torch.tensor([1, 0], dtype=torch.int32), "prefix"),
            ("context_parallel_info", object(), "context parallelism"),
            ("is_cuda_graph", True, "CUDA Graph"),
            ("cache_store_inputs", object(), "PD"),
        ):
            with self.subTest(field=field):
                inputs = self._inputs_by_tag(
                    is_prefill=True,
                    input_lengths=torch.tensor([1, 1], dtype=torch.int32),
                    prefix_lengths=torch.zeros(2, dtype=torch.int32),
                    sequence_lengths=torch.empty(0, dtype=torch.int32),
                )
                setattr(inputs[PLE_STATE_TAG], field, value)
                if field == "prefix_lengths":
                    setattr(inputs[PLE_NGRAM_CTX_TAG], field, value.clone())
                with self.assertRaisesRegex(RuntimeError, error):
                    self.model._apply_ple(
                        1,
                        torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16),
                        torch.ones(2, dtype=torch.long),
                        inputs,
                    )


if __name__ == "__main__":
    main()
