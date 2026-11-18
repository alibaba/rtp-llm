from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG
from rtp_llm.models_py.model_desc import qwen4_exp
from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import (
    _HC,
    _HIDDEN,
    PLERuntimeFixture,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.ple_page_reference import (
    _apply_ple as reference_apply_ple,
)


class Qwen4ExpPLEForwardPreparationTest(TestCase):
    def test_each_forward_shares_fresh_text_ids_between_prefetch_and_ple(self):
        model = qwen4_exp.Qwen4ExpModel.__new__(qwen4_exp.Qwen4ExpModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(special_tokens=SimpleNamespace(eos_token_id=7))
        model.ple_layers = {"0": object()}
        model.layers = [lambda hidden, *args, **kwargs: hidden]
        model.kv_cache = SimpleNamespace(get_layer_cache=lambda *args: object())
        model._try_fp8_hyper_embedding = lambda inputs: None
        model._initial_hyper_states = lambda hidden: hidden
        model._build_attn_meta = lambda *args: None
        model._attention_tag = lambda i: PLE_NGRAM_CTX_TAG
        model._layer_fmha_impl = lambda *args: None
        model.hyper_connection_mixer = lambda hidden: (hidden, None, None)
        model._start_eager_ple_prefetch = Mock(return_value={})
        model._apply_ple = Mock(side_effect=lambda i, hidden, *args, **kwargs: hidden)
        events = []
        model.word_embedding = lambda frame: (
            events.append("embedding") or torch.zeros(frame.input_ids.numel(), _HIDDEN)
        )
        normalize = model._ple_input_ids

        def prepare_ids(frame):
            events.append("ids")
            return normalize(frame)

        with (
            patch.object(model, "_ple_input_ids", side_effect=prepare_ids) as prepare,
            patch.object(
                qwen4_exp, "prepare_state_cache_inputs", side_effect=lambda a, k: a
            ),
            patch.object(
                qwen4_exp, "prepare_side_cache_store_writers", return_value={}
            ),
            patch.object(qwen4_exp, "publish_layer_side_caches"),
            patch.object(qwen4_exp, "_is_cuda_graph_forward", return_value=False),
        ):
            for prefill in (False, True):
                previous = None
                previous_metadata_cache = None
                for tokens in ([1, 2], [3, 4]):
                    with self.subTest(prefill=prefill, tokens=tokens):
                        events.clear()
                        prepare.reset_mock()
                        frame = SimpleNamespace(
                            input_ids=torch.tensor(tokens),
                            embedding_inputs=SimpleNamespace(
                                text_tokens_mask=torch.tensor([True, False])
                            ),
                            attention_inputs={
                                PLE_NGRAM_CTX_TAG: SimpleNamespace(
                                    is_prefill=prefill, is_cuda_graph=False
                                )
                            },
                        )
                        model._forward_with_ple_prefetch(frame, object())
                        prepare.assert_called_once_with(frame)
                        prepared = model._start_eager_ple_prefetch.call_args.args[2]
                        self.assertIs(model._apply_ple.call_args.args[2], prepared)
                        metadata_cache = model._apply_ple.call_args.kwargs[
                            "metadata_cache"
                        ]
                        self.assertIsInstance(metadata_cache, dict)
                        self.assertIsNot(metadata_cache, previous_metadata_cache)
                        self.assertIsNot(prepared, previous)
                        self.assertEqual(prepared.tolist(), [tokens[0], 7])
                        self.assertEqual(frame.input_ids.tolist(), tokens)
                        self.assertEqual(
                            events,
                            ["embedding", "ids"] if prefill else ["ids", "embedding"],
                        )
                        previous = prepared
                        previous_metadata_cache = metadata_cache


class Qwen4ExpPLEPrefetchTest(PLERuntimeFixture, TestCase):
    def test_native_decode_prefetch_replays_current_history_and_commits_original_states(
        self,
    ):
        embedding = self._enable_async_cpu_ple()
        execution = self.model._ple_execution()
        state_table, ctx_table = self.state_blocks.clone(), self.ctx_blocks.clone()
        graph_stream = torch.cuda.Stream()
        graphs = []
        try:
            for batch in (1, 2):
                self.state_blocks = state_table[:batch].clone()
                self.ctx_blocks = ctx_table[:batch].clone()
                host_lengths = torch.full((batch,), 7, dtype=torch.int32)
                device_lengths = (host_lengths + 1).cuda()
                attention = self._inputs_by_tag(
                    is_prefill=False,
                    input_lengths=torch.ones(batch, dtype=torch.int32),
                    prefix_lengths=torch.empty(0, dtype=torch.int32),
                    sequence_lengths=host_lengths,
                )
                for item in attention.values():
                    item.is_cuda_graph = item.is_s_padded = True
                    item.sequence_lengths_plus_1_device = device_lengths
                ids = torch.arange(batch, device="cuda", dtype=torch.long) + 2
                hyper = torch.randn(
                    batch, _HC * _HIDDEN, device="cuda", dtype=torch.bfloat16
                )
                frame = SimpleNamespace(input_ids=ids)
                if batch == 1:
                    ctx_inputs = attention[PLE_NGRAM_CTX_TAG]
                    for flag in ("is_prefill", "is_target_verify"):
                        setattr(ctx_inputs, flag, True)
                        with self.assertRaisesRegex(
                            RuntimeError, "ordinary decode only"
                        ):
                            execution._start_graph_ple_prefetch(frame, attention)
                        setattr(ctx_inputs, flag, False)
                    ctx_inputs.sequence_lengths_plus_1_device = None
                    with self.assertRaisesRegex(RuntimeError, "device decode metadata"):
                        execution._start_graph_ple_prefetch(frame, attention)
                    ctx_inputs.sequence_lengths_plus_1_device = device_lengths

                def run():
                    plans = execution._start_graph_ple_prefetch(frame, attention)
                    output = self.model._apply_ple(
                        1, hyper, ids, attention, prefetch=plans[1]
                    )
                    return output, plans[1][0].bank.rows.clone()

                # Warmup owns all bank allocations; early lookup is read-only.
                before_state, before_ctx = (
                    self.state_base.clone(),
                    self.ctx_base.clone(),
                )
                plans = execution._start_graph_ple_prefetch(frame, attention)
                embedding.consume_prefetch(plans[1][0])
                torch.testing.assert_close(
                    self.state_base, before_state, atol=0, rtol=0
                )
                torch.testing.assert_close(self.ctx_base, before_ctx, atol=0, rtol=0)
                run()
                embedding.finish_graph_warmup()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_stream):
                    actual, actual_rows = run()
                graphs.append(graph)
                for step, positions in enumerate(
                    ((7, 8), (8, 15), (15, 16), (0, 7), (7, 0), (0, 0))
                ):
                    positions = positions[:batch]
                    self.state_base.normal_()
                    self.ctx_base.random_(1, 8)
                    self.state_blocks.copy_(state_table[:batch].roll(step, dims=1))
                    self.ctx_blocks.copy_(ctx_table[:batch].roll(step + 1, dims=1))
                    ids.random_(1, 10)
                    hyper.normal_()
                    device_lengths.copy_(
                        torch.tensor(positions, device="cuda", dtype=torch.int32) + 1
                    )
                    expected_state, expected_ctx = (
                        self.state_base.clone(),
                        self.ctx_base.clone(),
                    )
                    expected = hyper.clone()
                    expected_rows = torch.zeros_like(actual_rows)
                    for row, position in enumerate(positions):
                        if position == 0:
                            self.state_blocks[row].zero_()
                            self.ctx_blocks[row].zero_()
                            continue
                        read_page = (position - 1) // self._PAGE
                        write_page = position // self._PAGE
                        state_read = int(self.state_blocks[row, read_page])
                        ctx_read = int(self.ctx_blocks[row, read_page])
                        history = torch.cat(
                            (
                                expected_ctx[ctx_read : ctx_read + 1],
                                ids[row : row + 1, None],
                            ),
                            dim=1,
                        )
                        # Independent dense CPU lookup for this ngram=2 fixture.
                        mixed = history[:, -1:].cpu() * 3 ^ (
                            history[:, -2:-1].cpu() * 5
                        )
                        hashed = mixed[:, :, None].remainder(
                            torch.tensor([5, 7])
                        ) + torch.tensor([0, 5])
                        rows = embedding.shards[0][hashed].cuda()
                        expected_rows[row : row + 1] = rows
                        output, new_state = self.ple.decode_step(
                            hyper[row : row + 1, None],
                            history,
                            expected_state[state_read : state_read + 1].view(
                                1, self.ple.short_conv_state_len, -1
                            ),
                            embeddings=rows.flatten(-2),
                        )
                        expected[row] += output[0, 0]
                        expected_state[self.state_blocks[row, write_page].long()] = (
                            new_state.flatten()
                        )
                        expected_ctx[self.ctx_blocks[row, write_page].long()] = ids[row]
                    graph.replay()
                    live = torch.tensor(positions, device="cuda").gt(0)
                    torch.testing.assert_close(
                        actual[live], expected[live], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        actual_rows, expected_rows, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        self.state_base, expected_state, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        self.ctx_base, expected_ctx, atol=0, rtol=0
                    )
                    self.assertTrue(torch.all(host_lengths == 7))
            self.assertFalse(embedding._prefetcher._active_slots)
        finally:
            for graph in graphs:
                graph.reset()
            embedding.close()

    def test_async_prefetch_ragged_prefill_preserves_every_page_and_prefix(self):
        embedding = self._enable_async_cpu_ple()
        device = torch.device("cuda")
        with self.subTest(mode="automatic"):
            self.state_base.normal_()
            self.ctx_base.random_(1, 7)
            before_state, before_context = (
                self.state_base.clone(),
                self.ctx_base.clone(),
            )
            ids = torch.randint(0, 10, (26,), device=device, dtype=torch.long)
            hyper = torch.randn(26, _HC * _HIDDEN, device=device, dtype=torch.bfloat16)
            inputs = self._inputs_by_tag(
                is_prefill=True,
                input_lengths=torch.tensor([17, 9], dtype=torch.int32),
                prefix_lengths=torch.tensor([0, 8], dtype=torch.int32),
                sequence_lengths=torch.empty(0, dtype=torch.int32),
            )
            expected = reference_apply_ple(
                self.model._ple_execution(), 1, hyper, ids, inputs
            )
            expected_state, expected_context = (
                self.state_base.clone(),
                self.ctx_base.clone(),
            )
            self.state_base.copy_(before_state)
            self.ctx_base.copy_(before_context)
            plans = self.model._start_eager_ple_prefetch(
                SimpleNamespace(input_ids=ids), inputs
            )
            torch.testing.assert_close(self.state_base, before_state, atol=0, rtol=0)
            torch.testing.assert_close(self.ctx_base, before_context, atol=0, rtol=0)
            actual = self.model._apply_ple(1, hyper, ids, inputs, prefetch=plans[1])
            embedding.finish_prefetch()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(self.state_base, expected_state, atol=0, rtol=0)
            torch.testing.assert_close(self.ctx_base, expected_context, atol=0, rtol=0)

    def test_async_target_verify_commit_and_rollback_preserve_state(self):
        embedding = self._enable_async_cpu_ple()
        device = torch.device("cuda")
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        before_state, before_context = self.state_base.clone(), self.ctx_base.clone()
        ids = torch.tensor([[4, 5, 6], [3, 4, 5]], device=device, dtype=torch.long)
        hyper = torch.randn(6, _HC * _HIDDEN, device=device, dtype=torch.bfloat16)
        inputs = self._target_inputs(prefixes, 3)
        expected = self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 3], device=device, dtype=torch.int32)
        )
        self.model.finish_speculative_target_commit(True)
        committed_state, committed_context = (
            self.state_base.clone(),
            self.ctx_base.clone(),
        )
        self.model.finish_speculative_target_commit(False)
        plans = self.model._start_eager_ple_prefetch(
            SimpleNamespace(input_ids=ids.flatten()), inputs
        )
        actual = self.model._apply_ple(
            1, hyper, ids.flatten(), inputs, prefetch=plans[1]
        )
        embedding.finish_prefetch()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(self.state_base, before_state, atol=0, rtol=0)
        torch.testing.assert_close(self.ctx_base, before_context, atol=0, rtol=0)

        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 3], device=device, dtype=torch.int32)
        )
        self.model.finish_speculative_target_commit(True)
        torch.testing.assert_close(self.state_base, committed_state, atol=0, rtol=0)
        torch.testing.assert_close(self.ctx_base, committed_context, atol=0, rtol=0)
        self.model.finish_speculative_target_commit(False)
        torch.testing.assert_close(self.state_base, before_state, atol=0, rtol=0)
        torch.testing.assert_close(self.ctx_base, before_context, atol=0, rtol=0)

    def test_async_budget_fallback_preserves_prefill_pages(self):
        embedding = self._enable_async_cpu_ple()
        embedding._prefetcher._max_eager_bytes = 1
        ids = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.long)
        hyper = torch.randn(4, _HC * _HIDDEN, device="cuda", dtype=torch.bfloat16)
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([3, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        before_state, before_context = self.state_base.clone(), self.ctx_base.clone()
        expected = reference_apply_ple(
            self.model._ple_execution(), 1, hyper, ids, inputs
        )
        expected_state, expected_context = (
            self.state_base.clone(),
            self.ctx_base.clone(),
        )
        self.state_base.copy_(before_state)
        self.ctx_base.copy_(before_context)
        plans = self.model._start_eager_ple_prefetch(
            SimpleNamespace(input_ids=ids), inputs
        )
        self.assertTrue(all(handle is None for handle in plans[1]))
        actual = self.model._apply_ple(1, hyper, ids, inputs, prefetch=plans[1])
        embedding.finish_prefetch()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(self.state_base, expected_state, atol=0, rtol=0)
        torch.testing.assert_close(self.ctx_base, expected_context, atol=0, rtol=0)

    def test_async_multimodal_prefetch_uses_eos_and_forward_finally_releases_slots(
        self,
    ):
        embedding = self._enable_async_cpu_ple()
        ids = torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.long)
        mask = torch.tensor([True, False, True, True], device="cuda")
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([3, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        frame = SimpleNamespace(
            input_ids=ids, embedding_inputs=SimpleNamespace(text_tokens_mask=mask)
        )
        plans = self.model._start_eager_ple_prefetch(frame, inputs)
        history = torch.tensor([[7, 1, 7, 3]], device="cuda", dtype=torch.long)
        expected = embedding._gather_torch(embedding.hashed_ids(history, 3).cpu()).to(
            "cuda"
        )
        torch.testing.assert_close(
            embedding.consume_prefetch(plans[1][0]), expected, atol=0, rtol=0
        )
        with patch.object(
            self.model,
            "_forward_with_ple_prefetch",
            side_effect=RuntimeError("before consume"),
        ):
            with self.assertRaisesRegex(RuntimeError, "before consume"):
                self.model.forward(frame)
        self.assertFalse(embedding._prefetcher._active_slots)


if __name__ == "__main__":
    main()
