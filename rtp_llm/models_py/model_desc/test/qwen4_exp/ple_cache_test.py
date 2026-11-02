from types import SimpleNamespace
from unittest import TestCase, main

import torch
from torch import nn

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import (
    _HC,
    _HIDDEN,
    PLERuntimeFixture,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.ple_page_reference import (
    _apply_ple as reference_apply_ple,
)


class Qwen4ExpPLECacheTest(PLERuntimeFixture, TestCase):
    def test_bulk_prefill_preserves_ragged_page_writes_and_reused_prefixes(self):
        for prefixes, lengths in (((0, 0), (17, 16)), ((8, 8), (16, 16))):
            with self.subTest(prefixes=prefixes, lengths=lengths):
                self.state_base.normal_()
                self.ctx_base.random_(1, 7)
                state_before = self.state_base.clone()
                context_before = self.ctx_base.clone()
                hyper = torch.randn(sum(lengths), _HC * _HIDDEN, dtype=torch.bfloat16)
                ids = torch.randint(0, 10, (sum(lengths),), dtype=torch.long)
                inputs = self._inputs_by_tag(
                    is_prefill=True,
                    input_lengths=torch.tensor(lengths, dtype=torch.int32),
                    prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
                    sequence_lengths=torch.empty(0, dtype=torch.int32),
                )
                expected = reference_apply_ple(
                    self.model._ple_execution(), 1, hyper, ids, inputs
                )
                expected_state = self.state_base.clone()
                expected_context = self.ctx_base.clone()
                self.state_base.copy_(state_before)
                self.ctx_base.copy_(context_before)
                actual = self.model._apply_ple(1, hyper, ids, inputs)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(self.state_base, expected_state)
                torch.testing.assert_close(
                    self.ctx_base, expected_context, atol=0, rtol=0
                )

    def test_ragged_prefill_and_decode_use_tag_local_state_pools(self):
        ids, hyper, got = self._prefill()
        expected = []
        offset = 0
        for length in (3, 1):
            seq = ids[offset : offset + length].view(1, length)
            history = torch.cat([seq.new_full((1, 1), 7), seq], dim=1)
            expected.append(
                self.ple(hyper[offset : offset + length].view(1, length, -1), history)[
                    0
                ]
            )
            offset += length
        torch.testing.assert_close(got, hyper + torch.cat(expected))
        torch.testing.assert_close(self.ctx_base[3], torch.tensor([3]))
        torch.testing.assert_close(self.ctx_base[4], torch.tensor([4]))

        decode_ids = torch.tensor([5, 6], dtype=torch.long)
        decode_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        old_state = self.state_base[[1, 2]].view(2, self.ple.short_conv_state_len, -1)
        old_ctx = self.ctx_base[[3, 4]]
        history = torch.cat([old_ctx, decode_ids.view(2, 1)], dim=1)
        expected_out, expected_state = self.ple.decode_step(
            decode_hyper.view(2, 1, -1), history, old_state
        )
        inputs = self._inputs_by_tag(
            is_prefill=False,
            # Decode input_lengths carries each request's original prompt
            # length; one current token per row is represented by ids.numel().
            input_lengths=torch.tensor([3, 1], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([3, 1], dtype=torch.int32),
        )
        got = self.model._apply_ple(1, decode_hyper, decode_ids, inputs)
        torch.testing.assert_close(got, decode_hyper + expected_out.squeeze(1))
        torch.testing.assert_close(
            self.state_base[[1, 2]].view_as(expected_state), expected_state
        )
        torch.testing.assert_close(self.ctx_base[[3, 4]], decode_ids.view(2, 1))

    def test_padded_target_commit_slices_live_rows_and_preserves_undo(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        ids = torch.tensor([[4, 5, 6], [3, 4, 5]], dtype=torch.long)
        hyper = torch.randn(6, _HC * _HIDDEN, dtype=torch.bfloat16)
        inputs = self._target_inputs(prefixes, 3)
        self.model._apply_ple(1, hyper, ids.flatten(), inputs)
        self.model.on_cuda_graph_capture(2, SimpleNamespace(attention_inputs=inputs))
        captured = self.model._ple_graph_target_transactions[2]
        before_state, before_ctx = self.state_base.clone(), self.ctx_base.clone()
        # Capacity two is reused by live batches one, two, then one. Padding
        # must never reach destination validation, candidate selection or undo.
        for accepted in ([3], [1, 2], [2]):
            self.model.on_cuda_graph_replay(2, SimpleNamespace(attention_inputs=inputs))
            self.model.prepare_speculative_target_commit(
                torch.tensor(accepted, dtype=torch.int32)
            )
            writes = captured.prepared_writes
            self.assertEqual(writes[0].state_blocks.numel(), len(accepted))
            self.model.finish_speculative_target_commit(True)
            if len(accepted) == 1:
                torch.testing.assert_close(
                    self.state_base[[2, 4, 8]], before_state[[2, 4, 8]]
                )
                torch.testing.assert_close(
                    self.ctx_base[[4, 2, 10]], before_ctx[[4, 2, 10]]
                )
            self.model.finish_speculative_target_commit(False)
            torch.testing.assert_close(self.state_base, before_state)
            torch.testing.assert_close(self.ctx_base, before_ctx)

    def test_target_verify_commit_finalize_and_next_decode_match_baseline(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        query_len = 5
        accepted = torch.tensor([1, query_len], dtype=torch.int32)
        initial_state, initial_context = self._seed_target_history(prefixes)
        ids = torch.tensor([[4, 5, 6, 8, 9], [3, 4, 5, 6, 8]], dtype=torch.long)
        hyper = torch.randn(2, query_len, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected_output, states, contexts = self._target_baseline(
            hyper, ids, initial_state, initial_context
        )
        inputs = self._target_inputs(prefixes, query_len)
        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()

        got = self.model._apply_ple(
            1, hyper.reshape(-1, _HC * _HIDDEN), ids.reshape(-1), inputs
        )

        torch.testing.assert_close(got.reshape_as(hyper), hyper + expected_output)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

        self.model.prepare_speculative_target_commit(accepted)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

        self.model.finish_speculative_target_commit(True)
        final_positions = torch.tensor(prefixes) + accepted.to(torch.long) - 1
        final_pages = torch.div(final_positions, self._PAGE, rounding_mode="floor")
        state_destinations = self.state_blocks.gather(
            1, final_pages.unsqueeze(1).to(torch.long)
        ).squeeze(1)
        context_destinations = self.ctx_blocks.gather(
            1, final_pages.unsqueeze(1).to(torch.long)
        ).squeeze(1)
        expected_state = torch.stack(
            [states[int(accepted[b]) - 1][b] for b in range(2)]
        )
        expected_context = torch.stack(
            [contexts[int(accepted[b]) - 1][b] for b in range(2)]
        )
        torch.testing.assert_close(
            self.state_base.index_select(0, state_destinations).view_as(expected_state),
            expected_state,
        )
        torch.testing.assert_close(
            self.ctx_base.index_select(0, context_destinations), expected_context
        )
        self.assertIsNotNone(self.model._ple_target_transaction)

        self.model.finalize_speculative_target_commit()
        self.assertIsNone(self.model._ple_target_transaction)

        next_ids = torch.tensor([10, 11], dtype=torch.long)
        next_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected_next, expected_next_state = self.ple.decode_step(
            next_hyper.view(2, 1, -1),
            torch.cat([expected_context, next_ids.view(2, 1)], dim=1),
            expected_state,
        )
        committed_lengths = torch.tensor(prefixes) + accepted
        decode_inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=committed_lengths.to(torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=committed_lengths.to(torch.int32),
        )
        got_next = self.model._apply_ple(1, next_hyper, next_ids, decode_inputs)
        torch.testing.assert_close(got_next, next_hyper + expected_next.squeeze(1))
        next_write_pages = torch.div(
            committed_lengths, self._PAGE, rounding_mode="floor"
        )
        next_state_destinations = self.state_blocks.gather(
            1, next_write_pages.unsqueeze(1).to(torch.long)
        ).squeeze(1)
        torch.testing.assert_close(
            self.state_base.index_select(0, next_state_destinations).view_as(
                expected_next_state
            ),
            expected_next_state,
        )

    def test_target_commit_refreshes_the_page_an_accepted_window_crosses(self):
        prefixes = (self._PAGE - 2, 2 * self._PAGE + 2)
        query_len = 5
        accepted = torch.tensor([4, 2], dtype=torch.int32)
        initial_state, initial_context = self._seed_target_history(prefixes)
        ids = torch.tensor([[4, 5, 6, 8, 9], [3, 4, 5, 6, 8]], dtype=torch.long)
        hyper = torch.randn(2, query_len, _HC * _HIDDEN, dtype=torch.bfloat16)
        _, states, contexts = self._target_baseline(
            hyper, ids, initial_state, initial_context
        )
        inputs = self._target_inputs(prefixes, query_len)

        self.model._apply_ple(
            1, hyper.reshape(-1, _HC * _HIDDEN), ids.reshape(-1), inputs
        )
        self.model.prepare_speculative_target_commit(accepted)
        self.model.finish_speculative_target_commit(True)

        # Request 0 starts at prefix 6 and accepts 4 tokens, crossing page
        # boundary 8: the page it completes must now hold the checkpoint that
        # ends at that boundary (the window after query token 1), not the one
        # this step originally read.
        crossed_state_row = int(self.state_blocks[0, 0])
        crossed_ctx_row = int(self.ctx_blocks[0, 0])
        torch.testing.assert_close(
            self.state_base[crossed_state_row].view_as(states[1][0]), states[1][0]
        )
        torch.testing.assert_close(self.ctx_base[crossed_ctx_row], contexts[1][0])

        # The final destinations keep the windows ending at the last accepted
        # token; request 1 never crosses a boundary and stays unaffected.
        final_positions = torch.tensor(prefixes) + accepted.to(torch.long) - 1
        final_pages = torch.div(final_positions, self._PAGE, rounding_mode="floor")
        state_destinations = self.state_blocks.gather(
            1, final_pages.unsqueeze(1).to(torch.long)
        ).squeeze(1)
        ctx_destinations = self.ctx_blocks.gather(
            1, final_pages.unsqueeze(1).to(torch.long)
        ).squeeze(1)
        expected_state = torch.stack(
            [states[int(accepted[b]) - 1][b] for b in range(2)]
        )
        expected_context = torch.stack(
            [contexts[int(accepted[b]) - 1][b] for b in range(2)]
        )
        torch.testing.assert_close(
            self.state_base.index_select(0, state_destinations).view_as(expected_state),
            expected_state,
        )
        torch.testing.assert_close(
            self.ctx_base.index_select(0, ctx_destinations), expected_context
        )
        self.model.finalize_speculative_target_commit()

    def test_target_verify_tentative_commit_can_be_rolled_back_idempotently(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        query_len = 3
        self._seed_target_history(prefixes)
        inputs = self._target_inputs(prefixes, query_len)
        hyper = torch.randn(2, query_len, _HC * _HIDDEN, dtype=torch.bfloat16)
        ids = torch.tensor([[4, 5, 6], [8, 9, 10]], dtype=torch.long)
        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()
        self.model._apply_ple(
            1, hyper.reshape(-1, _HC * _HIDDEN), ids.reshape(-1), inputs
        )
        self.model.prepare_speculative_target_commit(
            torch.tensor([2, 3], dtype=torch.int32)
        )

        self.model.finish_speculative_target_commit(True)
        self.assertFalse(torch.equal(self.state_base, state_before))
        self.assertFalse(torch.equal(self.ctx_base, context_before))

        self.model.finish_speculative_target_commit(False)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)
        self.assertIsNone(self.model._ple_target_transaction)
        self.model.finish_speculative_target_commit(False)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

    def test_target_prepare_rejects_invalid_accept_len_without_side_effects(self):
        cases = (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1, 4], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int32),
        )
        for accept_len in cases:
            with self.subTest(accept_len=accept_len):
                prefixes = (self._PAGE - 1, self._PAGE)
                self._seed_target_history(prefixes)
                inputs = self._target_inputs(prefixes, 3)
                state_before = self.state_base.clone()
                context_before = self.ctx_base.clone()
                self.model._apply_ple(
                    1,
                    torch.randn(6, _HC * _HIDDEN, dtype=torch.bfloat16),
                    torch.tensor([4, 5, 6, 8, 9, 10]),
                    inputs,
                )

                with self.assertRaisesRegex(RuntimeError, "accept_len"):
                    self.model.prepare_speculative_target_commit(accept_len)

                torch.testing.assert_close(self.state_base, state_before)
                torch.testing.assert_close(self.ctx_base, context_before)
                self.model.finish_speculative_target_commit(False)

    def test_target_prepare_rejects_missing_or_duplicate_destinations_before_write(
        self,
    ):
        prefixes = (2 * self._PAGE - 1, 2 * self._PAGE - 1)
        query_len = 2
        for failure in ("missing", "duplicate"):
            with self.subTest(failure=failure):
                self.state_base.zero_()
                self.ctx_base.zero_()
                self.model._ple_target_transaction = None
                initial_state, initial_context = self._seed_target_history(prefixes)
                self.assertEqual(initial_state.shape[0], 2)
                self.assertEqual(initial_context.shape[0], 2)
                inputs = self._target_inputs(prefixes, query_len)
                if failure == "missing":
                    inputs[PLE_STATE_TAG].kv_cache_block_id_device = (
                        self.state_blocks.clone()
                    )
                    inputs[PLE_STATE_TAG].kv_cache_block_id_device[0, 2] = 0
                else:
                    duplicate = self.state_blocks.clone()
                    duplicate[1, 2] = duplicate[0, 2]
                    inputs[PLE_STATE_TAG].kv_cache_block_id_device = duplicate
                state_before = self.state_base.clone()
                context_before = self.ctx_base.clone()
                self.model._apply_ple(
                    1,
                    torch.randn(
                        2 * query_len,
                        _HC * _HIDDEN,
                        dtype=torch.bfloat16,
                    ),
                    torch.tensor([4, 5, 8, 9]),
                    inputs,
                )

                expected = "unallocated" if failure == "missing" else "duplicate"
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.model.prepare_speculative_target_commit(
                        torch.tensor([2, 2], dtype=torch.int32)
                    )

                torch.testing.assert_close(self.state_base, state_before)
                torch.testing.assert_close(self.ctx_base, context_before)
                self.model.finish_speculative_target_commit(False)

    def test_target_prepare_requires_every_ple_layer_before_any_write(self):
        self.model.ple_layers["2"] = self.ple
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        inputs = self._target_inputs(prefixes, 2)
        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()
        self.model._apply_ple(
            1,
            torch.randn(4, _HC * _HIDDEN, dtype=torch.bfloat16),
            torch.tensor([4, 5, 8, 9]),
            inputs,
        )

        with self.assertRaisesRegex(RuntimeError, r"missing=\[2\]"):
            self.model.prepare_speculative_target_commit(
                torch.tensor([1, 2], dtype=torch.int32)
            )

        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)
        self.model.finish_speculative_target_commit(False)

    def test_target_prepare_syncs_before_publishing_and_abort_syncs_before_clear(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        inputs = self._target_inputs(prefixes, 2)
        self.model._apply_ple(
            1,
            torch.randn(4, _HC * _HIDDEN, dtype=torch.bfloat16),
            torch.tensor([4, 5, 8, 9]),
            inputs,
        )
        transaction = self.model._ple_target_transaction
        self.assertIsNotNone(transaction)
        sync_observations = []

        def observe_sync(current):
            sync_observations.append(
                (
                    current.prepared_writes is None,
                    self.model._ple_target_transaction is current,
                )
            )

        self.model._ple_coordinator()._synchronize_ple_transaction = observe_sync
        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 2], dtype=torch.int32)
        )
        self.assertEqual(sync_observations, [(True, True)])
        self.assertIsNotNone(transaction.prepared_writes)

        self.model.finish_speculative_target_commit(False)
        self.assertEqual(sync_observations, [(True, True), (False, True)])
        self.assertIsNone(self.model._ple_target_transaction)

    def test_target_prepare_sync_failure_is_settled_by_abort(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        inputs = self._target_inputs(prefixes, 2)
        self.model._apply_ple(
            1,
            torch.randn(4, _HC * _HIDDEN, dtype=torch.bfloat16),
            torch.tensor([4, 5, 8, 9]),
            inputs,
        )
        transaction = self.model._ple_target_transaction
        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()
        sync_calls = []

        def fail_prepare_sync(current):
            sync_calls.append(current)
            raise RuntimeError("injected prepare synchronize failure")

        self.model._ple_coordinator()._synchronize_ple_transaction = fail_prepare_sync
        with self.assertRaisesRegex(RuntimeError, "injected prepare"):
            self.model.prepare_speculative_target_commit(
                torch.tensor([1, 2], dtype=torch.int32)
            )
        self.assertEqual(len(sync_calls), 1)
        self.assertIs(sync_calls[0], transaction)
        self.assertIsNone(transaction.prepared_writes)
        self.assertIs(self.model._ple_target_transaction, transaction)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

        self.model._ple_coordinator()._synchronize_ple_transaction = sync_calls.append
        self.model.finish_speculative_target_commit(False)
        self.assertEqual(len(sync_calls), 2)
        self.assertIs(sync_calls[1], transaction)
        self.assertIsNone(self.model._ple_target_transaction)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

    def test_target_prepare_without_staging_fails_when_ple_is_enabled(self):
        self.assertIsNone(getattr(self.model, "_ple_target_transaction", None))
        with self.assertRaisesRegex(RuntimeError, "no staged transaction"):
            self.model.prepare_speculative_target_commit(
                torch.tensor([1, 1], dtype=torch.int32)
            )

        self.model.ple_layers.clear()
        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 1], dtype=torch.int32)
        )

    def test_multimodal_placeholders_hash_as_eos(self):
        """Vision positions hash as EOS, mirroring upstream's PLE ids."""
        model_inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2, 99, 4], dtype=torch.long),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([True, True, False, True])
            ),
        )
        torch.testing.assert_close(
            self.model._ple_input_ids(model_inputs),
            torch.tensor([1, 2, 7, 4], dtype=torch.long),
        )

        text_only = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.long),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([True, True])
            ),
        )
        torch.testing.assert_close(
            self.model._ple_input_ids(text_only),
            torch.tensor([1, 2], dtype=torch.long),
        )

        # The mask alone drives the substitution; features are not required.
        features_only = SimpleNamespace(
            input_ids=torch.tensor([5, 6], dtype=torch.long),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([False, True])
            ),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[torch.ones(1, _HIDDEN)]
            ),
        )
        torch.testing.assert_close(
            self.model._ple_input_ids(features_only),
            torch.tensor([7, 6], dtype=torch.long),
        )

        eos = self.model.config.special_tokens.eos_token_id
        self.model.config.special_tokens.eos_token_id = None
        try:
            with self.assertRaisesRegex(RuntimeError, "eos token id"):
                self.model._ple_input_ids(features_only)
        finally:
            self.model.config.special_tokens.eos_token_id = eos

    def test_ple_input_ids_rejects_mismatched_text_mask(self):
        model_inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.long),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([True], dtype=torch.bool)
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "mask size"):
            self.model._ple_input_ids(model_inputs)

    def test_decode_moves_state_across_a_page_boundary(self):
        ids = torch.arange(1, 2 * self._PAGE + 1, dtype=torch.long)
        hyper = torch.randn(2 * self._PAGE, _HC * _HIDDEN, dtype=torch.bfloat16)
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([self._PAGE, self._PAGE], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        self.model._apply_ple(1, hyper, ids, inputs)

        first_state = (
            self.state_base[[1, 2]].clone().view(2, self.ple.short_conv_state_len, -1)
        )
        first_ctx = self.ctx_base[[3, 4]].clone()
        decode_ids = torch.tensor([21, 22], dtype=torch.long)
        decode_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected_out, expected_state = self.ple.decode_step(
            decode_hyper.view(2, 1, -1),
            torch.cat([first_ctx, decode_ids.view(2, 1)], dim=1),
            first_state,
        )
        decode_inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=torch.tensor([self._PAGE, self._PAGE], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([self._PAGE, self._PAGE], dtype=torch.int32),
        )

        got = self.model._apply_ple(1, decode_hyper, decode_ids, decode_inputs)

        torch.testing.assert_close(got, decode_hyper + expected_out.squeeze(1))
        torch.testing.assert_close(
            self.state_base[[5, 4]].view_as(expected_state), expected_state
        )
        torch.testing.assert_close(self.ctx_base[[1, 2]], decode_ids.view(2, 1))
        torch.testing.assert_close(
            self.state_base[[1, 2]].view_as(first_state), first_state
        )
        torch.testing.assert_close(self.ctx_base[[3, 4]], first_ctx)

        # The next step is wholly inside page 1: it must consume and update the
        # migrated state rather than falling back to page 0.
        next_ids = torch.tensor([23, 24], dtype=torch.long)
        next_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected_out, expected_state = self.ple.decode_step(
            next_hyper.view(2, 1, -1),
            torch.cat([decode_ids.view(2, 1), next_ids.view(2, 1)], dim=1),
            expected_state,
        )
        for tag in (PLE_STATE_TAG, PLE_NGRAM_CTX_TAG):
            decode_inputs[tag].sequence_lengths = torch.tensor(
                [self._PAGE + 1, self._PAGE + 1], dtype=torch.int32
            )
        got = self.model._apply_ple(1, next_hyper, next_ids, decode_inputs)
        torch.testing.assert_close(got, next_hyper + expected_out.squeeze(1))
        torch.testing.assert_close(
            self.state_base[[5, 4]].view_as(expected_state), expected_state
        )
        torch.testing.assert_close(self.ctx_base[[1, 2]], next_ids.view(2, 1))

    def test_decode_rejects_an_unallocated_current_page(self):
        inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=torch.tensor([self._PAGE, 1], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([self._PAGE, 1], dtype=torch.int32),
        )
        inputs[PLE_STATE_TAG].kv_cache_block_id_device = self.state_blocks.clone()
        inputs[PLE_STATE_TAG].kv_cache_block_id_device[0, 1] = 0

        with self.assertRaisesRegex(RuntimeError, "unallocated logical page"):
            self.model._apply_ple(
                1,
                torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(2, dtype=torch.long),
                inputs,
            )

    def test_page_aligned_prefix_prefill_matches_one_full_prefill(self):
        """Prefix reuse: resumed prefills equal the same tokens in one prefill.

        Request 1 prefills three pages; requests 2 and 3 reuse the stored
        prefix page by page. Each step's outputs -- and therefore the
        state/context the previous step wrote -- must match the single
        32-token prefill.
        """
        chunk = self._PAGE
        # Request 1 writes 24 tokens (terminal page 2) and -- under the
        # checkpoint contract -- pages 0/1 as well; request 2 reuses the page-1
        # boundary, so it reads a checkpoint rather than a terminal write.
        steps = (
            (0, 3 * chunk, 0),
            (2 * chunk, chunk, 2 * chunk),
            (3 * chunk, chunk, 3 * chunk),
        )
        # One request walks four logical pages, so the block tables need four
        # columns (the shared fixture only has three).
        state_blocks = torch.tensor([[1, 5, 7, 3]], dtype=torch.int32)
        ctx_blocks = torch.tensor([[3, 1, 9, 4]], dtype=torch.int32)

        torch.manual_seed(11)
        ids = torch.arange(1, 4 * chunk + 1, dtype=torch.long)
        hyper = torch.randn(4 * chunk, _HC * _HIDDEN, dtype=torch.bfloat16)

        outputs = []
        for lo, length, prefix_len in steps:
            inputs = {
                PLE_STATE_TAG: self._side_inputs(
                    True,
                    torch.tensor([length], dtype=torch.int32),
                    torch.tensor([prefix_len], dtype=torch.int32),
                    torch.empty(0, dtype=torch.int32),
                    state_blocks,
                ),
                PLE_NGRAM_CTX_TAG: self._side_inputs(
                    True,
                    torch.tensor([length], dtype=torch.int32),
                    torch.tensor([prefix_len], dtype=torch.int32),
                    torch.empty(0, dtype=torch.int32),
                    ctx_blocks,
                ),
            }
            got = self.model._apply_ple(
                1, hyper[lo : lo + length], ids[lo : lo + length], inputs
            )
            outputs.append((lo, got))

        full_history = torch.cat(
            [ids.new_full((1, self.ple.ple_embedding.context_len), 7), ids.view(1, -1)],
            dim=1,
        )
        full_out, _ = self.ple.prefill(hyper.view(1, -1, _HC * _HIDDEN), full_history)
        for lo, got in outputs:
            length = int(got.shape[0])
            torch.testing.assert_close(
                got, hyper[lo : lo + length] + full_out[0, lo : lo + length, :]
            )

    def test_multi_page_ragged_prefill_checkpoints_every_page_then_decodes(self):
        lengths = (self._PAGE + 3, 2 * self._PAGE)
        token_count = sum(lengths)
        ids = torch.arange(1, token_count + 1, dtype=torch.long)
        hyper = torch.randn(token_count, _HC * _HIDDEN, dtype=torch.bfloat16)
        prefill_inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )

        got = self.model._apply_ple(1, hyper, ids, prefill_inputs)

        expected_outputs = []
        expected_states = []
        expected_contexts = []
        offset = 0
        for length in lengths:
            seq_ids = ids[offset : offset + length].view(1, length)
            history = torch.cat(
                [
                    seq_ids.new_full((1, self.ple.ple_embedding.context_len), 7),
                    seq_ids,
                ],
                dim=1,
            )
            output, state = self.ple.prefill(
                hyper[offset : offset + length].view(1, length, -1), history
            )
            expected_outputs.append(output.squeeze(0))
            expected_states.append(state.squeeze(0))
            expected_contexts.append(history[0, -self.ple.ple_embedding.context_len :])
            offset += length

        expected_states = torch.stack(expected_states)
        expected_contexts = torch.stack(expected_contexts)
        torch.testing.assert_close(got, hyper + torch.cat(expected_outputs, dim=0))

        # Both prompts end on logical page 1. Their page-0 rows carry the
        # intermediate checkpoints (a later request may reuse that boundary),
        # while the reserved page-2 rows stay untouched even though the
        # physical ids are sparse and differ between the two typed pools.
        terminal_state_blocks = self.state_blocks[:, 1].to(torch.long)
        terminal_ctx_blocks = self.ctx_blocks[:, 1].to(torch.long)
        torch.testing.assert_close(
            self.state_base.index_select(0, terminal_state_blocks).view_as(
                expected_states
            ),
            expected_states,
        )
        torch.testing.assert_close(
            self.ctx_base.index_select(0, terminal_ctx_blocks), expected_contexts
        )
        untouched_state_blocks = self.state_blocks[:, [2]].reshape(-1).to(torch.long)
        untouched_ctx_blocks = self.ctx_blocks[:, [2]].reshape(-1).to(torch.long)
        self.assertEqual(
            int(
                torch.count_nonzero(
                    self.state_base.index_select(0, untouched_state_blocks)
                ).item()
            ),
            0,
        )
        self.assertEqual(
            int(
                torch.count_nonzero(
                    self.ctx_base.index_select(0, untouched_ctx_blocks)
                ).item()
            ),
            0,
        )

        decode_ids = torch.tensor([31, 32], dtype=torch.long)
        decode_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        decode_history = torch.cat([expected_contexts, decode_ids.view(2, 1)], dim=1)
        expected_output, expected_new_state = self.ple.decode_step(
            decode_hyper.view(2, 1, -1), decode_history, expected_states
        )
        decode_inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor(lengths, dtype=torch.int32),
        )

        got = self.model._apply_ple(1, decode_hyper, decode_ids, decode_inputs)

        torch.testing.assert_close(got, decode_hyper + expected_output.squeeze(1))
        # P+3 stays within logical page 1; 2P crosses from page 1 to page 2.
        torch.testing.assert_close(
            self.state_base[5].view_as(expected_new_state[0]),
            expected_new_state[0],
        )
        torch.testing.assert_close(
            self.state_base[8].view_as(expected_new_state[1]),
            expected_new_state[1],
        )
        torch.testing.assert_close(
            self.ctx_base[1], decode_history[0, -self.ple.ple_embedding.context_len :]
        )
        torch.testing.assert_close(
            self.ctx_base[10], decode_history[1, -self.ple.ple_embedding.context_len :]
        )
        # The read-side checkpoint is not overwritten on the crossing row.
        torch.testing.assert_close(
            self.state_base[4].view_as(expected_states[1]), expected_states[1]
        )
        torch.testing.assert_close(self.ctx_base[2], expected_contexts[1])

    def test_prefill_skips_pages_the_allocator_did_not_materialize(self):
        """A dropped checkpoint chain still resumes from the terminal page.

        Without prefix reuse the allocator materializes only each request's
        tail page, so intermediate checkpoints have no home and must be
        skipped rather than failing -- the same contract as GDN's
        store_ssm_state_to_block_map kernel.
        """
        lengths = (3 * self._PAGE, 3 * self._PAGE)
        token_count = sum(lengths)
        ids = torch.arange(1, token_count + 1, dtype=torch.long)
        hyper = torch.randn(token_count, _HC * _HIDDEN, dtype=torch.bfloat16)
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        tail_only_state = torch.tensor([[0, 0, 7], [0, 0, 8]], dtype=torch.int32)
        tail_only_ctx = torch.tensor([[0, 0, 9], [0, 0, 10]], dtype=torch.int32)
        inputs[PLE_STATE_TAG].kv_cache_block_id_device = tail_only_state
        inputs[PLE_STATE_TAG].kv_cache_block_id = tail_only_state
        inputs[PLE_NGRAM_CTX_TAG].kv_cache_block_id_device = tail_only_ctx
        inputs[PLE_NGRAM_CTX_TAG].kv_cache_block_id = tail_only_ctx

        expected_outputs = []
        expected_states = []
        expected_contexts = []
        offset = 0
        for length in lengths:
            seq_ids = ids[offset : offset + length].view(1, length)
            history = torch.cat(
                [
                    seq_ids.new_full((1, self.ple.ple_embedding.context_len), 7),
                    seq_ids,
                ],
                dim=1,
            )
            output, state = self.ple.prefill(
                hyper[offset : offset + length].view(1, length, -1), history
            )
            expected_outputs.append(output.squeeze(0))
            expected_states.append(state.squeeze(0))
            expected_contexts.append(history[0, -self.ple.ple_embedding.context_len :])
            offset += length
        expected_states = torch.stack(expected_states)
        expected_contexts = torch.stack(expected_contexts)

        got = self.model._apply_ple(1, hyper, ids, inputs)

        torch.testing.assert_close(got, hyper + torch.cat(expected_outputs, dim=0))
        torch.testing.assert_close(
            self.state_base.index_select(
                0, tail_only_state[:, 2].to(torch.long)
            ).view_as(expected_states),
            expected_states,
        )
        torch.testing.assert_close(
            self.ctx_base.index_select(0, tail_only_ctx[:, 2].to(torch.long)),
            expected_contexts,
        )
        # Nothing leaked into the pages the allocator dropped: only the
        # terminal physical rows may hold data.
        untouched_state = torch.ones(self.state_base.shape[0], dtype=torch.bool)
        untouched_state[tail_only_state[:, 2].to(torch.long)] = False
        self.assertEqual(
            int(torch.count_nonzero(self.state_base[untouched_state]).item()), 0
        )
        untouched_ctx = torch.ones(self.ctx_base.shape[0], dtype=torch.bool)
        untouched_ctx[tail_only_ctx[:, 2].to(torch.long)] = False
        self.assertEqual(
            int(torch.count_nonzero(self.ctx_base[untouched_ctx]).item()), 0
        )

    def test_padded_prefill_skips_only_wholly_inactive_cache_rows(self):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([1, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        for item in inputs.values():
            item.is_s_padded = True
            item.kv_cache_block_id_device = item.kv_cache_block_id.clone()
            item.kv_cache_block_id_device[1].zero_()
            item.kv_cache_block_id = item.kv_cache_block_id_device
        hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        output = self.model._apply_ple(
            1, hyper, torch.ones(2, dtype=torch.long), inputs
        )
        self.assertEqual(output.shape, hyper.shape)
        self.assertEqual(torch.count_nonzero(self.state_base[0]), 0)
        self.assertEqual(torch.count_nonzero(self.ctx_base[0]), 0)
        self.assertGreater(torch.count_nonzero(self.state_base[1]), 0)
        # A live row with a missing terminal page must still fail, even when
        # another row in the same padded batch is wholly inactive.
        for item in inputs.values():
            item.kv_cache_block_id_device[0, 0] = 0
        with self.assertRaisesRegex(RuntimeError, "terminal page is unallocated"):
            self.model._apply_ple(1, hyper, torch.ones(2, dtype=torch.long), inputs)

    def test_cache_free_prefill_warmup_returns_placeholder_before_prefetch(self):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([1, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        for item in inputs.values():
            item.is_s_padded = True
        self.model.kv_cache = None
        self.model.layers = nn.ModuleList()
        hidden = torch.zeros(2, _HIDDEN, dtype=torch.bfloat16)
        self.model.word_embedding = lambda _: hidden
        args = SimpleNamespace(
            input_ids=torch.ones(2, dtype=torch.long), attention_inputs=inputs
        )
        output = self.model._forward_with_ple_prefetch(args)
        self.assertIs(output.hidden_states, hidden)
        for item in inputs.values():
            item.is_s_padded = False
        with self.assertRaisesRegex(RuntimeError, "outside prefill warmup"):
            self.model._forward_with_ple_prefetch(args)

    def test_prefill_rejects_an_unallocated_terminal_page(self):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([self._PAGE + 1, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        inputs[PLE_STATE_TAG].kv_cache_block_id_device = self.state_blocks.clone()
        inputs[PLE_STATE_TAG].kv_cache_block_id_device[0, 1] = 0

        with self.assertRaisesRegex(RuntimeError, "terminal page is unallocated"):
            self.model._apply_ple(
                1,
                torch.randn(self._PAGE + 2, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(self._PAGE + 2, dtype=torch.long),
                inputs,
            )

    def test_prefill_rejects_a_terminal_page_outside_the_block_table(self):
        length = 3 * self._PAGE + 1
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([length, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )

        with self.assertRaisesRegex(RuntimeError, "logical page exceeds"):
            self.model._apply_ple(
                1,
                torch.randn(length + 1, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(length + 1, dtype=torch.long),
                inputs,
            )

    def test_prefill_rejects_a_terminal_physical_block_outside_the_pool(self):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([self._PAGE + 1, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        inputs[PLE_STATE_TAG].kv_cache_block_id_device = self.state_blocks.clone()
        inputs[PLE_STATE_TAG].kv_cache_block_id_device[0, 1] = int(
            self.state_base.shape[0]
        )

        with self.assertRaisesRegex(RuntimeError, "physical block id exceeds"):
            self.model._apply_ple(
                1,
                torch.randn(self._PAGE + 2, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(self._PAGE + 2, dtype=torch.long),
                inputs,
            )

    def test_prefill_rejects_state_context_metadata_mismatch(self):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([self._PAGE + 1, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        inputs[PLE_NGRAM_CTX_TAG].input_lengths = torch.tensor(
            [self._PAGE, 2], dtype=torch.int32
        )

        with self.assertRaisesRegex(RuntimeError, "metadata is inconsistent"):
            self.model._apply_ple(
                1,
                torch.randn(self._PAGE + 2, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(self._PAGE + 2, dtype=torch.long),
                inputs,
            )


if __name__ == "__main__":
    main()
