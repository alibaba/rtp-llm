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
    def test_eager_dp_target_still_rejects_missing_live_pages(self):
        self.state_blocks[1].zero_()
        self.ctx_blocks[1].zero_()
        self.state_blocks[0, 0] = 0  # Other pages keep row zero live.
        inputs = self._target_inputs((1, 0), 3)
        for item in inputs.values():
            item.is_s_padded = True
        before_state, before_ctx = self.state_base.clone(), self.ctx_base.clone()
        with self.assertRaisesRegex(RuntimeError, "unallocated logical page"):
            self.model._apply_ple(
                1,
                torch.randn(6, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(6, dtype=torch.long),
                inputs,
            )
        torch.testing.assert_close(self.state_base, before_state, rtol=0, atol=0)
        torch.testing.assert_close(self.ctx_base, before_ctx, rtol=0, atol=0)

    def test_eager_dp_target_rejects_disagreeing_region_masks(self):
        self.state_blocks[0].zero_()
        inputs = self._target_inputs((1, 1), 3)
        for item in inputs.values():
            item.is_s_padded = True
        with self.assertRaisesRegex(RuntimeError, "disagree about padded rows"):
            self.model._apply_ple(
                1,
                torch.randn(6, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(6, dtype=torch.long),
                inputs,
            )

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

    def test_candidates_resume_after_a_crossed_page_and_a_prefix_hit(self):
        prefix, width = self._PAGE - 2, 4
        self.state_blocks = torch.tensor([[1, 5, 7, 6]], dtype=torch.int32)
        self.ctx_blocks = torch.tensor([[3, 1, 9, 6]], dtype=torch.int32)
        initial_state, initial_ctx = self._seed_target_history((prefix,))
        initial_state_pool = self.state_base.clone()
        initial_ctx_pool = self.ctx_base.clone()
        ids = torch.tensor([[4, 5, 6, 8]])
        hyper = torch.randn(1, width, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected, states, contexts = self._target_baseline(
            hyper, ids, initial_state, initial_ctx
        )
        got = self.model._apply_ple(
            1, hyper.flatten(0, 1), ids.flatten(), self._target_inputs((prefix,), width)
        )
        torch.testing.assert_close(got, (hyper + expected).flatten(0, 1))

        if torch.cuda.is_available():
            from rtp_llm.models_py.kernels.cuda.qwen4_ple.candidate_store import (
                store_candidate_states,
            )

            _, candidate = self.ple.decode_chunk(
                hyper, torch.cat((initial_ctx, ids), dim=1), initial_state
            )
            state_pool = initial_state_pool.cuda().view(
                -1, self.ple.short_conv_state_len, _HC * _HIDDEN
            )
            ctx_pool = initial_ctx_pool.cuda()
            store_candidate_states(
                initial_state.cuda(),
                candidate.cuda(),
                initial_ctx.cuda(),
                ids.cuda(),
                state_pool,
                ctx_pool,
                torch.tensor([prefix], dtype=torch.int32, device="cuda"),
                self._PAGE,
                self.state_blocks.cuda(),
                self.ctx_blocks.cuda(),
            )
            torch.testing.assert_close(state_pool.flatten(1).cpu(), self.state_base)
            torch.testing.assert_close(ctx_pool.cpu(), self.ctx_base)

        # Dispatch selects the page boundary (step 2) and the live tail (step 4).
        self.state_blocks = self.state_blocks[:, [1, 3, 2, 0]]
        self.ctx_blocks = self.ctx_blocks[:, [1, 3, 2, 0]]
        for page, step in enumerate((1, 3)):
            torch.testing.assert_close(
                self.state_base[self.state_blocks[0, page]].view_as(states[step][0]),
                states[step][0],
            )
            torch.testing.assert_close(
                self.ctx_base[self.ctx_blocks[0, page]], contexts[step][0]
            )

        next_id = torch.tensor([12])
        next_hidden = torch.randn(1, _HC * _HIDDEN, dtype=torch.bfloat16)
        for start, step, is_prefill in ((prefix + width, 3, False), (self._PAGE, 1, True)):
            if is_prefill:
                # The new request shares the completed page and owns a different tail.
                self.state_blocks[0, 1] = 10
                self.ctx_blocks[0, 1] = 11
            inputs = self._inputs_by_tag(
                is_prefill=is_prefill,
                input_lengths=torch.tensor([1], dtype=torch.int32),
                prefix_lengths=torch.tensor([start] if is_prefill else [], dtype=torch.int32),
                sequence_lengths=torch.tensor([] if is_prefill else [start], dtype=torch.int32),
            )
            expected, _, _ = self._target_baseline(
                next_hidden.unsqueeze(0), next_id.unsqueeze(0), states[step], contexts[step]
            )
            torch.testing.assert_close(
                self.model._apply_ple(1, next_hidden, next_id, inputs),
                next_hidden + expected.squeeze(0),
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
