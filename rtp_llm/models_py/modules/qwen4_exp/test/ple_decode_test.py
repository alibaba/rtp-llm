import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)

_REAL_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
_NGRAM = 3
_HEAD_DIM = 5
_EOS = 7
_HC = 4
_HIDDEN = 8
_KERNEL = 4
_EPS = 1e-6
_CONTEXT = _NGRAM - 1
_STATE_LEN = (_KERNEL - 1) * _NGRAM  # 9


def _make_layer():
    torch.manual_seed(0)
    hc_hidden = _HC * _HIDDEN
    embed_dim = _HEAD_DIM * 4
    ngram = Qwen4ExpNGramEmbedding(
        [torch.randn(20, _HEAD_DIM) for _ in range(3)],
        torch.tensor([11, 13, 17, 19], dtype=torch.long),
        torch.tensor([0, 11, 24, 41], dtype=torch.long),
        torch.tensor(_REAL_MULTIPLIERS, dtype=torch.long),
        ngram_size=_NGRAM,
        eos_token_id=_EOS,
    )
    return Qwen4ExpPLELayer(
        ngram,
        torch.randn(hc_hidden, embed_dim) * 0.05,
        torch.randn(_HIDDEN, embed_dim) * 0.05,
        torch.randn(hc_hidden, 1, _KERNEL) * 0.05,
        torch.randn(hc_hidden) * 0.1,
        torch.randn(hc_hidden) * 0.1,
        torch.randn(hc_hidden) * 0.1,
        hc_mult=_HC,
        hidden_size=_HIDDEN,
        conv_kernel_size=_KERNEL,
        norm_eps=_EPS,
    )


class PleDecodeEquivalenceTest(unittest.TestCase):
    def test_decode_chunk_matches_ordered_decode_steps(self):
        layer = _make_layer()
        hc_hidden = _HC * _HIDDEN
        batch, query_len = 3, 5
        hyper = torch.randn(batch, query_len, hc_hidden)
        initial_state = torch.randn(batch, _STATE_LEN, hc_hidden)
        initial_context = torch.randint(0, 40, (batch, _CONTEXT))
        candidate_ids = torch.randint(0, 40, (batch, query_len))
        history = torch.cat([initial_context, candidate_ids], dim=1)

        chunk_output, candidate_inputs = layer.decode_chunk(
            hyper, history, initial_state
        )

        step_outputs = []
        step_state = initial_state
        step_context = initial_context
        states_after_each_step = []
        for step in range(query_len):
            step_ids = candidate_ids[:, step : step + 1]
            output, step_state = layer.decode_step(
                hyper[:, step : step + 1],
                torch.cat([step_context, step_ids], dim=1),
                step_state,
            )
            step_outputs.append(output)
            states_after_each_step.append(step_state)
            step_context = torch.cat([step_context, step_ids], dim=1)[:, -_CONTEXT:]

        torch.testing.assert_close(chunk_output, torch.cat(step_outputs, dim=1))
        timeline = torch.cat([initial_state, candidate_inputs], dim=1)
        for accepted, expected_state in enumerate(states_after_each_step, start=1):
            selected = timeline[:, accepted : accepted + _STATE_LEN]
            torch.testing.assert_close(selected, expected_state)

    def test_chained_decode_matches_full_sequence_forward(self):
        layer = _make_layer()
        hc_hidden = _HC * _HIDDEN
        batch, prompt_len, gen_len = 2, 5, 6
        total = prompt_len + gen_len

        # A fresh sequence: the ngram history begins with EOS context.
        tokens = torch.randint(0, 40, (batch, total))
        hyper = torch.randn(batch, total, hc_hidden)
        ctx0 = tokens.new_full((batch, _CONTEXT), _EOS)
        history = torch.cat([ctx0, tokens], dim=1)

        reference = layer(hyper, history)

        # Prefill the prompt, then decode the remaining tokens one at a time.
        prompt_hist = torch.cat([ctx0, tokens[:, :prompt_len]], dim=1)
        _, prompt_gvn = layer._gated_values(hyper[:, :prompt_len], prompt_hist)
        conv_buffer = layer.prefill_conv_state(prompt_gvn)
        prefill_out = layer(hyper[:, :prompt_len], prompt_hist)
        torch.testing.assert_close(prefill_out, reference[:, :prompt_len])

        # ngram context after the prompt: last _CONTEXT tokens seen so far.
        ctx = prompt_hist[:, -_CONTEXT:]
        for step in range(prompt_len, total):
            current = tokens[:, step : step + 1]
            step_hist = torch.cat([ctx, current], dim=1)
            out, conv_buffer = layer.decode_step(
                hyper[:, step : step + 1], step_hist, conv_buffer
            )
            torch.testing.assert_close(out[:, 0], reference[:, step])
            ctx = step_hist[:, -_CONTEXT:]

    def test_decode_from_empty_prompt_matches_forward(self):
        # Pure autoregressive: zero-length prompt, decode every token.
        layer = _make_layer()
        hc_hidden = _HC * _HIDDEN
        batch, total = 1, 7
        tokens = torch.randint(0, 40, (batch, total))
        hyper = torch.randn(batch, total, hc_hidden)
        ctx0 = tokens.new_full((batch, _CONTEXT), _EOS)
        history = torch.cat([ctx0, tokens], dim=1)

        reference = layer(hyper, history)

        conv_buffer = hyper.new_zeros(batch, _STATE_LEN, hc_hidden)
        ctx = ctx0
        for step in range(total):
            current = tokens[:, step : step + 1]
            step_hist = torch.cat([ctx, current], dim=1)
            out, conv_buffer = layer.decode_step(
                hyper[:, step : step + 1], step_hist, conv_buffer
            )
            torch.testing.assert_close(out[:, 0], reference[:, step])
            ctx = step_hist[:, -_CONTEXT:]

    def test_prefill_conv_state_left_pads_a_short_prompt(self):
        layer = _make_layer()
        hc_hidden = _HC * _HIDDEN
        gvn = torch.randn(1, 3, hc_hidden)  # shorter than state_len (9)

        state = layer.prefill_conv_state(gvn)

        self.assertEqual(state.shape, (1, _STATE_LEN, hc_hidden))
        torch.testing.assert_close(state[:, -3:], gvn)
        self.assertTrue(torch.count_nonzero(state[:, : _STATE_LEN - 3]) == 0)

    def test_conv_buffer_rolls_by_one_each_step(self):
        layer = _make_layer()
        hc_hidden = _HC * _HIDDEN
        conv_buffer = torch.randn(1, _STATE_LEN, hc_hidden)
        ctx = torch.tensor([[3, 5]])
        current = torch.tensor([[9]])

        _, new_buffer = layer.decode_step(
            torch.randn(1, 1, hc_hidden), torch.cat([ctx, current], dim=1), conv_buffer
        )

        self.assertEqual(new_buffer.shape, conv_buffer.shape)
        # The oldest row is dropped; the rest shift left by one.
        torch.testing.assert_close(new_buffer[:, :-1], conv_buffer[:, 1:])


if __name__ == "__main__":
    unittest.main()
