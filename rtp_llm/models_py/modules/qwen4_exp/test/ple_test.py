import math
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.qwen4_exp.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)

# Real values read out of Qwen/Qwen3.8-Flash-Next; see design doc appendix C.
_REAL_VOCAB_SIZES = [
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
]
_REAL_OFFSETS = [
    0,
    20000003,
    40000026,
    60000059,
    80000106,
    100000165,
    120000228,
    140000297,
    160000374,
    180000455,
    200000548,
    220000655,
    240000802,
    260000955,
    280001114,
    300001275,
]
_REAL_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
_REAL_SHARDS = 128
_REAL_SHARD_ROWS = 2500012

_NGRAM_SIZE = 3
_HEADS = 4  # 2 orders x 2 heads per order, scaled down from the real 16
_HEAD_DIM = 5
_EOS = 7
_HC = 4
_HIDDEN = 8
_KERNEL = 4
_EPS = 1e-6


def _ref_shift(token_ids, shift, eos):
    if shift == 0:
        return token_ids
    batch_size, seq_len = token_ids.shape
    positions = torch.arange(seq_len, dtype=torch.long)
    eos_positions = torch.where(token_ids == eos, positions, -1)
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]],
        dim=1,
    )
    position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
    source_positions = positions - shift
    gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
    shifted = token_ids.gather(dim=1, index=gather_positions)
    valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, token_ids.new_full((), eos))


def _ref_hashed_ids(
    history, seq_len, vocab_sizes, offsets, multipliers, ngram_size, eos
):
    heads_per_ngram = vocab_sizes.shape[0] // (ngram_size - 1)
    shifted = [_ref_shift(history, s, eos) for s in range(ngram_size)]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        head_vocab = vocab_sizes[start:end].view(1, 1, -1)
        ids = torch.remainder(mixed.unsqueeze(-1), head_vocab)
        blocks.append(ids + offsets[start:end].view(1, 1, -1))
    return torch.cat(blocks, dim=-1)[:, -seq_len:]


class NGramVocabLayoutTest(unittest.TestCase):
    """The 128 shards are an equal row-split of the padded hashed vocab."""

    def test_offsets_are_the_running_sum_of_vocab_sizes(self):
        running = 0
        for size, offset in zip(_REAL_VOCAB_SIZES, _REAL_OFFSETS):
            self.assertEqual(offset, running)
            running += size
        self.assertEqual(running, 320_001_446)

    def test_padding_to_128_yields_exactly_the_shard_row_count(self):
        total = _REAL_OFFSETS[-1] + _REAL_VOCAB_SIZES[-1]
        padded = math.ceil(total / 128) * 128

        self.assertEqual(padded, 320_001_536)
        self.assertEqual(padded, _REAL_SHARDS * _REAL_SHARD_ROWS)

    def test_head_dim_tiles_the_ple_embed_dim(self):
        # 16 heads x 160 == ple_embed_dim 2560
        self.assertEqual(len(_REAL_VOCAB_SIZES) * 160, 2560)


class Qwen4ExpNGramEmbeddingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab_sizes = torch.tensor([11, 13, 17, 19], dtype=torch.long)
        self.offsets = torch.tensor([0, 11, 24, 41], dtype=torch.long)
        self.multipliers = torch.tensor(_REAL_MULTIPLIERS, dtype=torch.long)
        total = int(self.offsets[-1] + self.vocab_sizes[-1])  # 60
        self.shard_rows = 20
        self.shards = [torch.randn(self.shard_rows, _HEAD_DIM) for _ in range(3)]
        self.total = total

    def _build(self):
        return Qwen4ExpNGramEmbedding(
            self.shards,
            self.vocab_sizes,
            self.offsets,
            self.multipliers,
            ngram_size=_NGRAM_SIZE,
            eos_token_id=_EOS,
        )

    def test_hashed_ids_match_reference(self):
        history = torch.tensor([[7, 3, 5, 9, 7, 2], [1, 1, 2, 3, 5, 8]])

        got = self._build().hashed_ids(history, seq_len=4)
        ref = _ref_hashed_ids(
            history,
            4,
            self.vocab_sizes,
            self.offsets,
            self.multipliers,
            _NGRAM_SIZE,
            _EOS,
        )

        self.assertEqual(got.shape, (2, 4, _HEADS))
        torch.testing.assert_close(got, ref)

    def test_hashed_ids_stay_inside_their_head_slice(self):
        history = torch.randint(0, 50, (3, 12))

        ids = self._build().hashed_ids(history, seq_len=10)

        for head in range(_HEADS):
            low = int(self.offsets[head])
            high = low + int(self.vocab_sizes[head])
            self.assertTrue(bool((ids[..., head] >= low).all()))
            self.assertTrue(bool((ids[..., head] < high).all()))

    def test_eos_restarts_the_context(self):
        # Token after an EOS must hash as if it began a fresh segment.
        fresh = torch.tensor([[_EOS, _EOS, 4, 5]])
        after_eos = torch.tensor([[3, _EOS, 4, 5]])

        module = self._build()
        a = module.hashed_ids(fresh, seq_len=2)
        b = module.hashed_ids(after_eos, seq_len=2)

        torch.testing.assert_close(a, b)

    def test_sharded_gather_matches_a_single_concatenated_table(self):
        module = self._build()
        ids = module.hashed_ids(torch.randint(0, 50, (2, 9)), seq_len=7)
        full_table = torch.cat(self.shards, dim=0)

        got = module.gather(ids)
        ref = full_table[ids].flatten(-2)

        self.assertEqual(got.shape, (2, 7, _HEADS * _HEAD_DIM))
        torch.testing.assert_close(got, ref)

    def test_rejects_shards_too_small_for_the_hashed_vocab(self):
        with self.assertRaisesRegex(ValueError, "shards hold .* rows"):
            Qwen4ExpNGramEmbedding(
                [torch.randn(4, _HEAD_DIM)],
                self.vocab_sizes,
                self.offsets,
                self.multipliers,
                ngram_size=_NGRAM_SIZE,
                eos_token_id=_EOS,
            )

    def test_rejects_ragged_shards(self):
        with self.assertRaisesRegex(ValueError, "equal row counts"):
            Qwen4ExpNGramEmbedding(
                [torch.randn(20, _HEAD_DIM), torch.randn(19, _HEAD_DIM)],
                self.vocab_sizes,
                self.offsets,
                self.multipliers,
                ngram_size=_NGRAM_SIZE,
                eos_token_id=_EOS,
            )

    def test_rejects_non_integer_hash_metadata(self):
        with self.assertRaisesRegex(TypeError, "vocab_sizes must stay int64"):
            Qwen4ExpNGramEmbedding(
                self.shards,
                self.vocab_sizes.to(torch.bfloat16),
                self.offsets,
                self.multipliers,
                ngram_size=_NGRAM_SIZE,
                eos_token_id=_EOS,
            )


class Qwen4ExpPLELayerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hc_hidden = _HC * _HIDDEN
        self.embed_dim = _HEADS * _HEAD_DIM
        vocab_sizes = torch.tensor([11, 13, 17, 19], dtype=torch.long)
        offsets = torch.tensor([0, 11, 24, 41], dtype=torch.long)
        self.ngram = Qwen4ExpNGramEmbedding(
            [torch.randn(20, _HEAD_DIM) for _ in range(3)],
            vocab_sizes,
            offsets,
            torch.tensor(_REAL_MULTIPLIERS, dtype=torch.long),
            ngram_size=_NGRAM_SIZE,
            eos_token_id=_EOS,
        )
        self.key_proj = torch.randn(self.hc_hidden, self.embed_dim) * 0.05
        self.value_proj = torch.randn(_HIDDEN, self.embed_dim) * 0.05
        self.conv_weight = torch.randn(self.hc_hidden, 1, _KERNEL) * 0.05
        self.norm_key = torch.randn(self.hc_hidden) * 0.1
        self.norm_query = torch.randn(self.hc_hidden) * 0.1
        self.norm_conv = torch.randn(self.hc_hidden) * 0.1

    def _build(self):
        return Qwen4ExpPLELayer(
            self.ngram,
            self.key_proj,
            self.value_proj,
            self.conv_weight,
            self.norm_key,
            self.norm_query,
            self.norm_conv,
            hc_mult=_HC,
            hidden_size=_HIDDEN,
            conv_kernel_size=_KERNEL,
            norm_eps=_EPS,
        )

    def _reference(self, hyper_states, history):
        """Independent transcription of Qwen4ExpTextPLELayer.forward."""
        seq_len = hyper_states.shape[1]
        emb = self.ngram(history, seq_len)

        def norm(x, gamma):
            y = x.float().unflatten(-1, (_HC, _HIDDEN))
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + _EPS)
            return (y.flatten(-2) * (1.0 + gamma.float())).type_as(x)

        key_normed = norm(F.linear(emb, self.key_proj), self.norm_key).unflatten(
            -1, (_HC, _HIDDEN)
        )
        value = F.linear(emb, self.value_proj)
        query_normed = norm(hyper_states, self.norm_query).unflatten(-1, (_HC, _HIDDEN))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(
            _HIDDEN
        )
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = norm(gated_value.flatten(-2), self.norm_conv)
        gated_value = gated_value.flatten(-2)

        dilation = _NGRAM_SIZE
        state_len = (_KERNEL - 1) * dilation
        conv_in = gated_value_normed.transpose(1, 2)
        conv_in = F.pad(conv_in, (state_len, 0))
        conv_in = conv_in[..., -(state_len + seq_len) :]
        conv_out = F.silu(
            F.conv1d(
                conv_in, self.conv_weight, groups=self.hc_hidden, dilation=dilation
            )
        ).transpose(1, 2)
        return gated_value + conv_out

    def test_matches_upstream_reference(self):
        hyper_states = torch.randn(2, 6, self.hc_hidden)
        history = torch.tensor([[7, 3, 5, 9, 7, 2, 4, 6], [1, 1, 2, 3, 5, 8, 9, 0]])

        got = self._build()(hyper_states, history)
        ref = self._reference(hyper_states, history)

        self.assertEqual(got.shape, (2, 6, self.hc_hidden))
        torch.testing.assert_close(got, ref)

    def test_output_width_matches_the_residual_stream(self):
        hyper_states = torch.randn(1, 5, self.hc_hidden)
        history = torch.randint(0, 50, (1, 7))

        out = self._build()(hyper_states, history)

        self.assertEqual(out.shape, hyper_states.shape)

    def test_conv_is_causal_under_dilation(self):
        # Changing the last token must not change earlier outputs.
        layer = self._build()
        hyper_states = torch.randn(1, 8, self.hc_hidden)
        history_a = torch.tensor([[1, 2, 3, 4, 5, 6, 8, 9, 10, 11]])
        history_b = history_a.clone()
        history_b[0, -1] = 12

        out_a = layer(hyper_states, history_a)
        out_b = layer(hyper_states, history_b)

        torch.testing.assert_close(out_a[:, :-1], out_b[:, :-1])
        self.assertFalse(torch.allclose(out_a[:, -1], out_b[:, -1]))

    def test_padding_mask_zeroes_masked_positions(self):
        layer = self._build()
        hyper_states = torch.randn(1, 6, self.hc_hidden)
        history = torch.randint(0, 50, (1, 8))
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

        out = layer(hyper_states, history, padding_mask=mask)
        unmasked = layer(hyper_states, history)

        # Masked positions lose their direct (non-conv) contribution.
        self.assertFalse(torch.allclose(out[:, 4:], unmasked[:, 4:]))

    def test_rejects_mismatched_key_projection(self):
        with self.assertRaisesRegex(ValueError, "key_proj out width"):
            Qwen4ExpPLELayer(
                self.ngram,
                torch.randn(self.hc_hidden + 1, self.embed_dim),
                self.value_proj,
                self.conv_weight,
                self.norm_key,
                self.norm_query,
                self.norm_conv,
                hc_mult=_HC,
                hidden_size=_HIDDEN,
                conv_kernel_size=_KERNEL,
                norm_eps=_EPS,
            )

    def test_rejects_mismatched_conv_kernel(self):
        with self.assertRaisesRegex(ValueError, "conv weight"):
            Qwen4ExpPLELayer(
                self.ngram,
                self.key_proj,
                self.value_proj,
                torch.randn(self.hc_hidden, 1, _KERNEL + 1),
                self.norm_key,
                self.norm_query,
                self.norm_conv,
                hc_mult=_HC,
                hidden_size=_HIDDEN,
                conv_kernel_size=_KERNEL,
                norm_eps=_EPS,
            )


if __name__ == "__main__":
    unittest.main()
