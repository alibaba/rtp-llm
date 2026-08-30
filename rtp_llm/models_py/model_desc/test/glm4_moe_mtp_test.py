"""CPU contract tests for Glm4MoeMtpModel's eh_proj fusion and forward wiring.

Instantiating the model for real needs the full weight set, a MoE decoder layer and
pybind KV-cache blocks on a device, which no model_desc test in this repo does. The
parts that are cheap to get wrong and expensive to notice are the pure-tensor ones:
the token-sliced eh_proj must equal the unsliced result, the concat must stay in GLM's
[embed; hidden] order, and the decode loop must hand the norm a zero-initialised
residual and address the KV cache by layer index. Those are driven directly here.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.model_desc.glm4_moe_mtp import (
    _EH_PROJ_CHUNK_ENV,
    _EH_PROJ_CHUNK_TOKENS_DEFAULT,
    Glm4MoeMtpModel,
    _eh_proj_chunk_tokens,
)

HIDDEN = 8
VOCAB = 16


def _model(*, chunk, hidden_size=HIDDEN, eh_proj=None):
    """A Glm4MoeMtpModel with only the tensors _fuse_embed_and_hidden touches."""
    m = object.__new__(Glm4MoeMtpModel)
    m._hidden_size = hidden_size
    m._eh_proj_chunk = chunk
    torch.manual_seed(0)
    table = torch.randn(VOCAB, HIDDEN)
    m.embed_tokens = lambda ids: table[ids]
    # Scale rather than normalise: a real RMSNorm is row-independent too, and a
    # deterministic elementwise op keeps the slicing assertion exact.
    m.enorm = lambda x: x * 2.0
    m.hnorm = lambda x: x * 3.0
    m.eh_proj = eh_proj if eh_proj is not None else _SumHalves()
    return m


class _SumHalves:
    """Stand-in for eh_proj: [2*hidden] -> [hidden], keeps both halves visible."""

    def __call__(self, concat):
        embed_part, hidden_part = concat.chunk(2, dim=-1)
        return embed_part + 10.0 * hidden_part


class _FirstHalf:
    """Identity on the embedding half only, so concat order is observable."""

    def __call__(self, concat):
        return concat.chunk(2, dim=-1)[0]


class EhProjChunkTokensTest(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop(_EH_PROJ_CHUNK_ENV, None)

    def tearDown(self):
        os.environ.pop(_EH_PROJ_CHUNK_ENV, None)
        if self._saved is not None:
            os.environ[_EH_PROJ_CHUNK_ENV] = self._saved

    def test_unset_uses_the_default(self):
        self.assertEqual(_eh_proj_chunk_tokens(), _EH_PROJ_CHUNK_TOKENS_DEFAULT)

    def test_reads_an_override(self):
        os.environ[_EH_PROJ_CHUNK_ENV] = "128"
        self.assertEqual(_eh_proj_chunk_tokens(), 128)

    def test_rejects_a_non_integer(self):
        os.environ[_EH_PROJ_CHUNK_ENV] = "many"
        with self.assertRaises(ValueError):
            _eh_proj_chunk_tokens()

    def test_rejects_a_non_positive_value(self):
        os.environ[_EH_PROJ_CHUNK_ENV] = "0"
        with self.assertRaises(ValueError):
            _eh_proj_chunk_tokens()


class FuseEmbedAndHiddenTest(unittest.TestCase):
    """The slicing is an allocation optimisation and must not change the result."""

    def _inputs(self, num_tokens):
        torch.manual_seed(1)
        ids = torch.randint(0, VOCAB, (num_tokens,))
        last_hidden = torch.randn(num_tokens, HIDDEN)
        return ids, last_hidden

    def test_chunked_matches_unchunked_exactly(self):
        ids, last_hidden = self._inputs(37)
        unchunked = _model(chunk=1024)._fuse_embed_and_hidden(ids, last_hidden)
        chunked = _model(chunk=8)._fuse_embed_and_hidden(ids, last_hidden)
        # Exact, not approximate: RMSNorm is per-row and a linear is row-independent,
        # so a slice produces the same values it would inside the whole batch.
        torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)

    def test_chunk_boundary_that_divides_evenly(self):
        ids, last_hidden = self._inputs(32)
        unchunked = _model(chunk=1024)._fuse_embed_and_hidden(ids, last_hidden)
        chunked = _model(chunk=8)._fuse_embed_and_hidden(ids, last_hidden)
        torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)

    def test_single_token_takes_the_unsliced_path(self):
        # The decode step. Shape must survive a chunk size far above the token count.
        ids, last_hidden = self._inputs(1)
        out = _model(chunk=8192)._fuse_embed_and_hidden(ids, last_hidden)
        self.assertEqual(out.shape, (1, HIDDEN))

    def test_concat_is_embedding_first(self):
        # GLM's eh_proj is trained on [embed; hidden]; DeepSeek's is the reverse.
        # Flipping this lowers the accept rate instead of failing, so pin it.
        ids, last_hidden = self._inputs(5)
        m = _model(chunk=1024, eh_proj=_FirstHalf())
        expected = m.enorm(m.embed_tokens(ids))
        torch.testing.assert_close(
            m._fuse_embed_and_hidden(ids, last_hidden), expected, rtol=0, atol=0
        )

    def test_chunked_path_preserves_row_order(self):
        # A copy_ into the wrong slice would still return the right shape.
        ids, last_hidden = self._inputs(20)
        m_chunked = _model(chunk=7)
        m_whole = _model(chunk=1024)
        chunked = m_chunked._fuse_embed_and_hidden(ids, last_hidden)
        for row in range(20):
            torch.testing.assert_close(
                chunked[row],
                m_whole._fuse_embed_and_hidden(
                    ids[row : row + 1], last_hidden[row : row + 1]
                )[0],
                rtol=0,
                atol=0,
            )


class Glm4MoeMtpForwardWiringTest(unittest.TestCase):
    """forward's residual seeding, KV-cache indexing and eh_proj width check."""

    def _forward_model(self, *, hidden_size=HIDDEN, layer_num=1):
        m = _model(chunk=1024, hidden_size=hidden_size)
        m.layer_num = layer_num
        self.layer_calls = []

        def decoder_layer(hidden_states, residual, fmha, kv_cache=None):
            self.layer_calls.append(
                SimpleNamespace(
                    hidden_states=hidden_states.clone(),
                    residual=residual.clone(),
                    kv_cache=kv_cache,
                )
            )
            return SimpleNamespace(
                hidden_states=hidden_states + 1.0, residual=residual + 2.0
            )

        m.layers = [decoder_layer] * layer_num
        m.kv_cache = Mock()
        m.kv_cache.get_layer_cache.side_effect = lambda i: f"cache{i}"
        m.norm = Mock(side_effect=lambda hidden, residual: (hidden + residual, None))
        return m

    def _inputs(self, num_tokens=4):
        torch.manual_seed(2)
        return SimpleNamespace(
            input_ids=torch.randint(0, VOCAB, (num_tokens,)),
            input_hiddens=torch.randn(num_tokens, HIDDEN),
        )

    @patch("rtp_llm.models_py.model_desc.glm4_moe_mtp.select_fmha_impl_for_layer")
    def test_residual_starts_at_zero(self, select_impl):
        m = self._forward_model()
        m.forward(self._inputs(), fmha_impl="fmha")
        torch.testing.assert_close(
            self.layer_calls[0].residual,
            torch.zeros_like(self.layer_calls[0].residual),
            rtol=0,
            atol=0,
        )

    @patch("rtp_llm.models_py.model_desc.glm4_moe_mtp.select_fmha_impl_for_layer")
    def test_kv_cache_is_addressed_by_layer_index(self, select_impl):
        m = self._forward_model()
        m.forward(self._inputs(), fmha_impl="fmha")
        m.kv_cache.get_layer_cache.assert_called_once_with(0)
        self.assertEqual(self.layer_calls[0].kv_cache, "cache0")

    @patch("rtp_llm.models_py.model_desc.glm4_moe_mtp.select_fmha_impl_for_layer")
    def test_absent_kv_cache_passes_none(self, select_impl):
        m = self._forward_model()
        m.kv_cache = None
        m.forward(self._inputs(), fmha_impl="fmha")
        self.assertIsNone(self.layer_calls[0].kv_cache)

    @patch("rtp_llm.models_py.model_desc.glm4_moe_mtp.select_fmha_impl_for_layer")
    def test_final_norm_consumes_hidden_and_residual(self, select_impl):
        m = self._forward_model()
        out = m.forward(self._inputs(), fmha_impl="fmha")
        hidden_arg, residual_arg = m.norm.call_args.args
        # The layer stub adds 1 to hidden and 2 to a zero residual.
        torch.testing.assert_close(
            hidden_arg, self.layer_calls[0].hidden_states + 1.0, rtol=0, atol=0
        )
        torch.testing.assert_close(
            residual_arg, torch.full_like(residual_arg, 2.0), rtol=0, atol=0
        )
        torch.testing.assert_close(
            out.hidden_states, hidden_arg + residual_arg, rtol=0, atol=0
        )

    @patch("rtp_llm.models_py.model_desc.glm4_moe_mtp.select_fmha_impl_for_layer")
    def test_rejects_an_eh_proj_that_does_not_restore_hidden_size(self, select_impl):
        # A missing transpose on eh_proj must fail here, not produce wrong drafts.
        m = self._forward_model(hidden_size=HIDDEN + 1)
        with self.assertRaises(RuntimeError) as caught:
            m.forward(self._inputs(), fmha_impl="fmha")
        self.assertIn("eh_proj", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
