import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rtp_llm.models_py.speculative.dspark_proposer_mixin import (
    DSparkMarkovHead,
    DSparkProposerMixin,
)


class _TinyProposer(DSparkProposerMixin):
    """Smallest possible DSparkProposerMixin subclass for the sampling tail."""

    def __init__(self, *, width: int, vocab: int, rank: int):
        self.init_dspark_proposer(
            width=width,
            noise_token_id=1,
            aux_feature_dim=8,
            hidden_dim=4,
            vocab_size=vocab,
        )
        torch.manual_seed(7)
        self.w1 = torch.randn(vocab, rank)
        self.w2 = torch.randn(vocab, rank)
        self.markov_head = DSparkMarkovHead(
            self.w1, self.w2, vocab_size=vocab, rank=rank
        )


def _reference_chain(base_logits, anchors, w1, w2):
    """Direct transcription of the reference proposer loop.

    The reference applies softmax before the argmax; softmax is monotone so
    the argmax (the only consumed output — the deterministic proposal's q is
    the engine-built point mass) is numerically identical without it.
    """
    previous = anchors
    tokens = []
    for step in range(base_logits.shape[1]):
        markov_embed = F.embedding(previous, w1)
        markov_bias = F.linear(markov_embed, w2).float()
        logits = base_logits[:, step] + markov_bias
        next_token = torch.argmax(
            torch.softmax(logits, dim=-1, dtype=torch.float32), dim=-1
        )
        tokens.append(next_token.to(torch.int32))
        previous = next_token
    return torch.stack(tokens, dim=1).contiguous()


class SampleSequentialMarkovTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab, self.rank, self.batch, self.width = 23, 4, 3, 5
        self.proposer = _TinyProposer(
            width=self.width, vocab=self.vocab, rank=self.rank
        )
        self.base_logits = torch.randn(self.batch, self.width, self.vocab)
        self.anchors = torch.randint(
            0, self.vocab, (self.batch,), dtype=torch.int32
        )

    def test_matches_reference_chain_exactly(self) -> None:
        expected_tokens = _reference_chain(
            self.base_logits, self.anchors, self.proposer.w1, self.proposer.w2
        )

        tokens = self.proposer._sample_sequential_markov(
            self.base_logits, self.anchors
        )

        self.assertTrue(torch.equal(tokens, expected_tokens))

    def test_tokens_feed_next_step_bias(self) -> None:
        # Step 0 must consume the anchor bias; step 1 must consume the step-0
        # winner, not the anchor.
        head = self.proposer.markov_head
        tokens = self.proposer._sample_sequential_markov(
            self.base_logits, self.anchors
        )
        step0 = self.base_logits[:, 0] + head.bias(self.anchors)
        self.assertTrue(
            torch.equal(
                tokens[:, 0],
                torch.argmax(
                    torch.softmax(step0, dim=-1, dtype=torch.float32), dim=-1
                ).to(torch.int32),
            )
        )
        step1 = self.base_logits[:, 1] + head.bias(tokens[:, 0].long())
        self.assertTrue(
            torch.equal(
                tokens[:, 1],
                torch.argmax(
                    torch.softmax(step1, dim=-1, dtype=torch.float32), dim=-1
                ).to(torch.int32),
            )
        )

    def test_output_shape(self) -> None:
        # The chain is deterministic argmax; its q distribution is the point
        # mass built engine-side, so tokens are the sole output here.
        tokens = self.proposer._sample_sequential_markov(
            self.base_logits, self.anchors
        )
        self.assertEqual(tuple(tokens.shape), (self.batch, self.width))

    def test_rejects_bad_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, r"base logits must be \[B,width,V\]"):
            self.proposer._sample_sequential_markov(
                self.base_logits[:, 0], self.anchors
            )
        with self.assertRaisesRegex(
            ValueError, "does not match the configured width"
        ):
            self.proposer._sample_sequential_markov(
                self.base_logits[:, :3], self.anchors
            )
        with self.assertRaisesRegex(ValueError, "anchors must be"):
            self.proposer._sample_sequential_markov(
                self.base_logits, self.anchors[:1]
            )

    def test_rejects_missing_markov_head(self) -> None:
        self.proposer.markov_head = None
        with self.assertRaisesRegex(RuntimeError, "markov head is not loaded"):
            self.proposer._sample_sequential_markov(self.base_logits, self.anchors)


class ProposerContractTest(unittest.TestCase):
    def test_init_rejects_bad_geometry(self) -> None:
        proposer = DSparkProposerMixin()
        with self.assertRaisesRegex(ValueError, "width must be positive"):
            proposer.init_dspark_proposer(
                width=0,
                noise_token_id=1,
                aux_feature_dim=8,
                hidden_dim=4,
                vocab_size=9,
            )
        with self.assertRaisesRegex(
            ValueError, "noise token id must be non-negative"
        ):
            proposer.init_dspark_proposer(
                width=5,
                noise_token_id=-1,
                aux_feature_dim=8,
                hidden_dim=4,
                vocab_size=9,
            )
        with self.assertRaisesRegex(ValueError, "aux_feature_dim must be positive"):
            proposer.init_dspark_proposer(
                width=5,
                noise_token_id=1,
                aux_feature_dim=0,
                hidden_dim=4,
                vocab_size=9,
            )

    def test_empty_outputs_follow_configured_geometry(self) -> None:
        proposer = DSparkProposerMixin()
        proposer.init_dspark_proposer(
            width=3, noise_token_id=1, aux_feature_dim=24, hidden_dim=8, vocab_size=17
        )
        outputs = proposer.dspark_empty_outputs(2, torch.device("cpu"))
        self.assertEqual(tuple(outputs.hidden_states.shape), (6, 8))
        self.assertEqual(tuple(outputs.draft_tokens.shape), (2, 3))
        self.assertEqual(outputs.draft_tokens.dtype, torch.int32)

    def test_hooks_require_subclass_implementation(self) -> None:
        proposer = DSparkProposerMixin()
        with self.assertRaises(NotImplementedError):
            proposer.combine_hidden_states(torch.zeros(1, 8))
        with self.assertRaises(NotImplementedError):
            proposer.compute_draft_logits(torch.zeros(1, 3, 4, 8))
        with self.assertRaises(NotImplementedError):
            proposer.commit_feature_rows(
                torch.zeros(1, 8),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                None,
            )
        # Full-vocabulary default: identity mapping.
        ids = torch.tensor([3, 5])
        self.assertTrue(torch.equal(proposer.map_draft_to_target(ids), ids))

    def test_markov_head_rejects_mismatched_weights(self) -> None:
        w = torch.zeros(11, 4)
        with self.assertRaisesRegex(
            ValueError, "markov_w2 shape must match markov_w1"
        ):
            DSparkMarkovHead(w, torch.zeros(11, 5), vocab_size=11, rank=4)
        with self.assertRaisesRegex(ValueError, "unexpected DSpark markov_w1 shape"):
            DSparkMarkovHead(w, w, vocab_size=12, rank=4)


class _CommitProposer(DSparkProposerMixin):
    """Captures the rows handed to the projection and commit hooks."""

    def __init__(self, *, aux_dim: int):
        self.init_dspark_proposer(
            width=2,
            noise_token_id=1,
            aux_feature_dim=aux_dim,
            hidden_dim=aux_dim,
            vocab_size=7,
        )
        self.seen_features = None
        self.committed = None

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        self.seen_features = features
        return features

    def commit_feature_rows(self, main_x, req, positions, committed_ends, inputs):
        self.committed = (main_x, req, positions, committed_ends)


class CommitStepTest(unittest.TestCase):
    def test_commit_derives_windows_from_standard_prefill_geometry(self) -> None:
        # input_hiddens is a view of the shared MTP hidden buffer whose
        # DSpARK row width equals the aux payload — the commit step must not
        # copy it, and its row windows come straight from
        # input_lengths/prefix_lengths.
        aux_dim, rows = 6, 4
        proposer = _CommitProposer(aux_dim=aux_dim)
        hidden = torch.arange(
            rows * aux_dim, dtype=torch.float32
        ).reshape(rows, aux_dim)
        inputs = SimpleNamespace(
            input_hiddens=hidden,
            attention_inputs=SimpleNamespace(
                input_lengths=torch.tensor([3, 1], dtype=torch.int32),
                prefix_lengths=torch.tensor([10, 20], dtype=torch.int32),
            ),
        )

        proposer.run_commit_step(inputs, torch.device("cpu"))

        torch.testing.assert_close(proposer.seen_features, hidden)
        self.assertEqual(
            proposer.seen_features.data_ptr(), hidden.data_ptr()
        )
        main_x, req, positions, committed_ends = proposer.committed
        self.assertEqual(main_x.shape, (rows, aux_dim))
        self.assertEqual(req.tolist(), [0, 0, 0, 1])
        self.assertEqual(positions.tolist(), [10, 11, 12, 20])
        self.assertEqual(committed_ends.tolist(), [13, 21])


class _ProposeProposer(DSparkProposerMixin):
    """Fake propose half: records the query-block call, returns flat zeros."""

    def __init__(self, *, width: int = 3, vocab: int = 11, rank: int = 2):
        self.init_dspark_proposer(
            width=width,
            noise_token_id=9,
            aux_feature_dim=8,
            hidden_dim=4,
            vocab_size=vocab,
        )
        torch.manual_seed(3)
        self.markov_head = DSparkMarkovHead(
            torch.randn(vocab, rank),
            torch.randn(vocab, rank),
            vocab_size=vocab,
            rank=rank,
        )
        self.query_block_calls = []

    def forward_query_block(
        self, query_ids, query_positions, prefix_lengths, active_requests, inputs, fmha_impl
    ):
        self.query_block_calls.append(
            (query_ids, query_positions, prefix_lengths, active_requests)
        )
        rows = int(query_ids.shape[0]) * self._dspark_width
        return torch.zeros(rows, self._dspark_hidden_dim)

    def compute_draft_logits(self, hidden):
        batch = int(hidden.shape[0]) // self._dspark_width
        base = torch.zeros(
            batch, self._dspark_width, self._dspark_vocab_size, dtype=torch.float32
        )
        return hidden, base


def _propose_inputs(anchors, prefix_lengths, width):
    batch = int(anchors.numel())
    ids = torch.full((batch, width), 7, dtype=torch.int32)
    if batch:
        ids[:, 0] = anchors
    return SimpleNamespace(
        input_ids=ids.reshape(-1),
        attention_inputs=SimpleNamespace(
            input_lengths=torch.ones(batch, dtype=torch.int32),
            prefix_lengths=prefix_lengths,
        ),
    )


class ProposeStepTest(unittest.TestCase):
    def test_query_block_geometry_and_markov_tail(self) -> None:
        proposer = _ProposeProposer()
        anchors = torch.tensor([4, 8], dtype=torch.int32)
        prefix = torch.tensor([5, 12], dtype=torch.int32)

        outputs = proposer.run_propose_step(
            _propose_inputs(anchors, prefix, 3), None, torch.device("cpu")
        )

        (query_ids, query_positions, prefix_lengths, active) = (
            proposer.query_block_calls[0]
        )
        # Column 0 carries the anchor; every other column is forced to the
        # configured noise token regardless of what the engine sent.
        self.assertEqual(query_ids[:, 0].tolist(), [4, 8])
        self.assertTrue(torch.all(query_ids[:, 1:] == 9))
        # Positions continue each request's committed prefix.
        self.assertEqual(
            query_positions.tolist(), [[5, 6, 7], [12, 13, 14]]
        )
        self.assertEqual(prefix_lengths.tolist(), [5, 12])
        self.assertEqual(active.tolist(), [True, True])
        # The tail must equal the Markov chain over the fake base logits.
        expected = proposer._sample_sequential_markov(
            torch.zeros(2, 3, 11), anchors
        )
        self.assertTrue(torch.equal(outputs.draft_tokens, expected))

    def test_zero_prefix_padding_slot_is_inactive(self) -> None:
        proposer = _ProposeProposer()
        anchors = torch.tensor([4, 8], dtype=torch.int32)
        prefix = torch.tensor([5, 0], dtype=torch.int32)

        proposer.run_propose_step(
            _propose_inputs(anchors, prefix, 3), None, torch.device("cpu")
        )

        (_, _, _, active) = proposer.query_block_calls[0]
        self.assertEqual(active.tolist(), [True, False])

    def test_empty_batch_still_runs_collective_block_once(self) -> None:
        # Empty DP ranks must execute every collective layer so EP stays
        # balanced; only the head is skipped and the outputs are empty.
        proposer = _ProposeProposer()
        anchors = torch.empty(0, dtype=torch.int32)
        prefix = torch.empty(0, dtype=torch.int32)

        outputs = proposer.run_propose_step(
            _propose_inputs(anchors, prefix, 3), None, torch.device("cpu")
        )

        self.assertEqual(len(proposer.query_block_calls), 1)
        self.assertEqual(tuple(outputs.hidden_states.shape), (0, 4))
        self.assertEqual(tuple(outputs.draft_tokens.shape), (0, 3))

    def test_rejects_wrong_token_count(self) -> None:
        proposer = _ProposeProposer()
        inputs = _propose_inputs(
            torch.tensor([4], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            3,
        )
        inputs.input_ids = inputs.input_ids[:-1]
        with self.assertRaisesRegex(
            RuntimeError, r"must contain exactly B\*gamma tokens"
        ):
            proposer.run_propose_step(inputs, None, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
