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
    """Direct transcription of the reference proposer loop."""
    previous = anchors
    tokens, probabilities = [], []
    for step in range(base_logits.shape[1]):
        markov_embed = F.embedding(previous, w1)
        markov_bias = F.linear(markov_embed, w2).float()
        logits = base_logits[:, step] + markov_bias
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        next_token = torch.argmax(probs, dim=-1)
        probabilities.append(probs)
        tokens.append(next_token.to(torch.int32))
        previous = next_token
    return (
        torch.stack(tokens, dim=1).contiguous(),
        torch.stack(probabilities, dim=1).contiguous(),
    )


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
        expected_tokens, _ = _reference_chain(
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
        with self.assertRaises(ValueError):
            self.proposer._sample_sequential_markov(
                self.base_logits[:, 0], self.anchors
            )
        with self.assertRaises(ValueError):
            self.proposer._sample_sequential_markov(
                self.base_logits[:, :3], self.anchors
            )
        with self.assertRaises(ValueError):
            self.proposer._sample_sequential_markov(
                self.base_logits, self.anchors[:1]
            )

    def test_rejects_missing_markov_head(self) -> None:
        self.proposer.markov_head = None
        with self.assertRaises(RuntimeError):
            self.proposer._sample_sequential_markov(self.base_logits, self.anchors)


class ProposerContractTest(unittest.TestCase):
    def test_init_rejects_bad_geometry(self) -> None:
        proposer = DSparkProposerMixin()
        with self.assertRaises(ValueError):
            proposer.init_dspark_proposer(
                width=0,
                noise_token_id=1,
                aux_feature_dim=8,
                hidden_dim=4,
                vocab_size=9,
            )
        with self.assertRaises(ValueError):
            proposer.init_dspark_proposer(
                width=5,
                noise_token_id=-1,
                aux_feature_dim=8,
                hidden_dim=4,
                vocab_size=9,
            )
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            DSparkMarkovHead(w, torch.zeros(11, 5), vocab_size=11, rank=4)
        with self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
