import unittest

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
        expected_tokens, expected_probs = _reference_chain(
            self.base_logits, self.anchors, self.proposer.w1, self.proposer.w2
        )

        tokens, probs = self.proposer._sample_sequential_markov(
            self.base_logits, self.anchors
        )

        self.assertTrue(torch.equal(tokens, expected_tokens))
        assert probs is not None
        self.assertTrue(torch.equal(probs, expected_probs))

    def test_tokens_feed_next_step_bias(self) -> None:
        # Step 0 must consume the anchor bias; step 1 must consume the step-0
        # winner, not the anchor.
        head = self.proposer.markov_head
        tokens, _ = self.proposer._sample_sequential_markov(
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

    def test_probabilities_can_be_skipped(self) -> None:
        tokens, probs = self.proposer._sample_sequential_markov(
            self.base_logits, self.anchors, need_probabilities=False
        )
        self.assertIsNone(probs)
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
        self.assertEqual(tuple(outputs.draft_probs.shape), (2, 3, 17))
        self.assertEqual(outputs.draft_probs.dtype, torch.float32)

    def test_hooks_require_subclass_implementation(self) -> None:
        proposer = DSparkProposerMixin()
        with self.assertRaises(NotImplementedError):
            proposer.combine_hidden_states(torch.zeros(1, 8))
        with self.assertRaises(NotImplementedError):
            proposer.compute_draft_logits(torch.zeros(1, 3, 4, 8))
        # Full-vocabulary default: identity mapping.
        ids = torch.tensor([3, 5])
        self.assertTrue(torch.equal(proposer.map_draft_to_target(ids), ids))

    def test_markov_head_rejects_mismatched_weights(self) -> None:
        w = torch.zeros(11, 4)
        with self.assertRaises(ValueError):
            DSparkMarkovHead(w, torch.zeros(11, 5), vocab_size=11, rank=4)
        with self.assertRaises(ValueError):
            DSparkMarkovHead(w, w, vocab_size=12, rank=4)


if __name__ == "__main__":
    unittest.main()
