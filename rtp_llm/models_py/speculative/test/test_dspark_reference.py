import unittest

import torch

from rtp_llm.models_py.speculative.dspark_reference import sample_markov_drafts


class DSparkReferenceTest(unittest.TestCase):
    def test_greedy_markov_chain_uses_previous_draft(self):
        base = torch.zeros(1, 3, 4)
        # token x biases token (x + 1) % 4, proving that every step consumes
        # the previous sampled token rather than reusing the anchor.
        w1 = torch.eye(4)
        w2 = torch.zeros(4, 4)
        for token in range(4):
            w2[(token + 1) % 4, token] = 20.0
        tokens, probabilities = sample_markov_drafts(
            base, torch.tensor([0]), w1, w2, temperature=0.0
        )
        self.assertEqual(tokens.tolist(), [[1, 2, 3]])
        self.assertEqual(tuple(probabilities.shape), (1, 3, 4))
        self.assertTrue(torch.equal(probabilities.max(dim=-1).values, torch.ones(1, 3)))

    def test_seeded_sampling_is_reproducible(self):
        base = torch.zeros(2, 2, 3)
        w1 = torch.zeros(3, 2)
        w2 = torch.zeros(3, 2)
        first = torch.Generator().manual_seed(17)
        second = torch.Generator().manual_seed(17)
        lhs = sample_markov_drafts(base, torch.tensor([0, 1]), w1, w2, generator=first)[
            0
        ]
        rhs = sample_markov_drafts(
            base, torch.tensor([0, 1]), w1, w2, generator=second
        )[0]
        self.assertTrue(torch.equal(lhs, rhs))

    def test_reduced_vocab_maps_sample_back_before_next_markov_step(self):
        base = torch.tensor([[[-100.0, -100.0, 100.0], [0.0, 0.0, 0.0]]])
        w1 = torch.zeros(6, 1)
        w1[5, 0] = 1.0
        w2 = torch.tensor([[-100.0], [0.0], [100.0]])
        tokens, probabilities = sample_markov_drafts(
            base,
            torch.tensor([0]),
            w1,
            w2,
            temperature=0.0,
            draft_to_target=torch.tensor([1, 3, 5]),
        )
        self.assertEqual(tokens.tolist(), [[5, 5]])
        self.assertEqual(tuple(probabilities.shape), (1, 2, 3))

    def test_rejects_transposed_runtime_w2_layout(self):
        with self.assertRaisesRegex(ValueError, "low-rank"):
            sample_markov_drafts(
                torch.zeros(1, 1, 3),
                torch.tensor([0]),
                torch.zeros(4, 2),
                torch.zeros(2, 3),
            )

    def test_rejects_negative_temperature(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            sample_markov_drafts(
                torch.zeros(1, 1, 3),
                torch.tensor([0]),
                torch.zeros(3, 2),
                torch.zeros(3, 2),
                temperature=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
