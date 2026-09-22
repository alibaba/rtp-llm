import unittest

import torch

from rtp_llm.models_py.speculative.dspark_single_simulator import (
    SingleRequestDSparkSimulator,
    run_demo,
)


class SingleRequestDSparkSimulatorTest(unittest.TestCase):
    def test_full_accept_reject_and_next_round_overwrite(self):
        full, rejected, healed = run_demo()

        self.assertEqual(full.proposals, [1, 2, 3])
        self.assertEqual(full.accepted_draft_tokens, 3)
        self.assertEqual(full.committed_tokens, [1, 2, 3, 4])
        self.assertEqual(full.query_ids, [0, 5, 5])

        self.assertEqual(rejected.proposals, [5, 2, 3])
        self.assertEqual(rejected.accepted_draft_tokens, 1)
        self.assertEqual(rejected.committed_tokens, [5, 0])
        self.assertEqual(rejected.correction_position, 6)
        self.assertEqual(rejected.draft_cache[-1], 3)  # unreachable stale row

        self.assertEqual(healed.proposal_overwrites, [7])
        self.assertEqual(healed.proposals, [1, 2, 3])
        self.assertEqual(healed.accepted_draft_tokens, 3)
        self.assertEqual(healed.committed_sequence, [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4])
        self.assertEqual(healed.draft_cache, healed.committed_sequence)

    def test_query_width_excludes_anchor_prediction_when_configured(self):
        vocab = 4
        w1 = torch.eye(vocab)
        w2 = torch.zeros(vocab, vocab)
        for previous in range(vocab):
            w2[(previous + 1) % vocab, previous] = 10.0

        def target_logits(prefix):
            logits = torch.full((vocab,), -10.0)
            logits[(prefix[-1] + 1) % vocab] = 10.0
            return logits

        simulator = SingleRequestDSparkSimulator(
            prompt_tokens=[0],
            proposal_width=2,
            noise_token_id=3,
            markov_w1=w1,
            markov_w2=w2,
            target_logits=target_logits,
            sample_from_anchor=False,
        )
        trace = simulator.run_greedy_round(torch.zeros(2, vocab))
        self.assertEqual(trace.query_ids, [0, 3, 3])
        self.assertEqual(trace.proposals, [1, 2])


if __name__ == "__main__":
    unittest.main()
