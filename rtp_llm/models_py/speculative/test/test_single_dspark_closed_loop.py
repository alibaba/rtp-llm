import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
from rtp_llm.models_py.speculative.dspark_single_simulator import (
    SingleRequestDSparkSimulator,
)


class _SyntheticDSparkProposer(DSparkProposerMixin):
    """Actual shared proposer flow with deterministic synthetic backbone rows."""

    def __init__(self, width: int, vocab: int):
        self.init_dspark_proposer(
            width=width,
            noise_token_id=vocab - 1,
            aux_feature_dim=2,
            hidden_dim=vocab,
        )
        self.next_base_logits = torch.zeros(width, vocab)
        self.committed_positions = []
        self.last_query_ids = None

    def combine_hidden_states(self, features):
        return torch.nn.functional.pad(features, (0, self._dspark_hidden_dim - 2))

    def commit_feature_rows(
        self, main_x, request_ids, positions, committed_ends, inputs, commit_ctx=None
    ):
        del main_x, request_ids, committed_ends, inputs, commit_ctx
        self.committed_positions.extend(int(value) for value in positions.tolist())

    def forward_query_block(
        self,
        query_ids,
        query_positions,
        prefix_lengths,
        active_requests,
        inputs,
        fmha_impl,
    ):
        del query_positions, prefix_lengths, active_requests, inputs, fmha_impl
        self.last_query_ids = query_ids.clone()
        return self.next_base_logits.clone()


def _inputs(*, input_ids=None, hidden=None, length: int, prefix: int):
    return SimpleNamespace(
        input_ids=(
            torch.empty(0, dtype=torch.int32) if input_ids is None else input_ids
        ),
        input_hiddens=hidden,
        attention_inputs=SimpleNamespace(
            input_lengths=torch.tensor([length], dtype=torch.int32),
            prefix_lengths=torch.tensor([prefix], dtype=torch.int32),
        ),
    )


class SingleDSparkClosedLoopTest(unittest.TestCase):
    def test_commit_propose_markov_verify_reject_and_recommit(self):
        vocab, width = 6, 3
        w1 = torch.eye(vocab)
        w2 = torch.zeros(vocab, vocab)
        for previous in range(vocab):
            w2[(previous + 1) % vocab, previous] = 20.0

        def target_logits(prefix):
            logits = torch.full((vocab,), -20.0)
            logits[(prefix[-1] + 1) % vocab] = 20.0
            return logits

        proposer = _SyntheticDSparkProposer(width, vocab)
        simulator = SingleRequestDSparkSimulator(
            prompt_tokens=[0],
            proposal_width=width,
            noise_token_id=vocab - 1,
            markov_w1=w1,
            markov_w2=w2,
            target_logits=target_logits,
        )

        proposer.run_commit_step(
            _inputs(hidden=torch.tensor([[0.0, 1.0]]), length=1, prefix=0),
            torch.device("cpu"),
        )

        traces = []
        round_logits = [
            torch.zeros(width, vocab),
            torch.zeros(width, vocab),
            torch.zeros(width, vocab),
        ]
        round_logits[1][1, 2] = 50.0  # force a middle rejection
        for base_logits in round_logits:
            proposer.next_base_logits = base_logits
            anchor = simulator.committed[-1]
            proposal_output = proposer.run_propose_step(
                _inputs(
                    input_ids=torch.tensor(
                        [anchor] + [0] * (width - 1), dtype=torch.int32
                    ),
                    length=width,
                    prefix=len(simulator.committed),
                ),
                fmha_impl=None,
                device=torch.device("cpu"),
            )
            trace = simulator.run_greedy_round(proposal_output.hidden_states)
            traces.append(trace)
            self.assertEqual(proposer.last_query_ids[0].tolist(), trace.query_ids)

            commit_start = len(trace.committed_sequence) - len(trace.committed_tokens)
            proposer.run_commit_step(
                _inputs(
                    hidden=torch.ones(len(trace.committed_tokens), 2),
                    length=len(trace.committed_tokens),
                    prefix=commit_start,
                ),
                torch.device("cpu"),
            )

        self.assertEqual([trace.accepted_draft_tokens for trace in traces], [3, 1, 3])
        self.assertEqual(traces[2].proposal_overwrites, [7])
        self.assertEqual(
            proposer.committed_positions,
            list(range(len(simulator.committed))),
        )
        self.assertEqual(simulator.draft_cache, simulator.committed)


if __name__ == "__main__":
    unittest.main()
