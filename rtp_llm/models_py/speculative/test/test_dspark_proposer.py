import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin


class _TinyProposer(DSparkProposerMixin):
    """Smallest possible DSparkProposerMixin subclass."""

    def __init__(self, *, width: int, query_width=None):
        self.init_dspark_proposer(
            width=width,
            query_width=query_width,
            noise_token_id=1,
            aux_feature_dim=8,
            hidden_dim=4,
        )


class ProposerContractTest(unittest.TestCase):
    def test_init_rejects_bad_geometry(self) -> None:
        proposer = DSparkProposerMixin()
        with self.assertRaises(ValueError):
            proposer.init_dspark_proposer(
                width=0,
                noise_token_id=1,
                aux_feature_dim=8,
                hidden_dim=4,
            )
        with self.assertRaises(ValueError):
            proposer.init_dspark_proposer(
                width=5,
                noise_token_id=-1,
                aux_feature_dim=8,
                hidden_dim=4,
            )
        with self.assertRaises(ValueError):
            proposer.init_dspark_proposer(
                width=5,
                noise_token_id=1,
                aux_feature_dim=0,
                hidden_dim=4,
            )

    def test_empty_outputs_follow_configured_geometry(self) -> None:
        proposer = DSparkProposerMixin()
        proposer.init_dspark_proposer(
            width=3, noise_token_id=1, aux_feature_dim=24, hidden_dim=8
        )
        outputs = proposer.dspark_empty_outputs(2, torch.device("cpu"))
        self.assertEqual(tuple(outputs.hidden_states.shape), (6, 8))
        self.assertEqual(outputs.hidden_states.dtype, torch.bfloat16)

    def test_query_width_may_exclude_anchor_from_predictions(self) -> None:
        proposer = _TinyProposer(width=3, query_width=4)
        outputs = proposer.dspark_empty_outputs(2, torch.device("cpu"))
        self.assertEqual(tuple(outputs.hidden_states.shape), (8, 4))

    def test_hooks_require_subclass_implementation(self) -> None:
        proposer = DSparkProposerMixin()
        with self.assertRaises(NotImplementedError):
            proposer.combine_hidden_states(torch.zeros(1, 8))
        hidden = torch.zeros(1, 3, 4, 8)
        self.assertIs(proposer.compute_draft_hidden_states(hidden), hidden)
        with self.assertRaises(NotImplementedError):
            proposer.commit_feature_rows(
                torch.zeros(1, 8),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                None,
            )


class _CommitProposer(DSparkProposerMixin):
    """Captures the rows handed to the projection and commit hooks."""

    def __init__(self, *, aux_dim: int):
        self.init_dspark_proposer(
            width=2,
            noise_token_id=1,
            aux_feature_dim=aux_dim,
            hidden_dim=aux_dim,
        )
        self.seen_features = None
        self.committed = None

    def combine_hidden_states(self, features: torch.Tensor) -> torch.Tensor:
        self.seen_features = features
        return features

    def commit_feature_rows(
        self, main_x, req, positions, committed_ends, inputs, commit_ctx=None
    ):
        self.committed = (main_x, req, positions, committed_ends)


class CommitStepTest(unittest.TestCase):
    def test_commit_derives_windows_from_standard_prefill_geometry(self) -> None:
        # input_hiddens is a view of the shared MTP hidden buffer whose
        # DSpARK row width equals the aux payload — the commit step must not
        # copy it, and its row windows come straight from
        # input_lengths/prefix_lengths.
        aux_dim, rows = 6, 4
        proposer = _CommitProposer(aux_dim=aux_dim)
        hidden = torch.arange(rows * aux_dim, dtype=torch.float32).reshape(
            rows, aux_dim
        )
        inputs = SimpleNamespace(
            input_hiddens=hidden,
            attention_inputs=SimpleNamespace(
                input_lengths=torch.tensor([3, 1], dtype=torch.int32),
                prefix_lengths=torch.tensor([10, 20], dtype=torch.int32),
            ),
        )

        outputs = proposer.run_commit_step(inputs, torch.device("cpu"))

        torch.testing.assert_close(proposer.seen_features, hidden)
        self.assertEqual(proposer.seen_features.data_ptr(), hidden.data_ptr())
        main_x, req, positions, committed_ends = proposer.committed
        self.assertEqual(main_x.shape, (rows, aux_dim))
        self.assertEqual(req.tolist(), [0, 0, 0, 1])
        self.assertEqual(positions.tolist(), [10, 11, 12, 20])
        self.assertEqual(committed_ends.tolist(), [13, 21])
        torch.testing.assert_close(outputs.hidden_states, hidden)
        self.assertEqual(outputs.hidden_states.data_ptr(), hidden.data_ptr())


class _ProposeProposer(_TinyProposer):
    """Records the query block and synthesizes normalized hidden states."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.query_call = None
        self.seen_hidden = None

    def forward_query_block(
        self,
        query_ids,
        query_positions,
        prefix_lengths,
        active_requests,
        inputs,
        fmha_impl,
    ):
        self.query_call = (query_ids, query_positions, prefix_lengths, active_requests)
        return torch.zeros(query_ids.numel(), 4)

    def compute_draft_hidden_states(self, hidden):
        self.seen_hidden = hidden + 1
        return self.seen_hidden


def _propose_inputs(input_ids, input_lengths, prefix_lengths):
    return SimpleNamespace(
        input_ids=input_ids,
        attention_inputs=SimpleNamespace(
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
        ),
    )


class ProposeStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 5
        self.proposer = _ProposeProposer(width=self.width)
        self.device = torch.device("cpu")

    def test_query_block_geometry_and_hidden_states(self) -> None:
        anchors = torch.tensor([3, 9], dtype=torch.int32)
        input_ids = torch.zeros(2 * self.width, dtype=torch.int32)
        input_ids[0], input_ids[self.width] = anchors[0], anchors[1]
        inputs = _propose_inputs(
            input_ids,
            torch.tensor([self.width, self.width], dtype=torch.int32),
            torch.tensor([7, 0], dtype=torch.int32),
        )

        outputs = self.proposer.run_propose_step(
            inputs, fmha_impl=None, device=self.device
        )

        query_ids, positions, prefix, active = self.proposer.query_call
        self.assertEqual(query_ids[:, 0].tolist(), anchors.tolist())
        self.assertTrue((query_ids[:, 1:] == 1).all())  # noise_token_id
        self.assertEqual(positions[0].tolist(), [7, 8, 9, 10, 11])
        self.assertEqual(positions[1].tolist(), [0, 1, 2, 3, 4])
        # Zero-prefix rows are CUDA-graph padding, not live requests.
        self.assertEqual(active.tolist(), [True, False])
        self.assertTrue(torch.equal(outputs.hidden_states, self.proposer.seen_hidden))

    def test_rejects_wrong_token_count(self) -> None:
        inputs = _propose_inputs(
            torch.zeros(3, dtype=torch.int32),
            torch.tensor([self.width], dtype=torch.int32),
            torch.tensor([7], dtype=torch.int32),
        )
        with self.assertRaisesRegex(RuntimeError, "exactly B\\*gamma tokens"):
            self.proposer.run_propose_step(inputs, fmha_impl=None, device=self.device)

    def test_requires_per_request_prefix(self) -> None:
        inputs = _propose_inputs(
            torch.zeros(self.width, dtype=torch.int32),
            torch.tensor([self.width], dtype=torch.int32),
            None,
        )
        with self.assertRaisesRegex(RuntimeError, "prefix_lengths"):
            self.proposer.run_propose_step(inputs, fmha_impl=None, device=self.device)

    def test_empty_batch_still_runs_collective_layers(self) -> None:
        inputs = _propose_inputs(
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
        )

        outputs = self.proposer.run_propose_step(
            inputs, fmha_impl=None, device=self.device
        )

        # Empty DP ranks must still execute the collective attention layers.
        self.assertIsNotNone(self.proposer.query_call)
        self.assertEqual(tuple(outputs.hidden_states.shape), (0, 4))


if __name__ == "__main__":
    unittest.main()
