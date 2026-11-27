import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel


class Glm5MtpIndexerStateTest(unittest.TestCase):
    def setUp(self):
        self.model = GenericMoeMTPModel.__new__(GenericMoeMTPModel)
        nn.Module.__init__(self.model)
        self.model.config = SimpleNamespace(attn_config=SimpleNamespace(indexer_topk=4))
        self.model._mtp_indexer_share_enabled = True
        self.model._mtp_shared_topk_indices = torch.full((3, 4), -1, dtype=torch.int32)

    def reuse(self, positions):
        return self.model._get_mtp_reuse_topk_indices(
            torch.zeros(len(positions), 8),
            SimpleNamespace(
                fmha_params=SimpleNamespace(positions_d=torch.tensor(positions))
            ),
        )

    def test_seed_rows_follow_request_order(self):
        seed = torch.tensor(
            [[0, 1, -1, -1], [2, 1, 0, -1], [3, 2, 1, 0]], dtype=torch.int32
        )
        self.model._store_mtp_topk_indices(
            seed, torch.tensor([2, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(
            self.model.snapshot_mtp_indexer_topk(2), seed[[2, 0]]
        )
        seed.zero_()
        self.assertEqual(self.model.snapshot_mtp_indexer_topk(2)[0, 0].item(), 3)

    def test_consecutive_reuse_preserves_new_positions_and_evicts_tail(self):
        self.model.load_mtp_indexer_topk(
            torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)
        )
        for position, expected in (
            (2, [2, 0, 1, -1]),
            (3, [3, 2, 0, 1]),
            (4, [4, 3, 2, 0]),
            (3, [4, 3, 2, 0]),
        ):
            with self.subTest(position=position):
                torch.testing.assert_close(
                    self.reuse([position]), torch.tensor([expected], dtype=torch.int32)
                )

    def test_batch_shrink_and_reload(self):
        seed = torch.tensor(
            [[0, 1, -1, -1], [0, 1, 2, -1], [0, 1, 2, 3]], dtype=torch.int32
        )
        self.model.load_mtp_indexer_topk(seed)
        self.reuse([2])
        torch.testing.assert_close(
            self.model.snapshot_mtp_indexer_topk(3)[1:], seed[1:]
        )
        self.model.load_mtp_indexer_topk(seed.flip(0))
        torch.testing.assert_close(
            self.model.snapshot_mtp_indexer_topk(3), seed.flip(0)
        )

    def test_reject_invalid_seed_and_batch(self):
        for seed in (
            torch.zeros(2, 4),
            torch.zeros(2, 3, dtype=torch.int32),
            torch.zeros(4, 4, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
        ):
            with self.subTest(shape=seed.shape, dtype=seed.dtype):
                with self.assertRaises(RuntimeError):
                    self.model.load_mtp_indexer_topk(seed)
        for batch in (-1, 4):
            with self.assertRaises(RuntimeError):
                self.model.snapshot_mtp_indexer_topk(batch)
        with self.assertRaises(RuntimeError):
            self.reuse([1, 2, 3, 4])
        with self.assertRaises(RuntimeError):
            self.model._get_mtp_reuse_topk_indices(
                torch.zeros(2, 8),
                SimpleNamespace(
                    fmha_params=SimpleNamespace(positions_d=torch.tensor([1]))
                ),
            )

    def test_disabled_share_is_noop(self):
        self.model._mtp_indexer_share_enabled = False
        original = self.model._mtp_shared_topk_indices.clone()
        self.model.load_mtp_indexer_topk(torch.ones(5))
        self.model.set_mtp_indexer_role(2)
        self.assertEqual(self.model._mtp_indexer_role, 0)
        torch.testing.assert_close(self.model.snapshot_mtp_indexer_topk(99), original)
        with self.assertRaises(ValueError):
            self.model.set_mtp_indexer_role(3)

    def test_graph_clone_runs_draft_forward_with_its_frontend(self):
        model = self.model
        model.config.num_layers = 0
        model.config.vocab_size = 4
        model.config.capture_aux_hidden_layer_ids = []
        model.parallelism_config = model.weight = model.moe_config = None
        model.fmha_config = model.py_hw_kernel_config = None
        model.max_generate_batch_size = 2
        model.device_type = torch.device("cpu")
        model.layers = nn.ModuleList()
        model.embed_tokens = nn.Embedding(4, 2)
        model.embed_tokens.tp_size = 1
        model.pre_fc_norm_embedding = model.pre_fc_norm_hidden = nn.Identity()
        model.fc = nn.Identity()
        model.norm = lambda hidden, residual: (hidden + residual, residual)
        model._mtp_indexer_share_enabled = False
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1]),
            input_hiddens=torch.tensor([[2.0, 3.0]]),
            prefill_layer_progress=None,
        )
        expected = torch.cat(
            (model.embed_tokens(inputs.input_ids), inputs.input_hiddens), -1
        )
        for frontend in (None, object()):
            with self.subTest(fused=frontend is not None):
                model._mtp_frontend_native = frontend
                clone = model.clone_for_cuda_graph()
                actual = clone(inputs, fmha_impl=object()).hidden_states
                torch.testing.assert_close(actual, expected)
                self.assertIs(clone._mtp_frontend_native, frontend)

    def test_only_position_zero_embedding_is_masked(self):
        embeddings = torch.arange(12).reshape(3, 4).float()
        result = self.model._mask_position_zero_embeddings(
            embeddings,
            torch.tensor([0, 5, 0]),
        )
        expected = embeddings.clone()
        expected[[0, 2]] = 0
        torch.testing.assert_close(result, expected)
        self.assertEqual(embeddings[2, 0].item(), 8)


if __name__ == "__main__":
    unittest.main()
