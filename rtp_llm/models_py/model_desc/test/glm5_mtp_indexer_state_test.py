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

    def test_only_position_zero_embedding_is_masked(self):
        embeddings = torch.arange(12).reshape(3, 4).float()
        result = self.model._mask_position_zero_embeddings(
            embeddings,
            SimpleNamespace(
                fmha_params=SimpleNamespace(positions_d=torch.tensor([0, 5, 0]))
            ),
        )
        expected = embeddings.clone()
        expected[[0, 2]] = 0
        torch.testing.assert_close(result, expected)
        self.assertEqual(embeddings[2, 0].item(), 8)


if __name__ == "__main__":
    unittest.main()
