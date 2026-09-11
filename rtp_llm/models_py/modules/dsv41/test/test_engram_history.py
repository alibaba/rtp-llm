import os
import unittest
from types import SimpleNamespace

import torch
from standalone_load import load_component

_engram = load_component("rtp_v41_engram_test", "models_py/modules/dsv41/engram.py")
EngramHash = _engram.EngramHash
committed_history = _engram.committed_history


class EngramHistoryTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get("DSV41_TEST_DEVICE", "cuda"))
        config = SimpleNamespace(
            text={
                "engram_compressed_vocab_size": 4,
                "engram_pad_token_id": 2,
                "engram_max_ngram_size": 4,
                "engram_layer_ids": [1, 14],
                "engram_num_embeddings": [56, 180],
                "engram_n_heads": 2,
                "engram_vocab_size": 3,
            }
        )
        self.hasher = EngramHash(config, [0, 1, 2, 3]).to(self.device)
        self.history = torch.full((2, 3), 2, dtype=torch.int64, device=self.device)
        self.valid = torch.zeros_like(self.history, dtype=torch.bool)
        self.ids = torch.tensor(
            [[0, 1, 3, 1, 0, 3, 1], [3, 0, 1, 0, 3, 1, 0]], device=self.device
        )

    def test_chunk_and_batch_order_are_request_local(self):
        full = self.hasher(self.ids, self.history, self.valid)
        prefix = self.hasher(self.ids[:, :3], self.history, self.valid)
        history, valid = committed_history(
            self.history,
            self.valid,
            self.ids[:, :3],
            torch.ones_like(self.ids[:, :3], dtype=torch.bool),
        )
        suffix = self.hasher(self.ids[:, 3:], history, valid)
        self.assertTrue(torch.equal(torch.cat((prefix, suffix), 1), full))
        self.assertTrue(
            torch.equal(
                self.hasher(self.ids.flip(0), self.history, self.valid), full.flip(0)
            )
        )

    def test_image_boundary_blocks_older_text(self):
        mask = torch.ones_like(self.ids, dtype=torch.bool)
        mask[:, 2:4] = False
        changed = self.ids.clone()
        changed[:, :2] = 2
        actual = self.hasher(self.ids, self.history, self.valid, mask)
        other = self.hasher(changed, self.history, self.valid, mask)
        self.assertTrue(torch.equal(actual[:, 2:], other[:, 2:]))

    def test_rejected_verify_rows_do_not_mutate_history(self):
        expected = self.hasher(self.ids[:, :1], self.history, self.valid)
        self.hasher(self.ids, self.history, self.valid)
        self.assertTrue(
            torch.equal(
                self.hasher(self.ids[:, :1], self.history, self.valid), expected
            )
        )
        ids, valid = committed_history(
            self.history,
            self.valid,
            self.ids[:, :1],
            torch.ones_like(self.ids[:, :1], dtype=torch.bool),
        )
        self.assertTrue(torch.equal(ids[:, -1], self.ids[:, 0]))
        self.assertTrue(
            torch.equal(
                valid.sum(1), torch.ones(2, dtype=torch.int64, device=self.device)
            )
        )


if __name__ == "__main__":
    unittest.main()
