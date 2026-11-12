"""SM120 cases split out of test_prefill_fast_path.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


class PrefillFastPathTestSm120(unittest.TestCase):
    """Sibling of ``PrefillFastPathTest`` carrying the SM120 cases."""

    def test_embed_full_reassembles_sharded_embedding(self):
        class ShardedEmbedding:
            args = SimpleNamespace(dim=4)

            @staticmethod
            def embed(input_ids):
                values = input_ids.to(torch.float32).reshape(-1, 1)
                return torch.cat((values, values + 1), dim=-1).view(*input_ids.shape, 2)

        def fake_all_gather(local, group):
            self.assertEqual(group.name, "TP")
            return torch.cat((local, local + 100), dim=0)

        input_ids = torch.tensor([[3, 4], [5, 6]], dtype=torch.long)
        with patch(
            "rtp_llm.models_py.distributed.collective_torch.all_gather",
            side_effect=fake_all_gather,
        ):
            out = V4Transformer.embed_full(ShardedEmbedding(), input_ids)

        torch.testing.assert_close(
            out,
            torch.tensor(
                [
                    [[3.0, 4.0, 103.0, 104.0], [4.0, 5.0, 104.0, 105.0]],
                    [[5.0, 6.0, 105.0, 106.0], [6.0, 7.0, 106.0, 107.0]],
                ]
            ),
        )

    def test_embed_full_rejects_incompatible_shard_width(self):
        class BadEmbedding:
            args = SimpleNamespace(dim=5)

            @staticmethod
            def embed(input_ids):
                return torch.zeros(*input_ids.shape, 2)

        with self.assertRaisesRegex(ValueError, "incompatible hidden size"):
            V4Transformer.embed_full(BadEmbedding(), torch.tensor([1, 2]))


if __name__ == "__main__":
    unittest.main()
