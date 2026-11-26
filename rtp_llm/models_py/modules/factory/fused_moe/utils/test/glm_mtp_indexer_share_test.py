import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.glm53_mtp_indexer import (
    Glm53MtpIndexerSeeds,
    mtp_indexer_share_active,
)


class GlmMtpIndexerShareTest(unittest.TestCase):
    def test_activation_requires_checkpoint_opt_in_and_supported_geometry(self):
        config = SimpleNamespace(index_share_for_mtp_iteration=True)
        parallelism = SimpleNamespace(
            pp_size=1, prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
        )
        self.assertTrue(mtp_indexer_share_active(config, parallelism, 1, 2))
        self.assertFalse(mtp_indexer_share_active(config, parallelism, 2, 2))
        self.assertFalse(mtp_indexer_share_active(config, parallelism, 1, 0))
        config.index_share_for_mtp_iteration = False
        self.assertFalse(mtp_indexer_share_active(config, parallelism, 1, 2))
        config.index_share_for_mtp_iteration = True
        parallelism.pp_size = 2
        self.assertFalse(mtp_indexer_share_active(config, parallelism, 1, 2))
        parallelism.pp_size = 1
        parallelism.prefill_cp_config.is_enabled = lambda: True
        self.assertFalse(mtp_indexer_share_active(config, parallelism, 1, 2))

    def test_accepted_rows_expand_in_request_order_with_frozen_raw_tail(self):
        seeds = Glm53MtpIndexerSeeds(4, 2, 4, "cpu")
        pooled = torch.tensor([[0, -1], [0, -1], [1, 0], [1, 0]], dtype=torch.int32)
        positions = torch.tensor([3, 4, 7, 9], dtype=torch.int32)
        seeds.store(pooled, torch.tensor([3, 1], dtype=torch.int32), positions)
        expected = torch.tensor(
            [
                [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1],
                [0, 1, 2, 3, -1, -1, -1, -1, 4, -1, -1],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(seeds.snapshot(2), expected))
        pointer = seeds.indices.data_ptr()
        seeds.load(expected.flip(0))
        self.assertEqual(seeds.indices.data_ptr(), pointer)
        self.assertTrue(torch.equal(seeds.snapshot(2), expected.flip(0)))
        # Executor snapshots own their result across subsequent batch reorders.
        owned = seeds.snapshot(2).clone()
        seeds.load(torch.full_like(expected, -1))
        self.assertTrue(torch.equal(owned, expected.flip(0)))
        self.assertEqual(tuple(seeds.snapshot(0).shape), (0, 11))

    def test_dense_and_cold_start_keep_incomplete_tail(self):
        seeds = Glm53MtpIndexerSeeds(2, 2, 4, "cpu")
        rows = torch.tensor([0, 1], dtype=torch.int32)
        seeds.store(None, rows, torch.tensor([0, 6], dtype=torch.int32))
        expected = torch.tensor(
            [[-1] * 8 + [0, -1, -1], [0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6]],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(seeds.snapshot(2), expected))
        seeds.store(
            torch.empty((2, 0), dtype=torch.int32),
            rows,
            torch.tensor([0, 4], dtype=torch.int32),
        )
        self.assertTrue(
            torch.equal(
                seeds.snapshot(2),
                torch.tensor(
                    [[-1] * 8 + [0, -1, -1], [-1] * 8 + [4, -1, -1]], dtype=torch.int32
                ),
            )
        )

    def test_cp_seed_collects_final_query_without_global_prompt_gather(self):
        from rtp_llm.models_py.distributed.sequence_parallel import token_shard_layout
        from rtp_llm.models_py.distributed.zigzag_token_layout import ZigzagTokenLayout

        # Short requests, padding, unequal shards and different final-row owners.
        for q_lens, world in (([3, 17, 33], 8), ([1, 2], 8), ([9], 2)):
            with self.subTest(lengths=q_lens, world=world):
                ends = torch.tensor(q_lens).cumsum(0).to(torch.int32) - 1
                expected_pooled = torch.stack((ends, torch.zeros_like(ends)), dim=1)
                prefix = torch.full((len(q_lens),), 512, dtype=torch.int32)
                contributions = []
                for rank in range(world):
                    layout = ZigzagTokenLayout(
                        q_lens,
                        token_shard_layout(sum(q_lens), world, rank),
                        world,
                        rank,
                        torch.device("cpu"),
                    )
                    canonical = {}
                    for fragment in layout.fragments:
                        if fragment.cp_rank == rank:
                            for offset in range(fragment.length):
                                canonical[fragment.cp_offset + offset] = (
                                    fragment.canonical_offset + offset
                                )
                    ids = torch.tensor(sorted(canonical), dtype=torch.int32)
                    pooled = torch.tensor(
                        [[canonical[int(row)], 0] for row in ids], dtype=torch.int32
                    ).reshape(-1, 2)
                    seeds = Glm53MtpIndexerSeeds(len(q_lens), 2, 4, "cpu")
                    # This is the old failure: local CP top-k and global positions
                    # are different coordinate systems, even with TP=8/DP=1.
                    if pooled.shape[0] != sum(q_lens):
                        with self.assertRaisesRegex(ValueError, "invalid pooled"):
                            seeds.store(
                                pooled,
                                ends,
                                torch.arange(sum(q_lens), dtype=torch.int32),
                            )

                    def reduce(tensor, group):
                        self.assertEqual(group, Group.TP)
                        self.assertEqual(tuple(tensor.shape), (len(q_lens), 2))
                        contributions.append(tensor.clone())
                        return expected_pooled

                    with patch(
                        "rtp_llm.models_py.modules.glm53_mtp_indexer.all_reduce",
                        side_effect=reduce,
                    ):
                        seeds.store_prefill_cp(pooled, layout, ids, prefix)
                    reference = Glm53MtpIndexerSeeds(len(q_lens), 2, 4, "cpu")
                    reference.store(
                        expected_pooled,
                        torch.arange(len(q_lens), dtype=torch.int32),
                        prefix + torch.tensor(q_lens, dtype=torch.int32) - 1,
                    )
                    torch.testing.assert_close(
                        seeds.snapshot(len(q_lens)), reference.snapshot(len(q_lens))
                    )
                torch.testing.assert_close(
                    torch.stack(contributions).sum(0).to(torch.int32), expected_pooled
                )

    def test_cp_dense_seed_preserves_short_context_tail(self):
        seeds = Glm53MtpIndexerSeeds(2, 2, 4, "cpu")
        layout = SimpleNamespace(q_lens=[1, 6])
        with patch("rtp_llm.models_py.modules.glm53_mtp_indexer.all_reduce") as reduce:
            seeds.store_prefill_cp(
                None, layout, None, torch.zeros(2, dtype=torch.int32)
            )
            reduce.assert_not_called()
        reference = Glm53MtpIndexerSeeds(2, 2, 4, "cpu")
        reference.store(
            None,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 5], dtype=torch.int32),
        )
        torch.testing.assert_close(seeds.snapshot(2), reference.snapshot(2))

    def test_bad_seed_geometry_and_capacity_fail_before_replay(self):
        seeds = Glm53MtpIndexerSeeds(2, 2, 4, "cpu")
        for value in (
            torch.zeros((2, 11), dtype=torch.int64),
            torch.zeros((2, 2), dtype=torch.int32),
            torch.zeros((3, 11), dtype=torch.int32),
            torch.zeros(11, dtype=torch.int32),
        ):
            with self.assertRaises(ValueError):
                seeds.load(value)
        for count in (-1, 3):
            with self.assertRaises(ValueError):
                seeds.snapshot(count)
        with self.assertRaises(ValueError):
            seeds.store(
                torch.zeros((2, 3), dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
