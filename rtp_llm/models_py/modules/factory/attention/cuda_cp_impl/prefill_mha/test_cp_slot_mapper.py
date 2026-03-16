import unittest

import torch

from rtp_llm.ops.compute_ops import CPSlotMapper


class TestCPSlotMapper(unittest.TestCase):
    def test_not_sharded(self):
        mapper = CPSlotMapper(cp_rank=0, cp_size=1, block_size=32)
        self.assertFalse(mapper.is_sharded)
        positions = torch.arange(16)
        self.assertTrue(mapper.is_owned(positions).all())

    def test_basic_interleaving(self):
        rank0 = CPSlotMapper(cp_rank=0, cp_size=2, block_size=4)
        rank1 = CPSlotMapper(cp_rank=1, cp_size=2, block_size=4)

        self.assertTrue(rank0.is_sharded)
        self.assertEqual(rank0.virtual_block_size, 8)

        positions = torch.arange(8)
        mask0 = rank0.is_owned(positions)
        mask1 = rank1.is_owned(positions)

        # Token 0,2,4,6 -> rank0; Token 1,3,5,7 -> rank1
        expected0 = torch.tensor([True, False, True, False, True, False, True, False])
        expected1 = torch.tensor([False, True, False, True, False, True, False, True])
        torch.testing.assert_close(mask0, expected0)
        torch.testing.assert_close(mask1, expected1)

    def test_every_token_owned_by_exactly_one_rank(self):
        cp_size = 4
        block_size = 8
        total = 128
        owner = torch.full((total,), -1, dtype=torch.long)

        for r in range(cp_size):
            mapper = CPSlotMapper(r, cp_size, block_size)
            positions = torch.arange(total)
            mask = mapper.is_owned(positions)
            owned_pos = torch.where(mask)[0]
            for pos in owned_pos:
                self.assertEqual(
                    owner[pos].item(),
                    -1,
                    f"Token {pos.item()} owned by both rank {owner[pos].item()} and {r}",
                )
                owner[pos] = r

        for i in range(total):
            self.assertGreaterEqual(
                owner[i].item(), 0, f"Token {i} not owned by any rank"
            )

    def test_local_block_offset_within_range(self):
        mapper = CPSlotMapper(cp_rank=0, cp_size=2, block_size=32)
        positions = torch.arange(256)
        mask = mapper.is_owned(positions)
        owned_pos = positions[mask]
        offsets = mapper.local_block_offset(owned_pos)
        self.assertTrue((offsets >= 0).all())
        self.assertTrue((offsets < 32).all())

    def test_local_block_count(self):
        mapper = CPSlotMapper(cp_rank=0, cp_size=2, block_size=32)
        self.assertEqual(mapper.local_block_count(0), 0)
        self.assertEqual(mapper.local_block_count(1), 1)  # ceil(1/64) = 1
        self.assertEqual(mapper.local_block_count(64), 1)  # ceil(64/64) = 1
        self.assertEqual(mapper.local_block_count(65), 2)  # ceil(65/64) = 2
        self.assertEqual(mapper.local_block_count(128), 2)  # ceil(128/64) = 2

    def test_compute_slot_mapping_not_sharded(self):
        mapper = CPSlotMapper(cp_rank=0, cp_size=1, block_size=4)
        positions = torch.tensor([0, 1, 2, 3, 4, 5])
        block_table = torch.tensor([[10, 20]])  # 1 batch, 2 blocks
        batch_indices = torch.zeros(6, dtype=torch.int32)

        slots = mapper.compute_slot_mapping(positions, block_table, batch_indices)
        # block0: pos 0->10*4+0=40, pos 1->41, pos 2->42, pos 3->43
        # block1: pos 4->20*4+0=80, pos 5->81
        expected = torch.tensor([40, 41, 42, 43, 80, 81], dtype=torch.long)
        torch.testing.assert_close(slots, expected)

    def test_compute_slot_mapping_sharded(self):
        mapper = CPSlotMapper(cp_rank=0, cp_size=2, block_size=4)
        # virtual_block_size = 8, so positions 0-7 form one virtual block
        positions = torch.arange(8)
        block_table = torch.tensor(
            [[10]]
        )  # 1 batch, 1 virtual block -> physical block 10
        batch_indices = torch.zeros(8, dtype=torch.int32)

        slots = mapper.compute_slot_mapping(positions, block_table, batch_indices)
        # rank0 owns positions 0,2,4,6
        # local_offset(0) = 0/2 = 0 -> slot = 10*4 + 0 = 40
        # local_offset(2) = 2/2 = 1 -> slot = 10*4 + 1 = 41
        # local_offset(4) = 4/2 = 2 -> slot = 10*4 + 2 = 42
        # local_offset(6) = 6/2 = 3 -> slot = 10*4 + 3 = 43
        # positions 1,3,5,7 are not owned -> slot = -1
        expected = torch.tensor([40, -1, 41, -1, 42, -1, 43, -1], dtype=torch.long)
        torch.testing.assert_close(slots, expected)

    def test_cross_request_consistency(self):
        cp_size = 4
        block_size = 16
        for r in range(cp_size):
            mapper = CPSlotMapper(r, cp_size, block_size)
            vbs = mapper.virtual_block_size
            for pos in range(256):
                expected = (pos % vbs) % cp_size == r
                positions = torch.tensor([pos])
                self.assertEqual(
                    mapper.is_owned(positions).item(), expected, f"rank={r} pos={pos}"
                )


if __name__ == "__main__":
    unittest.main()
