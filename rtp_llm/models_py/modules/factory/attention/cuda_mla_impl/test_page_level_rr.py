"""Unit tests for page-level round-robin CP KV cache.

Tests cover:
1. CPSlotMapper page-level mode (pybind)
2. filter_topk_for_sharded_cache (Triton kernel)
3. PageRoundRobinSparseMlaFp8CPOp plan() logic (pure Python)
"""

import unittest

import torch

# ---------------------------------------------------------------------------
# 1. CPSlotMapper page-level tests (pybind)
# ---------------------------------------------------------------------------


class TestCPSlotMapperPageLevel(unittest.TestCase):
    """Test CPSlotMapper with sharded=True (page-level RR)."""

    def _make_mapper(self, cp_rank, cp_size, block_size):
        from rtp_llm.ops.compute_ops import CPSlotMapper

        return CPSlotMapper(cp_rank=cp_rank, cp_size=cp_size, block_size=block_size)

    def test_is_sharded(self):
        m = self._make_mapper(0, 2, 4)
        self.assertTrue(m.is_sharded)

    def test_ownership_block_granularity(self):
        """Block b belongs to rank (b % cp_size)."""
        block_size = 4
        cp_size = 2
        m0 = self._make_mapper(0, cp_size, block_size)
        m1 = self._make_mapper(1, cp_size, block_size)

        # positions 0-3 = block 0 -> rank 0
        # positions 4-7 = block 1 -> rank 1
        # positions 8-11 = block 2 -> rank 0
        # positions 12-15 = block 3 -> rank 1
        positions = torch.arange(16)
        mask0 = m0.is_owned(positions)
        mask1 = m1.is_owned(positions)

        expected0 = torch.tensor([True] * 4 + [False] * 4 + [True] * 4 + [False] * 4)
        expected1 = torch.tensor([False] * 4 + [True] * 4 + [False] * 4 + [True] * 4)
        torch.testing.assert_close(mask0, expected0)
        torch.testing.assert_close(mask1, expected1)

    def test_every_position_owned_exactly_once(self):
        cp_size = 3
        block_size = 8
        total = 96
        owner = torch.full((total,), -1, dtype=torch.long)
        for r in range(cp_size):
            m = self._make_mapper(r, cp_size, block_size)
            positions = torch.arange(total)
            mask = m.is_owned(positions)
            for pos in torch.where(mask)[0]:
                self.assertEqual(
                    owner[pos].item(),
                    -1,
                    f"pos {pos.item()} owned by rank {owner[pos].item()} and {r}",
                )
                owner[pos] = r
        for i in range(total):
            self.assertGreaterEqual(owner[i].item(), 0, f"pos {i} not owned")

    def test_local_block_offset(self):
        """local_block_offset = position % block_size for page-level."""
        m = self._make_mapper(0, 2, 4)
        # rank 0 owns block 0 (pos 0-3) and block 2 (pos 8-11)
        positions = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
        offsets = m.local_block_offset(positions)
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        torch.testing.assert_close(offsets, expected)

    def test_local_block_count(self):
        """Count blocks where block_idx % cp_size == cp_rank."""
        m0 = self._make_mapper(0, 2, 4)
        m1 = self._make_mapper(1, 2, 4)
        # seq_len=16 => 4 blocks: rank0 gets blocks 0,2 -> 2 blocks
        self.assertEqual(m0.local_block_count(16), 2)
        self.assertEqual(m1.local_block_count(16), 2)
        # seq_len=12 => 3 blocks: rank0 gets blocks 0,2 -> 2; rank1 gets block 1 -> 1
        self.assertEqual(m0.local_block_count(12), 2)
        self.assertEqual(m1.local_block_count(12), 1)
        # seq_len=4 => 1 block: rank0 gets 1, rank1 gets 0
        self.assertEqual(m0.local_block_count(4), 1)
        self.assertEqual(m1.local_block_count(4), 0)

    def test_compute_slot_mapping_page_level(self):
        """Verify slot mapping for page-level RR."""
        m = self._make_mapper(0, 2, 4)
        # positions 0-7: block 0 (rank 0), block 1 (rank 1)
        positions = torch.arange(8)
        # block_table has virtual blocks: virtual_block_size = block_size * cp_size = 8
        # virtual block 0 -> physical block 10
        block_table = torch.tensor([[10]])
        batch_indices = torch.zeros(8, dtype=torch.int32)
        slots = m.compute_slot_mapping(positions, block_table, batch_indices)

        # rank 0 owns pos 0-3 (block 0):
        #   global_block_idx = 0, vblock_idx = 0 // 2 = 0, local_offset = pos % 4
        #   slot = block_table[0][0] * 4 + local_offset = 10*4 + {0,1,2,3} = {40,41,42,43}
        # rank 0 does NOT own pos 4-7 (block 1 -> rank 1) -> -1
        expected = torch.tensor([40, 41, 42, 43, -1, -1, -1, -1], dtype=torch.long)
        torch.testing.assert_close(slots, expected)

    def test_compute_slot_mapping_page_level_rank1(self):
        """Verify slot mapping for rank 1 in page-level RR."""
        m = self._make_mapper(1, 2, 4)
        positions = torch.arange(8)
        block_table = torch.tensor([[10]])
        batch_indices = torch.zeros(8, dtype=torch.int32)
        slots = m.compute_slot_mapping(positions, block_table, batch_indices)

        # rank 1 owns pos 4-7 (block 1):
        #   global_block_idx = 1, vblock_idx = 1 // 2 = 0, local_offset = pos % 4
        #   slot = block_table[0][0] * 4 + local_offset = 10*4 + {0,1,2,3} = {40,41,42,43}
        # rank 1 does NOT own pos 0-3 (block 0 -> rank 0) -> -1
        expected = torch.tensor([-1, -1, -1, -1, 40, 41, 42, 43], dtype=torch.long)
        torch.testing.assert_close(slots, expected)


# ---------------------------------------------------------------------------
# 2. filter_topk_for_sharded_cache tests (Triton kernel)
# ---------------------------------------------------------------------------


class TestFilterTopkPageLevel(unittest.TestCase):
    """Test the Triton kernel for page-level RR topk filtering."""

    def _run_filter(
        self, req_id, block_table, token_indices, cp_rank, cp_size, block_size
    ):
        from rtp_llm.models_py.triton_kernels.sparse_mla.filter_topk_for_sharded_cache import (
            triton_filter_topk_for_sharded_cache,
        )

        BLOCK_N = token_indices.shape[1]
        return triton_filter_topk_for_sharded_cache(
            req_id.cuda(),
            block_table.cuda(),
            token_indices.cuda(),
            cp_rank,
            cp_size,
            block_size,
            BLOCK_N=BLOCK_N,
        ).cpu()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_page_level_filter_rank0(self):
        """Rank 0 should own blocks 0, 2, ... (block_idx % 2 == 0)."""
        block_size = 4
        cp_size = 2
        # 1 request, 2 virtual blocks (4 physical blocks in total)
        # block_table: virtual block 0 -> phys 10, virtual block 1 -> phys 20
        block_table = torch.tensor([[10, 20]], dtype=torch.int32)
        # 1 token, 128 topk positions (padded with -1)
        topk = 128
        token_indices = torch.full((1, topk), -1, dtype=torch.int32)
        # Set some positions:
        # pos 0 -> block 0 -> rank 0 (owned)
        # pos 5 -> block 1 -> rank 1 (not owned)
        # pos 8 -> block 2 -> rank 0 (owned)
        # pos 13 -> block 3 -> rank 1 (not owned)
        token_indices[0, 0] = 0
        token_indices[0, 1] = 5
        token_indices[0, 2] = 8
        token_indices[0, 3] = 13

        req_id = torch.tensor([0], dtype=torch.int32)
        out = self._run_filter(
            req_id,
            block_table,
            token_indices,
            cp_rank=0,
            cp_size=cp_size,
            block_size=block_size,
        )

        # pos 0: block_idx=0, rank=0 (owned), vblock_idx=0, local_offset=0
        #   -> block_table[0][0] * 4 + 0 = 40
        self.assertEqual(out[0, 0].item(), 40)
        # pos 5: block_idx=1, rank=1 (not owned) -> -1
        self.assertEqual(out[0, 1].item(), -1)
        # pos 8: block_idx=2, rank=0 (owned), vblock_idx=1, local_offset=0
        #   -> block_table[0][1] * 4 + 0 = 80
        self.assertEqual(out[0, 2].item(), 80)
        # pos 13: block_idx=3, rank=1 (not owned) -> -1
        self.assertEqual(out[0, 3].item(), -1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_page_level_filter_rank1(self):
        """Rank 1 should own blocks 1, 3, ... (block_idx % 2 == 1)."""
        block_size = 4
        cp_size = 2
        block_table = torch.tensor([[10, 20]], dtype=torch.int32)
        topk = 128
        token_indices = torch.full((1, topk), -1, dtype=torch.int32)
        token_indices[0, 0] = 0  # block 0 -> rank 0
        token_indices[0, 1] = 5  # block 1 -> rank 1
        token_indices[0, 2] = 8  # block 2 -> rank 0
        token_indices[0, 3] = 13  # block 3 -> rank 1

        req_id = torch.tensor([0], dtype=torch.int32)
        out = self._run_filter(
            req_id,
            block_table,
            token_indices,
            cp_rank=1,
            cp_size=cp_size,
            block_size=block_size,
        )

        # pos 0: not owned -> -1
        self.assertEqual(out[0, 0].item(), -1)
        # pos 5: block_idx=1, rank=1 (owned), vblock_idx=0, local_offset=1
        #   -> block_table[0][0] * 4 + 1 = 41
        self.assertEqual(out[0, 1].item(), 41)
        # pos 8: not owned -> -1
        self.assertEqual(out[0, 2].item(), -1)
        # pos 13: block_idx=3, rank=1 (owned), vblock_idx=1, local_offset=1
        #   -> block_table[0][1] * 4 + 1 = 81
        self.assertEqual(out[0, 3].item(), 81)


# ---------------------------------------------------------------------------
# 3. PageRoundRobinSparseMlaFp8CPOp plan() logic (pure Python)
# ---------------------------------------------------------------------------


class TestPageRoundRobinPlan(unittest.TestCase):
    """Test the plan() logic for PageRoundRobinSparseMlaFp8CPOp.

    We test the pure computation logic: block ownership, local token mapping,
    and AG restore indices.
    """

    def _compute_plan_indices(
        self,
        cp_rank,
        cp_size,
        page_size,
        seq_lengths,
        input_lengths,
        prefix_lengths=None,
    ):
        """Reproduce the core plan() logic from PageRoundRobinSparseMlaFp8CPOp."""
        device = "cpu"
        if prefix_lengths is None:
            prefix_lengths = [0] * len(seq_lengths)

        batch_size = len(seq_lengths)
        total_q_positions = []
        for b in range(batch_size):
            kv_len = input_lengths[b] + prefix_lengths[b]
            for j in range(input_lengths[b]):
                total_q_positions.append(prefix_lengths[b] + j)

        total_q = len(total_q_positions)

        # Build kv positions for each request
        kv_positions_per_req = []
        for b in range(batch_size):
            kv_len = input_lengths[b] + prefix_lengths[b]
            kv_positions_per_req.append(list(range(kv_len)))

        # Compute local KV tokens per request
        local_kv_per_req = []
        for b in range(batch_size):
            kv_len = input_lengths[b] + prefix_lengths[b]
            local = []
            for pos in range(kv_len):
                block_idx = pos // page_size
                if block_idx % cp_size == cp_rank:
                    local.append(pos)
            local_kv_per_req.append(local)

        # Compute AG restore indices
        # After all-gather, the layout is: [rank0_local | rank1_local | ...]
        # We need indices to restore global order
        total_local_kv_per_req = [len(lk) for lk in local_kv_per_req]

        restore_indices_per_req = []
        for b in range(batch_size):
            kv_len = input_lengths[b] + prefix_lengths[b]
            total_blocks = (kv_len + page_size - 1) // page_size
            restore = []
            for pos in range(kv_len):
                block_idx = pos // page_size
                source_rank = block_idx % cp_size
                local_block_idx = block_idx // cp_size
                token_in_block = pos % page_size
                # offset in gathered buffer = source_rank * total_local_kv + local_block_idx * page_size + token_in_block
                local_offset = local_block_idx * page_size + token_in_block
                source_offset = source_rank * total_local_kv_per_req[b] + local_offset
                restore.append(source_offset)
            restore_indices_per_req.append(restore)

        return {
            "local_kv_per_req": local_kv_per_req,
            "total_local_kv_per_req": total_local_kv_per_req,
            "restore_indices_per_req": restore_indices_per_req,
        }

    def test_single_request_even_blocks(self):
        """2 ranks, 4 blocks (evenly split)."""
        page_size = 4
        cp_size = 2
        kv_len = 16  # 4 blocks
        input_len = 16
        prefix_len = 0

        for rank in range(cp_size):
            result = self._compute_plan_indices(
                rank,
                cp_size,
                page_size,
                seq_lengths=[kv_len],
                input_lengths=[input_len],
                prefix_lengths=[prefix_len],
            )
            # Each rank gets 2 blocks = 8 tokens
            self.assertEqual(result["total_local_kv_per_req"][0], 8)

        # Rank 0 gets blocks 0, 2 -> positions [0,1,2,3, 8,9,10,11]
        r0 = self._compute_plan_indices(
            0, cp_size, page_size, [kv_len], [input_len], [prefix_len]
        )
        self.assertEqual(r0["local_kv_per_req"][0], [0, 1, 2, 3, 8, 9, 10, 11])

        # Rank 1 gets blocks 1, 3 -> positions [4,5,6,7, 12,13,14,15]
        r1 = self._compute_plan_indices(
            1, cp_size, page_size, [kv_len], [input_len], [prefix_len]
        )
        self.assertEqual(r1["local_kv_per_req"][0], [4, 5, 6, 7, 12, 13, 14, 15])

    def test_restore_indices_roundtrip(self):
        """Verify that restore indices correctly reconstruct global order."""
        page_size = 4
        cp_size = 2
        kv_len = 16
        input_len = 16

        # Simulate all-gather
        # Rank 0 local: positions [0,1,2,3, 8,9,10,11] -> values [0,1,2,3, 8,9,10,11]
        # Rank 1 local: positions [4,5,6,7, 12,13,14,15] -> values [4,5,6,7, 12,13,14,15]
        # Gathered buffer = [rank0_local | rank1_local]
        #                 = [0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15]
        gathered = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]

        # Get restore indices from rank 0's perspective (any rank gives same result)
        r0 = self._compute_plan_indices(0, cp_size, page_size, [kv_len], [input_len])
        restore = r0["restore_indices_per_req"][0]

        # Apply restore indices
        restored = [gathered[i] for i in restore]
        expected = list(range(kv_len))
        self.assertEqual(restored, expected)

    def test_uneven_blocks(self):
        """3 ranks, 5 blocks (uneven split)."""
        page_size = 4
        cp_size = 3
        kv_len = 20  # 5 blocks
        input_len = 20

        # Block 0 -> rank 0, Block 1 -> rank 1, Block 2 -> rank 2
        # Block 3 -> rank 0, Block 4 -> rank 1
        r0 = self._compute_plan_indices(0, cp_size, page_size, [kv_len], [input_len])
        r1 = self._compute_plan_indices(1, cp_size, page_size, [kv_len], [input_len])
        r2 = self._compute_plan_indices(2, cp_size, page_size, [kv_len], [input_len])

        self.assertEqual(r0["total_local_kv_per_req"][0], 8)  # blocks 0,3
        self.assertEqual(r1["total_local_kv_per_req"][0], 8)  # blocks 1,4
        self.assertEqual(r2["total_local_kv_per_req"][0], 4)  # block 2 only

        # Verify restore roundtrip
        local_r0 = r0["local_kv_per_req"][0]  # [0,1,2,3, 12,13,14,15]
        local_r1 = r1["local_kv_per_req"][0]  # [4,5,6,7, 16,17,18,19]
        local_r2 = r2["local_kv_per_req"][0]  # [8,9,10,11]

        # After padding r2 to 8 tokens: [8,9,10,11, pad,pad,pad,pad]
        # But for restore, we only care about the actual tokens
        # gathered = [local_r0 | local_r1 | local_r2]
        gathered = local_r0 + local_r1 + local_r2
        restore = r0["restore_indices_per_req"][0]
        restored = [gathered[i] for i in restore]
        self.assertEqual(restored, list(range(kv_len)))

    def test_partial_last_block(self):
        """Test with a partial last block."""
        page_size = 4
        cp_size = 2
        kv_len = 10  # 3 blocks: 0(full), 1(full), 2(partial: 2 tokens)
        input_len = 10

        r0 = self._compute_plan_indices(0, cp_size, page_size, [kv_len], [input_len])
        r1 = self._compute_plan_indices(1, cp_size, page_size, [kv_len], [input_len])

        # Block 0 -> rank 0: [0,1,2,3]
        # Block 1 -> rank 1: [4,5,6,7]
        # Block 2 -> rank 0: [8,9]
        self.assertEqual(r0["local_kv_per_req"][0], [0, 1, 2, 3, 8, 9])
        self.assertEqual(r1["local_kv_per_req"][0], [4, 5, 6, 7])
        self.assertEqual(r0["total_local_kv_per_req"][0], 6)
        self.assertEqual(r1["total_local_kv_per_req"][0], 4)


if __name__ == "__main__":
    unittest.main()
