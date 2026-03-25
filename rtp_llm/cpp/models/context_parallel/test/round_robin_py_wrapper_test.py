import os
import sys
import unittest

import torch

from rtp_llm.cpp.models.context_parallel.test import (
    libth_round_robin_py_wrapper_test as rr_test,
)


class TestRoundRobinPlan(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.fake_input_tokens = list(range(100, 132))  # [100, 101, ..., 131]

    def test_basic_split_cp2(self):
        """Test round-robin split with cp_size=2, no padding."""
        total_tokens = self.fake_input_tokens[:8]  # [100..107]
        cp_size = 2
        cp_chunk_size = 4
        cp_padding_size = 0

        # rank 0: positions 0,2,4,6 → tokens 100,102,104,106
        result0, tokens0, indices0 = rr_test.round_robin_plan(
            total_tokens, [], [], 0, cp_size, cp_chunk_size, cp_padding_size
        )
        self.assertTrue(result0)
        self.assertEqual(indices0, [0, 2, 4, 6])
        self.assertEqual(tokens0, [100, 102, 104, 106])

        # rank 1: positions 1,3,5,7 → tokens 101,103,105,107
        result1, tokens1, indices1 = rr_test.round_robin_plan(
            total_tokens, [], [], 1, cp_size, cp_chunk_size, cp_padding_size
        )
        self.assertTrue(result1)
        self.assertEqual(indices1, [1, 3, 5, 7])
        self.assertEqual(tokens1, [101, 103, 105, 107])

    def test_with_padding(self):
        """Test round-robin split with padding."""
        total_tokens = self.fake_input_tokens[:6]  # 6 tokens, need 8 (cp_size=2, align to 2)
        cp_size = 2
        cp_chunk_size = 4
        cp_padding_size = 2  # pad to 8

        # rank 0: positions 0,2,4,6 → tokens 100,102,104,0(pad)
        result0, tokens0, indices0 = rr_test.round_robin_plan(
            total_tokens, [], [], 0, cp_size, cp_chunk_size, cp_padding_size
        )
        self.assertTrue(result0)
        self.assertEqual(indices0, [0, 2, 4, 6])
        self.assertEqual(tokens0, [100, 102, 104, 0])

        # rank 1: positions 1,3,5,7 → tokens 101,103,105,0(pad)
        result1, tokens1, indices1 = rr_test.round_robin_plan(
            total_tokens, [], [], 1, cp_size, cp_chunk_size, cp_padding_size
        )
        self.assertTrue(result1)
        self.assertEqual(indices1, [1, 3, 5, 7])
        self.assertEqual(tokens1, [101, 103, 105, 0])

    def test_multiple_ranks_cp4(self):
        """Test round-robin split with cp_size=4."""
        total_tokens = self.fake_input_tokens[:16]  # [100..115]
        cp_size = 4
        cp_chunk_size = 4
        cp_padding_size = 0

        expected_indices = [
            [0, 4, 8, 12],   # rank 0
            [1, 5, 9, 13],   # rank 1
            [2, 6, 10, 14],  # rank 2
            [3, 7, 11, 15],  # rank 3
        ]
        expected_tokens = [
            [total_tokens[i] for i in idx] for idx in expected_indices
        ]

        for cp_rank in range(cp_size):
            result, tokens, indices = rr_test.round_robin_plan(
                total_tokens, [], [], cp_rank, cp_size, cp_chunk_size, cp_padding_size
            )
            self.assertTrue(result)
            self.assertEqual(indices, expected_indices[cp_rank],
                             f"rank {cp_rank} indices mismatch")
            self.assertEqual(tokens, expected_tokens[cp_rank],
                             f"rank {cp_rank} tokens mismatch")

    def test_all_ranks_cover_all_tokens(self):
        """Verify all ranks together cover all token positions exactly once."""
        total_tokens = self.fake_input_tokens[:12]
        cp_size = 3
        cp_chunk_size = 4
        cp_padding_size = 0

        all_indices = []
        all_tokens = []
        for cp_rank in range(cp_size):
            result, tokens, indices = rr_test.round_robin_plan(
                total_tokens, [], [], cp_rank, cp_size, cp_chunk_size, cp_padding_size
            )
            self.assertTrue(result)
            all_indices.extend(indices)
            all_tokens.extend(tokens)

        self.assertEqual(sorted(all_indices), list(range(12)))
        self.assertEqual(sorted(all_tokens), sorted(total_tokens))


class TestRoundRobinRestoreIndices(unittest.TestCase):
    def test_basic_cp2(self):
        """Test restore indices for cp_size=2, single stream."""
        # chunk_length=4 means each rank has 4 tokens, total 8 tokens
        prefill_cp_chunk_lengths = torch.tensor([4], dtype=torch.int32)
        cp_size = 2

        restore_indices = rr_test.round_robin_generate_qkv_restore_indices(
            prefill_cp_chunk_lengths, cp_size
        )

        # After all-gather: [rank0: pos 0,2,4,6 | rank1: pos 1,3,5,7]
        # = flat indices [0,1,2,3 | 4,5,6,7]
        # restore_indices[global_pos] = flat_idx
        # global_pos 0 → rank0, local_j=0 → flat 0
        # global_pos 1 → rank1, local_j=0 → flat 4
        # global_pos 2 → rank0, local_j=1 → flat 1
        # global_pos 3 → rank1, local_j=1 → flat 5
        # global_pos 4 → rank0, local_j=2 → flat 2
        # global_pos 5 → rank1, local_j=2 → flat 5  → wait, 6? no...
        # rank1 offset = 1*4 = 4
        # global_pos 5 → rank1, local_j=2 → flat 4+2 = 6
        # global_pos 6 → rank0, local_j=3 → flat 3
        # global_pos 7 → rank1, local_j=3 → flat 7
        expected = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32)
        self.assertTrue(torch.equal(restore_indices, expected),
                        f"Expected {expected.tolist()}, got {restore_indices.tolist()}")

    def test_cp4_single_stream(self):
        """Test restore indices for cp_size=4, single stream."""
        prefill_cp_chunk_lengths = torch.tensor([3], dtype=torch.int32)
        cp_size = 4
        # total_token_size = 3, each rank has 3 tokens
        # total global positions = 12
        # rank r, local j → global pos = r + j*4
        # flat_idx = r * 3 + j
        # restore[global_pos] = flat_idx

        restore_indices = rr_test.round_robin_generate_qkv_restore_indices(
            prefill_cp_chunk_lengths, cp_size
        )

        expected = []
        total_token_size = 3
        for pos in range(12):
            rank = pos % cp_size
            j = pos // cp_size
            expected.append(rank * total_token_size + j)
        expected = torch.tensor(expected, dtype=torch.int32)
        self.assertTrue(torch.equal(restore_indices, expected),
                        f"Expected {expected.tolist()}, got {restore_indices.tolist()}")

    def test_multi_stream(self):
        """Test restore indices with multiple prefill streams."""
        prefill_cp_chunk_lengths = torch.tensor([2, 4], dtype=torch.int32)
        cp_size = 2
        # total_token_size = 6
        # Stream 0: chunk_length=2, prefill_qkv_len=4
        #   rank0: global 0,2 → flat 0*6+0=0, 0*6+1=1
        #   rank1: global 1,3 → flat 1*6+0=6, 1*6+1=7
        # Stream 1: chunk_length=4, prefill_qkv_len=8, seq_offset=4
        #   rank0: global 4,6,8,10 → flat 0*6+2=2, 0*6+3=3, 0*6+4=4, 0*6+5=5
        #   rank1: global 5,7,9,11 → flat 1*6+2=8, 1*6+3=9, 1*6+4=10, 1*6+5=11
        # restore[0]=0, restore[1]=6, restore[2]=1, restore[3]=7
        # restore[4]=2, restore[5]=8, restore[6]=3, restore[7]=9
        # restore[8]=4, restore[9]=10, restore[10]=5, restore[11]=11

        restore_indices = rr_test.round_robin_generate_qkv_restore_indices(
            prefill_cp_chunk_lengths, cp_size
        )
        expected = torch.tensor([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11], dtype=torch.int32)
        self.assertTrue(torch.equal(restore_indices, expected),
                        f"Expected {expected.tolist()}, got {restore_indices.tolist()}")

    def test_restore_is_valid_permutation(self):
        """Verify restore indices form a valid permutation."""
        prefill_cp_chunk_lengths = torch.tensor([4, 6], dtype=torch.int32)
        cp_size = 2
        total = cp_size * torch.sum(prefill_cp_chunk_lengths).item()

        restore_indices = rr_test.round_robin_generate_qkv_restore_indices(
            prefill_cp_chunk_lengths, cp_size
        )
        self.assertEqual(restore_indices.numel(), total)
        self.assertEqual(sorted(restore_indices.tolist()), list(range(total)))


class TestRoundRobinPaddingMask(unittest.TestCase):
    def test_no_padding(self):
        """Test padding mask with no padding."""
        prefill_cp_chunk_lengths = torch.tensor([4, 6], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([0, 0], dtype=torch.int32)
        cp_size = 2

        mask = rr_test.round_robin_generate_qkv_padding_mask(
            prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size
        )
        total = cp_size * torch.sum(prefill_cp_chunk_lengths).item()
        expected = torch.ones(total, dtype=torch.int32)
        self.assertTrue(torch.equal(mask, expected))

    def test_with_padding(self):
        """Test padding mask with some padding."""
        prefill_cp_chunk_lengths = torch.tensor([4], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([2], dtype=torch.int32)
        cp_size = 2
        # padded_length = 4*2 = 8, valid = 6, padding = 2
        # mask = [1,1,1,1,1,1,0,0]

        mask = rr_test.round_robin_generate_qkv_padding_mask(
            prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size
        )
        expected = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int32)
        self.assertTrue(torch.equal(mask, expected),
                        f"Expected {expected.tolist()}, got {mask.tolist()}")

    def test_multi_stream_padding(self):
        """Test padding mask with multiple streams, some with padding."""
        prefill_cp_chunk_lengths = torch.tensor([3, 4], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([0, 2], dtype=torch.int32)
        cp_size = 2
        # Stream 0: padded=6, valid=6, pad=0 → [1,1,1,1,1,1]
        # Stream 1: padded=8, valid=6, pad=2 → [1,1,1,1,1,1,0,0]

        mask = rr_test.round_robin_generate_qkv_padding_mask(
            prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size
        )
        expected = torch.tensor(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=torch.int32
        )
        self.assertTrue(torch.equal(mask, expected),
                        f"Expected {expected.tolist()}, got {mask.tolist()}")


if __name__ == "__main__":
    unittest.main()
