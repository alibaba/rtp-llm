import os
import sys
import unittest

import torch

from rtp_llm.cpp.models.context_parallel.test import (
    libth_context_parallel_py_wrapper_test as cp_test,
)


class TestContextParallelLoadBalanceSplit(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(
            os.environ.get("TEST_SRCDIR", ".")
            + "/rtp_llm/rtp_llm/cpp/models/context_parallel/test"
        )
        self.fake_input_tokens = [
            45231,
            12089,
            58934,
            2341,
            61234,
            9876,
            34567,
            51234,
            8901,
            43210,
            15678,
            29345,
            48901,
            5234,
            62345,
            18765,
            38901,
            54321,
            7890,
            41234,
            26789,
            52345,
            13456,
            39012,
            55678,
            4567,
            47890,
            21345,
            59012,
            11234,
            35678,
            49012,
        ]

    def test_basic_split(self):
        """Test basic load balance split with simple parameters"""
        # Test with 16 tokens, 2 ranks, chunk size 8
        total_tokens = self.fake_input_tokens[:16]
        input_tokens = []
        shuffle_indices = []
        cp_rank = 0
        cp_size = 2
        cp_chunk_size = 8
        cp_padding_size = 0
        expect_shuffle_indices = [0, 1, 2, 3, 12, 13, 14, 15]
        expect_input_tokens = [total_tokens[i] for i in expect_shuffle_indices]

        result, input_tokens, shuffle_indices = (
            cp_test.context_parallel_load_balance_split(
                total_tokens,
                input_tokens,
                shuffle_indices,
                cp_rank,
                cp_size,
                cp_chunk_size,
                cp_padding_size,
            )
        )
        self.assertTrue(result)
        self.assertEqual(len(input_tokens), cp_chunk_size)
        self.assertEqual(len(shuffle_indices), cp_chunk_size)
        self.assertEqual(input_tokens, expect_input_tokens)
        self.assertEqual(shuffle_indices, expect_shuffle_indices)

    def test_with_padding(self):
        """Test load balance split with padding"""
        # Test with padding
        total_tokens = self.fake_input_tokens[:14]  # 14 tokens
        input_tokens = []
        shuffle_indices = []
        cp_rank = 0
        cp_size = 2
        cp_chunk_size = 8
        cp_padding_size = 2
        expect_shuffle_indices = [0, 1, 2, 3, 12, 13, 14, 15]
        expect_input_tokens = [total_tokens[i] for i in expect_shuffle_indices[0:6]] + [
            0,
            0,
        ]

        result, input_tokens, shuffle_indices = (
            cp_test.context_parallel_load_balance_split(
                total_tokens,
                input_tokens,
                shuffle_indices,
                cp_rank,
                cp_size,
                cp_chunk_size,
                cp_padding_size,
            )
        )
        self.assertTrue(result)
        self.assertEqual(len(input_tokens), cp_chunk_size)
        self.assertEqual(len(shuffle_indices), cp_chunk_size)
        self.assertEqual(input_tokens, expect_input_tokens)
        self.assertEqual(shuffle_indices, expect_shuffle_indices)

    def test_multiple_ranks(self):
        """Test load balance split across multiple ranks"""
        total_tokens = self.fake_input_tokens[:32]
        cp_size = 4
        cp_chunk_size = 8
        cp_padding_size = 0

        expect_shuffle_indices = [
            [0, 1, 2, 3, 28, 29, 30, 31],
            [4, 5, 6, 7, 24, 25, 26, 27],
            [8, 9, 10, 11, 20, 21, 22, 23],
            [12, 13, 14, 15, 16, 17, 18, 19],
        ]
        expect_input_tokens = [
            [total_tokens[i] for i in indices] for indices in expect_shuffle_indices
        ]

        # Test each rank
        for cp_rank in range(cp_size):
            input_tokens = []
            shuffle_indices = []
            result, input_tokens, shuffle_indices = (
                cp_test.context_parallel_load_balance_split(
                    total_tokens,
                    input_tokens,
                    shuffle_indices,
                    cp_rank,
                    cp_size,
                    cp_chunk_size,
                    cp_padding_size,
                )
            )
            self.assertTrue(result)
            self.assertEqual(input_tokens, expect_input_tokens[cp_rank])
            self.assertEqual(shuffle_indices, expect_shuffle_indices[cp_rank])


class TestGenerateQKVRestoreIndices(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(
            os.environ.get("TEST_SRCDIR", ".")
            + "/rtp_llm/rtp_llm/cpp/models/context_parallel/test"
        )

    def test_basic_restore_indices(self):
        """Test basic QKV restore indices generation"""
        # Create chunk lengths for 2 ranks
        prefill_cp_chunk_lengths = torch.tensor([8, 8], dtype=torch.int32)
        cp_size = 2

        expect_restore_indices = torch.tensor(
            [
                0,
                1,
                2,
                3,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                12,
                13,
                14,
                15,
            ],
            dtype=torch.int32,
        )

        restore_indices = cp_test.generate_qkv_restore_indices(
            prefill_cp_chunk_lengths, cp_size
        )
        self.assertTrue(torch.equal(restore_indices, expect_restore_indices))

    def test_different_chunk_sizes(self):
        """Test with different chunk sizes across ranks"""
        # Create uneven chunk lengths
        prefill_cp_chunk_lengths = torch.tensor([2, 4, 8, 2], dtype=torch.int32)
        cp_size = 4

        restore_indices = cp_test.generate_qkv_restore_indices(
            prefill_cp_chunk_lengths, cp_size
        )
        # rank0:[0,7,8,9,22,23,24,25,26,27,52,53,54,55,56,63]
        # rank1:[1,6,10,11,20,21,28,29,30,31,48,49,50,51,57,62]
        # rank2:[2,5,12,13,18,19,32,33,34,35,44,45,46,47,58,61]
        # rank3:[3,4,14,15,16,17,36,37,38,39,40,41,42,43,59,60]
        # gather all rank position_ids and then argsort
        expect_restore_indices = torch.tensor(
            [
                0,
                16,
                32,
                48,
                49,
                33,
                17,
                1,
                2,
                3,
                18,
                19,
                34,
                35,
                50,
                51,
                52,
                53,
                36,
                37,
                20,
                21,
                4,
                5,
                6,
                7,
                8,
                9,
                22,
                23,
                24,
                25,
                38,
                39,
                40,
                41,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                42,
                43,
                44,
                45,
                26,
                27,
                28,
                29,
                10,
                11,
                12,
                13,
                14,
                30,
                46,
                62,
                63,
                47,
                31,
                15,
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(restore_indices, expect_restore_indices))


class TestGenerateQKVPaddingMask(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        sys.path.append(
            os.environ.get("TEST_SRCDIR", ".")
            + "/rtp_llm/rtp_llm/cpp/models/context_parallel/test"
        )

    def test_with_padding(self):
        """Test padding mask with actual padding"""
        # Some ranks have padding
        prefill_cp_chunk_lengths = torch.tensor([4, 8, 2, 6], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([0, 2, 0, 2], dtype=torch.int32)
        cp_size = 2
        padding_mask = cp_test.generate_qkv_padding_mask(
            prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size
        )
        expect_padding_mask = torch.ones(
            cp_size * torch.sum(prefill_cp_chunk_lengths), dtype=torch.int32
        )
        expect_padding_mask[22:24] = 0
        expect_padding_mask[38:] = 0
        self.assertTrue(torch.equal(expect_padding_mask, padding_mask))

    def test_no_padding(self):
        """Test when there is no padding"""
        prefill_cp_chunk_lengths = torch.tensor([8, 8, 8], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([0, 0, 0], dtype=torch.int32)
        cp_size = 4

        padding_mask = cp_test.generate_qkv_padding_mask(
            prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size
        )

        expect_padding_mask = torch.ones(
            cp_size * torch.sum(prefill_cp_chunk_lengths), dtype=torch.bool
        )
        self.assertTrue(torch.equal(expect_padding_mask, padding_mask))


class TestComputeLocalLastHidden(unittest.TestCase):
    @staticmethod
    def _pad_to(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def _run_case(self, stream_lens, cp_size, hidden_dim=3, device="cpu"):
        padded_lens = [self._pad_to(s, 2 * cp_size) for s in stream_lens]
        chunk_lens = [p // cp_size for p in padded_lens]
        padding_lens = [p - s for p, s in zip(padded_lens, stream_lens)]

        chunk_lengths = torch.tensor(chunk_lens, dtype=torch.int32)
        padding_lengths = torch.tensor(padding_lens, dtype=torch.int32)
        restore_indice = cp_test.generate_qkv_restore_indices(chunk_lengths, cp_size)
        padding_mask = cp_test.generate_qkv_padding_mask(
            chunk_lengths, padding_lengths, cp_size
        )

        total_token_size = int(chunk_lengths.sum().item())
        padded_total = cp_size * total_token_size
        base = torch.arange(1, padded_total + 1, dtype=torch.float32).unsqueeze(1)
        scale = torch.tensor([[1.0, 10.0, 100.0][:hidden_dim]], dtype=torch.float32)
        original_hidden = base * scale

        gathered_layout = torch.zeros((padded_total, hidden_dim), dtype=torch.float32)
        gathered_layout.index_copy_(0, restore_indice.to(torch.long), original_hidden)
        rank_chunks = gathered_layout.reshape(cp_size, total_token_size, hidden_dim).to(
            device
        )
        restore_indice = restore_indice.to(device)
        padding_mask = padding_mask.to(device)

        valid_indices = torch.nonzero(padding_mask.cpu()).squeeze(-1)
        restored = original_hidden.index_select(0, valid_indices.to(torch.long))
        ends = []
        offset = 0
        for stream_len in stream_lens:
            offset += stream_len
            ends.append(offset - 1)
        lm_output_indexes = torch.tensor(ends, dtype=torch.int32)
        expected = restored.index_select(0, lm_output_indexes.to(torch.long))

        actual = torch.zeros_like(expected)
        contributors = torch.zeros(len(ends), dtype=torch.long)
        for rank in range(cp_size):
            local = cp_test.compute_local_last_hidden(
                rank_chunks[rank].contiguous(),
                restore_indice,
                padding_mask,
                lm_output_indexes,
                rank,
                cp_size,
            )
            actual += local
            contributors += (local.abs().sum(dim=1) > 0).to(torch.long)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(
            torch.equal(contributors, torch.ones(len(ends), dtype=torch.long))
        )

    def test_single_stream_with_padding(self):
        self._run_case(stream_lens=[14], cp_size=2)

    def test_multi_stream_cp4_with_padding(self):
        self._run_case(stream_lens=[10, 20, 7], cp_size=4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cpu_indexes_with_cuda_hidden(self):
        self._run_case(stream_lens=[14], cp_size=2, device="cuda")


if __name__ == "__main__":
    unittest.main()
