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


class TestHandleInputsWithHidden(unittest.TestCase):
    def test_hidden_states_split_with_input_tokens(self):
        total_tokens = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
        input_lengths = torch.tensor([6], dtype=torch.int32)
        sequence_lengths = torch.empty((0,), dtype=torch.int32)
        hidden_states = torch.tensor(
            [
                [0.0, 0.5],
                [1.0, 1.5],
                [2.0, 2.5],
                [3.0, 3.5],
                [4.0, 4.5],
                [5.0, 5.5],
            ],
            dtype=torch.float32,
        )

        tokens0, lengths0, hidden0, shuffle0 = cp_test.handle_inputs_with_hidden(
            total_tokens, input_lengths, sequence_lengths, hidden_states, 0, 2
        )
        self.assertTrue(
            torch.equal(tokens0, torch.tensor([10, 11, 0, 0], dtype=torch.int32))
        )
        self.assertTrue(torch.equal(lengths0, torch.tensor([4], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(shuffle0, torch.tensor([0, 1, 6, 7], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                hidden0,
                torch.tensor(
                    [[0.0, 0.5], [1.0, 1.5], [0.0, 0.0], [0.0, 0.0]],
                    dtype=torch.float32,
                ),
            )
        )

        tokens1, lengths1, hidden1, shuffle1 = cp_test.handle_inputs_with_hidden(
            total_tokens, input_lengths, sequence_lengths, hidden_states, 1, 2
        )
        self.assertTrue(
            torch.equal(tokens1, torch.tensor([12, 13, 14, 15], dtype=torch.int32))
        )
        self.assertTrue(torch.equal(lengths1, torch.tensor([4], dtype=torch.int32)))
        self.assertTrue(
            torch.equal(shuffle1, torch.tensor([2, 3, 4, 5], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                hidden1,
                torch.tensor(
                    [[2.0, 2.5], [3.0, 3.5], [4.0, 4.5], [5.0, 5.5]],
                    dtype=torch.float32,
                ),
            )
        )


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
    """Validate the CP gather-last-hidden path (handleOutputsLastHidden).

    handleOutputsLastHidden = computeLocalLastHidden (per rank, no comm) +
    all-reduce-sum across ranks. Here we drive computeLocalLastHidden for every
    rank in-process and sum the results to simulate the all-reduce, then check it
    reconstructs exactly the rows the legacy full-gather path would index_select.
    """

    @staticmethod
    def _pad_to(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def _run_case(self, stream_lens, cp_size, hidden_dim=3):
        # Per-stream padded length must be divisible by 2 * cp_size (see plan()).
        padded_lens = [self._pad_to(s, 2 * cp_size) for s in stream_lens]
        chunk_lens = [p // cp_size for p in padded_lens]
        padding_lens = [p - s for p, s in zip(padded_lens, stream_lens)]

        chunk_lengths = torch.tensor(chunk_lens, dtype=torch.int32)
        padding_lengths = torch.tensor(padding_lens, dtype=torch.int32)

        restore_indice = cp_test.generate_qkv_restore_indices(chunk_lengths, cp_size)
        padding_mask = cp_test.generate_qkv_padding_mask(
            chunk_lengths, padding_lengths, cp_size
        )

        total_token_size = int(chunk_lengths.sum().item())  # per-rank chunk length
        p_total = cp_size * total_token_size
        self.assertEqual(restore_indice.numel(), p_total)
        self.assertEqual(padding_mask.numel(), p_total)

        # Deterministic, all-nonzero per-position hidden in original padded order.
        # Padding rows are never an lm_output position and never gathered, so any
        # leak of a padding/wrong row would show up as a value mismatch.
        base = torch.arange(1, p_total + 1, dtype=torch.float32).unsqueeze(1)
        scale = torch.tensor([[1.0, 10.0, 100.0][:hidden_dim]], dtype=torch.float32)
        orig_hidden_padded = base * scale  # [p_total, hidden_dim]

        # Lay the original rows out in chunk-concat (flat) order: all_hidden_flat is
        # what the all-gather would produce; rank r owns rows [r*tts, (r+1)*tts).
        all_hidden_flat = torch.zeros((p_total, hidden_dim), dtype=torch.float32)
        all_hidden_flat.index_copy_(
            0, restore_indice.to(torch.long), orig_hidden_padded
        )
        rank_chunks = all_hidden_flat.reshape(cp_size, total_token_size, hidden_dim)

        # Reference restored hidden (original valid-seq order), like handleOutputs.
        valid_indices = torch.nonzero(padding_mask).squeeze(-1)
        restored = orig_hidden_padded.index_select(0, valid_indices.to(torch.long))

        # Pick last valid token of each stream (positions into valid-seq order).
        ends = []
        acc = 0
        for s in stream_lens:
            acc += s
            ends.append(acc - 1)
        lm_output_indexes = torch.tensor(ends, dtype=torch.int32)
        num_lm = lm_output_indexes.numel()

        expected = restored.index_select(0, lm_output_indexes.to(torch.long))

        # Per-rank contribution + simulated all-reduce-sum, plus an ownership count.
        summed = torch.zeros((num_lm, hidden_dim), dtype=torch.float32)
        contributors = torch.zeros(num_lm, dtype=torch.long)
        for r in range(cp_size):
            local_buf = cp_test.compute_local_last_hidden(
                rank_chunks[r].contiguous(),
                restore_indice,
                padding_mask,
                lm_output_indexes,
                r,
                cp_size,
            )
            self.assertEqual(list(local_buf.shape), [num_lm, hidden_dim])
            summed += local_buf
            contributors += (local_buf.abs().sum(dim=1) > 0).to(torch.long)

        # 1) Summing the per-rank buffers reconstructs exactly the gathered rows.
        self.assertTrue(
            torch.equal(summed, expected),
            f"summed != expected\nsummed={summed}\nexpected={expected}",
        )
        # 2) Each output row is contributed by exactly one rank (zigzag bijection).
        self.assertTrue(
            torch.equal(contributors, torch.ones(num_lm, dtype=torch.long)),
            f"each row must have exactly one owner, got {contributors.tolist()}",
        )

    def test_single_stream_no_padding(self):
        # 16 tokens, cp=2 -> 16 % 4 == 0, no padding.
        self._run_case(stream_lens=[16], cp_size=2)

    def test_single_stream_with_padding(self):
        # 14 tokens, cp=2 -> padded to 16, 2 padding tokens.
        self._run_case(stream_lens=[14], cp_size=2)

    def test_single_stream_cp4(self):
        # 32 tokens, cp=4 -> 32 % 8 == 0, no padding.
        self._run_case(stream_lens=[32], cp_size=4)

    def test_multi_stream_with_padding(self):
        # Two streams, both need padding under cp=2.
        self._run_case(stream_lens=[6, 10], cp_size=2)

    def test_multi_stream_cp4_with_padding(self):
        self._run_case(stream_lens=[10, 20, 7], cp_size=4)


if __name__ == "__main__":
    unittest.main()
