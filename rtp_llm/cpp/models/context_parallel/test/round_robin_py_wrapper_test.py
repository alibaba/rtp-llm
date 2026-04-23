import os
import sys
import unittest

import torch

from rtp_llm.cpp.models.context_parallel.test import (
    libth_round_robin_py_wrapper_test as rr_test,
)


class TestRoundRobinPlan(unittest.TestCase):
    """Page-level round-robin plan tests with page_size=64/256.

    Distribution: block b → rank (b % cp_size).
    Each rank gets contiguous pages of page_size tokens.
    cpAlignSize = page_size * cp_size.
    """

    @staticmethod
    def _align_up(n, align):
        return ((n + align - 1) // align) * align

    def _verify_plan(self, num_tokens, cp_size, page_size, token_offset=1000):
        """Generic verification for page-level RR plan.

        Uses token_offset to make token values differ from positions,

        ensuring we verify actual token content, not just indices.

        Checks:
        1. Each rank gets correct shuffle_indices (page-level interleave)
        2. Valid tokens match original content (token_offset + position)
        3. Padding tokens are 0
        4. All ranks together cover all padded positions exactly once
        5. All original tokens collected across ranks match the input
        """
        align = page_size * cp_size
        padded = self._align_up(num_tokens, align)
        cp_chunk_size = padded // cp_size
        cp_padding_size = padded - num_tokens
        # Use non-trivial token values: token[i] = token_offset + i
        # so token value != position index, preventing false positives
        total_tokens = [token_offset + i for i in range(num_tokens)]

        all_indices = []
        all_valid_tokens = []
        for cp_rank in range(cp_size):
            result, tokens, indices = rr_test.round_robin_plan(
                total_tokens,
                [],
                [],
                cp_rank,
                cp_size,
                cp_chunk_size,
                cp_padding_size,
                page_size=page_size,
            )
            self.assertTrue(result)
            self.assertEqual(len(tokens), cp_chunk_size)
            self.assertEqual(len(indices), cp_chunk_size)

            num_local_blocks = cp_chunk_size // page_size
            for j in range(num_local_blocks):
                global_block = cp_rank + j * cp_size
                for t in range(page_size):
                    local_pos = j * page_size + t
                    global_pos = global_block * page_size + t
                    # shuffle_indices must map to correct global position
                    self.assertEqual(
                        indices[local_pos],
                        global_pos,
                        f"rank={cp_rank}, block={j}, t={t}: "
                        f"indices[{local_pos}] expected {global_pos}, got {indices[local_pos]}",
                    )
                    # valid tokens must match original content
                    if global_pos < num_tokens:
                        expected_token = total_tokens[global_pos]
                        self.assertEqual(
                            tokens[local_pos],
                            expected_token,
                            f"rank={cp_rank}, global_pos={global_pos}: "
                            f"expected token {expected_token}, got {tokens[local_pos]}",
                        )
                        all_valid_tokens.append(tokens[local_pos])
                    else:
                        self.assertEqual(
                            tokens[local_pos],
                            0,
                            f"rank={cp_rank}, global_pos={global_pos}: "
                            f"padding should be 0, got {tokens[local_pos]}",
                        )

            all_indices.extend(indices)

        # All padded positions [0..padded) covered exactly once
        self.assertEqual(sorted(all_indices), list(range(padded)))
        # All original tokens collected across ranks match the full input
        self.assertEqual(sorted(all_valid_tokens), sorted(total_tokens))

    def test_page64_cp2_aligned(self):
        """page_size=64, cp_size=2, 256 tokens (aligned, no padding)."""
        self._verify_plan(num_tokens=256, cp_size=2, page_size=64)

    def test_page64_cp2_unaligned(self):
        """page_size=64, cp_size=2, 200 tokens → pad to 256."""
        self._verify_plan(num_tokens=200, cp_size=2, page_size=64)

    def test_page64_cp4_unaligned(self):
        """page_size=64, cp_size=4, 300 tokens → pad to 512."""
        self._verify_plan(num_tokens=300, cp_size=4, page_size=64)

    def test_page256_cp2_unaligned(self):
        """page_size=256, cp_size=2, 600 tokens → pad to 1024."""
        self._verify_plan(num_tokens=600, cp_size=2, page_size=256)

    def test_page256_cp4_aligned(self):
        """page_size=256, cp_size=4, 2048 tokens (aligned, no padding)."""
        self._verify_plan(num_tokens=2048, cp_size=4, page_size=256)

    def test_page256_cp4_unaligned(self):
        """page_size=256, cp_size=4, 1500 tokens → pad to 2048."""
        self._verify_plan(num_tokens=1500, cp_size=4, page_size=256)

    def test_page64_cp2_less_than_one_page(self):
        """Edge case: 30 tokens < page_size=64, cp_size=2 → pad to 128."""
        self._verify_plan(num_tokens=30, cp_size=2, page_size=64)

    def test_page64_cp2_one_token_over_page(self):
        """Edge case: 65 tokens (one over page_size=64), cp_size=2 → pad to 256."""
        self._verify_plan(num_tokens=65, cp_size=2, page_size=64)

    def test_page64_cp2_explicit_token_distribution(self):
        """Explicitly verify page-level token distribution with readable values.

        page_size=64, cp_size=2, 256 tokens (4 blocks, aligned).
        Tokens = [1000..1255], value != position.

        Expected distribution:
          Rank 0: block 0 [1000..1063] + block 2 [1128..1191]
          Rank 1: block 1 [1064..1127] + block 3 [1192..1255]
        """
        page_size = 64
        cp_size = 2
        num_tokens = 256
        total_tokens = [1000 + i for i in range(num_tokens)]

        # Rank 0: blocks 0, 2
        result0, tokens0, _ = rr_test.round_robin_plan(
            total_tokens,
            [],
            [],
            0,
            cp_size,
            128,
            0,
            page_size=page_size,
        )
        self.assertTrue(result0)
        # First page: block 0, tokens [1000..1063]
        self.assertEqual(tokens0[:64], list(range(1000, 1064)))
        # Second page: block 2, tokens [1128..1191]
        self.assertEqual(tokens0[64:128], list(range(1128, 1192)))

        # Rank 1: blocks 1, 3
        result1, tokens1, _ = rr_test.round_robin_plan(
            total_tokens,
            [],
            [],
            1,
            cp_size,
            128,
            0,
            page_size=page_size,
        )
        self.assertTrue(result1)
        # First page: block 1, tokens [1064..1127]
        self.assertEqual(tokens1[:64], list(range(1064, 1128)))
        # Second page: block 3, tokens [1192..1255]
        self.assertEqual(tokens1[64:128], list(range(1192, 1256)))

    def test_page64_cp2_unaligned_explicit_padding(self):
        """Explicitly verify padding tokens for unaligned case.

        page_size=64, cp_size=2, 200 tokens → pad to 256, padding=56.
        Tokens = [500..699].

        Expected:
          Rank 0: block 0 [500..563] + block 2 [628..691]
          Rank 1: block 1 [564..627] + block 3 [692..699, 0*56]
                  (block 3 has 8 valid + 56 padding)
        """
        page_size = 64
        cp_size = 2
        num_tokens = 200
        total_tokens = [500 + i for i in range(num_tokens)]
        padded = 256
        cp_chunk_size = 128
        cp_padding_size = 56

        # Rank 0: blocks 0, 2 — all valid
        _, tokens0, _ = rr_test.round_robin_plan(
            total_tokens,
            [],
            [],
            0,
            cp_size,
            cp_chunk_size,
            cp_padding_size,
            page_size=page_size,
        )
        self.assertEqual(tokens0[:64], list(range(500, 564)))
        self.assertEqual(tokens0[64:128], list(range(628, 692)))

        # Rank 1: blocks 1, 3
        _, tokens1, _ = rr_test.round_robin_plan(
            total_tokens,
            [],
            [],
            1,
            cp_size,
            cp_chunk_size,
            cp_padding_size,
            page_size=page_size,
        )
        # Block 1: fully valid [564..627]
        self.assertEqual(tokens1[:64], list(range(564, 628)))
        # Block 3: first 8 valid [692..699], then 56 padding zeros
        self.assertEqual(tokens1[64:72], list(range(692, 700)))
        self.assertEqual(tokens1[72:128], [0] * 56)


class TestRoundRobinRestoreIndices(unittest.TestCase):
    """Page-level restore indices tests with page_size=64/256."""

    def _compute_expected_restore(self, chunk_lengths_list, cp_size, page_size):
        """Compute expected restore indices for page-level RR.

        After all-gather: [rank0_chunk | rank1_chunk | ...] (flat).
        restore_indices[dst_pos] = src_pos in flat all-gathered tensor.

        For each stream, rank r owns local blocks j=0,1,...
        which map to global blocks (r + j*cp_size).
        dst_pos = stream_offset + global_block * page_size + t
        src_pos = rank * total_tokens + stream_local_offset + j * page_size + t
        """
        total_tokens = sum(chunk_lengths_list)
        total_padded = total_tokens * cp_size
        expected = [0] * total_padded

        offset = 0  # cumulative stream offset in global space
        local_offset = 0  # cumulative per-rank offset
        for cl in chunk_lengths_list:
            num_local_blocks = cl // page_size
            for rank in range(cp_size):
                for j in range(num_local_blocks):
                    global_block = rank + j * cp_size
                    for t in range(page_size):
                        dst = offset + global_block * page_size + t
                        src = rank * total_tokens + local_offset + j * page_size + t
                        expected[dst] = src
            offset += cl * cp_size
            local_offset += cl

        return torch.tensor(expected, dtype=torch.int32)

    def test_page64_cp2_single_stream(self):
        """page_size=64, cp_size=2, single stream with 128 tokens/rank."""
        page_size = 64
        cp_size = 2
        chunk_len = 128  # 2 blocks per rank
        prefill_cp_chunk_lengths = torch.tensor([chunk_len], dtype=torch.int32)

        restore_indices = rr_test.round_robin_generate_qkv_restore_indices(
            prefill_cp_chunk_lengths,
            cp_size,
            page_size=page_size,
        )

        total = cp_size * chunk_len
        self.assertEqual(restore_indices.numel(), total)
        self.assertEqual(sorted(restore_indices.tolist()), list(range(total)))

        expected = self._compute_expected_restore([chunk_len], cp_size, page_size)
        self.assertTrue(
            torch.equal(restore_indices, expected),
            f"Mismatch:\nexpected={expected.tolist()[:20]}...\ngot={restore_indices.tolist()[:20]}...",
        )

    def test_page64_cp4_single_stream(self):
        """page_size=64, cp_size=4, single stream with 256 tokens/rank (4 blocks)."""
        page_size = 64
        cp_size = 4
        chunk_len = 256
        prefill_cp_chunk_lengths = torch.tensor([chunk_len], dtype=torch.int32)

        restore_indices = rr_test.round_robin_generate_qkv_restore_indices(
            prefill_cp_chunk_lengths,
            cp_size,
            page_size=page_size,
        )

        total = cp_size * chunk_len
        self.assertEqual(restore_indices.numel(), total)
        self.assertEqual(sorted(restore_indices.tolist()), list(range(total)))

        expected = self._compute_expected_restore([chunk_len], cp_size, page_size)
        self.assertTrue(
            torch.equal(restore_indices, expected),
            f"Mismatch at first diff",
        )

    def test_page64_cp2_multi_stream(self):
        """page_size=64, cp_size=2, two streams [128, 256] tokens/rank."""
        page_size = 64
        cp_size = 2
        chunk_lengths = [128, 256]
        prefill_cp_chunk_lengths = torch.tensor(chunk_lengths, dtype=torch.int32)

        restore_indices = rr_test.round_robin_generate_qkv_restore_indices(
            prefill_cp_chunk_lengths,
            cp_size,
            page_size=page_size,
        )

        total = cp_size * sum(chunk_lengths)
        self.assertEqual(restore_indices.numel(), total)
        self.assertEqual(sorted(restore_indices.tolist()), list(range(total)))

        expected = self._compute_expected_restore(chunk_lengths, cp_size, page_size)
        self.assertTrue(
            torch.equal(restore_indices, expected),
            f"Multi-stream restore mismatch",
        )

    def test_page256_cp2_single_stream(self):
        """page_size=256, cp_size=2, single stream with 512 tokens/rank."""
        page_size = 256
        cp_size = 2
        chunk_len = 512
        prefill_cp_chunk_lengths = torch.tensor([chunk_len], dtype=torch.int32)

        restore_indices = rr_test.round_robin_generate_qkv_restore_indices(
            prefill_cp_chunk_lengths,
            cp_size,
            page_size=page_size,
        )

        total = cp_size * chunk_len
        self.assertEqual(restore_indices.numel(), total)
        self.assertEqual(sorted(restore_indices.tolist()), list(range(total)))

        expected = self._compute_expected_restore([chunk_len], cp_size, page_size)
        self.assertTrue(torch.equal(restore_indices, expected))


class TestRoundRobinPaddingMask(unittest.TestCase):
    """Page-level padding mask tests with page_size=64/256."""

    def test_page64_no_padding(self):
        """page_size=64, no padding needed."""
        page_size = 64
        cp_size = 2
        # chunk=128 (2 blocks), no padding
        prefill_cp_chunk_lengths = torch.tensor([128], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([0], dtype=torch.int32)

        mask = rr_test.round_robin_generate_qkv_padding_mask(
            prefill_cp_chunk_lengths,
            prefill_cp_padding_lengths,
            cp_size,
            page_size=page_size,
        )
        total = cp_size * 128
        self.assertEqual(mask.numel(), total)
        self.assertTrue(torch.all(mask == 1))

    def test_page64_with_padding(self):
        """page_size=64, cp_size=2, 200 tokens → padded 256, padding=56."""
        page_size = 64
        cp_size = 2
        # chunk=128, padding=56 (in global padded space)
        prefill_cp_chunk_lengths = torch.tensor([128], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([56], dtype=torch.int32)

        mask = rr_test.round_robin_generate_qkv_padding_mask(
            prefill_cp_chunk_lengths,
            prefill_cp_padding_lengths,
            cp_size,
            page_size=page_size,
        )
        total = cp_size * 128  # 256
        valid = total - 56  # 200
        self.assertEqual(mask.numel(), total)
        self.assertEqual(mask.sum().item(), valid)
        self.assertTrue(torch.all(mask[:valid] == 1))
        self.assertTrue(torch.all(mask[valid:] == 0))

    def test_page256_with_padding(self):
        """page_size=256, cp_size=2, 600 tokens → padded 1024, padding=424."""
        page_size = 256
        cp_size = 2
        prefill_cp_chunk_lengths = torch.tensor([512], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([424], dtype=torch.int32)

        mask = rr_test.round_robin_generate_qkv_padding_mask(
            prefill_cp_chunk_lengths,
            prefill_cp_padding_lengths,
            cp_size,
            page_size=page_size,
        )
        total = cp_size * 512  # 1024
        valid = total - 424  # 600
        self.assertEqual(mask.numel(), total)
        self.assertEqual(mask.sum().item(), valid)
        self.assertTrue(torch.all(mask[:valid] == 1))
        self.assertTrue(torch.all(mask[valid:] == 0))

    def test_page64_multi_stream_mixed_padding(self):
        """page_size=64, cp_size=2, two streams: one aligned, one with padding."""
        page_size = 64
        cp_size = 2
        # Stream 0: chunk=128, no padding → 256 valid
        # Stream 1: chunk=128, padding=56 → 200 valid
        prefill_cp_chunk_lengths = torch.tensor([128, 128], dtype=torch.int32)
        prefill_cp_padding_lengths = torch.tensor([0, 56], dtype=torch.int32)

        mask = rr_test.round_robin_generate_qkv_padding_mask(
            prefill_cp_chunk_lengths,
            prefill_cp_padding_lengths,
            cp_size,
            page_size=page_size,
        )
        # Stream 0: 256 all valid
        self.assertTrue(torch.all(mask[:256] == 1))
        # Stream 1: 200 valid + 56 padding
        self.assertTrue(torch.all(mask[256 : 256 + 200] == 1))
        self.assertTrue(torch.all(mask[256 + 200 :] == 0))


if __name__ == "__main__":
    unittest.main()
