from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.triton_kernels.zigzag_permute import (
    canonical_to_rank_major_zigzag,
    is_triton_zigzag_permute_supported,
    rank_major_zigzag_to_canonical,
)


def _canonical(
    world_size: int,
    block_tokens: int,
    hidden_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.arange(
        2 * world_size * block_tokens * hidden_size,
        device=device,
        dtype=torch.int64,
    )
    return (values % 251).to(dtype).reshape(2 * world_size * block_tokens, hidden_size)


def _reference_canonical_to_rank_major(
    canonical: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    blocks = canonical.reshape(2 * world_size, -1, canonical.shape[-1])
    shards = [
        torch.cat((blocks[rank], blocks[2 * world_size - 1 - rank]), dim=0)
        for rank in range(world_size)
    ]
    return torch.cat(shards, dim=0)


class ZigzagPermuteValidationTest(unittest.TestCase):
    def test_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 \\* world_size"):
            canonical_to_rank_major_zigzag(torch.zeros(15, 3), 8)

    def test_cpu_input_is_rejected(self) -> None:
        tensor = torch.zeros(16, 3, dtype=torch.bfloat16)
        self.assertFalse(is_triton_zigzag_permute_supported(tensor))
        with self.assertRaisesRegex(ValueError, "contiguous CUDA tensor"):
            canonical_to_rank_major_zigzag(tensor, 8)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ZigzagPermuteTritonTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.cuda.set_device(0)

    def test_tp8_bf16_hidden7168_exact(self) -> None:
        world_size = 8
        canonical = _canonical(
            world_size,
            block_tokens=3,
            hidden_size=7168,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.assertTrue(is_triton_zigzag_permute_supported(canonical))
        expected_rank_major = _reference_canonical_to_rank_major(canonical, world_size)
        rank_major_buffer = torch.empty_like(canonical)
        canonical_buffer = torch.empty_like(canonical)

        rank_major = canonical_to_rank_major_zigzag(
            canonical,
            world_size,
            output=rank_major_buffer,
        )
        restored = rank_major_zigzag_to_canonical(
            rank_major,
            world_size,
            output=canonical_buffer,
        )

        self.assertEqual(rank_major.data_ptr(), rank_major_buffer.data_ptr())
        self.assertEqual(restored.data_ptr(), canonical_buffer.data_ptr())
        torch.testing.assert_close(rank_major, expected_rank_major, rtol=0, atol=0)
        torch.testing.assert_close(restored, canonical, rtol=0, atol=0)

    def test_in_place_is_rejected(self) -> None:
        tensor = torch.zeros(16, 3, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(ValueError, "in-place"):
            canonical_to_rank_major_zigzag(tensor, 8, output=tensor)

    def test_one_token_per_block(self) -> None:
        world_size = 8
        canonical = _canonical(
            world_size,
            block_tokens=1,
            hidden_size=3,
            device="cuda",
            dtype=torch.bfloat16,
        )
        expected = _reference_canonical_to_rank_major(canonical, world_size)

        actual = canonical_to_rank_major_zigzag(canonical, world_size)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_row_width_one(self) -> None:
        world_size = 8
        canonical = _canonical(
            world_size,
            block_tokens=2,
            hidden_size=1,
            device="cuda",
            dtype=torch.bfloat16,
        )
        expected = _reference_canonical_to_rank_major(canonical, world_size)

        actual = canonical_to_rank_major_zigzag(canonical, world_size)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tp8_bf16_global_1m_uses_int64_offsets(self) -> None:
        world_size = 8
        global_tokens = 1_000_000
        hidden_size = 7168
        block_tokens = global_tokens // (2 * world_size)
        num_elements = global_tokens * hidden_size
        local_num_elements = (global_tokens // world_size) * hidden_size
        int32_max = (1 << 31) - 1
        uint32_max = (1 << 32) - 1
        self.assertEqual(global_tokens % (2 * world_size), 0)
        self.assertLess(local_num_elements, int32_max)
        self.assertGreater(num_elements - 1, int32_max)
        self.assertGreater(num_elements - 1, uint32_max)

        bytes_per_tensor = (
            num_elements * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        required_free_bytes = 2 * bytes_per_tensor + 4 * 1024**3
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < required_free_bytes:
            self.skipTest(
                "1M int64-offset test requires at least "
                f"{required_free_bytes / 1024**3:.1f} GiB free GPU memory, "
                f"got {free_bytes / 1024**3:.1f} GiB"
            )

        canonical = torch.zeros(
            global_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        rank_major = torch.empty_like(canonical)
        self.assertTrue(is_triton_zigzag_permute_supported(canonical))

        sentinel_tokens = (
            0,
            8 * block_tokens,
            15 * block_tokens,
            global_tokens - 1,
        )
        sentinel_values = (11.0, 22.0, 33.0, 44.0)

        def rank_major_token(canonical_token: int) -> int:
            canonical_block, token_in_block = divmod(canonical_token, block_tokens)
            if canonical_block < world_size:
                destination_block = 2 * canonical_block
            else:
                rank = 2 * world_size - 1 - canonical_block
                destination_block = 2 * rank + 1
            return destination_block * block_tokens + token_in_block

        destination_tokens = tuple(rank_major_token(token) for token in sentinel_tokens)
        max_destination_offset = max(destination_tokens) * hidden_size + hidden_size - 1
        self.assertGreater(max_destination_offset, uint32_max)

        try:
            for token, value in zip(sentinel_tokens, sentinel_values):
                canonical[token].fill_(value)

            canonical_to_rank_major_zigzag(
                canonical,
                world_size,
                output=rank_major,
            )
            for token, value in zip(destination_tokens, sentinel_values):
                self.assertTrue(
                    torch.equal(
                        rank_major[token],
                        torch.full_like(rank_major[token], value),
                    )
                )

            rank_major_zigzag_to_canonical(
                rank_major,
                world_size,
                output=canonical,
            )
            for token, value in zip(sentinel_tokens, sentinel_values):
                self.assertTrue(
                    torch.equal(
                        canonical[token],
                        torch.full_like(canonical[token], value),
                    )
                )
        finally:
            del rank_major
            del canonical
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
