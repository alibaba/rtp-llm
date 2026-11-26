import os
import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
    triton_convert_req_index_to_global_index as convert,
)


class BlockIndexToGlobalTest(unittest.TestCase):
    def integers(self, values):
        device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
        return torch.tensor(values, dtype=torch.int32, device=device)

    def test_compressed_pages_and_mixed_prefill_workspace(self):
        req = self.integers([1, 0, 0])
        table = self.integers([[1, 0, -2], [4, 5, 3]])
        local = self.integers(
            [
                [0, 1, 2, 3, 4, 8, 11, 12],
                [0, 1, 4, 5, 8, 9, -1, 12],
                [0, 1, 4, 5, 8, -2, 11, 12],
            ]
        )
        decoded = [
            [8, 9, 8, 9, 10, 6, 7, -1],
            [2, 3, -1, -1, -1, -1, -1, -1],
            [2, 3, -1, -1, -1, -1, -1, -1],
        ]
        torch.testing.assert_close(
            convert(req, table, local, 4, 2, 8, 4), self.integers(decoded)
        )
        # Prefill bypasses the cache table (including unallocated pages),
        # while negative indices and out-of-range local pages still map to -1.
        prefill = self.integers([-1, -1, 1])
        starts = self.integers([100, 30])
        actual = convert(req, table, local, 4, 2, 8, 4, True, prefill, starts)
        decoded[2] = [30, 31, 34, 35, 38, -1, 41, -1]
        torch.testing.assert_close(actual, self.integers(decoded))

    def test_ragged_rows_preserve_page_pitch_and_skip_reserved_children(self):
        req = self.integers([0, 0])
        table = self.integers([[1, 6, 7, 0, -1]])
        indices = [0, 3, 4, 7, 8, 11, 12, 16, 20, -1, -3]
        for pages, pitch, slots in [
            (1, 6, [6, 9, 36, 39, 42, 45]),
            (2, 11, [-1, -1, 33, 36, 37, 40]),
            (2, None, [-1, -1, 24, 27, 28, 31]),
        ]:
            slots = slots + [-1] * 5
            for width in (3, 129, 2051):
                with self.subTest(pages=pages, pitch=pitch, width=width):
                    row = torch.arange(width, device=req.device) % len(indices)
                    local = self.integers(indices)[
                        torch.stack((row, row.flip(0)), dim=1)
                    ].T
                    expected = self.integers(slots)[torch.stack((row, row.flip(0)))]
                    actual = convert(
                        req,
                        table,
                        local,
                        4,
                        4,
                        width,
                        None,
                        PHYSICAL_ENTRIES_PER_BLOCK=pitch,
                        BLOCKS_PER_PHYSICAL_BLOCK=pages,
                    )
                    torch.testing.assert_close(actual, expected)

    def test_empty_rows_and_existing_validation(self):
        req = self.integers([0])
        table = self.integers([[2]])
        local = self.integers([[0, 1, 2]])
        for ids, local_indices in [(req[:0], local[:0]), (req, local[:, :0])]:
            out = convert(ids, table, local_indices, 4, 4, local_indices.shape[1], None)
            torch.testing.assert_close(out, local_indices)
        with self.assertRaisesRegex(AssertionError, "must be divisible by BLOCK_N"):
            convert(req, table, local, 4, 4, 3, 2)
        with self.assertRaises(AssertionError):
            convert(req, table, local, 4, 4, 3, None, True)
        with self.assertRaises(AssertionError):
            convert(
                req,
                table,
                local,
                4,
                4,
                3,
                None,
                PHYSICAL_ENTRIES_PER_BLOCK=7,
                BLOCKS_PER_PHYSICAL_BLOCK=2,
            )


if __name__ == "__main__":
    unittest.main()
