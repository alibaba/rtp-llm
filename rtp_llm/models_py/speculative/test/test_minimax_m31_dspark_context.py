import unittest

import torch

from rtp_llm.models_py.speculative.minimax_m31_dspark_context import map_cp_context_rows


class TestMiniMaxM31DSparkContext(unittest.TestCase):
    def test_cp4_ragged_prefix_and_padding(self):
        # Suffix9 pads to16, suffix2 pads to8. Each rank gets two halves.
        shuffles = (
            [0, 1, -1, -1, 0, -1],
            [2, 3, -1, -1, 1, -1],
            [4, 5, -1, -1, -1, -1],
            [6, 7, 8, -1, -1, -1],
        )
        seen = []
        for shuffle in shuffles:
            requests, positions = map_cp_context_rows(
                torch.tensor([4, 2]),
                torch.tensor(shuffle),
                torch.tensor([128, 4096]),
                torch.tensor([9, 2]),
            )
            for request, position, offset in zip(
                requests.tolist(), positions.tolist(), shuffle
            ):
                if offset < 0:
                    self.assertEqual((request, position), (-1, -1))
                else:
                    seen.append((request, position))
        self.assertCountEqual(
            seen, [(0, 128 + i) for i in range(9)] + [(1, 4096 + i) for i in range(2)]
        )
        self.assertEqual(len(seen), len(set(seen)))

    def test_empty_requests_and_outside_suffix(self):
        requests, positions = map_cp_context_rows(
            torch.tensor([0, 2, 0]),
            torch.tensor([1, 2]),
            torch.tensor([0, 128, 0]),
            torch.tensor([0, 2, 0]),
        )
        self.assertEqual(requests.tolist(), [1, -1])
        self.assertEqual(positions.tolist(), [129, -1])
        empty = torch.empty(0, dtype=torch.int32)
        req, pos = map_cp_context_rows(empty, empty, empty, empty)
        self.assertEqual(req.numel(), 0)
        self.assertEqual(pos.numel(), 0)

    def test_missing_request_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "sizes differ"):
            map_cp_context_rows(
                torch.tensor([2]),
                torch.tensor([0, 1]),
                torch.tensor([0, 1]),
                torch.tensor([2]),
            )


if __name__ == "__main__":
    unittest.main()
