"""Verify the production combined-table split with request-local fixtures."""

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

DSV4 = Path(__file__).resolve().parent.parent


class PrefillIndicesTest(unittest.TestCase):
    def test_multiple_requests_and_empty_rows(self):
        cache = types.ModuleType("rtp_llm.models_py.modules.dsv4.const_cache")
        cache.__dict__.update(runpy.run_path(str(DSV4 / "const_cache.py")))
        split = runpy.run_path(str(DSV4 / "fp8/_sm120_prefill_indices.py"))[
            "split_sm120_combined_tables"
        ]
        combined = torch.tensor(
            [
                [1, 3, 4, 5, -1, -1, -1],
                [12, 14, 15, 16, -1, -1, -1],
                [24, 25, 26, 27, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        for ratio, page_size in ((4, 64), (128, 2)):
            for extra_axis in (False, True):
                with self.subTest(ratio=ratio, extra_axis=extra_axis), patch.dict(
                    sys.modules, {"rtp_llm.models_py.modules.dsv4.const_cache": cache}
                ):
                    swa, swa_lens, extra, extra_lens, page = split(
                        combined.unsqueeze(1) if extra_axis else combined,
                        torch.tensor([4, 4, 4, 0], dtype=torch.int32),
                        M=10,
                        N=4,
                        window_size=4,
                        extra_width=3,
                        ratio=ratio,
                        device=torch.device("cpu"),
                    )
                self.assertEqual(
                    swa.tolist(),
                    [[0, 1, 0, 0], [6, 7, 8, 0], [12, 13, 14, 15], [0, 0, 0, 0]],
                )
                self.assertEqual(swa_lens.tolist(), [2, 3, 4, 0])
                self.assertEqual(
                    extra[:, :3].tolist(), [[1, 3, 0], [6, 0, 0], [0, 0, 0], [0, 0, 0]]
                )
                self.assertEqual(extra_lens.tolist(), [2, 1, 0, 0])
                self.assertEqual(extra.shape, (4, 64))
                self.assertTrue(torch.all(extra[:, 3:] == 0))
                self.assertEqual(page, page_size)


if __name__ == "__main__":
    unittest.main()
