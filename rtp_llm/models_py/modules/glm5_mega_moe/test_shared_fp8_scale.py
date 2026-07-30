import unittest

import torch

from rtp_llm.models_py.modules.glm5_mega_moe.shared_fp8_scale import (
    shared_fp8_scale_row_indices,
    stage_shared_fp8_input_scales,
)


class SharedFp8ScaleTest(unittest.TestCase):
    def test_row_indices_match_deepgemm_layout(self):
        indices = shared_fp8_scale_row_indices(65, 64)
        self.assertEqual(indices[:5].tolist(), [0, 4, 8, 12, 16])
        self.assertEqual(indices[31].item(), 124)
        self.assertEqual(indices[32].item(), 1)
        self.assertEqual(indices[63].item(), 125)
        self.assertEqual(indices[64].item(), 128)

    def test_cpu_staging_copies_scales_and_clears_padding(self):
        tokens = 65
        source = torch.arange(tokens * 2, dtype=torch.int32).view(tokens, 2)
        destination = torch.empty_strided((384, 2), (1, 384), dtype=torch.int32)
        destination.fill_(-1)

        stage_shared_fp8_input_scales(source, destination, tokens, block_m=64)

        indices = shared_fp8_scale_row_indices(tokens, 64)
        torch.testing.assert_close(destination[indices], source)
        self.assertEqual(destination[2].tolist(), [0, 0])
        self.assertEqual(destination[383].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
