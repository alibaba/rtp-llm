import unittest

import torch

from rtp_llm.models_py.modules.glm5_mega_moe.input_packer_se_triton import (
    fused_pack_mega_moe_se_inputs,
)
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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_packer_matches_shared_scale_staging(self):
        device = torch.device("cuda")
        for tokens, block_m in (
            (0, 64),
            (1, 64),
            (63, 64),
            (65, 64),
            (129, 32),
            (257, 192),
        ):
            with self.subTest(tokens=tokens, block_m=block_m):
                hidden = 256
                topk = 4
                x = torch.randn((tokens, hidden), device=device, dtype=torch.bfloat16)
                weights = torch.randn((tokens, topk), device=device)
                indices = torch.randint(0, 16, (tokens, topk), device=device)
                out_fp8 = torch.empty(
                    (tokens, hidden), device=device, dtype=torch.float8_e4m3fn
                )
                out_sf = torch.empty(
                    (tokens, hidden // 128), device=device, dtype=torch.int32
                )
                out_indices = torch.empty(
                    (tokens, topk), device=device, dtype=torch.int64
                )
                out_weights = torch.empty((tokens, topk), device=device)
                aligned_block_m = ((block_m + 127) // 128) * 128
                active_rows = ((tokens + block_m - 1) // block_m) * aligned_block_m
                shared = torch.empty_strided(
                    (active_rows + 128, hidden // 128),
                    (1, active_rows + 128),
                    device=device,
                    dtype=torch.int32,
                )
                shared.fill_(-1)
                expected = torch.empty_like(shared)
                expected.fill_(-1)

                fused_pack_mega_moe_se_inputs(
                    x,
                    weights,
                    indices,
                    out_fp8,
                    out_sf,
                    shared,
                    out_indices,
                    out_weights,
                    block_m,
                )
                stage_shared_fp8_input_scales(out_sf, expected, tokens, block_m)

                torch.testing.assert_close(shared, expected)
                torch.testing.assert_close(out_weights, weights)
                torch.testing.assert_close(out_indices, indices)


if __name__ == "__main__":
    unittest.main()
