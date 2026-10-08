import runpy
import unittest
from pathlib import Path

import torch


gather_fp8_prefix_slice = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "mla_fp8_kernels.py")
)["gather_fp8_prefix_slice"]


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MlaPrefixSliceGatherTest(unittest.TestCase):
    def test_page_crossing_and_nonzero_page_offset(self):
        device = torch.device("cuda:0")
        page_size, latent, rope = 128, 16, 8
        torch.manual_seed(19)
        cache = torch.randn(6, page_size, latent + rope, device=device).to(
            torch.float8_e4m3fn
        )
        pages = torch.tensor([5, 2, 4, 1], dtype=torch.int32, device=device)
        info = torch.tensor([[0, 1, 0, 1], [1, 300, 1, 3]], dtype=torch.int32, device=device)
        start, length = 128, 170
        c = torch.empty(length, latent, dtype=torch.bfloat16, device=device)
        r = torch.empty(length, rope, dtype=torch.bfloat16, device=device)
        gather_fp8_prefix_slice(c, r, cache, pages, info, page_size,
                                owner=1, start=start, prefix_len=300, scale=1.0)
        expected = torch.stack([
            cache[pages[1 + i // page_size].item(), i % page_size]
            for i in range(start, start + length)
        ]).to(torch.bfloat16)
        torch.testing.assert_close(c, expected[:, :latent], rtol=0, atol=0)
        torch.testing.assert_close(r, expected[:, latent:], rtol=0, atol=0)

    def test_slice_cannot_read_past_owner_prefix(self):
        cache = torch.empty(1, 128, 24, dtype=torch.float8_e4m3fn, device="cuda:0")
        pages = torch.zeros(1, dtype=torch.int32, device="cuda:0")
        info = torch.tensor([[0, 16, 0, 1]], dtype=torch.int32, device="cuda:0")
        c = torch.empty(9, 16, dtype=torch.bfloat16, device="cuda:0")
        r = torch.empty(9, 8, dtype=torch.bfloat16, device="cuda:0")
        with self.assertRaisesRegex(ValueError, "exceeds historical prefix"):
            gather_fp8_prefix_slice(c, r, cache, pages, info, 128,
                                    owner=0, start=8, prefix_len=16, scale=1.0)


if __name__ == "__main__":
    unittest.main()
