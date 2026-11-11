"""SM80 decode output-projection fallback contract."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.fp8.decode.output_proj import (
    _needs_reference_output_proj,
    decode_output_proj,
)


class _FakeAttn:
    rope_head_dim = 64
    n_groups = 2
    n_heads = 4
    head_dim = 512
    o_lora_rank = 8
    tp_size = 1

    def __init__(self, device):
        self.wo_a_w = torch.randn(16, 1024, device=device).to(torch.float8_e4m3fn)
        self.wo_a_s = torch.full(
            (1, 8), 127, dtype=torch.float8_e8m0fnu, device=device
        )
        self.wo_b = torch.empty(8, 8, device=device, dtype=torch.bfloat16)

    @staticmethod
    def _lin(weight, x):
        return x[..., :8]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeOutputProjSm80Test(unittest.TestCase):
    def test_sm80_uses_eager_inverse_rope_path(self):
        torch.manual_seed(19)
        device = "cuda"
        attn = _FakeAttn(device)
        o = torch.randn(1, 1, 4, 512, device=device, dtype=torch.bfloat16)
        freqs = torch.polar(
            torch.ones(1, 32, device=device),
            torch.randn(1, 32, device=device) * 0.1,
        ).to(torch.complex64)
        self.assertTrue(_needs_reference_output_proj(o))
        out = decode_output_proj(attn, o, freqs, 1, 1)
        self.assertEqual(tuple(out.shape), (1, 1, 8))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
