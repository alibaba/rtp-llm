import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.moe.input_packer import (
    FusedMegaMoeInputPacker,
    TorchMegaMoeInputPacker,
    get_mega_moe_input_packer,
)


@contextmanager
def _env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _make_buf(tokens, dim, topk, device):
    return SimpleNamespace(
        x=torch.empty((tokens, dim), dtype=torch.float8_e4m3fn, device=device),
        x_sf=torch.empty((tokens, dim // 128), dtype=torch.int32, device=device),
        topk_idx=torch.empty((tokens, topk), dtype=torch.int64, device=device),
        topk_weights=torch.empty((tokens, topk), dtype=torch.float32, device=device),
    )


class TestMegaMoeInputPacker(unittest.TestCase):
    def test_dispatch(self):
        with _env("DSV4_MEGA_MOE_INPUT_PACKER", "torch"):
            self.assertIsInstance(get_mega_moe_input_packer(), TorchMegaMoeInputPacker)
        with _env("DSV4_MEGA_MOE_INPUT_PACKER", "fused"):
            self.assertIsInstance(get_mega_moe_input_packer(), FusedMegaMoeInputPacker)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fused_matches_torch_buffer_bits(self):
        torch.manual_seed(3)
        for tokens in (1, 17, 128):
            with self.subTest(tokens=tokens):
                dim = 256
                topk = 8
                x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
                weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
                indices = torch.randint(0, 256, (tokens, topk), device="cuda", dtype=torch.int64)
                ref = _make_buf(tokens, dim, topk, "cuda")
                got = _make_buf(tokens, dim, topk, "cuda")
                TorchMegaMoeInputPacker().pack(x, weights, indices, ref, tokens)
                FusedMegaMoeInputPacker().pack(x, weights, indices, got, tokens)
                self.assertTrue(torch.equal(ref.x.view(torch.uint8).cpu(), got.x.view(torch.uint8).cpu()))
                self.assertTrue(torch.equal(ref.x_sf.cpu(), got.x_sf.cpu()))
                self.assertTrue(torch.equal(ref.topk_idx.cpu(), got.topk_idx.cpu()))
                self.assertTrue(torch.equal(ref.topk_weights.cpu(), got.topk_weights.cpu()))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_zero_tokens_noop(self):
        buf = _make_buf(1, 128, 8, "cuda")
        FusedMegaMoeInputPacker().pack(
            torch.empty((0, 128), device="cuda", dtype=torch.bfloat16),
            torch.empty((0, 8), device="cuda", dtype=torch.float32),
            torch.empty((0, 8), device="cuda", dtype=torch.int64),
            buf,
            0,
        )


if __name__ == "__main__":
    unittest.main()
