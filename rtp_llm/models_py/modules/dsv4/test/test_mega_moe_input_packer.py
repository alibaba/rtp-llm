import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
    TorchMegaMoEInputPacker,
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


class TestMegaMoEInputPacker(unittest.TestCase):
    def test_dispatch(self):
        old = os.environ.pop("MEGA_MOE_INPUT_PACKER", None)
        try:
            self.assertIsInstance(get_mega_moe_input_packer(), FusedMegaMoEInputPacker)
        finally:
            if old is not None:
                os.environ["MEGA_MOE_INPUT_PACKER"] = old
        with _env("MOE_STRICT_FUSED", "0"), _env("MEGA_MOE_INPUT_PACKER", "torch"):
            self.assertIsInstance(get_mega_moe_input_packer(), TorchMegaMoEInputPacker)
        with _env("MEGA_MOE_INPUT_PACKER", "fused"):
            self.assertIsInstance(get_mega_moe_input_packer(), FusedMegaMoEInputPacker)

    def test_fused_rejects_unsupported_without_fallback(self):
        tokens = 2
        dim = 128
        topk = 8
        x = torch.randn(tokens, dim, dtype=torch.bfloat16)
        weights = torch.randn(tokens, topk, dtype=torch.float32)
        indices = torch.randint(0, 256, (tokens, topk), dtype=torch.int64)
        buf = _make_buf(tokens, dim, topk, "cpu")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA bf16"):
            FusedMegaMoEInputPacker().pack(x, weights, indices, buf, tokens)

    def test_strict_rejects_torch_packer(self):
        with _env("MEGA_MOE_INPUT_PACKER", "torch"):
            with self.assertRaisesRegex(RuntimeError, "forbids"):
                get_mega_moe_input_packer()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fused_matches_torch_buffer_bits(self):
        torch.manual_seed(3)
        for tokens in (1, 17, 128):
            with self.subTest(tokens=tokens):
                dim = 256
                topk = 8
                x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
                weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
                indices = torch.randint(
                    0, 256, (tokens, topk), device="cuda", dtype=torch.int64
                )
                ref = _make_buf(tokens, dim, topk, "cuda")
                got = _make_buf(tokens, dim, topk, "cuda")
                with _env("MOE_STRICT_FUSED", "0"):
                    TorchMegaMoEInputPacker().pack(x, weights, indices, ref, tokens)
                FusedMegaMoEInputPacker().pack(x, weights, indices, got, tokens)
                self.assertTrue(
                    torch.equal(
                        ref.x.view(torch.uint8).cpu(), got.x.view(torch.uint8).cpu()
                    )
                )
                self.assertTrue(torch.equal(ref.x_sf.cpu(), got.x_sf.cpu()))
                self.assertTrue(torch.equal(ref.topk_idx.cpu(), got.topk_idx.cpu()))
                self.assertTrue(
                    torch.equal(ref.topk_weights.cpu(), got.topk_weights.cpu())
                )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_zero_tokens_noop(self):
        buf = _make_buf(1, 128, 8, "cuda")
        FusedMegaMoEInputPacker().pack(
            torch.empty((0, 128), device="cuda", dtype=torch.bfloat16),
            torch.empty((0, 8), device="cuda", dtype=torch.float32),
            torch.empty((0, 8), device="cuda", dtype=torch.int64),
            buf,
            0,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_nonfinite_activations_are_zeroed(self):
        tokens, dim, topk = 3, 256, 8
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        x[0, 0] = float("nan")
        x[1, 1] = float("inf")
        x[2, 2] = -float("inf")
        weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
        indices = torch.randint(
            0, 256, (tokens, topk), device="cuda", dtype=torch.int64
        )
        ref = _make_buf(tokens, dim, topk, "cuda")
        got = _make_buf(tokens, dim, topk, "cuda")

        with _env("MOE_STRICT_FUSED", "0"):
            TorchMegaMoEInputPacker().pack(x, weights, indices, ref, tokens)
        FusedMegaMoEInputPacker().pack(x, weights, indices, got, tokens)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(ref.x.view(torch.uint8), got.x.view(torch.uint8)))
        self.assertTrue(torch.equal(ref.x_sf, got.x_sf))
        self.assertTrue(torch.isfinite(got.x.float()).all().item())

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fused_rejects_noncontiguous_input_without_copying(self):
        tokens, dim, topk = 2, 128, 8
        x = torch.randn(dim, tokens, device="cuda", dtype=torch.bfloat16).transpose(
            0, 1
        )
        self.assertNotEqual(x.stride(-1), 1)
        weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
        indices = torch.randint(
            0, 256, (tokens, topk), device="cuda", dtype=torch.int64
        )
        buf = _make_buf(tokens, dim, topk, "cuda")
        with self.assertRaisesRegex(ValueError, "x must be contiguous"):
            FusedMegaMoEInputPacker().pack(x, weights, indices, buf, tokens)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fused_rejects_noncontiguous_output_without_copying(self):
        tokens, dim, topk = 2, 128, 8
        x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(tokens, topk, device="cuda", dtype=torch.float32)
        indices = torch.randint(
            0, 256, (tokens, topk), device="cuda", dtype=torch.int64
        )
        buf = _make_buf(tokens, dim, topk, "cuda")
        buf.x = torch.empty(
            dim, tokens, device="cuda", dtype=torch.float8_e4m3fn
        ).transpose(0, 1)
        self.assertNotEqual(buf.x.stride(-1), 1)
        with self.assertRaisesRegex(ValueError, "out_fp8 must be contiguous"):
            FusedMegaMoEInputPacker().pack(x, weights, indices, buf, tokens)


if __name__ == "__main__":
    unittest.main()
