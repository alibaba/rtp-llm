"""Byte exact generic MLA FP8 epilogue checks, including paged cache writes."""

import importlib.util
import unittest
from pathlib import Path

import torch


def fused_epilogue():
    path = Path(__file__).parents[1] / "mla_fused_fp8_epilogue.py"
    if not path.is_file():
        raise AssertionError("generic MLA FP8 fused epilogue is missing")
    spec = importlib.util.spec_from_file_location("mla_fused_fp8_epilogue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fused_mla_fp8_epilogue


class FusedMlaFp8EpilogueTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_strided_kv_and_sparse_slots_match_reference_bytes(self):
        op = fused_epilogue()
        torch.manual_seed(147)
        device = "cuda"
        tokens, heads, nope, rope, value_dim, latent, block = 12, 3, 16, 8, 16, 32, 4
        q = (torch.randn(tokens, heads, nope + rope, device=device) * 0.5).bfloat16()
        joined = (torch.randn(tokens, heads, nope + value_dim, device=device) * 0.5).bfloat16()
        k_nope, v = joined[..., :nope], joined[..., nope:]
        pe = (torch.randn(tokens, rope, device=device) * 0.5).bfloat16()
        compressed = (torch.randn(tokens, latent, device=device) * 0.5).bfloat16()
        slots = torch.tensor([0, 3, 4, -1, 2, 8, 7, -1, 11, 1, 6, 9],
                             dtype=torch.int64, device=device)
        q_scale = torch.tensor([0.5], dtype=torch.float32, device=device)
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
        v_scale = torch.tensor([1.25], dtype=torch.float32, device=device)
        cache_scale = torch.tensor([0.75], dtype=torch.float32, device=device)
        cache = torch.zeros((3, block, latent + rope), dtype=torch.float8_e4m3fn,
                            device=device)
        actual = op(q, k_nope, pe, compressed, v, cache, slots,
                    q_scale, k_scale, v_scale, cache_scale)
        key = torch.cat((k_nope, pe[:, None, :].expand(-1, heads, -1)), dim=-1)
        expected = tuple((x.float() * scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                         for x, scale in ((q, q_scale), (key, k_scale), (v, v_scale)))
        expected_cache = torch.zeros_like(cache).view(-1, latent + rope)
        latent_pe = torch.cat((compressed, pe), dim=-1)
        expected_cache[slots[slots >= 0]] = (
            latent_pe[slots >= 0].float() * cache_scale
        ).clamp(-448, 448).to(torch.float8_e4m3fn)
        for got, want in zip(actual, expected):
            self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))
        self.assertTrue(torch.equal(cache.view(torch.uint8),
                                    expected_cache.view_as(cache).view(torch.uint8)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_k3_shape_without_special_case(self):
        op = fused_epilogue()
        device = "cuda"
        tokens, heads = 16, 12
        q = torch.randn(tokens, heads, 192, device=device).bfloat16()
        kv = torch.randn(tokens, heads, 256, device=device).bfloat16()
        pe = torch.randn(tokens, 64, device=device).bfloat16()
        compressed = torch.randn(tokens, 512, device=device).bfloat16()
        q[0, 0, 0] = 1000.0
        kv[0, 0, 0] = -1000.0
        pe[0, 0] = 1000.0
        compressed[0, 0] = -1000.0
        slots = torch.arange(tokens, dtype=torch.int64, device=device)
        scale = torch.ones((1,), dtype=torch.float32, device=device)
        cache = torch.zeros((1, 128, 576), dtype=torch.float8_e4m3fn, device=device)
        try:
            q8, k8, v8 = op(q, kv[..., :128], pe, compressed, kv[..., 128:],
                             cache, slots, scale, scale, scale, scale,
                             assume_unit_scales=True)
        except TypeError as exc:
            self.fail(f"unit-scale MLA path is missing: {exc}")
        self.assertEqual((tokens, heads, 192), tuple(q8.shape))
        self.assertEqual((tokens, heads, 192), tuple(k8.shape))
        self.assertEqual((tokens, heads, 128), tuple(v8.shape))
        def reference(x):
            return x.float().clamp(-448, 448).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(q8.view(torch.uint8),
                                    reference(q).view(torch.uint8)))
        expected_key = torch.cat((kv[..., :128],
                                  pe[:, None, :].expand(-1, heads, -1)), -1)
        self.assertTrue(torch.equal(k8.view(torch.uint8),
                                    reference(expected_key).view(torch.uint8)))
        self.assertTrue(torch.equal(v8.view(torch.uint8),
                                    reference(kv[..., 128:]).view(torch.uint8)))
        self.assertTrue(torch.equal(cache.view(-1, 576)[:tokens].view(torch.uint8),
                                    reference(torch.cat((compressed, pe), -1))
                                    .view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
