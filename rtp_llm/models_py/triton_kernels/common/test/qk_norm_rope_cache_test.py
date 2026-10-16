"""Numerical, paged-cache and CUDA Graph regressions for fused QK norm/RoPE."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.common.qk_norm_rope_cache import (
    _vllm_rope_cache,
    fused_qk_norm_rope_cache,
    maybe_fused_qk_norm_rope_cache,
)


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda, "CUDA required")
class QKNormRopeCacheTest(unittest.TestCase):
    def make_case(self, fp8=False, fp8q=False, mrope=True):
        torch.manual_seed(379)
        t, qh, kh, d, page = 3, 4, 2, 256, 64
        # An exactly representable sum of squares makes the independent Torch
        # reference insensitive to a different reduction tree.
        x = torch.arange(d, device="cuda").remainder(4).add(1).bfloat16()
        qkv = x.repeat(t, qh + 2 * kh)
        qw = (torch.randn(d, device="cuda") * 0.025).bfloat16()
        kw = (torch.randn(d, device="cuda") * 0.025).bfloat16()
        qw[0] = 2**-10
        pos = torch.tensor(
            [[7, 2, 5], [11, 8, 3], [19, 4, 13]], dtype=torch.int32, device="cuda"
        )
        if not mrope:
            pos = pos[:, :1]
        table = _vllm_rope_cache(10000.0, 64, 256, qkv.device)
        cu = torch.arange(t + 1, device="cuda", dtype=torch.int32)
        prefix = torch.tensor([63, 64, 129], device="cuda", dtype=torch.int32)
        pages = torch.tensor(
            [[6, 3, 9], [1, 8, 5], [7, 2, 4]], device="cuda", dtype=torch.int32
        )
        offsets = (
            torch.stack((pages * 2, pages * 2 + 1), dim=1).unsqueeze(1).contiguous()
        )
        cache = torch.full(
            (10, 2, kh, page, d), 0.125, device="cuda", dtype=torch.bfloat16
        )
        if fp8:
            cache = cache.to(torch.float8_e4m3fn)
        scales = torch.full((10, 2, kh, page), 0.375, device="cuda") if fp8 else None
        return SimpleNamespace(
            qkv=qkv,
            qw=qw,
            kw=kw,
            pos=pos,
            table=table,
            cu=cu,
            prefix=prefix,
            pages=pages,
            offsets=offsets,
            cache=cache,
            scales=scales,
            qh=qh,
            kh=kh,
            d=d,
            page=page,
            args=dict(
                num_q_heads=qh,
                num_kv_heads=kh,
                rotary_dim=64,
                page_size=page,
                mrope_section=[11, 11, 10] if mrope else None,
                kv_scales=scales,
                fp8_query=fp8q,
            ),
        )

    def run_fused(self, c, **extra):
        return fused_qk_norm_rope_cache(
            c.qkv,
            c.qw,
            c.kw,
            c.table,
            c.pos,
            c.cu,
            c.prefix,
            c.cache,
            c.offsets,
            **c.args,
            **extra
        )

    def reference(self, c):
        q, k, v = c.qkv.split([c.qh * c.d, c.kh * c.d, c.kh * c.d], dim=-1)
        values = []
        for source, weight, heads in ((q, c.qw, c.qh), (k, c.kw, c.kh)):
            x = source.view(-1, heads, c.d).float()
            z = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
            # Removing this cast is observably incorrect for nonzero positions.
            z = (z * (weight.float() + 1)).bfloat16().float()
            axis = torch.zeros(32, device="cuda", dtype=torch.long)
            if c.args["mrope_section"] is not None:
                axis[1:33:3] = 1
                axis[2:30:3] = 2
            positions = c.pos[:, axis].long()
            pair = torch.arange(32, device="cuda")
            cos = c.table[positions, pair].float().unsqueeze(1)
            sin = c.table[positions, pair + 32].float().unsqueeze(1)
            lo, hi = z[..., :32], z[..., 32:64]
            # CUDA/Triton can fuse one multiply-add. Assert after BF16 storage.
            low = (lo * cos - hi * sin).bfloat16()
            high = (hi * cos + lo * sin).bfloat16()
            values.append(torch.cat((low, high, z[..., 64:].bfloat16()), dim=-1))
        query, key = values
        expected = c.cache.clone()
        expected_scales = c.scales.clone() if c.scales is not None else None
        for token, physical in enumerate(c.prefix.cpu().tolist()):
            page = int(c.pages[token, physical // c.page])
            slot = physical % c.page
            for kind, value in enumerate((key[token], v.view(-1, c.kh, c.d)[token])):
                if c.cache.dtype == torch.float8_e4m3fn:
                    value = value.float().clamp(-448, 448).to(c.cache.dtype)
                expected[page, kind, :, slot] = value
                if expected_scales is not None:
                    expected_scales[page, kind, :, slot] = 1
            if c.cache.dtype == torch.float8_e4m3fn and (physical + 1) % c.page:
                expected[page, 1, :, (physical + 1) % c.page :] = 0
        if c.args["fp8_query"]:
            query = query.float().clamp(-448, 448).to(torch.float8_e4m3fn)
        return query, expected, expected_scales

    def assert_bytes(self, actual, expected):
        torch.testing.assert_close(
            actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
        )

    def test_norm_rope_cache_and_unit_scales(self):
        for fp8, fp8q, mrope in (
            (False, False, False),
            (False, False, True),
            (True, False, True),
            (True, True, True),
        ):
            with self.subTest(fp8=fp8, fp8q=fp8q, mrope=mrope):
                c = self.make_case(fp8, fp8q, mrope)
                source = c.qkv.clone()
                eq, ec, es = self.reference(c)
                actual = self.run_fused(c)
                self.assert_bytes(actual, eq)
                self.assert_bytes(c.cache, ec)
                self.assert_bytes(c.qkv, source)
                if es is not None:
                    self.assert_bytes(c.scales, es)

    def test_graph_replay_reads_updated_positions_lengths_and_pages(self):
        c = self.make_case(True, True)
        self.run_fused(c)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.run_fused(c)
        c.prefix.add_(1)
        c.pos.add_(2)
        c.pages.copy_(torch.roll(c.pages, 1, dims=1))
        c.offsets.copy_(torch.stack((c.pages * 2, c.pages * 2 + 1), dim=1).unsqueeze(1))
        eq, ec, es = self.reference(c)
        graph.replay()
        self.assert_bytes(out, eq)
        self.assert_bytes(c.cache, ec)
        self.assert_bytes(c.scales, es)

    def test_empty_rows_and_invalid_pages_do_not_write_cache(self):
        c = self.make_case(True, True)
        c.cu.zero_()
        before = c.cache.clone()
        scales = c.scales.clone()
        out = self.run_fused(c)
        self.assert_bytes(out, torch.zeros_like(out))
        self.assert_bytes(c.cache, before)
        self.assert_bytes(c.scales, scales)
        c.cu.copy_(torch.arange(4, device="cuda", dtype=torch.int32))
        c.offsets.fill_(-1)
        self.run_fused(c)
        self.assert_bytes(c.cache, before)
        self.assert_bytes(c.scales, scales)

    def test_context_parallel_metadata_falls_back_before_writing(self):
        c = self.make_case()
        impl = SimpleNamespace(
            attn_configs=None,
            rope_params=None,
            attn_inputs=SimpleNamespace(context_parallel_info=object()),
        )
        norm = SimpleNamespace(zero_centered=True)
        before = c.cache.clone()
        self.assertIsNone(maybe_fused_qk_norm_rope_cache(c.qkv, norm, impl, c.cache))
        self.assert_bytes(c.cache, before)

    def test_query_output_cannot_alias_raw_projection(self):
        c = self.make_case()
        out = c.qkv.flatten()[: 3 * c.qh * c.d].view(3, c.qh, c.d)
        with self.assertRaisesRegex(ValueError, "must not alias"):
            self.run_fused(c, out=out)

    def test_fp8_recycled_value_tail_nan_is_cleared(self):
        c = self.make_case(True, True)
        for row, physical in enumerate(c.prefix.cpu().tolist()):
            page = int(c.pages[row, physical // c.page])
            tail = (physical + 1) % c.page
            if tail:
                c.cache[page, 1, :, tail:] = float("nan")
        eq, ec, es = self.reference(c)
        self.assert_bytes(self.run_fused(c), eq)
        self.assert_bytes(c.cache, ec)

    def test_disabled_dispatch_does_not_touch_inputs(self):
        c = self.make_case()
        with patch.dict("os.environ", {"RTP_QWEN35_QK_NORM_ROPE_CACHE_FUSION": "0"}):
            self.assertIsNone(maybe_fused_qk_norm_rope_cache(c.qkv, None, None, None))


if __name__ == "__main__":
    unittest.main()
