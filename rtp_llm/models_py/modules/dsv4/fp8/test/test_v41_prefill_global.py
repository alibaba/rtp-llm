"""Prefill compact compressor, typed stores and immutable halo contracts."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_global as fused
from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    quantize_and_insert_k_cache_fp4,
    quantize_indexer_k_fp4,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    compress_pairs,
    rms_norm,
    rope_only,
)


def _case(
    rows,
    ratio,
    *,
    strided=False,
    norm_dtype=torch.bfloat16,
    tile_offset=0,
    batch=1,
    seed=123,
    device="cuda",
):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(rows, 1024 if strided else 512, generator=generator).to(device)
    values = raw[:, :512]
    scores = (
        raw[:, 512:]
        if strided
        else torch.randn(rows, 512, generator=generator).to(device)
    )
    starts_cpu = [i % 2 + 127 * i for i in range(batch)]
    lengths = [rows // batch + (i < rows % batch) for i in range(batch)]
    pos_cpu, req_cpu = [], []
    for req, (start, length) in enumerate(zip(starts_cpu, lengths)):
        pos_cpu.extend(range(start + tile_offset, start + tile_offset + length))
        req_cpu.extend([req] * length)
    boundaries = [i for i, pos in enumerate(pos_cpu) if (pos + 1) % ratio == 0]
    device_tensor = lambda x: torch.tensor(x, dtype=torch.int64, device=device)
    c = len(boundaries)
    norm = (torch.rand(512, generator=generator) + 0.5).to(
        device=device, dtype=norm_dtype
    )
    index_norm = (torch.rand(128, generator=generator) + 0.5).to(
        device=device, dtype=norm_dtype
    )
    previous = torch.randn(batch, 1024, generator=generator).to(device)
    carry = tuple(torch.randn(1, 512, generator=generator).to(device) for _ in range(2))
    freq_rows = max(pos_cpu, default=0) + 1
    angles = torch.outer(
        torch.arange(freq_rows).float(), torch.linspace(0.001, 0.1, 32)
    )
    freqs = torch.polar(torch.ones_like(angles), angles).to(device)
    entries = 64
    pages = (c + entries - 1) // entries + 2

    def pool(width):
        # Page padding catches accidental use of pool.numel()/blocks as stride.
        return torch.full(
            (pages, entries * width + 128), 0x5A, dtype=torch.uint8, device=device
        )[:, : entries * width].view(pages, entries, width)

    main_slots = torch.arange(c, device=device) + entries
    index_slots = main_slots.flip(0).contiguous()
    main_slots[::4] = -1
    index_slots[1::4] = -1
    state_slots = torch.arange(rows, device=device) + 1
    state_slots[::3] = -1
    state = torch.full((rows + 2, 1056), -123.0, device=device)[:, :1024]
    projection = (torch.randn(128, 512, generator=generator) / 512**0.5).to(
        device=device, dtype=torch.bfloat16
    )
    return SimpleNamespace(
        values=values,
        scores=scores,
        positions=device_tensor(pos_cpu),
        req_ids=device_tensor(req_cpu),
        starts=device_tensor(starts_cpu),
        previous=previous,
        carry=carry,
        boundary_indices=device_tensor(boundaries),
        freqs=freqs,
        norm=norm,
        index_norm=index_norm,
        main_pool=pool(288),
        index_pool=pool(68),
        main_slots=main_slots,
        index_slots=index_slots,
        state_slots=state_slots,
        state=state,
        projection=projection,
        ratio=ratio,
        eps=1e-6,
    )


def _latent_reference(c):
    if c.ratio == 1:
        return rms_norm(c.values, c.norm, c.eps).to(torch.bfloat16)[c.boundary_indices]
    value_prev = torch.cat((c.carry[0], c.values[:-1]), 0)
    score_prev = torch.cat((c.carry[1], c.scores[:-1]), 0)
    first = c.positions == c.starts[c.req_ids]
    value_prev = torch.where(first[:, None], c.previous[c.req_ids, :512], value_prev)
    score_prev = torch.where(first[:, None], c.previous[c.req_ids, 512:], score_prev)
    idx = c.boundary_indices
    return compress_pairs(
        torch.stack((value_prev[idx], c.values[idx]), 1),
        torch.stack((score_prev[idx], c.scores[idx]), 1),
        c.norm,
        c.eps,
    )


def _compress(c):
    return fused.compress_main(
        c.values,
        c.scores,
        c.norm,
        c.eps,
        c.positions,
        c.req_ids,
        c.starts,
        c.previous,
        c.boundary_indices,
        c.freqs,
        c.main_pool,
        c.main_slots,
        c.ratio,
        c.carry,
    )


def _reference(c):
    latent = _latent_reference(c)
    positions = c.positions[c.boundary_indices]
    freqs = c.freqs[positions // c.ratio * c.ratio]
    keys = rope_only(latent.clone(), freqs, 64)
    quantize_and_insert_k_cache_fp4(keys, c.main_pool, c.main_slots)
    projected = F.linear(latent, c.projection)
    keys = rope_only(rms_norm(projected, c.index_norm, c.eps), freqs, 64)
    quantize_indexer_k_fp4(keys, c.index_slots, c.index_pool)
    data = torch.cat((c.values, c.scores), -1)
    idx = c.state_slots.clamp_min(0)
    c.state[idx] = torch.where((c.state_slots >= 0)[:, None], data, c.state[idx])
    return latent


def _candidate(c):
    latent = _compress(c)
    assert latent is not None
    projected = F.linear(latent, c.projection)
    assert fused.store_index(
        projected,
        c.index_norm,
        c.eps,
        c.positions[c.boundary_indices],
        c.freqs,
        c.index_pool,
        c.index_slots,
        c.ratio,
    )
    assert fused.store_states(c.values, c.scores, c.state_slots, c.state)
    return latent


@triton.jit
def _rope_probe(X, FREQ, O, D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, D)
    x = tl.load(X + row * D + cols)
    y = fused._rope_bf16(x, FREQ, row, D)
    tl.store(O + row * D + cols, y)


class PrefillGlobalGateTest(unittest.TestCase):
    def test_cpu_and_disabled_fallback(self):
        c = _case(8, 2, device="cpu")
        self.assertIsNone(_compress(c))
        self.assertFalse(fused.store_states(c.values, c.scores, c.state_slots, c.state))
        with patch.dict(os.environ, {"DSV41_FUSED_PREFILL_GLOBAL": "0"}):
            self.assertFalse(fused._enabled(c.values))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillGlobalCudaTest(unittest.TestCase):
    def _compare(self, rows, ratio, **kwargs):
        ref, got = _case(rows, ratio, **kwargs), _case(rows, ratio, **kwargs)
        unchanged = [
            x.clone() for x in (got.values, got.scores, got.previous, *got.carry)
        ]
        expected = _reference(ref)
        actual = _candidate(got)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name in ("main_pool", "index_pool", "state"):
            torch.testing.assert_close(
                getattr(got, name), getattr(ref, name), rtol=0, atol=0, msg=name
            )
        for actual, expected in zip(
            (got.values, got.scores, got.previous, *got.carry), unchanged
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_small_compact_requests_and_tile_halo(self):
        for ratio in (1, 2):
            for dtype in (torch.bfloat16, torch.float32):
                for offset in (0, 17):
                    with self.subTest(ratio=ratio, dtype=dtype, offset=offset):
                        self._compare(
                            97,
                            ratio,
                            batch=4,
                            strided=True,
                            norm_dtype=dtype,
                            tile_offset=offset,
                        )

    def test_small_reduction_widths(self):
        for rows in (1, 2, 3, 8, 17):
            for ratio in (1, 2):
                with self.subTest(rows=rows, ratio=ratio):
                    self._compare(rows, ratio, strided=True)

    def test_large_projection_tiles(self):
        for rows, ratio in ((1024, 2), (32768, 2), (32768, 1)):
            with self.subTest(rows=rows, ratio=ratio):
                self._compare(rows, ratio, strided=True, tile_offset=32769)

    def test_index_epilogue_bytes_and_zero_signs(self):
        for dtype in (torch.bfloat16, torch.float32):
            c = _case(513, 1, norm_dtype=dtype)
            projected = torch.randn(513, 128, device="cuda", dtype=torch.bfloat16)
            projected[0].zero_()
            projected[1].fill_(-0.0)
            projected[2].fill_(2.0**-60)
            projected[3].fill_(2.0**30)
            expected = c.index_pool.clone()
            keys = rope_only(
                rms_norm(projected, c.index_norm, c.eps), c.freqs[c.positions], 64
            )
            quantize_indexer_k_fp4(keys, c.index_slots, expected)
            self.assertTrue(
                fused.store_index(
                    projected,
                    c.index_norm,
                    c.eps,
                    c.positions,
                    c.freqs,
                    c.index_pool,
                    c.index_slots,
                    1,
                )
            )
            torch.testing.assert_close(c.index_pool, expected, rtol=0, atol=0)

    def test_dynamic_graph_replay(self):
        c = _case(129, 2, strided=True, tile_offset=19)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _candidate(c)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = _candidate(c)
        torch.cuda.current_stream().wait_stream(stream)
        for seed in (67, 68, 69):
            ref = _case(129, 2, strided=True, tile_offset=19, seed=seed)
            for name in (
                "values",
                "scores",
                "previous",
                "norm",
                "index_norm",
                "projection",
            ):
                getattr(c, name).copy_(getattr(ref, name))
            for a, b in zip(c.carry, ref.carry):
                a.copy_(b)
            graph.replay()
            expected = _reference(ref)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            for name in ("main_pool", "index_pool", "state"):
                torch.testing.assert_close(
                    getattr(c, name), getattr(ref, name), rtol=0, atol=0
                )

    def test_rope_intermediate_bits(self):
        for dim in (128, 512):
            for rows in (17, 32768):
                c = _case(rows, 1)
                x = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
                x[0].zero_()
                x[1].fill_(-0.0)
                expected = rope_only(x.clone(), c.freqs, 64)
                actual = torch.empty_like(x)
                _rope_probe[(rows,)](
                    x,
                    c.freqs.view(torch.float32),
                    actual,
                    dim,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                torch.testing.assert_close(
                    actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
                )

    def test_layout_fallback_and_empty(self):
        c = _case(8, 2)
        c.values = c.values.to(torch.bfloat16)
        self.assertIsNone(_compress(c))
        c = _case(1, 2)
        self.assertEqual(_compress(c).shape, (0, 512))


if __name__ == "__main__":
    unittest.main()
