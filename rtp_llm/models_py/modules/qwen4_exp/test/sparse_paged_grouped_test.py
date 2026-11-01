"""Batch-matrix correctness tests for the grouped tensor-core sparse paged GQA path.

Covers the SM90 `_sparse_paged_grouped_split_kernel` dispatch in
``sparse_paged_gqa_attn`` across batch sizes, query lengths, cache dtypes, and
the packed selection format, against a vectorised PyTorch reference. The large
prefill shapes validate sampled rows so the reference stays tractable.
"""

import importlib.util
import os
import sys
import unittest

import torch

import rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha as _installed

_DEVICE = "cuda"

_OVERRIDE = os.environ.get("QSA_OPT_MODULE_OVERRIDE")
if _OVERRIDE:
    _spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha", _OVERRIDE
    )
    _candidate = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _candidate
    _spec.loader.exec_module(_candidate)
    _active = _candidate
else:
    _candidate = None
    _active = _installed

sparse_paged_gqa_attn = _active.sparse_paged_gqa_attn
grouped_supported = getattr(_active, "_grouped_split_supported", None)


def _reference(q, cache, block_table, kv_lens, selected, page_size, scale=None):
    """Vectorised reference: same math as the production python oracle."""
    batch, q_heads, query_len, head_dim = q.shape
    kv_heads = cache.shape[2]
    group = q_heads // kv_heads
    if cache.dtype == torch.float8_e4m3fn:
        cache_f = cache.to(torch.float32) * scale.unsqueeze(-1)
    else:
        cache_f = cache.to(torch.float32)
    out = torch.zeros_like(q)
    for b in range(batch):
        for s in range(query_len):
            row = selected[b, s]
            valid = (row >= 0) & (row < kv_lens[b, s])
            if not bool(valid.any()):
                continue
            idx = row[valid].long()
            logical = idx // page_size
            phys = block_table[b, logical].long()
            slot = idx % page_size
            for kvh in range(kv_heads):
                keys = cache_f[phys, 0, kvh, slot]
                values = cache_f[phys, 1, kvh, slot]
                for h in range(kvh * group, (kvh + 1) * group):
                    score = (q[b, h, s].float() @ keys.T) / head_dim**0.5
                    out[b, h, s] = (score.softmax(dim=-1) @ values).to(out.dtype)
    return out


def _make_inputs(
    batch,
    query_len,
    width,
    *,
    q_heads=12,
    kv_heads=1,
    head_dim=256,
    page_size=128,
    fp8=False,
    packed=False,
    fill=None,
    seed=0,
    max_blocks=None,
):
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tokens = kv_heads * page_size * 32
    cache_blocks = 64
    while cache_blocks * page_size < tokens:
        cache_blocks *= 2
    if max_blocks is not None:
        cache_blocks = max_blocks
    q = torch.randn(
        batch, q_heads, query_len, head_dim, device=_DEVICE, dtype=torch.bfloat16
    )
    if fp8:
        cache = (
            torch.randn(
                cache_blocks,
                2,
                kv_heads,
                page_size,
                head_dim,
                device=_DEVICE,
                dtype=torch.float32,
            )
            .clamp(-2.5, 2.5)
            .to(torch.float8_e4m3fn)
        )
        scale = torch.rand(
            cache_blocks, 2, kv_heads, page_size, device=_DEVICE, dtype=torch.float32
        ) * 0.5 + 0.5
    else:
        cache = torch.randn(
            cache_blocks,
            2,
            kv_heads,
            page_size,
            head_dim,
            device=_DEVICE,
            dtype=torch.bfloat16,
        )
        scale = None
    span = min((cache_blocks - 1) * page_size, query_len + 64)
    pages_needed = (span + page_size - 1) // page_size
    pool = torch.randperm(cache_blocks - 1, generator=gen)[:pages_needed] + 1
    block_table = (
        pool.to(torch.int32)
        .to(_DEVICE)
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    kv_lens = torch.full(
        (batch, query_len), span, dtype=torch.int32, device=_DEVICE
    )
    slots = width - 1 if packed else width
    selected = torch.full(
        (batch, query_len, width), -1, dtype=torch.int32, device=_DEVICE
    )
    if fill is None:
        fill = slots
    fill = min(fill, slots, span)
    for b in range(batch):
        for s in range(query_len):
            picks = torch.randperm(span, generator=gen)[:fill].to(torch.int32)
            selected[b, s, :fill] = picks.to(_DEVICE)
    if packed:
        selected[..., -1] = fill
    return q, cache, block_table, kv_lens, selected, (scale if fp8 else None)


def _run(q, cache, block_table, kv_lens, selected, page_size, kv_heads, scale,
         has_count=False, graph_capture=False):
    kwargs = {
        "page_size": page_size,
        "kv_head_num": kv_heads,
        "selected_has_count": has_count,
        "graph_capture": graph_capture,
    }
    if scale is not None:
        kwargs["kv_scale"] = scale
    return sparse_paged_gqa_attn(q, cache, block_table, kv_lens, selected, **kwargs)


class GroupedSplitDispatchTest(unittest.TestCase):
    def test_dispatch_reaches_grouped_kernel_on_sm90(self):
        q, cache, block_table, kv_lens, selected, scale = _make_inputs(
            1, 4, 131, packed=True, fp8=True
        )
        self.assertTrue(
            grouped_supported(
                q, cache, selected, page_size=128, selected_width=131
            )
        )

    def test_unsupported_geometry_stays_off(self):
        q, cache, block_table, kv_lens, selected, scale = _make_inputs(
            1, 4, 131, packed=True, fp8=True, q_heads=24, kv_heads=1
        )
        # group 24 -> BLOCK_M 32 allowed; push past the cap instead
        q2, cache2, _, _, selected2, _ = _make_inputs(
            1, 4, 131, q_heads=48, kv_heads=1
        )
        self.assertFalse(
            grouped_supported(q2, cache2, selected2, page_size=128, selected_width=131)
        )


class GroupedSplitBatchMatrixTest(unittest.TestCase):
    def _check(self, got, expected, context):
        torch.testing.assert_close(
            got, expected, atol=3e-2, rtol=3e-2, msg=lambda m: f"{context}: {m}"
        )

    def test_batch_matrix_bf16(self):
        for batch in (1, 2, 4, 8):
            for query_len in (1, 4, 128, 1024):
                with self.subTest(batch=batch, query_len=query_len):
                    q, cache, table, lens, sel, scale = _make_inputs(
                        batch, query_len, 131, seed=100 + batch * 10 + query_len
                    )
                    got = _run(q, cache, table, lens, sel, 128, 1, scale)
                    expected = _reference(q, cache, table, lens, sel, 128)
                    self._check(got, expected, f"bf16 B={batch} S={query_len}")

    def test_batch_matrix_fp8(self):
        for batch in (1, 2, 4, 8):
            for query_len in (1, 4, 128, 1024):
                with self.subTest(batch=batch, query_len=query_len):
                    q, cache, table, lens, sel, scale = _make_inputs(
                        batch,
                        query_len,
                        131,
                        fp8=True,
                        seed=200 + batch * 10 + query_len,
                    )
                    got = _run(q, cache, table, lens, sel, 128, 1, scale)
                    expected = _reference(q, cache, table, lens, sel, 128, scale)
                    self._check(got, expected, f"fp8 B={batch} S={query_len}")

    def test_batch_matrix_packed_counts(self):
        for batch in (1, 2, 4, 8):
            for fill in (0, 1, 37, 2051):
                with self.subTest(batch=batch, fill=fill):
                    q, cache, table, lens, sel, scale = _make_inputs(
                        batch,
                        4,
                        2052,
                        fp8=True,
                        packed=True,
                        fill=fill,
                        seed=300 + batch * 10 + fill,
                    )
                    got = _run(
                        q, cache, table, lens, sel, 128, 1, scale, has_count=True
                    )
                    row = sel[..., :-1]
                    expected = _reference(q, cache, table, lens, row, 128, scale)
                    self._check(got, expected, f"packed B={batch} fill={fill}")

    def test_kv_heads_two_group(self):
        for kv_heads in (2,):
            q, cache, table, lens, sel, scale = _make_inputs(
                2, 128, 131, kv_heads=kv_heads, fp8=True, seed=42
            )
            got = _run(q, cache, table, lens, sel, 128, kv_heads, scale)
            expected = _reference(q, cache, table, lens, sel, 128, scale)
            self._check(got, expected, f"kv_heads={kv_heads}")

    def test_large_prefill_shapes_sampled_rows(self):
        for query_len in (2048, 8192):
            with self.subTest(query_len=query_len):
                q, cache, table, lens, sel, scale = _make_inputs(
                    1, query_len, 2052, fp8=True, packed=True, fill=2051, seed=7
                )
                got = _run(
                    q, cache, table, lens, sel, 128, 1, scale, has_count=True
                )
                probes = [0, 1, query_len // 2, query_len - 1]
                sub_sel = sel[:, probes]
                sub_lens = lens[:, probes]
                sub_q = q[:, :, probes]
                expected = _reference(
                    sub_q, cache, table, sub_lens, sub_sel[..., :-1], 128, scale
                )
                self._check(
                    got[:, :, probes],
                    expected,
                    f"prefill S={query_len}",
                )

    def test_all_invalid_selection_rows_are_zero(self):
        q, cache, table, lens, sel, scale = _make_inputs(
            2, 8, 131, fp8=True, seed=5
        )
        sel[:, :4] = -1
        lens[:, :4] = 0
        got = _run(q, cache, table, lens, sel, 128, 1, scale)
        self.assertTrue(torch.equal(got[:, :, :4], torch.zeros_like(got[:, :, :4])))
        expected = _reference(q, cache, table, lens, sel, 128, scale)
        self._check(got, expected, "invalid rows")

    def test_cuda_graph_capture_and_replay(self):
        q, cache, table, lens, sel, scale = _make_inputs(
            1, 4, 2052, fp8=True, packed=True, fill=500, seed=11
        )
        stream = torch.cuda.Stream()
        inputs = [q, cache, table, lens, sel, scale]
        with torch.cuda.stream(stream):
            got = _run(q, cache, table, lens, sel, 128, 1, scale, has_count=True)
        stream.synchronize()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_out = _run(
                q,
                cache,
                table,
                lens,
                sel,
                128,
                1,
                scale,
                has_count=True,
                graph_capture=True,
            )
        # Refresh every live input in place and replay.
        torch.manual_seed(12)
        q2, cache2, table2, lens2, sel2, scale2 = _make_inputs(
            1, 4, 2052, fp8=True, packed=True, fill=900, seed=99
        )
        q.copy_(q2)
        cache.copy_(cache2)
        table.copy_(table2)
        lens.copy_(lens2)
        sel.copy_(sel2)
        scale.copy_(scale2)
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(
            q, cache, table, lens, sel[..., :-1], 128, scale
        )
        torch.testing.assert_close(
            replay_out,
            expected,
            atol=3e-2,
            rtol=3e-2,
            msg=lambda m: f"graph replay: {m}",
        )


class GroupedSplitOldNewParityTest(unittest.TestCase):
    """Direct old-vs-new comparison by gating the new path off."""

    def test_old_and_new_paths_agree(self):
        if _OVERRIDE is None:
            self.skipTest("needs QSA_OPT_MODULE_OVERRIDE to compare old vs new")
        for batch, query_len, fill in ((1, 128, 2051), (2, 8, 300), (4, 4, 130)):
            with self.subTest(batch=batch, query_len=query_len):
                q, cache, table, lens, sel, scale = _make_inputs(
                    batch, query_len, 2052, fp8=True, packed=True, fill=fill, seed=21
                )
                got_new = _candidate.sparse_paged_gqa_attn(
                    q,
                    cache,
                    table,
                    lens,
                    sel,
                    page_size=128,
                    kv_head_num=1,
                    kv_scale=scale,
                    selected_has_count=True,
                )
                got_old = _installed.sparse_paged_gqa_attn(
                    q,
                    cache,
                    table,
                    lens,
                    sel,
                    page_size=128,
                    kv_head_num=1,
                    kv_scale=scale,
                    selected_has_count=True,
                )
                torch.testing.assert_close(
                    got_new,
                    got_old,
                    atol=3e-2,
                    rtol=3e-2,
                    msg=lambda m: f"B={batch} S={query_len}: {m}",
                )


if __name__ == "__main__":
    unittest.main()
