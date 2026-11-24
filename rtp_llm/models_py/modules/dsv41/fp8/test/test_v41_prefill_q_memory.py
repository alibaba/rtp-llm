"""V4.1 streamed Q projection, RoPE numerical boundaries and scratch lifetime."""

import os
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41 import chunk_env, prefill_workspace
from rtp_llm.models_py.modules.dsv41.fp8 import attention_v41 as attention
from rtp_llm.models_py.modules.dsv41.fp8.attention import PrefillQKV
from rtp_llm.models_py.modules.dsv41.prefill_workspace import PrefillWorkspace


class PrefillQMemoryTest(unittest.TestCase):
    def test_short_metadata_fails_before_any_projection_or_mla(self):
        owner = SimpleNamespace(
            dim=5,
            _mega_attention=SimpleNamespace(
                supports_query=lambda _: True,
                supports_prefill=lambda *args: True,
            ),
            _project_prefill_q=Mock(),
            _prefill_output_all_reduce=Mock(),
        )
        qr = torch.zeros(9, 4, dtype=torch.bfloat16)
        workspace = PrefillWorkspace(
            torch.device("cpu"), q_rows=4, q_dim=16, reserve_cp=False, align_bytes=1
        )
        sources = (
            torch.ones(9, 2, dtype=torch.complex64),
            torch.zeros(9, 1, dtype=torch.int32),
            torch.ones(9, dtype=torch.int32),
        )
        mla = Mock()
        with patch.object(attention, "_FLASH_MLA_SPARSE_Q_CHUNK", 4), patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv41.flash_mla": SimpleNamespace(
                    flash_mla_sparse_fwd=mla
                )
            },
        ):
            for source in range(3):
                for rows in (0, 4, 8):
                    values = list(sources)
                    values[source] = values[source][:rows]
                    with self.subTest(source=source, rows=rows), self.assertRaises(
                        ValueError
                    ):
                        attention.AttentionV41FP8._prefill_sparse_attention(
                            owner,
                            SimpleNamespace(qr=qr),
                            SimpleNamespace(freqs_cis=values[0], workspace=workspace),
                            kv=torch.empty(1, 1, 8),
                            indices=values[1],
                            topk_length=values[2],
                            profile_name="test.short.metadata",
                        )
        owner._project_prefill_q.assert_not_called()
        owner._prefill_output_all_reduce.assert_not_called()
        mla.assert_not_called()

    def test_workspace_capacity_is_chunk_bounded_despite_stale_switch(self):
        with patch.object(chunk_env, "FLASH_MLA_SPARSE_Q_CHUNK", 4):
            for legacy_flag in (None, "0", "1"):
                with patch.dict(os.environ):
                    if legacy_flag is None:
                        os.environ.pop("DSV41_PREFILL_Q_CHUNKED", None)
                    else:
                        os.environ["DSV41_PREFILL_Q_CHUNKED"] = legacy_flag
                    for rows in (0, 1, 4, 5, 131072):
                        self.assertEqual(
                            prefill_workspace.prefill_q_workspace_rows(rows),
                            min(rows, 4),
                        )

    def _case(self, rows, reuse_output=False):
        torch.manual_seed(7)
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        owner.dim, owner.n_heads, owner.head_dim, owner.rope_head_dim = 5, 2, 8, 4
        owner.softmax_scale, owner.attn_sink = 0.5, torch.zeros(2)
        qr = torch.randn(rows, 4).bfloat16()
        wq = torch.randn(16, 4).bfloat16()
        wo = torch.randn(5, 16).bfloat16()
        events, pointers = [], []

        def project(x, *, out):
            events.append(("q", len(x)))
            pointers.append(out.data_ptr())
            return out.copy_(F.linear(x, wq))

        owner.wq_b = project
        owner._prefill_output_all_reduce = Mock()

        def output_project(o, freqs, *, out, workspace=None):
            events.append(("output", len(o)))
            attention.apply_rotary_emb(
                o[..., -owner.rope_head_dim :].unsqueeze(0), freqs, inverse=True
            )
            out.copy_(F.linear(o.flatten(1), wo))

        owner._prefill_output_proj_into = output_project
        owner._shared_attention = {}
        owner.wo_b = object()
        angles = torch.randn(rows * 2, owner.rope_head_dim // 2)
        freqs = torch.polar(torch.ones_like(angles), angles)[::2]
        indices = (torch.arange(rows) % 7).view(rows, 1, 1).int()
        lengths = (torch.arange(rows) % 3 != 1).int()
        kv = torch.randn(7, 1, 8).bfloat16()

        def mla(q, kv, indices, sm_scale=1.0, *, topk_length, **kwargs):
            events.append(("mla", len(q)))
            gathered = kv[indices[:, 0, 0].long()]
            return (
                (
                    q.float() + gathered.float() * topk_length[:, None, None].float()
                ).bfloat16(),
                torch.zeros(q.shape[:2]),
                torch.zeros(q.shape[:2]),
            )

        reference_q = F.linear(qr, wq).view(rows, 2, 8)
        attention.rope_only(reference_q, freqs, 4)
        reference_o = mla(q=reference_q, kv=kv, indices=indices, topk_length=lengths)[0]
        reference = torch.empty(rows, owner.dim, dtype=torch.bfloat16)
        output_project(reference_o, freqs, out=reference)
        events.clear()

        def mega_prefill(q, pool, idx, positions, lens):
            selected_freqs = freqs[positions.long()]
            attention.rope_only(q, selected_freqs, 4)
            o = mla(q=q, kv=pool, indices=idx, topk_length=lens)[0]
            return o, selected_freqs

        owner._mega_attention = SimpleNamespace(
            supports_query=lambda _: True,
            supports_prefill=lambda *args: True,
            project_q=lambda linear, value, out: linear(value, out=out).view(-1, 2, 8),
            prefill=mega_prefill,
            project_output=lambda pair, linear, out, prefill_wo_a: output_project(
                *pair, out=out
            ),
        )

        common = SimpleNamespace(
            freqs_cis=freqs,
            cp_on=False,
            position_ids=torch.arange(rows),
            workspace=PrefillWorkspace(
                torch.device("cpu"),
                q_rows=min(rows, 4),
                q_dim=16,
                reserve_cp=False,
                align_bytes=1,
            ),
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(attention, "_FLASH_MLA_SPARSE_Q_CHUNK", 4))
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "rtp_llm.models_py.modules.dsv41.flash_mla": SimpleNamespace(
                            flash_mla_sparse_fwd=mla
                        )
                    },
                )
            )
            output = (
                torch.empty(rows, owner.dim, dtype=torch.bfloat16)
                if reuse_output
                else None
            )
            result = owner._prefill_sparse_attention(
                PrefillQKV(qr=qr, q=None, kv_full=kv[:, 0]),
                common,
                kv=kv,
                indices=indices,
                topk_length=lengths,
                profile_name="test.v41.q",
                out=output,
            )
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        if reuse_output:
            self.assertIs(result, output)
        owner._prefill_output_all_reduce.assert_called_once_with(result)
        expected = []
        for start in range(0, rows, 4):
            count = min(4, rows - start)
            expected.extend((name, count) for name in ("q", "mla", "output"))
        self.assertEqual(events, expected)
        self.assertLessEqual(len(set(pointers)), 1)

    def test_chunked_matches_full_projection_including_padding_and_tail(self):
        for rows in (0, 1, 4, 5, 11):
            for legacy_flag in ("0", "1"):
                for reuse_output in (False, True):
                    with self.subTest(
                        rows=rows,
                        legacy_flag=legacy_flag,
                        reuse_output=reuse_output,
                    ), patch.dict(os.environ, {"DSV41_PREFILL_Q_CHUNKED": legacy_flag}):
                        self._case(rows, reuse_output=reuse_output)

    def test_prefill_query_reuses_dead_input_as_output_buffer(self):
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        owner.dim = 5
        owner.head_dim = 8
        owner.compress_ratio = 0
        owner._prefill_sparse_attention = Mock(return_value="output")
        x = torch.zeros(3, owner.dim, dtype=torch.bfloat16)
        qkv = PrefillQKV(
            qr=torch.zeros(3, 4, dtype=torch.bfloat16),
            q=None,
            kv_full=torch.zeros(3, owner.head_dim, dtype=torch.bfloat16),
        )
        common = SimpleNamespace(
            topk_idxs=torch.zeros(3, 1, 1, dtype=torch.int32),
            swa_meta=SimpleNamespace(
                topk_length_kv_full=torch.ones(3, dtype=torch.int32)
            ),
        )

        result = owner._prefill_query(x, common, qkv, None, None)

        self.assertEqual(result, "output")
        kwargs = owner._prefill_sparse_attention.call_args.kwargs
        self.assertIs(kwargs["out"], x)

    def test_prefill_output_buffer_requires_owned_contiguous_bf16_storage(self):
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        owner.dim = 5
        valid = torch.zeros(3, 5, dtype=torch.bfloat16)
        self.assertIs(owner._prefill_output_buffer(valid, 3), valid)
        self.assertIsNone(owner._prefill_output_buffer(valid.view(3, 5), 3))
        self.assertIsNone(owner._prefill_output_buffer(valid[:, 1:], 3))
        self.assertIsNone(owner._prefill_output_buffer(valid.float(), 3))
        self.assertIsNone(owner._prefill_output_buffer(valid[:2], 3))

    def test_cpu_rope_falls_back_and_empty_does_not_launch(self):
        x = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.bfloat16)
        freqs = torch.tensor([[1j]], dtype=torch.complex64)
        with patch.object(attention, "rope_only_inplace") as kernel:
            result = attention._prefill_q_rope(x, freqs, 2)
            self.assertIs(result, x)
            self.assertEqual(result.tolist(), [[[1, 2, -4, 3]]])
            empty = x[:0]
            self.assertIs(attention._prefill_q_rope(empty, freqs[:0], 2), empty)
        kernel.assert_not_called()

    def test_fp8_only_prefill_sparse_attention_slices_quantized_chunks(self):
        rows, chunk, q_rank = 10, 4, 128
        padded = (rows + 3) // 4 * 4
        q8 = torch.zeros(rows, q_rank, dtype=torch.float8_e4m3fn)
        sf = torch.empty_strided(
            (padded, q_rank // 128), (1, padded), dtype=torch.int32
        )[:rows]
        for r in range(rows):
            sf[r, 0] = r + 100
        seen = []

        def fake_project_q(linear, qr, qr_quantized=None, out=None):
            self.assertIsNone(qr)
            self.assertIsNotNone(qr_quantized)
            cq8, csf = qr_quantized
            seen.append((cq8.shape[0], csf.shape, csf.stride(), csf[:, 0].tolist()))
            out.zero_()
            return out.view(-1, 2, 8)

        mega = SimpleNamespace(
            supports_query=lambda _: False,
            supports_quantized_query=lambda pair: pair is not None,
            supports_prefill=lambda *args: True,
            project_q=fake_project_q,
            prefill=lambda q, kv, idx, pos, lengths: (q, None),
            project_output=lambda pair, linear, out, prefill_wo_a=None: out.zero_(),
        )
        owner = SimpleNamespace(
            dim=6,
            n_heads=2,
            head_dim=8,
            rope_head_dim=2,
            wq_b=object(),
            wo_b=object(),
            _mega_attention=mega,
            _mega_position_chunks=attention.AttentionV41FP8._mega_position_chunks.__get__(
                SimpleNamespace(_shared_attention={})
            ),
            _prefill_output_all_reduce=lambda out: None,
        )
        workspace = PrefillWorkspace(
            torch.device("cpu"), q_rows=chunk, q_dim=16, reserve_cp=False, align_bytes=1
        )
        common = SimpleNamespace(
            freqs_cis=torch.ones(rows, 1, dtype=torch.complex64),
            position_ids=torch.arange(rows, dtype=torch.int32),
            cp_on=False,
            workspace=workspace,
        )
        qkv = PrefillQKV(
            qr=None, q=None, kv_full=torch.zeros(rows, 8), qr_quantized=(q8, sf)
        )
        with patch.object(attention, "_FLASH_MLA_SPARSE_Q_CHUNK", chunk):
            out = attention.AttentionV41FP8._prefill_sparse_attention(
                owner,
                qkv,
                common,
                kv=torch.zeros(1, 1, 8),
                indices=torch.zeros(rows, 1, dtype=torch.int32),
                topk_length=torch.ones(rows, dtype=torch.int32),
                profile_name="test",
            )
        self.assertEqual(tuple(out.shape), (rows, 6))
        self.assertEqual(
            seen,
            [
                (4, (4, 1), (1, 12), [100, 101, 102, 103]),
                (4, (4, 1), (1, 12), [104, 105, 106, 107]),
                (2, (2, 1), (1, 12), [108, 109]),
            ],
        )

    def test_try_fused_qr_kv_uses_fp8_only_and_emits_norm_for_hot_hooks(self):
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        x = torch.zeros(1, 4, 16, dtype=torch.bfloat16)
        common = SimpleNamespace(workspace=SimpleNamespace(fp8_quant_kernel="auto"))
        captured = []

        def fake_proj(self_arg, x_arg, sq_arg, *, quant_kernel, emit_norm):
            captured.append((quant_kernel, emit_norm))
            return (None if not emit_norm else x_arg, x_arg, (x_arg, x_arg))

        with patch(
            "rtp_llm.models_py.modules.dsv41._v41_query_quant.try_project_prefill_quantized_qkv",
            side_effect=fake_proj,
        ):
            with patch(
                "rtp_llm.utils.hot_hook_runtime.runtime",
                return_value=SimpleNamespace(enabled=False),
            ):
                owner._try_fused_qr_kv(x, common=common)
            with patch(
                "rtp_llm.utils.hot_hook_runtime.runtime",
                return_value=SimpleNamespace(enabled=True),
            ):
                owner._try_fused_qr_kv(x, common=common)
        self.assertEqual(captured, [("auto", False), ("auto", True)])

    def test_ced_l20_forwards_compact_fp8_only_tma_layout(self):
        from contextlib import nullcontext

        from rtp_llm.models_py.modules.dsv41 import block as block_mod

        for keep_idx in ([0, 2, 4, 5, 8], [1, 3, 6, 7], []):
            with self.subTest(keep_len=len(keep_idx)):
                rows, dim, cols, packs = 10, 4, 1280, 10
                padded = (rows + 3) // 4 * 4
                q8 = (
                    (torch.arange(rows * cols) % 251)
                    .to(torch.uint8)
                    .view(rows, cols)
                    .view(torch.float8_e4m3fn)
                )
                sf = torch.empty_strided(
                    (padded, packs), (1, padded), dtype=torch.int32
                )[:rows]
                for r in range(rows):
                    for c in range(packs):
                        sf[r, c] = r * 100 + c
                idx = torch.tensor(keep_idx, dtype=torch.long)
                t_c = len(keep_idx)
                compact_q8 = q8.view(torch.uint8).index_select(0, idx).view(q8.dtype)
                stride = (t_c + 3) // 4 * 4 or 1
                compact_sf = torch.empty_strided(
                    (t_c, packs), (1, stride), dtype=sf.dtype
                )
                compact_sf.copy_(sf.index_select(0, idx))
                qkv = PrefillQKV(
                    qr=None,
                    q=None,
                    kv_full=torch.zeros(t_c, 8),
                    qr_quantized=(compact_q8, compact_sf),
                )
                seen_qkv = []
                common = SimpleNamespace(cp_ctx=None)
                replay_context = object()

                def produce(x, pos, *, l20_plan, l20_query_x):
                    self.assertIs(l20_plan.context, replay_context)
                    self.assertEqual(l20_query_x.shape[0], len(keep_idx))
                    return common, qkv, None, None, None

                attn = SimpleNamespace(
                    _shared_attention={},
                    freqs_cis=torch.ones(rows, 1, dtype=torch.complex64),
                    index_weights=torch.ones(2, dim, dtype=torch.bfloat16),
                    can_fuse_prefill_attn_norm_input_quant=lambda *args: False,
                    _set_compressor_pool_context=lambda: None,
                    _clear_compressor_pool_context=lambda: None,
                    _prefill_produce=produce,
                    _prefill_query=lambda x, meta, qkv_arg, swa, starts, *, indexer_weights=None: (
                        seen_qkv.append(qkv_arg)
                        or torch.zeros(len(keep_idx), dim, dtype=torch.bfloat16)
                    ),
                )
                exchange = SimpleNamespace(
                    send_sizes=[len(keep_idx)],
                    compact=lambda t, _idx=idx: t.index_select(0, _idx),
                )
                plan = SimpleNamespace(
                    early_indexer=False,
                    early_indexer_fallback=False,
                    original_context=SimpleNamespace(chunk_length=rows),
                    context=replay_context,
                    exchange=exchange,
                    compact_l20=lambda blk, r, p, c, ids: (
                        r.index_select(0, idx),
                        p,
                        c,
                        ids.index_select(0, idx),
                    ),
                    l20_query_metadata=lambda meta, freqs: meta,
                    router_projection=lambda ffn: nullcontext(),
                )
                norm = SimpleNamespace(
                    weight=SimpleNamespace(data=torch.ones(dim, dtype=torch.bfloat16)),
                    variance_epsilon=1e-6,
                    __call__=lambda t: t,
                )
                blk = SimpleNamespace(
                    attn=attn,
                    attn_norm=norm,
                    ffn_norm=norm,
                    ffn=lambda x, ids, expert_load=None: x,
                    _try_mega_mhc=lambda *a, **kw: None,
                    _sync_after_first_cp_prefill_attention=lambda: None,
                    _prefill_fast_hc_impls=lambda: (
                        lambda x: (x, None, None),
                        lambda x: (x, None, None),
                        lambda out, res, p, c: out,
                        lambda out, res, p, c: out,
                    ),
                )
                with (
                    patch.object(
                        block_mod, "_prefill_fast_norm", side_effect=lambda n, t: t
                    ),
                    patch(
                        "rtp_llm.models_py.modules.dsv41.fp8.attention.bind_attn_cache",
                        side_effect=lambda *a, **kw: nullcontext(),
                    ),
                ):
                    block_mod.Block.forward_prefill_ced_l20(
                        blk,
                        torch.zeros(rows, dim, dtype=torch.bfloat16),
                        torch.arange(rows, dtype=torch.int32),
                        torch.arange(rows, dtype=torch.int32),
                        plan,
                        None,
                    )
                self.assertEqual(len(seen_qkv), 1)
                compacted = seen_qkv[0]
                self.assertIs(compacted, qkv)
                self.assertIsNone(compacted.qr)
                self.assertIsNotNone(compacted.qr_quantized)
                c_q8, c_sf = compacted.qr_quantized
                t_c = len(keep_idx)
                expected_padded = (t_c + 3) // 4 * 4
                self.assertEqual(c_q8.dtype, torch.float8_e4m3fn)
                self.assertEqual(tuple(c_q8.shape), (t_c, cols))
                torch.testing.assert_close(
                    c_q8.view(torch.uint8),
                    q8.view(torch.uint8).index_select(0, idx),
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(tuple(c_sf.shape), (t_c, packs))
                self.assertEqual(c_sf.stride(), (1, expected_padded or 1))
                torch.testing.assert_close(
                    c_sf.contiguous(),
                    sf.contiguous().index_select(0, idx),
                    rtol=0,
                    atol=0,
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillQRopeCudaTest(unittest.TestCase):
    def test_inplace_rope_matches_bf16_eager_with_strided_frequencies(self):
        torch.manual_seed(9)
        for rows, heads in ((0, 128), (1, 128), (17, 7), (257, 128)):
            for strided in (False, True):
                with self.subTest(rows=rows, heads=heads, strided=strided):
                    base = torch.randn(
                        rows, heads, 512, device="cuda", dtype=torch.bfloat16
                    )
                    angles = torch.randn(rows * 2, 64, device="cuda")
                    freqs = torch.polar(torch.ones_like(angles), angles)[::2, ::2]
                    if not strided:
                        freqs = freqs.contiguous()
                    expected = base.clone()
                    # Keep the eager algorithm as the independent test oracle.
                    if rows:
                        attention.rope_only(expected, freqs, 64)
                    actual = base.clone()
                    pointer = actual.data_ptr()
                    # The removed A/B switch must not disable the in-place path.
                    with patch.dict(os.environ, {"DSV41_PREFILL_Q_ROPE_INPLACE": "0"}):
                        result = attention._prefill_q_rope(actual, freqs, 64)
                    self.assertIs(result, actual)
                    self.assertEqual(actual.data_ptr(), pointer)
                    torch.testing.assert_close(
                        actual[..., :-64], base[..., :-64], rtol=0, atol=0
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_mxfp8_q_projection_is_identical_across_chunk_boundaries(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MXFP8 Q projection requires SM100")
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        torch.manual_seed(23)
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        owner.n_heads, owner.head_dim, owner.rope_head_dim = 64, 512, 64
        q_dim, q_rank = owner.n_heads * owner.head_dim, 1280
        weight = (torch.randn(q_dim, q_rank, device="cuda") * 0.02).to(
            torch.float8_e4m3fn
        )
        scales = torch.ones(q_dim // 32, q_rank // 32, device="cuda").to(
            torch.float8_e8m0fnu
        )
        owner.wq_b = V41MXFP8Linear(weight, scales)
        # Include an unaligned final tile and padded zero rows from two CP
        # request halves. The latter still have their own RoPE positions.
        for rows, chunk_rows in ((257, 128), (1025, 384)):
            with self.subTest(rows=rows, chunk_rows=chunk_rows):
                qr = torch.randn(rows, q_rank, device="cuda", dtype=torch.bfloat16)
                qr[121:128].zero_()
                qr[-3:].zero_()
                angles = torch.randn(rows * 2, 32, device="cuda")
                freqs = torch.polar(torch.ones_like(angles), angles)[::2]
                expected = owner._lin(owner.wq_b, qr).view(rows, 64, 512)
                attention.rope_only(expected, freqs, 64)
                workspace = PrefillWorkspace(
                    torch.device("cuda"),
                    q_rows=chunk_rows,
                    q_dim=q_dim,
                    reserve_cp=False,
                    align_bytes=1,
                )
                with patch.dict(os.environ, {"DSV41_PREFILL_Q_ROPE_INPLACE": "1"}):
                    for start in range(0, rows, chunk_rows):
                        end = min(start + chunk_rows, rows)
                        actual = owner._project_prefill_q(
                            qr[start:end], freqs[start:end], workspace
                        )
                        torch.testing.assert_close(
                            actual, expected[start:end], rtol=0, atol=0
                        )

    def test_rope_does_not_allocate_a_q_sized_temporary(self):
        q = torch.randn(1025, 128, 512, device="cuda", dtype=torch.bfloat16)
        angles = torch.randn(1025, 32, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        with patch.dict(os.environ, {"DSV41_PREFILL_Q_ROPE_INPLACE": "1"}):
            attention._prefill_q_rope(q, freqs, 64)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            allocated = torch.cuda.memory_allocated()
            attention._prefill_q_rope(q, freqs, 64)
            torch.cuda.synchronize()
            self.assertLess(torch.cuda.max_memory_allocated() - allocated, 1 << 20)

    def test_fp8_only_chunked_q_projection_and_indexer_bit_exact(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MXFP8 prefill Q projection requires SM10x")
        from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import strided_q_rmsnorm
        from rtp_llm.models_py.modules.dsv41._v41_query_quant import query_norm_quant
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        torch.manual_seed(31)
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        owner.n_heads, owner.head_dim, owner.rope_head_dim = 64, 512, 64
        q_dim, q_rank = owner.n_heads * owner.head_dim, 1280
        owner.wq_b = V41MXFP8Linear(
            (torch.randn(q_dim, q_rank, device="cuda") * 0.02).to(torch.float8_e4m3fn),
            torch.ones(q_dim // 32, q_rank // 32, device="cuda").to(
                torch.float8_e8m0fnu
            ),
        )
        owner.index_wq = V41MXFP8Linear(
            (torch.randn(1024, q_rank, device="cuda") * 0.02).to(torch.float8_e4m3fn),
            torch.ones(1024 // 32, q_rank // 32, device="cuda").to(
                torch.float8_e8m0fnu
            ),
        )
        q_norm = torch.randn(q_rank, device="cuda", dtype=torch.bfloat16)
        for rows, chunk_rows in ((257, 128), (1025, 384)):
            with self.subTest(rows=rows, chunk_rows=chunk_rows):
                raw = torch.randn(1, rows, 1792, device="cuda", dtype=torch.bfloat16)
                x_q = raw[..., :q_rank]
                qr_ref = strided_q_rmsnorm(x_q, q_norm, 1e-6).squeeze(0)
                qr_none, (q8_3d, sf) = query_norm_quant(
                    x_q, q_norm, 1e-6, emit_norm=False
                )
                self.assertIsNone(qr_none)
                q8 = q8_3d.squeeze(0)
                # Verify fp8_only working set is strictly smaller than BF16 qr.
                self.assertLess(
                    q8.numel() * q8.element_size() + sf.numel() * sf.element_size(),
                    qr_ref.numel() * qr_ref.element_size(),
                )
                # Indexer query projection bit-exactness.
                torch.testing.assert_close(
                    owner.index_wq.forward_quantized(q8, sf),
                    owner.index_wq(qr_ref),
                    rtol=0,
                    atol=0,
                )
                # Chunked Q projection + RoPE bit-exactness across unaligned tail.
                angles = torch.randn(rows, 32, device="cuda")
                freqs = torch.polar(torch.ones_like(angles), angles)
                expected = owner._lin(owner.wq_b, qr_ref).view(rows, 64, 512)
                attention.rope_only(expected, freqs, 64)
                workspace = PrefillWorkspace(
                    torch.device("cuda"),
                    q_rows=chunk_rows,
                    q_dim=q_dim,
                    reserve_cp=False,
                    align_bytes=1,
                )
                for start in range(0, rows, chunk_rows):
                    end = min(start + chunk_rows, rows)
                    actual = owner._project_prefill_q(
                        None,
                        freqs[start:end],
                        workspace,
                        qr_quantized=(q8[start:end], sf[start:end]),
                    )
                    torch.testing.assert_close(
                        actual, expected[start:end], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
