import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill import (
    try_causal_qsa_prefill,
    try_paged_causal_qsa_prefill,
    try_ragged_causal_qsa_prefill,
    try_ragged_paged_qsa_prefill,
)


def make_selection(tokens):
    lengths = torch.arange(1, tokens + 1, device="cuda")
    columns = (tokens + 3) // 4
    scores = torch.randn(tokens, columns, device="cuda")
    scores.masked_fill_(
        torch.arange(columns, device="cuda")[None, :] >= lengths[:, None] // 4,
        -torch.inf,
    )
    values, blocks = scores.topk(min(512, columns), dim=-1)
    indices = blocks[:, :, None] * 4 + torch.arange(4, device="cuda")
    indices.masked_fill_(~torch.isfinite(values)[:, :, None], -1)
    selected = torch.full((1, tokens, 2051), -1, device="cuda", dtype=torch.int32)
    selected[0, :, : indices.numel() // tokens] = indices.reshape(tokens, -1)
    tail = torch.arange(3, device="cuda")[None, :]
    selected[0, :, -3:] = torch.where(
        tail < (lengths % 4)[:, None], lengths[:, None] // 4 * 4 + tail, -1
    )
    return selected


def make_inputs(tokens, heads):
    packed = torch.randn(tokens, (heads + 2) * 256, dtype=torch.bfloat16, device="cuda")
    q, k, v = packed.split([heads * 256, 256, 256], dim=-1)
    q = q.reshape(1, tokens, heads, 256).transpose(1, 2)
    k = k.reshape(1, tokens, 1, 256).transpose(1, 2)
    v = v.reshape(1, tokens, 1, 256).transpose(1, 2)
    return packed, q, k, v


def reference(q, k, v, selected):
    tokens = q.shape[2]
    indices = selected[0].long()
    membership = torch.zeros(
        tokens, tokens, dtype=torch.int32, device=q.device
    ).scatter_add_(1, indices.clamp_min(0), (indices >= 0).int())
    scores = q.double() @ k.double().transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores += membership.clamp_min(1).double().log()[None, None]
    scores.masked_fill_(membership[None, None] == 0, -torch.inf)
    probability = scores.softmax(-1).nan_to_num(0)
    return (probability @ v.double()).to(q.dtype)


class TmaPrefillTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            self.skipTest("SM120 required")
        torch.manual_seed(471)

    def test_packed_qkv_matches_fp64_with_partial_tiles_holes_and_empty_rows(self):
        for heads, tokens in ((6, 257), (12, 257), (6, 513), (12, 513)):
            with self.subTest(heads=heads, tokens=tokens):
                _, q, k, v = make_inputs(tokens, heads)
                selected = make_selection(tokens)
                selected[:, [0, 13, tokens - 1]] = -1
                selected[:, 16:, :4] = -1
                actual = try_causal_qsa_prefill(q, k, v, selected)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(
                    actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
                )
                self.assertTrue((actual[:, :, [0, 13, tokens - 1]] == 0).all())
                packed_output = actual.transpose(1, 2).reshape(tokens, heads * 256)
                self.assertTrue(packed_output.is_contiguous())
                self.assertEqual(packed_output.data_ptr(), actual.data_ptr())

    def test_short_requests_match_fp64_and_refresh_graph_inputs(self):
        for heads in (6, 12):
            for tokens in (1, 2, 3, 7, 17, 32, 67, 127):
                with self.subTest(heads=heads, tokens=tokens):
                    packed, q, k, v = make_inputs(tokens, heads)
                    selected = make_selection(tokens)
                    actual = try_causal_qsa_prefill(q, k, v, selected)
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(
                        actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
                    )
                    initial = selected.clone()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = try_causal_qsa_prefill(q, k, v, selected)
                    selected.fill_(-1)
                    graph.replay()
                    self.assertTrue((actual == 0).all())
                    selected.copy_(initial)
                    packed.copy_(torch.randn_like(packed))
                    graph.replay()
                    torch.testing.assert_close(
                        actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
                    )

    @staticmethod
    def _paged_case(heads, prefix, tokens, fp8):
        page = 128
        total = prefix + tokens
        pages = (total + page - 1) // page + 5
        _, q, _, _ = make_inputs(tokens, heads)
        table = torch.zeros(
            1, (total + page - 1) // page + 2, device=q.device, dtype=torch.int32
        )
        table[0, : (total + page - 1) // page] = (
            torch.randperm(pages - 1, device=q.device) + 1
        )[: (total + page - 1) // page].int()
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        backing = torch.full(
            (pages, (2 * page + 32) * 256),
            torch.nan,
            device=q.device,
            dtype=torch.bfloat16,
        )
        cache = torch.as_strided(
            backing,
            (pages, 2, 1, page, 256),
            (backing.stride(0), page * 256, page * 256, 256, 1),
        )
        positions = torch.arange(total, device=q.device)
        physical, offsets = table[0, positions // page].long(), positions % page
        cache[physical, 0, 0, offsets] = torch.randn(
            total, 256, device=q.device, dtype=torch.bfloat16
        )
        cache[physical, 1, 0, offsets] = (
            torch.randn(total, 256, device=q.device, dtype=torch.bfloat16) * 2
        )
        if fp8:
            backing = backing.to(dtype)
            cache = torch.as_strided(backing, cache.shape, cache.stride())
        scale_backing = torch.full(
            (pages, 2 * page + 16), torch.nan, device=q.device, dtype=torch.float32
        )
        scale = (
            torch.as_strided(
                scale_backing,
                (pages, 2, 1, page),
                (scale_backing.stride(0), page, page, 1),
            )
            if fp8
            else None
        )
        if fp8:
            scale[physical, 0, 0, offsets] = (
                0.125 + torch.rand(total, device=q.device) * 1.875
            )
            scale[physical, 1, 0, offsets] = (
                0.25 + torch.rand(total, device=q.device) * 3.5
            ) * torch.where(positions % 3 == 0, -1.0, 1.0)
        selected = make_selection(total)[:, prefix:].contiguous()
        return q, backing, cache, table, selected, scale

    @staticmethod
    def _paged_reference(q, cache, table, selected, prefix, scale):
        positions = torch.arange(prefix + q.shape[2], device=q.device)
        physical, offsets = table[0, positions // 128].long(), positions % 128
        source = cache.float()
        k = source[physical, 0, 0, offsets]
        v = source[physical, 1, 0, offsets]
        if scale is not None:
            k = k * scale[physical, 0, 0, offsets, None]
            v = v * scale[physical, 1, 0, offsets, None]
        membership = torch.zeros(
            q.shape[2], positions.numel(), device=q.device, dtype=torch.int32
        ).scatter_add_(1, selected[0].long().clamp_min(0), (selected[0] >= 0).int())
        scores = q.double() @ k.double().T / 16
        scores.masked_fill_(membership[None, None] == 0, -torch.inf)
        return (scores.softmax(-1).nan_to_num(0) @ v.double()).to(q.dtype)

    def test_paged_cache_padding_scales_and_graph_updates(self):
        for heads in (6, 12):
            for fp8 in (False, True):
                for prefix, tokens in (
                    (127, 129),
                    (1024, 129),
                    (3072, 257),
                    (1024, 513),
                    (3072, 1024),
                ):
                    with self.subTest(
                        heads=heads, fp8=fp8, prefix=prefix, tokens=tokens
                    ):
                        q, backing, cache, table, selected, scale = self._paged_case(
                            heads, prefix, tokens, fp8
                        )
                        selected[:, [0, 13, tokens - 1]] = -1
                        selected[(selected >= 0) & (selected < 32)] = -1
                        kwargs = dict(
                            prefix=prefix, page_size=128, kv_head_num=1, kv_scale=scale
                        )
                        # Both native HybridPool rows and HND views retain their padding.
                        for storage in (backing, cache):
                            actual = try_paged_causal_qsa_prefill(
                                q, storage, table, selected, **kwargs
                            )
                            self.assertIsNotNone(actual)
                            torch.testing.assert_close(
                                actual,
                                self._paged_reference(
                                    q, cache, table, selected, prefix, scale
                                ),
                                atol=0.0078125,
                                rtol=0.008,
                            )
                            self.assertTrue(
                                (actual[:, :, [0, 13, tokens - 1]] == 0).all()
                            )
                            packed = actual.transpose(1, 2).reshape(tokens, heads * 256)
                            self.assertEqual(packed.data_ptr(), actual.data_ptr())
                        initial = selected.clone()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = try_paged_causal_qsa_prefill(
                                q, backing, table, selected, **kwargs
                            )
                        selected.fill_(-1)
                        graph.replay()
                        self.assertTrue((actual == 0).all())
                        selected.copy_(initial)
                        table[0, :2].copy_(table[0, :2].flip(0))
                        q.copy_(torch.randn_like(q))
                        if fp8:
                            scale[:, 1].mul_(2)
                        graph.replay()
                        torch.testing.assert_close(
                            actual,
                            self._paged_reference(
                                q, cache, table, selected, prefix, scale
                            ),
                            atol=0.0078125,
                            rtol=0.008,
                        )

    def test_paged_value_and_scale_ranges_preserve_bf16_exponents(self):
        for heads in (6, 12):
            for fp8 in (False, True):
                with self.subTest(heads=heads, fp8=fp8):
                    q, backing, cache, table, selected, scale = self._paged_case(
                        heads, 128, 129, fp8
                    )
                    kwargs = dict(
                        prefix=128, page_size=128, kv_head_num=1, kv_scale=scale
                    )
                    unscaled = try_paged_causal_qsa_prefill(
                        q, backing, table, selected, **kwargs
                    )
                    if fp8:
                        scale[:, 1].mul_(2**17)
                    else:
                        cache[:, 1].mul_(2**17)
                    actual = try_paged_causal_qsa_prefill(
                        q, backing, table, selected, **kwargs
                    )
                    torch.testing.assert_close(actual, unscaled * 2**17, atol=0, rtol=0)
                    expected = self._paged_reference(
                        q, cache, table, selected, 128, scale
                    )
                    torch.testing.assert_close(
                        actual / 2**17, expected / 2**17, atol=0.0078125, rtol=0.008
                    )

    @classmethod
    def _paged_batch_case(cls, heads, lengths, prefixes, fp8):
        cases = [cls._paged_case(heads, p, s, fp8) for p, s in zip(prefixes, lengths)]
        tokens = sum(lengths)
        packed = torch.empty(
            tokens, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
        )
        q = packed[:, : heads * 256]
        q.copy_(
            torch.cat(
                [c[0].transpose(1, 2).reshape(s, -1) for c, s in zip(cases, lengths)]
            )
        )
        backing = torch.cat([c[1].float() for c in cases]).to(cases[0][1].dtype)
        cache = torch.as_strided(
            backing,
            (backing.shape[0], 2, 1, 128, 256),
            (backing.stride(0), 128 * 256, 128 * 256, 256, 1),
        )
        width = max(c[3].shape[1] for c in cases)
        table = torch.zeros(len(cases), width + 3, device=q.device, dtype=torch.int32)[
            :, :width
        ]
        offset = 0
        for row, c in enumerate(cases):
            table[row, : c[3].shape[1]] = torch.where(c[3][0] > 0, c[3][0] + offset, 0)
            offset += c[1].shape[0]
        scale = None
        if fp8:
            scale_backing = torch.full(
                (backing.shape[0], 272), torch.nan, device=q.device
            )
            scale = torch.as_strided(
                scale_backing, (backing.shape[0], 2, 1, 128), (272, 128, 128, 1)
            )
            scale.copy_(torch.cat([c[5] for c in cases]))
        ids = torch.cat([c[4][0] for c in cases])
        cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            device=q.device,
            dtype=torch.int32,
        )
        pd = torch.tensor(prefixes, device=q.device, dtype=torch.int32)
        return q, backing, cache, table, ids, scale, cu, pd

    @classmethod
    def _paged_batch_reference(cls, q, cache, table, ids, lengths, prefixes, scale):
        outputs = []
        start = 0
        for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
            end = start + length
            query = q[start:end].view(1, length, -1, 256).transpose(1, 2)
            value = cls._paged_reference(
                query, cache, table[row : row + 1], ids[None, start:end], prefix, scale
            )
            outputs.append(value.transpose(1, 2).reshape(length, -1))
            start = end
        return torch.cat(outputs)

    def test_batched_paged_refreshes_partition_prefixes_and_cache_inputs(self):
        for heads in (6, 12):
            for fp8 in (False, True):
                with self.subTest(heads=heads, fp8=fp8):
                    lengths, prefixes = [129, 257, 356], [127, 128, 384]
                    q, backing, cache, table, ids, scale, cu, pd = (
                        self._paged_batch_case(heads, lengths, prefixes, fp8)
                    )
                    ids[[0, 13, 129, 742 - 1]] = -1
                    ids[(ids >= 0) & (ids < 32)] = -1
                    original = ids.clone()

                    def run():
                        return try_ragged_paged_qsa_prefill(
                            q,
                            backing,
                            table,
                            ids,
                            lengths,
                            prefixes,
                            cu,
                            pd,
                            page_size=128,
                            kv_head_num=1,
                            kv_scale=scale,
                        )

                    def expected(active_lengths=lengths, active_prefixes=prefixes):
                        return self._paged_batch_reference(
                            q, cache, table, ids, active_lengths, active_prefixes, scale
                        )

                    actual = run()
                    self.assertIsNotNone(actual)
                    self.assertTrue(actual.is_contiguous())
                    torch.testing.assert_close(
                        actual, expected(), atol=0.0078125, rtol=0.008
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    ids.fill_(-1)
                    graph.replay()
                    self.assertTrue((actual == 0).all())
                    ids.copy_(original)
                    q.mul_(0.75)
                    if scale is not None:
                        scale[:, 1].mul_(-0.5)
                    table[:, :2].copy_(table[:, :2].flip(1))
                    graph.replay()
                    torch.testing.assert_close(
                        actual, expected(), atol=0.0078125, rtol=0.008
                    )
                    new_lengths, new_prefixes = lengths[::-1], prefixes[::-1]
                    cu.copy_(
                        torch.tensor(
                            [0, 356, 613, 742], device=q.device, dtype=torch.int32
                        )
                    )
                    pd.copy_(pd.flip(0))
                    table.copy_(table.flip(0))
                    ids.copy_(
                        torch.cat(
                            [
                                make_selection(p + s)[0, p:]
                                for p, s in zip(new_prefixes, new_lengths)
                            ]
                        )
                    )
                    for _ in range(20):
                        graph.replay()
                    torch.testing.assert_close(
                        actual,
                        expected(new_lengths, new_prefixes),
                        atol=0.0078125,
                        rtol=0.008,
                    )

    def test_batched_paged_guards_keep_unsupported_metadata_on_general_path(self):
        q, backing, _, table, ids, scale, cu, pd = self._paged_batch_case(
            12, [129, 257], [128, 128], True
        )

        def run(**changes):
            args = dict(
                q=q,
                cache=backing,
                table=table,
                selected=ids,
                lengths=[129, 257],
                prefixes=[128, 128],
                cu_seqlens=cu,
                prefixes_device=pd,
                page_size=128,
                kv_head_num=1,
                kv_scale=scale,
            )
            args.update(changes)
            return try_ragged_paged_qsa_prefill(**args)

        for changes in (
            dict(cu_seqlens=None),
            dict(cu_seqlens=cu.long()),
            dict(prefixes=[None, 128]),
            dict(prefixes_device=pd.long()),
            dict(kv_scale=scale.bfloat16()),
            dict(lengths=[127, 259]),
            dict(prefixes=[4096, 128]),
            dict(kv_head_num=2),
        ):
            with self.subTest(changes=list(changes)):
                self.assertIsNone(run(**changes))

    def test_batched_paged_runtime_requires_trusted_selection_and_plan(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_gqa_impl
        from rtp_llm.models_py.modules.qwen4_exp.qsa.fmha_plans import (
            IncrementalPrefillInputs,
            PrefillRequest,
        )

        q, backing, cache, table, ids, scale, cu, pd = self._paged_batch_case(
            12, [129, 257], [128, 128], True
        )
        requests = [
            PrefillRequest(
                a,
                b,
                (128 + torch.arange(1, b - a + 1, device=q.device, dtype=torch.int32))[
                    None
                ],
                ids[None, a:b],
                128,
            )
            for a, b in ((0, 129), (129, 386))
        ]
        plan = IncrementalPrefillInputs(2, 386, table, 128, requests, ids, pd, cu)
        impl = object.__new__(sparse_gqa_impl.SparseGqaFmhaImpl)
        impl.head_num, impl.kv_head_num, impl.head_dim, impl.q_width = 12, 1, 256, 3072
        impl._is_mtp_draft = False
        impl.attn_configs = SimpleNamespace(
            indexer_topk=512, kernel_tokens_per_block=128
        )
        layer = SimpleNamespace(kv_cache_base=backing, kv_scale_base=scale)
        name = "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_ragged_paged_qsa_prefill"
        with patch(name, wraps=try_ragged_paged_qsa_prefill) as fast:
            actual = impl._forward_ragged_paged(q, layer, plan, trusted_selected=True)
            fast.assert_called_once()
            self.assertEqual(fast.call_args.args[0].data_ptr(), q.data_ptr())
            self.assertIs(fast.call_args.args[6], cu)
        expected = self._paged_batch_reference(
            q, cache, table, ids, [129, 257], [128, 128], scale
        )
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)
        ids[:, 4:8] = ids[:, :4]
        with patch(
            name, side_effect=AssertionError("injected selection reached batched TMA")
        ):
            actual = impl._forward_ragged_paged(q, layer, plan)
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )

        outputs = []
        for row, request in enumerate(requests):
            query = (
                q[request.start : request.end]
                .view(1, request.end - request.start, 12, 256)
                .transpose(1, 2)
                .contiguous()
            )
            value = sparse_paged_gqa_attn(
                query,
                backing,
                table[row : row + 1].contiguous(),
                request.kv_lens,
                request.selected_indices,
                page_size=128,
                kv_head_num=1,
                kv_scale=scale,
            )
            outputs.append(
                value.transpose(1, 2).reshape(request.end - request.start, -1)
            )
        torch.testing.assert_close(actual, torch.cat(outputs), atol=0, rtol=0)

    def test_long_prefix_fp8_value_scales_preserve_bf16_range(self):
        q, backing, cache, table, selected, scale = self._paged_case(
            12, 1024, 129, True
        )
        kwargs = dict(prefix=1024, page_size=128, kv_head_num=1, kv_scale=scale)
        original = try_paged_causal_qsa_prefill(q, backing, table, selected, **kwargs)
        scale[:, 1].mul_(2**17)
        actual = try_paged_causal_qsa_prefill(q, backing, table, selected, **kwargs)
        torch.testing.assert_close(actual, original * (2**17), atol=0, rtol=0)
        torch.testing.assert_close(
            actual,
            self._paged_reference(q, cache, table, selected, 1024, scale),
            atol=1024,
            rtol=0.008,
        )

    def test_paged_runtime_preserves_selection_provenance(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_gqa_impl
        from rtp_llm.models_py.modules.qwen4_exp.qsa.fmha_plans import (
            IncrementalPrefillInputs,
            PrefillRequest,
        )

        q, backing, cache, table, selected, scale = self._paged_case(12, 128, 129, True)
        impl = object.__new__(sparse_gqa_impl.SparseGqaFmhaImpl)
        impl.head_num, impl.kv_head_num, impl.head_dim, impl.q_width = (
            12,
            1,
            256,
            12 * 256,
        )
        impl._is_mtp_draft = False
        impl.attn_configs = SimpleNamespace(
            indexer_topk=512, kernel_tokens_per_block=128
        )
        lengths = (128 + torch.arange(1, 130, device=q.device, dtype=torch.int32))[None]
        plan = IncrementalPrefillInputs(
            1,
            129,
            table,
            128,
            [PrefillRequest(0, 129, lengths, selected, 128)],
            selected[0],
            torch.tensor([128], device=q.device),
        )
        layer = SimpleNamespace(kv_cache_base=backing, kv_scale_base=scale)
        packed = q.transpose(1, 2).reshape(129, -1)
        expected = (
            self._paged_reference(q, cache, table, selected, 128, scale)
            .transpose(1, 2)
            .reshape(129, -1)
        )
        name = "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_paged_causal_qsa_prefill"
        with patch(name, wraps=try_paged_causal_qsa_prefill) as fast:
            actual = impl._forward_ragged_paged(
                packed, layer, plan, trusted_selected=True
            )
            fast.assert_called_once()
            self.assertEqual(
                fast.call_args.args[0].untyped_storage().data_ptr(),
                packed.untyped_storage().data_ptr(),
            )
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)
        # Injected duplicate indices must reach the general multiset attention.
        selected[..., 4:8] = selected[..., :4]
        with patch(name, side_effect=AssertionError("injected selection reached TMA")):
            actual = impl._forward_ragged_paged(packed, layer, plan)
        from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_paged_fmha import (
            sparse_paged_gqa_attn,
        )

        expected = (
            sparse_paged_gqa_attn(
                q.contiguous(),
                backing,
                table,
                lengths,
                selected,
                page_size=128,
                kv_head_num=1,
                kv_scale=scale,
            )
            .transpose(1, 2)
            .reshape(129, -1)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        impl.attn_configs.indexer_topk = 1025
        with patch(
            name, side_effect=AssertionError("different compression reached TMA")
        ):
            impl._forward_ragged_paged(packed, layer, plan, trusted_selected=True)
        self.assertIsNone(
            try_paged_causal_qsa_prefill(
                q,
                backing,
                table,
                selected,
                prefix=None,
                page_size=128,
                kv_head_num=1,
                kv_scale=scale,
            )
        )

    def test_bf16_value_range_is_preserved(self):
        _, q, k, v = make_inputs(129, 12)
        selected = make_selection(129)
        unscaled = try_causal_qsa_prefill(q, k, v, selected)
        scale = 2**17
        v.mul_(scale)
        actual = try_causal_qsa_prefill(q, k, v, selected)
        torch.testing.assert_close(actual, unscaled * scale, atol=0, rtol=0)
        # Compare in the original units so the absolute error has the same scale.
        torch.testing.assert_close(
            actual / scale,
            reference(q, k, v, selected) / scale,
            atol=0.0078125,
            rtol=0.008,
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_runtime_uses_packed_views_only_for_indexer_owned_selections(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_gqa_impl

        tokens, heads = 257, 12
        packed, q, k, v = make_inputs(tokens, heads)
        selected = make_selection(tokens)
        configs = SimpleNamespace(
            head_num=heads,
            kv_head_num=1,
            size_per_head=256,
            kernel_tokens_per_block=128,
            indexer_topk=512,
        )
        inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            input_lengths=torch.tensor([tokens], dtype=torch.int32),
            prefix_lengths=torch.zeros(1, dtype=torch.int32),
            sequence_lengths=torch.zeros(1, dtype=torch.int32),
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            kv_cache_kernel_block_id_device=None,
        )
        writer = SimpleNamespace(
            prepare=lambda _: object(), forward=lambda qkv, *args: qkv
        )
        with patch.object(
            sparse_gqa_impl, "FusedRopeKVCachePrefillOpQKVOut", return_value=writer
        ):
            impl = sparse_gqa_impl.SparseGqaFmhaImpl(configs, inputs)
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_causal_qsa_prefill",
            wraps=try_causal_qsa_prefill,
        ) as fast:
            impl.set_trusted_selected_indices(selected[0])
            actual = impl.forward(packed, None)
            fast.assert_called_once()
            for tensor in fast.call_args.args[:3]:
                self.assertEqual(
                    tensor.untyped_storage().data_ptr(),
                    packed.untyped_storage().data_ptr(),
                )
        expected = reference(q, k, v, selected).transpose(1, 2).reshape(tokens, -1)
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)

        # A different compression ratio can produce the same selection width.
        configs.indexer_topk = 1025
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_causal_qsa_prefill",
            side_effect=AssertionError("different block layout reached TMA"),
        ):
            impl.set_trusted_selected_indices(selected[0])
            actual = impl.forward(packed, None)
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)
        configs.indexer_topk = 512

        # Injected selections may repeat keys and must retain multiplicity.
        selected[..., 4:8] = selected[..., :4]
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_causal_qsa_prefill",
            side_effect=AssertionError("untrusted selection reached TMA"),
        ):
            actual = impl.forward(packed, None, selected_indices=selected[0])
        expected = reference(q, k, v, selected).transpose(1, 2).reshape(tokens, -1)
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)

        # Ragged requests keep their own local key indices in the packed batch.
        inputs.input_lengths = torch.tensor([129, 128], dtype=torch.int32)
        inputs.prefix_lengths = torch.zeros(2, dtype=torch.int32)
        inputs.sequence_lengths = torch.zeros(2, dtype=torch.int32)
        selections = [make_selection(129), make_selection(128)]
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_causal_qsa_prefill",
            wraps=try_causal_qsa_prefill,
        ) as fast:
            impl.set_trusted_selected_indices(torch.cat(selections, dim=1)[0])
            actual = impl.forward(packed, None)
            self.assertEqual(fast.call_count, 2)
        expected = torch.cat(
            [
                reference(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], indices)
                .transpose(1, 2)
                .reshape(b - a, -1)
                for (a, b), indices in zip(((0, 129), (129, 257)), selections)
            ]
        )
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)

        # Packed launches require CU metadata supplied by the indexer owner.
        inputs.cu_seqlens_device = torch.tensor(
            [0, 129, 257], dtype=torch.int32, device=packed.device
        )
        packed_selection = torch.cat(selections, dim=1)[0]
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_ragged_causal_qsa_prefill",
            wraps=try_ragged_causal_qsa_prefill,
        ) as fast:
            impl.set_trusted_selected_indices(
                packed_selection, cu_seqlens=inputs.cu_seqlens_device
            )
            actual = impl.forward(packed, None)
            fast.assert_called_once()
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)
        self.assertTrue(actual.is_contiguous())
        self.assertIsNone(impl._trusted_prefill_cu_seqlens)

        old_cu = inputs.cu_seqlens_device
        inputs.cu_seqlens_device = old_cu.clone()
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_prefill.try_ragged_causal_qsa_prefill",
            side_effect=AssertionError("CU owner changed"),
        ):
            impl.set_trusted_selected_indices(packed_selection, cu_seqlens=old_cu)
            actual = impl.forward(packed, None)
        torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)

    def test_graph_replay_refreshes_membership_and_qkv(self):
        packed, q, k, v = make_inputs(257, 12)
        selected = make_selection(257)
        for _ in range(3):
            try_causal_qsa_prefill(q, k, v, selected)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_causal_qsa_prefill(q, k, v, selected)
        selected.fill_(-1)
        graph.replay()
        self.assertTrue((actual == 0).all())
        packed.copy_(torch.randn_like(packed))
        selected.copy_(make_selection(257))
        graph.replay()
        torch.testing.assert_close(
            actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
        )

    def test_unsupported_shapes_and_alignment_return_to_general_kernel(self):
        _, q, k, v = make_inputs(257, 12)
        selected = make_selection(257)
        self.assertIsNone(try_causal_qsa_prefill(q[:, :3], k, v, selected))
        self.assertIsNone(try_causal_qsa_prefill(q[:, :, :32], k, v, selected))
        self.assertIsNone(try_causal_qsa_prefill(q, k, v, selected[:, :, :-1]))
        self.assertIsNone(try_causal_qsa_prefill(q, k, v, selected.long()))
        misaligned = torch.empty(257, 257, dtype=v.dtype, device=v.device)[:, 1:]
        self.assertIsNone(
            try_causal_qsa_prefill(q, k, misaligned[None, None], selected)
        )

    def test_no_device_scalar_readback(self):
        _, q, k, v = make_inputs(257, 12)
        selected = make_selection(257)
        try_causal_qsa_prefill(q, k, v, selected)
        with (
            patch.object(torch.Tensor, "item", side_effect=AssertionError("readback")),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("readback")
            ),
        ):
            actual = try_causal_qsa_prefill(q, k, v, selected)
        torch.testing.assert_close(
            actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
        )

    def test_missing_gluon_retains_standard_tma_fallback(self):
        _, q, k, v = make_inputs(257, 6)
        selected = make_selection(257)
        module = "rtp_llm.models_py.modules.qwen4_exp.qsa.tma_pipeline"
        with patch.dict(sys.modules, {module: None}):
            actual = try_causal_qsa_prefill(q, k, v, selected)
        self.assertIsNotNone(actual)
        torch.testing.assert_close(
            actual, reference(q, k, v, selected), atol=0.0078125, rtol=0.008
        )

    def test_packed_ragged_refreshes_request_partition_and_isolates_tail_tiles(self):
        for heads in (6, 12):
            with self.subTest(heads=heads):
                packed, q, k, v = make_inputs(257, heads)
                flat_q, flat_k, flat_v = packed.split([heads * 256, 256, 256], -1)
                selected = torch.cat([make_selection(129), make_selection(128)], 1)[0]
                cu = torch.tensor([0, 129, 257], dtype=torch.int32, device=q.device)

                def run():
                    return try_ragged_causal_qsa_prefill(
                        flat_q, flat_k, flat_v, selected, [129, 128], cu
                    )

                actual = run()
                self.assertIsNotNone(actual)
                expected = reference(
                    q[:, :, :129], k[:, :, :129], v[:, :, :129], selected[None, :129]
                )
                # Only the following request is poisoned; the first tail tile stays isolated.
                flat_v[129:] = torch.nan
                actual = run()
                torch.testing.assert_close(
                    actual[:129],
                    expected.transpose(1, 2).reshape(129, -1),
                    atol=0.0078125,
                    rtol=0.008,
                )
                packed.copy_(torch.randn_like(packed))
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                selected.fill_(-1)
                graph.replay()
                self.assertTrue((actual == 0).all())
                cu.copy_(
                    torch.tensor([0, 128, 257], dtype=torch.int32, device=q.device)
                )
                selected.copy_(
                    torch.cat([make_selection(128), make_selection(129)], 1)[0]
                )
                graph.replay()
                expected = torch.cat(
                    [
                        reference(
                            q[:, :, a:b],
                            k[:, :, a:b],
                            v[:, :, a:b],
                            selected[None, a:b],
                        )
                        .transpose(1, 2)
                        .reshape(b - a, -1)
                        for a, b in ((0, 128), (128, 257))
                    ]
                )
                torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)

    def test_packed_ragged_guards_keep_unsupported_metadata_on_general_path(self):
        packed, _, _, _ = make_inputs(257, 12)
        q, k, v = packed.split([12 * 256, 256, 256], -1)
        ids = torch.cat([make_selection(129), make_selection(128)], 1)[0]
        cu = torch.tensor([0, 129, 257], dtype=torch.int32, device=q.device)
        self.assertIsNone(try_ragged_causal_qsa_prefill(q, k, v, ids, [257], cu))
        self.assertIsNone(try_ragged_causal_qsa_prefill(q, k, v, ids, [129, 129], cu))
        self.assertIsNone(try_ragged_causal_qsa_prefill(q, k, v, ids, [129, 128], None))
        self.assertIsNone(
            try_ragged_causal_qsa_prefill(q, k, v, ids, [129, 128], cu.long())
        )

    def test_batched_paged_table_geometry_reuses_compiled_pipeline(self):
        try:
            from rtp_llm.models_py.modules.qwen4_exp.qsa import (
                tma_paged_ragged_pipeline,
            )
        except ImportError:
            self.skipTest("Gluon paged pipeline unavailable")
        kernel = tma_paged_ragged_pipeline._batched_paged_prefill
        run = kernel.run
        hashes = []

        def record(*args, **kwargs):
            compiled = run(*args, **kwargs)
            hashes.append(compiled.hash)
            return compiled

        for heads in (6, 12):
            for fp8 in (False, True):
                with self.subTest(heads=heads, fp8=fp8):
                    lengths, prefixes = [229, 239, 259], [512] * 3
                    q, backing, cache, original, ids, scale, cu, pd = (
                        self._paged_batch_case(heads, lengths, prefixes, fp8)
                    )
                    expected = self._paged_batch_reference(
                        q, cache, original, ids, lengths, prefixes, scale
                    )
                    hashes.clear()
                    with patch.object(kernel, "run", side_effect=record):
                        for capacity, padding in ((8, 0), (16, 0), (31, 3), (64, 3)):
                            storage = torch.zeros(
                                (3, capacity + padding),
                                dtype=torch.int32,
                                device=q.device,
                            )
                            table = storage[:, :capacity]
                            copied = min(capacity, original.shape[1])
                            table[:, :copied].copy_(original[:, :copied])
                            actual = try_ragged_paged_qsa_prefill(
                                q,
                                backing,
                                table,
                                ids,
                                lengths,
                                prefixes,
                                cu,
                                pd,
                                page_size=128,
                                kv_head_num=1,
                                kv_scale=scale,
                            )
                            torch.testing.assert_close(
                                actual, expected, atol=0.0078125, rtol=0.008
                            )
                    self.assertEqual(len(hashes), 4)
                    self.assertEqual(len(set(hashes)), 1)

    def test_runtime_lengths_reuse_compiled_pipeline(self):
        try:
            from rtp_llm.models_py.modules.qwen4_exp.qsa import tma_pipeline
        except ImportError:
            self.skipTest("Gluon TMA pipeline unavailable")
        kernel = tma_pipeline._pipelined_prefill
        run = kernel.run
        hashes = []

        def record(*args, **kwargs):
            compiled = run(*args, **kwargs)
            hashes.append(compiled.hash)
            return compiled

        for heads, lengths in ((6, (257, 509)), (12, (769, 1021))):
            with self.subTest(heads=heads):
                hashes.clear()
                with patch.object(kernel, "run", side_effect=record):
                    for tokens in lengths:
                        _, q, k, v = make_inputs(tokens, heads)
                        selected = make_selection(tokens)
                        actual = try_causal_qsa_prefill(q, k, v, selected)
                        torch.testing.assert_close(
                            actual,
                            reference(q, k, v, selected),
                            atol=0.0078125,
                            rtol=0.008,
                        )
                self.assertEqual(len(hashes), len(lengths))
                self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main()
