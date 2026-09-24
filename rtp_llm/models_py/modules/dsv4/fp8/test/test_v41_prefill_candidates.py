"""Exact prefill candidate selection, packed flags, and graph replay tests."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_candidates as fused


def reference_select(logits, visible, block_size, topk_blocks):
    """Frozen old attention_v41.select_candidate_blocks, including sorted=True."""
    nblocks = (logits.shape[1] + block_size - 1) // block_size
    padded = F.pad(
        logits, (0, nblocks * block_size - logits.shape[1]), value=-torch.inf
    )
    scores = padded.view(logits.shape[0], nblocks, block_size).amax(-1)
    newest = ((visible - 1).clamp_min(0) // block_size).long()
    scores.scatter_(
        1, newest[:, None], torch.where(visible > 0, torch.inf, -torch.inf)[:, None]
    )
    values, ids = scores.topk(min(topk_blocks, nblocks), dim=-1)
    return torch.where(values > -torch.inf, ids, -1).int()


def reference_mask(logits, candidates, block_size):
    nblocks = (logits.shape[1] + block_size - 1) // block_size
    allowed = torch.zeros(
        logits.shape[0], nblocks + 1, dtype=torch.bool, device=logits.device
    )
    ids = torch.where(candidates >= 0, candidates, nblocks).long()
    allowed.scatter_(1, ids, True)
    columns = torch.arange(logits.shape[1], device=logits.device) // block_size
    return logits.masked_fill_(~allowed[:, columns], -torch.inf)


def make_case(rows, width, seed=41):
    torch.manual_seed(seed)
    storage = torch.randn(rows, width + 256, device="cuda")
    logits = storage[:, :width]
    visible_storage = torch.randint(0, width + 1, (rows * 2,), device="cuda")
    visible = visible_storage[::2]
    if rows >= 3:
        visible[:3] = torch.tensor([0, 1, width], device="cuda")
    logits.masked_fill_(
        torch.arange(width, device="cuda")[None, :] >= visible[:, None], -torch.inf
    )
    return logits, visible


def metadata(rows=17, width=32768):
    logits = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda"),
        dtype=torch.float32,
        ndim=2,
        shape=(rows, width),
        stride=lambda dim: width + 256 if dim == 0 else 1,
        requires_grad=False,
    )
    visible = SimpleNamespace(
        device=logits.device,
        ndim=1,
        dtype=torch.int64,
        numel=lambda: rows,
        stride=lambda dim: 2,
    )
    return logits, visible


class V41PrefillCandidatesCPU(unittest.TestCase):
    def test_default_gate_and_unsupported_block_size(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(fused.is_supported(*metadata(), 8, 2048))
        self.assertFalse(fused.is_supported(*metadata(), 0, 2048))
        x = torch.empty(3, 41)
        self.assertIsNone(fused.select_candidates(x, torch.ones(3).long(), 8, 5))
        self.assertFalse(fused.mask_candidates(x, torch.ones(3, 1).int(), 8))

    def test_layout_dtype_shape_and_grad_gates(self):
        for field, value in (
            ("dtype", torch.bfloat16),
            ("ndim", 3),
            ("shape", (0, 32768)),
            ("shape", (17, 0)),
            ("shape", (17, (1 << 20) + 1)),
            ("stride", lambda dim: 2),
            ("requires_grad", True),
        ):
            x, visible = metadata()
            setattr(x, field, value)
            with torch.enable_grad(), self.subTest(field=field):
                self.assertFalse(fused.is_supported(x, visible, 8, 2048))
        x, visible = metadata()
        visible.dtype = torch.float32
        self.assertFalse(fused.is_supported(x, visible, 8, 2048))
        self.assertFalse(fused.is_supported(*metadata(), 3, 2048))
        self.assertFalse(fused.is_supported(*metadata(), 8, 0))

    def test_full_cache_memory_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            for tokens, mib in (
                (32768, 4),
                (65536, 16),
                (131072, 64),
                (262144, 256),
                (524288, 1024),
            ):
                self.assertEqual(
                    fused.bitmap_size_bytes(tokens // 4, tokens, 8), mib * 1024**2
                )
                self.assertEqual(
                    fused.bitmap_is_bounded(tokens // 4, tokens, 8), mib <= 256
                )
        with patch.dict(os.environ, {"DSV41_PREFILL_CANDIDATE_FLAGS_MAX_BYTES": "0"}):
            self.assertFalse(fused.bitmap_is_bounded(1, 8, 8))


class V41PrefillCandidatesCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

    @torch.no_grad()
    def test_pool_torch_bits_with_causal_mask_and_token_epilogue(self):
        patterns = torch.tensor(
            [
                0,
                -2147483648,
                -1082130432,
                0x3F800000,
                0x7F800000,
                -8388608,
                0x7FC00001,
                0x7FC01234,
                -4194303,
            ],
            dtype=torch.int32,
            device="cuda",
        )
        for block in (1, 2, 4, 8, 16, 32):
            width = 521 * block - 1
            torch.manual_seed(33000 + block)
            raw = patterns[torch.randint(len(patterns), (4, width + 17), device="cuda")]
            raw = raw.view(torch.float32)[:, :width]
            ends = torch.tensor(
                [0, 1, width - 3, width], device="cuda", dtype=torch.int32
            )
            clean = raw.masked_fill(
                torch.arange(width, device="cuda")[None] >= ends[:, None], -torch.inf
            )
            nblocks = (width + block - 1) // block
            expected = F.pad(clean, (0, nblocks * block - width), value=-torch.inf)
            expected = expected.reshape(4, nblocks, block).amax(-1)
            expected.scatter_(
                1,
                ((ends.long() - 1).clamp_min(0) // block)[:, None],
                torch.where(ends > 0, torch.inf, -torch.inf)[:, None],
            )
            for fused_epilogue in (False, True):
                with self.subTest(block=block, fused=fused_epilogue):
                    tokens = torch.arange(512, device="cuda", dtype=torch.int32)[
                        None
                    ].repeat(4, 1)
                    expected_tokens = torch.where(
                        (tokens < ends[:, None]) & clean[:, :512].isfinite(), tokens, -1
                    )
                    output = torch.empty_like(expected)
                    source = raw if fused_epilogue else clean
                    fused._prefill_candidate_pool_kernel[(4, (nblocks + 127) // 128)](
                        source,
                        ends,
                        output,
                        width,
                        source.stride(0),
                        1,
                        block,
                        nblocks,
                        128,
                        MASK_TAIL=fused_epilogue,
                        token_indices=tokens if fused_epilogue else None,
                        token_ends=ends if fused_epilogue else None,
                        FILTER_TOKENS=fused_epilogue,
                    )
                    self.assertTrue(
                        torch.equal(
                            output.view(torch.int32), expected.view(torch.int32)
                        )
                    )
                    if fused_epilogue:
                        self.assertTrue(torch.equal(tokens, expected_tokens))

    @torch.no_grad()
    def test_publication_filter_graph_replay_matches_masked_reference(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk

        for width in (1033, 32769):
            for bitmap in (False, True):
                with self.subTest(width=width, bitmap=bitmap):
                    torch.manual_seed(33001)
                    logits = torch.randn(4, width + 17, device="cuda")[:, :width]
                    ends = torch.full((4,), width, device="cuda", dtype=torch.int32)
                    bounds = (torch.zeros_like(ends), ends)
                    tokens = torch.empty(4, 512, device="cuda", dtype=torch.int32)
                    candidates = torch.empty(4, 2048, device="cuda", dtype=torch.int32)
                    flags = (
                        torch.empty(
                            4,
                            fused.bitmap_words(width, 8),
                            device="cuda",
                            dtype=torch.int32,
                        )
                        if bitmap
                        else None
                    )

                    def run():
                        self.assertIs(
                            topk.try_select_tokens(
                                logits,
                                ends,
                                bounds=bounds,
                                out=tokens,
                                filter_finite=False,
                            ),
                            tokens,
                        )
                        self.assertIsNotNone(
                            fused.select_candidates(
                                logits,
                                ends,
                                8,
                                2048,
                                out=candidates,
                                flags=flags,
                                build_bitmap=bitmap,
                                mask_tail=True,
                                token_indices=tokens,
                                token_ends=ends,
                            )
                        )

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    for changed in (False, True):
                        if changed:
                            ends.copy_(
                                torch.tensor(
                                    [0, 1, 511, width - 3],
                                    device="cuda",
                                    dtype=torch.int32,
                                )
                            )
                            logits[:, ::19] = torch.nan
                            logits[:, ::31] = torch.inf
                            logits[:, ::43] = -torch.inf
                            logits[:, 4::47] = -0.0
                            logits[:, 5::47] = 0.0
                        graph.replay()
                        clean = logits.masked_fill(
                            torch.arange(width, device="cuda")[None] >= ends[:, None],
                            -torch.inf,
                        )
                        expected_tokens = topk.try_select_tokens(
                            clean, ends, bounds=bounds
                        )
                        self.assertTrue(
                            (
                                (tokens == -1)
                                | ((tokens >= 0) & (tokens < ends[:, None]))
                            )
                            .all()
                            .item()
                        )
                        if not changed:
                            self.assertTrue(
                                torch.equal(
                                    tokens.sort(-1).values,
                                    expected_tokens.sort(-1).values,
                                )
                            )
                        actual_values = clean.gather(
                            1, tokens.clamp_min(0).long()
                        ).masked_fill(tokens < 0, -torch.inf)
                        expected_values = clean.gather(
                            1, expected_tokens.clamp_min(0).long()
                        ).masked_fill(expected_tokens < 0, -torch.inf)
                        self.assertTrue(
                            torch.equal(
                                actual_values.sort(-1).values,
                                expected_values.sort(-1).values,
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                (tokens >= 0).sum(-1), (expected_tokens >= 0).sum(-1)
                            )
                        )
                        for row in tokens:
                            self.assertEqual(
                                row[row >= 0].numel(), row[row >= 0].unique().numel()
                            )
                        expected = reference_select(clean, ends, 8, 2048)
                        expected = F.pad(
                            expected, (0, 2048 - expected.shape[1]), value=-1
                        )
                        self.assertTrue(
                            torch.equal(
                                candidates.sort(-1).values, expected.sort(-1).values
                            )
                        )
                        if bitmap:
                            self.assertTrue(torch.equal(candidates, expected))
                            self.assertTrue(
                                torch.equal(
                                    flags, fused.build_flags(expected, width, 8)
                                )
                            )

    def check_case(self, logits, visible, block=8, k=2048):
        before = logits.clone()
        expected = reference_select(logits, visible, block, k)
        result = fused.select_candidates(logits, visible, block, k)
        self.assertIsNotNone(result)
        actual, flags = result
        self.assertEqual(actual.dtype, torch.int32)
        self.assertEqual(flags.dtype, torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        rebuilt = fused.build_flags(actual, logits.shape[1], block)
        torch.testing.assert_close(rebuilt, flags, rtol=0, atol=0)
        unordered, unused_flags = fused.select_candidates(
            logits, visible, block, k, build_bitmap=False
        )
        self.assertIsNone(unused_flags)
        torch.testing.assert_close(
            unordered.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
        )
        torch.testing.assert_close(
            fused.build_flags(unordered, logits.shape[1], block), flags, rtol=0, atol=0
        )
        torch.testing.assert_close(
            logits.view(torch.int32), before.view(torch.int32), rtol=0, atol=0
        )
        reference = reference_mask(before, expected, block)
        self.assertTrue(fused.mask_candidates(logits, flags, block))
        torch.testing.assert_close(
            logits.view(torch.int32), reference.view(torch.int32), rtol=0, atol=0
        )

    @torch.no_grad()
    def test_real_chunk_shapes(self):
        for rows, width in ((1024, 65536), (2048, 32768), (4096, 16384)):
            with self.subTest(rows=rows, width=width):
                self.check_case(*make_case(rows, width))

    @torch.no_grad()
    def test_boundaries_nan_infinity_ties_and_partial_blocks(self):
        for width, block, k in (
            (1, 1, 3),
            (37, 8, 7),
            (2059, 8, 257),
            (32769, 8, 2048),
        ):
            with self.subTest(width=width):
                logits, visible = make_case(11, width)
                visible[:] = width
                visible[0] = 0
                logits[0] = -torch.inf
                logits[1] = 0
                logits[2] = -torch.inf
                logits[3] = float("nan")
                logits[4] = torch.inf
                logits[5, ::2] = -0.0
                logits[5, 1::2] = 0.0
                logits[6, : min(width, 33)] = float("nan")
                self.check_case(logits, visible, block, k)

    @torch.no_grad()
    def test_sliced_outputs_extra_words_and_padding(self):
        rows, width = 17, 2059
        logits, visible = make_case(rows, width)
        reference = reference_select(logits, visible, 8, 257)
        out_base = torch.full((rows + 4, 280), 123, dtype=torch.int32, device="cuda")
        flag_base = torch.full((rows + 4, 23), 123, dtype=torch.int32, device="cuda")
        out, flags = out_base[2:-2, :270], flag_base[2:-2, :19]
        result = fused.select_candidates(logits, visible, 8, 257, out=out, flags=flags)
        self.assertIs(result[0], out)
        self.assertIs(result[1], flags)
        torch.testing.assert_close(out[:, :257], reference, rtol=0, atol=0)
        self.assertTrue(torch.all(out[:, 257:] == -1).item())
        self.assertTrue(torch.all(flags[:, fused.bitmap_words(width, 8) :] == 0).item())
        self.assertTrue(torch.all(out_base[:2] == 123).item())
        self.assertTrue(torch.all(flag_base[-2:] == 123).item())
        self.assertTrue(torch.all(flag_base[:, 19:] == 123).item())
        expected = reference_mask(logits.clone(), reference, 8)
        self.assertTrue(fused.mask_candidates(logits, flags, 8))
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_sparse_publication_skips_unused_bitmap(self):
        logits, visible = make_case(17, 32769)
        expected = reference_select(logits, visible, 8, 2048)
        with patch.dict(os.environ, {"DSV41_PREFILL_CANDIDATE_FLAGS_MAX_BYTES": "0"}):
            result = fused.select_candidates(
                logits, visible, 8, 2048, build_bitmap=False
            )
        self.assertIsNotNone(result)
        self.assertIsNone(result[1])
        torch.testing.assert_close(
            result[0].sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
        )

    @torch.no_grad()
    def test_all_blocks_unordered_sliced_output_and_graph(self):
        logits, visible = make_case(11, 1033)
        logits[2:, ::17] = float("nan")
        logits[2:, ::53] = torch.inf
        logits.masked_fill_(
            torch.arange(1033, device="cuda")[None] >= visible[:, None], -torch.inf
        )
        expected = reference_select(logits, visible, 8, 2048)
        before = logits.view(torch.int32).clone()
        storage = torch.full((15, 151), 123, dtype=torch.int32, device="cuda")
        out = storage[2:-2, 1:138]

        def run():
            result = fused.select_candidates(
                logits, visible, 8, 2048, out=out, build_bitmap=False
            )
            self.assertIs(result[0], out)
            self.assertIsNone(result[1])

        with patch.object(
            torch.Tensor, "topk", side_effect=AssertionError("unused TopK")
        ):
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            out.fill_(99)
            graph.replay()
        torch.testing.assert_close(
            out[:, :130].sort(-1).values,
            expected.sort(-1).values,
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(out[:, 130:] == -1).item())
        for guard in (storage[:2], storage[-2:], storage[:, :1], storage[:, 138:]):
            self.assertTrue(torch.all(guard == 123).item())
        torch.testing.assert_close(logits.view(torch.int32), before, rtol=0, atol=0)

    @torch.no_grad()
    def test_candidate_only_preserves_sparse_plan_scores_and_remap(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as indexer
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_sparse_prefill_indexer as sparse,
        )

        if (
            torch.cuda.get_device_capability()[0] != 10
            or sparse._get_deep_gemm() is None
        ):
            self.skipTest("SM100 sparse DeepGEMM is required")
        rows, width = 13, 32769
        q = torch.randn(rows, 32, 128, device="cuda").bfloat16()
        key = torch.randn(width, 128, device="cuda").bfloat16()
        q_payload, q_sf = indexer.quantize_indexer_q(q)
        keys = indexer.PrefillIndexerKeys(*indexer.quantize_indexer_k_reference(key))
        weights = torch.randn(rows, 32, device="cuda") / 64
        for pattern in ("random", "ties", "nonfinite"):
            with self.subTest(pattern=pattern):
                logits, visible = make_case(rows, width)
                if pattern == "ties":
                    logits.round_()
                elif pattern == "nonfinite":
                    logits[2:, ::17] = float("nan")
                    logits[2:, ::53] = torch.inf
                    logits[2:, ::71] = -torch.inf
                    logits.masked_fill_(
                        torch.arange(width, device="cuda")[None] >= visible[:, None],
                        -torch.inf,
                    )
                reference = reference_select(logits, visible, 8, 2048)
                actual, flags = fused.select_candidates(
                    logits, visible, 8, 2048, build_bitmap=False
                )
                self.assertIsNone(flags)
                torch.testing.assert_close(
                    actual.sort(1).values, reference.sort(1).values, rtol=0, atol=0
                )
                old_plan = sparse.prepare_plan(reference, visible, width)
                new_plan = sparse.prepare_plan(actual, visible, width)
                for name in ("sparse_indices", "end", "row_ks", "row_ke"):
                    torch.testing.assert_close(
                        getattr(new_plan, name), getattr(old_plan, name), rtol=0, atol=0
                    )
                old_scores = sparse.score(q_payload, q_sf, keys, weights, old_plan)
                new_scores = sparse.score(q_payload, q_sf, keys, weights, new_plan)
                valid = (
                    torch.arange(old_plan.sparse_columns, device="cuda")[None]
                    < old_plan.end[:, None]
                )
                torch.testing.assert_close(
                    new_scores[valid], old_scores[valid], rtol=0, atol=0
                )
                columns = (
                    old_scores.float()
                    .masked_fill(~valid, -torch.inf)
                    .topk(512, dim=-1)
                    .indices.int()
                )
                torch.testing.assert_close(
                    sparse.remap(columns, new_plan, logits=new_scores),
                    sparse.remap(columns, old_plan, logits=old_scores),
                    rtol=0,
                    atol=0,
                )

    @torch.no_grad()
    def test_runtime_count_fixed_output_does_not_recompile(self):
        def compiled_count():
            return sum(len(cache) for cache, *_ in kernel.device_caches.values())

        rows, out_k = 11, 2048
        out = torch.empty((rows, out_k), dtype=torch.int32, device="cuda")
        flags = torch.empty((rows, 64), dtype=torch.int32, device="cuda")
        counts = (
            1,
            3,
            7,
            17,
            31,
            33,
            63,
            65,
            127,
            129,
            257,
            511,
            513,
            1023,
            1025,
            1537,
            2047,
            2048,
        )
        for build_bitmap in (False, True):
            kernel = (
                fused._prefill_candidate_store_bitmap_kernel
                if build_bitmap
                else fused._prefill_candidate_pool_kernel
            )
            kernel.device_caches.clear()
            before = compiled_count()
            for count in counts:
                with self.subTest(build_bitmap=build_bitmap, count=count):
                    # Keep output/bitmap geometry fixed; only runtime K changes.
                    width = count * 8 - count % 7
                    logits, visible = make_case(rows, width)
                    expected = reference_select(logits, visible, 8, out_k)
                    result = fused.select_candidates(
                        logits,
                        visible,
                        8,
                        out_k,
                        out=out,
                        flags=flags if build_bitmap else None,
                        build_bitmap=build_bitmap,
                    )
                    self.assertIsNotNone(result)
                    self.assertIs(result[0], out)
                    torch.testing.assert_close(
                        (
                            out[:, :count]
                            if build_bitmap
                            else out[:, :count].sort(dim=-1).values
                        ),
                        expected if build_bitmap else expected.sort(dim=-1).values,
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(torch.all(out[:, count:] == -1).item())
                    if build_bitmap:
                        self.assertIs(result[1], flags)
                        rebuilt = fused.build_flags(expected, width, 8)
                        words = fused.bitmap_words(width, 8)
                        torch.testing.assert_close(
                            flags[:, :words], rebuilt, rtol=0, atol=0
                        )
                        self.assertTrue(torch.all(flags[:, words:] == 0).item())
                        masked = reference_mask(logits.clone(), expected, 8)
                        self.assertTrue(fused.mask_candidates(logits, flags, 8))
                        torch.testing.assert_close(logits, masked, rtol=0, atol=0)
                    else:
                        self.assertIsNone(result[1])
                    # Exactly one specialization per WRITE_FLAGS geometry,
                    # including K=1, odd K, and aligned K=2048.
                    self.assertEqual(compiled_count(), before + 1)

    @torch.no_grad()
    def test_dynamic_cuda_graph_replay(self):
        rows, width = 17, 2059
        logits, visible = make_case(rows, width)
        out = torch.empty(rows, 257, dtype=torch.int32, device="cuda")
        flags = torch.empty(
            rows, fused.bitmap_words(width, 8), dtype=torch.int32, device="cuda"
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fused.select_candidates(logits, visible, 8, 257, out=out, flags=flags)
                fused.mask_candidates(logits, flags, 8)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fused.select_candidates(
                logits, visible, 8, 257, out=out, flags=flags
            )
            fused.mask_candidates(logits, flags, 8)
        for seed in (7, 19, 41, 103):
            source, lengths = make_case(rows, width, seed)
            expected_ids = reference_select(source, lengths, 8, 257)
            expected_logits = reference_mask(source.clone(), expected_ids, 8)
            logits.copy_(source)
            visible.copy_(lengths)
            graph.replay()
            torch.testing.assert_close(captured[0], expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)

    @torch.no_grad()
    def test_512k_chunk_bitmap_rebuild_and_duplicate_ids(self):
        # Full-query caching is gated at 1 GiB; chunk rebuilding stays 8 MiB.
        rows, width = 1024, 524288
        storage = torch.randint(
            0, width // 8, (rows + 2, 2053), device="cuda", dtype=torch.int32
        )
        candidates = storage[1:-1, :2048]
        candidates[:, :8] = torch.tensor(
            [-1, 0, 31, 32, 63, 64, 65535, 31], device="cuda", dtype=torch.int32
        )
        flags = fused.build_flags(candidates, width, 8)
        self.assertEqual(flags.numel() * flags.element_size(), 8 * 1024**2)
        # Compare bitmap bits directly without allocating a 2 GiB logits tensor.
        bit_ids = torch.arange(width // 8, device="cuda")
        unpacked = ((flags[:, bit_ids // 32].long() >> (bit_ids % 32)) & 1).bool()
        expected = torch.zeros(rows, width // 8 + 1, dtype=torch.bool, device="cuda")
        expected.scatter_(
            1, torch.where(candidates >= 0, candidates, width // 8).long(), True
        )
        torch.testing.assert_close(unpacked, expected[:, :-1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
