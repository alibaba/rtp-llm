import ast
import gc
import sys
import threading
import types
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    _slice_relative_mqa_logits,
    fp8_paged_indexer_score,
)
from rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla import (
    SM120_EXTRA_TOPK_WIDTHS,
    canonical_topk,
    run,
    run_chunked_reference,
    sparse_mla_available,
    token_lens,
    validate_sm120_head_count,
    validate_sm120_swa_topk_width,
)


class Sm120SparseMlaCanonicalTest(unittest.TestCase):
    def test_native_head_counts_include_padded_eight_head_case(self):
        for heads in (8, 16, 32, 64, 128):
            validate_sm120_head_count(heads)
        for heads in (0, 4, 12, 256):
            with self.assertRaisesRegex(ValueError, "local query heads"):
                validate_sm120_head_count(heads)

    def test_sparse_mla_available_reflects_provider_load(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        for provider, expected in ((None, False), (Mock(), True)):
            with self.subTest(provider=provider):
                with patch.object(
                    sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=provider
                ):
                    self.assertIs(sparse_mla_available(), expected)

    def test_run_rejects_invalid_prefill_page_or_extra_width(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        provider = Mock()
        query = torch.zeros((1, 16, 512), dtype=torch.bfloat16)
        swa_indices = torch.zeros((1, 128), dtype=torch.int32)
        swa_lens = torch.ones((1,), dtype=torch.int32)
        out = torch.full_like(query, 7)
        sinks = torch.zeros((16,), dtype=torch.float32)

        with patch.object(
            sm120_sparse_mla, "_load_sm120_sparse_mla", return_value=provider
        ):
            bad_page_cache = torch.zeros((1, 128, 584), dtype=torch.uint8)
            with self.assertRaisesRegex(ValueError, "page_block_size=128"):
                run(
                    query=query,
                    swa_cache=bad_page_cache,
                    swa_indices=swa_indices,
                    swa_lens=swa_lens,
                    out=out,
                    scale=1.0,
                    sinks=sinks,
                    is_prefill=True,
                )

            good_cache = torch.zeros((1, 64, 584), dtype=torch.uint8)
            extra_cache = torch.zeros((1, 2, 584), dtype=torch.uint8)
            for width in (2, 65):
                with self.subTest(width=width):
                    with self.assertRaisesRegex(
                        ValueError,
                        "prefill extra_indices width must be a multiple of 64, "
                        f"got {width}",
                    ):
                        run(
                            query=query,
                            swa_cache=good_cache,
                            swa_indices=swa_indices,
                            swa_lens=swa_lens,
                            out=out,
                            scale=1.0,
                            sinks=sinks,
                            extra_cache=extra_cache,
                            extra_indices=torch.zeros((1, width), dtype=torch.int32),
                            extra_lens=swa_lens,
                            is_prefill=True,
                        )

        provider.assert_not_called()
        self.assertTrue(torch.all(out == 7))

    def test_none_length_compacts_non_negative_slots(self):
        indices = torch.tensor([[11, -1, 13, -1], [-1, 7, 8, 9]], dtype=torch.int64)
        canonical, lengths = canonical_topk(indices, None, (4, 8))
        self.assertEqual(canonical.dtype, torch.int32)
        self.assertEqual(lengths.tolist(), [2, 3])
        self.assertEqual(canonical.tolist(), [[11, 13, -1, -1], [7, 8, 9, -1]])

    def test_short_explicit_length_is_preserved(self):
        indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, torch.tensor([2]), (4, 8))
        self.assertEqual(canonical.tolist(), [[1, 2, -1, -1]])
        self.assertEqual(lengths.tolist(), [2])

    def test_unsupported_width_padding_keeps_invalid_slots(self):
        indices = torch.tensor([[4, -1, 9]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, None, (4, 8))
        self.assertEqual(canonical.shape, (1, 4))
        self.assertEqual(canonical.tolist(), [[4, 9, -1, -1]])
        self.assertEqual(lengths.tolist(), [2])

    def test_explicit_length_ignores_holes_and_tail(self):
        indices = torch.tensor([[5, -1, 7, 8]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, torch.tensor([2]), (4, 8))
        self.assertEqual(canonical.tolist(), [[5, -1, -1, -1]])
        self.assertEqual(lengths.tolist(), [1])

    def test_length_is_clamped_to_kernel_width(self):
        indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        canonical, lengths = canonical_topk(indices, torch.tensor([99]), (4, 8))
        self.assertEqual(canonical.tolist(), [[1, 2, 3, 4]])
        self.assertEqual(lengths.tolist(), [4])

    def test_normalization_preserves_shared_length_views(self):
        for valid_mask in (
            None,
            torch.tensor([[True] * 4, [True, False, True, False]]),
        ):
            with self.subTest(masked=valid_mask is not None):
                storage = torch.tensor([-1, 17, 9, 23], dtype=torch.int32)
                before = storage.clone()
                lengths = storage[::2]
                result = token_lens(lengths, 2, 4, lengths.device, valid_mask)
                torch.testing.assert_close(storage, before)
                self.assertEqual(result.tolist(), [0, 4 if valid_mask is None else 2])
                result.fill_(1)
                torch.testing.assert_close(storage, before)

    def test_canonicalization_preserves_out_of_range_lengths(self):
        indices = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
        lengths = torch.tensor([-1, 99], dtype=torch.int32)
        canonical, normalized = canonical_topk(indices, lengths, (4, 8))
        self.assertEqual(lengths.tolist(), [-1, 99])
        self.assertEqual(normalized.tolist(), [0, 4])
        self.assertEqual(canonical.tolist(), [[-1] * 4, [5, 6, 7, 8]])

    def test_width_overflow_fails_with_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds the largest"):
            canonical_topk(torch.zeros(1, 9, dtype=torch.int32), None, (4, 8))

    def test_normal_decode_width_is_validated_before_first_request(self):
        self.assertEqual(
            validate_sm120_swa_topk_width(513, context="DeepSeek-V4 decode"),
            1024,
        )
        self.assertEqual(
            validate_sm120_swa_topk_width(1025, context="DeepSeek-V4 decode"),
            1025,
        )

    def test_dspark_gamma_is_included_in_startup_width_validation(self):
        window = 1024
        gamma = 3
        dspark_width = ((window + gamma + 127) // 128) * 128
        self.assertEqual(dspark_width, 1152)
        self.assertEqual(
            validate_sm120_swa_topk_width(
                dspark_width,
                context="DeepSeek-V4 DSpark decode",
            ),
            1152,
        )

    def test_chunked_large_width_fallback_matches_dense_softmax(self):
        torch.manual_seed(7)
        rows, heads, dim = 2, 2, 512
        query = torch.randn(rows, heads, dim, dtype=torch.float32) / 16
        cache_rows = torch.randn(8, dim, dtype=torch.float32) / 16
        indices = torch.tensor([[0, 2, 4, -1, -1], [1, 3, 5, 7, -1]])
        lengths = torch.tensor([3, 4], dtype=torch.int32)
        sinks = torch.tensor([0.2, -0.3], dtype=torch.float32)
        actual = torch.empty_like(query)

        def fake_dequantize(_cache, slots):
            safe = slots.clamp_min(0).to(torch.long)
            result = cache_rows.index_select(0, safe)
            return result.masked_fill((slots < 0).unsqueeze(1), 0.0)

        with patch(
            "rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton."
            "dequantize_slots_to_bf16",
            side_effect=fake_dequantize,
        ):
            run_chunked_reference(
                query=query,
                swa_cache=torch.empty(0),
                swa_indices=indices,
                swa_lens=lengths,
                out=actual,
                scale=0.5,
                sinks=sinks,
                chunk_size=2,
            )

        expected = torch.empty_like(actual)
        for row in range(rows):
            selected = cache_rows.index_select(
                0, indices[row, : lengths[row]].to(torch.long)
            )
            logits = torch.einsum("hd,kd->hk", query[row], selected) * 0.5
            denom_logits = torch.cat((logits, sinks[:, None]), dim=1)
            weights = torch.softmax(denom_logits, dim=1)[:, :-1]
            expected[row] = torch.einsum("hk,kd->hd", weights, selected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_hca_long_context_widths_cover_one_million_tokens(self):
        for source_width, expected_width in (
            (0, 2),
            (1, 2),
            (1562, 2048),
            (2049, 4096),
            (4097, 8192),
            (8192, 8192),
        ):
            with self.subTest(source_width=source_width):
                indices = torch.arange(source_width, dtype=torch.int32).view(1, -1)
                canonical, lengths = canonical_topk(
                    indices, None, SM120_EXTRA_TOPK_WIDTHS
                )
                self.assertEqual(canonical.shape, (1, expected_width))
                self.assertEqual(lengths.tolist(), [source_width])

        with self.assertRaisesRegex(RuntimeError, "exceeds the largest"):
            canonical_topk(
                torch.zeros(1, 8193, dtype=torch.int32),
                None,
                SM120_EXTRA_TOPK_WIDTHS,
            )


class Sm120IndexerFallbackTest(unittest.TestCase):
    @staticmethod
    def _pool(num_blocks: int, block_size: int) -> torch.Tensor:
        # Zero K/scales are sufficient to exercise address/mask logic; the
        # expected score for every valid token is therefore exactly zero.
        pool = torch.zeros(
            num_blocks * block_size,
            132,
            dtype=torch.uint8,
        )
        return pool

    def test_invalid_block_table_slots_are_masked(self):
        q = torch.ones(1, 1, 1, 128, dtype=torch.float8_e4m3fn)
        weights = torch.ones(1, 1, dtype=torch.float32)
        pool = self._pool(num_blocks=2, block_size=2)
        # The second logical block is padding.  Its positions must not be
        # turned into scores by clamping -1 to physical block zero.
        block_table = torch.tensor([[0, -1]], dtype=torch.int32)
        lengths = torch.tensor([[4]], dtype=torch.int32)

        out = fp8_paged_indexer_score(
            q,
            weights,
            pool,
            block_table,
            lengths,
            block_size=2,
            max_ctx_len=4,
        )
        self.assertTrue(torch.isfinite(out[0, :2]).all())
        self.assertTrue(torch.isneginf(out[0, 2:]).all())

    def test_context_length_masks_tail_without_large_intermediate(self):
        q = torch.ones(2, 1, 1, 128, dtype=torch.float8_e4m3fn)
        weights = torch.ones(2, 1, dtype=torch.float32)
        pool = self._pool(num_blocks=4, block_size=2)
        block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        lengths = torch.tensor([[1], [3]], dtype=torch.int32)

        out = fp8_paged_indexer_score(
            q,
            weights,
            pool,
            block_table,
            lengths,
            block_size=2,
            max_ctx_len=4,
        )
        self.assertTrue(torch.isfinite(out[0, :1]).all())
        self.assertTrue(torch.isneginf(out[0, 1:]).all())
        self.assertTrue(torch.isfinite(out[1, :3]).all())
        self.assertTrue(torch.isneginf(out[1, 3:]).all())

    def test_block_table_capacity_is_validated(self):
        q = torch.ones(1, 1, 1, 128, dtype=torch.float8_e4m3fn)
        weights = torch.ones(1, 1, dtype=torch.float32)
        pool = self._pool(num_blocks=1, block_size=2)
        with self.assertRaisesRegex(ValueError, "block-table capacity"):
            fp8_paged_indexer_score(
                q,
                weights,
                pool,
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([[2]], dtype=torch.int32),
                block_size=2,
                max_ctx_len=3,
            )

    def test_sm120_mqa_finalize_relative_max_seqlen_k(self):
        out = torch.tensor(
            [
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            ],
            dtype=torch.float32,
        )
        starts = torch.tensor([1, 3], dtype=torch.int32)
        ends = torch.tensor([4, 5], dtype=torch.int32)

        rel_4 = _slice_relative_mqa_logits(
            out.clone(), starts, ends, clean_logits=False, max_seqlen_k=4
        )
        expected_4 = torch.tensor(
            [
                [11.0, 12.0, 13.0, float("-inf")],
                [23.0, 24.0, float("-inf"), float("-inf")],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(rel_4, expected_4, equal_nan=True)

        rel_2 = _slice_relative_mqa_logits(
            out.clone(), starts, ends, clean_logits=False, max_seqlen_k=2
        )
        torch.testing.assert_close(
            rel_2,
            torch.tensor([[11.0, 12.0], [23.0, 24.0]], dtype=torch.float32),
        )

        cleaned = _slice_relative_mqa_logits(
            out.clone(), starts, ends, clean_logits=True, max_seqlen_k=0
        )
        expected_clean = torch.tensor(
            [
                [float("-inf"), 11.0, 12.0, 13.0, float("-inf"), float("-inf")],
                [
                    float("-inf"),
                    float("-inf"),
                    float("-inf"),
                    23.0,
                    24.0,
                    float("-inf"),
                ],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(cleaned, expected_clean, equal_nan=True)

        empty_cols = _slice_relative_mqa_logits(
            torch.empty((2, 0), dtype=torch.float32),
            starts,
            ends,
            clean_logits=False,
            max_seqlen_k=4,
        )
        self.assertEqual(tuple(empty_cols.shape), (2, 4))
        self.assertTrue(torch.isneginf(empty_cols).all())

        with self.assertRaisesRegex(ValueError, "max_seqlen_k"):
            _slice_relative_mqa_logits(
                out, starts, ends, clean_logits=False, max_seqlen_k=-1
            )


class AttentionFp8OomRetryTest(unittest.TestCase):
    def test_swa_oom_retry_releases_failed_frame_before_empty_cache(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        layer = AttentionFP8.__new__(AttentionFP8)
        torch.nn.Module.__init__(layer)
        layer.oom_defrag_retry = True
        layer._dispose_kv_full = lambda *_args, **_kwargs: None

        sentinel_ref = None
        expected_out = torch.ones((2, 4), dtype=torch.bfloat16)
        attempts = 0
        observations = []

        class _HeldBuffer:
            pass

        def flaky_forward(**_kwargs):
            nonlocal attempts, sentinel_ref
            attempts += 1
            if attempts == 1:
                buf = _HeldBuffer()
                sentinel_ref = weakref.ref(buf)
                raise torch.OutOfMemoryError("synthetic OOM after output alloc")
            return expected_out

        layer._flash_mla_sparse_fwd_chunked_projected = flaky_forward
        qkv = SimpleNamespace(
            q=SimpleNamespace(device=torch.device("cuda", 0)),
            kv_full=torch.empty((2, 4), dtype=torch.bfloat16),
        )
        common = SimpleNamespace(
            cp_on=False,
            topk_idxs=torch.zeros((2, 1), dtype=torch.int32),
            swa_meta=SimpleNamespace(
                topk_length_kv_full=torch.ones((2,), dtype=torch.int32)
            ),
            freqs_cis=None,
            workspace=None,
        )

        def fake_empty_cache():
            gc.collect()
            observations.append(
                (
                    sys.exc_info()[1] is None,
                    sentinel_ref is not None and sentinel_ref() is None,
                )
            )

        with (
            patch("torch.cuda.device", side_effect=lambda *_a: nullcontext()),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.empty_cache", side_effect=fake_empty_cache),
        ):
            actual = layer._attn_fp8_swa_via_kv_full(qkv, common)

        self.assertIs(actual, expected_out)
        self.assertEqual(attempts, 2)
        self.assertEqual(observations, [(True, True)])


class _TrackingLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(self, *args):
        self._lock.release()
        return False


class Sm120SparseMlaWorkspaceAndProbeTest(unittest.TestCase):
    def test_workspace_separates_eager_streams_owners_and_graph_scratch(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        stream_id = 101
        capturing = False
        warmup_enabled = False
        alloc_calls = 0

        def fake_empty(size, dtype=None, device=None):
            nonlocal alloc_calls
            alloc_calls += 1
            return torch.tensor([alloc_calls], dtype=torch.uint8)

        owner_a = sm120_sparse_mla.Sm120SparseMlaWorkspace(max_eager_streams=2)
        owner_b = sm120_sparse_mla.Sm120SparseMlaWorkspace(max_eager_streams=2)
        dev = torch.device("cuda", 0)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=lambda: capturing,
            ),
            patch(
                "torch.cuda.current_stream",
                side_effect=lambda _d=None: SimpleNamespace(cuda_stream=stream_id),
            ),
            patch.object(
                sm120_sparse_mla,
                "cuda_graph_warmup_forward_enabled",
                side_effect=lambda: warmup_enabled,
            ),
            patch("torch.empty", side_effect=fake_empty),
        ):
            ws_s1 = sm120_sparse_mla.workspace(dev, owner=owner_a)
            stream_id = 202
            ws_s2 = sm120_sparse_mla.workspace(dev, owner=owner_a)
            self.assertIsNot(ws_s1, ws_s2)

            # Eager decode uses only the current eager stream buffer and
            # never materializes an unused graph buffer on eager-only lanes.
            self.assertIsNone(owner_a._buffers.get(("cuda", 0, None)))

            # Capture after only eager run() fails fast on an un-warmed graph slot.
            capturing = True
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                sm120_sparse_mla.workspace(dev, owner=owner_a)

            capturing = False
            ws_graph_a = sm120_sparse_mla.warmup(dev, owner=owner_a)
            ws_graph_b = sm120_sparse_mla.warmup(dev, owner=owner_b)
            self.assertIsNot(ws_graph_a, ws_s1)
            self.assertIsNot(ws_graph_a, ws_s2)
            self.assertIsNot(ws_graph_a, ws_graph_b)

            capturing = True
            self.assertIs(sm120_sparse_mla.workspace(dev, owner=owner_a), ws_graph_a)
            self.assertIs(sm120_sparse_mla.workspace(dev, owner=owner_b), ws_graph_b)

            # Warmup forward on owner_b populates graph scratch cleanly too.
            capturing = False
            warmup_enabled = True
            owner_c = sm120_sparse_mla.Sm120SparseMlaWorkspace(max_eager_streams=2)
            ws_graph_c = sm120_sparse_mla.workspace(dev, owner=owner_c)
            capturing = True
            warmup_enabled = False
            self.assertIs(sm120_sparse_mla.workspace(dev, owner=owner_c), ws_graph_c)

            # LRU eviction caps eager stream slots without dropping graph scratch.
            capturing = False
            for sid in (303, 404, 505):
                stream_id = sid
                sm120_sparse_mla.workspace(dev, owner=owner_a)
            eager_keys = [
                k for k in owner_a._buffers if k[:2] == ("cuda", 0) and k[2] is not None
            ]
            self.assertEqual(eager_keys, [("cuda", 0, 404), ("cuda", 0, 505)])
            self.assertIs(owner_a._buffers[("cuda", 0, None)], ws_graph_a)

    def test_ownerless_default_workspace_is_eager_only(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        capturing = False
        warmup_enabled = False
        allocs = []

        def fake_empty(size, dtype=None, device=None):
            buf = torch.tensor([len(allocs) + 1], dtype=torch.uint8)
            allocs.append(buf)
            return buf

        dev = torch.device("cuda", 0)
        saved_default_buffers = dict(sm120_sparse_mla._DEFAULT_WORKSPACE._buffers)
        sm120_sparse_mla._DEFAULT_WORKSPACE._buffers.clear()
        try:
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.device", return_value=nullcontext()),
                patch(
                    "torch.cuda.is_current_stream_capturing",
                    side_effect=lambda: capturing,
                ),
                patch(
                    "torch.cuda.current_stream",
                    side_effect=lambda _d=None: SimpleNamespace(cuda_stream=707),
                ),
                patch.object(
                    sm120_sparse_mla,
                    "cuda_graph_warmup_forward_enabled",
                    side_effect=lambda: warmup_enabled,
                ),
                patch("torch.empty", side_effect=fake_empty),
            ):
                eager_ws = sm120_sparse_mla.workspace(dev)
                self.assertIn(
                    ("cuda", 0, 707), sm120_sparse_mla._DEFAULT_WORKSPACE._buffers
                )
                self.assertIsNotNone(eager_ws)

                with self.assertRaisesRegex(RuntimeError, "explicit"):
                    sm120_sparse_mla.warmup(dev)

                warmup_enabled = True
                with self.assertRaisesRegex(RuntimeError, "explicit"):
                    sm120_sparse_mla.workspace(dev)

                warmup_enabled = False
                capturing = True
                with self.assertRaisesRegex(RuntimeError, "explicit"):
                    sm120_sparse_mla.workspace(dev)
        finally:
            sm120_sparse_mla._DEFAULT_WORKSPACE._buffers.clear()
            sm120_sparse_mla._DEFAULT_WORKSPACE._buffers.update(saved_default_buffers)

    def test_independent_ownerless_prepare_prefill_handles_isolate_graph_workspaces(
        self,
    ):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla
        from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import (
            SparsePrefillInputs,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        capturing = False
        warmup_enabled = False
        allocs = []
        recorded_workspaces = []

        def fake_empty(size, dtype=None, device=None):
            buf = torch.tensor([len(allocs) + 1], dtype=torch.uint8)
            allocs.append(buf)
            return buf

        def fake_provider(**kwargs):
            recorded_workspaces.append(kwargs["workspace_buffer"])

        q = torch.zeros(1, 16, 512, dtype=torch.bfloat16)
        cache = torch.zeros(2, 64, 584, dtype=torch.uint8)
        indices = torch.zeros(1, 128, dtype=torch.int32)
        lengths = torch.ones(1, dtype=torch.int32)
        sink = torch.zeros(16, dtype=torch.float32)
        real_workspace_key = sm120_sparse_mla._workspace_key

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=lambda: capturing,
            ),
            patch(
                "torch.cuda.current_stream",
                side_effect=lambda _d=None: SimpleNamespace(cuda_stream=808),
            ),
            patch.object(
                sm120_sparse_mla,
                "cuda_graph_warmup_forward_enabled",
                side_effect=lambda: warmup_enabled,
            ),
            patch.object(
                sm120_sparse_mla,
                "_workspace_key",
                side_effect=lambda _dev, **kw: real_workspace_key(
                    torch.device("cuda", 0), **kw
                ),
            ),
            patch.object(
                sm120_sparse_mla,
                "_load_sm120_sparse_mla",
                return_value=fake_provider,
            ),
            patch("torch.empty", side_effect=fake_empty),
        ):
            handle_a = SM120_ATTENTION.prepare_prefill(
                n_heads=16,
                q=q,
                kv=None,
                indices=indices,
                topk_length=lengths,
                head_dim=512,
                softmax_scale=0.25,
                attn_sink=sink,
                prefill_workspace=None,
                backend_inputs=SparsePrefillInputs(swa_cache=cache),
            )
            handle_b = SM120_ATTENTION.prepare_prefill(
                n_heads=16,
                q=q,
                kv=None,
                indices=indices,
                topk_length=lengths,
                head_dim=512,
                softmax_scale=0.25,
                attn_sink=sink,
                prefill_workspace=None,
                backend_inputs=SparsePrefillInputs(swa_cache=cache),
            )
            warmup_enabled = True
            handle_a.forward(0, 1)
            handle_b.forward(0, 1)
            warmup_enabled = False
            capturing = True
            handle_a.forward(0, 1)
            handle_b.forward(0, 1)

            handle_c = SM120_ATTENTION.prepare_prefill(
                n_heads=16,
                q=q,
                kv=None,
                indices=indices,
                topk_length=lengths,
                head_dim=512,
                softmax_scale=0.25,
                attn_sink=sink,
                prefill_workspace=None,
                backend_inputs=SparsePrefillInputs(swa_cache=cache),
            )
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                handle_c.forward(0, 1)

        self.assertEqual(len(allocs), 2)
        self.assertEqual(len(recorded_workspaces), 4)
        ws_a_warm, ws_b_warm, ws_a_cap, ws_b_cap = recorded_workspaces
        self.assertIsNot(ws_a_warm, ws_b_warm)
        self.assertIs(ws_a_cap, ws_a_warm)
        self.assertIs(ws_b_cap, ws_b_warm)

    def test_explicit_eager_hint_rejected_during_capture_and_warmup(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        capturing = False
        warmup_enabled = False
        allocs = []

        def fake_empty(size, dtype=None, device=None):
            buf = torch.tensor([len(allocs) + 1], dtype=torch.uint8)
            allocs.append(buf)
            return buf

        owner = sm120_sparse_mla.Sm120SparseMlaWorkspace(max_eager_streams=2)
        dev = torch.device("cuda", 0)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=lambda: capturing,
            ),
            patch(
                "torch.cuda.current_stream",
                side_effect=lambda _d=None: SimpleNamespace(cuda_stream=909),
            ),
            patch.object(
                sm120_sparse_mla,
                "cuda_graph_warmup_forward_enabled",
                side_effect=lambda: warmup_enabled,
            ),
            patch("torch.empty", side_effect=fake_empty),
        ):
            ws_eager = sm120_sparse_mla.workspace(dev, owner=owner)
            self.assertIs(ws_eager, allocs[0])

            warmup_enabled = True
            with self.assertRaisesRegex(RuntimeError, "eager.*during CUDA graph"):
                sm120_sparse_mla.workspace(dev, for_graph=False, owner=owner)

            warmup_enabled = False
            capturing = True
            with self.assertRaisesRegex(RuntimeError, "eager.*during CUDA graph"):
                sm120_sparse_mla.workspace(dev, for_graph=False, owner=owner)

    def test_capture_check_uses_target_device_not_ambient_device(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        current_dev = 0
        capturing_by_device = {0: False, 1: False}
        warmup_enabled = False
        allocs = []

        class _FakeCudaDevice:
            def __init__(self, target):
                self._target = int(torch.device(target).index)
                self._prev = None

            def __enter__(self):
                nonlocal current_dev
                self._prev = current_dev
                current_dev = self._target
                return self

            def __exit__(self, *_args):
                nonlocal current_dev
                current_dev = self._prev
                return False

        def fake_empty(size, dtype=None, device=None):
            buf = torch.tensor([len(allocs) + 1], dtype=torch.uint8)
            allocs.append(buf)
            return buf

        owner = sm120_sparse_mla.Sm120SparseMlaWorkspace(max_eager_streams=2)
        dev1 = torch.device("cuda", 1)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device", side_effect=_FakeCudaDevice),
            patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=lambda: capturing_by_device[current_dev],
            ),
            patch(
                "torch.cuda.current_stream",
                side_effect=lambda _d=None: SimpleNamespace(cuda_stream=1010),
            ),
            patch.object(
                sm120_sparse_mla,
                "cuda_graph_warmup_forward_enabled",
                side_effect=lambda: warmup_enabled,
            ),
            patch("torch.empty", side_effect=fake_empty),
        ):
            ws_eager_1 = sm120_sparse_mla.workspace(dev1, owner=owner)
            self.assertIs(ws_eager_1, allocs[0])

            capturing_by_device[1] = True
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                sm120_sparse_mla.workspace(dev1, owner=owner)
            with self.assertRaisesRegex(RuntimeError, "eager.*during CUDA graph"):
                sm120_sparse_mla.workspace(dev1, for_graph=False, owner=owner)
            self.assertEqual(current_dev, 0)

            capturing_by_device[1] = False
            ws_graph_1 = sm120_sparse_mla.warmup(dev1, owner=owner)
            self.assertIsNot(ws_graph_1, ws_eager_1)

            capturing_by_device[1] = True
            self.assertIs(sm120_sparse_mla.workspace(dev1, owner=owner), ws_graph_1)
            self.assertEqual(current_dev, 0)

    def test_chunked_fallback_skips_cuda_host_sync_and_ambient_capture_check(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        query = torch.ones((1, 2, 512), dtype=torch.float32)
        indices = torch.tensor([[0, 1]], dtype=torch.int32)
        lengths = torch.tensor([2], dtype=torch.int32)
        actual = torch.empty_like(query)

        # Fake CUDA query/sinks tensors without CUDA hardware to verify
        # run_chunked_reference never invokes ambient capture checks or
        # host-syncing torch.isfinite on CUDA tensors.
        fake_sinks = Mock()
        fake_sinks.is_cuda = True
        fake_sinks.to.return_value = torch.zeros((2,), dtype=torch.float32)

        with (
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError("ambient capture check"),
            ),
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton."
                "dequantize_slots_to_bf16",
                return_value=torch.zeros((2, 512), dtype=torch.float32),
            ),
        ):
            run_chunked_reference(
                query=query,
                swa_cache=torch.empty(0),
                swa_indices=indices,
                swa_lens=lengths,
                out=actual,
                scale=0.25,
                sinks=fake_sinks,
            )

    def test_prefill_wide_swa_routes_chunks_above_64_to_chunked_fallback(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla
        from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import (
            SparsePrefillInputs,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        rows, heads, dim = 65, 16, 512
        q = torch.zeros((rows, heads, dim), dtype=torch.bfloat16)
        sink = torch.zeros((heads,), dtype=torch.float32)
        cache = torch.zeros((2, 64, 584), dtype=torch.uint8)
        lengths = torch.full((rows,), 64, dtype=torch.int32)

        for width, expect_fallback in ((1024, False), (2048, True)):
            with self.subTest(width=width):
                indices = torch.zeros((rows, width), dtype=torch.int32)
                with (
                    patch.object(sm120_sparse_mla, "run") as native_runner,
                    patch.object(
                        sm120_sparse_mla, "run_chunked_reference"
                    ) as fallback_runner,
                ):
                    handle = SM120_ATTENTION.prepare_prefill(
                        n_heads=heads,
                        q=q,
                        kv=None,
                        indices=indices,
                        topk_length=lengths,
                        head_dim=dim,
                        softmax_scale=0.25,
                        attn_sink=sink,
                        prefill_workspace=None,
                        backend_inputs=SparsePrefillInputs(swa_cache=cache),
                    )
                    handle.forward(0, rows)
                if expect_fallback:
                    fallback_runner.assert_called_once()
                    native_runner.assert_not_called()
                else:
                    native_runner.assert_called_once()
                    fallback_runner.assert_not_called()

    def test_module_does_not_import_fused_moe_at_top_level(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        tree = ast.parse(Path(sm120_sparse_mla.__file__).read_text())
        top_modules = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_modules.append(node.module)
        self.assertFalse(any("fused_moe" in mod for mod in top_modules), top_modules)

    def test_concurrent_first_probe_waits_for_provider_load(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        sentinel = object()
        entered = threading.Event()
        release = threading.Event()
        tracking_lock = _TrackingLock()

        class _SlowDecodeModule(types.ModuleType):
            def __getattr__(self, name):
                if name == "trtllm_batch_decode_sparse_mla_dsv4":
                    entered.set()
                    assert release.wait(timeout=5)
                    return sentinel
                raise AttributeError(name)

        fake_decode = _SlowDecodeModule("flashinfer.decode")
        fake_fi = types.ModuleType("flashinfer")
        fake_fi.decode = fake_decode

        saved_fn = sm120_sparse_mla._sm120_sparse_mla
        saved_attempted = sm120_sparse_mla._sm120_sparse_mla_load_attempted
        try:
            sm120_sparse_mla._sm120_sparse_mla = None
            sm120_sparse_mla._sm120_sparse_mla_load_attempted = False
            results = [None, None]
            errors = []

            def worker(idx):
                try:
                    results[idx] = sm120_sparse_mla._load_sm120_sparse_mla()
                except BaseException as exc:  # pragma: no cover
                    errors.append(exc)

            with (
                patch.dict(
                    sys.modules,
                    {"flashinfer": fake_fi, "flashinfer.decode": fake_decode},
                ),
                patch.object(sm120_sparse_mla, "_SM120_SPARSE_MLA_LOCK", tracking_lock),
            ):
                t0 = threading.Thread(target=worker, args=(0,))
                t0.start()
                self.assertTrue(entered.wait(timeout=5))
                t1 = threading.Thread(target=worker, args=(1,))
                t1.start()
                self.assertTrue(tracking_lock.contended.wait(timeout=5))
                release.set()
                t0.join(timeout=5)
                t1.join(timeout=5)
            self.assertEqual(errors, [])
            self.assertIs(results[0], sentinel)
            self.assertIs(results[1], sentinel)
        finally:
            sm120_sparse_mla._sm120_sparse_mla = saved_fn
            sm120_sparse_mla._sm120_sparse_mla_load_attempted = saved_attempted

    def test_forward_decode_empty_rows_and_out_of_pool_slot_exclusion(self) -> None:
        empty_q = torch.empty((0, 1, 16, 512), dtype=torch.bfloat16)
        empty_idx = torch.empty((0, 1, 128), dtype=torch.int32)
        kv_cache = torch.zeros((2, 64, 584), dtype=torch.uint8)
        sink = torch.zeros((16,), dtype=torch.float32)
        empty_out = sm120_sparse_mla.forward_decode(
            empty_q, kv_cache, sink, empty_idx, softmax_scale=0.1
        )
        self.assertEqual(tuple(empty_out.shape), (0, 1, 16, 512))

        pool_capacity = 2 * 64
        canon, eff_lens = sm120_sparse_mla.canonical_topk(
            torch.tensor([[0, pool_capacity, -1], [pool_capacity + 5, 1, 2]], dtype=torch.int32),
            None,
            (3,),
            max_slot=pool_capacity,
        )
        self.assertEqual(eff_lens.tolist(), [1, 2])
        self.assertEqual(canon.tolist(), [[0, -1, -1], [1, 2, -1]])

    def test_run_chunked_reference_masks_tail_and_out_of_pool_slots_before_dequant(self) -> None:
        seen_slots = []

        def fake_dequant(cache, flat_slots):
            seen_slots.append(flat_slots.clone())
            if bool((flat_slots >= 128).any()):
                raise AssertionError(
                    "trap: out-of-pool slot reached dequantize_slots_to_bf16"
                )
            out = torch.zeros((flat_slots.numel(), 512), dtype=torch.bfloat16)
            # Slot 5 is in-pool (< 128) yet past row 0's length=1; returning NaN
            # catches any mask failure where masked tail weights still compute 0 * NaN.
            out[flat_slots == 5] = float("nan")
            return out

        query = torch.randn((2, 16, 512), dtype=torch.bfloat16)
        cache = torch.zeros((2, 64, 584), dtype=torch.uint8)
        indices = torch.tensor([[1, 999, 5], [1000, -1, 3]], dtype=torch.int32)
        lengths = torch.tensor([1, 2], dtype=torch.int32)
        out = torch.empty_like(query)
        with patch.dict(
            sys.modules,
            {
                "rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton": types.SimpleNamespace(
                    dequantize_slots_to_bf16=fake_dequant
                )
            },
        ):
            sm120_sparse_mla.run_chunked_reference(
                query=query,
                swa_cache=cache,
                swa_indices=indices,
                swa_lens=lengths,
                out=out,
                scale=0.1,
                sinks=torch.zeros((16,), dtype=torch.float32),
            )
        self.assertEqual(len(seen_slots), 1)
        self.assertTrue(torch.isfinite(out).all())
        # Verify both out-of-pool slots (999, 1000) and past-length in-pool slots (5, 3)
        # were masked to -1 before reaching dequantize_slots_to_bf16.
        self.assertEqual(seen_slots[0].tolist(), [1, -1, -1, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
