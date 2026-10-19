"""Decode token K512 dispatch, score precision, and changing graph inputs."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as selector
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_indexer import make_case
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_deepselect import (
    assert_equivalent as assert_prefill_equivalent,
)
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_deepselect import (
    reference as prefill_reference,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _static_select_candidates(logits, lengths, block_size, topk_blocks):
    """Reference: the padded static pool kernel + identical selection chain.

    Mirrors selector.select_candidates but always launches the capacity-wide
    _candidate_pool_kernel, so the bounded production path can be compared
    against the historical padded scores end to end.
    """
    rows, width = logits.shape
    nblocks = selector.triton.cdiv(width, block_size)
    count = min(topk_blocks, nblocks)
    score_stride = selector.triton.cdiv(nblocks, 256) * 256
    scores = torch.empty(
        (rows, score_stride), dtype=torch.float32, device=logits.device
    )[:, :nblocks]
    block_lengths = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    selector._candidate_pool_kernel[(rows, selector.triton.cdiv(nblocks, 128))](
        logits,
        lengths,
        scores,
        block_lengths,
        width,
        logits.stride(0),
        block_size,
        nblocks,
        score_stride,
        128,
    )
    candidates = torch.empty((rows, count), dtype=torch.int32, device=logits.device)
    if nblocks > count:
        rtp_llm_ops.topk_v3(
            scores,
            block_lengths,
            candidates,
            selector._get_topk_workspace(logits.device),
            count,
            nblocks,
        )
    flags = torch.empty((rows, nblocks), dtype=torch.uint8, device=logits.device)
    selector._candidate_store_flags_kernel[(rows,)](
        scores, candidates, flags, count, nblocks, score_stride
    )
    return candidates, flags


def reference(logits, ends):
    """Decode preserves every visible index on short rows, including mask holes."""
    result = prefill_reference(logits, ends)
    dense = torch.arange(512, device=logits.device)[None].expand_as(result)
    dense = torch.where(dense < ends[:, None].clamp(0, logits.shape[1]), dense, -1)
    return torch.where((ends <= 512)[:, None], dense, result).int()


def assert_equivalent(case, actual, expected, logits, ends):
    short = ends <= 512
    torch.testing.assert_close(
        actual[short], reference(logits, ends)[short], rtol=0, atol=0
    )
    long = ~short
    if long.any().item():
        assert_prefill_equivalent(
            case, actual[long], expected[long], logits[long], ends[long]
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeTopKTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            self.skipTest("DeepSelect requires SM100/SM103")
        torch.manual_seed(714)
        env = patch.dict(os.environ, {"DSV41_PREFILL_DEEPSELECT": "0"})
        env.start()
        self.addCleanup(env.stop)
        # Default-on decode selection must not depend on the prefill switch.
        previous = os.environ.pop("DSV41_DECODE_DEEPSELECT", None)
        self.addCleanup(self.restore_switch, previous)

    def require_native(self, dtype):
        name = (
            "deepselect_bf16_available"
            if dtype == torch.bfloat16
            else "deepselect_fp32_available"
        )
        # This suite explicitly targets SM100/SM103 builds with native DeepSelect.
        self.assertTrue(getattr(rtp_llm_ops, name, lambda: False)(), name + " missing")

    @staticmethod
    def restore_switch(previous):
        if previous is None:
            os.environ.pop("DSV41_DECODE_DEEPSELECT", None)
        else:
            os.environ["DSV41_DECODE_DEEPSELECT"] = previous

    def test_decode_uses_native_dense_deepselect_for_both_score_dtypes(self):
        for dtype, name in (
            (torch.float32, "deepselect_fp32"),
            (torch.bfloat16, "deepselect_bf16_dense"),
        ):
            self.require_native(dtype)
            logits = torch.randn(6, 4096, device="cuda", dtype=dtype)
            ends = torch.tensor(
                [0, 1, 511, 512, 513, 4096], device="cuda", dtype=torch.int32
            )
            with self.subTest(dtype=dtype), patch.object(
                rtp_llm_ops, name, wraps=getattr(rtp_llm_ops, name)
            ) as native, patch.object(
                rtp_llm_ops,
                "topk_v3",
                side_effect=AssertionError("decode skipped DeepSelect"),
            ):
                result = selector.select_tokens(logits, ends, 512)
                native.assert_called_once()
                self.assertIs(native.call_args.args[0], logits)
                assert_equivalent(self, result, reference(logits, ends), logits, ends)
                torch.testing.assert_close(
                    result[3], torch.arange(512, device="cuda").int()
                )

    def test_opt_out_and_missing_backend_retain_radix_fallback(self):
        logits = torch.randn(6, 4096, device="cuda")
        ends = torch.full((6,), 4096, device="cuda", dtype=torch.int32)
        for disable in (True, False):
            with self.subTest(disable=disable), patch.dict(
                os.environ, {"DSV41_DECODE_DEEPSELECT": "0" if disable else "1"}
            ), patch.object(
                rtp_llm_ops,
                "deepselect_fp32_available",
                return_value=disable,
                create=True,
            ), patch.object(
                rtp_llm_ops, "topk_v3", wraps=rtp_llm_ops.topk_v3
            ) as fallback:
                result = selector.select_tokens(logits, ends, 512)
                fallback.assert_called_once()
                assert_equivalent(self, result, reference(logits, ends), logits, ends)

    def test_missing_both_native_backends_still_runs_bf16_radix(self):
        logits = torch.randn(6, 1024, device="cuda", dtype=torch.bfloat16)
        logits[:, :64] = -float("inf")
        ends = torch.tensor(
            [0, 1, 128, 511, 512, 1024], device="cuda", dtype=torch.int32
        )
        with patch.object(
            rtp_llm_ops, "deepselect_fp32_available", return_value=False, create=True
        ), patch.object(
            rtp_llm_ops, "deepselect_bf16_available", return_value=False, create=True
        ), patch.object(
            rtp_llm_ops, "topk_v3", wraps=rtp_llm_ops.topk_v3
        ) as fallback:
            result = selector.select_tokens(logits, ends, 512)
            fallback.assert_called_once()
            assert_equivalent(self, result, reference(logits, ends), logits, ends)

    def test_unsupported_bf16_layout_and_other_k_use_radix(self):
        for dtype, width, k in (
            (torch.bfloat16, 768, 512),
            (torch.float32, 2048, 1024),
        ):
            logits = torch.randn(6, width, device="cuda", dtype=dtype)
            ends = torch.full((6,), width, device="cuda", dtype=torch.int32)
            with patch.object(
                rtp_llm_ops, "topk_v3", wraps=rtp_llm_ops.topk_v3
            ) as fallback:
                actual = selector.select_tokens(logits, ends, k)
                fallback.assert_called_once()
                values = logits.gather(1, actual.long()).sort(-1).values
                torch.testing.assert_close(
                    values, logits.topk(k).values.sort(-1).values, rtol=0, atol=0
                )

    def test_native_execution_error_is_not_hidden_by_fallback(self):
        self.require_native(torch.float32)
        logits = torch.randn(6, 4096, device="cuda")
        ends = torch.full((6,), 4096, device="cuda", dtype=torch.int32)
        with patch.object(
            rtp_llm_ops, "deepselect_fp32", side_effect=RuntimeError("native failure")
        ), patch.object(
            rtp_llm_ops, "topk_v3", side_effect=AssertionError("unexpected fallback")
        ):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                selector.select_tokens(logits, ends, 512)

    def test_production_rows_lengths_nonfinite_and_cutoff_ties(self):
        for dtype in (torch.bfloat16, torch.float32):
            self.require_native(dtype)
            logits = torch.randn(192, 32768, device="cuda", dtype=dtype)
            # Exercise short rows, full rows, clamping, and different MTP positions.
            ends = torch.arange(192, device="cuda", dtype=torch.int32) * 127 + 513
            ends[:6] = torch.tensor([-1, 0, 511, 512, 513, 40000], device="cuda")
            logits[6:9, :20] = float("nan")
            logits[9:12, :20] = float("inf")
            logits[12:15, :20] = -float("inf")
            logits[15:18, :1024] = 10
            result = selector.select_tokens(logits, ends, 512)
            assert_equivalent(self, result, reference(logits, ends), logits, ends)

    def test_graph_replay_refreshes_scores_and_causal_lengths(self):
        for dtype in (torch.bfloat16, torch.float32):
            self.require_native(dtype)
            logits = torch.randn(160, 32768, device="cuda", dtype=dtype)
            ends = torch.full((160,), 13000, device="cuda", dtype=torch.int32)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                selector.select_tokens(logits, ends, 512)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = selector.select_tokens(logits, ends, 512)
            for length in (0, 511, 512, 513, 13000, 32768):
                logits.normal_()
                logits[:, :64] = -float("inf")
                ends.fill_(length)
                graph.replay()
                assert_equivalent(self, result, reference(logits, ends), logits, ends)

    def test_bf16_candidate_publication_keeps_original_block_membership(self):
        logits = torch.randn(192, 32768, device="cuda", dtype=torch.bfloat16)
        logits[:3].zero_()
        ends = torch.full((192,), 32768, device="cuda", dtype=torch.int32)
        expected = selector.select_candidates(logits.float(), ends, 8, 2048)
        actual = selector.select_candidates(logits, ends, 8, 2048)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_candidate_pool_bounded_elides_padding_and_matches_static(self):
        """Production-shape pooling: width >> visible, count=2048 (topk_v3 path).

        The bounded row-loop kernel must (a) write bitwise-identical scores on
        the visible blocks and identical block_lengths, (b) leave the >=
        block_lengths tail unwritten while the static kernel pads it with -inf,
        and (c) drive select_candidates to the exact same candidates+flags as
        the padded static chain, with every selected block < block_lengths.
        """
        rows, width, block_size, count = 192, 524288, 8, 2048
        nblocks = selector.triton.cdiv(width, block_size)
        score_stride = selector.triton.cdiv(nblocks, 256) * 256
        torch.manual_seed(1522)
        logits = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        # Mixed lengths: production ~32.8k, plus short/full/zero/non-multiple-of-8
        # rows to exercise every bounding edge. ends never exceeds width in
        # production (visible_per_token <= capacity), so the full row uses width
        # and the near-full row width-1 (both pin the newest block identically).
        ends = torch.full((rows,), 32820, device="cuda", dtype=torch.int32)
        ends += (torch.arange(rows, device="cuda", dtype=torch.int32) % 11) - 5
        ends[:8] = torch.tensor(
            [0, 1, 7, 8, 2047, 16384, width, width - 1], device="cuda"
        ).to(torch.int32)
        ends = torch.clamp(ends, min=0)
        # Poison the allocator tail so any unbounded read would surface garbage.
        logits[:, 40000:] = float("nan")

        # Pre-fill the bounded score buffer with a sentinel so elision of the
        # >= block_lengths tail is proven deterministically (not left to chance
        # allocator contents).
        s_b = torch.full(
            (rows, score_stride), 777.0, device="cuda", dtype=torch.float32
        )
        b_b = torch.empty(rows, device="cuda", dtype=torch.int32)
        selector._candidate_pool_bounded_kernel[(rows,)](
            logits,
            ends,
            s_b,
            b_b,
            width,
            logits.stride(0),
            block_size,
            score_stride,
            256,
            num_warps=8,
        )
        s_s = torch.empty(rows, score_stride, device="cuda", dtype=torch.float32)
        b_s = torch.empty(rows, device="cuda", dtype=torch.int32)
        selector._candidate_pool_kernel[(rows, selector.triton.cdiv(nblocks, 128))](
            logits,
            ends,
            s_s,
            b_s,
            width,
            logits.stride(0),
            block_size,
            nblocks,
            score_stride,
            128,
        )
        torch.cuda.synchronize()
        # (a) block_lengths identical; visible-block scores bitwise identical
        #     (equal_nan: full rows pool the poisoned tail identically).
        torch.testing.assert_close(b_b, b_s, rtol=0, atol=0)
        for r in range(rows):
            n = int(b_s[r].item())
            torch.testing.assert_close(
                s_b[r, :n], s_s[r, :n], rtol=0, atol=0, equal_nan=True
            )
        # (b) static pads the tail with -inf; the bounded kernel leaves a
        #     representative elided row's tail at the sentinel (no write).
        r = 10
        n = int(b_s[r].item())
        self.assertLess(n, nblocks)
        self.assertTrue(bool((s_s[r, n:] == -float("inf")).all()))
        self.assertTrue(bool((s_b[r, n:] == 777.0).all()))
        # (c) end-to-end select_candidates (auto-selects bounded) == static chain.
        cand_b, flags_b = selector.select_candidates(logits, ends, block_size, count)
        cand_s, flags_s = _static_select_candidates(logits, ends, block_size, count)
        torch.testing.assert_close(cand_b, cand_s, rtol=0, atol=0)
        torch.testing.assert_close(flags_b, flags_s, rtol=0, atol=0)
        # Every selected block is within the row's visible span (no padding leak).
        valid = cand_b >= 0
        bounds = b_s[:, None].expand_as(cand_b)
        self.assertTrue(bool((cand_b[valid] < bounds[valid]).all()))

    def test_candidate_pool_bounded_graph_replay_tracks_lengths(self):
        """Capture select_candidates in a CUDA graph; replay across changing
        visible lengths (grow/shrink/zero/full). The device-bounded row loop
        must re-derive block_lengths each replay with no host read, matching a
        fresh eager static computation at every length."""
        rows, width, block_size, count = 160, 65536, 8, 2048
        torch.manual_seed(1523)
        logits = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        ends = torch.full((rows,), 20000, device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            selector.select_candidates(logits, ends, block_size, count)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cand, flags = selector.select_candidates(logits, ends, block_size, count)
        for length in (0, 8, 2047, 16384, 20000, width):
            logits.normal_()
            ends.fill_(length)
            graph.replay()
            torch.cuda.synchronize()
            exp_cand, exp_flags = _static_select_candidates(
                logits, ends, block_size, count
            )
            torch.testing.assert_close(cand, exp_cand, rtol=0, atol=0)
            torch.testing.assert_close(flags, exp_flags, rtol=0, atol=0)

    def test_attention_paged_caller_reaches_dense_deepselect(self):
        self.require_native(torch.bfloat16)
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_attention_fusion import (
            _fixture,
        )

        for ratio in (1, 2):
            attn, _, x, qr, positions = _fixture("cuda", ratio)
            # BF16 TMA selection requires complete 512-element tiles.
            keys = attn._shared_attention["global"][attn.kv_source_layer_id]
            table = torch.nn.functional.pad(
                keys.block_table,
                (0, 1024 // keys.logical_entries_per_block - keys.block_table.shape[1]),
            )
            keys = indexer.DecodeIndexerKeys(
                keys.pool, table, 1024, keys.logical_entries_per_block
            )
            attn._shared_attention["global"][attn.kv_source_layer_id] = keys
            with patch.object(
                indexer, "score_decode_indexer", wraps=indexer.score_decode_indexer
            ) as score, patch.object(
                rtp_llm_ops,
                "deepselect_bf16_dense",
                wraps=rtp_llm_ops.deepselect_bf16_dense,
            ) as select, patch.object(
                rtp_llm_ops, "topk_v3", side_effect=AssertionError("unexpected radix")
            ):
                selected = AttentionV41FP8._select_indices_decode(
                    attn, x, qr, positions
                )
                select.assert_called_once()
                self.assertTrue(score.call_args.kwargs["preserve_score_dtype"])
                self.assertEqual(select.call_args.args[0].dtype, torch.bfloat16)
                visible = ((positions + 1) // ratio).int()
                assert_equivalent(
                    self,
                    selected,
                    reference(select.call_args.args[0], visible),
                    select.call_args.args[0],
                    visible,
                )

    def test_short_rows_preserve_nonfinite_and_candidate_mask_indices(self):
        for dtype, name in (
            (torch.float32, "deepselect_fp32"),
            (torch.bfloat16, "deepselect_bf16_dense"),
        ):
            self.require_native(dtype)
            lengths = [0, 1, 128, 511, 512, 513]
            ends = torch.tensor(lengths, device="cuda", dtype=torch.int32)
            for value in (float("nan"), float("inf"), -float("inf")):
                logits = torch.randn(6, 1024, device="cuda", dtype=dtype)
                logits[:, :64] = value
                with patch.object(
                    rtp_llm_ops, name, wraps=getattr(rtp_llm_ops, name)
                ) as native:
                    actual = selector.select_tokens(logits, ends, 512)
                    native.assert_called_once()
                for row, length in enumerate(lengths[:-1]):
                    expected = torch.full((512,), -1, device="cuda", dtype=torch.int32)
                    expected[:length] = torch.arange(length, device="cuda")
                    torch.testing.assert_close(actual[row], expected, rtol=0, atol=0)
                with patch.dict(os.environ, {"DSV41_DECODE_DEEPSELECT": "0"}):
                    old = selector.select_tokens(logits, ends, 512)
                assert_equivalent(self, actual, old, logits, ends)

    def test_paged_candidate_fallback_preserves_fp32_membership(self):
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_attention_fusion import (
            _fixture,
        )

        for k, capacity in ((256, 1024), (512, 256)):
            attn, keys, x, qr, positions = _fixture("cuda", 1, candidate=True)
            attn.index_topk = k
            positions = positions.clamp_max(capacity - 1)
            table = keys.block_table
            pages = capacity // keys.logical_entries_per_block
            table = torch.nn.functional.pad(table, (0, max(0, pages - table.shape[1])))[
                :, :pages
            ].contiguous()
            attn._shared_attention["global"][attn.kv_source_layer_id] = (
                indexer.DecodeIndexerKeys(
                    keys.pool, table, capacity, keys.logical_entries_per_block
                )
            )
            original_score = indexer.score_decode_indexer

            def score_with_dtype(*args, **kwargs):
                logits = original_score(*args, **kwargs)
                self.assertIsNotNone(logits)
                self.assertEqual(logits.dtype, torch.bfloat16)
                return logits

            def score_fp32(*args, **kwargs):
                kwargs["preserve_score_dtype"] = False
                return original_score(*args, **kwargs)

            with patch.object(
                indexer, "score_decode_indexer", side_effect=score_with_dtype
            ), patch.object(
                selector,
                "select_tokens",
                side_effect=AssertionError("must enter old fallback"),
            ):
                actual = AttentionV41FP8._select_indices_decode(attn, x, qr, positions)
                candidates = attn._shared_attention["candidates"].clone()
            with patch.object(indexer, "score_decode_indexer", side_effect=score_fp32):
                expected = AttentionV41FP8._select_indices_decode(
                    attn, x, qr, positions
                )
            torch.testing.assert_close(
                candidates, attn._shared_attention["candidates"], rtol=0, atol=0
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_paged_compaction_preserves_bf16_without_changing_scores(self):
        for logical in (32, 64):
            case, _ = make_case(2, 6, 1024, 64, logical, 64 // logical)
            expected = indexer.score_decode_indexer(**case)
            actual = indexer.score_decode_indexer(**case, preserve_score_dtype=True)
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
