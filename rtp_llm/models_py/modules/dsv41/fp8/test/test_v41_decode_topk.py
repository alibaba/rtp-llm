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
    reference as prefill_reference,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


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
        env = patch.dict(
            os.environ,
            {"DSV41_PREFILL_DEEPSELECT": "0", "DSV41_DECODE_CANDIDATE_TOPK_V3": "0"},
        )
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

    def test_candidate_v3_short_capacity_preserves_invalid_blocks(self):
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import (
            select_candidate_blocks,
        )

        for dtype in (torch.float32, torch.bfloat16):
            for width, block_size in ((16376, 8), (16375, 8), (15, 7)):
                logits = torch.randn(6, width, device="cuda", dtype=dtype)
                ends = torch.tensor(
                    [0, 1, 7, 8, width - 1, width], device="cuda", dtype=torch.int32
                )
                columns = torch.arange(width, device="cuda")
                if width > 32:
                    logits[:, 8:16] = float("nan")
                    logits[:, 16:24] = -float("inf")
                    logits[:, 24:32] = float("inf")
                logits.masked_fill_(columns[None] >= ends[:, None], -torch.inf)
                expected = select_candidate_blocks(
                    logits.float(), ends, block_size, 2048
                )
                expected_flags = torch.zeros(
                    6,
                    (width + block_size - 1) // block_size,
                    device="cuda",
                    dtype=torch.uint8,
                )
                for row in range(6):
                    expected_flags[row, expected[row][expected[row] >= 0].long()] = 1
                with patch.object(
                    torch.Tensor,
                    "topk",
                    side_effect=AssertionError("short rows need no topk"),
                ), patch.dict(
                    os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "1"}
                ), patch.object(
                    rtp_llm_ops, "topk_v3", wraps=rtp_llm_ops.topk_v3
                ) as native:
                    ids, flags = selector.select_candidates(
                        logits, ends, block_size, 2048
                    )
                torch.testing.assert_close(flags, expected_flags, rtol=0, atol=0)
                torch.testing.assert_close(
                    ids.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0
                )

    def test_candidate_v3_is_default_and_opt_out_or_missing_backend_retains_torch(self):
        logits = torch.randn(4, 32768, device="cuda", dtype=torch.bfloat16)
        ends = torch.tensor([0, 13000, 16384, 32768], device="cuda", dtype=torch.int32)
        # Force cutoff ties so another valid block set cannot fail this test.
        logits[-1].zero_()
        logits.masked_fill_(
            torch.arange(logits.shape[1], device="cuda")[None] >= ends[:, None],
            -torch.inf,
        )
        with patch.dict(
            os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "0"}
        ), patch.object(
            rtp_llm_ops,
            "topk_v3",
            side_effect=AssertionError("opt-out must retain torch"),
        ):
            expected = selector.select_candidates(logits, ends, 8, 2048)
        with patch.dict(
            os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "1"}
        ), patch.object(selector.candidate_topk, "_TOPK_V3_OK", False):
            actual = selector.select_candidates(logits, ends, 8, 2048)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
        with patch.dict(
            os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "1"}
        ), patch.object(rtp_llm_ops, "topk_v3", wraps=rtp_llm_ops.topk_v3) as native:
            ids, flags = selector.select_candidates(logits, ends, 8, 2048)
            native.assert_called_once()
            self.assertEqual(native.call_args.args[-2], 2048)
            torch.testing.assert_close(native.call_args.args[1], (ends + 7) // 8)
            from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_candidates import (
                assert_candidate_values,
            )

            assert_candidate_values(self, ids, expected[0], logits, ends, 8)
            self.assertTrue(
                ((ids < 0) | (ids < ((ends + 7) // 8)[:, None])).all().item()
            )
            nblocks = flags.shape[1]
            rebuilt = torch.zeros(
                (ids.shape[0], nblocks + 1), device=ids.device, dtype=torch.uint8
            )
            rebuilt.scatter_(1, torch.where(ids >= 0, ids, nblocks).long(), 1)
            torch.testing.assert_close(flags, rebuilt[:, :nblocks], rtol=0, atol=0)
            self.assertEqual(ids.dtype, torch.int32)

    def test_candidate_v3_keeps_newest_and_filters_invalid_blocks(self):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.randn(8, 32776, device="cuda", dtype=dtype)
            logits[:, 8:16] = float("nan")
            logits[:, 16:24] = -float("inf")
            logits[:, 24:32] = float("inf")
            ends = torch.tensor(
                [0, 1, 7, 8, 9, 16384, 16385, 32776], device="cuda", dtype=torch.int32
            )
            with patch.dict(os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "0"}):
                expected = selector.select_candidates(logits, ends, 8, 2048)
            with patch.dict(os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "1"}):
                ids, flags = selector.select_candidates(logits, ends, 8, 2048)
            torch.testing.assert_close(
                flags.sum(-1), expected[1].sum(-1), rtol=0, atol=0
            )
            newest = ((ends - 1).clamp_min(0) // 8).long()
            self.assertTrue(
                (flags.gather(1, newest[:, None]).flatten()[ends > 0] == 1).all().item()
            )
            self.assertFalse(flags[0].any().item())
            self.assertEqual(int(flags[-1, 1]), 0)
            self.assertEqual(int(flags[-1, 2]), 0)
            self.assertEqual(int(flags[-1, 3]), 1)
            for row in range(8):
                valid = ids[row][ids[row] >= 0].long()
                self.assertEqual(valid.numel(), valid.unique().numel())
                self.assertTrue((valid < ((ends[row] + 7) // 8)).all().item())

    def test_candidate_v3_graph_refreshes_short_long_lengths_and_masks(self):
        logits = torch.randn(6, 32768, device="cuda", dtype=torch.bfloat16)
        ends = torch.full((6,), 13000, device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with patch.dict(os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "1"}):
            with torch.cuda.stream(stream):
                selector.select_candidates(logits, ends, 8, 2048)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                ids, flags = selector.select_candidates(logits, ends, 8, 2048)
            for length in (32768, 0, 16384, 16385, 13000, 1):
                logits.normal_()
                ends.fill_(length)
                graph.replay()
                reference_ids, reference_flags = selector.select_candidates(
                    logits, ends, 8, 2048
                )
                torch.testing.assert_close(ids, reference_ids, rtol=0, atol=0)
                torch.testing.assert_close(flags, reference_flags, rtol=0, atol=0)

    def test_candidate_v3_cutoff_ties_preserve_values_without_promising_membership(
        self,
    ):
        logits = torch.rand(4, 32768, device="cuda", dtype=torch.bfloat16) * 0.5
        logits[:, ::8] = 1
        ends = torch.full((4,), 32768, device="cuda", dtype=torch.int32)
        expected = selector.select_candidates(logits, ends, 8, 2048)
        with patch.dict(os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "1"}):
            ids, flags = selector.select_candidates(logits, ends, 8, 2048)
        torch.testing.assert_close(flags.sum(-1), expected[1].sum(-1), rtol=0, atol=0)
        scores = logits.float().view(4, 4096, 8).amax(-1)
        scores[:, -1] = torch.inf
        got = scores.gather(1, ids.long()).sort(-1).values
        ref = scores.gather(1, expected[0].long()).sort(-1).values
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        self.assertTrue((flags[:, -1] == 1).all().item())

    def test_candidate_native_execution_failure_is_not_hidden(self):
        logits = torch.randn(4, 32768, device="cuda")
        ends = torch.full((4,), 32768, device="cuda", dtype=torch.int32)
        with patch.dict(
            os.environ, {"DSV41_DECODE_CANDIDATE_TOPK_V3": "1"}
        ), patch.object(
            rtp_llm_ops, "topk_v3", side_effect=RuntimeError("candidate failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "candidate failure"):
                selector.select_candidates(logits, ends, 8, 2048)

    def test_candidate_warmup_uses_v3_without_legacy_jit(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import (
            _v41_attention_jit_warmup as warmup,
        )
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_candidate_topk as legacy

        with patch.dict(
            os.environ, {"DSV41_PREFILL_CANDIDATE_TOPK_V3": "1"}
        ), patch.object(
            legacy, "warmup", side_effect=AssertionError("legacy JIT must not run")
        ), patch.object(
            rtp_llm_ops, "topk_v3", wraps=rtp_llm_ops.topk_v3
        ) as native:
            warmup._warm_candidate_topk(torch.device("cuda"))
            native.assert_called_once()
        with patch.dict(
            os.environ,
            {
                "DSV41_PREFILL_CANDIDATE_TOPK_V3": "0",
                "DSV41_DECODE_CANDIDATE_TOPK_V3": "0",
            },
        ), patch.object(
            rtp_llm_ops,
            "topk_v3",
            side_effect=AssertionError("disabled warmup must skip"),
        ):
            warmup._warm_candidate_topk(torch.device("cuda"))

    def test_paged_compaction_preserves_bf16_without_changing_scores(self):
        for logical in (32, 64):
            case, _ = make_case(2, 6, 1024, 64, logical, 64 // logical)
            expected = indexer.score_decode_indexer(**case)
            actual = indexer.score_decode_indexer(**case, preserve_score_dtype=True)
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
