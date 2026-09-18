"""V4.1 fused prefill indexer contracts and opt-in CUDA equivalence tests.

The default invocation runs CPU contracts without probing a CUDA device.
Set DSV41_TEST_FUSED_INDEXER_GPU=1 to additionally run the real DeepGEMM path.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as score_backend
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as indexer
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    fp8_roundtrip,
    mask_candidate_logits,
    select_candidate_blocks,
)


def _reference_logits(q, weights, k, visible):
    """The original V4.1 FP32 scoring path, including FP8 round trips."""
    scores = torch.einsum("mhd,kd->mhk", fp8_roundtrip(q), fp8_roundtrip(k))
    scores = (scores.relu_() * weights[:, :, None]).sum(1)
    columns = torch.arange(k.shape[0], device=k.device)
    return scores.masked_fill(columns[None] >= visible[:, None], -torch.inf)


class V41PrefillIndexerCPU(unittest.TestCase):
    def test_default_enable_and_supported_device_contract(self):
        with patch.dict(os.environ), patch.object(
            score_backend, "has_fp8_mqa_logits", return_value=True
        ), patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            os.environ.pop("DSV41_FUSED_PREFILL_INDEXER", None)
            self.assertTrue(indexer.is_supported(torch.device("cuda"), 32, 128))

        with patch.dict(os.environ, {"DSV41_FUSED_PREFILL_INDEXER": "1"}), patch.object(
            score_backend, "has_fp8_mqa_logits", return_value=True
        ):
            for major in (8, 9, 10, 11):
                for heads, dim in ((32, 128), (64, 128), (16, 128), (32, 64)):
                    with self.subTest(major=major, heads=heads, dim=dim), patch.object(
                        torch.cuda, "get_device_capability", return_value=(major, 0)
                    ):
                        self.assertEqual(
                            indexer.is_supported(torch.device("cuda"), heads, dim),
                            major in (9, 10) and heads in (32, 64) and dim == 128,
                        )

    def test_missing_deepgemm_api_falls_back(self):
        with patch.dict(os.environ, {"DSV41_FUSED_PREFILL_INDEXER": "1"}), patch.object(
            score_backend, "has_fp8_mqa_logits", return_value=False
        ), patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            self.assertFalse(indexer.is_supported(torch.device("cuda"), 32, 128))

    def test_cpu_is_unsupported_without_querying_cuda(self):
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("CPU fallback must not query CUDA"),
        ):
            self.assertFalse(indexer.is_supported(torch.device("cpu"), 32, 128))

    def test_explicit_disable_is_unsupported_without_querying_cuda(self):
        with patch.dict(os.environ, {"DSV41_FUSED_PREFILL_INDEXER": "0"}), patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("Disabled path must not query CUDA"),
        ):
            self.assertFalse(indexer.is_supported(torch.device("cuda"), 32, 128))

    def test_k_quantization_preserves_continuous_scale_and_zero_rows(self):
        torch.manual_seed(41)
        k = torch.randn(9, 128, dtype=torch.bfloat16)
        k[0].zero_()
        # This row distinguishes continuous scales from power-of-two scaling.
        k[1].fill_(1.5)
        quantized, scales = indexer.quantize_indexer_k_reference(k)
        expected_scales = (k.float().abs().amax(-1) / 448.0).clamp_min(1e-12)
        self.assertEqual(quantized.shape, k.shape)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.shape, (k.shape[0],))
        self.assertEqual(scales.dtype, torch.float32)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(
            quantized.float() * scales[:, None], fp8_roundtrip(k), rtol=0, atol=0
        )
        self.assertTrue(torch.isfinite(scales).all())

    def test_logit_chunks_bound_long_context_workspace(self):
        cap = 256 * 1024 * 1024
        for keys in (1, 32768, 65536, 131072, 262144, 524288, cap):
            with self.subTest(keys=keys):
                rows = indexer.logits_chunk_rows(keys)
                self.assertGreaterEqual(rows, 1)
                self.assertLessEqual(rows, 512)
                if keys * 4 <= cap:
                    self.assertLessEqual(rows * keys * 4, cap)
                else:
                    self.assertEqual(rows, 1)

    def test_score_clamps_bounds_and_masks_zero_visible_rows(self):
        q = torch.zeros(3, 32, 128).to(torch.float8_e4m3fn)
        k = torch.zeros(5, 128).to(torch.float8_e4m3fn)
        visible = torch.tensor([-2, 3, 8], dtype=torch.int32)
        raw_logits = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        with patch.object(
            score_backend, "fp8_mqa_indexer_score", return_value=raw_logits.clone()
        ) as fused:
            actual = indexer.score_indexer_chunk(
                q, torch.ones(3, 32), k, torch.ones(5), visible
            )
        self.assertTrue(fused.call_args.kwargs["clean_logits"])
        self.assertEqual(fused.call_args.kwargs["max_seqlen_k"], 0)
        starts, ends = fused.call_args.args[4:6]
        torch.testing.assert_close(starts, torch.zeros(3, dtype=torch.int32))
        torch.testing.assert_close(ends, torch.tensor([0, 3, 5], dtype=torch.int32))
        expected = raw_logits.masked_fill(
            torch.arange(5)[None] >= torch.tensor([0, 3, 5])[:, None], -torch.inf
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_kernel_failure_is_not_silently_replaced_by_reference(self):
        with patch.object(
            score_backend,
            "fp8_mqa_indexer_score",
            side_effect=RuntimeError("injected CUDA launch failure"),
        ), self.assertRaisesRegex(RuntimeError, "injected CUDA launch failure"):
            indexer.score_indexer_chunk(
                torch.zeros(1, 32, 128).to(torch.float8_e4m3fn),
                torch.ones(1, 32),
                torch.zeros(5, 128).to(torch.float8_e4m3fn),
                torch.ones(5),
                torch.tensor([5], dtype=torch.int32),
            )

    def test_cp4_gather_preserves_cache_bytes_and_global_order(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _indexer_cp_assembler as assembler,
        )
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _indexer_cp_gather_triton as gather_backend,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
            cp_kv_slot_mapping,
        )

        for ratio in (1, 2):
            owner_entries = 1024 // ratio
            page = 128 // ratio
            for count in (5, owner_entries + 3, 4 * owner_entries + 3):
                with self.subTest(ratio=ratio, count=count):
                    expected_q = (
                        torch.arange(count * 128).remainder(120).byte().view(count, 128)
                    )
                    expected_s = torch.arange(count, dtype=torch.float32) + 1.25
                    local_count = (
                        (count + 4 * owner_entries - 1) // (4 * owner_entries)
                    ) * owner_entries
                    peers_q, peers_s, pools, tables = [], [], [], []
                    for rank in range(4):
                        # Scalar ownership oracle, independent of CP plan/restore.
                        owned = [
                            i for i in range(count) if (i * ratio // 1024) % 4 == rank
                        ]
                        local_q = torch.zeros(local_count, 128, dtype=torch.uint8)
                        local_s = torch.zeros(local_count, 4, dtype=torch.uint8)
                        local_q[: len(owned)] = expected_q[owned]
                        local_s[: len(owned)] = (
                            expected_s[owned].view(torch.uint8).view(-1, 4)
                        )
                        peers_q.append(local_q)
                        peers_s.append(local_s)
                        pages = local_count // page
                        table = torch.stack(
                            (
                                torch.arange(1, pages + 1),
                                torch.arange(2 * pages, pages, -1),
                            )
                        )
                        pool = torch.zeros(2 * pages + 1, page, 132, dtype=torch.uint8)
                        raw = pool.view(2 * pages + 1, -1)
                        local_ids = torch.arange(len(owned))
                        blocks = table[1, local_ids // page]
                        offsets = local_ids % page
                        raw[
                            blocks[:, None], offsets[:, None] * 128 + torch.arange(128)
                        ] = local_q[: len(owned)]
                        raw[
                            blocks[:, None],
                            page * 128 + offsets[:, None] * 4 + torch.arange(4),
                        ] = local_s[: len(owned)]
                        pools.append(pool)
                        tables.append(table)

                    def read_packed_cache(pool, slots):
                        valid = slots >= 0
                        blocks, offsets = slots[valid] // page, slots[valid] % page
                        raw = pool.view(pool.shape[0], -1)
                        q = torch.zeros(len(slots), 128, dtype=torch.uint8)
                        s = torch.zeros(len(slots), 4, dtype=torch.uint8)
                        q[valid] = raw[
                            blocks[:, None], offsets[:, None] * 128 + torch.arange(128)
                        ]
                        s[valid] = raw[
                            blocks[:, None],
                            page * 128 + offsets[:, None] * 4 + torch.arange(4),
                        ]
                        return (
                            q.view(torch.float8_e4m3fn),
                            s.view(torch.float32).flatten(),
                        )

                    for rank in range(4):

                        def slots_for_request(positions, requests):
                            self.assertTrue((requests == 1).all())
                            self.assertTrue((positions < count * ratio).all())
                            return cp_kv_slot_mapping(
                                positions,
                                tables[rank],
                                requests,
                                128,
                                page,
                                ratio,
                                4,
                                rank,
                                owner_tokens_per_block=1024,
                            )

                        def gather_peers(local, group):
                            peers = peers_q if local.shape[1] == 128 else peers_s
                            torch.testing.assert_close(
                                local, peers[rank], rtol=0, atol=0
                            )
                            return torch.cat(peers)

                        with patch.object(
                            gather_backend,
                            "gather_indexer_k_for_prefill",
                            side_effect=read_packed_cache,
                        ), patch.object(
                            assembler, "all_gather", side_effect=gather_peers
                        ):
                            result = indexer.gather_indexer_keys(
                                pools[rank],
                                slots_for_request,
                                count,
                                1,
                                ratio,
                                SimpleNamespace(
                                    cp_size=4, cp_rank=rank, kv_cache_sharded=True
                                ),
                                owner_tokens_per_block=1024,
                            )
                        torch.testing.assert_close(
                            result.quant.view(torch.uint8), expected_q, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            result.scale, expected_s, rtol=0, atol=0
                        )

    def test_empty_gather_does_not_map_or_read_slots(self):
        def unexpected_slots(*args):
            self.fail("Empty K gather must not resolve cache slots")

        result = indexer.gather_indexer_keys(
            torch.empty(1, 64, 132, dtype=torch.uint8),
            unexpected_slots,
            0,
            0,
            2,
            SimpleNamespace(cp_size=4, cp_rank=3, kv_cache_sharded=True),
            owner_tokens_per_block=1024,
        )
        self.assertEqual(result.quant.shape, (0, 128))
        self.assertEqual(result.scale.shape, (0,))


@unittest.skipUnless(
    os.environ.get("DSV41_TEST_FUSED_INDEXER_GPU") == "1",
    "Set DSV41_TEST_FUSED_INDEXER_GPU=1 to run CUDA equivalence tests",
)
class V41PrefillIndexerCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # An explicit GPU test invocation must fail, rather than silently skip,
        # when the expected production kernel is unavailable.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA test requested but CUDA is unavailable")
        cls.device = torch.device("cuda")
        if not indexer.is_supported(cls.device, 32, 128):
            raise RuntimeError(
                "CUDA test requested but fused V4.1 indexer is unsupported"
            )

    def setUp(self):
        self.original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.original_tf32

    def _inputs(self, rows, heads, keys, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        q = torch.randn(
            rows, heads, 128, generator=generator, device=self.device
        ).bfloat16()
        k = torch.randn(keys, 128, generator=generator, device=self.device).bfloat16()
        weights = (
            torch.randn(rows, heads, generator=generator, device=self.device)
            / (heads * 128) ** 0.5
        )
        return q, weights, k

    def _score(self, q, weights, k, visible):
        q_fp8, w_fold = indexer.quantize_indexer_q(q, weights)
        k_fp8, k_scale = indexer.quantize_indexer_k_reference(k)
        return indexer.score_indexer_chunk(q_fp8, w_fold, k_fp8, k_scale, visible)

    def _assert_scores_and_topk(self, actual, expected, topk=512):
        finite = torch.isfinite(expected)
        self.assertTrue(torch.equal(torch.isfinite(actual), finite))
        self.assertTrue(torch.isneginf(actual[~finite]).all())
        torch.testing.assert_close(
            actual[finite], expected[finite], rtol=3e-4, atol=8e-4
        )
        # Top-k ties may choose different column IDs. Every chosen column must
        # still reach the reference cutoff within the measured score error.
        for row in range(actual.shape[0]):
            count = min(topk, int(finite[row].sum()))
            if not count:
                continue
            selected = actual[row].topk(count).indices
            self.assertTrue(finite[row, selected].all())
            error = (actual[row, finite[row]] - expected[row, finite[row]]).abs().max()
            cutoff = expected[row].topk(count).values[-1]
            self.assertGreaterEqual(
                float(expected[row, selected].min()), float(cutoff - 2 * error - 1e-6)
            )

    def test_ratio_one_two_noncontiguous_global_positions_and_prefix(self):
        for heads in (32, 64):
            for ratio in (1, 2):
                for prefix in (0, 513):
                    with self.subTest(heads=heads, ratio=ratio, prefix=prefix):
                        positions = (
                            torch.tensor(
                                [0, 1, 2, 7, 63, 255, 256, 511, 512, 767, 1023, 1024],
                                dtype=torch.int32,
                                device=self.device,
                            )
                            + prefix
                        )
                        keys = (1025 + prefix) // ratio
                        visible = (positions + 1) // ratio
                        visible[0] = 0
                        q, weights, k = self._inputs(len(positions), heads, keys, 41)
                        actual = self._score(q, weights, k, visible)
                        expected = _reference_logits(q, weights, k, visible)
                        self._assert_scores_and_topk(actual, expected)
                        # Row chunk boundaries must not change global causality.
                        chunked = torch.cat(
                            [
                                self._score(
                                    q[i : i + 3],
                                    weights[i : i + 3],
                                    k,
                                    visible[i : i + 3],
                                )
                                for i in range(0, len(positions), 3)
                            ]
                        )
                        self._assert_scores_and_topk(chunked, expected)

    def test_q_quantization_folds_signed_weights_and_keeps_zero_rows_finite(self):
        q, weights, _ = self._inputs(7, 32, 1, 42)
        q[0].zero_()
        weights[1].zero_()
        quantized, folded = indexer.quantize_indexer_q(q, weights)
        scale = (q.float().abs().amax(-1) / 448.0).clamp_min(1e-12)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(quantized.shape, q.shape)
        self.assertEqual(folded.dtype, torch.float32)
        self.assertTrue(torch.isfinite(folded).all())
        torch.testing.assert_close(folded, weights * scale, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(
            quantized.float() * scale[:, :, None],
            fp8_roundtrip(q),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_candidate_source_and_consumers_use_causal_compressed_positions(self):
        # Real V4.1 candidate geometry; the final block is only partly visible
        # on one row, and the zero-visible row must publish only -1 padding.
        keys = 2049 * 8
        visible = torch.tensor(
            [0, 8, 8193, keys], dtype=torch.int32, device=self.device
        )
        q, weights, k = self._inputs(4, 32, keys, 43)
        source = self._score(q, weights, k, visible)
        source_ref = _reference_logits(q, weights, k, visible)
        self._assert_scores_and_topk(source, source_ref)
        candidates = select_candidate_blocks(source, visible, 8, 2048)
        self.assertTrue((candidates[0] == -1).all())
        for row in range(1, 4):
            self.assertTrue((candidates[row] == (visible[row] - 1) // 8).any())
            chosen = candidates[row][candidates[row] >= 0]
            self.assertEqual(chosen.unique().numel(), chosen.numel())
            self.assertTrue((chosen * 8 < visible[row]).all())
            reference_blocks = source_ref[row].reshape(-1, 8).amax(-1)
            reference_blocks[(visible[row] - 1) // 8] = torch.inf
            cutoff = reference_blocks.topk(chosen.numel()).values[-1]
            valid = torch.isfinite(source_ref[row])
            error = (source[row, valid] - source_ref[row, valid]).abs().max()
            self.assertGreaterEqual(
                float(reference_blocks[chosen].min()), float(cutoff - 2 * error - 1e-6)
            )

        q2, weights2, _ = self._inputs(4, 32, 1, 44)
        consumer = self._score(q2, weights2, k, visible)
        consumer_ref = _reference_logits(q2, weights2, k, visible)
        mask_candidate_logits(consumer, candidates, 8)
        mask_candidate_logits(consumer_ref, candidates, 8)
        self._assert_scores_and_topk(consumer, consumer_ref)


if __name__ == "__main__":
    unittest.main()
