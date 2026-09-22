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
    mask_candidate_logits,
    select_candidate_blocks,
)


def _reference_logits(q, weights, k, visible):
    """The V4.1 FP32 scoring path over FP4 fake-quantized Q and K."""
    q_fake = indexer._fp4_rows_torch(q)[2].view(q.shape)
    k_fake = indexer._fp4_rows_torch(k)[2]
    scores = torch.einsum("mhd,kd->mhk", q_fake, k_fake)
    scores = (scores.relu_() * weights[:, :, None]).sum(1)
    columns = torch.arange(k.shape[0], device=k.device)
    return scores.masked_fill(columns[None] >= visible[:, None], -torch.inf)


class V41PrefillIndexerCPU(unittest.TestCase):
    def test_clean_logits_only_defaults_to_verified_sm100(self):
        for major in (8, 9, 10, 11, 12):
            with self.subTest(major=major), patch.object(
                torch.cuda, "get_device_capability", return_value=(major, 0)
            ):
                self.assertEqual(
                    indexer._use_clean_logits_only(torch.device("cuda")),
                    major == 10,
                )

    def test_clean_logits_cpu_does_not_query_cuda(self):
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("Fallback must not query CUDA"),
        ):
            self.assertFalse(indexer._use_clean_logits_only(torch.device("cpu")))

    def test_clean_logits_only_preserves_backend_tensor_without_dense_mask(self):
        clean = torch.tensor(
            [[-torch.inf, -torch.inf, -torch.inf], [2.0, 1.0, -torch.inf]]
        )
        with patch.object(
            score_backend, "fp8_fp4_mqa_indexer_score", return_value=clean
        ) as fused, patch.object(
            indexer, "_use_clean_logits_only", return_value=True
        ), patch.object(
            torch.Tensor,
            "masked_fill_",
            side_effect=AssertionError("Clean logits must not be masked again"),
        ):
            actual = indexer.score_indexer_chunk(
                torch.zeros(2, 32, 64, dtype=torch.int8),
                torch.ones(2, 32, dtype=torch.int32),
                torch.zeros(3, 64, dtype=torch.int8),
                torch.ones(3, dtype=torch.int32),
                torch.ones(2, 32),
                torch.tensor([0, 2], dtype=torch.int32),
            )
        self.assertIs(actual, clean)
        self.assertTrue(fused.call_args.kwargs["clean_logits"])
        self.assertEqual(fused.call_args.kwargs["max_seqlen_k"], 0)

    def test_empty_score_shape_does_not_launch_backend(self):
        for rows, keys in ((0, 5), (3, 0), (0, 0)):
            with self.subTest(rows=rows, keys=keys), patch.object(
                score_backend,
                "fp8_fp4_mqa_indexer_score",
                side_effect=AssertionError("Empty logits must not launch DeepGEMM"),
            ):
                actual = indexer.score_indexer_chunk(
                    torch.empty(rows, 32, 64, dtype=torch.int8),
                    torch.empty(rows, 32, dtype=torch.int32),
                    torch.empty(keys, 64, dtype=torch.int8),
                    torch.empty(keys, dtype=torch.int32),
                    torch.empty(rows, 32),
                    torch.zeros(rows, dtype=torch.int32),
                )
                self.assertEqual(actual.shape, (rows, keys))
                self.assertEqual(actual.dtype, torch.float32)

    def test_default_enable_and_supported_device_contract(self):
        with patch.object(
            score_backend, "has_fp8_fp4_mqa_logits", return_value=True
        ), patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            self.assertTrue(indexer.is_supported(torch.device("cuda"), 32, 128))

        with patch.object(score_backend, "has_fp8_fp4_mqa_logits", return_value=True):
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
        with patch.object(
            score_backend, "has_fp8_fp4_mqa_logits", return_value=False
        ), patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            self.assertFalse(indexer.is_supported(torch.device("cuda"), 32, 128))

    def test_cpu_is_unsupported_without_querying_cuda(self):
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("CPU fallback must not query CUDA"),
        ):
            self.assertFalse(indexer.is_supported(torch.device("cpu"), 32, 128))

    def test_unsupported_head_dim_does_not_query_cuda(self):
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("Disabled path must not query CUDA"),
        ):
            self.assertFalse(indexer.is_supported(torch.device("cuda"), 32, 64))

    def test_k_quantization_uses_power_of_two_scales_and_exact_six(self):
        torch.manual_seed(41)
        k = torch.randn(9, 128, dtype=torch.bfloat16)
        k[0].zero_()
        # 1.5 / (1.5/6 rounded up to a power of two) == 6.0 exactly.
        k[1].fill_(1.5)
        payload, sf = indexer.quantize_indexer_k_reference(k)
        _, _, values = indexer._fp4_rows_torch(k)
        self.assertEqual(tuple(payload.shape), (9, 64))
        self.assertEqual(payload.dtype, torch.int8)
        self.assertEqual(tuple(sf.shape), (9,))
        self.assertEqual(sf.dtype, torch.int32)
        # Zero rows encode zero payload; the floor scale keeps every group
        # representable (packed exponents stay in the normal UE8M0 range).
        self.assertTrue((payload[0] == 0).all())
        exponent = sf.view(torch.uint8).view(9, 4)
        self.assertTrue(((exponent > 0) & (exponent < 255)).all())
        torch.testing.assert_close(values[1], torch.full((128,), 1.5), rtol=0, atol=0)
        # Round-trip error stays bounded by the coarsest e2m1 grid step
        # (one full scale in the [4, 6] magnitude band, half elsewhere).
        error = (values - k.float()).abs().reshape(9, 4, 32)
        scale = torch.exp2(exponent.float() - 127.0)
        self.assertTrue((error <= scale[:, :, None] + 1e-9).all())

    def test_logit_chunks_bound_long_context_workspace(self):
        cap = 256 * 1024 * 1024
        for keys in (1, 32768, 65536, 131072, 262144, 524288, cap):
            with self.subTest(keys=keys):
                rows = indexer.logits_chunk_rows(keys)
                self.assertGreaterEqual(rows, 1)
                self.assertLessEqual(rows, 4096)
                if keys * 4 <= cap:
                    self.assertLessEqual(rows * keys * 4, cap)
                else:
                    self.assertEqual(rows, 1)

    def test_score_clamps_bounds_and_masks_zero_visible_rows(self):
        q_payload = torch.zeros(3, 32, 64, dtype=torch.int8)
        q_sf = torch.ones(3, 32, dtype=torch.int32)
        k_payload = torch.zeros(5, 64, dtype=torch.int8)
        k_sf = torch.ones(5, dtype=torch.int32)
        visible = torch.tensor([-2, 3, 8], dtype=torch.int32)
        raw_logits = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        with patch.object(
            score_backend, "fp8_fp4_mqa_indexer_score", return_value=raw_logits.clone()
        ) as fused:
            actual = indexer.score_indexer_chunk(
                q_payload, q_sf, k_payload, k_sf, torch.ones(3, 32), visible
            )
        self.assertTrue(fused.call_args.kwargs["clean_logits"])
        self.assertEqual(fused.call_args.kwargs["max_seqlen_k"], 0)
        starts, ends = fused.call_args.args[5:7]
        torch.testing.assert_close(starts, torch.zeros(3, dtype=torch.int32))
        torch.testing.assert_close(ends, torch.tensor([0, 3, 5], dtype=torch.int32))
        expected = raw_logits.masked_fill(
            torch.arange(5)[None] >= torch.tensor([0, 3, 5])[:, None], -torch.inf
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_kernel_failure_is_not_silently_replaced_by_reference(self):
        with patch.object(
            score_backend,
            "fp8_fp4_mqa_indexer_score",
            side_effect=RuntimeError("injected CUDA launch failure"),
        ), self.assertRaisesRegex(RuntimeError, "injected CUDA launch failure"):
            indexer.score_indexer_chunk(
                torch.zeros(1, 32, 64, dtype=torch.int8),
                torch.ones(1, 32, dtype=torch.int32),
                torch.zeros(5, 64, dtype=torch.int8),
                torch.ones(5, dtype=torch.int32),
                torch.ones(1, 32),
                torch.tensor([5], dtype=torch.int32),
            )

    def test_cp4_gather_preserves_cache_bytes_and_global_order(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _indexer_cp_assembler as assembler,
        )
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as codec
        from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
            cp_kv_slot_mapping,
        )

        for ratio in (1, 2):
            owner_entries = 1024 // ratio
            page = 128 // ratio
            for count in (5, owner_entries + 3, 4 * owner_entries + 3):
                with self.subTest(ratio=ratio, count=count):
                    expected_q = (
                        torch.arange(count * 64).remainder(120).byte().view(count, 64)
                    )
                    expected_s = (
                        (torch.arange(count * 4).remainder(120) + 1)
                        .byte()
                        .view(count, 4)
                        .view(torch.int32)
                        .flatten()
                    )
                    local_count = (
                        (count + 4 * owner_entries - 1) // (4 * owner_entries)
                    ) * owner_entries
                    peers_q, peers_s, pools, tables = [], [], [], []
                    for rank in range(4):
                        # Scalar ownership oracle, independent of CP plan/restore.
                        owned = [
                            i for i in range(count) if (i * ratio // 1024) % 4 == rank
                        ]
                        local_q = torch.zeros(local_count, 64, dtype=torch.uint8)
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
                        pool = torch.zeros(2 * pages + 1, page, 68, dtype=torch.uint8)
                        raw = pool.view(2 * pages + 1, -1)
                        local_ids = torch.arange(len(owned))
                        blocks = table[1, local_ids // page]
                        offsets = local_ids % page
                        raw[
                            blocks[:, None], offsets[:, None] * 64 + torch.arange(64)
                        ] = local_q[: len(owned)]
                        raw[
                            blocks[:, None],
                            page * 64 + offsets[:, None] * 4 + torch.arange(4),
                        ] = local_s[: len(owned)]
                        pools.append(pool)
                        tables.append(table)

                    def read_packed_cache(pool, slots):
                        valid = slots >= 0
                        blocks, offsets = slots[valid] // page, slots[valid] % page
                        raw = pool.view(pool.shape[0], -1)
                        q = torch.zeros(len(slots), 64, dtype=torch.uint8)
                        s = torch.zeros(len(slots), 4, dtype=torch.uint8)
                        q[valid] = raw[
                            blocks[:, None], offsets[:, None] * 64 + torch.arange(64)
                        ]
                        s[valid] = raw[
                            blocks[:, None],
                            page * 64 + offsets[:, None] * 4 + torch.arange(4),
                        ]
                        return (
                            q.view(torch.int8),
                            s.contiguous().view(torch.int32).flatten(),
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
                            peers = peers_q if local.shape[1] == 64 else peers_s
                            torch.testing.assert_close(
                                local, peers[rank], rtol=0, atol=0
                            )
                            return torch.cat(peers)

                        with patch.object(
                            codec,
                            "gather_indexer_k_fp4",
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
            torch.empty(1, 64, 68, dtype=torch.uint8),
            unexpected_slots,
            0,
            0,
            2,
            SimpleNamespace(cp_size=4, cp_rank=3, kv_cache_sharded=True),
            owner_tokens_per_block=1024,
        )
        self.assertEqual(result.quant.shape, (0, 64))
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
        q_payload, q_sf = indexer.quantize_indexer_q(q)
        k_payload, k_sf = indexer.quantize_indexer_k_reference(k)
        return indexer.score_indexer_chunk(
            q_payload, q_sf, k_payload, k_sf, weights, visible
        )

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

    def test_clean_logits_only_mixed_tile_boundaries_and_padded_stride(self):
        if torch.cuda.get_device_capability(self.device)[0] != 10:
            self.skipTest("Clean-logits-only is enabled only on SM100")
        for heads in (32, 64):
            for ratio in (1, 2):
                for keys in (1, 255, 256, 257, 509, 1021, 16555):
                    with self.subTest(heads=heads, ratio=ratio, keys=keys):
                        # Non-aligned M and N, mixed bounds within each Q
                        # tile, and ratio-2 odd/even original token positions.
                        if ratio == 2:
                            positions = [
                                0,
                                1,
                                2,
                                509,
                                510,
                                511,
                                512,
                                513,
                                2 * keys - 3,
                                2 * keys - 2,
                                2 * keys - 1,
                                2 * keys,
                                2 * keys + 1,
                            ]
                            ends = [(p + 1) // ratio for p in positions]
                        else:
                            ends = [
                                -1,
                                0,
                                1,
                                255,
                                256,
                                257,
                                keys - 1,
                                keys,
                                keys + 1,
                                0,
                                keys,
                                1,
                                128,
                            ]
                        visible = torch.tensor(
                            ends, dtype=torch.int32, device=self.device
                        )
                        bounded = visible.clamp(0, keys)
                        q, weights, k = self._inputs(len(ends), heads, keys, 61)
                        actual = self._score(q, weights, k, visible)
                        with patch.object(
                            indexer, "_use_clean_logits_only", return_value=False
                        ):
                            masked = self._score(q, weights, k, visible)
                        # The optimization changes no arithmetic, valid value,
                        # or invalid (-inf) position in the full logical output.
                        torch.testing.assert_close(actual, masked, rtol=0, atol=0)
                        self._assert_scores_and_topk(
                            actual, _reference_logits(q, weights, k, bounded)
                        )
                        self.assertGreater(actual.stride(0), keys)
                        padded = actual.as_strided(
                            (len(ends), actual.stride(0)), actual.stride()
                        )
                        self.assertTrue(torch.isneginf(padded[:, keys:]).all())

    def test_clean_logits_graph_replay_overwrites_poisoned_reused_storage(self):
        if torch.cuda.get_device_capability(self.device)[0] != 10:
            self.skipTest("Clean-logits-only is enabled only on SM100")
        for rows, keys in ((1, 257), (3, 509), (13, 1021), (13, 16555)):
            with self.subTest(rows=rows, keys=keys):
                q, weights, k = self._inputs(rows, 32, keys, 62)
                q_payload, q_sf = indexer.quantize_indexer_q(q)
                k_payload, k_sf = indexer.quantize_indexer_k_reference(k)
                visible = torch.full(
                    (rows,), keys, dtype=torch.int32, device=self.device
                )

                def score():
                    return indexer.score_indexer_chunk(
                        q_payload, q_sf, k_payload, k_sf, weights, visible
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        score()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = score()
                torch.cuda.synchronize()
                storage = actual.as_strided((rows, actual.stride(0)), actual.stride())
                mixed = [0, 1, 255, 256, 257, keys - 1, keys]
                # Zero-only Q tiles visit no K tile; the independent cleaner
                # still has to overwrite the entire reused output allocation.
                for bounds, poison in (
                    ([0] * rows, float("nan")),
                    ([mixed[i % len(mixed)] for i in range(rows)], 1.0e30),
                    ([keys] * rows, float("nan")),
                ):
                    storage.fill_(poison)
                    visible.copy_(
                        torch.tensor(bounds, dtype=torch.int32, device=self.device)
                    )
                    graph.replay()
                    self._assert_scores_and_topk(
                        actual,
                        _reference_logits(q, weights, k, visible.clamp(0, keys)),
                    )
                    self.assertTrue(torch.isneginf(storage[:, keys:]).all())

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

    def test_q_quantization_packs_mx_scales_and_keeps_zero_rows_finite(self):
        q, weights, _ = self._inputs(7, 32, 1, 42)
        q[0].zero_()
        payload, sf = indexer.quantize_indexer_q(q)
        expected_payload, expected_sf, values = indexer._fp4_rows_torch(
            q.reshape(-1, 128)
        )
        self.assertEqual(payload.dtype, torch.int8)
        self.assertEqual(tuple(payload.shape), (7, 32, 64))
        self.assertEqual(sf.dtype, torch.int32)
        self.assertEqual(tuple(sf.shape), (7, 32))
        torch.testing.assert_close(
            payload.reshape(-1, 64), expected_payload, rtol=0, atol=0
        )
        torch.testing.assert_close(sf.reshape(-1), expected_sf, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((payload.reshape(7, 32, 64)[0] == 0).all())

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

    def test_batched_source_scan_rows_match_512_row_chunks(self):
        """The batched source scan (up to 4096 query rows per scorer call) is
        row-independent: one large call equals the 512-row chunk sequence it
        replaces, bit-for-bit."""
        keys = 16555
        rows = 4096
        generator = torch.Generator(device=self.device).manual_seed(45)
        q = torch.randn(
            rows, 32, 128, generator=generator, device=self.device
        ).bfloat16()
        k = torch.randn(keys, 128, generator=generator, device=self.device).bfloat16()
        weights = torch.randn(rows, 32, generator=generator, device=self.device) / 32.0
        visible = torch.full((rows,), keys, dtype=torch.int32, device=self.device)
        visible[0] = 0
        q_payload, q_sf = indexer.quantize_indexer_q(q)
        k_payload, k_sf = indexer.quantize_indexer_k_reference(k)
        batched = indexer.score_indexer_chunk(
            q_payload, q_sf, k_payload, k_sf, weights, visible
        )
        chunked = torch.cat(
            [
                indexer.score_indexer_chunk(
                    q_payload[i : i + 512],
                    q_sf[i : i + 512],
                    k_payload,
                    k_sf,
                    weights[i : i + 512],
                    visible[i : i + 512],
                )
                for i in range(0, rows, 512)
            ]
        )
        torch.testing.assert_close(batched, chunked, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
