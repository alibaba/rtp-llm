"""Paged indexer regression for heterogeneous shared-cache block pitches."""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

# The host contracts exercise layout/reference code and mock the native call.
# DeepGEMM's wheel initializes CUDA at import, even on a CPU-only worker.
# Restore just the blocked entry so Triton's native modules stay registered.
_block_deep_gemm = not torch.cuda.is_available() and "deep_gemm" not in sys.modules
if _block_deep_gemm:
    sys.modules["deep_gemm"] = None
try:
    from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score, glm53_indexer_score
finally:
    if _block_deep_gemm:
        del sys.modules["deep_gemm"]

from rtp_llm.models_py.modules.dsv4.fp8._indexer_cache_view import (
    paged_indexer_cache_view,
)


def _inputs(page, device):
    torch.manual_seed(53)
    pages, heads = 6, 32
    pitch = page * 132 + 256
    storage = torch.full((pages, pitch), 173, dtype=torch.uint8, device=device)
    raw = storage[:, : page * 132]
    keys = torch.randn(pages, page, 128, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(pages, page, device=device) * 0.01
    raw[:, : page * 128] = keys.view(torch.uint8).reshape(pages, page * 128)
    raw[:, page * 128 :] = scales.view(torch.uint8).reshape(pages, page * 4)
    pool = raw.view(pages, page, 132)
    q = torch.randn(2, 2, heads, 128, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(4, heads, device=device)
    table = torch.tensor([[5, 1, 3], [0, 4, 2]], dtype=torch.int32, device=device)
    lengths = torch.tensor(
        [[0, page - 1], [page + 3, page * 3]], dtype=torch.int32, device=device
    )
    expected = []
    for batch in range(2):
        k = keys[table[batch].long()].float().reshape(-1, 128)
        sf = scales[table[batch].long()].reshape(-1)
        dots = torch.einsum("nhd,kd->nhk", q[batch].float(), k)
        score = (dots.relu() * weights[batch * 2 : (batch + 1) * 2, :, None]).sum(
            1
        ) * sf
        score.masked_fill_(
            torch.arange(page * 3, device=device) >= lengths[batch, :, None], 0
        )
        expected.append(score)
    return storage, pool, q, weights, table, lengths, torch.cat(expected)


class GlmIndexerCacheContractTest(unittest.TestCase):
    def test_window_topk_handles_ties_negative_infinity_and_short_rows(self):
        from rtp_llm.models_py.modules.dsv4.indexer_topk import (
            select_stable_window_topk,
        )

        scores = torch.full((4, 600), float("-inf"))
        scores[0, 1:560] = 1.0
        scores[1, :511] = 99.0
        starts = torch.tensor([1, 511, 17, 3], dtype=torch.int32)
        ends = torch.tensor([560, 512, 17, 5], dtype=torch.int32)
        for k in (3, 512, 700):
            out = torch.empty(4, k, dtype=torch.int32)
            native = mock.Mock(
                side_effect=AssertionError("unexpected native call on CPU")
            )
            select_stable_window_topk(scores, starts, ends, out, native)
            expected = torch.full_like(out, -1)
            for row, count in enumerate((559, 1, 0, 2)):
                valid = min(count, k)
                expected[row, :valid] = torch.arange(valid)
            torch.testing.assert_close(out, expected)
            native.assert_not_called()

    def test_flat_and_pitched_views_alias_the_original_storage(self):
        for page in (16, 32, 64):
            storage, pool, *_ = _inputs(page, "cpu")
            for cache in (pool, pool.unsqueeze(2), pool.contiguous().view(-1, 132)):
                view = paged_indexer_cache_view(cache, page)
                self.assertEqual(view.data_ptr(), cache.data_ptr())
                self.assertEqual(view.shape, (6, page, 1, 132))
                self.assertEqual(
                    view.stride(0), cache.stride(0) if cache.ndim > 2 else page * 132
                )
            self.assertTrue(torch.all(storage[:, page * 132 :] == 173))

    def test_native_dispatch_preserves_block_pitch_without_a_pool_copy(self):
        _, pool, q, weights, table, lengths, _ = _inputs(32, "cpu")
        backend = glm53_indexer_score.fp8_mqa
        with (
            mock.patch.object(glm53_indexer_score, "_HAS_NATIVE_MQA", True),
            mock.patch.object(backend, "get_num_sms", return_value=1),
            mock.patch.object(
                backend, "get_paged_mqa_logits_metadata", return_value=torch.empty(1)
            ),
            mock.patch.object(
                backend, "fp8_paged_mqa_logits", return_value=torch.empty(4, 96)
            ) as kernel,
        ):
            glm53_indexer_score.fp8_paged_indexer_score(
                q, weights, pool, table, lengths, 32, 96
            )
        cache = kernel.call_args.args[1]
        self.assertEqual(cache.data_ptr(), pool.data_ptr())
        self.assertEqual(cache.stride(0), pool.stride(0))
        self.assertFalse(cache.is_contiguous())

    def test_fallback_scores_pitched_block_planar_cache(self):
        storage, pool, q, weights, table, lengths, expected = _inputs(32, "cpu")
        before = storage.clone()
        with mock.patch.object(
            _indexer_score, "supports_paged_mqa_logits", return_value=False
        ):
            actual = _indexer_score.fp8_paged_indexer_score(
                q, weights, pool, table, lengths, 32, 96
            )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_invalid_or_overlapping_pages_are_rejected(self):
        for cache in (
            torch.empty(2, 16, 133, dtype=torch.uint8),
            torch.empty(2, 16, 132, dtype=torch.float32),
            torch.empty(5000, dtype=torch.uint8).as_strided(
                (2, 16, 132), (100, 132, 1)
            ),
        ):
            with self.assertRaises(ValueError):
                paged_indexer_cache_view(cache, 16)


class GlmIndexerCacheGpuTest(unittest.TestCase):
    def test_prefill_roles_order_the_same_selected_window(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import glm53_indexer as indexer
        from rtp_llm.ops import RoleType

        scores = torch.tensor([[99, 3, 3, 3, 8, 2], [9, 5, 1, 1, 9, 9]]).float()
        starts = torch.tensor([1, 1], dtype=torch.int32)
        ends = torch.tensor([6, 3], dtype=torch.int32)
        meta = SimpleNamespace(M=2, T=6, ks=starts, ke=ends)
        keys = torch.zeros(6)
        for role in (
            RoleType.PREFILL,
            RoleType.DECODE,
            RoleType.PDFUSION,
            RoleType.VIT,
        ):
            op = SimpleNamespace(
                role_type=role,
                index_topk=3,
                compressor=SimpleNamespace(kpool_mode=False),
            )
            with (
                self.subTest(role=role),
                mock.patch.object(
                    indexer, "fp8_mqa_indexer_score", return_value=scores
                ),
            ):
                out = indexer.IndexerFP8._prefill_score_topk(
                    op, scores, scores, keys, keys, meta
                )
                first = [0, 1, 3] if role == RoleType.PREFILL else [3, 0, 1]
                self.assertEqual(out.tolist(), [first, [0, 1, -1]])

    def test_decode_target_verify_keeps_selected_order(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import glm53_indexer as indexer
        from rtp_llm.ops import RoleType

        x = torch.ones(1, 3, 1)
        positions = torch.tensor([12, 13, 14])
        lengths = torch.tensor([5, 2, 0], dtype=torch.int32)
        meta = SimpleNamespace(positions=positions, compressed_lens_per_token=lengths)
        scores = torch.zeros(3, 32)
        scores[:, :5] = torch.tensor([2, 5, 5, 5, 9])
        op = SimpleNamespace(
            role_type=RoleType.PREFILL,
            index_topk=3,
            compress_ratio=4,
            n_heads=1,
            freqs_cis=None,
            _kv_block_table=torch.zeros(1, 1),
            _kv_eb=32,
            _kv_pool_view=None,
            compressor=mock.Mock(kpool_mode=True),
            _propagate_pool_to_nested=mock.Mock(),
            _clear_nested_pool=mock.Mock(),
            _fuse_glm_q_quant=lambda: False,
            _compute_indexer_q=lambda *a, **kw: x,
            _compute_weights=lambda x: x,
            _quantize_q=lambda q, w: (q, w),
        )
        with mock.patch.object(indexer, "fp8_paged_indexer_score", return_value=scores):
            actual = indexer.IndexerFP8.forward_decode_vectorized(
                op,
                x,
                x,
                positions[:1],
                torch.empty(1, 3, 3, dtype=torch.int32),
                position_ids=positions,
                compressor_meta=meta,
            )
        self.assertEqual(actual.tolist(), [[[4, 1, 2], [0, 1, -1], [-1, -1, -1]]])

    def test_mla_gather_uses_role_and_query_phase(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.glm53_sparse_mla import (
            SparseMlaFp8Op,
        )
        from rtp_llm.models_py.modules.hybrid.glm53_indexer import Indexer
        from rtp_llm.ops import RoleType

        phases = (
            (True, False, False),
            (False, False, False),
            (True, True, False),
            (True, False, True),
        )
        for role in (
            RoleType.PREFILL,
            RoleType.DECODE,
            RoleType.PDFUSION,
            RoleType.VIT,
        ):
            for graph in (False, True):
                op = SparseMlaFp8Op(
                    64,
                    512,
                    0,
                    512,
                    128,
                    1.0,
                    2051,
                    parallelism_config=SimpleNamespace(role_type=role),
                    indexer_top_k=512,
                    indexer_group_size=4,
                    use_cuda_graph=graph,
                )
                indexer = SimpleNamespace(_prefill_cp_enabled=lambda: True)
                params = SimpleNamespace(
                    batch_indice_h=torch.empty(6),
                    prefill_ragged_kv_len_indptr_d=torch.tensor([0, 10, 20]),
                )
                for prefill, verify, extend in phases:
                    inputs = SimpleNamespace(
                        is_prefill=prefill,
                        is_target_verify=verify,
                        is_draft_extend=extend,
                    )
                    regular_prefill = prefill and not verify and not extend
                    with (
                        self.subTest(role=role, graph=graph, phase=inputs),
                        mock.patch.object(op, "_reset_sched_meta"),
                    ):
                        op.plan(params, torch.zeros(2, 1), inputs)
                        self.assertEqual(
                            op._gather is not None,
                            regular_prefill and role == RoleType.PREFILL and not graph,
                        )
                        self.assertEqual(
                            op._native_tokens_per_request, 3 if verify else 1
                        )
                        self.assertEqual(
                            Indexer._is_sparse_prefill_sharded(indexer, inputs),
                            regular_prefill,
                        )
                        op._plan_prefill_gather(inputs, kv_cache_sharded=True)
                        self.assertEqual(op._gather is not None, regular_prefill)
                op._plan_prefill_gather(None, kv_cache_sharded=True)
                self.assertIsNone(op._gather)

    def test_sparse_query_chunk_limits_follow_cp_geometry(self):
        from rtp_llm.models_py.distributed.zigzag_token_layout import zigzag_local_rows
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            glm53_sparse_mla as sparse,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            glm53_sparse_mla_cp as sparse_cp,
        )
        from rtp_llm.ops import RoleType

        for rows, expected in (
            (0, 0),
            (7, 2),
            (8, 2),
            (9, 2),
            (15, 2),
            (16, 2),
            (17, 4),
            (299999, 37500),
            (300000, 37500),
            (300001, 37502),
            (1048576, 131072),
        ):
            self.assertEqual(zigzag_local_rows(rows, 8), expected)

        # Tiny head/output widths keep numerical tensors small; FlashMLA is replaced
        # by a shape recorder so the real dispatch can exercise long launches on CPU.
        for role, cp, heads, limit, expected_chunks in (
            (RoleType.PREFILL, True, 64, 1048576, [4097]),
            (RoleType.PREFILL, True, 128, 1048576, [4097]),
            (RoleType.PREFILL, True, 8, 1048576, [4096, 1]),
            (RoleType.PREFILL, False, 64, 1048576, [4096, 1]),
            (RoleType.DECODE, True, 64, 1048576, [4096, 1]),
            (RoleType.PDFUSION, True, 64, 1048576, [4096, 1]),
            (RoleType.PREFILL, True, 64, 0, [4096, 1]),
            (RoleType.PREFILL, True, 64, -1, [4096, 1]),
            (RoleType.PREFILL, True, 64, None, [4096, 1]),
        ):
            op = sparse.SparseMlaOp(
                heads,
                1,
                0,
                1,
                128,
                1.0,
                1,
                parallelism_config=SimpleNamespace(role_type=role),
            )
            op.sequence_parallel, op.prefill_cp_size = cp, 8
            sparse_cp.SparseMlaFp8CPOp.configure_native_query_chunk(op, limit)
            if (
                role == RoleType.PREFILL
                and cp
                and limit == 1048576
                and heads in (64, 128)
            ):
                self.assertEqual(op.native_q_chunk_rows, 131072)
            else:
                self.assertEqual(op.native_q_chunk_rows, 4096)
            q = torch.ones(4097, heads, 1)
            kv = torch.ones(1, 1, 1)
            indices = torch.zeros(4097, 1, 128, dtype=torch.int32)
            with (
                self.subTest(role=role, cp=cp, heads=heads, limit=limit),
                mock.patch.object(
                    sparse,
                    "flash_mla_sparse_fwd",
                    side_effect=lambda rows, *a, **kw: (rows, None, None),
                    create=True,
                ) as kernel,
            ):
                self.assertTrue(torch.equal(op._forward_sparse(q, kv, indices), q))
                self.assertEqual(
                    [call.args[0].shape[0] for call in kernel.call_args_list],
                    expected_chunks,
                )
        # Every row materializes 2176 selected BF16 keys of 512 channels.
        self.assertEqual(sparse._SELECTED_KV_CHUNK_ROWS * 2176 * 512 * 2, 544 * 1024**2)

    def test_native_window_topk_matches_stable_fallback_membership(self):
        from rtp_llm.models_py.modules.dsv4.indexer_topk import (
            select_stable_window_topk,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        scores = torch.ones(3, 3200, device="cuda")
        scores[0, 17::3] = 2
        scores[1, 10:13] = torch.tensor([4.0, 1.0, -float("inf")], device="cuda")
        starts = torch.tensor([7, 10, 100], dtype=torch.int32, device="cuda")
        ends = torch.tensor([3199, 13, 100], dtype=torch.int32, device="cuda")
        for k in (512, 1024, 2048):
            native = torch.empty(3, k, dtype=torch.int32, device="cuda")
            fallback = torch.empty_like(native)
            select_stable_window_topk(
                scores, starts, ends, native, rtp_llm_ops.topk_v3_tie_break
            )
            select_stable_window_topk(scores, starts, ends, fallback)
            torch.testing.assert_close(native.sort().values, fallback.sort().values)
            torch.testing.assert_close(native[1:], fallback[1:])

    def test_strided_small_page_and_legacy_kernels_match_reference(self):
        self.assertTrue(torch.cuda.is_available())
        for page in (16, 32):
            storage, pool, q, weights, table, lengths, expected = _inputs(page, "cuda")
            before = storage.clone()
            with (
                mock.patch.object(glm53_indexer_score, "_HAS_NATIVE_MQA", False),
                mock.patch.object(
                    _indexer_score, "supports_paged_mqa_logits", return_value=False
                ),
            ):
                actual = glm53_indexer_score.fp8_paged_indexer_score(
                    q, weights, pool, table, lengths, page, 3 * page
                )
            columns = torch.arange(page * 3, device="cuda")
            valid = columns < lengths.reshape(-1, 1)
            torch.testing.assert_close(
                actual[valid], expected[valid], rtol=3e-5, atol=3e-5
            )
            if page == 16:
                self.assertTrue(torch.isneginf(actual[~valid]).all())
            else:
                self.assertTrue((actual[~valid] == 0).all())
            torch.testing.assert_close(storage, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
