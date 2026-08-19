from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase, main, mock, skipUnless

import torch

from rtp_llm.models_py.modules.base.cuda import indexer_op as cuda_indexer
from rtp_llm.models_py.modules.base.cuda.indexer_op import (
    _resolve_paged_indexer_inputs,
)
from rtp_llm.models_py.modules.hybrid.indexer import Indexer

# deep_gemm is only importable on accelerator lanes; on a CPU node cuda_indexer.deep_gemm
# is None, so a test that patches attributes on it would ERROR rather than skip and turn
# the lane red. Gate those tests on its presence.
_HAS_DEEP_GEMM = cuda_indexer.deep_gemm is not None


class IndexerTargetVerifyTest(TestCase):
    def test_decode_preserves_existing_paged_metadata(self):
        context_lens = torch.tensor([1024, 1536], dtype=torch.int32)
        cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
        fmha_params = SimpleNamespace(kvlen_d=context_lens)
        attention_inputs = SimpleNamespace(
            is_target_verify=False,
            decode_cu_seqlens_device=cu_seqlens_q,
        )

        lengths, request_indices, actual_cu_seqlens_q = (
            _resolve_paged_indexer_inputs(2, fmha_params, attention_inputs)
        )

        self.assertIs(lengths, context_lens)
        self.assertIsNone(request_indices)
        self.assertIs(actual_cu_seqlens_q, cu_seqlens_q)

    def test_target_verify_uses_paged_topk(self):
        paged_result = object()
        indexer_op = mock.Mock()
        indexer_op._get_topk_paged.return_value = paged_result
        owner = SimpleNamespace(
            indexer_op=indexer_op,
            _is_sparse_prefill_cp=lambda _: False,
        )
        attention_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
        )
        q_fp8 = object()
        weights = object()
        kv_cache = object()
        fmha_params = object()

        actual = Indexer._compute_topk(
            owner,
            q_fp8,
            weights,
            kv_cache,
            fmha_params,
            attention_inputs,
            None,
        )

        self.assertIs(actual, paged_result)
        indexer_op._get_topk_paged.assert_called_once_with(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )
        indexer_op._get_topk_ragged.assert_not_called()

    def test_target_verify_with_prefill_cp_fails_closed(self):
        # The paged entry-count guard cannot catch this: at next_n + 1 == 2 the
        # rank-local and full-length row counts coincide for cp_size 2, 4 and 8, so
        # the mismatch would pass through and produce silently wrong topk indices.
        indexer_op = mock.Mock()
        owner = SimpleNamespace(
            indexer_op=indexer_op,
            _is_sparse_prefill_cp=lambda _: True,
        )
        attention_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
        )

        with self.assertRaises(ValueError):
            Indexer._compute_topk(
                owner,
                object(),
                object(),
                object(),
                object(),
                attention_inputs,
                object(),
            )
        indexer_op._get_topk_paged.assert_not_called()
        indexer_op._get_topk_ragged_cp.assert_not_called()

    def test_target_verify_expands_per_query_metadata(self):
        expanded_seq_lens = torch.tensor([2047, 2048, 2049, 2050])
        batch_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        fmha_params = SimpleNamespace(
            expanded_seq_lens=expanded_seq_lens,
            batch_indice_d=batch_indices,
            kvlen_d=torch.tensor([2048, 2050]),
        )
        attention_inputs = SimpleNamespace(is_target_verify=True)

        lengths, request_indices, cu_seqlens_q = _resolve_paged_indexer_inputs(
            4, fmha_params, attention_inputs
        )

        self.assertIs(lengths, expanded_seq_lens)
        self.assertIs(request_indices, batch_indices)
        torch.testing.assert_close(
            cu_seqlens_q,
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    @skipUnless(_HAS_DEEP_GEMM, "requires deep_gemm (accelerator lane)")
    def test_cuda_paged_topk_uses_target_verify_metadata(self):
        query_count = 4
        op = object.__new__(cuda_indexer.IndexerOp)
        op.index_n_heads = 2
        op.index_head_dim = 8
        op.index_topk = 2048
        op.blocksize = 64
        op.block_size = 8
        q_fp8 = torch.zeros(
            (query_count, op.index_n_heads, op.index_head_dim),
            dtype=torch.float8_e4m3fn,
        )
        weights = torch.ones((query_count, op.index_n_heads))
        kv_cache = SimpleNamespace(
            kv_scale_base=torch.zeros((4, op.blocksize, 12), dtype=torch.uint8)
        )
        context_lens = torch.tensor([2047, 2048, 2049, 2050], dtype=torch.int32)
        request_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        fmha_params = SimpleNamespace(
            expanded_seq_lens=context_lens,
            batch_indice_d=request_indices,
            kvlen_d=torch.tensor([2048, 2050], dtype=torch.int32),
        )
        block_tables = torch.tensor(
            [[0] * 33, [1] * 33], dtype=torch.int32
        )
        attention_inputs = SimpleNamespace(
            is_target_verify=True,
            kv_cache_kernel_block_id_device=block_tables,
        )
        logits = torch.zeros((query_count, 33 * op.blocksize))
        expected = torch.zeros((query_count, op.index_topk), dtype=torch.int32)

        with mock.patch.object(
            cuda_indexer.deep_gemm,
            "get_paged_mqa_logits_metadata",
            return_value=object(),
        ) as metadata_mock, mock.patch.object(
            cuda_indexer.deep_gemm,
            "get_num_sms",
            return_value=64,
        ), mock.patch.object(
            cuda_indexer.deep_gemm,
            "fp8_paged_mqa_logits",
            return_value=logits,
        ) as logits_mock, mock.patch(
            "rtp_llm.models_py.kernels.cuda.fast_topk.fast_topk_transform_fused",
            return_value=expected,
        ) as topk_mock:
            actual = op._get_topk_paged(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )

        self.assertIs(actual, expected)
        self.assertIs(metadata_mock.call_args.args[0], context_lens)
        expanded_tables = logits_mock.call_args.args[4]
        torch.testing.assert_close(
            expanded_tables,
            block_tables.index_select(0, request_indices.to(torch.int64)),
            rtol=0,
            atol=0,
        )
        self.assertIs(topk_mock.call_args.kwargs["lengths"], context_lens)
        torch.testing.assert_close(
            topk_mock.call_args.kwargs["cu_seqlens_q"],
            torch.arange(query_count + 1, dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def _target_verify_op(self):
        op = object.__new__(cuda_indexer.IndexerOp)
        op.index_n_heads = 2
        op.index_head_dim = 8
        op.index_topk = 2048
        op.blocksize = 64
        op.block_size = 8
        return op

    def _target_verify_fixtures(
        self, op, context_lens, request_indices, block_rows=2
    ):
        query_count = context_lens.numel()
        q_fp8 = torch.zeros(
            (query_count, op.index_n_heads, op.index_head_dim),
            dtype=torch.float8_e4m3fn,
        )
        weights = torch.ones((query_count, op.index_n_heads))
        kv_cache = SimpleNamespace(
            kv_scale_base=torch.zeros((4, op.blocksize, 12), dtype=torch.uint8)
        )
        fmha_params = SimpleNamespace(
            expanded_seq_lens=context_lens,
            batch_indice_d=request_indices,
            kvlen_d=torch.tensor([2048, 2050], dtype=torch.int32),
        )
        block_tables = torch.stack(
            [torch.full((33,), r, dtype=torch.int32) for r in range(block_rows)]
        )
        attention_inputs = SimpleNamespace(
            is_target_verify=True,
            kv_cache_kernel_block_id_device=block_tables,
        )
        return q_fp8, weights, kv_cache, fmha_params, attention_inputs

    @contextmanager
    def _patch_deep_gemm(self, logits):
        with mock.patch.object(
            cuda_indexer.deep_gemm,
            "get_paged_mqa_logits_metadata",
            return_value=object(),
        ) as metadata_mock, mock.patch.object(
            cuda_indexer.deep_gemm,
            "get_num_sms",
            return_value=64,
        ), mock.patch.object(
            cuda_indexer.deep_gemm,
            "fp8_paged_mqa_logits",
            return_value=logits,
        ) as logits_mock, mock.patch(
            "rtp_llm.models_py.kernels.cuda.fast_topk.fast_topk_transform_fused",
            return_value=torch.zeros((logits.size(0), 2048), dtype=torch.int32),
        ) as topk_mock:
            yield metadata_mock, logits_mock, topk_mock

    @skipUnless(_HAS_DEEP_GEMM, "requires deep_gemm (accelerator lane)")
    def test_target_verify_dispatch_selects_decode_transform_kernel(self):
        """Pin the load-bearing kernel dispatch for target verify.

        Replays the C++ predicate from fast_topk_transform_fused
        (``is_decode = row_starts is None and prefill_bs == B``) on the exact
        arguments _get_topk_paged hands to the kernel wrapper, proving target
        verify lands on the decode transform kernel (request-local output),
        never the prefill branch (which needs an unsupplied src_page_table).
        """
        op = self._target_verify_op()
        context_lens = torch.tensor([2047, 2048, 2049, 2050], dtype=torch.int32)
        request_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        q_fp8, weights, kv_cache, fmha_params, attention_inputs = (
            self._target_verify_fixtures(op, context_lens, request_indices)
        )
        logits = torch.zeros((context_lens.numel(), 33 * op.blocksize))

        with self._patch_deep_gemm(logits) as (_, _, topk_mock):
            op._get_topk_paged(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )

        kwargs = topk_mock.call_args.kwargs
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        score = kwargs["score"]
        prefill_bs = int(cu_seqlens_q.numel() - 1)
        batch = int(score.size(0))
        self.assertIsNone(kwargs["row_starts"])
        self.assertEqual(cu_seqlens_q.dtype, torch.int32)
        # is_decode := row_starts is None and prefill_bs == B  (fast_topk.cu)
        self.assertTrue(kwargs["row_starts"] is None and prefill_bs == batch)

    @skipUnless(_HAS_DEEP_GEMM, "requires deep_gemm (accelerator lane)")
    def test_paged_dispatch_fails_closed_on_row_count_mismatch(self):
        op = self._target_verify_op()
        context_lens = torch.tensor([2047, 2048, 2049, 2050], dtype=torch.int32)
        request_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        q_fp8, weights, kv_cache, fmha_params, attention_inputs = (
            self._target_verify_fixtures(op, context_lens, request_indices)
        )
        # Extra score row => prefill_bs != B => the kernel would take the
        # prefill branch (needs src_page_table) instead of decode; fail closed.
        logits = torch.zeros((context_lens.numel() + 1, 33 * op.blocksize))

        with self._patch_deep_gemm(logits):
            with self.assertRaises(ValueError):
                op._get_topk_paged(
                    q_fp8, weights, kv_cache, fmha_params, attention_inputs
                )

    def test_decode_cu_seqlens_cast_to_int32(self):
        context_lens = torch.tensor([1024, 1536], dtype=torch.int32)
        cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int64)
        fmha_params = SimpleNamespace(kvlen_d=context_lens)
        attention_inputs = SimpleNamespace(
            is_target_verify=False,
            decode_cu_seqlens_device=cu_seqlens_q,
        )

        _, _, actual_cu_seqlens_q = _resolve_paged_indexer_inputs(
            2, fmha_params, attention_inputs
        )

        self.assertEqual(actual_cu_seqlens_q.dtype, torch.int32)

    def test_context_len_count_mismatch_fails_closed(self):
        context_lens = torch.tensor([2048, 2050], dtype=torch.int32)
        request_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        fmha_params = SimpleNamespace(
            expanded_seq_lens=context_lens,
            batch_indice_d=request_indices,
            kvlen_d=context_lens,
        )
        attention_inputs = SimpleNamespace(is_target_verify=True)

        with self.assertRaises(ValueError):
            _resolve_paged_indexer_inputs(4, fmha_params, attention_inputs)

    def test_request_index_count_mismatch_fails_closed(self):
        context_lens = torch.tensor([2047, 2048, 2049, 2050], dtype=torch.int32)
        request_indices = torch.tensor([0, 1], dtype=torch.int32)
        fmha_params = SimpleNamespace(
            expanded_seq_lens=context_lens,
            batch_indice_d=request_indices,
            kvlen_d=torch.tensor([2048, 2050], dtype=torch.int32),
        )
        attention_inputs = SimpleNamespace(is_target_verify=True)

        with self.assertRaises(ValueError):
            _resolve_paged_indexer_inputs(4, fmha_params, attention_inputs)


if __name__ == "__main__":
    main()
