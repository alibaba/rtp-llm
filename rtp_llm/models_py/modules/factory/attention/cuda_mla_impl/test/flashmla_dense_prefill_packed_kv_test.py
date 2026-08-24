import os
import sys
import unittest
from typing import Sequence
from unittest import mock

_LOCAL_DEEP_GEMM_PATH = os.environ.get("RTP_LOCAL_DEEP_GEMM_PATH")
if _LOCAL_DEEP_GEMM_PATH:
    sys.path.insert(0, _LOCAL_DEEP_GEMM_PATH)

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    MlaFlashMLAPrefillOp,
)
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
    CudaF16Linear,
)
from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import (
    plan_kimi_k3_chunk_rounds,
)
from rtp_llm.ops.compute_ops import LayerKVCache


class FlashMLADensePrefillPackedKVTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("packed K3 projection requires SM100")
        import deep_gemm

        self.assertTrue(callable(deep_gemm.bf16_gemm_nt_skip_head_mid))
        if _LOCAL_DEEP_GEMM_PATH:
            self.assertTrue(deep_gemm.__file__.startswith(_LOCAL_DEEP_GEMM_PATH))

    @staticmethod
    def _make_op(num_heads: int = 12) -> MlaFlashMLAPrefillOp:
        op = object.__new__(MlaFlashMLAPrefillOp)
        op.num_heads = num_heads
        op.kv_lora_rank = 512
        op.qk_nope_head_dim = 128
        op.qk_rope_head_dim = 64
        op.v_head_dim = 128
        op.weights = [{}]
        op.quant_config = None
        op._direct_attn_inputs = None
        op._direct_block_table_width = 0
        return op

    def _assert_close_chunked(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        chunk_tokens: int = 4096,
    ) -> None:
        self.assertEqual(actual.shape, expected.shape)
        for start in range(0, actual.shape[0], chunk_tokens):
            torch.testing.assert_close(
                actual[start : start + chunk_tokens],
                expected[start : start + chunk_tokens],
                rtol=0,
                atol=0,
            )

    @staticmethod
    def _indptr(lengths: Sequence[int]) -> list[int]:
        result = [0]
        for length in lengths:
            result.append(result[-1] + length)
        return result

    @staticmethod
    def _reference_projection(
        op: MlaFlashMLAPrefillOp,
        linear: CudaF16Linear,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = compressed_kv.shape[0]
        expanded_dim = op.qk_nope_head_dim + op.v_head_dim
        kv = linear(compressed_kv).view(tokens, op.num_heads, expanded_dim)
        k = compressed_kv.new_empty(
            tokens,
            op.num_heads,
            op.qk_nope_head_dim + op.qk_rope_head_dim,
        )
        k[..., : op.qk_nope_head_dim].copy_(kv[..., : op.qk_nope_head_dim])
        k[..., op.qk_nope_head_dim :].copy_(k_pe.view(tokens, 1, -1))
        return k, kv[..., op.qk_nope_head_dim :]

    def _run_synthetic_cache_pipeline(
        self,
        q_lens: Sequence[int],
        reuse_lens: Sequence[int],
        *,
        projection_repeats: int = 1,
        num_heads: int = 12,
        strided_query_k_pe: bool = False,
    ) -> None:
        self.assertEqual(len(q_lens), len(reuse_lens))
        self.assertTrue(q_lens)
        self.assertTrue(all(length > 0 for length in q_lens))

        torch.manual_seed(sum(q_lens) + sum(reuse_lens) + 1234)
        op = self._make_op(num_heads)
        op.page_size = 128

        import flash_mla.cuda as flash_mla_cuda

        op.flash_mla_cuda = flash_mla_cuda
        qo_indptr = self._indptr(q_lens)
        kv_lens = [q_len + reuse_len for q_len, reuse_len in zip(q_lens, reuse_lens)]
        kv_indptr = self._indptr(kv_lens)
        op.qo_indptr = torch.tensor(qo_indptr, device="cuda", dtype=torch.int32)
        op.kv_indptr = torch.tensor(kv_indptr, device="cuda", dtype=torch.int32)
        op.max_q_len = max(q_lens)
        op.max_kv_len = max(kv_lens)
        op.total_kv_lens = sum(kv_lens)
        op.batch_size = len(q_lens)
        op.scale = (op.qk_nope_head_dim + op.qk_rope_head_dim) ** -0.5

        suffix_tokens = sum(q_lens)
        compressed_kv = torch.randn(
            (suffix_tokens, op.kv_lora_rank),
            device="cuda",
            dtype=torch.bfloat16,
        )
        if strided_query_k_pe:
            k_pe_storage = torch.randn(
                (suffix_tokens, op.qk_rope_head_dim + 1),
                device="cuda",
                dtype=torch.bfloat16,
            )
            k_pe = k_pe_storage[:, : op.qk_rope_head_dim]
            self.assertFalse(k_pe.is_contiguous())
            self.assertEqual(k_pe.stride(1), 1)
        else:
            k_pe = torch.randn(
                (suffix_tokens, op.qk_rope_head_dim),
                device="cuda",
                dtype=torch.bfloat16,
            )

        page_counts = [
            (length + op.page_size - 1) // op.page_size for length in reuse_lens
        ]
        total_pages = sum(page_counts)
        cache = torch.randn(
            (
                max(total_pages + 1, 1),
                op.page_size,
                op.kv_lora_rank + op.qk_rope_head_dim,
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )
        physical_pages = list(range(total_pages, 0, -1))
        reuse_page_indices: list[int] = []
        batch_reuse_info: list[list[int]] = []
        expected_compressed_parts = []
        expected_k_pe_parts = []
        page_cursor = 0
        for batch_idx, (q_len, reuse_len, page_count) in enumerate(
            zip(q_lens, reuse_lens, page_counts)
        ):
            batch_page_ids = physical_pages[page_cursor : page_cursor + page_count]
            block_start = len(reuse_page_indices)
            reuse_page_indices.extend(batch_page_ids)
            batch_reuse_info.append([batch_idx, reuse_len, block_start, page_count])

            if page_count:
                prefix = cache[batch_page_ids].reshape(
                    -1, op.kv_lora_rank + op.qk_rope_head_dim
                )[:reuse_len]
            else:
                prefix = cache.new_empty((0, op.kv_lora_rank + op.qk_rope_head_dim))
            suffix_start = qo_indptr[batch_idx]
            suffix_end = suffix_start + q_len
            expected_compressed_parts.append(
                torch.cat(
                    [
                        prefix[:, : op.kv_lora_rank],
                        compressed_kv[suffix_start:suffix_end],
                    ],
                    dim=0,
                )
            )
            expected_k_pe_parts.append(
                torch.cat(
                    [prefix[:, op.kv_lora_rank :], k_pe[suffix_start:suffix_end]],
                    dim=0,
                )
            )
            page_cursor += page_count

        expected_compressed = torch.cat(expected_compressed_parts, dim=0)
        expected_k_pe = torch.cat(expected_k_pe_parts, dim=0)
        op.has_reuse_cache = total_pages > 0
        op.reuse_cache_page_indice = torch.tensor(
            reuse_page_indices, device="cuda", dtype=torch.int32
        )
        op.batch_reuse_info_vec = torch.tensor(
            batch_reuse_info, device="cuda", dtype=torch.int32
        )

        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = cache
        gathered_compressed, gathered_k_pe = op._gather_reused_kv(
            compressed_kv,
            k_pe,
            kv_cache if op.has_reuse_cache else None,
        )
        self._assert_close_chunked(gathered_compressed, expected_compressed)
        self._assert_close_chunked(gathered_k_pe, expected_k_pe)

        checkpoint_weight = torch.randn(
            (
                op.kv_lora_rank,
                op.num_heads * (op.qk_nope_head_dim + op.v_head_dim),
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)
        reference_k, reference_v = self._reference_projection(
            op, linear, expected_compressed, expected_k_pe
        )
        with mock.patch.object(
            LinearFactory, "create_linear_from_weights", return_value=linear
        ):
            for _ in range(projection_repeats):
                packed_k, packed_v = op._project_kv(
                    gathered_compressed, gathered_k_pe, 0
                )
            fused_kv = op._project_reused_kv_with_gap_fill(
                compressed_kv,
                k_pe,
                kv_cache if op.has_reuse_cache else None,
                linear,
            )

        if op.has_reuse_cache:
            self.assertIsNotNone(fused_kv)
            fused_k, fused_v = fused_kv
            self._assert_close_chunked(fused_k, reference_k)
            self._assert_close_chunked(fused_v, reference_v)
            self.assertEqual(
                fused_k.untyped_storage().data_ptr(),
                fused_v.untyped_storage().data_ptr(),
            )
            self.assertEqual(fused_k.stride(), packed_k.stride())
            self.assertEqual(fused_v.stride(), packed_v.stride())
        else:
            self.assertIsNone(fused_kv)

        self._assert_close_chunked(packed_k, reference_k)
        self._assert_close_chunked(packed_v, reference_v)
        self.assertEqual(
            packed_k.untyped_storage().data_ptr(),
            packed_v.untyped_storage().data_ptr(),
        )
        expected_stride = (op.num_heads * 320, 320, 1)
        self.assertEqual(packed_k.stride(), expected_stride)
        self.assertEqual(packed_v.stride(), expected_stride)
        self.assertEqual(packed_k.storage_offset(), 0)
        self.assertEqual(packed_v.storage_offset(), 192)

        baseline_bytes = (
            reference_k.untyped_storage().nbytes()
            + reference_v.untyped_storage().nbytes()
        )
        packed_bytes = packed_k.untyped_storage().nbytes()
        expected_saved_bytes = (
            op.total_kv_lens
            * op.num_heads
            * op.qk_nope_head_dim
            * gathered_compressed.element_size()
        )
        self.assertEqual(baseline_bytes - packed_bytes, expected_saved_bytes)

        q = torch.randn(
            (
                suffix_tokens,
                op.num_heads,
                op.qk_nope_head_dim + op.qk_rope_head_dim,
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )
        reference_out = op._dense_attention(q, reference_k, reference_v)
        packed_out = op._dense_attention(q, packed_k, packed_v)
        self._assert_close_chunked(packed_out, reference_out)
        if op.has_reuse_cache:
            fused_out = op._dense_attention(q, fused_k, fused_v)
            self._assert_close_chunked(fused_out, reference_out)

    def test_synthetic_cache_reuse_scenarios(self) -> None:
        cases = (
            ("cache_miss", [257], [0], 24),
            ("full_hit", [1], [4096], 1),
            ("partial_hit", [257], [4096], 1),
            ("partial_page_hit", [31], [4103], 1),
            ("mixed_batch", [17, 33, 65, 9], [0, 4103, 8192, 2051], 1),
        )
        for name, q_lens, reuse_lens, repeats in cases:
            with self.subTest(name=name):
                self._run_synthetic_cache_pipeline(
                    q_lens,
                    reuse_lens,
                    projection_repeats=repeats,
                    strided_query_k_pe=name == "mixed_batch",
                )

    def test_prefix_projection_tp_head_counts(self) -> None:
        for tensor_parallel_size, num_heads in ((2, 48), (4, 24), (8, 12)):
            with self.subTest(tensor_parallel_size=tensor_parallel_size):
                self._run_synthetic_cache_pipeline(
                    [31, 17],
                    [257, 129],
                    num_heads=num_heads,
                    strided_query_k_pe=True,
                )

    def test_synthetic_whole_model_64k_chunk_boundary(self) -> None:
        rounds = plan_kimi_k3_chunk_rounds(
            [65537],
            [0],
            chunk_budget=65536,
            page_size=4096,
        )
        self.assertEqual(len(rounds), 2)
        final_slice = rounds[-1].slices[0]
        self.assertEqual(final_slice.new_length, 1)
        self.assertEqual(final_slice.absolute_start, 65536)
        self.assertTrue(final_slice.terminal)

        self._run_synthetic_cache_pipeline(
            [final_slice.new_length],
            [final_slice.absolute_start],
        )

    def test_project_kv_and_attention_match_reference(self) -> None:
        torch.manual_seed(456)
        tokens = 257
        op = self._make_op()
        compressed_kv = torch.randn(
            (tokens, op.kv_lora_rank), device="cuda", dtype=torch.bfloat16
        )
        k_pe = torch.randn(
            (tokens, op.qk_rope_head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        checkpoint_weight = torch.randn(
            (
                op.kv_lora_rank,
                op.num_heads * (op.qk_nope_head_dim + op.v_head_dim),
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )
        linear = CudaF16Linear(checkpoint_weight)

        expanded_dim = op.qk_nope_head_dim + op.v_head_dim
        reference_kv = linear(compressed_kv).view(tokens, op.num_heads, expanded_dim)
        reference_k = compressed_kv.new_empty(
            tokens,
            op.num_heads,
            op.qk_nope_head_dim + op.qk_rope_head_dim,
        )
        reference_k[..., : op.qk_nope_head_dim].copy_(
            reference_kv[..., : op.qk_nope_head_dim]
        )
        reference_k[..., op.qk_nope_head_dim :].copy_(k_pe.view(tokens, 1, -1))
        reference_v = reference_kv[..., op.qk_nope_head_dim :]

        with mock.patch.object(
            LinearFactory, "create_linear_from_weights", return_value=linear
        ):
            packed_k, packed_v = op._project_kv(compressed_kv, k_pe, 0)

        torch.testing.assert_close(packed_k, reference_k, rtol=0, atol=0)
        torch.testing.assert_close(packed_v, reference_v, rtol=0, atol=0)
        self.assertEqual(
            packed_k.untyped_storage().data_ptr(),
            packed_v.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            reference_k.untyped_storage().data_ptr(),
            reference_v.untyped_storage().data_ptr(),
        )
        self.assertEqual(packed_k.stride(), (3840, 320, 1))
        self.assertEqual(packed_v.stride(), (3840, 320, 1))
        self.assertEqual(packed_k.storage_offset(), 0)
        self.assertEqual(packed_v.storage_offset(), 192)

        import flash_mla.cuda as flash_mla_cuda

        query_tokens = 17
        op.flash_mla_cuda = flash_mla_cuda
        op.qo_indptr = torch.tensor([0, query_tokens], device="cuda", dtype=torch.int32)
        op.kv_indptr = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        op.max_q_len = query_tokens
        op.max_kv_len = tokens
        op.scale = (op.qk_nope_head_dim + op.qk_rope_head_dim) ** -0.5
        q = torch.randn(
            (
                query_tokens,
                op.num_heads,
                op.qk_nope_head_dim + op.qk_rope_head_dim,
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )

        reference_out = op._dense_attention(q, reference_k, reference_v)
        packed_out = op._dense_attention(q, packed_k, packed_v)
        torch.testing.assert_close(packed_out, reference_out, rtol=0, atol=0)

    def test_project_kv_keeps_linear_implementations_without_packed_api(
        self,
    ) -> None:
        torch.manual_seed(789)
        tokens = 17
        op = self._make_op()
        compressed_kv = torch.randn(
            (tokens, op.kv_lora_rank), device="cuda", dtype=torch.bfloat16
        )
        k_pe = torch.randn(
            (tokens, op.qk_rope_head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        checkpoint_weight = torch.randn(
            (
                op.kv_lora_rank,
                op.num_heads * (op.qk_nope_head_dim + op.v_head_dim),
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )

        class LinearWithoutPackedAPI:
            bias = None

            def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.linear(inputs, checkpoint_weight.T)

        linear = LinearWithoutPackedAPI()
        op.has_reuse_cache = True
        self.assertIsNone(
            op._project_reused_kv_with_gap_fill(compressed_kv, k_pe, None, linear)
        )
        with mock.patch.object(
            LinearFactory, "create_linear_from_weights", return_value=linear
        ):
            actual_k, actual_v = op._project_kv(compressed_kv, k_pe, 0)

        expanded_dim = op.qk_nope_head_dim + op.v_head_dim
        reference_kv = linear(compressed_kv).view(tokens, op.num_heads, expanded_dim)
        reference_k = compressed_kv.new_empty(
            tokens,
            op.num_heads,
            op.qk_nope_head_dim + op.qk_rope_head_dim,
        )
        reference_k[..., : op.qk_nope_head_dim].copy_(
            reference_kv[..., : op.qk_nope_head_dim]
        )
        reference_k[..., op.qk_nope_head_dim :].copy_(k_pe.view(tokens, 1, -1))

        torch.testing.assert_close(actual_k, reference_k, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_v,
            reference_kv[..., op.qk_nope_head_dim :],
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
