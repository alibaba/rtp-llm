import os
import sys
import unittest
from types import SimpleNamespace
from typing import Any, Sequence
from unittest import mock

_LOCAL_DEEP_GEMM_PATH = os.environ.get("RTP_LOCAL_DEEP_GEMM_PATH")
if _LOCAL_DEEP_GEMM_PATH:
    sys.path.insert(0, _LOCAL_DEEP_GEMM_PATH)

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_dense_prefill as flashmla_dense_prefill_module,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    MlaFlashMLAPrefillOp,
    _FlashMLAPrefixChunkWorkspace,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_state_merge_triton import (
    is_merge_attention_states_in_place_supported,
    merge_attention_states_in_place,
)
from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import plan_kimi_k3_chunk_rounds
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
    def _make_op(
        prefix_chunk_tokens: int = 0,
        num_heads: int = 12,
        qk_rope_head_dim: int = 64,
    ) -> MlaFlashMLAPrefillOp:
        return MlaFlashMLAPrefillOp(
            num_heads=num_heads,
            kv_lora_rank=512,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=128,
            v_head_dim=128,
            page_size=128,
            softmax_extra_scale=1.0,
            use_mla=True,
            weights=[{}],
            prefix_chunk_tokens=prefix_chunk_tokens,
        )

    def test_gather_without_reuse_supports_zero_rope_dim(self) -> None:
        op = self._make_op(qk_rope_head_dim=0)
        tokens = 8
        compressed_kv = torch.empty(
            (tokens, op.kv_lora_rank), dtype=torch.bfloat16, device="cuda"
        )
        k_pe = torch.empty((tokens, 1, 0), dtype=torch.bfloat16, device="cuda")

        gathered_compressed, gathered_k_pe = op._gather_reused_kv(
            compressed_kv, k_pe, None
        )

        self.assertIs(gathered_compressed, compressed_kv)
        self.assertEqual(tuple(gathered_k_pe.shape), (tokens, 0))

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
        prefix_chunk_tokens: int = 0,
    ) -> MlaFlashMLAPrefillOp:
        self.assertEqual(len(q_lens), len(reuse_lens))
        self.assertTrue(q_lens)
        self.assertTrue(all(length > 0 for length in q_lens))

        torch.manual_seed(sum(q_lens) + sum(reuse_lens) + 1234)
        op = self._make_op(prefix_chunk_tokens, num_heads=num_heads)
        qo_indptr = self._indptr(q_lens)
        kv_lens = [q_len + reuse_len for q_len, reuse_len in zip(q_lens, reuse_lens)]
        kv_indptr = self._indptr(kv_lens)

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
        reuse_cache_page_indice = torch.tensor(
            reuse_page_indices, device="cuda", dtype=torch.int32
        )
        batch_reuse_info_vec = torch.tensor(
            batch_reuse_info, device="cuda", dtype=torch.int32
        )
        params = SimpleNamespace(
            qo_indptr_h=torch.tensor(qo_indptr, dtype=torch.int32),
            prefill_ragged_kv_len_indptr_h=torch.tensor(kv_indptr, dtype=torch.int32),
            qo_indptr_d=torch.tensor(qo_indptr, device="cuda", dtype=torch.int32),
            prefill_ragged_kv_len_indptr_d=torch.tensor(
                kv_indptr, device="cuda", dtype=torch.int32
            ),
            reuse_cache_page_indice_d=reuse_cache_page_indice,
            batch_reuse_info_vec_h=torch.tensor(batch_reuse_info, dtype=torch.int32),
            batch_reuse_info_vec_d=batch_reuse_info_vec,
        )
        op.plan(params)
        if op.prefix_chunks:
            self.assertIsNotNone(op._prefix_chunk_metadata)
            metadata_ptr = op._prefix_chunk_metadata.untyped_storage().data_ptr()
            for chunk in op.prefix_chunks:
                for tensor in (
                    chunk.qo_indptr,
                    chunk.kv_indptr,
                    chunk.gather_qo_indptr,
                    chunk.batch_reuse_info,
                ):
                    self.assertEqual(tensor.untyped_storage().data_ptr(), metadata_ptr)

        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = cache
        gathered_compressed, gathered_k_pe = op._gather_reused_kv(
            compressed_kv,
            k_pe,
            kv_cache if op.has_reuse_cache else None,
        )
        self._assert_close_chunked(gathered_compressed, expected_compressed)
        self._assert_close_chunked(gathered_k_pe, expected_k_pe)

        if op.prefix_chunks:
            page_indices = op._current_reuse_cache_page_indices()
            flat_k_pe = k_pe.contiguous()
            for chunk in op.prefix_chunks:
                actual_compressed, actual_k_pe = op._gather_prefix_chunk(
                    chunk,
                    compressed_kv,
                    flat_k_pe,
                    kv_cache,
                    page_indices,
                )
                expected_chunk_compressed = []
                expected_chunk_k_pe = []
                for request_idx, start, length in zip(
                    chunk.spec.request_indices,
                    chunk.spec.prefix_starts,
                    chunk.spec.prefix_lens,
                ):
                    expected_chunk_compressed.append(
                        expected_compressed_parts[request_idx][start : start + length]
                    )
                    expected_chunk_k_pe.append(
                        expected_k_pe_parts[request_idx][start : start + length]
                    )
                torch.testing.assert_close(
                    actual_compressed,
                    torch.cat(expected_chunk_compressed),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual_k_pe,
                    torch.cat(expected_chunk_k_pe),
                    rtol=0,
                    atol=0,
                )

        checkpoint_weight = torch.randn(
            (
                op.kv_lora_rank,
                op.num_heads * (op.qk_nope_head_dim + op.v_head_dim),
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )
        checkpoint_weight.mul_(op.kv_lora_rank**-0.5)
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

        if op.prefix_chunks:
            reference_out, reference_lse = op._run_dense_attention(
                q,
                reference_k,
                reference_v,
                qo_indptr=op.qo_indptr,
                kv_indptr=op.kv_indptr,
                max_q_len=op.max_q_len,
                max_kv_len=op.max_kv_len,
                causal=True,
            )
            allocated_workspaces: list[_FlashMLAPrefixChunkWorkspace] = []
            allocate_workspace = _FlashMLAPrefixChunkWorkspace.allocate

            def tracked_allocate_workspace(
                **kwargs: Any,
            ) -> _FlashMLAPrefixChunkWorkspace:
                workspace = allocate_workspace(**kwargs)
                allocated_workspaces.append(workspace)
                return workspace

            with mock.patch.object(
                LinearFactory, "create_linear_from_weights", return_value=linear
            ), mock.patch.object(
                _FlashMLAPrefixChunkWorkspace,
                "allocate",
                side_effect=tracked_allocate_workspace,
            ), mock.patch.object(
                flashmla_dense_prefill_module.rtp_llm_ops,
                "gather_mla_latent_and_fill_k_pe",
                wraps=flashmla_dense_prefill_module.rtp_llm_ops.gather_mla_latent_and_fill_k_pe,
            ) as fused_prefix_gather, mock.patch.object(
                linear,
                "forward_skip_head_mid",
                wraps=linear.forward_skip_head_mid,
            ) as packed_projection, mock.patch.object(
                op,
                "_gather_prefix_chunk",
                wraps=op._gather_prefix_chunk,
            ) as gather_prefix_chunk, mock.patch.object(
                op,
                "_project_kv",
                wraps=op._project_kv,
            ) as project_kv, mock.patch.object(
                op,
                "_run_dense_attention",
                wraps=op._run_dense_attention,
            ) as run_dense_attention, mock.patch.object(
                flashmla_dense_prefill_module,
                "merge_attention_states_in_place",
                wraps=merge_attention_states_in_place,
            ) as merge_attention_states:
                chunked_out = op.forward(q, compressed_kv, k_pe, kv_cache, 0)
            self.assertEqual(len(allocated_workspaces), 1)
            chunk_workspace = allocated_workspaces.pop()

            self.assertEqual(gather_prefix_chunk.call_count, 0)
            self.assertEqual(fused_prefix_gather.call_count, len(op.prefix_chunks))
            self.assertIsNone(chunk_workspace.k_pe)
            packed_storage = chunk_workspace.packed_kv
            self.assertIsNotNone(packed_storage)
            assert packed_storage is not None
            for chunk, call in zip(
                op.prefix_chunks,
                fused_prefix_gather.call_args_list,
                strict=True,
            ):
                workspace_compressed, workspace_packed = call.args[:2]
                self.assertEqual(
                    tuple(workspace_compressed.shape),
                    (chunk.spec.kv_tokens, op.kv_lora_rank),
                )
                self.assertEqual(
                    tuple(workspace_packed.shape),
                    (chunk.spec.kv_tokens, op.num_heads * 320),
                )
                self.assertEqual(
                    workspace_compressed.untyped_storage().data_ptr(),
                    chunk_workspace.compressed_kv.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    workspace_packed.untyped_storage().data_ptr(),
                    packed_storage.untyped_storage().data_ptr(),
                )
                self.assertIs(call.args[6], chunk.batch_reuse_info)
                self.assertIs(call.args[7], chunk.gather_qo_indptr)
                self.assertEqual(call.args[9], 320)
                self.assertEqual(call.args[10], op.qk_nope_head_dim)

            self.assertEqual(project_kv.call_count, 1)
            self.assertIsNone(project_kv.call_args_list[0].kwargs.get("packed_output"))
            self.assertEqual(packed_projection.call_count, len(op.prefix_chunks) + 1)
            self.assertIsNone(packed_projection.call_args_list[0].kwargs.get("output"))
            for chunk, call in zip(
                op.prefix_chunks,
                packed_projection.call_args_list[1:],
                strict=True,
            ):
                workspace_packed = call.kwargs["output"]
                self.assertIsNotNone(workspace_packed)
                assert workspace_packed is not None
                self.assertEqual(
                    tuple(workspace_packed.shape),
                    (chunk.spec.kv_tokens, op.num_heads * 320),
                )
                self.assertEqual(
                    workspace_packed.untyped_storage().data_ptr(),
                    packed_storage.untyped_storage().data_ptr(),
                )

            self.assertEqual(run_dense_attention.call_count, len(op.prefix_chunks) + 1)
            self.assertIsNone(run_dense_attention.call_args_list[0].kwargs.get("out"))
            self.assertIsNone(run_dense_attention.call_args_list[0].kwargs.get("lse"))
            self.assertTrue(run_dense_attention.call_args_list[0].kwargs["causal"])
            self.assertIs(
                run_dense_attention.call_args_list[0].kwargs["kv_indptr"],
                op.qo_indptr,
            )
            for call_index, (chunk, call) in enumerate(
                zip(
                    op.prefix_chunks,
                    run_dense_attention.call_args_list[1:],
                    strict=True,
                )
            ):
                workspace_packed = fused_prefix_gather.call_args_list[call_index].args[
                    1
                ]
                assert workspace_packed is not None
                self.assertFalse(call.kwargs["causal"])
                self.assertLessEqual(call.kwargs["max_kv_len"], prefix_chunk_tokens)
                attention_k, attention_v = call.args[1:3]
                packed_storage_ptr = workspace_packed.untyped_storage().data_ptr()
                self.assertEqual(
                    attention_k.untyped_storage().data_ptr(), packed_storage_ptr
                )
                self.assertEqual(
                    attention_v.untyped_storage().data_ptr(), packed_storage_ptr
                )
                self.assertEqual(
                    attention_k.storage_offset(), workspace_packed.storage_offset()
                )
                self.assertEqual(
                    attention_v.storage_offset(),
                    workspace_packed.storage_offset()
                    + op.qk_nope_head_dim
                    + op.qk_rope_head_dim,
                )
                expected_kv_stride = (op.num_heads * 320, 320, 1)
                self.assertEqual(attention_k.stride(), expected_kv_stride)
                self.assertEqual(attention_v.stride(), expected_kv_stride)
                workspace_out = call.kwargs["out"]
                workspace_lse = call.kwargs["lse"]
                self.assertEqual(
                    tuple(workspace_out.shape),
                    (chunk.spec.q_tokens, op.num_heads, op.v_head_dim),
                )
                self.assertEqual(
                    workspace_out.untyped_storage().data_ptr(),
                    chunk_workspace.attention_out.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    workspace_lse.untyped_storage().data_ptr(),
                    chunk_workspace.attention_lse_storage.untyped_storage().data_ptr(),
                )
                self.assertEqual(workspace_lse.stride(), (1, chunk.spec.q_tokens))
            self.assertEqual(merge_attention_states.call_count, len(op.prefix_chunks))
            expected_accumulator_dtype = (
                torch.float32 if len(op.prefix_chunks) > 1 else torch.bfloat16
            )
            for call in merge_attention_states.call_args_list:
                self.assertEqual(call.args[0].dtype, expected_accumulator_dtype)
            del chunk_workspace, allocated_workspaces

            with mock.patch.object(
                LinearFactory, "create_linear_from_weights", return_value=linear
            ):
                _, chunked_lse = op._forward_chunked_prefix(
                    q, compressed_kv, k_pe, kv_cache, 0
                )
            self.assertEqual(chunked_out.dtype, q.dtype)
            self.assertEqual(chunked_lse.stride(), reference_lse.stride())
            torch.testing.assert_close(
                chunked_out, reference_out, rtol=2e-2, atol=0.03125
            )
            torch.testing.assert_close(chunked_lse, reference_lse, rtol=1e-4, atol=1e-4)
        else:
            with mock.patch.object(
                LinearFactory, "create_linear_from_weights", return_value=linear
            ), mock.patch.object(
                op,
                "_forward_chunked_prefix",
                side_effect=AssertionError("cache miss must use the full path"),
            ):
                routed_out = op.forward(q, compressed_kv, k_pe, kv_cache, 0)
            torch.testing.assert_close(routed_out, reference_out, rtol=0, atol=0)
        return op

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

        op = self._run_synthetic_cache_pipeline(
            [final_slice.new_length],
            [final_slice.absolute_start],
            prefix_chunk_tokens=16384,
        )
        self.assertEqual(len(op.prefix_chunks), 4)

    def test_chunked_prefix_matches_full_attention(self) -> None:
        self._run_synthetic_cache_pipeline(
            [17, 33, 65, 9],
            [0, 256, 640, 384],
            prefix_chunk_tokens=256,
        )

    def test_cache_miss_with_chunking_enabled_uses_full_path(self) -> None:
        op = self._run_synthetic_cache_pipeline(
            [1, 17, 257],
            [0, 0, 0],
            prefix_chunk_tokens=16384,
        )
        self.assertEqual(op.prefix_chunks, ())

    def test_prefix_query_boundary_matrix_matches_full_attention(self) -> None:
        cases = (
            ("single_token", [1], [1], 1),
            ("page_minus_one", [17], [127], 1),
            ("page_exact", [17], [128], 1),
            ("page_plus_one", [17], [129], 1),
            ("chunk_minus_one", [1], [16383], 1),
            ("chunk_exact", [17], [16384], 1),
            ("chunk_plus_one", [257], [16385], 2),
            ("two_chunks_exact", [1], [32768], 2),
            ("large_query", [4096], [129], 1),
            ("mixed_batch", [1, 17, 257], [0, 16384, 16385], 3),
        )
        for name, q_lens, prefix_lens, expected_chunks in cases:
            with self.subTest(name=name):
                op = self._run_synthetic_cache_pipeline(
                    q_lens,
                    prefix_lens,
                    prefix_chunk_tokens=16384,
                )
                self.assertEqual(len(op.prefix_chunks), expected_chunks)

    def test_single_prefix_chunk_matches_full_attention(self) -> None:
        op = self._run_synthetic_cache_pipeline(
            [17],
            [257],
            prefix_chunk_tokens=16384,
        )
        self.assertEqual(len(op.prefix_chunks), 1)

    def test_non_page_aligned_prefix_and_mixed_batch(self) -> None:
        self._run_synthetic_cache_pipeline(
            [1, 17, 65],
            [257, 0, 4097],
            prefix_chunk_tokens=256,
        )

    def test_one_million_prefix_uses_64_real_chunks(self) -> None:
        op = self._run_synthetic_cache_pipeline(
            [1],
            [1024 * 1024],
            prefix_chunk_tokens=16384,
        )
        self.assertEqual(len(op.prefix_chunks), 64)

    def test_state_merge_supports_k3_head_count_and_flashmla_lse_layout(self) -> None:
        tokens = 257
        heads = 96
        head_size = 128
        torch.manual_seed(20260820)
        output = torch.randn(
            (tokens, heads, head_size), device="cuda", dtype=torch.bfloat16
        )
        other_output = torch.randn_like(output)
        output_lse = torch.randn(
            (heads, tokens), device="cuda", dtype=torch.float32
        ).transpose(0, 1)
        other_lse = torch.randn(
            (heads, tokens), device="cuda", dtype=torch.float32
        ).transpose(0, 1)
        self.assertEqual(output_lse.stride(), (1, tokens))

        max_lse = torch.maximum(output_lse, other_lse)
        output_exp = torch.exp(output_lse - max_lse)
        other_exp = torch.exp(other_lse - max_lse)
        denominator = output_exp + other_exp
        reference_output = (
            output.float() * (output_exp / denominator).unsqueeze(-1)
            + other_output.float() * (other_exp / denominator).unsqueeze(-1)
        ).to(torch.bfloat16)
        reference_lse = torch.log(denominator) + max_lse
        output_ptr = output.data_ptr()
        lse_ptr = output_lse.data_ptr()

        merge_attention_states_in_place(
            output,
            output_lse,
            other_output,
            other_lse,
        )

        self.assertEqual(output.data_ptr(), output_ptr)
        self.assertEqual(output_lse.data_ptr(), lse_ptr)
        self.assertEqual(output_lse.stride(), (1, tokens))
        torch.testing.assert_close(output, reference_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(output_lse, reference_lse, rtol=1e-5, atol=1e-5)

    def test_state_merge_rejects_internally_overlapping_output(self) -> None:
        tokens = 3
        heads = 4
        head_size = 8
        output_storage = torch.empty(
            (tokens, head_size), device="cuda", dtype=torch.float32
        )
        output = output_storage.as_strided(
            (tokens, heads, head_size),
            (head_size, 0, 1),
        )
        output_lse = torch.empty((tokens, heads), device="cuda", dtype=torch.float32)
        other_output = torch.empty_like(output.contiguous())
        other_lse = torch.empty_like(output_lse)

        self.assertFalse(
            is_merge_attention_states_in_place_supported(
                output, output_lse, other_output, other_lse
            )
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            merge_attention_states_in_place(output, output_lse, other_output, other_lse)

    def test_state_merge_rejects_equal_stride_output_overlap(self) -> None:
        output_storage = torch.empty(640, device="cuda", dtype=torch.float32)
        output = output_storage.as_strided(
            (2, 2, 128),
            (256, 256, 1),
        )
        output_lse = torch.empty((2, 2), device="cuda", dtype=torch.float32)
        other_output = torch.empty((2, 2, 128), device="cuda", dtype=torch.float32)
        other_lse = torch.empty_like(output_lse)

        self.assertEqual(output[1, 0].data_ptr(), output[0, 1].data_ptr())
        self.assertFalse(
            is_merge_attention_states_in_place_supported(
                output, output_lse, other_output, other_lse
            )
        )

    def test_state_merge_rejects_equal_stride_lse_overlap(self) -> None:
        output = torch.empty((2, 2, 128), device="cuda", dtype=torch.float32)
        output_lse_storage = torch.empty(5, device="cuda", dtype=torch.float32)
        output_lse = output_lse_storage.as_strided((2, 2), (2, 2))
        other_output = torch.empty_like(output)
        other_lse = torch.empty((2, 2), device="cuda", dtype=torch.float32)

        self.assertEqual(output_lse[1, 0].data_ptr(), output_lse[0, 1].data_ptr())
        self.assertFalse(
            is_merge_attention_states_in_place_supported(
                output, output_lse, other_output, other_lse
            )
        )

    def test_state_merge_rejects_internally_overlapping_lse(self) -> None:
        tokens = 3
        heads = 4
        head_size = 8
        output = torch.empty(
            (tokens, heads, head_size), device="cuda", dtype=torch.float32
        )
        lse_storage = torch.empty((tokens,), device="cuda", dtype=torch.float32)
        output_lse = lse_storage.as_strided((tokens, heads), (1, 0))
        other_output = torch.empty_like(output)
        other_lse = torch.empty((tokens, heads), device="cuda", dtype=torch.float32)

        self.assertFalse(
            is_merge_attention_states_in_place_supported(
                output, output_lse, other_output, other_lse
            )
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            merge_attention_states_in_place(output, output_lse, other_output, other_lse)

    def test_state_merge_rejects_accumulator_alias_with_other_state(self) -> None:
        tokens = 3
        heads = 4
        head_size = 8
        output = torch.empty(
            (tokens, heads, head_size), device="cuda", dtype=torch.float32
        )
        output_lse = torch.empty((tokens, heads), device="cuda", dtype=torch.float32)
        other_lse = torch.empty_like(output_lse)

        self.assertFalse(
            is_merge_attention_states_in_place_supported(
                output, output_lse, output, other_lse
            )
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            merge_attention_states_in_place(output, output_lse, output, other_lse)

    def test_state_merge_65_states_keeps_fp32_accumulator(self) -> None:
        tokens = 7
        heads = 96
        head_size = 128
        torch.manual_seed(33)
        output_bf16 = torch.randn(
            (tokens, heads, head_size), device="cuda", dtype=torch.bfloat16
        )
        output = output_bf16.float()
        output_lse = torch.randn(
            (heads, tokens), device="cuda", dtype=torch.float32
        ).transpose(0, 1)
        reference_output = output.clone()
        reference_lse = output_lse.clone()

        for _ in range(64):
            other_output = torch.randn_like(output_bf16)
            other_lse = torch.randn_like(output_lse)
            max_lse = torch.maximum(reference_lse, other_lse)
            output_exp = torch.exp(reference_lse - max_lse)
            other_exp = torch.exp(other_lse - max_lse)
            denominator = output_exp + other_exp
            reference_output = reference_output * (output_exp / denominator).unsqueeze(
                -1
            ) + other_output.float() * (other_exp / denominator).unsqueeze(-1)
            reference_lse = torch.log(denominator) + max_lse
            merge_attention_states_in_place(
                output,
                output_lse,
                other_output,
                other_lse,
            )

        actual_bf16 = output.to(torch.bfloat16)
        reference_bf16 = reference_output.to(torch.bfloat16)
        max_abs = float((actual_bf16.float() - reference_bf16.float()).abs().max())
        self.assertLessEqual(max_abs, 0.015625)
        torch.testing.assert_close(output_lse, reference_lse, rtol=1e-4, atol=1e-4)

    def test_state_merge_large_grid_orders_in_place_lse_update(self) -> None:
        """Cover the large multi-warp layout involved in the LSE RAW hazard."""

        tokens = 65536
        heads = 96
        head_size = 128
        torch.manual_seed(20260823)
        output = torch.randn(
            (tokens, heads, head_size), device="cuda", dtype=torch.float32
        ).mul_(0.125)
        other_output = torch.randn(
            (tokens, heads, head_size), device="cuda", dtype=torch.bfloat16
        ).mul_(0.125)
        # Match the scale separation seen between a short causal suffix and a
        # 16K historical-prefix state.  If a late warp observes the newly
        # stored LSE instead of the old one, its output scale is visibly wrong.
        output_lse = (
            torch.randn((heads, tokens), device="cuda", dtype=torch.float32)
            .mul_(0.05)
            .add_(6.5)
            .transpose(0, 1)
        )
        other_lse = (
            torch.randn((heads, tokens), device="cuda", dtype=torch.float32)
            .mul_(0.05)
            .add_(10.25)
            .transpose(0, 1)
        )
        reference_output = output.clone()
        reference_lse = output_lse.clone()
        for start in range(0, tokens, 4096):
            end = min(start + 4096, tokens)
            output_lse_chunk = reference_lse[start:end]
            other_lse_chunk = other_lse[start:end]
            max_lse = torch.maximum(output_lse_chunk, other_lse_chunk)
            output_exp = torch.exp(output_lse_chunk - max_lse)
            other_exp = torch.exp(other_lse_chunk - max_lse)
            denominator = output_exp + other_exp
            reference_output[start:end].mul_(
                (output_exp / denominator).unsqueeze(-1)
            ).add_(
                other_output[start:end].float()
                * (other_exp / denominator).unsqueeze(-1)
            )
            output_lse_chunk.copy_(torch.log(denominator) + max_lse)

        merge_attention_states_in_place(
            output,
            output_lse,
            other_output,
            other_lse,
        )

        for start in range(0, tokens, 4096):
            end = min(start + 4096, tokens)
            torch.testing.assert_close(
                output[start:end],
                reference_output[start:end],
                rtol=1e-4,
                atol=1e-5,
            )
            torch.testing.assert_close(
                output_lse[start:end],
                reference_lse[start:end],
                rtol=1e-5,
                atol=1e-5,
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
        checkpoint_weight.mul_(op.kv_lora_rank**-0.5)
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

    def test_project_kv_falls_back_when_packed_capability_is_disabled(
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

        linear = CudaF16Linear(checkpoint_weight)
        with mock.patch.object(
            LinearFactory, "create_linear_from_weights", return_value=linear
        ), mock.patch.object(
            linear,
            "supports_skip_head_mid",
            return_value=False,
        ) as supports_skip_head_mid:
            op.has_reuse_cache = True
            self.assertIsNone(
                op._project_reused_kv_with_gap_fill(compressed_kv, k_pe, None, linear)
            )
            actual_k, actual_v = op._project_kv(compressed_kv, k_pe, 0)
        self.assertGreaterEqual(supports_skip_head_mid.call_count, 2)

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

    def test_chunked_prefix_requires_packed_projection_capability(self) -> None:
        op = self._make_op(prefix_chunk_tokens=128)
        op.q_lens = [1]
        op.qo_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        op.prefix_chunks = (
            SimpleNamespace(
                spec=SimpleNamespace(
                    kv_tokens=128,
                    q_tokens=1,
                    q_start=0,
                    request_indices=(0,),
                    prefix_lens=(128,),
                ),
                qo_indptr=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
                kv_indptr=torch.tensor([0, 128], device="cuda", dtype=torch.int32),
                gather_qo_indptr=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
                batch_reuse_info=torch.tensor(
                    [[0, 128, 0, 1]], device="cuda", dtype=torch.int32
                ),
            ),
        )
        q = torch.empty((1, 12, 192), device="cuda", dtype=torch.bfloat16)
        compressed_kv = torch.empty((1, 512), device="cuda", dtype=torch.bfloat16)
        k_pe = torch.empty((1, 64), device="cuda", dtype=torch.bfloat16)
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.empty(
            (1, 128, 576), device="cuda", dtype=torch.bfloat16
        )
        unsupported_linear = mock.Mock()
        unsupported_linear.supports_skip_head_mid.return_value = False

        with mock.patch.object(
            op, "_create_kv_b_proj", return_value=unsupported_linear
        ), mock.patch.object(
            op,
            "_project_kv",
            return_value=(
                torch.empty((1, 12, 192), device="cuda", dtype=torch.bfloat16),
                torch.empty((1, 12, 128), device="cuda", dtype=torch.bfloat16),
            ),
        ), mock.patch.object(
            op,
            "_run_dense_attention",
            return_value=(
                torch.empty((1, 12, 128), device="cuda", dtype=torch.bfloat16),
                torch.empty((1, 12), device="cuda", dtype=torch.float32),
            ),
        ), mock.patch.object(
            op,
            "_current_reuse_cache_page_indices",
            return_value=torch.tensor([0], device="cuda", dtype=torch.int32),
        ), mock.patch.object(
            op,
            "_gather_prefix_chunk",
            side_effect=lambda *args, outputs=None, **kwargs: outputs,
        ), mock.patch.object(
            flashmla_dense_prefill_module,
            "merge_attention_states_in_place",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires packed.*projection"):
                op._forward_chunked_prefix(q, compressed_kv, k_pe, kv_cache, 0)


if __name__ == "__main__":
    unittest.main()
