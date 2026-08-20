import os
import sys
import tempfile
import unittest
from typing import Sequence
from unittest import mock

os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}_{os.getpid()}"),
)
os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

_LOCAL_DEEP_GEMM_PATH = os.environ.get("RTP_LOCAL_DEEP_GEMM_PATH")
if _LOCAL_DEEP_GEMM_PATH:
    sys.path.insert(0, _LOCAL_DEEP_GEMM_PATH)

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    FlashMLADeviceParams,
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
    def _make_op() -> MlaFlashMLAPrefillOp:
        op = object.__new__(MlaFlashMLAPrefillOp)
        op.num_heads = 12
        op.kv_lora_rank = 512
        op.qk_nope_head_dim = 128
        op.qk_rope_head_dim = 64
        op.v_head_dim = 128
        op.weights = [{}]
        op.quant_config = None
        op._direct_attn_inputs = None
        op._direct_block_table_width = 0
        op._chunk_prefill_kv_tile_tokens = 0
        op._chunk_prefill_kv_tiles = ()
        op._packed_kv_tile_workspace = None
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
    ) -> None:
        self.assertEqual(len(q_lens), len(reuse_lens))
        self.assertTrue(q_lens)
        self.assertTrue(all(length > 0 for length in q_lens))

        torch.manual_seed(sum(q_lens) + sum(reuse_lens) + 1234)
        op = self._make_op()
        op.page_size = 128
        self.assertTrue(all(length % op.page_size == 0 for length in reuse_lens))

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
        op.scale = (op.qk_nope_head_dim + op.qk_rope_head_dim) ** -0.5

        suffix_tokens = sum(q_lens)
        compressed_kv = torch.randn(
            (suffix_tokens, op.kv_lora_rank),
            device="cuda",
            dtype=torch.bfloat16,
        )
        k_pe = torch.randn(
            (suffix_tokens, op.qk_rope_head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )

        page_counts = [length // op.page_size for length in reuse_lens]
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
            batch_reuse_info.append(
                [batch_idx, reuse_len, block_start, page_count]
            )

            if page_count:
                prefix = cache[batch_page_ids].reshape(
                    -1, op.kv_lora_rank + op.qk_rope_head_dim
                )[:reuse_len]
            else:
                prefix = cache.new_empty(
                    (0, op.kv_lora_rank + op.qk_rope_head_dim)
                )
            suffix_start = qo_indptr[batch_idx]
            suffix_end = suffix_start + q_len
            expected_compressed_parts.append(
                torch.cat(
                    [prefix[:, : op.kv_lora_rank], compressed_kv[suffix_start:suffix_end]],
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
        self.assertIsNone(op._packed_kv_tile_workspace)

        self._assert_close_chunked(packed_k, reference_k)
        self._assert_close_chunked(packed_v, reference_v)
        self.assertEqual(
            packed_k.untyped_storage().data_ptr(),
            packed_v.untyped_storage().data_ptr(),
        )
        self.assertEqual(packed_k.stride(), (3840, 320, 1))
        self.assertEqual(packed_v.stride(), (3840, 320, 1))
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

    def test_synthetic_cache_reuse_scenarios(self) -> None:
        cases = (
            ("cache_miss", [257], [0], 24),
            ("full_hit", [1], [4096], 1),
            ("partial_hit", [257], [4096], 1),
            ("mixed_batch", [17, 33, 65, 9], [0, 4096, 8192, 4096], 1),
        )
        for name, q_lens, reuse_lens, repeats in cases:
            with self.subTest(name=name):
                self._run_synthetic_cache_pipeline(
                    q_lens,
                    reuse_lens,
                    projection_repeats=repeats,
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

    def test_long_prefix_is_partitioned_into_chunk_sized_kv_tiles(self) -> None:
        op = self._make_op()
        q_len = 32768
        prefix_len = 491520
        kv_len = q_len + prefix_len
        params = FlashMLADeviceParams(
            attn_inputs=None,
            q_lens_host=[q_len],
            kv_lens_host=[kv_len],
            prefix_lens_host=[prefix_len],
            qo_indptr_d=torch.tensor([0, q_len], device="cuda", dtype=torch.int32),
            kv_indptr_d=torch.tensor([0, kv_len], device="cuda", dtype=torch.int32),
            positions_d=torch.empty(0, device="cuda", dtype=torch.int32),
            batch_indice_d=torch.empty(0, device="cuda", dtype=torch.int32),
            reuse_cache_page_indice_d=torch.empty(
                0, device="cuda", dtype=torch.int32
            ),
            batch_reuse_info_vec_d=torch.empty(
                0, device="cuda", dtype=torch.int32
            ),
            block_table_width=0,
        )

        op.set_chunk_prefill_kv_tile_tokens(q_len)
        op.plan(params)

        self.assertEqual(op.total_kv_lens, kv_len)
        self.assertEqual(len(op._chunk_prefill_kv_tiles), 16)
        self.assertTrue(
            all(tile.kv_length <= q_len for tile in op._chunk_prefill_kv_tiles)
        )
        self.assertTrue(
            all(not tile.causal for tile in op._chunk_prefill_kv_tiles[:-1])
        )
        self.assertTrue(op._chunk_prefill_kv_tiles[-1].causal)
        self.assertEqual(op._chunk_prefill_kv_tiles[-1].kv_length, q_len)
        self.assertIsNone(op._packed_kv_tile_workspace)

        op.set_chunk_prefill_kv_tile_tokens(0)
        op.plan(params)
        self.assertEqual(op._chunk_prefill_kv_tiles, ())
        self.assertIsNone(op._packed_kv_tile_workspace)

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
        reference_kv = linear(compressed_kv).view(
            tokens, op.num_heads, expanded_dim
        )
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

        with (
            mock.patch.object(
                linear,
                "forward_skip_head_mid",
                wraps=linear.forward_skip_head_mid,
            ) as packed_projection,
            mock.patch.object(
                LinearFactory, "create_linear_from_weights", return_value=linear
            ),
        ):
            packed_k, packed_v = op._project_kv(compressed_kv, k_pe, 0)

        packed_projection.assert_called_once()
        self.assertNotIn("out", packed_projection.call_args.kwargs)
        self.assertIsNone(op._packed_kv_tile_workspace)

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
        op.qo_indptr = torch.tensor(
            [0, query_tokens], device="cuda", dtype=torch.int32
        )
        op.kv_indptr = torch.tensor(
            [0, tokens], device="cuda", dtype=torch.int32
        )
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

    def test_chunk_prefill_tiles_match_full_kv_attention(self) -> None:
        torch.manual_seed(654)
        op = self._make_op()
        op.num_heads = 96
        op.page_size = 128
        op.scale = (op.qk_nope_head_dim + op.qk_rope_head_dim) ** -0.5
        q_len = 128
        prefix_len = 512
        kv_len = prefix_len + q_len
        tile_tokens = 128

        import flash_mla.cuda as flash_mla_cuda

        op.flash_mla_cuda = flash_mla_cuda
        op.q_lens = [q_len]
        op.kv_lens = [kv_len]
        op.qo_indptr = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
        op.kv_indptr = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
        op.max_q_len = q_len
        op.max_kv_len = kv_len
        op.total_kv_lens = kv_len
        op.set_chunk_prefill_kv_tile_tokens(tile_tokens)
        op._chunk_prefill_kv_tiles = op._build_chunk_prefill_kv_tiles(
            [prefix_len],
            device=op.qo_indptr.device,
        )

        q = torch.randn(
            (q_len, op.num_heads, op.qk_nope_head_dim + op.qk_rope_head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        compressed_kv = torch.randn(
            (kv_len, op.kv_lora_rank), device="cuda", dtype=torch.bfloat16
        )
        k_pe = torch.randn(
            (kv_len, op.qk_rope_head_dim), device="cuda", dtype=torch.bfloat16
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
        ):
            full_k, full_v = op._project_kv(compressed_kv, k_pe, 0)
            reference = op._dense_attention(q, full_k, full_v)
            tiled_reference = None
            tiled_reference_lse = None
            for tile in op._chunk_prefill_kv_tiles:
                tile_k = full_k.narrow(0, tile.kv_start, tile.kv_length)
                tile_v = full_v.narrow(0, tile.kv_start, tile.kv_length)
                tile_out, tile_lse = op._dense_attention_tile(
                    q,
                    tile_k,
                    tile_v,
                    tile,
                )
                if tiled_reference is None:
                    tiled_reference = tile_out
                    tiled_reference_lse = tile_lse
                else:
                    tiled_reference, tiled_reference_lse = (
                        op._merge_attention_state_in_place(
                            tiled_reference,
                            tiled_reference_lse,
                            tile_out,
                            tile_lse,
                        )
                    )
            actual = op._forward_chunk_prefill_kv_tiles(
                q,
                compressed_kv,
                k_pe,
                0,
            )

        self.assertIsNotNone(tiled_reference)
        torch.testing.assert_close(
            tiled_reference, reference, rtol=3e-2, atol=5e-1
        )
        self.assertEqual(
            op._packed_kv_tile_workspace.shape,
            (tile_tokens, op.num_heads * sum((128, 64, 128))),
        )
        torch.testing.assert_close(actual, tiled_reference, rtol=3e-2, atol=5e-1)

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
        with mock.patch.object(
            LinearFactory, "create_linear_from_weights", return_value=linear
        ):
            actual_k, actual_v = op._project_kv(compressed_kv, k_pe, 0)

        expanded_dim = op.qk_nope_head_dim + op.v_head_dim
        reference_kv = linear(compressed_kv).view(
            tokens, op.num_heads, expanded_dim
        )
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
