"""Unit coverage for MSA page-RR prefix scratch reconstruction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.msa_attention import (
    MSAAttention,
    _scatter_cp_prefix_pages,
)

_MSA_MODULE = "rtp_llm.models_py.modules.hybrid.msa_attention"


class TestMSACpShardedPrefixRestore(unittest.TestCase):

    def test_direct_paged_live_shape_does_not_cross_multiply_old_cp_high_watermarks(
        self,
    ):
        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn.cp_enabled = True
        attn.page_size = 128
        attn._scratch_batch_size = 0
        attn._scratch_seq_len = 0
        attn._scratch_slots = 0

        # The fallback path intentionally retains independent high watermarks.
        attn._ensure_gather_scratch(
            SimpleNamespace(),
            torch.device("cpu"),
            torch.bfloat16,
            bsz=8,
            max_kv=80_000,
        )
        self.assertEqual(attn._scratch_batch_size, 8)
        self.assertEqual(attn._scratch_seq_len, 80_128)

        # Direct-paged prefill uses only the current request geometry. A later
        # long batch-1 request must not allocate 8 * 320K logical pages.
        attn._ensure_gather_scratch(
            SimpleNamespace(),
            torch.device("cpu"),
            torch.bfloat16,
            bsz=1,
            max_kv=320_000,
            exact_cp_shape=True,
        )
        self.assertEqual(attn._scratch_batch_size, 1)
        self.assertEqual(attn._scratch_seq_len, 320_000)
        self.assertEqual(attn._scratch_slots, 320_000)

    def test_direct_paged_prefill_gate_is_narrow_and_default_off(self):
        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn._kv_sharded = True
        attn.disable_index_value = True
        attn.num_idx_heads = 4
        attn.kv_head_num = 4
        attn._paged_kv_base_view = lambda _cache: torch.empty(
            1, 2, 4, 128, 128, dtype=torch.float8_e4m3fn
        )

        with patch.dict("os.environ", {"M3_SPARSE_ATTN_CHUNK_ENABLE": "1"}):
            with patch(f"{_MSA_MODULE}._USE_CP_DIRECT_PAGED_PREFILL", False):
                self.assertFalse(
                    attn._should_use_cp_direct_paged_prefill(SimpleNamespace())
                )
            with patch(f"{_MSA_MODULE}._USE_CP_DIRECT_PAGED_PREFILL", True):
                self.assertTrue(
                    attn._should_use_cp_direct_paged_prefill(SimpleNamespace())
                )
                with patch(f"{_MSA_MODULE}._USE_FUSED_CP_PAGED_WRITE", False):
                    self.assertFalse(
                        attn._should_use_cp_direct_paged_prefill(SimpleNamespace())
                    )
                attn._kv_sharded = False
                self.assertFalse(
                    attn._should_use_cp_direct_paged_prefill(SimpleNamespace())
                )
                attn._kv_sharded = True
                attn.disable_index_value = False
                self.assertFalse(
                    attn._should_use_cp_direct_paged_prefill(SimpleNamespace())
                )
                attn.disable_index_value = True
                attn.num_idx_heads = 5
                self.assertFalse(
                    attn._should_use_cp_direct_paged_prefill(SimpleNamespace())
                )
                attn.num_idx_heads = 4
                attn._paged_kv_base_view = lambda _cache: torch.empty(
                    1, 2, 4, 128, 128, dtype=torch.bfloat16
                )
                self.assertFalse(
                    attn._should_use_cp_direct_paged_prefill(SimpleNamespace())
                )

        attn._paged_kv_base_view = lambda _cache: torch.empty(
            1, 2, 4, 128, 128, dtype=torch.float8_e4m3fn
        )
        with patch.dict("os.environ", {"M3_SPARSE_ATTN_CHUNK_ENABLE": "0"}):
            with patch(f"{_MSA_MODULE}._USE_CP_DIRECT_PAGED_PREFILL", True):
                self.assertFalse(
                    attn._should_use_cp_direct_paged_prefill(SimpleNamespace())
                )

    @patch(f"{_MSA_MODULE}._fused_cp_paged_write")
    @patch(f"{_MSA_MODULE}._IDX_K_SCRATCH.acquire")
    def test_direct_paged_source_always_builds_bf16_working_pages(
        self, acquire_idx_scratch, fused_write
    ):
        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn.page_size = 2
        attn.kv_head_num = 1
        attn.head_dim = 1
        attn.idx_head_dim = 1
        attn._scratch_slots = 4
        attn._scratch_seq_len = 4
        attn._paged_kv_base_view = lambda _cache: torch.empty(
            2, 2, 1, 2, 1, dtype=torch.float8_e4m3fn
        )
        attn._idx_k_paged_view = lambda _cache: torch.empty(
            2, 2, 1, dtype=torch.bfloat16
        )
        acquire_idx_scratch.return_value = torch.zeros(4, 1, 1, dtype=torch.bfloat16)
        packed = torch.zeros(1, 3, dtype=torch.bfloat16)
        one = torch.zeros(1, dtype=torch.int64)

        k_paged, v_paged = attn._write_cp_suffix_to_bf16_working_pages(
            SimpleNamespace(),
            packed,
            one,
            one,
            one,
            one.to(torch.int32),
            1,
            1,
            1,
        )

        self.assertEqual(k_paged.dtype, torch.bfloat16)
        self.assertEqual(v_paged.dtype, torch.bfloat16)
        self.assertEqual(tuple(k_paged.shape), (2, 1, 2, 1))
        self.assertIsNone(attn._scratch_k)
        self.assertIsNone(attn._scratch_v)
        self.assertIs(attn._scratch_idx_k, acquire_idx_scratch.return_value)
        self.assertTrue(fused_write.call_args.kwargs["scratch_is_paged"])

        with self.assertRaisesRegex(RuntimeError, "requires BF16 packed"):
            attn._write_cp_suffix_to_bf16_working_pages(
                SimpleNamespace(),
                packed.float(),
                one,
                one,
                one,
                one.to(torch.int32),
                1,
                1,
                1,
            )

    @patch(f"{_MSA_MODULE}.gather_cp_sharded_prefix_pool")
    def test_restores_multi_request_main_kv_and_idx_in_logical_order(self, gather):
        # Three logical pages, two tokens per page. Request 0 owns pages 0/1;
        # request 1 owns page 2. Values encode the global logical token order.
        main_pages = torch.zeros(3, 2, 1, 2, 1, dtype=torch.bfloat16)
        main_pages[:, 0, 0, :, 0] = torch.tensor([[1, 2], [3, 4], [5, 6]])
        main_pages[:, 1, 0, :, 0] = torch.tensor([[11, 12], [13, 14], [15, 16]])
        idx_pages = torch.tensor(
            [[[21], [22]], [[23], [24]], [[25], [26]]], dtype=torch.bfloat16
        )
        gather.side_effect = [main_pages, idx_pages]

        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn._kv_sharded = True
        attn._cp_size = 2
        attn._cp_rank = 0
        attn.page_size = 2
        attn.kv_head_num = 1
        attn.head_dim = 1
        attn.idx_head_dim = 1
        attn._scratch_k = torch.zeros(12, 1, 1, dtype=torch.bfloat16)
        attn._scratch_v = torch.zeros_like(attn._scratch_k)
        attn._scratch_idx_k = torch.zeros_like(attn._scratch_k)
        attn._physical_block_table = lambda _inputs: torch.tensor([[3], [7]])
        attn._paged_kv_base_view = lambda _cache: torch.empty(8, 2, 1, 2, 1)
        attn._idx_k_paged_view = lambda _cache: torch.empty(8, 2, 1)
        req_to_token = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

        attn._restore_cp_sharded_prefix_scratch(
            SimpleNamespace(), [4, 2], req_to_token, SimpleNamespace()
        )

        self.assertEqual(gather.call_count, 2)
        self.assertTrue(
            torch.equal(attn._scratch_k[[0, 1, 2, 3, 6, 7], 0, 0], torch.arange(1, 7))
        )
        self.assertTrue(
            torch.equal(attn._scratch_v[[0, 1, 2, 3, 6, 7], 0, 0], torch.arange(11, 17))
        )
        self.assertTrue(
            torch.equal(
                attn._scratch_idx_k[[0, 1, 2, 3, 6, 7], 0, 0],
                torch.arange(21, 27),
            )
        )

    @patch(f"{_MSA_MODULE}.gather_cp_sharded_prefix_pool")
    def test_zero_prefix_does_not_collect(self, gather):
        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn._kv_sharded = True
        attn._restore_cp_sharded_prefix_scratch(
            SimpleNamespace(), [0], torch.zeros(1, 1), SimpleNamespace()
        )
        gather.assert_not_called()

    @patch(f"{_MSA_MODULE}.gather_cp_sharded_prefix_pool")
    def test_restores_prefix_pages_into_each_request_page_run(self, gather):
        # CPU index_copy does not implement float8; BF16 is sufficient here to
        # validate the request-local destination page namespace. The CUDA writer
        # test covers byte-exact E4M3 stores.
        main_pages = torch.zeros(3, 2, 1, 2, 1, dtype=torch.bfloat16)
        main_pages[:, 0, 0, :, 0] = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16
        )
        main_pages[:, 1, 0, :, 0] = torch.tensor(
            [[11, 12], [13, 14], [15, 16]], dtype=torch.bfloat16
        )
        idx_pages = torch.tensor(
            [[[21], [22]], [[23], [24]], [[25], [26]]], dtype=torch.bfloat16
        )
        gather.side_effect = [main_pages, idx_pages]

        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn._kv_sharded = True
        attn._cp_size = 2
        attn._cp_rank = 0
        attn.page_size = 2
        attn.idx_head_dim = 1
        attn._scratch_idx_k = torch.zeros(12, 1, 1, dtype=torch.bfloat16)
        attn._physical_block_table = lambda _inputs: torch.tensor([[3], [7]])
        attn._paged_kv_base_view = lambda _cache: torch.empty(
            8, 2, 1, 2, 1, dtype=torch.bfloat16
        )
        attn._idx_k_paged_view = lambda _cache: torch.empty(8, 2, 1)
        req_to_token = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        k_paged = torch.zeros(6, 1, 2, 1, dtype=torch.bfloat16)
        v_paged = torch.zeros_like(k_paged)

        attn._restore_cp_sharded_prefix_working_pages(
            SimpleNamespace(),
            [4, 2],
            req_to_token,
            SimpleNamespace(),
            k_paged,
            v_paged,
        )

        self.assertEqual(gather.call_count, 2)
        self.assertTrue(
            torch.equal(
                k_paged[[0, 1, 3], 0, :, 0].to(torch.bfloat16),
                torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16),
            )
        )
        self.assertTrue(
            torch.equal(
                v_paged[[0, 1, 3], 0, :, 0].to(torch.bfloat16),
                torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.bfloat16),
            )
        )
        self.assertEqual(torch.count_nonzero(k_paged[2]).item(), 0)
        self.assertTrue(
            torch.equal(
                attn._scratch_idx_k[[0, 1, 2, 3, 6, 7], 0, 0],
                torch.arange(21, 27),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for FP8 copy")
    @patch(f"{_MSA_MODULE}.gather_cp_sharded_prefix_pool")
    def test_restores_cuda_fp8_prefix_pages_as_raw_bytes(self, gather):
        device = torch.device("cuda")
        main_bf16 = torch.zeros(3, 2, 1, 2, 1, dtype=torch.bfloat16, device=device)
        main_bf16[:, 0, 0, :, 0] = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16, device=device
        )
        main_bf16[:, 1, 0, :, 0] = torch.tensor(
            [[11, 12], [13, 14], [15, 16]], dtype=torch.bfloat16, device=device
        )
        main_pages = main_bf16.to(torch.float8_e4m3fn)
        idx_pages = torch.tensor(
            [[[21], [22]], [[23], [24]], [[25], [26]]],
            dtype=torch.bfloat16,
            device=device,
        )
        gather.side_effect = [main_pages, idx_pages]

        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn._kv_sharded = True
        attn._cp_size = 2
        attn._cp_rank = 0
        attn.page_size = 2
        attn.idx_head_dim = 1
        attn._scratch_idx_k = torch.zeros(12, 1, 1, dtype=torch.bfloat16, device=device)
        attn._physical_block_table = lambda _inputs: torch.tensor(
            [[3], [7]], device=device
        )
        attn._paged_kv_base_view = lambda _cache: torch.empty(
            8, 2, 1, 2, 1, dtype=torch.float8_e4m3fn, device=device
        )
        attn._idx_k_paged_view = lambda _cache: torch.empty(
            8, 2, 1, dtype=torch.bfloat16, device=device
        )
        req_to_token = torch.tensor(
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], device=device
        )
        k_paged = torch.zeros(6, 1, 2, 1, dtype=torch.float8_e4m3fn, device=device)
        v_paged = torch.zeros_like(k_paged)

        attn._restore_cp_sharded_prefix_working_pages(
            SimpleNamespace(),
            [4, 2],
            req_to_token,
            SimpleNamespace(),
            k_paged,
            v_paged,
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(
                k_paged.view(torch.uint8)[[0, 1, 3]],
                main_pages[:, 0].contiguous().view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                v_paged.view(torch.uint8)[[0, 1, 3]],
                main_pages[:, 1].contiguous().view(torch.uint8),
            )
        )
        self.assertEqual(torch.count_nonzero(k_paged[2]).item(), 0)
        self.assertEqual(torch.count_nonzero(v_paged[2]).item(), 0)
        self.assertTrue(
            torch.equal(
                attn._scratch_idx_k[[0, 1, 2, 3, 6, 7], 0, 0],
                torch.arange(21, 27, dtype=torch.bfloat16, device=device),
            )
        )
        self.assertEqual(
            torch.count_nonzero(attn._scratch_idx_k[[4, 5, 8, 9, 10, 11]]).item(),
            0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for FP8 conversion")
    def test_scatter_converts_fp8_prefix_pages_to_bf16(self):
        device = torch.device("cuda")
        src = torch.arange(1, 25, dtype=torch.bfloat16, device=device).view(
            3, 2, 1, 2, 2
        )
        main_pages = src.to(torch.float8_e4m3fn)
        idx_pages = torch.arange(1, 7, dtype=torch.bfloat16, device=device).view(
            3, 2, 1
        )
        dst_pages = torch.tensor([0, 2, 4], dtype=torch.long, device=device)
        k_paged = torch.zeros(5, 1, 2, 2, dtype=torch.bfloat16, device=device)
        v_paged = torch.zeros_like(k_paged)
        idx_scratch = torch.zeros(5, 2, 1, dtype=torch.bfloat16, device=device)

        _scatter_cp_prefix_pages(
            main_pages,
            idx_pages,
            dst_pages,
            k_paged,
            v_paged,
            idx_scratch,
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(k_paged[dst_pages], main_pages[:, 0].to(torch.bfloat16))
        )
        self.assertTrue(
            torch.equal(v_paged[dst_pages], main_pages[:, 1].to(torch.bfloat16))
        )
        self.assertTrue(torch.equal(idx_scratch[dst_pages], idx_pages))
        self.assertEqual(torch.count_nonzero(k_paged[[1, 3]]).item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for FP8 conversion")
    def test_scatter_reads_rank_major_gather_with_cached_restore_indices(self):
        device = torch.device("cuda")
        logical = torch.arange(1, 25, dtype=torch.bfloat16, device=device).view(
            3, 2, 1, 2, 2
        )
        # Rank-major gather contains one padded row and a non-logical order.
        src_pages = torch.tensor([2, 0, 3], dtype=torch.long, device=device)
        main_rank_major = torch.zeros(
            4, 2, 1, 2, 2, dtype=torch.float8_e4m3fn, device=device
        )
        idx_rank_major = torch.zeros(4, 2, 1, dtype=torch.bfloat16, device=device)
        main_rank_major[src_pages] = logical.to(torch.float8_e4m3fn)
        logical_idx = torch.arange(1, 7, dtype=torch.bfloat16, device=device).view(
            3, 2, 1
        )
        idx_rank_major[src_pages] = logical_idx
        dst_pages = torch.tensor([0, 2, 4], dtype=torch.long, device=device)
        k_paged = torch.zeros(5, 1, 2, 2, dtype=torch.bfloat16, device=device)
        v_paged = torch.zeros_like(k_paged)
        idx_scratch = torch.zeros(5, 2, 1, dtype=torch.bfloat16, device=device)

        _scatter_cp_prefix_pages(
            main_rank_major,
            idx_rank_major,
            dst_pages,
            k_paged,
            v_paged,
            idx_scratch,
            src_pages=src_pages,
        )
        torch.cuda.synchronize()

        logical_fp8 = logical.to(torch.float8_e4m3fn)
        self.assertTrue(
            torch.equal(k_paged[dst_pages], logical_fp8[:, 0].to(torch.bfloat16))
        )
        self.assertTrue(
            torch.equal(v_paged[dst_pages], logical_fp8[:, 1].to(torch.bfloat16))
        )
        self.assertTrue(torch.equal(idx_scratch[dst_pages], logical_idx))


if __name__ == "__main__":
    unittest.main()
