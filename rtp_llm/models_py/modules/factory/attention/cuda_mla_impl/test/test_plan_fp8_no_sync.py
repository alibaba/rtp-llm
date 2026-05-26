"""
Unit test: _plan_fp8_from_device produces identical outputs whether the inputs
need device/dtype conversion or are already in the target format.

The fast path (skip .to() when input matches device+dtype) avoids per-layer
cudaStreamSynchronize but must produce bitwise-identical _fp8_indices_buf
and _fp8_topk_len_buf as the slow path.
"""

from unittest import TestCase, main, skipIf

import torch


def _has_cuda():
    return torch.cuda.is_available()


@skipIf(not _has_cuda(), "requires CUDA")
class TestPlanFp8NoSync(TestCase):
    """Bypass the heavyweight MlaFlashInferDecodeOp ctor by building a stub
    that has just the buffers _plan_fp8_from_device touches."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0")
        self.batch_size = 4
        self.token_per_block = 64
        self.max_blocks = 16
        self.max_bs = 8
        self.padded_topk_max = self.max_blocks * self.token_per_block

        # Inputs: realistic device int32 tensors (the no-sync path)
        self.seqlens_d_i32 = torch.tensor(
            [400, 500, 600, 700], dtype=torch.int32, device=self.device
        )
        self.block_table_d_i32 = torch.randint(
            0,
            1000,
            (self.batch_size, self.max_blocks),
            dtype=torch.int32,
            device=self.device,
        )

    def _make_stub(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
            MlaFlashInferDecodeOp,
        )

        # Skip __init__; manually set just the fields _plan_fp8_from_device reads
        op = MlaFlashInferDecodeOp.__new__(MlaFlashInferDecodeOp)
        op._fp8_indices_buf = torch.full(
            (self.max_bs, 1, self.padded_topk_max),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        op._fp8_topk_len_buf = torch.zeros(
            self.max_bs, dtype=torch.int32, device=self.device
        )
        op._fp8_position_buf = torch.arange(
            self.padded_topk_max, dtype=torch.int32, device=self.device
        )
        op._fp8_block_ids_buf = (op._fp8_position_buf // self.token_per_block).to(
            torch.long
        )
        op._fp8_offsets_buf = op._fp8_position_buf % self.token_per_block
        op._fp8_plan_B = 0
        op._fp8_prefill_indices = None
        op._fp8_prefill_topk_length = None
        op._fp8_prefill_qo_indptr_h = None
        op._fp8_prefill_q_lens_h = None
        op._fp8_prefill_max_q_len = 0
        op._fp8_prefill_total_q = 0
        op._fp8_prefill_sched_meta = None

        class _SchedMeta:
            tile_scheduler_metadata = None
            num_splits = None

        op._sched_meta = _SchedMeta()
        op.token_per_block = self.token_per_block
        return op

    def test_no_sync_path_matches_conversion_path(self):
        """Run with already-correct inputs (no .to() needed) AND with
        host int64 inputs (needs .to() conversion). Outputs must match.
        """
        # Path A: already on device int32 (the no-sync hot path)
        op_a = self._make_stub()
        op_a._plan_fp8_from_device(self.seqlens_d_i32, self.block_table_d_i32)
        torch.cuda.synchronize()
        a_indices = op_a._fp8_indices_buf.clone()
        a_topk_len = op_a._fp8_topk_len_buf.clone()
        a_plan_B = op_a._fp8_plan_B

        # Path B: host int64 input (forces .to() conversion path)
        op_b = self._make_stub()
        seqlens_h_i64 = self.seqlens_d_i32.cpu().to(torch.int64)
        block_table_h_i64 = self.block_table_d_i32.cpu().to(torch.int64)
        op_b._plan_fp8_from_device(seqlens_h_i64, block_table_h_i64)
        torch.cuda.synchronize()
        b_indices = op_b._fp8_indices_buf.clone()
        b_topk_len = op_b._fp8_topk_len_buf.clone()
        b_plan_B = op_b._fp8_plan_B

        self.assertEqual(a_plan_B, b_plan_B)
        torch.testing.assert_close(a_indices, b_indices, msg="indices buf mismatch")
        torch.testing.assert_close(a_topk_len, b_topk_len, msg="topk_len buf mismatch")

    def test_zero_batch_no_op(self):
        """B=0 (empty seqlens) should leave buffers unchanged."""
        op = self._make_stub()
        empty_seq = torch.tensor([], dtype=torch.int32, device=self.device)
        before = op._fp8_indices_buf.clone()
        op._plan_fp8_from_device(empty_seq, self.block_table_d_i32)
        torch.cuda.synchronize()
        torch.testing.assert_close(op._fp8_indices_buf, before)
        self.assertEqual(op._fp8_plan_B, 0)


@skipIf(not _has_cuda(), "requires CUDA")
class TestPlanFp8HostRead(TestCase):
    """Verify that plan() reads seqlens from kvlen_h (host) and produces
    identical _fp8_indices_buf as reading from the GPU copy."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0")
        self.batch_size = 4
        self.token_per_block = 64
        self.max_blocks = 16
        self.max_bs = 8
        self.padded_topk_max = self.max_blocks * self.token_per_block

    def _make_stub(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
            MlaFlashInferDecodeOp,
        )

        op = MlaFlashInferDecodeOp.__new__(MlaFlashInferDecodeOp)
        op._fp8_kv = True
        op._fp8_indices_buf = torch.full(
            (self.max_bs, 1, self.padded_topk_max),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        op._fp8_topk_len_buf = torch.zeros(
            self.max_bs, dtype=torch.int32, device=self.device
        )
        op._fp8_position_buf = torch.arange(
            self.padded_topk_max, dtype=torch.int32, device=self.device
        )
        op._fp8_block_ids_buf = (op._fp8_position_buf // self.token_per_block).to(
            torch.long
        )
        op._fp8_offsets_buf = op._fp8_position_buf % self.token_per_block
        op._fp8_plan_B = 0
        op._fp8_prefill_indices = None
        op._fp8_prefill_topk_length = None
        op._fp8_prefill_qo_indptr_h = None
        op._fp8_prefill_q_lens_h = None
        op._fp8_prefill_max_q_len = 0
        op._fp8_prefill_total_q = 0
        op._fp8_prefill_sched_meta = None

        class _SchedMeta:
            tile_scheduler_metadata = None
            num_splits = None

        op._sched_meta = _SchedMeta()
        op.token_per_block = self.token_per_block
        op.use_cuda_graph = False
        return op

    def _make_fmha_params(self, kvlen_values):
        """Create a minimal fmha_params mock with decode_page_indptr_h,
        page_indice_d, kvlen_h, and qo_indptr_h."""
        B = len(kvlen_values)

        # page_indice: flatten block table
        block_table = torch.randint(
            0, 1000, (B, self.max_blocks), dtype=torch.int32, device=self.device
        )
        # indptr: each batch uses ceil(kvlen/token_per_block) pages
        pages_per_batch = [
            (v + self.token_per_block - 1) // self.token_per_block for v in kvlen_values
        ]
        indptr = [0]
        for n in pages_per_batch:
            indptr.append(indptr[-1] + n)

        total_pages = indptr[-1]
        page_indice_d = torch.zeros(total_pages, dtype=torch.int32, device=self.device)
        for i in range(B):
            start = indptr[i]
            n = pages_per_batch[i]
            page_indice_d[start : start + n].copy_(block_table[i, :n])

        class Params:
            pass

        p = Params()
        p.decode_page_indptr_h = torch.tensor(indptr, dtype=torch.int32)
        p.page_indice_d = page_indice_d
        p.kvlen_h = torch.tensor(kvlen_values, dtype=torch.int32)
        p.qo_indptr_h = torch.arange(B + 1, dtype=torch.int32)
        return p

    def test_plan_reads_from_host(self):
        """plan() with host kvlen_h produces correct indices."""
        kvlens = [200, 300, 128, 64]
        fmha_params = self._make_fmha_params(kvlens)
        op = self._make_stub()
        op.plan(fmha_params)
        torch.cuda.synchronize()

        # Verify indices are populated correctly
        B = len(kvlens)
        for i in range(B):
            seqlen = kvlens[i]
            indices_row = op._fp8_indices_buf[i, 0, :seqlen]
            # All valid entries should be >= 0
            self.assertTrue(
                (indices_row >= 0).all().item(),
                f"batch {i}: expected all indices >= 0 for seqlen={seqlen}",
            )
            # Entries beyond seqlen should be -1
            if seqlen < self.padded_topk_max:
                beyond = op._fp8_indices_buf[i, 0, seqlen:]
                self.assertTrue(
                    (beyond == -1).all().item(), f"batch {i}: expected -1 beyond seqlen"
                )

        # Verify topk_len_buf matches kvlen
        expected_topk = torch.tensor(kvlens, dtype=torch.int32, device=self.device)
        torch.testing.assert_close(
            op._fp8_topk_len_buf[:B],
            expected_topk,
            msg="topk_len_buf should match kvlen values",
        )

    def test_plan_vs_direct_device_read(self):
        """Verify plan() reading from kvlen_h matches manually computing
        the same result by reading seqlens from the GPU tensor."""
        kvlens = [150, 250, 400, 100]
        fmha_params = self._make_fmha_params(kvlens)

        # Run plan() (reads from kvlen_h on host)
        op = self._make_stub()
        op.plan(fmha_params)
        torch.cuda.synchronize()
        result_indices = op._fp8_indices_buf.clone()
        result_topk = op._fp8_topk_len_buf.clone()

        # Compute expected result using _plan_fp8_from_device
        # (which uses vectorized device operations)
        op2 = self._make_stub()
        seqlens_d = torch.tensor(kvlens, dtype=torch.int32, device=self.device)
        B = len(kvlens)
        pages_per_batch = [
            (v + self.token_per_block - 1) // self.token_per_block for v in kvlens
        ]
        max_pages = max(pages_per_batch)
        block_table_2d = torch.full(
            (B, max_pages), -1, dtype=torch.int32, device=self.device
        )
        indptr = fmha_params.decode_page_indptr_h
        for i in range(B):
            start = int(indptr[i].item())
            n = pages_per_batch[i]
            block_table_2d[i, :n].copy_(fmha_params.page_indice_d[start : start + n])

        op2._plan_fp8_from_device(seqlens_d, block_table_2d)
        torch.cuda.synchronize()
        expected_indices = op2._fp8_indices_buf.clone()
        expected_topk = op2._fp8_topk_len_buf.clone()

        torch.testing.assert_close(
            result_indices[:B],
            expected_indices[:B],
            msg="plan() indices should match _plan_fp8_from_device",
        )
        torch.testing.assert_close(
            result_topk[:B],
            expected_topk[:B],
            msg="plan() topk_len should match _plan_fp8_from_device",
        )


if __name__ == "__main__":
    main()
