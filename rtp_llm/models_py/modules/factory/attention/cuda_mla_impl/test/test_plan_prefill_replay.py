"""
Unit test: plan_prefill_cuda_graph_replay (device-only, 0 syncs) produces
identical _fp8_prefill_indices as plan_prefill_cuda_graph (which reads host data).
"""

from unittest import TestCase, main, skipIf

import torch


def _has_cuda():
    return torch.cuda.is_available()


@skipIf(not _has_cuda(), "requires CUDA")
class TestPlanPrefillCudaGraphReplay(TestCase):
    """Verify plan_prefill_cuda_graph_replay matches plan_prefill_cuda_graph."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0")
        self.batch_size = 4
        self.token_per_block = 64
        self.max_blocks = 16
        self.max_bs = 8
        self.num_heads = 16
        self.max_context_len = self.max_blocks * self.token_per_block
        self.num_mtp_heads = 3  # q_len per batch in MTP

    def _make_op(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
            MlaFlashInferDecodeOp,
        )
        from rtp_llm.ops import KvCacheDataType

        op = MlaFlashInferDecodeOp.__new__(MlaFlashInferDecodeOp)
        op.num_heads = self.num_heads
        op.kv_lora_rank = 512
        op.qk_rope_head_dim = 64
        op.qk_nope_head_dim = 128
        op.token_per_block = self.token_per_block
        op.softmax_extra_scale = 1.0
        op.use_mla = True
        op.is_sparse = False
        op.use_cuda_graph = True
        op._fp8_kv = True
        op._fmha_params = None
        op._sched_meta = None
        op._fp8_prefill_sched_meta = None

        padded_topk_max = self.max_context_len
        op._fp8_max_bs = self.max_bs
        op._fp8_max_context_len = self.max_context_len
        op._fp8_indices_buf = torch.full(
            (self.max_bs, 1, padded_topk_max),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        op._fp8_topk_len_buf = torch.zeros(
            self.max_bs, dtype=torch.int32, device=self.device
        )
        op._fp8_position_buf = torch.arange(
            padded_topk_max, dtype=torch.int32, device=self.device
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
        return op

    def _make_fmha_params(self, kvlen_values):
        """Create mock fmha_params with host+device tensors."""
        B = len(kvlen_values)
        q_len = self.num_mtp_heads

        # qo_indptr: [0, q_len, 2*q_len, ...]
        qo_indptr = torch.arange(B + 1, dtype=torch.int32) * q_len
        kvlen = torch.tensor(kvlen_values, dtype=torch.int32)

        # Block table: pages per batch
        pages_per_batch = [
            (v + self.token_per_block - 1) // self.token_per_block for v in kvlen_values
        ]
        max_pages = max(pages_per_batch) if pages_per_batch else 0

        # Build decode_page_indptr (uniform stride for cuda graph)
        decode_page_indptr = torch.arange(B + 1, dtype=torch.int32) * max_pages

        # Build page_indice_d (flat, B * max_pages)
        page_indice = torch.randint(
            0, 1000, (B * max_pages,), dtype=torch.int32, device=self.device
        )

        class Params:
            pass

        p = Params()
        p.qo_indptr_h = qo_indptr
        p.qo_indptr_d = qo_indptr.to(self.device)
        p.kvlen_h = kvlen
        p.kvlen_d = kvlen.to(self.device)
        p.decode_page_indptr_h = decode_page_indptr
        p.decode_page_indptr_d = decode_page_indptr.to(self.device)
        p.page_indice_d = page_indice
        p.page_indice_h = page_indice.cpu()
        return p

    def _init_prefill_indices(self, op, B, max_q_len, max_kv_len):
        """Initialize _fp8_prefill_indices buffer (done during capture)."""
        op._fp8_prefill_indices = torch.full(
            (B, max_q_len, max_kv_len),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        op._fp8_prefill_topk_length = torch.zeros(
            B, dtype=torch.int32, device=self.device
        )

    def test_replay_matches_capture(self):
        """plan_prefill_cuda_graph_replay produces same indices as plan_prefill_cuda_graph."""
        kvlens = [200, 300, 400, 500]
        B = len(kvlens)
        max_kv_len = max(kvlens)
        max_q_len = self.num_mtp_heads
        fmha_params = self._make_fmha_params(kvlens)

        # Path A: capture (reads host data)
        op_a = self._make_op()
        self._init_prefill_indices(op_a, B, max_q_len, max_kv_len)
        op_a.plan_prefill_cuda_graph(fmha_params)
        torch.cuda.synchronize()
        a_indices = op_a._fp8_prefill_indices.clone()
        a_topk_len = op_a._fp8_prefill_topk_length.clone()

        # Verify cache was set
        self.assertIsNotNone(op_a._cached_q_lens_d)
        self.assertEqual(op_a._cached_prefill_B, B)
        self.assertEqual(op_a._cached_max_q_len, max_q_len)

        # Path B: replay (device-only)
        op_b = self._make_op()
        self._init_prefill_indices(op_b, B, max_q_len, max_kv_len)
        # First call plan_prefill_cuda_graph to populate cache
        op_b.plan_prefill_cuda_graph(fmha_params)
        # Clear indices, then call replay
        op_b._fp8_prefill_indices.fill_(-1)
        op_b._fp8_prefill_topk_length.zero_()
        op_b.plan_prefill_cuda_graph_replay(fmha_params)
        torch.cuda.synchronize()
        b_indices = op_b._fp8_prefill_indices.clone()
        b_topk_len = op_b._fp8_prefill_topk_length.clone()

        torch.testing.assert_close(
            a_indices, b_indices, msg="replay indices must match capture indices"
        )
        torch.testing.assert_close(
            a_topk_len, b_topk_len, msg="replay topk_len must match capture topk_len"
        )

    def test_replay_with_growing_kvlen(self):
        """Simulate CUDA graph replay with kvlen growing each step."""
        B = self.batch_size
        max_q_len = self.num_mtp_heads
        max_kv_len = self.max_context_len

        # Initial kvlens (capture)
        initial_kvlens = [100, 150, 200, 250]
        fmha_params = self._make_fmha_params(initial_kvlens)

        op = self._make_op()
        self._init_prefill_indices(op, B, max_q_len, max_kv_len)
        op.plan_prefill_cuda_graph(fmha_params)

        # Simulate growing kvlen (replay steps)
        for step in range(3):
            new_kvlens = [v + step + 1 for v in initial_kvlens]
            fmha_params_new = self._make_fmha_params(new_kvlens)

            # Compute reference with capture path
            op_ref = self._make_op()
            self._init_prefill_indices(op_ref, B, max_q_len, max_kv_len)
            op_ref.plan_prefill_cuda_graph(fmha_params_new)
            torch.cuda.synchronize()
            ref_indices = op_ref._fp8_prefill_indices.clone()

            # Compute with replay path
            op._fp8_prefill_indices.fill_(-1)
            op._fp8_prefill_topk_length.zero_()
            op.plan_prefill_cuda_graph_replay(fmha_params_new)
            torch.cuda.synchronize()
            replay_indices = op._fp8_prefill_indices.clone()

            torch.testing.assert_close(
                ref_indices,
                replay_indices,
                msg=f"step {step}: replay must match capture with kvlens={new_kvlens}",
            )

    def test_replay_not_fp8_is_noop(self):
        """Non-FP8 op: replay returns True without error."""
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
            MlaFlashInferDecodeOp,
        )

        op = MlaFlashInferDecodeOp.__new__(MlaFlashInferDecodeOp)
        op._fp8_kv = False
        op._fp8_prefill_indices = None

        class FakeParams:
            pass

        result = op.plan_prefill_cuda_graph_replay(FakeParams())
        self.assertTrue(result)


if __name__ == "__main__":
    main()
