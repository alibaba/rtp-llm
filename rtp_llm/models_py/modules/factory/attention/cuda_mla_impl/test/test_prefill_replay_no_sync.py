"""
Unit test: verify plan_prefill_cuda_graph_replay produces correct indices when
caches are NOT pre-populated (first call after capture), and that the entire
path is device-only (0 cudaStreamSynchronize events).
"""

from unittest import TestCase, main, skipIf

import torch


def _has_cuda():
    return torch.cuda.is_available()


@skipIf(not _has_cuda(), "requires CUDA")
class TestPrefillReplayNoSync(TestCase):
    """Verify plan_prefill_cuda_graph_replay self-initializes caches from device
    data and runs without any D2H sync."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0")
        self.batch_size = 4
        self.token_per_block = 64
        self.max_blocks = 16
        self.max_bs = 8
        self.num_heads = 16
        self.max_context_len = self.max_blocks * self.token_per_block
        self.num_mtp_heads = 3

    def _make_op(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
            MlaFlashInferDecodeOp,
        )

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
        # Ensure caches are NOT set (simulating first replay after capture)
        op._cached_prefill_B = None
        op._cached_max_q_len = None
        op._cached_q_lens_d = None
        return op

    def _make_fmha_params(self, kvlen_values):
        """Create mock fmha_params with device tensors (simulating replay state)."""
        B = len(kvlen_values)
        q_len = self.num_mtp_heads
        total_tokens = B * q_len

        qo_indptr = torch.arange(B + 1, dtype=torch.int32) * q_len
        kvlen = torch.tensor(kvlen_values, dtype=torch.int32)

        pages_per_batch = [
            (v + self.token_per_block - 1) // self.token_per_block for v in kvlen_values
        ]
        max_pages = max(pages_per_batch) if pages_per_batch else 0

        decode_page_indptr = torch.arange(B + 1, dtype=torch.int32) * max_pages
        page_indice = torch.randint(
            0, 1000, (B * max_pages,), dtype=torch.int32, device=self.device
        )

        # batch_indice_h: shape {total_tokens} — signals B and total_tokens
        batch_indice_h = torch.zeros(total_tokens, dtype=torch.int32)

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
        p.batch_indice_h = batch_indice_h
        p.batch_indice_d = batch_indice_h.to(self.device)
        return p

    def _init_prefill_indices(self, op, B, max_q_len, max_kv_len):
        op._fp8_prefill_indices = torch.full(
            (B, max_q_len, max_kv_len),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        op._fp8_prefill_topk_length = torch.zeros(
            B, dtype=torch.int32, device=self.device
        )

    def test_first_replay_self_initializes(self):
        """First call with None caches should self-initialize and produce
        correct indices (matching plan_prefill_cuda_graph reference)."""
        kvlens = [200, 300, 400, 500]
        B = len(kvlens)
        max_kv_len = max(kvlens)
        max_q_len = self.num_mtp_heads
        fmha_params = self._make_fmha_params(kvlens)

        # Path A: reference via plan_prefill_cuda_graph (reads host)
        op_a = self._make_op()
        self._init_prefill_indices(op_a, B, max_q_len, max_kv_len)
        op_a.plan_prefill_cuda_graph(fmha_params)
        torch.cuda.synchronize()
        a_indices = op_a._fp8_prefill_indices.clone()

        # Path B: replay with NO pre-populated caches
        op_b = self._make_op()
        self._init_prefill_indices(op_b, B, max_q_len, max_kv_len)
        assert op_b._cached_q_lens_d is None
        op_b.plan_prefill_cuda_graph_replay(fmha_params)
        torch.cuda.synchronize()
        b_indices = op_b._fp8_prefill_indices.clone()

        # Caches should now be populated
        self.assertIsNotNone(op_b._cached_q_lens_d)
        self.assertEqual(op_b._cached_prefill_B, B)
        self.assertEqual(op_b._cached_max_q_len, max_q_len)

        torch.testing.assert_close(
            a_indices,
            b_indices,
            msg="First replay (self-init) must match plan_prefill_cuda_graph reference",
        )

    def test_replay_no_cuda_stream_synchronize(self):
        """Verify plan_prefill_cuda_graph_replay triggers 0 cudaStreamSynchronize."""
        kvlens = [200, 300, 400, 500]
        B = len(kvlens)
        max_kv_len = max(kvlens)
        max_q_len = self.num_mtp_heads
        fmha_params = self._make_fmha_params(kvlens)

        op = self._make_op()
        self._init_prefill_indices(op, B, max_q_len, max_kv_len)

        # Warm up (populate caches)
        op.plan_prefill_cuda_graph_replay(fmha_params)
        torch.cuda.synchronize()

        # Profile the SECOND call (steady-state replay)
        op._fp8_prefill_indices.fill_(-1)
        op._fp8_prefill_topk_length.zero_()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
        ) as prof:
            op.plan_prefill_cuda_graph_replay(fmha_params)

        events = prof.key_averages()
        sync_events = [e for e in events if "cudaStreamSynchronize" in e.key]
        sync_count = sum(e.count for e in sync_events)
        self.assertEqual(
            sync_count,
            0,
            f"Expected 0 cudaStreamSynchronize, got {sync_count}. "
            f"Events: {[(e.key, e.count) for e in sync_events]}",
        )

    def test_first_replay_no_cuda_stream_synchronize(self):
        """Even the FIRST replay (cache self-init) should have 0 syncs."""
        kvlens = [200, 300, 400, 500]
        B = len(kvlens)
        max_kv_len = max(kvlens)
        max_q_len = self.num_mtp_heads
        fmha_params = self._make_fmha_params(kvlens)

        op = self._make_op()
        self._init_prefill_indices(op, B, max_q_len, max_kv_len)
        assert op._cached_q_lens_d is None

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
        ) as prof:
            op.plan_prefill_cuda_graph_replay(fmha_params)

        events = prof.key_averages()
        sync_events = [e for e in events if "cudaStreamSynchronize" in e.key]
        sync_count = sum(e.count for e in sync_events)
        self.assertEqual(
            sync_count,
            0,
            f"First replay should also have 0 syncs, got {sync_count}. "
            f"Events: {[(e.key, e.count) for e in sync_events]}",
        )


if __name__ == "__main__":
    main()
