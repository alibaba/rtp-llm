"""Test PyFlashinferPrefillPagedAttnOp CUDA graph path vs normal path.

Verifies that forward() with prefill_cuda_graph_copy_params produces
identical results to forward() without copy_params.
"""

import logging
import math
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillPagedAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    compare_tensors,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyPrefillCudaGaphCopyParams,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

PAGE_SIZE = 16


class TestPrefillPagedCudaGraph(BaseAttentionTest):
    """Compare forward() output: CUDA graph copy path vs normal path."""

    def _make_inputs(
        self,
        input_lengths,
        prefix_lengths,
        with_copy_params=False,
        max_seq_len=0,
        cg_max_batch_size=0,
    ):
        """Create PyAttentionInputs for prefill (single or multi batch).

        Args:
            cg_max_batch_size: If > 0 and with_copy_params=True, pad to this
                batch size to simulate non-full CUDA graph capture buckets
                (active_bs < max_bs).
        """
        if isinstance(input_lengths, int):
            input_lengths = [input_lengths]
            prefix_lengths = [prefix_lengths]

        active_batch_size = len(input_lengths)

        # Pad for non-full bucket simulation
        if (
            cg_max_batch_size > 0
            and with_copy_params
            and cg_max_batch_size > active_batch_size
        ):
            pad_count = cg_max_batch_size - active_batch_size
            input_lengths = list(input_lengths) + [0] * pad_count
            prefix_lengths = list(prefix_lengths) + [0] * pad_count

        batch_size = len(input_lengths)
        inp = PyAttentionInputs()
        inp.is_cuda_graph = with_copy_params
        inp.is_prefill = True
        inp.input_lengths = torch.tensor(input_lengths, dtype=torch.int32).pin_memory()
        inp.prefix_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32
        ).pin_memory()
        seq_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        inp.sequence_lengths = torch.tensor(seq_lengths, dtype=torch.int32).pin_memory()

        cu = [0]
        for il in input_lengths:
            cu.append(cu[-1] + il)

        if with_copy_params:
            inp.cu_seqlens_device = torch.tensor(cu, dtype=torch.int32).pin_memory()
            inp.cu_kv_seqlens_device = torch.tensor(cu, dtype=torch.int32).pin_memory()
        else:
            inp.cu_seqlens_device = torch.tensor(cu, dtype=torch.int32, device="cuda")
            inp.cu_kv_seqlens_device = torch.tensor(
                cu, dtype=torch.int32, device="cuda"
            )

        active_seq_lengths = [s for s in seq_lengths if s > 0]
        max_blocks = (
            max(math.ceil(s / PAGE_SIZE) for s in active_seq_lengths)
            if active_seq_lengths
            else 1
        )
        block_ids = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
        offset = 0
        for i, s in enumerate(seq_lengths):
            if s > 0:
                nb = math.ceil(s / PAGE_SIZE)
                block_ids[i, :nb] = torch.arange(offset, offset + nb)
                offset += nb
        inp.kv_cache_kernel_block_id = block_ids

        if with_copy_params:
            ms = (
                max_seq_len
                if max_seq_len > 0
                else max(il for il in input_lengths if il > 0)
            )
            cp = PyPrefillCudaGaphCopyParams()
            cp.cuda_graph_prefill_batch_size = torch.tensor(
                [active_batch_size], dtype=torch.int32
            ).pin_memory()
            cp.max_seq_len = ms
            cp.max_batch_size = batch_size
            inp.prefill_cuda_graph_copy_params = cp

        return inp

    def _make_paged_kv_cache(self, k, v, seq_lengths, num_kv_heads, head_dim):
        if isinstance(seq_lengths, int):
            seq_lengths = [seq_lengths]
        total_pages = sum(math.ceil(s / PAGE_SIZE) for s in seq_lengths)
        cache = torch.zeros(
            total_pages,
            2,
            num_kv_heads,
            PAGE_SIZE,
            head_dim,
            dtype=k.dtype,
            device=self.device,
        )
        page_idx, token_offset = 0, 0
        for seq_len in seq_lengths:
            for i in range(math.ceil(seq_len / PAGE_SIZE)):
                s, e = i * PAGE_SIZE, min((i + 1) * PAGE_SIZE, seq_len)
                n = e - s
                cache[page_idx, 0, :, :n, :] = k[
                    token_offset + s : token_offset + e
                ].transpose(0, 1)
                cache[page_idx, 1, :, :n, :] = v[
                    token_offset + s : token_offset + e
                ].transpose(0, 1)
                page_idx += 1
            token_offset += seq_len
        kv = LayerKVCache()
        kv.kv_cache_base = cache
        return kv

    def _test_forward_match(
        self,
        input_lengths,
        prefix_lengths,
        max_seq_len=0,
        head_num=8,
        head_num_kv=2,
        size_per_head=64,
    ):
        if isinstance(input_lengths, int):
            input_lengths = [input_lengths]
            prefix_lengths = [prefix_lengths]
        if max_seq_len == 0:
            max_seq_len = max(input_lengths)

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=PAGE_SIZE,
        )
        seq_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        total_q = sum(input_lengths)
        total_kv = sum(seq_lengths)

        q = torch.randn(
            total_q, head_num, size_per_head, dtype=torch.float16, device=self.device
        )
        k = torch.randn(
            total_kv,
            head_num_kv,
            size_per_head,
            dtype=torch.float16,
            device=self.device,
        )
        v = torch.randn(
            total_kv,
            head_num_kv,
            size_per_head,
            dtype=torch.float16,
            device=self.device,
        )
        kv_cache = self._make_paged_kv_cache(
            k, v, seq_lengths, head_num_kv, size_per_head
        )

        # Normal path
        normal_inp = self._make_inputs(input_lengths, prefix_lengths)
        normal_op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, normal_inp)
        normal_op.prepare(normal_inp)
        normal_out = normal_op.forward(q, kv_cache)

        # CUDA graph path: forward after init prepare (simulates capture phase).
        # In real CG, forward()/run() only executes during capture (with ideal
        # qo_indptr matching q_aligned.shape[0]). During replay, the captured
        # graph is replayed — run() is never called directly, so FlashInfer's
        # shape validation doesn't apply.
        cg_init = self._make_inputs(input_lengths, prefix_lengths, True, max_seq_len)
        cg_op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, cg_init)
        cg_op.prepare(cg_init)

        # For non-uniform input_lengths (some < max_seq_len), FlashInfer computes
        # causal positions as: pos = qo_offset + kv_len - q_seq_len. With ideal
        # qo_indptr, q_seq_len = max_seq_len (padded), shifting positions and
        # producing numerically different results. Only compare forward output
        # when all inputs are uniform (match max_seq_len).
        is_uniform = all(il == max_seq_len for il in input_lengths)
        if is_uniform:
            cg_out = cg_op.forward(q, kv_cache)
            compare_tensors(
                normal_out,
                cg_out,
                rtol=1e-3,
                atol=1e-3,
                name=f"input={input_lengths}, prefix={prefix_lengths}",
            )

    # === Single batch ===

    def test_no_prefix(self):
        self._test_forward_match(5, 0)

    def test_with_prefix(self):
        self._test_forward_match(5, 100)

    def test_single_token(self):
        self._test_forward_match(1, 200)

    def test_large_prefix(self):
        self._test_forward_match(5, 500)

    def test_varying_input_same_max(self):
        for n in [1, 2, 3, 4, 5]:
            self._test_forward_match(n, 100, max_seq_len=5)

    # === Multi batch ===

    def test_multi_batch_uniform(self):
        self._test_forward_match([5, 5, 5], [100, 100, 100])

    def test_multi_batch_varied_input(self):
        self._test_forward_match([2, 4, 3], [100, 50, 200])

    def test_multi_batch_varied_input_and_prefix(self):
        self._test_forward_match([1, 3, 5, 2], [200, 50, 100, 300])

    def test_multi_batch_single_tokens(self):
        self._test_forward_match([1, 1, 1], [100, 200, 300])

    # === Real CUDA graph capture/replay ===

    def test_non_full_bucket_real_cuda_graph_replay(self):
        """Verify non-full bucket correctness with real CUDA graph capture/replay.

        Captures a real torch.cuda.CUDAGraph during init (ideal qo_indptr),
        replays with actual batch data (update block qo_indptr), and compares
        output against the normal path. This is the only test that exercises
        the full capture→replay flow for non-full buckets.
        """
        head_num, head_num_kv, size_per_head = 8, 2, 64
        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=PAGE_SIZE,
        )

        active_input_lengths = [5, 5, 5]
        active_prefix_lengths = [100, 100, 100]
        max_batch_size = 4
        max_seq_len = 5

        seq_lengths = [
            p + i for p, i in zip(active_prefix_lengths, active_input_lengths)
        ]
        total_q = sum(active_input_lengths)
        total_kv = sum(seq_lengths)

        q = torch.randn(
            total_q, head_num, size_per_head, dtype=torch.float16, device=self.device
        )
        k = torch.randn(
            total_kv,
            head_num_kv,
            size_per_head,
            dtype=torch.float16,
            device=self.device,
        )
        v = torch.randn(
            total_kv,
            head_num_kv,
            size_per_head,
            dtype=torch.float16,
            device=self.device,
        )
        kv_cache = self._make_paged_kv_cache(
            k, v, seq_lengths, head_num_kv, size_per_head
        )

        # 1. Normal path reference
        normal_inp = self._make_inputs(active_input_lengths, active_prefix_lengths)
        normal_op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, normal_inp)
        normal_op.prepare(normal_inp)
        normal_out = normal_op.forward(q, kv_cache).clone()

        # 2. CG path: init prepare (ideal qo_indptr = [0,5,10,15,20])
        cg_init = self._make_inputs(
            active_input_lengths,
            active_prefix_lengths,
            with_copy_params=True,
            max_seq_len=max_seq_len,
            cg_max_batch_size=max_batch_size,
        )
        cg_op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, cg_init)
        cg_op.prepare(cg_init)

        # 3. Warmup forward (allocate internal buffers before capture)
        cg_op.forward(q, kv_cache)

        # 4. Capture the graph (forward/run recorded as CUDA ops)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cg_out = cg_op.forward(q, kv_cache)

        # 5. Replay: update qo_indptr to actual batch data, then replay graph
        cg_replay = self._make_inputs(
            active_input_lengths,
            active_prefix_lengths,
            with_copy_params=True,
            max_seq_len=max_seq_len,
            cg_max_batch_size=max_batch_size,
        )
        cg_op.prepare(cg_replay, forbid_realloc=True)
        graph.replay()
        torch.cuda.synchronize()

        # 6. Compare: replay output should match normal path
        compare_tensors(
            normal_out,
            cg_out,
            rtol=1e-3,
            atol=1e-3,
            name="real_cg_non_full_bucket",
        )

    # === Non-full bucket (active_bs < max_bs) ===

    def test_non_full_bucket_prepare_no_crash(self):
        """Verify first prepare() doesn't crash when active_bs < max_bs.

        This covers the MTP+CG crash scenario: during CUDA graph capture with
        a non-full bucket, the init block must set qo_indptr to ideal padding
        [0, max_sl, 2*max_sl, ...] without the update block overwriting it.

        Without the _is_init guard fix, prepare() would set qo_indptr[-1] to
        the actual token count (< max_batch_size * max_seq_len), causing
        FlashInfer's run() to fail with shape mismatch during capture.
        """
        config = self._create_config(
            head_num=8, head_num_kv=2, size_per_head=64, seq_size_per_block=PAGE_SIZE
        )

        # 3 active batches in a max_batch_size=4 bucket
        active_input_lengths = [5, 5, 5]
        active_prefix_lengths = [100, 100, 100]
        max_batch_size = 4
        max_seq_len = 5

        cg_init = self._make_inputs(
            active_input_lengths,
            active_prefix_lengths,
            with_copy_params=True,
            max_seq_len=max_seq_len,
            cg_max_batch_size=max_batch_size,
        )
        op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, cg_init)

        # This call crashed before the fix:
        # ValueError: q.shape[0] (20) does not match qo_indptr[-1] (15)
        op.prepare(cg_init)  # Must not raise

        # Verify qo_indptr is ideal padding (update block was skipped)
        expected = torch.arange(max_batch_size + 1, dtype=torch.int32) * max_seq_len
        actual = op.qo_indptr.cpu()
        self.assertTrue(
            torch.equal(actual, expected),
            f"qo_indptr should be ideal padding {expected.tolist()}, got {actual.tolist()}",
        )

    def test_non_full_bucket_replay_updates_qo_indptr(self):
        """Verify that the update block runs correctly on subsequent prepare() calls.

        After the init prepare (capture), replay calls must update qo_indptr
        with actual batch data so the kernel processes only real tokens.
        """
        config = self._create_config(
            head_num=8, head_num_kv=2, size_per_head=64, seq_size_per_block=PAGE_SIZE
        )

        active_input_lengths = [5, 5, 5]
        active_prefix_lengths = [100, 100, 100]
        max_batch_size = 4
        max_seq_len = 5

        cg_init = self._make_inputs(
            active_input_lengths,
            active_prefix_lengths,
            with_copy_params=True,
            max_seq_len=max_seq_len,
            cg_max_batch_size=max_batch_size,
        )
        op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, cg_init)
        op.prepare(cg_init)  # init

        # Simulate replay with same data
        cg_replay = self._make_inputs(
            active_input_lengths,
            active_prefix_lengths,
            with_copy_params=True,
            max_seq_len=max_seq_len,
            cg_max_batch_size=max_batch_size,
        )
        op.prepare(cg_replay, forbid_realloc=True)  # replay

        # After replay, qo_indptr should reflect actual batch layout:
        # offsets = [0, 5, 10, 15] * 1 (stride=max_sl=5 per batch slot)
        # qo_indptr[i+1] = i*max_sl + input_lengths[i]
        # For active: [0+5, 5+5, 10+5] = [5, 10, 15]
        # For inactive (input_len=0): [15+0] = [15]
        actual = op.qo_indptr.cpu().tolist()
        # qo_indptr[0] = 0, active batches contribute, inactive stays
        self.assertEqual(actual[0], 0)
        # Active entries: qo_indptr[i+1] = i*5 + 5 for i in [0,1,2]
        for i in range(3):
            self.assertEqual(actual[i + 1], (i * max_seq_len) + 5)
        # Inactive batch: qo_indptr[4] = 3*5 + 0 = 15
        self.assertEqual(actual[4], 3 * max_seq_len + 0)

    def test_non_full_bucket_varied_ratios(self):
        """Test multiple active_bs / max_bs ratios: prepare must not crash and
        qo_indptr must equal ideal padding [0, max_sl, 2*max_sl, ...].

        Without the _is_init guard fix, prepare() would set qo_indptr[-1] to
        the actual token count (< max_bs * max_sl), causing FlashInfer's run()
        to fail with shape mismatch during capture.
        """
        config = self._create_config(
            head_num=8, head_num_kv=2, size_per_head=64, seq_size_per_block=PAGE_SIZE
        )

        test_cases = [
            # (active_input_lengths, max_batch_size, max_seq_len)
            ([5], 4, 5),  # 1 active out of 4
            ([3, 3], 4, 3),  # 2 active out of 4
            ([1, 1, 1, 1, 1], 8, 1),  # 5 active out of 8
            ([7, 7, 7], 4, 7),  # 3 active out of 4 (larger seq)
        ]

        for active_lens, max_bs, max_sl in test_cases:
            with self.subTest(active_lens=active_lens, max_bs=max_bs):
                prefix = [100] * len(active_lens)
                cg_init = self._make_inputs(
                    active_lens,
                    prefix,
                    with_copy_params=True,
                    max_seq_len=max_sl,
                    cg_max_batch_size=max_bs,
                )
                op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, cg_init)
                op.prepare(cg_init)  # Must not crash

                # Full qo_indptr must be ideal padding (init block skips update)
                expected = torch.arange(max_bs + 1, dtype=torch.int32) * max_sl
                actual = op.qo_indptr.cpu()
                self.assertTrue(
                    torch.equal(actual, expected),
                    f"qo_indptr should be ideal padding {expected.tolist()}, "
                    f"got {actual.tolist()} for active={active_lens}, max_bs={max_bs}",
                )


if __name__ == "__main__":
    unittest.main()
