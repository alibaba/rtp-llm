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
    quantize_to_fp8_if_needed,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    compare_tensors,
)
from rtp_llm.ops import KvCacheDataType
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
        self, input_lengths, prefix_lengths, with_copy_params=False, max_seq_len=0
    ):
        """Create PyAttentionInputs for prefill (single or multi batch)."""
        if isinstance(input_lengths, int):
            input_lengths = [input_lengths]
            prefix_lengths = [prefix_lengths]

        batch_size = len(input_lengths)
        inp = PyAttentionInputs()
        inp.is_cuda_graph = with_copy_params
        inp.is_prefill = True
        inp.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device="cuda"
        )
        inp.prefix_lengths = torch.tensor(
            prefix_lengths, dtype=torch.int32, device="cuda"
        )
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

        max_blocks = max(math.ceil(s / PAGE_SIZE) for s in seq_lengths)
        block_ids = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
        offset = 0
        for i, s in enumerate(seq_lengths):
            nb = math.ceil(s / PAGE_SIZE)
            block_ids[i, :nb] = torch.arange(offset, offset + nb)
            offset += nb
        inp.kv_cache_kernel_block_id = block_ids

        if with_copy_params:
            ms = max_seq_len if max_seq_len > 0 else max(input_lengths)
            cp = PyPrefillCudaGaphCopyParams()
            cp.cuda_graph_prefill_batch_size = torch.tensor(
                [batch_size], dtype=torch.int32
            ).pin_memory()
            cp.max_seq_len = ms
            cp.max_batch_size = batch_size
            inp.prefill_cuda_graph_copy_params = cp

        return inp

    def _make_paged_kv_cache(
        self, k, v, seq_lengths, num_kv_heads, head_dim, cache_dtype
    ):
        if isinstance(seq_lengths, int):
            seq_lengths = [seq_lengths]
        total_pages = sum(math.ceil(s / PAGE_SIZE) for s in seq_lengths)
        cache = torch.zeros(
            total_pages,
            2,
            num_kv_heads,
            PAGE_SIZE,
            head_dim,
            dtype=cache_dtype,
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
        capture_input_lengths=None,
        capture_prefix_lengths=None,
        kv_cache_dtype=KvCacheDataType.BASE,
        cache_dtype=torch.float16,
        verify_cast_buffer_reuse=False,
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
        config.attn_configs.kv_cache_dtype = kv_cache_dtype
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
            k, v, seq_lengths, head_num_kv, size_per_head, cache_dtype
        )

        # Normal path
        normal_inp = self._make_inputs(input_lengths, prefix_lengths)
        normal_op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, normal_inp)
        normal_op.prepare(normal_inp)
        normal_out = normal_op.forward(q, kv_cache)

        # CUDA graph path: capture then replay
        capture_input_lengths = capture_input_lengths or input_lengths
        capture_prefix_lengths = capture_prefix_lengths or prefix_lengths
        cg_init = self._make_inputs(
            capture_input_lengths, capture_prefix_lengths, True, max_seq_len
        )
        cg_op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, cg_init)
        cg_op.prepare(cg_init)
        cg_replay = self._make_inputs(input_lengths, prefix_lengths, True, max_seq_len)
        cg_op.prepare(cg_replay, forbid_realloc=True)
        cg_out = cg_op.forward(q, kv_cache)

        if verify_cast_buffer_reuse:
            cast_buf = cg_op._aligned_q_cast_buf
            self.assertIsNotNone(cast_buf)
            first_out = cg_out.clone()
            replay_out = cg_op.forward(q, kv_cache)
            self.assertIs(cg_op._aligned_q_cast_buf, cast_buf)
            compare_tensors(
                first_out,
                replay_out,
                rtol=0,
                atol=0,
                name="FP8 CUDA graph replay",
            )

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

    def test_replay_smaller_batch_with_varied_input(self):
        self._test_forward_match(
            [2, 4, 3],
            [100, 50, 200],
            max_seq_len=5,
            capture_input_lengths=[5, 5, 5, 5],
            capture_prefix_lengths=[200, 200, 200, 200],
        )

    def test_fp8_q_cast_buffer_reused_across_replay(self):
        self._test_forward_match(
            [2, 4, 3],
            [100, 50, 200],
            max_seq_len=5,
            capture_input_lengths=[5, 5, 5, 5],
            capture_prefix_lengths=[200, 200, 200, 200],
            kv_cache_dtype=KvCacheDataType.FP8,
            cache_dtype=torch.float8_e4m3fn,
            verify_cast_buffer_reuse=True,
        )

    def test_fp8_unit_scale_cast_saturates_outliers(self):
        values = torch.tensor(
            [-1000.0, -448.0, 448.0, 1000.0],
            dtype=torch.float16,
            device=self.device,
        )
        quantized = quantize_to_fp8_if_needed(values, torch.float8_e4m3fn).float()
        expected = torch.tensor(
            [-448.0, -448.0, 448.0, 448.0],
            dtype=torch.float32,
            device=self.device,
        )
        torch.testing.assert_close(quantized, expected, rtol=0, atol=0)
        self.assertFalse(torch.isnan(quantized).any().item())

    def test_multi_batch_single_tokens(self):
        self._test_forward_match([1, 1, 1], [100, 200, 300])

    def test_non_fp8_dtype_mismatch_raises(self):
        values = torch.ones(1, dtype=torch.float16, device=self.device)
        with self.assertRaisesRegex(
            ValueError,
            "unsupported dtype conversion from torch.float16 to torch.bfloat16",
        ):
            quantize_to_fp8_if_needed(values, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
