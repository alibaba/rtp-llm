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
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
    TRTLLMFMHAv2PagedPrefillOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.utils import (
    is_cuda_12_9_or_later,
)
from rtp_llm.ops import KvCacheDataType, RopeStyle
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
        is_target_verify=False,
    ):
        """Create PyAttentionInputs for prefill (single or multi batch)."""
        if isinstance(input_lengths, int):
            input_lengths = [input_lengths]
            prefix_lengths = [prefix_lengths]

        batch_size = len(input_lengths)
        inp = PyAttentionInputs()
        inp.is_cuda_graph = with_copy_params
        inp.is_prefill = True
        inp.is_target_verify = is_target_verify
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
            # The production graph replay path copies these changing values H2D.
            inp.cu_seqlens_device = torch.tensor(
                cu, dtype=torch.int32
            ).pin_memory()
            inp.cu_kv_seqlens_device = torch.tensor(
                cu, dtype=torch.int32
            ).pin_memory()
        else:
            inp.cu_seqlens_device = torch.tensor(
                cu, dtype=torch.int32, device="cuda"
            )
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
        capture_input_lengths=None,
        capture_prefix_lengths=None,
        is_target_verify=False,
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
        normal_inp = self._make_inputs(
            input_lengths,
            prefix_lengths,
            is_target_verify=is_target_verify,
        )
        backend = "fa2" if is_target_verify else "auto"
        normal_op = PyFlashinferPrefillPagedAttnOp(
            config.attn_configs,
            normal_inp,
            backend=backend,
        )
        if is_target_verify:
            self.assertEqual(normal_op.backend, "fa2")
        normal_op.prepare(normal_inp)
        normal_out = normal_op.forward(q, kv_cache)

        # CUDA graph path: capture then replay
        capture_input_lengths = capture_input_lengths or input_lengths
        capture_prefix_lengths = capture_prefix_lengths or prefix_lengths
        cg_init = self._make_inputs(
            capture_input_lengths,
            capture_prefix_lengths,
            True,
            max_seq_len,
            is_target_verify,
        )
        self.assertFalse(cg_init.cu_seqlens_device.is_cuda)
        self.assertTrue(cg_init.cu_seqlens_device.is_pinned())
        cg_op = PyFlashinferPrefillPagedAttnOp(
            config.attn_configs,
            cg_init,
            backend=backend,
        )
        if is_target_verify:
            self.assertEqual(cg_op.backend, "fa2")
        cg_op.prepare(cg_init)
        cg_replay = self._make_inputs(
            input_lengths,
            prefix_lengths,
            True,
            max_seq_len,
            is_target_verify,
        )
        self.assertFalse(cg_replay.cu_seqlens_device.is_cuda)
        self.assertTrue(cg_replay.cu_seqlens_device.is_pinned())
        cg_op.prepare(cg_replay, forbid_realloc=True)

        # Warm up allocations/JIT on a side stream before capture. Capture uses
        # different query values so the comparison below requires graph replay.
        static_q = torch.zeros_like(q)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            cg_op.forward(static_q, kv_cache)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cg_out = cg_op.forward(static_q, kv_cache)

        static_q.copy_(q)
        cg_op.prepare(cg_replay, forbid_realloc=True)
        graph.replay()
        torch.cuda.synchronize()

        compare_tensors(
            normal_out,
            cg_out,
            rtol=1e-3,
            atol=1e-3,
            name=f"input={input_lengths}, prefix={prefix_lengths}",
        )

    def test_mrope_target_verify_fa2_matches_trtllm_fmha_v2(self):
        """Compare the post-MRoPE FMHA backend boundary for Qwen2-VL D128."""
        if not is_cuda_12_9_or_later():
            self.skipTest("TRTLLM FMHA v2 requires CUDA 12.9 or later")

        input_lengths = [5]
        prefix_lengths = [1024]
        head_num = 8
        head_num_kv = 1
        head_dim = 128

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=PAGE_SIZE,
            data_type="bf16",
        )
        config.attn_configs.rope_config.style = RopeStyle.Mrope
        config.attn_configs.need_rope_kv_cache = True
        config.attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        config.attn_configs.is_causal = True

        attn_inputs = self._make_inputs(
            input_lengths,
            prefix_lengths,
            is_target_verify=True,
        )
        total_kv = prefix_lengths[0] + input_lengths[0]
        attn_inputs.cu_kv_seqlens_device = torch.tensor(
            [0, total_kv],
            dtype=torch.int32,
            device=self.device,
        )
        attn_inputs.kv_cache_kernel_block_id_device = (
            attn_inputs.kv_cache_kernel_block_id.to(self.device)
        )

        # These tensors represent the output of the shared fused MRoPE/KV-cache
        # stage, so this A/B isolates only the target-verify FMHA backend.
        q = torch.randn(
            input_lengths[0],
            head_num,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k = torch.randn(
            total_kv,
            head_num_kv,
            head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        v = torch.randn_like(k)
        kv_cache = self._make_paged_kv_cache(
            k,
            v,
            [total_kv],
            head_num_kv,
            head_dim,
        )

        flashinfer_op = PyFlashinferPrefillPagedAttnOp(
            config.attn_configs,
            attn_inputs,
            backend="fa2",
        )
        flashinfer_op.prepare(attn_inputs)
        flashinfer_output = flashinfer_op.forward(q, kv_cache)

        self.assertTrue(
            TRTLLMFMHAv2PagedPrefillOp.support(
                config.attn_configs,
                attn_inputs,
            )
        )
        trtllm_op = TRTLLMFMHAv2PagedPrefillOp(config.attn_configs)
        trtllm_params = trtllm_op.prepare(attn_inputs)
        trtllm_output = trtllm_op.forward(q, kv_cache, trtllm_params).view_as(
            flashinfer_output
        )

        torch.testing.assert_close(
            flashinfer_output,
            trtllm_output,
            rtol=2e-2,
            atol=2e-2,
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

    def test_target_verify_long_prefix(self):
        self._test_forward_match(
            5,
            12000,
            head_num=8,
            head_num_kv=1,
            size_per_head=256,
            is_target_verify=True,
        )

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

    def test_multi_batch_single_tokens(self):
        self._test_forward_match([1, 1, 1], [100, 200, 300])


if __name__ == "__main__":
    unittest.main()
