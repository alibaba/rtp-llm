import math
import os
import random
import sys
from typing import List, Optional
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.atten_test_util import (
    attention_prefill_ref,
    gen_attention_inputs,
    write_kv_cache,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
    FlashInferTRTLLMDecodeOp,
    FlashInferTRTLLMPrefillOp,
    _compute_cg_grid,
    _prepare_cg_decode_kernel,
    _prepare_cg_prefill_kernel,
    _prepare_cg_spec_decode_kernel,
)
from rtp_llm.test.utils.numeric_util import assert_close_with_mismatch_tolerance

device = torch.device("cuda")

from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlashInferPythonMHATest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda")
        set_seed(25536)
        self.num_pages = 1024
        self.page_size = 64
        self.head_dim = 128
        self.num_kv_heads = 8
        self.num_heads = 64

    def _init_kv_cache(self, dtype: torch.dtype = torch.float8_e4m3fn) -> LayerKVCache:
        k_cache = (
            torch.rand(
                self.num_pages,
                self.num_kv_heads,
                self.page_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            * 2
            - 1
        ).to(dtype)
        v_cache = (
            torch.rand(
                self.num_pages,
                self.num_kv_heads,
                self.page_size,
                self.head_dim,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            * 2
            - 1
        ).to(dtype)
        kv_cache: LayerKVCache = LayerKVCache()
        kv_cache.kv_cache_base = torch.stack([k_cache, v_cache], dim=1)
        return kv_cache

    def _create_config(self, dtype) -> AttentionConfigs:
        """Create a standard AttentionConfigs config for testing."""
        config = AttentionConfigs()
        config.head_num = self.num_heads
        config.kv_head_num = self.num_kv_heads
        config.size_per_head = self.head_dim
        config.tokens_per_block = self.page_size
        config.kernel_tokens_per_block = self.page_size
        config.kv_cache_dtype = (
            KvCacheDataType.FP8
            if dtype is torch.float8_e4m3fn
            else KvCacheDataType.BASE
        )
        config.use_mla = False
        config.is_causal = True
        config.fuse_qkv_add_bias = True
        config.q_scaling = 1.0
        return config

    def _test_flashinfer_trtllm_base(
        self,
        dtype: torch.dtype,
        lengths: List[int],
        is_prefill: bool,
        use_prefill_op: bool,
    ):
        """Test FlashInferTRTLLM attention with reference comparison.

        Args:
            lengths: sequence lengths for each request.
            is_prefill: if True, treat lengths as input_lengths (prefill data);
                        if False, treat as sequence_lengths (decode data).
            use_prefill_op: if True, use FlashInferTRTLLMPrefillOp;
                            if False, use FlashInferTRTLLMDecodeOp.
        """
        is_sm_100 = torch.cuda.get_device_capability()[0] in [10]
        if not is_sm_100:
            raise SkipTest("FlashInferTRTLLM requires SM_100 (compute capability 10.0)")

        config = self._create_config(dtype)
        q_size = self.head_dim * self.num_heads
        k_size = self.head_dim * self.num_kv_heads
        v_size = self.head_dim * self.num_kv_heads
        qkv_dim = self.head_dim * self.num_heads + 2 * self.num_kv_heads * self.head_dim
        num_tokens = sum(lengths)
        if is_prefill:
            attn_inputs = gen_attention_inputs(
                self.page_size, self.num_pages, input_lengths=lengths
            )
        else:
            attn_inputs = gen_attention_inputs(
                self.page_size, self.num_pages, sequence_lengths=lengths
            )

        qkv = (
            torch.rand([num_tokens, qkv_dim], dtype=torch.bfloat16, device=self.device)
            * 2
            - 1
        )
        q_ref = qkv[:, :q_size].reshape(num_tokens, self.num_heads, self.head_dim)
        k_ref = qkv[:, q_size : q_size + k_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v_ref = qkv[:, q_size + k_size : q_size + k_size + v_size].reshape(
            num_tokens, self.num_kv_heads, self.head_dim
        )

        kv_cache = self._init_kv_cache(dtype)
        kv_write_lengths = (
            attn_inputs.input_lengths if is_prefill else attn_inputs.sequence_lengths
        )
        write_kv_cache(
            k_ref, v_ref, kv_cache, kv_write_lengths, attn_inputs.kv_cache_block_id_host
        )

        out_ref = attention_prefill_ref(
            q_ref,
            k_ref,
            v_ref,
            attn_inputs.sequence_lengths,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            causal=True,
        )

        if use_prefill_op:
            op = FlashInferTRTLLMPrefillOp(config)
            input_params = op.prepare(attn_inputs)
            out_trtllm = op.forward(q_ref, kv_cache, input_params)

            out_trtllm_f32 = out_trtllm.reshape(
                num_tokens, self.num_heads, self.head_dim
            ).float()
            out_ref_f32 = out_ref.float()
            allowed_mismatch_rate = 1e-5
        else:
            last_token_idx = attn_inputs.cu_seqlens[1:] - 1
            if is_prefill:
                q_len_per_req = 6
                attn_inputs.prefix_lengths = attn_inputs.input_lengths - q_len_per_req
                attn_inputs.input_lengths = (
                    torch.ones_like(
                        attn_inputs.input_lengths, dtype=attn_inputs.input_lengths.dtype
                    )
                    * q_len_per_req
                )
                last_token_idx = last_token_idx.unsqueeze(-1).repeat(
                    1, q_len_per_req
                ) - torch.arange(q_len_per_req).flip([0]).view(1, q_len_per_req)
                last_token_idx = last_token_idx.reshape(-1)
            out_ref = out_ref[last_token_idx]
            op = FlashInferTRTLLMDecodeOp(config)
            q = q_ref[last_token_idx]
            attn_inputs.sequence_lengths -= 1
            input_params = op.prepare(attn_inputs)
            out_trtllm = op.forward(q, kv_cache, input_params)

            out_trtllm_f32 = out_trtllm.reshape(
                -1, self.num_heads, self.head_dim
            ).float()
            out_ref_f32 = out_ref.float()
            allowed_mismatch_rate = 1e-3
        assert_close_with_mismatch_tolerance(
            out_trtllm_f32,
            out_ref_f32,
            atol=0.04,
            rtol=0.04,
            max_mismatched_elements=int(allowed_mismatch_rate * out_ref_f32.numel()),
        )

    def test_flashinfer_trtllm_prefill_op_bf16(self):
        self._test_flashinfer_trtllm_base(
            torch.bfloat16, [2, 129, 255, 63], is_prefill=True, use_prefill_op=True
        )

    def test_flashinfer_trtllm_prefill_op_fp8(self):
        self._test_flashinfer_trtllm_base(
            torch.float8_e4m3fn, [2, 129, 255, 63], is_prefill=True, use_prefill_op=True
        )

    def test_flashinfer_trtllm_spec_op_bf16(self):
        self._test_flashinfer_trtllm_base(
            torch.bfloat16, [11, 129, 255, 63], is_prefill=True, use_prefill_op=False
        )

    def test_flashinfer_trtllm_spec_op_fp8(self):
        self._test_flashinfer_trtllm_base(
            torch.float8_e4m3fn,
            [11, 129, 255, 63],
            is_prefill=True,
            use_prefill_op=False,
        )

    def test_flashinfer_trtllm_decode_op_bf16(self):
        self._test_flashinfer_trtllm_base(
            torch.bfloat16, [2, 129, 255, 63], is_prefill=False, use_prefill_op=False
        )

    def test_flashinfer_trtllm_decode_op_fp8(self):
        self._test_flashinfer_trtllm_base(
            torch.float8_e4m3fn,
            [11, 129, 255, 63],
            is_prefill=False,
            use_prefill_op=False,
        )


class PrepareCudaGraphKernelTest(TestCase):
    """Tests for the _prepare_cg_*_kernel Triton kernels."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda")
        set_seed(42)

    def _run_kernel(
        self,
        src1,
        src2,
        seq_lens_out,
        cu_kv_seqlens_out,
        block_id,
        kv_offset_out,
        page_size,
        N,
        M,
        mode,
    ):
        grid, _, _, total_bm, BLOCK_SIZE = _compute_cg_grid(N, M)
        if mode == 0:
            _prepare_cg_decode_kernel[grid](
                src1,
                seq_lens_out,
                block_id,
                kv_offset_out,
                N,
                M,
                total_bm,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif mode == 1:
            _prepare_cg_spec_decode_kernel[grid](
                src1,
                src2,
                seq_lens_out,
                block_id,
                kv_offset_out,
                N,
                M,
                total_bm,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        else:
            _prepare_cg_prefill_kernel[grid](
                src1,
                src2,
                seq_lens_out,
                cu_kv_seqlens_out,
                block_id,
                kv_offset_out,
                page_size,
                N,
                M,
                total_bm,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        torch.cuda.synchronize()

    def _make_block_id(self, N, M):
        return torch.randint(0, 512, (N, M), dtype=torch.int32, device=self.device)

    def _reference_kv_offset(self, block_id):
        """Reference: block_id[B,M] -> kv_offset[B,2,M] with K=id*2, V=id*2+1."""
        B, M = block_id.shape
        kv_offset = torch.zeros(B, 2, M, dtype=torch.int32, device=self.device)
        for b in range(B):
            for m in range(M):
                bid = block_id[b, m].item()
                kv_offset[b, 0, m] = bid * 2
                kv_offset[b, 1, m] = bid * 2 + 1
        return kv_offset

    # -- decode: seq_lens = copy(src1) --

    def test_mode0_base(self):
        N, M = 4, 8
        src1 = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=self.device)
        src2 = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            src1, src2, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=0
        )
        expected_kv = self._reference_kv_offset(block_id)
        torch.testing.assert_close(kv_offset, expected_kv)
        torch.testing.assert_close(seq_lens_out, src1)

    def test_mode0_large_batch(self):
        N, M = 128, 16
        src1 = torch.randint(1, 1000, (N,), dtype=torch.int32, device=self.device)
        src2 = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            src1, src2, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=0
        )

        torch.testing.assert_close(seq_lens_out, src1)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    # -- spec-decode prefill: seq_lens = prefix + src2[0] --

    def test_mode1_base(self):
        N, M = 4, 8
        prefix = torch.tensor(
            [100, 200, 300, 400], dtype=torch.int32, device=self.device
        )
        q_len_tensor = torch.tensor([5], dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            prefix,
            q_len_tensor,
            seq_lens_out,
            cu_kv,
            block_id,
            kv_offset,
            0,
            N,
            M,
            mode=1,
        )

        expected = prefix + 5
        torch.testing.assert_close(seq_lens_out, expected)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_mode1_large_batch(self):
        N, M = 64, 32
        prefix = torch.randint(50, 500, (N,), dtype=torch.int32, device=self.device)
        q_len_tensor = torch.tensor([7], dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            prefix,
            q_len_tensor,
            seq_lens_out,
            cu_kv,
            block_id,
            kv_offset,
            0,
            N,
            M,
            mode=1,
        )

        torch.testing.assert_close(seq_lens_out, prefix + 7)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    # -- prefill: seq_lens = input + prefix, cu_kv_seqlens --

    def test_mode2_base(self):
        N, M = 4, 8
        page_size = 64
        input_lens = torch.tensor(
            [10, 20, 30, 40], dtype=torch.int32, device=self.device
        )
        prefix_lens = torch.tensor(
            [5, 15, 25, 35], dtype=torch.int32, device=self.device
        )
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            input_lens,
            prefix_lens,
            seq_lens_out,
            cu_kv,
            block_id,
            kv_offset,
            page_size,
            N,
            M,
            mode=2,
        )

        expected_seq = input_lens + prefix_lens
        torch.testing.assert_close(seq_lens_out, expected_seq)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))
        total_seq = (input_lens + prefix_lens).cpu()
        pages_per_seq = (total_seq + page_size - 1) // page_size
        expected_cu = torch.zeros(N + 1, dtype=torch.int32)
        expected_cu[1:] = torch.cumsum(pages_per_seq, dim=0)
        torch.testing.assert_close(cu_kv.cpu(), expected_cu)

    def test_mode2_large_batch(self):
        N, M = 128, 32
        page_size = 128
        input_lens = torch.randint(1, 500, (N,), dtype=torch.int32, device=self.device)
        prefix_lens = torch.randint(0, 200, (N,), dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            input_lens,
            prefix_lens,
            seq_lens_out,
            cu_kv,
            block_id,
            kv_offset,
            page_size,
            N,
            M,
            mode=2,
        )

        expected_seq = input_lens + prefix_lens
        torch.testing.assert_close(seq_lens_out, expected_seq)

        total_cpu = expected_seq.cpu()
        pages = (total_cpu + page_size - 1) // page_size
        expected_cu = torch.zeros(N + 1, dtype=torch.int32)
        expected_cu[1:] = torch.cumsum(pages, dim=0)
        torch.testing.assert_close(cu_kv.cpu(), expected_cu)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    # -- Edge cases --

    def test_single_batch(self):
        """Single-element batch for all modes."""
        M = 4
        for mode in [0, 1, 2]:
            src1 = torch.tensor([42], dtype=torch.int32, device=self.device)
            src2 = torch.tensor([10], dtype=torch.int32, device=self.device)
            seq_lens_out = torch.zeros(1, dtype=torch.int32, device=self.device)
            cu_kv = torch.zeros(2, dtype=torch.int32, device=self.device)
            block_id = self._make_block_id(1, M)
            kv_offset = torch.zeros(1, 2, M, dtype=torch.int32, device=self.device)
            page_size = 64 if mode == 2 else 0

            self._run_kernel(
                src1,
                src2,
                seq_lens_out,
                cu_kv,
                block_id,
                kv_offset,
                page_size,
                1,
                M,
                mode=mode,
            )

            if mode == 0:
                self.assertEqual(seq_lens_out.item(), 42)
            elif mode == 1:
                self.assertEqual(seq_lens_out.item(), 42 + 10)
            else:
                self.assertEqual(seq_lens_out.item(), 42 + 10)
                expected_pages = (52 + 64 - 1) // 64
                self.assertEqual(cu_kv[0].item(), 0)
                self.assertEqual(cu_kv[1].item(), expected_pages)

            torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))

    def test_mode2_page_boundary(self):
        """Sequences exactly on page boundaries."""
        N, M = 3, 4
        page_size = 64
        input_lens = torch.tensor([64, 128, 192], dtype=torch.int32, device=self.device)
        prefix_lens = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            input_lens,
            prefix_lens,
            seq_lens_out,
            cu_kv,
            block_id,
            kv_offset,
            page_size,
            N,
            M,
            mode=2,
        )

        torch.testing.assert_close(seq_lens_out, input_lens)
        # Exact page boundaries: 1, 2, 3 pages
        expected_cu = torch.tensor([0, 1, 3, 6], dtype=torch.int32)
        torch.testing.assert_close(cu_kv.cpu(), expected_cu)

    def test_mode2_one_over_page_boundary(self):
        """Sequences one token over page boundaries need an extra page."""
        N, M = 3, 8
        page_size = 64
        input_lens = torch.tensor([65, 129, 193], dtype=torch.int32, device=self.device)
        prefix_lens = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            input_lens,
            prefix_lens,
            seq_lens_out,
            cu_kv,
            block_id,
            kv_offset,
            page_size,
            N,
            M,
            mode=2,
        )

        # 65 -> 2 pages, 129 -> 3 pages, 193 -> 4 pages
        expected_cu = torch.tensor([0, 2, 5, 9], dtype=torch.int32)
        torch.testing.assert_close(cu_kv.cpu(), expected_cu)

    def test_large_M_multi_block(self):
        """total_bm exceeds BLOCK_SIZE, exercising multi-block grid."""
        N, M = 8, 256
        src1 = torch.randint(1, 100, (N,), dtype=torch.int32, device=self.device)
        src2 = torch.zeros(N, dtype=torch.int32, device=self.device)
        seq_lens_out = torch.zeros(N, dtype=torch.int32, device=self.device)
        cu_kv = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        block_id = self._make_block_id(N, M)
        kv_offset = torch.zeros(N, 2, M, dtype=torch.int32, device=self.device)

        self._run_kernel(
            src1, src2, seq_lens_out, cu_kv, block_id, kv_offset, 0, N, M, mode=0
        )

        torch.testing.assert_close(seq_lens_out, src1)
        torch.testing.assert_close(kv_offset, self._reference_kv_offset(block_id))


if __name__ == "__main__":
    main()
