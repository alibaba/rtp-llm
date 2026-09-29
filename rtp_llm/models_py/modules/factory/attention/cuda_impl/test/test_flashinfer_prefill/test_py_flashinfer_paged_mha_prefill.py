import logging
import math
import unittest
from types import SimpleNamespace
from typing import List
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferHybridPrefillAttnOp,
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillPagedAttnOp,
    attn_kv_dtype,
    attn_q_dtype,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    fill_paged_kv_cache,
)
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferPrefillPagedAttnOp(BaseAttentionTest):
    """Test suite for PyFlashinferPrefillPagedAttnOp with paged KV cache"""

    atol = 5e-3

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - this test requires CUDA")

        # Call parent setUp for common initialization
        super().setUp()

    def _create_paged_kv_cache(
        self,
        k_ragged: torch.Tensor,
        v_ragged: torch.Tensor,
        sequence_lengths: List[int],
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        block_table: torch.Tensor,
        cache_dtype: torch.dtype,
    ) -> LayerKVCache:
        """
        Convert ragged K, V to paged KV cache format (HND layout)

        Args:
            k_ragged: [total_tokens, num_kv_heads, head_dim]
            v_ragged: [total_tokens, num_kv_heads, head_dim]
            sequence_lengths: List of sequence lengths
            page_size: Page size
            num_kv_heads: Number of KV heads
            head_dim: Head dimension
            block_table: [batch, max_pages] page ids the op will read

        Returns:
            paged_kv_cache: [num_pages, 2, num_kv_heads, page_size, head_dim] (HND layout)
        """
        offsets = [0]
        for seq_len in sequence_lengths:
            offsets.append(offsets[-1] + seq_len)
        per_batch = [
            (
                k_ragged[offsets[i] : offsets[i + 1]],
                v_ragged[offsets[i] : offsets[i + 1]],
            )
            for i in range(len(sequence_lengths))
        ]
        return fill_paged_kv_cache(
            [entry[0] for entry in per_batch],
            [entry[1] for entry in per_batch],
            sequence_lengths,
            block_table,
            page_size,
            num_kv_heads,
            head_dim,
            cache_dtype,
            self.device,
        )

    def _test_prefill_correctness(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        page_size: int = 64,
        causal: bool = True,
    ):
        """Test prefill correctness by comparing with flashinfer reference implementation"""

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=page_size,
        )
        config.attn_configs.is_causal = causal

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, sequence_lengths, config.seq_size_per_block
        )

        # Create PyFlashinferPrefillPagedAttnOp instance
        attn_op = PyFlashinferPrefillPagedAttnOp(
            config.attn_configs,
            attn_inputs,
        )

        # Check support
        if not attn_op.support(attn_inputs):
            raise RuntimeError(
                "PyFlashinferPrefillPagedAttnOp does not support this configuration"
            )

        # Prepare params
        params = attn_op.prepare(attn_inputs)

        # Create Q input
        total_tokens = sum(sequence_lengths)
        hidden_size_q = config.size_per_head * config.head_num

        # Create Q tensor [total_tokens, hidden_size_q]
        q_flat = torch.randn(
            total_tokens,
            hidden_size_q,
            dtype=torch.float16,
            device=self.device,
        )
        q = q_flat.reshape(total_tokens, config.head_num, config.size_per_head)

        # Create K, V for paged cache
        hidden_size_k = config.size_per_head * config.head_num_kv
        hidden_size_v = config.size_per_head * config.head_num_kv

        k_flat = torch.randn(
            total_tokens,
            hidden_size_k,
            dtype=torch.float16,
            device=self.device,
        )
        v_flat = torch.randn(
            total_tokens,
            hidden_size_v,
            dtype=torch.float16,
            device=self.device,
        )
        k = k_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        v = v_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)

        cache_dtype = self.cache_dtype(config.attn_configs)
        k = k.to(cache_dtype)
        v = v.to(cache_dtype)

        # Create paged KV cache
        paged_kv_cache = self._create_paged_kv_cache(
            k,
            v,
            sequence_lengths,
            page_size,
            config.head_num_kv,
            config.size_per_head,
            attn_inputs.kv_cache_kernel_block_id,
            cache_dtype,
        )

        # Forward pass through PyFlashinferPrefillPagedAttnOp
        output = attn_op.forward(q, paged_kv_cache)  # Use layer 0

        # Compute reference outputs using flashinfer's reference (with round-trip)
        ref_output = compute_flashinfer_prefill_reference(
            q.to(attn_op.q_dtype).to(q.dtype),
            k.to(q.dtype),
            v.to(q.dtype),
            attn_inputs.cu_seqlens_device,
            causal=causal,
        )

        # Compare outputs
        print(
            f"Testing batch_size={batch_size}, seq_lens={sequence_lengths}, "
            f"head_num={head_num}, kv_head_num={head_num_kv}, "
            f"size_per_head={size_per_head}, page_size={page_size}"
        )

        # Assert closeness (with relaxed tolerance for FP16)
        try:
            self._assert_output_close(output, ref_output, name="Prefill output")
            print("✓ Test passed")
        except AssertionError as e:
            logging.error(f"✗ Test failed: {e}")
            raise

    # ========== Test Cases: Single Batch ==========

    def test_single_sequence_small(self):
        """Test single sequence with small length"""
        self._test_prefill_correctness(
            batch_size=1,
            sequence_lengths=[32],
            head_num=8,
            head_num_kv=2,
            size_per_head=64,
            page_size=16,
        )

    def test_non_causal_prefill(self):
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[64, 96],
            head_num=8,
            head_num_kv=2,
            size_per_head=64,
            page_size=16,
            causal=False,
        )

    def test_single_sequence_medium(self):
        """Test single sequence with medium length"""
        self._test_prefill_correctness(
            batch_size=1,
            sequence_lengths=[128],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_single_sequence_large(self):
        """Test single sequence with large length"""
        self._test_prefill_correctness(
            batch_size=1,
            sequence_lengths=[512],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    # ========== Test Cases: Multi Batch ==========

    def test_multi_batch_uniform(self):
        """Test multiple sequences with uniform lengths"""
        self._test_prefill_correctness(
            batch_size=4,
            sequence_lengths=[64, 64, 64, 64],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    def test_multi_batch_varied(self):
        """Test multiple sequences with varied lengths"""
        self._test_prefill_correctness(
            batch_size=4,
            sequence_lengths=[32, 64, 128, 256],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=64,
        )

    # ========== Test Cases: Chunked Prefill (Prefix Caching) ==========

    def test_chunked_prefill_single_batch(self):
        """Test chunked prefill with single batch (mimics your real scenario)

        Scenario:
        - Existing KV cache: 4884 tokens
        - New Q input: 5 tokens
        - Total KV: 4889 tokens
        """
        print("\n" + "=" * 70)
        print("Testing CHUNKED PREFILL scenario")
        print("  prefix_length: 4884 (existing KV cache)")
        print("  input_length: 5 (new Q tokens)")
        print("  Expected: Q[i] attends to KV[0:4884+i+1]")
        print("=" * 70)

        prefix_lengths = [4884]
        input_lengths = [5]
        page_size = 64
        head_num = 40
        head_num_kv = 8
        size_per_head = 128

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=page_size,
        )

        # Create chunked prefill attention inputs
        attn_inputs = self._create_chunked_prefill_attention_inputs(
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            seq_size_per_block=config.seq_size_per_block,
        )

        # Create PyFlashinferPrefillPagedAttnOp instance
        attn_op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, attn_inputs)

        # Check support
        if not attn_op.support(attn_inputs):
            raise RuntimeError(
                "PyFlashinferPrefillPagedAttnOp does not support chunked prefill"
            )

        # Prepare params
        params = attn_op.prepare(attn_inputs)

        # Create Q input (only for new tokens)
        total_q_tokens = sum(input_lengths)  # 5

        q = torch.randn(
            total_q_tokens,
            config.head_num,
            config.size_per_head,
            dtype=torch.float16,
            device=self.device,
        )

        # Create K, V for FULL sequence (prefix + input)
        total_kv_tokens = sum(
            [p + i for p, i in zip(prefix_lengths, input_lengths)]
        )  # 4889

        k = torch.randn(
            total_kv_tokens,
            config.head_num_kv,
            config.size_per_head,
            dtype=torch.float16,
            device=self.device,
        )
        v = torch.randn(
            total_kv_tokens,
            config.head_num_kv,
            config.size_per_head,
            dtype=torch.float16,
            device=self.device,
        )

        cache_dtype = self.cache_dtype(config.attn_configs)
        k = k.to(cache_dtype)
        v = v.to(cache_dtype)

        # Create paged KV cache
        sequence_lengths = [p + i for p, i in zip(prefix_lengths, input_lengths)]
        paged_kv_cache = self._create_paged_kv_cache(
            k,
            v,
            sequence_lengths,
            page_size,
            config.head_num_kv,
            config.size_per_head,
            attn_inputs.kv_cache_kernel_block_id,
            cache_dtype,
        )

        # Forward pass through PyFlashinferPrefillPagedAttnOp
        print("\nRunning FlashInfer forward pass...")
        output = attn_op.forward(q, paged_kv_cache)

        print(f"Output shape: {output.shape}")
        print(f"Output has NaN: {torch.isnan(output).any().item()}")
        print(f"Output has Inf: {torch.isinf(output).any().item()}")

        # Try to verify the output is not all NaN
        if torch.isnan(output).all():
            raise RuntimeError(
                "❌ All output is NaN! FlashInfer chunked prefill failed!"
            )

        print("✅ Test completed (output not all NaN)")

        # ========== Correctness Verification (Simplified) ==========
        print("\n" + "=" * 70)
        print("Computing reference output (simplified approach)...")
        print("=" * 70)

        # 简化方法：构造完整的 Q（前面用0填充），然后只取最后几个输出
        # 这样可以直接用标准的 single_prefill_with_kv_cache

        from flashinfer.prefill import single_prefill_with_kv_cache

        # 构造完整长度的 Q（和 K/V 一样长）
        # 前 prefix_len 个位置填0，后 input_len 个位置是真实的 Q
        prefix_len = prefix_lengths[0]
        input_len = input_lengths[0]
        seq_len = prefix_len + input_len

        # Q_full: [seq_len, num_heads, head_dim]
        q_full = torch.zeros(
            seq_len,
            config.head_num,
            config.size_per_head,
            dtype=torch.float16,
            device=self.device,
        )
        # with round-trip
        q_full[prefix_len:] = q.to(attn_op.q_dtype).to(q.dtype)  # 把真实的 Q 放在后面

        print(f"  Q_full shape: {q_full.shape} (padded)")
        print(f"  K shape: {k.shape}")
        print(f"  Prefix: {prefix_len}, Input: {input_len}")

        # 用 FlashInfer 计算完整的 attention
        ref_output_full = single_prefill_with_kv_cache(
            q_full, k.to(q.dtype), v.to(q.dtype), causal=True, kv_layout="NHD"
        )

        # 只取最后 input_len 个输出（对应真实的 Q）
        ref_output = ref_output_full[prefix_len:]

        print(f"\n[Reference Output]")
        print(f"  Shape: {ref_output.shape}")
        print(f"  Has NaN: {torch.isnan(ref_output).any().item()}")
        print(
            f"  Range: [{ref_output.min().item():.4f}, {ref_output.max().item():.4f}]"
        )

        print(f"\n[Test Output]")
        print(f"  Shape: {output.shape}")
        print(f"  Has NaN: {torch.isnan(output).any().item()}")
        if not torch.isnan(output).any():
            print(f"  Range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # Compare outputs
        print(f"\n[Correctness Check]")
        try:
            self._assert_output_close(
                output,
                ref_output,
                rtol=max(self.rtol, 1e-2),
                atol=max(self.atol, 1e-2),
                name="Chunked prefill output",
            )
            print("✅ Correctness check PASSED!")
        except AssertionError as e:
            print(f"❌ Correctness check FAILED: {e}")

            # Detailed debugging
            diff = (output - ref_output).abs()
            print(f"\n[Debugging Info]")
            print(f"  Max absolute difference: {diff.max().item():.6f}")
            print(f"  Mean absolute difference: {diff.mean().item():.6f}")
            print(f"  Median absolute difference: {diff.median().item():.6f}")

            # Find tokens with largest errors
            max_diff_idx = int(diff.view(-1).argmax().item())
            token_idx = max_diff_idx // (config.head_num * config.size_per_head)
            print(f"  Token with max error: {token_idx}")
            print(f"    Test output: {output.view(-1)[max_diff_idx].item():.6f}")
            print(f"    Ref output: {ref_output.view(-1)[max_diff_idx].item():.6f}")

            raise

    def test_multi_batch_small_lengths(self):
        """Test multiple sequences with small lengths"""
        self._test_prefill_correctness(
            batch_size=3,
            sequence_lengths=[8, 16, 24],
            head_num=8,
            head_num_kv=2,
            size_per_head=64,
            page_size=16,
        )

    # ========== Test Cases: Different Page Sizes ==========

    def test_small_page_size(self):
        """Test with small page size"""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[128, 256],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=32,
        )

    def test_large_page_size(self):
        """Test with large page size"""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[128, 256],
            head_num=32,
            head_num_kv=8,
            size_per_head=128,
            page_size=128,
        )

    # ========== Test Cases: Different Head Configurations ==========

    def test_many_heads(self):
        """Test with many heads"""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[64, 128],
            head_num=64,
            head_num_kv=16,
            size_per_head=128,
            page_size=64,
        )

    def test_gqa_4(self):
        """Test with GQA group size 4"""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[64, 128],
            head_num=32,
            head_num_kv=8,  # 32/8 = 4 queries per KV
            size_per_head=128,
            page_size=64,
        )


class TestPyFlashinferPrefillPagedAttnOpFP8(TestPyFlashinferPrefillPagedAttnOp):
    kv_cache_dtype = KvCacheDataType.FP8
    rtol = 4e-2
    atol = 4e-2
    max_mismatch_rate = 1e-5

    def test_mode1_interleaved_mrope_writes_before_paged_attention(self):
        sequence_lengths = [3, 5]
        page_size = 4
        config = self._create_config(
            head_num=4,
            head_num_kv=2,
            size_per_head=256,
            seq_size_per_block=page_size,
        )
        self._enable_qwen35_mrope_mode1(config)
        inputs = self._create_prefill_attention_inputs(
            len(sequence_lengths), sequence_lengths, page_size
        )
        position_ids = self._add_qwen35_mrope_inputs(inputs, sequence_lengths)
        self.assertTrue(
            PyFlashinferPagedPrefillImpl.support(config.attn_configs, inputs)
        )

        total_tokens = sum(sequence_lengths)
        qkv = torch.randn(
            total_tokens,
            (config.head_num + 2 * config.head_num_kv) * config.size_per_head,
            dtype=config.attn_configs.dtype,
            device=self.device,
        )
        q, k, v = torch.split(
            qkv,
            [
                config.head_num * config.size_per_head,
                config.head_num_kv * config.size_per_head,
                config.head_num_kv * config.size_per_head,
            ],
            dim=-1,
        )
        q = q.reshape(total_tokens, config.head_num, config.size_per_head)
        k = k.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        v = v.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        expected_q, expected_k = self._apply_qwen35_mrope_reference(q, k, position_ids)

        page_count = sum(math.ceil(length / page_size) for length in sequence_lengths)
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = torch.zeros(
            page_count,
            2,
            config.head_num_kv,
            page_size,
            config.size_per_head,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        kv_cache.kv_scale_base = torch.ones(
            page_count,
            2 * config.head_num_kv * page_size,
            dtype=torch.float32,
            device=self.device,
        )
        expected_cache = kv_cache.kv_cache_base.clone()
        token_offset = 0
        for batch_idx, sequence_length in enumerate(sequence_lengths):
            for position in range(sequence_length):
                page_id = int(
                    inputs.kv_cache_kernel_block_id[
                        batch_idx, position // page_size
                    ].item()
                )
                page_offset = position % page_size
                expected_cache[page_id, 0, :, page_offset] = expected_k[
                    token_offset + position
                ].to(torch.float8_e4m3fn)
                expected_cache[page_id, 1, :, page_offset] = v[
                    token_offset + position
                ].to(torch.float8_e4m3fn)
            token_offset += sequence_length

        impl = PyFlashinferPagedPrefillImpl(
            config.attn_configs, inputs, config.parallelism_config
        )
        events = []
        fused_forward = impl.fused_mrope_impl.forward
        cache_write = impl.kv_cache_write_op.forward

        def observed_fused_forward(qkv_input, cache, params):
            events.append("fused_rope")
            self.assertIsNone(cache)
            return fused_forward(qkv_input, cache, params)

        def observed_cache_write(key, value, cache):
            events.append("cache_write")
            return cache_write(key, value, cache)

        def observed_cache_store(*args, **kwargs):
            events.append("cache_store")

        def observed_attention(query, cache):
            events.append("paged_attention")
            self.assertIs(cache, kv_cache)
            self.assertEqual(cache.kv_cache_base.dtype, torch.float8_e4m3fn)
            torch.testing.assert_close(
                cache.kv_cache_base.float(), expected_cache.float(), rtol=0, atol=0
            )
            return query

        with mock.patch.object(
            impl.fused_mrope_impl,
            "forward",
            side_effect=observed_fused_forward,
        ) as fused_mock, mock.patch.object(
            impl.kv_cache_write_op,
            "forward",
            side_effect=observed_cache_write,
        ) as write_mock, mock.patch.object(
            common,
            "apply_write_cache_store",
            side_effect=observed_cache_store,
        ), mock.patch.object(
            impl.fmha_impl,
            "forward",
            side_effect=observed_attention,
        ):
            output = impl.forward(qkv.clone(), kv_cache)

        fused_mock.assert_called_once()
        write_mock.assert_called_once()
        self.assertEqual(
            events, ["fused_rope", "cache_write", "cache_store", "paged_attention"]
        )
        torch.testing.assert_close(output, expected_q, rtol=1e-2, atol=1e-2)


class _NoReleasePagedAttnOp(PyFlashinferPrefillPagedAttnOp):
    def __del__(self):
        pass


class TestDynamicFp8PagedPrefillUnit(unittest.TestCase):
    @staticmethod
    def _config() -> SimpleNamespace:
        return SimpleNamespace(
            dtype=torch.bfloat16,
            kv_cache_dtype=KvCacheDataType.FP8,
            fp8_kv_cache_mode=2,
            head_num=4,
            kv_head_num=2,
            size_per_head=4,
            tokens_per_block=32,
            kernel_tokens_per_block=8,
            is_causal=True,
        )

    def test_mode2_uses_base_plan_dtypes_and_compact_page_ids(self):
        config = self._config()
        self.assertEqual(attn_q_dtype(config), torch.bfloat16)
        self.assertEqual(attn_kv_dtype(config), torch.bfloat16)

        params = SimpleNamespace(
            fill_params=mock.Mock(),
            decode_page_indptr_d=torch.tensor([0, 2], dtype=torch.int32),
            page_indice_d=torch.tensor([9, 3, 77], dtype=torch.int32),
            paged_kv_last_page_len_d=torch.tensor([2], dtype=torch.int32),
        )
        wrapper = SimpleNamespace(_qo_indptr_buf=None, plan=mock.Mock())
        op = _NoReleasePagedAttnOp.__new__(_NoReleasePagedAttnOp)
        op.g_workspace_buffer = torch.empty(0)
        op.local_head_num = 4
        op.local_kv_head_num = 2
        op.head_dim_qk = 4
        op.page_size = 8
        op.dtype = torch.bfloat16
        op.kv_dtype = torch.bfloat16
        op.q_dtype = torch.bfloat16
        op.is_causal = True
        op.dynamic_fp8 = True
        op.direct_scale_fp8 = False
        op.enable_cuda_graph = False
        op.prefill_cuda_graph_copy_params = None
        op.fmha_params = params
        op.prefill_wrapper = wrapper
        inputs = SimpleNamespace(
            kv_cache_kernel_block_id=torch.tensor([[9, 3]], dtype=torch.int32),
            kv_cache_kernel_block_id_device=None,
            input_lengths=torch.tensor([2], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            sequence_lengths=torch.tensor([2], dtype=torch.int32),
            cu_seqlens_device=torch.tensor([0, 2], dtype=torch.int32),
            prefill_cuda_graph_copy_params=None,
        )

        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "check_attention_inputs"
        ):
            op.prepare(inputs)

        plan_args = wrapper.plan.call_args.args
        torch.testing.assert_close(
            plan_args[2], torch.tensor([0, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            op._active_source_page_indices, torch.tensor([9, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(
            params.page_indice_d, torch.tensor([9, 3, 77], dtype=torch.int32)
        )
        self.assertEqual(wrapper.plan.call_args.kwargs["q_data_type"], torch.bfloat16)
        self.assertEqual(wrapper.plan.call_args.kwargs["kv_data_type"], torch.bfloat16)

    def test_mode2_forward_gathers_original_pages_without_unit_cast(self):
        op = _NoReleasePagedAttnOp.__new__(_NoReleasePagedAttnOp)
        op.g_workspace_buffer = torch.empty(0)
        op.dynamic_fp8 = True
        op.direct_scale_fp8 = False
        op.dtype = torch.bfloat16
        op.local_kv_head_num = 2
        op.head_dim_qk = 4
        op.physical_page_size = 32
        op.page_size = 8
        op.subdivision = 4
        op.prefill_cuda_graph_copy_params = None
        op.fmha_params = SimpleNamespace(
            page_indice_d=torch.tensor([9, 3, 77], dtype=torch.int32)
        )
        op._active_source_page_indices = op.fmha_params.page_indice_d[:2]
        op.prefill_wrapper = SimpleNamespace(run=mock.Mock(side_effect=lambda q, _: q))
        q = torch.randn(2, 4, 4, dtype=torch.bfloat16)
        cache = SimpleNamespace(
            kv_cache_base=torch.empty(12, 2, 2, 8, 4, dtype=torch.float8_e4m3fn),
            kv_scale_base=torch.empty(12, 2 * 2 * 8, dtype=torch.float32),
        )

        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "rtp_llm_ops.gather_and_dequantize_fp8_kv_cache",
            create=True,
        ) as gather_mock, mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha."
            "quantize_to_fp8_if_needed",
            side_effect=AssertionError("unit-scale cast must not run"),
        ):
            output = op.forward(q, cache)

        self.assertIs(output, q)
        gather_mock.assert_called_once()
        gather_args = gather_mock.call_args.args
        self.assertIs(gather_args[2], op._active_source_page_indices)
        torch.testing.assert_close(
            gather_args[2], torch.tensor([9, 3], dtype=torch.int32)
        )
        self.assertEqual(gather_args[3].shape, (2, 2, 2, 8, 4))
        self.assertEqual(gather_args[3].dtype, torch.bfloat16)
        self.assertEqual(gather_args[4:], (32, 8, 4))

    def test_eagle3_direct_scale_binding_for_eager_and_graph(self):
        module = (
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha"
        )
        config = self._config()
        config.max_seq_len = 128
        for role in ("is_target_verify", "is_spec_draft_prefill"):
            inputs = SimpleNamespace(
                is_target_verify=False, is_spec_draft_prefill=False, is_cuda_graph=False
            )
            setattr(inputs, role, True)
            wrapper = SimpleNamespace()
            jit_module = object()
            with self.subTest(role=role), mock.patch(
                f"{module}.get_py_flashinfer_workspace_buffer",
                return_value=torch.empty(0),
            ), mock.patch(
                f"{module}.BatchPrefillWithPagedKVCacheWrapper", return_value=wrapper
            ), mock.patch(
                f"{module}._get_dynamic_fp8_jit_module",
                return_value=(jit_module, ["kv_scale"]),
            ) as get_module, mock.patch(
                f"{module}._bind_dynamic_fp8_prefill_module"
            ) as bind_gather:
                op = _NoReleasePagedAttnOp(config, inputs)
                self.assertTrue(op.direct_scale_fp8)
                self.assertIs(wrapper._jit_module, jit_module)
                self.assertEqual(wrapper._jit_additional_tensor_names, ["kv_scale"])
                self.assertEqual(get_module.call_args.args[0][2], torch.float8_e4m3fn)
                bind_gather.assert_not_called()
                inputs.is_cuda_graph = True
                graph_op = _NoReleasePagedAttnOp(config, inputs)
                self.assertTrue(graph_op.direct_scale_fp8)
                self.assertTrue(graph_op.enable_cuda_graph)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_eagle3_graph_replay_updates_pages_and_skips_padding(self):
        from rtp_llm.ops import RopeStyle

        torch.manual_seed(2026)
        harness = BaseAttentionTest()
        harness.device = torch.device("cuda")
        for role, capture_lengths in (
            ("is_target_verify", [6, 6, 6, 6]),
            ("is_spec_draft_prefill", [6, 6, 0, 0]),
        ):
            with self.subTest(role=role):
                config = harness._create_config(
                    head_num=10,
                    head_num_kv=2,
                    size_per_head=64,
                    seq_size_per_block=8,
                    data_type="bf16",
                ).attn_configs
                config.tokens_per_block = 16
                config.fp8_kv_cache_mode = 2
                config.kv_cache_dtype = KvCacheDataType.FP8
                config.gen_num_per_cycle = 5
                config.max_seq_len = 128
                config.need_rope_kv_cache = True
                config.is_causal = True
                config.rope_config.style = RopeStyle.Base
                config.rope_config.dim = 64
                config.rope_config.base = 10000
                table = torch.randperm(32, dtype=torch.int32).reshape(4, 8)
                table = (
                    table[..., None] * 2 + torch.arange(2, dtype=torch.int32)
                ).reshape(4, 16)

                def inputs_for(lengths, prefixes, table, graph):
                    inputs = harness._create_chunked_prefill_attention_inputs(
                        lengths,
                        prefixes,
                        8,
                        dtype=torch.bfloat16,
                        kv_cache_block_id=table,
                        is_cuda_graph=graph,
                    )
                    inputs.sequence_lengths = torch.empty(
                        0, dtype=torch.int32
                    ).pin_memory()
                    setattr(inputs, role, True)
                    return inputs

                cache = LayerKVCache()
                payload = (torch.randn(64, 2, 2, 8, 64, device="cuda") * 8).to(
                    torch.float8_e4m3fn
                )
                scales = torch.rand(64, 32, device="cuda") * 0.02 + 0.001
                cache.kv_cache_base = payload.clone()
                cache.kv_scale_base = scales.clone()
                reference_cache = LayerKVCache()
                reference_cache.kv_cache_base = payload.clone()
                reference_cache.kv_scale_base = scales.clone()
                inputs = inputs_for(capture_lengths, [122] * 4, table, True)
                impl = PyFlashinferPagedPrefillImpl(config, inputs)
                qkv = torch.randn(
                    sum(capture_lengths), 14 * 64, dtype=torch.bfloat16, device="cuda"
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        impl.forward(qkv, cache)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = impl.forward(qkv, cache)
                state = impl.fmha_impl._graph_state
                cases = [
                    ([6, 0, 0, 0], [7, 0, 0, 0]),
                    (capture_lengths, [15, 31, 63, 94]),
                    ([6, 6, 0, 0], [8, 16, 0, 0]),
                ]
                if role == "is_target_verify":
                    cases.append(([6, 6, 6, 0], [63, 64, 65, 0]))
                for index, (lengths, prefixes) in enumerate(cases):
                    live = sum(lengths)
                    live_table = table.roll(index + 1, dims=1).contiguous()
                    inputs = inputs_for(lengths, prefixes, live_table, True)
                    qkv.normal_()
                    # RoPE updates the packed Q/K views in place on both paths.
                    reference_qkv = qkv[:live].clone()
                    cache.kv_cache_base.copy_(payload)
                    cache.kv_scale_base.copy_(scales)
                    reference_cache.kv_cache_base.copy_(payload)
                    reference_cache.kv_scale_base.copy_(scales)
                    with mock.patch(
                        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha._host_i32",
                        side_effect=AssertionError(
                            "graph requires producer-owned host mirrors"
                        ),
                    ):
                        impl.prepare_cuda_graph(inputs)
                    self.assertEqual(impl.fmha_impl._graph_state, state)
                    torch.testing.assert_close(
                        impl.fmha_impl.graph_pages[live:].cpu(),
                        torch.full((qkv.size(0) - live,), -1, dtype=torch.int32),
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    reference = PyFlashinferPagedPrefillImpl(
                        config, inputs_for(lengths, prefixes, live_table, False)
                    )
                    expected = reference.forward(reference_qkv, reference_cache)
                    torch.testing.assert_close(
                        output[:live], expected, rtol=0.03, atol=0.02
                    )
                    # Whole-cache equality detects writes by padded rows, including page 0.
                    torch.testing.assert_close(
                        cache.kv_cache_base.view(torch.uint8),
                        reference_cache.kv_cache_base.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        cache.kv_scale_base,
                        reference_cache.kv_scale_base,
                        rtol=0,
                        atol=0,
                    )
                bad = inputs_for(capture_lengths, [0] * 4, table, True)
                bad.input_lengths = bad.input_lengths.cuda()
                with self.assertRaisesRegex(ValueError, "requires contiguous host"):
                    impl.prepare_cuda_graph(bad)
                with self.assertRaisesRegex(RuntimeError, "cannot be replaced"):
                    impl.fmha_impl.set_params(object())
                impl.fmha_impl.prefill_wrapper._int_workspace_buffer = (
                    impl.fmha_impl.prefill_wrapper._int_workspace_buffer.clone()
                )
                with self.assertRaisesRegex(
                    RuntimeError, "plan or buffer addresses changed"
                ):
                    impl.prepare_cuda_graph(
                        inputs_for(capture_lengths, [0] * 4, table, True)
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_eagle3_direct_scale_qwen3_geometry_causal_and_decode_agree(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferDecodeAttnOp,
        )

        torch.manual_seed(2026)
        harness = BaseAttentionTest()
        harness.device = torch.device("cuda")
        for dtype in (torch.bfloat16, torch.float16):
            for role in ("is_target_verify", "is_spec_draft_prefill"):
                with self.subTest(dtype=dtype, role=role):
                    config = harness._create_config(
                        head_num=40,
                        head_num_kv=8,
                        size_per_head=128,
                        seq_size_per_block=64,
                    ).attn_configs
                    config.dtype = dtype
                    config.fp8_kv_cache_mode = 2
                    config.kv_cache_dtype = KvCacheDataType.FP8
                    config.is_causal = True
                    config.gen_num_per_cycle = 5
                    lengths, prefixes = [6, 17], [61, 113]
                    totals = [p + n for p, n in zip(prefixes, lengths)]
                    inputs = harness._create_chunked_prefill_attention_inputs(
                        lengths, prefixes, 64, dtype=dtype
                    )
                    inputs.sequence_lengths = torch.empty(
                        0, dtype=torch.int32
                    ).pin_memory()
                    setattr(inputs, role, True)
                    pages = int(inputs.kv_cache_kernel_block_id.max()) + 1
                    cache = LayerKVCache()
                    # Deliberately vary each token/head scale across powers of two.
                    cache.kv_cache_base = (
                        torch.randn(pages, 2, 8, 64, 128, device="cuda") * 16
                    ).to(torch.float8_e4m3fn)
                    cache.kv_scale_base = (
                        2.0 ** torch.randint(-7, -2, (pages, 2 * 8 * 64), device="cuda")
                    ).float()
                    query = torch.randn(
                        sum(lengths), 40, 128, dtype=dtype, device="cuda"
                    )
                    op = PyFlashinferPrefillPagedAttnOp(config, inputs)
                    self.assertTrue(op.direct_scale_fp8)
                    self.assertEqual(op.kv_dtype, torch.float8_e4m3fn)
                    op.prepare(inputs)
                    self.assertIsNone(op._active_source_page_indices)
                    with mock.patch(
                        "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha._gather_dynamic_fp8_cache",
                        side_effect=AssertionError(
                            "direct-scale verify must not gather/dequantize"
                        ),
                    ):
                        output = op.forward(query, cache)
                    restored = cache.kv_cache_base.float() * cache.kv_scale_base.view(
                        pages, 2, 8, 64, 1
                    )
                    references = []
                    start = 0
                    for batch_idx, (prefix, length, total) in enumerate(
                        zip(prefixes, lengths, totals)
                    ):
                        ids = (
                            inputs.kv_cache_kernel_block_id[
                                batch_idx, : math.ceil(total / 64)
                            ]
                            .long()
                            .cuda()
                        )
                        keys, values = [
                            restored[ids, kv]
                            .permute(1, 0, 2, 3)
                            .reshape(8, -1, 128)[:, :total]
                            .repeat_interleave(5, dim=0)
                            for kv in (0, 1)
                        ]
                        mask = (
                            torch.arange(total, device="cuda")[None]
                            <= prefix + torch.arange(length, device="cuda")[:, None]
                        )
                        references.append(
                            torch.nn.functional.scaled_dot_product_attention(
                                query[start : start + length].transpose(0, 1).float(),
                                keys,
                                values,
                                attn_mask=mask,
                            ).transpose(0, 1)
                        )
                        start += length
                    torch.testing.assert_close(
                        output.float(), torch.cat(references), rtol=0.04, atol=0.025
                    )
                    decode_inputs = harness._create_attention_inputs_base(
                        2, totals, 64, dtype=dtype
                    )
                    decode = PyFlashinferDecodeAttnOp(config, decode_inputs)
                    decode.prepare(decode_inputs)
                    last_rows = torch.tensor([5, 22], device="cuda")
                    decode_output = decode.forward(
                        query[last_rows], cache, decode.fmha_params
                    )
                    torch.testing.assert_close(
                        output[last_rows], decode_output, rtol=0.03, atol=0.02
                    )
                    # Large negative logits must still honor per-row causal ends.
                    cache.kv_cache_base[:, 0].fill_(-32)
                    cache.kv_scale_base.view(pages, 2, 8, 64)[:, 0].fill_(1)
                    query.fill_(32)
                    output = op.forward(query, cache)
                    self.assertTrue(torch.isfinite(output).all().item())
                    refs = []
                    restored = cache.kv_cache_base.float() * cache.kv_scale_base.view(
                        pages, 2, 8, 64, 1
                    )
                    for b, (prefix, length, total) in enumerate(
                        zip(prefixes, lengths, totals)
                    ):
                        ids = (
                            inputs.kv_cache_kernel_block_id[b, : math.ceil(total / 64)]
                            .long()
                            .cuda()
                        )
                        values = (
                            restored[ids, 1]
                            .permute(1, 0, 2, 3)
                            .reshape(8, -1, 128)[:, :total]
                            .repeat_interleave(5, dim=0)
                        )
                        refs.extend(
                            values[:, : prefix + j + 1].mean(1) for j in range(length)
                        )
                    torch.testing.assert_close(
                        output.float(), torch.stack(refs), rtol=0.04, atol=0.025
                    )

    def test_mode2_end_to_end_attention_matches_base_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        device = torch.device("cuda")
        harness = BaseAttentionTest()
        harness.device = device
        config = harness._create_config(
            head_num=4,
            head_num_kv=2,
            size_per_head=64,
            seq_size_per_block=4,
        )
        config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        config.attn_configs.fp8_kv_cache_mode = 2
        config.attn_configs.is_causal = True
        sequence_lengths = [7, 11]
        inputs = harness._create_prefill_attention_inputs(
            len(sequence_lengths), sequence_lengths, config.seq_size_per_block
        )
        op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, inputs)
        params = op.prepare(inputs)

        total_tokens = sum(sequence_lengths)
        q = torch.randn(total_tokens, 4, 64, device=device, dtype=torch.float16)
        k = torch.randn(total_tokens, 2, 64, device=device, dtype=torch.float16)
        v = torch.randn(total_tokens, 2, 64, device=device, dtype=torch.float16)
        page_count = sum(
            math.ceil(length / config.seq_size_per_block) for length in sequence_lengths
        )
        cache = LayerKVCache()
        cache.kv_cache_base = torch.empty(
            page_count,
            2,
            2,
            config.seq_size_per_block,
            64,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        cache.kv_scale_base = torch.empty(
            page_count,
            2 * 2 * config.seq_size_per_block,
            device=device,
            dtype=torch.float32,
        )
        writer = KVCacheWriteOp(
            num_kv_heads=2,
            head_size=64,
            physical_page_size=config.seq_size_per_block,
            kernel_page_size=config.seq_size_per_block,
            dynamic_mode=True,
        )
        writer.set_params(params)
        writer.forward(k, v, cache)

        scale_view = cache.kv_scale_base.view(
            page_count, 2, 2, config.seq_size_per_block
        )
        restored_k_cache = (
            cache.kv_cache_base[:, 0].float() * scale_view[:, 0].unsqueeze(-1)
        ).to(k.dtype)
        restored_v_cache = (
            cache.kv_cache_base[:, 1].float() * scale_view[:, 1].unsqueeze(-1)
        ).to(v.dtype)
        restored_k = []
        restored_v = []
        for batch_idx, sequence_length in enumerate(sequence_lengths):
            page_ids = inputs.kv_cache_kernel_block_id[
                batch_idx, : math.ceil(sequence_length / config.seq_size_per_block)
            ].tolist()
            restored_k.append(
                restored_k_cache[page_ids]
                .permute(1, 0, 2, 3)
                .reshape(2, -1, 64)
                .permute(1, 0, 2)[:sequence_length]
            )
            restored_v.append(
                restored_v_cache[page_ids]
                .permute(1, 0, 2, 3)
                .reshape(2, -1, 64)
                .permute(1, 0, 2)[:sequence_length]
            )

        output = op.forward(q, cache)
        reference = compute_flashinfer_prefill_reference(
            q,
            torch.cat(restored_k),
            torch.cat(restored_v),
            inputs.cu_seqlens_device,
            causal=True,
        )
        torch.testing.assert_close(output, reference, rtol=0.06, atol=0.04)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_eagle3_verify_rejection_overwrites_payload_and_scales(self):
        device = torch.device("cuda")
        dtype = torch.bfloat16
        harness = BaseAttentionTest()
        harness.device = device
        torch.manual_seed(42)

        def quantized(tensor):
            scale = tensor.float().abs().amax(-1) / 448.0
            scale = torch.where(scale == 0, 1.0, scale)
            payload = (
                (tensor.float() * scale.reciprocal().unsqueeze(-1))
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            return payload.float() * scale.unsqueeze(-1), scale

        for subdivision in (1, 2):
            for hybrid in (False, True):
                with self.subTest(subdivision=subdivision, hybrid=hybrid):
                    physical_size, page_size = 8, 8 // subdivision
                    config = harness._create_config(
                        head_num=4,
                        head_num_kv=2,
                        size_per_head=64,
                        seq_size_per_block=page_size,
                        data_type="bf16",
                    )
                    config.attn_configs.tokens_per_block = physical_size
                    config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
                    config.attn_configs.fp8_kv_cache_mode = 2
                    config.attn_configs.gen_num_per_cycle = 5
                    config.attn_configs.is_causal = True
                    # Noncontiguous physical pages; all three streams can cross pages.
                    physical_ids = torch.tensor(
                        [[11, 1, 9, 3], [6, 4, 10, 0], [7, 2, 8, 5]], dtype=torch.int32
                    )
                    block_table = (
                        physical_ids[..., None] * subdivision
                        + torch.arange(subdivision)
                    ).reshape(3, -1)
                    page_count = 12 * subdivision
                    cache = LayerKVCache()
                    cache.kv_cache_base = torch.zeros(
                        page_count,
                        2,
                        2,
                        page_size,
                        64,
                        dtype=torch.float8_e4m3fn,
                        device=device,
                    )
                    cache.kv_scale_base = torch.ones(
                        page_count,
                        2 * 2 * page_size,
                        dtype=torch.float32,
                        device=device,
                    )
                    writer = KVCacheWriteOp(
                        num_kv_heads=2,
                        head_size=64,
                        physical_page_size=physical_size,
                        kernel_page_size=page_size,
                        dynamic_mode=True,
                    )

                    def inputs_for(prefixes, lengths):
                        inputs = harness._create_chunked_prefill_attention_inputs(
                            lengths,
                            prefixes,
                            page_size,
                            dtype=dtype,
                            kv_cache_block_id=block_table,
                        )
                        inputs.sequence_lengths = torch.empty(
                            0, dtype=torch.int32
                        ).pin_memory()
                        inputs.is_target_verify = True
                        return inputs

                    prefixes = [7, 10, 15]
                    keys = [
                        torch.randn(n, 2, 64, dtype=dtype, device=device)
                        for n in prefixes
                    ]
                    values = [torch.randn_like(k) for k in keys]
                    bootstrap_inputs = inputs_for([0, 0, 0], prefixes)
                    bootstrap = PyFlashinferPrefillPagedAttnOp(
                        config.attn_configs, bootstrap_inputs
                    )
                    writer.set_params(bootstrap.prepare(bootstrap_inputs))
                    writer.forward(torch.cat(keys), torch.cat(values), cache)
                    op = None
                    for round_id, magnitude in enumerate((8.0, 0.125)):
                        inputs = inputs_for(prefixes, [6, 6, 6])
                        if op is None:
                            op = (
                                PyFlashinferHybridPrefillAttnOp(
                                    config.attn_configs, inputs
                                )
                                if hybrid
                                else PyFlashinferPrefillPagedAttnOp(
                                    config.attn_configs, inputs
                                )
                            )
                        writer.set_params(op.prepare(inputs))
                        new_k = (
                            torch.randn(18, 2, 64, dtype=dtype, device=device)
                            * magnitude
                        )
                        new_v = torch.randn_like(new_k) * magnitude
                        q = (
                            torch.randn(18, 4, 64, dtype=dtype, device=device)
                            / magnitude
                        )
                        if hybrid:
                            output = op.forward(q, new_k, new_v, cache, writer)
                        else:
                            writer.forward(new_k, new_v, cache)
                            output = op.forward(q, cache)
                        expected_outputs = []
                        scale_view = cache.kv_scale_base.view(
                            page_count, 2, 2, page_size
                        )
                        for batch_idx, prefix in enumerate(prefixes):
                            k_chunk = new_k[batch_idx * 6 : (batch_idx + 1) * 6]
                            v_chunk = new_v[batch_idx * 6 : (batch_idx + 1) * 6]
                            keys[batch_idx] = torch.cat(
                                (keys[batch_idx][:prefix], k_chunk)
                            )
                            values[batch_idx] = torch.cat(
                                (values[batch_idx][:prefix], v_chunk)
                            )
                            positions = torch.arange(prefix + 6, device=device)
                            pages = block_table[batch_idx].to(device)[
                                positions // page_size
                            ]
                            offsets = positions % page_size
                            restored = []
                            for kv, full in enumerate(
                                (keys[batch_idx], values[batch_idx])
                            ):
                                expected, scales = quantized(full)
                                actual_scales = scale_view[pages, kv, :, offsets]
                                actual = cache.kv_cache_base[
                                    pages, kv, :, offsets
                                ].float() * actual_scales.unsqueeze(-1)
                                torch.testing.assert_close(
                                    actual_scales, scales, rtol=1e-6, atol=1e-7
                                )
                                torch.testing.assert_close(
                                    actual, expected, rtol=1e-5, atol=1e-5
                                )
                                # Direct-scale verify applies FP32 scales in the
                                # attention kernel, without a BF16 KV round-trip.
                                restored.append(
                                    expected.to(dtype) if hybrid else expected
                                )
                            k_ref, v_ref = restored
                            if hybrid:
                                k_ref = torch.cat((k_ref[:prefix], k_chunk))
                                v_ref = torch.cat((v_ref[:prefix], v_chunk))
                            mask = (
                                positions[None, :]
                                <= prefix + torch.arange(6, device=device)[:, None]
                            )
                            reference = (
                                torch.nn.functional.scaled_dot_product_attention(
                                    q[batch_idx * 6 : (batch_idx + 1) * 6]
                                    .transpose(0, 1)
                                    .float(),
                                    k_ref.repeat_interleave(2, dim=1)
                                    .transpose(0, 1)
                                    .float(),
                                    v_ref.repeat_interleave(2, dim=1)
                                    .transpose(0, 1)
                                    .float(),
                                    attn_mask=mask,
                                ).transpose(0, 1)
                            )
                            expected_outputs.append(reference)
                        self.assertTrue(torch.isfinite(output).all().item())
                        torch.testing.assert_close(
                            output.float(),
                            torch.cat(expected_outputs),
                            rtol=0.03,
                            atol=0.05,
                        )
                        # Keep the mandatory target token plus 0, 2, or all 5 proposals.
                        # The next round must overwrite rejected payload AND scales.
                        if round_id == 0:
                            prefixes = [
                                p + accepted for p, accepted in zip(prefixes, (1, 3, 6))
                            ]

    def test_mode2_large_negative_logits_preserve_constant_values_with_subdivision(
        self,
    ):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        device = torch.device("cuda")
        harness = BaseAttentionTest()
        harness.device = device
        sequence_lengths = [33, 129, 4807]
        values = [1.0, 2.0, 3.0]
        physical_page_size = 64
        for dtype in (torch.float16, torch.bfloat16):
            for subdivision in (1, 2, 4):
                with self.subTest(dtype=dtype, subdivision=subdivision):
                    kernel_page_size = physical_page_size // subdivision
                    config = harness._create_config(
                        head_num=28,
                        head_num_kv=4,
                        size_per_head=128,
                        seq_size_per_block=kernel_page_size,
                    )
                    config.attn_configs.dtype = dtype
                    config.attn_configs.tokens_per_block = physical_page_size
                    config.attn_configs.kv_cache_dtype = KvCacheDataType.FP8
                    config.attn_configs.fp8_kv_cache_mode = 2
                    config.attn_configs.is_causal = True
                    inputs = harness._create_prefill_attention_inputs(
                        len(sequence_lengths),
                        sequence_lengths,
                        kernel_page_size,
                        dtype=dtype,
                    )

                    physical_pages = sum(
                        math.ceil(length / physical_page_size)
                        for length in sequence_lengths
                    )
                    page_ids = list(range(1, 2 * physical_pages, 2))
                    page_ids = page_ids[::2] + page_ids[1::2]
                    physical_blocks = harness._create_kv_cache_block_ids(
                        len(sequence_lengths), sequence_lengths, physical_page_size
                    )
                    block_table = torch.zeros_like(inputs.kv_cache_kernel_block_id)
                    page_offset = 0
                    for batch_idx, length in enumerate(sequence_lengths):
                        num_physical_pages = math.ceil(length / physical_page_size)
                        physical_blocks[batch_idx, :num_physical_pages] = torch.tensor(
                            page_ids[page_offset : page_offset + num_physical_pages],
                            dtype=torch.int32,
                        )
                        num_kernel_pages = math.ceil(length / kernel_page_size)
                        for logical_page in range(num_kernel_pages):
                            block_table[batch_idx, logical_page] = (
                                physical_blocks[batch_idx, logical_page // subdivision]
                                * subdivision
                                + logical_page % subdivision
                            )
                        page_offset += num_physical_pages
                    inputs.kv_cache_block_id = physical_blocks
                    inputs.kv_cache_block_id_device = physical_blocks.to(device)
                    inputs.kv_cache_kernel_block_id = block_table
                    inputs.kv_cache_kernel_block_id_device = block_table.to(device)

                    op = PyFlashinferPrefillPagedAttnOp(config.attn_configs, inputs)
                    params = op.prepare(inputs)
                    total_tokens = sum(sequence_lengths)
                    q = torch.full(
                        (total_tokens, 28, 128), 32, dtype=dtype, device=device
                    )
                    k = torch.full(
                        (total_tokens, 4, 128), -32, dtype=dtype, device=device
                    )
                    v = torch.cat(
                        [
                            torch.full(
                                (length, 4, 128), value, dtype=dtype, device=device
                            )
                            for length, value in zip(sequence_lengths, values)
                        ]
                    )
                    kernel_page_count = 2 * physical_pages * subdivision
                    cache = LayerKVCache()
                    cache.kv_cache_base = torch.zeros(
                        kernel_page_count,
                        2,
                        4,
                        kernel_page_size,
                        128,
                        dtype=torch.float8_e4m3fn,
                        device=device,
                    )
                    cache.kv_scale_base = torch.ones(
                        kernel_page_count,
                        2 * 4 * kernel_page_size,
                        dtype=torch.float32,
                        device=device,
                    )
                    writer = KVCacheWriteOp(
                        num_kv_heads=4,
                        head_size=128,
                        physical_page_size=physical_page_size,
                        kernel_page_size=kernel_page_size,
                        dynamic_mode=True,
                    )
                    writer.set_params(params)
                    writer.forward(k, v, cache)
                    output = op.forward(q, cache)
                    self.assertEqual(output.shape, q.shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(torch.isfinite(output).all().item())

                    scale_view = cache.kv_scale_base.view(
                        kernel_page_count, 2, 4, kernel_page_size
                    )
                    restored_cache = cache.kv_cache_base.float() * scale_view.unsqueeze(
                        -1
                    )
                    token_offset = 0
                    for batch_idx, (length, value) in enumerate(
                        zip(sequence_lengths, values)
                    ):
                        with self.subTest(length=length):
                            positions = torch.arange(length, device=device)
                            pages = inputs.kv_cache_kernel_block_id_device[
                                batch_idx, positions // kernel_page_size
                            ]
                            offsets = positions % kernel_page_size
                            for kv, expected_value in enumerate((-32.0, value)):
                                restored = restored_cache[pages, kv, :, offsets]
                                torch.testing.assert_close(
                                    restored,
                                    torch.full_like(restored, expected_value),
                                    rtol=1e-5,
                                    atol=1e-5,
                                )
                            actual = output[token_offset : token_offset + length]
                            torch.testing.assert_close(
                                actual,
                                torch.full_like(actual, value),
                                rtol=1e-2,
                                atol=1e-2,
                            )
                        token_offset += length


if __name__ == "__main__":
    unittest.main()
