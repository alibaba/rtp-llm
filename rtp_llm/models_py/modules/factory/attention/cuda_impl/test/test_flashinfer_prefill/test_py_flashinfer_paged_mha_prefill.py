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

        batch_size = 1
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
            batch_size, prefix_lengths, input_lengths, config.seq_size_per_block
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


if __name__ == "__main__":
    unittest.main()
