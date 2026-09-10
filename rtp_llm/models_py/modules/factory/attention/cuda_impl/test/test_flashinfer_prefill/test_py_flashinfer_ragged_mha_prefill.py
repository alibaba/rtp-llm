import logging
import math
import unittest
from typing import List
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillAttnOp,
    PyFlashinferPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    FP8_CACHE_DTYPES,
    BaseAttentionTest,
    make_fp8_unit_scale,
)
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferPrefillAttnOp(BaseAttentionTest):
    """Test suite for PyFlashinferPrefillAttnOp with correctness verification"""

    def setUp(self):
        """Set up test fixtures"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - this test requires CUDA")

        # Call parent setUp for common initialization
        super().setUp()

    def _create_kv_cache(
        self,
        total_blocks: int,
        seq_size_per_block: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        """Helper to create empty KV cache for prefill"""
        kv_cache = LayerKVCache()

        # Create combined KV cache with shape [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        # Initialize with zeros since we'll write to it during prefill
        kv_cache_combined = torch.zeros(
            total_blocks,
            2,  # K and V
            num_kv_heads,
            seq_size_per_block,
            head_dim,
            dtype=dtype,
            device=self.device,
        )

        kv_cache.kv_cache_base = kv_cache_combined
        if dtype in FP8_CACHE_DTYPES:
            kv_cache.kv_scale_base = make_fp8_unit_scale(
                total_blocks, num_kv_heads, seq_size_per_block, self.device
            )

        # Extract separate K and V for reference computation
        k_cache = kv_cache_combined[:, 0, :, :, :]
        v_cache = kv_cache_combined[:, 1, :, :, :]

        return kv_cache, k_cache, v_cache

    def _test_prefill_correctness(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        seq_size_per_block: int = 64,
        causal: bool = True,
        with_kv_cache_block_ids: bool = True,
    ):
        """Test prefill correctness by comparing with flashinfer reference implementation"""

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )
        config.attn_configs.is_causal = causal

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size,
            sequence_lengths,
            config.seq_size_per_block,
            with_kv_cache_block_ids=with_kv_cache_block_ids,
        )

        # Create PyFlashinferPrefillAttnOp instance
        attn_op = PyFlashinferPrefillAttnOp(config.attn_configs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())

        # Check support
        if not attn_op.support(attn_inputs):
            raise RuntimeError(
                "PyFlashinferPrefillAttnOp does not support this configuration"
            )

        # Prepare params
        params = attn_op.prepare(attn_inputs)

        # Create QKV input in the format expected by PyFlashinferPrefillAttnOp
        # Input shape: [total_tokens, hidden_size_q + hidden_size_k + hidden_size_v]
        # where hidden_size_q = head_dim * head_num, hidden_size_k = hidden_size_v = head_dim * kv_head_num
        total_tokens = sum(sequence_lengths)

        hidden_size_q = config.size_per_head * config.head_num
        hidden_size_k = config.size_per_head * config.head_num_kv
        hidden_size_v = config.size_per_head * config.head_num_kv

        # Create QKV tensor in flattened format [total_tokens, hidden_size_q + hidden_size_k + hidden_size_v]
        qkv = torch.randn(
            total_tokens,
            hidden_size_q + hidden_size_k + hidden_size_v,
            dtype=torch.float16,
            device=self.device,
        )

        # Extract Q, K, V for reference computation
        # Split and reshape to [total_tokens, num_heads, head_dim]
        q_flat, k_flat, v_flat = torch.split(
            qkv,
            [hidden_size_q, hidden_size_k, hidden_size_v],
            dim=-1,
        )
        q = q_flat.reshape(total_tokens, config.head_num, config.size_per_head)
        k = k_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)
        v = v_flat.reshape(total_tokens, config.head_num_kv, config.size_per_head)

        kv_cache = None
        if with_kv_cache_block_ids:
            total_blocks = self._calculate_total_blocks(
                sequence_lengths, config.seq_size_per_block
            )
            kv_cache, _, _ = self._create_kv_cache(
                total_blocks,
                config.seq_size_per_block,
                config.head_num_kv,
                config.size_per_head,
                dtype=self.cache_dtype(config.attn_configs),
            )

        # Forward pass through PyFlashinferPrefillAttnOp
        output = attn_op.forward(q, k, v, kv_cache)

        # Compute reference outputs using flashinfer's single_prefill_with_kv_cache (with round-trip)
        ref_output = compute_flashinfer_prefill_reference(
            q.to(attn_op.q_dtype).to(q.dtype),
            k.to(attn_op.kv_dtype).to(k.dtype),
            v.to(attn_op.kv_dtype).to(v.dtype),
            attn_inputs.cu_seqlens_device,
            causal=causal,
        )

        # Compare outputs
        self._assert_output_close(
            output,
            ref_output,
            name=f"Prefill output (batch={batch_size}, seq_lens={sequence_lengths})",
        )

        logging.info(
            f"✓ Test passed: batch_size={batch_size}, sequence_lengths={sequence_lengths}"
        )

    def test_single_batch_prefill(self):
        """Test prefill for a single batch"""
        logging.info("\n=== Testing single batch prefill ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=1,
                sequence_lengths=[128],
                size_per_head=head_dim,
            )

    def test_non_causal_prefill(self):
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[64, 96],
            head_num=8,
            head_num_kv=2,
            size_per_head=64,
            causal=False,
        )

    def test_non_causal_prefill_without_kv_block_table(self):
        """Test encoder-only ragged prefill without a paged KV block table."""
        self._test_prefill_correctness(
            batch_size=2,
            sequence_lengths=[10, 20],
            head_num=8,
            head_num_kv=8,
            size_per_head=128,
            causal=False,
            with_kv_cache_block_ids=False,
        )

    def test_multi_batch_prefill(self):
        """Test prefill for multiple batches with varying sequence lengths"""
        logging.info("\n=== Testing multi-batch prefill ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=4,
                sequence_lengths=[64, 128, 256, 512],
                size_per_head=head_dim,
            )

    def test_different_block_sizes(self):
        """Test with different block sizes"""
        logging.info("\n=== Testing different block sizes ===")
        for head_dim in [128, 256]:
            for block_size in [16, 32, 64, 128]:
                logging.info(
                    f"\n--- Testing head_dim={head_dim}, block_size={block_size} ---"
                )
                self._test_prefill_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    size_per_head=head_dim,
                    seq_size_per_block=block_size,
                )

    def test_different_head_configurations(self):
        """Test with different head configurations (GQA)"""
        logging.info("\n=== Testing different head configurations ===")
        test_cases = [
            (32, 32, "MHA"),  # MHA: head_num == head_num_kv
            (32, 8, "GQA"),  # GQA: head_num > head_num_kv (group_size=4)
            (32, 4, "GQA-4"),  # GQA with group_size=8
        ]

        for head_dim in [128, 256]:
            for head_num, head_num_kv, name in test_cases:
                logging.info(
                    f"\n--- Testing {name}: head_num={head_num}, head_num_kv={head_num_kv}, head_dim={head_dim} ---"
                )
                self._test_prefill_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    head_num=head_num,
                    head_num_kv=head_num_kv,
                    size_per_head=head_dim,
                )

    def test_edge_case_sequence_lengths(self):
        """Test edge cases with sequence lengths"""
        logging.info("\n=== Testing edge case sequence lengths ===")

        for head_dim in [128, 256]:
            # Sequence length exactly equal to block size
            logging.info(
                f"\n--- Testing seq_len == block_size, head_dim={head_dim} ---"
            )
            self._test_prefill_correctness(
                batch_size=1,
                sequence_lengths=[64],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Sequence length slightly more than block size
            logging.info(f"\n--- Testing seq_len > block_size, head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=1,
                sequence_lengths=[65],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Very short sequences
            logging.info(f"\n--- Testing short sequences, head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=2,
                sequence_lengths=[10, 20],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

    def test_variable_sequence_lengths(self):
        """Test prefill with highly variable sequence lengths

        PyFlashinferPrefillAttnOp uses BatchPrefillWithRaggedKVCacheWrapper which
        handles ragged tensors efficiently. This test verifies it works correctly
        with sequences of very different lengths.

        Note: This implementation uses ragged tensor format (via cu_seqlens), not
        padded format. Padding would be wasteful for highly variable lengths.
        """
        logging.info("\n=== Testing variable sequence lengths (ragged format) ===")

        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing varied lengths, head_dim={head_dim} ---")

            # Test with very different sequence lengths
            self._test_prefill_correctness(
                batch_size=4,
                sequence_lengths=[32, 96, 200, 512],  # Highly variable
                size_per_head=head_dim,
            )

            # Test with extreme variation
            logging.info(f"\n--- Testing extreme variation, head_dim={head_dim} ---")
            self._test_prefill_correctness(
                batch_size=3,
                sequence_lengths=[16, 128, 1024],  # 64x difference
                size_per_head=head_dim,
            )


class TestPyFlashinferPrefillAttnOpFP8(TestPyFlashinferPrefillAttnOp):
    kv_cache_dtype = KvCacheDataType.FP8
    rtol = 4e-2
    atol = 4e-2
    max_mismatch_rate = 5e-5

    def test_mode1_interleaved_mrope_rotates_qk_and_writes_unit_scale_fp8(self):
        sequence_lengths = [5, 3]
        page_size = 4
        config = self._create_config(
            head_num=4,
            head_num_kv=2,
            size_per_head=256,
            seq_size_per_block=page_size,
        )
        self._enable_qwen35_mrope_mode1(config)
        config.attn_configs.is_causal = True
        inputs = self._create_prefill_attention_inputs(
            len(sequence_lengths), sequence_lengths, page_size
        )
        position_ids = self._add_qwen35_mrope_inputs(inputs, sequence_lengths)
        self.assertTrue(PyFlashinferPrefillImpl.support(config.attn_configs, inputs))

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

        total_blocks = sum(math.ceil(length / page_size) for length in sequence_lengths)
        kv_cache, _, _ = self._create_kv_cache(
            total_blocks,
            page_size,
            config.head_num_kv,
            config.size_per_head,
            dtype=torch.float8_e4m3fn,
        )
        impl = PyFlashinferPrefillImpl(
            config.attn_configs, inputs, config.parallelism_config
        )
        self.assertIsNone(impl.rope_impl)
        self.assertIsNotNone(impl.fused_mrope_impl)

        with mock.patch.object(
            impl.fused_mrope_impl,
            "forward",
            wraps=impl.fused_mrope_impl.forward,
        ) as fused_forward, mock.patch.object(
            impl.kv_cache_write_op,
            "forward",
            wraps=impl.kv_cache_write_op.forward,
        ) as cache_write, mock.patch.object(
            impl.fmha_impl,
            "forward",
            wraps=impl.fmha_impl.forward,
        ) as fmha_forward:
            output = impl.forward(qkv.clone(), kv_cache)

        fused_forward.assert_called_once()
        self.assertIsNone(fused_forward.call_args.args[1])
        cache_write.assert_called_once()
        fmha_forward.assert_called_once()
        query, written_k, written_v = fmha_forward.call_args.args[:3]
        self.assertEqual(written_k.dtype, torch.float8_e4m3fn)
        self.assertEqual(written_v.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(query, expected_q, rtol=1e-2, atol=1e-2)
        reference = compute_flashinfer_prefill_reference(
            expected_q.to(impl.fmha_impl.q_dtype).to(expected_q.dtype),
            written_k.to(expected_k.dtype),
            written_v.to(v.dtype),
            inputs.cu_seqlens_device,
            causal=True,
        )
        self._assert_output_close(output, reference, name="MRoPE ragged output")
        torch.testing.assert_close(
            written_k.float(),
            expected_k.to(torch.float8_e4m3fn).float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            written_v.float(), v.to(torch.float8_e4m3fn).float(), rtol=0, atol=0
        )

        token_offset = 0
        for batch_idx, sequence_length in enumerate(sequence_lengths):
            for position in range(sequence_length):
                page_id = int(
                    inputs.kv_cache_kernel_block_id[
                        batch_idx, position // page_size
                    ].item()
                )
                page_offset = position % page_size
                torch.testing.assert_close(
                    kv_cache.kv_cache_base[page_id, 0, :, page_offset].float(),
                    expected_k[token_offset + position].to(torch.float8_e4m3fn).float(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    kv_cache.kv_cache_base[page_id, 1, :, page_offset].float(),
                    v[token_offset + position].to(torch.float8_e4m3fn).float(),
                    rtol=0,
                    atol=0,
                )
            token_offset += sequence_length
        self.assertTrue(torch.all(kv_cache.kv_scale_base == 1).item())

    def test_mode1_mrope_support_rejects_invalid_variants(self):
        config = self._create_config(
            head_num=4,
            head_num_kv=2,
            size_per_head=64,
            seq_size_per_block=4,
        )
        self._enable_qwen35_mrope_mode1(config)
        inputs = self._create_prefill_attention_inputs(1, [2], 4)
        self._add_qwen35_mrope_inputs(inputs, [2])

        cases = (
            ("non_interleaved", "mrope_interleaved", False),
            ("invalid_index_factor", "index_factor", 1),
            ("invalid_sections", "mrope_dim3", 9),
        )
        for name, field, value in cases:
            with self.subTest(name=name):
                setattr(config.attn_configs.rope_config, field, value)
                self.assertFalse(
                    PyFlashinferPrefillImpl.support(config.attn_configs, inputs)
                )
                self._enable_qwen35_mrope_mode1(config)

        inputs.combo_position_ids = torch.empty(
            0, dtype=torch.int32, device=self.device
        )
        self.assertFalse(PyFlashinferPrefillImpl.support(config.attn_configs, inputs))
        self._add_qwen35_mrope_inputs(inputs, [2])
        config.attn_configs.fp8_kv_cache_mode = 2
        self.assertFalse(PyFlashinferPrefillImpl.support(config.attn_configs, inputs))

    def test_out_of_fp8_range_kv(self):
        config = self._create_config(
            head_num=1,
            head_num_kv=1,
            size_per_head=64,
            seq_size_per_block=16,
        )
        config.attn_configs.is_causal = False
        attn_inputs = self._create_prefill_attention_inputs(
            batch_size=1,
            sequence_lengths=[2],
            seq_size_per_block=config.seq_size_per_block,
            with_kv_cache_block_ids=False,
        )

        attn_op = PyFlashinferPrefillAttnOp(config.attn_configs)
        attn_op.set_params(rtp_llm_ops.FlashInferMlaAttnParams())
        attn_op.prepare(attn_inputs)
        self.assertEqual(attn_op.kv_dtype, torch.float8_e4m3fn)

        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        q = torch.full(
            (2, config.head_num, config.size_per_head),
            0.125,
            dtype=torch.float16,
            device=self.device,
        )
        k = torch.empty(
            (2, config.head_num_kv, config.size_per_head),
            dtype=torch.float16,
            device=self.device,
        )
        v = torch.empty_like(k)
        k[0].fill_(2 * fp8_max)
        k[1].fill_(-2 * fp8_max)
        v.copy_(k)

        output = attn_op.forward(q, k, v, kv_cache=None)

        saturated_k = k.clamp(-fp8_max, fp8_max).to(attn_op.kv_dtype).to(k.dtype)
        saturated_v = v.clamp(-fp8_max, fp8_max).to(attn_op.kv_dtype).to(v.dtype)
        ref_output = compute_flashinfer_prefill_reference(
            q.to(attn_op.q_dtype).to(q.dtype),
            saturated_k,
            saturated_v,
            attn_inputs.cu_seqlens_device,
            causal=False,
        )

        self.assertTrue(torch.isfinite(output).all().item())
        self._assert_output_close(
            output,
            ref_output,
            name="FP8 ragged prefill output with saturated K/V",
        )


if __name__ == "__main__":
    unittest.main()
