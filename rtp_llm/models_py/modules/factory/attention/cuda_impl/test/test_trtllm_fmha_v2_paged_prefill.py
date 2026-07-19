"""Tests for TRTLLMFMHAv2PagedPrefillOp (non-padded mode)

Tests paged prefill with prefix cache via trtllm_fmha_v2_prefill Q_PAGED_KV_HND layout.
This mode is used when there's existing KV cache (prefix/prompt caching).
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTLLMFMHAv2TestBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.trt_test_utils import (
    apply_base_rope_to_qkv_reference,
    compute_pytorch_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
    FlashInferTRTLLMFMHAv2PagedPrefillImpl,
    TRTLLMFMHAv2PagedPrefillOp,
)
from rtp_llm.models_py.utils.arch import is_sm12x
from rtp_llm.ops import KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import get_typemeta


class TestTRTLLMFMHAv2PagedPrefillOpBF16(TRTLLMFMHAv2TestBase):
    """Test suite for TRTLLMFMHAv2PagedPrefillOp in non-padded mode

    TRTLLMFMHAv2PagedPrefillOp:
    - Paged prefill with prefix cache (prompt caching)
    - Uses Q_PAGED_KV_HND layout via FlashInfer trtllm_fmha_v2_prefill
    - prefix_lengths must be > 0 (already cached KV)
    - Processes new input_lengths tokens with existing prefix_lengths cache
    - Total KV length = prefix_lengths + input_lengths
    """

    kv_cache_dtype = KvCacheDataType.BASE

    def _create_config(self, *args, **kwargs):
        kwargs["data_type"] = "bf16"
        attn_configs = super()._create_config(*args, **kwargs)
        attn_configs.kv_cache_dtype = self.kv_cache_dtype
        return attn_configs

    def run_correctness_test(self, *args, **kwargs):
        kwargs.setdefault("use_packed_kv_cache", True)
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            kwargs.setdefault("rtol", 4e-2)
            kwargs.setdefault("atol", 4e-2)
            kwargs.setdefault("max_mismatch_rate", 1e-5)
        return super().run_correctness_test(*args, **kwargs)

    def test_basic(self):
        """Test basic paged prefill with single sequence and prefix cache"""
        print("\n=== Test FMHAv2PagedPrefill: Basic ===", flush=True)

        batch_size = 1
        input_lengths = [128]
        prefix_lengths = [64]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTLLMFMHAv2PagedPrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def test_short_sequence(self):
        batch_size = 1
        input_lengths = [8]
        prefix_lengths = [8]
        head_num = 8
        head_num_kv = 2
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )
        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        self.run_correctness_test(
            attn_op=TRTLLMFMHAv2PagedPrefillOp(attn_configs),
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def test_batch(self):
        """Test paged prefill with multiple sequences and different prefix lengths"""
        print("\n=== Test FMHAv2PagedPrefill: Batch ===", flush=True)

        batch_size = 4
        input_lengths = [32, 64, 128, 256]
        prefix_lengths = [32, 64, 128, 256]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTLLMFMHAv2PagedPrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def test_non_causal(self):
        batch_size = 2
        input_lengths = [24, 32]
        prefix_lengths = [16, 24]
        head_num = 8
        head_num_kv = 2
        head_dim = 128
        tokens_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=tokens_per_block,
        )
        attn_configs.is_causal = False
        attn_inputs = self._create_prefill_attention_inputs(
            batch_size,
            input_lengths,
            tokens_per_block,
            prefix_lengths=prefix_lengths,
        )

        self.run_correctness_test(
            attn_op=TRTLLMFMHAv2PagedPrefillOp(attn_configs),
            op_name="TRTLLMFMHAv2PagedPrefillOp non-causal",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=tokens_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def test_cuda_graph(self):
        test_cases = [
            ("aligned", [24, 32], [48, 32], [32, 24], [32, 48]),
            ("compact", [32, 24], [48, 32], [24, 32], [48, 32]),
            (
                "compact_smaller_batch",
                [32, 8, 8, 8],
                [32, 32, 32, 32],
                [24, 32, 0, 0],
                [32, 32, 0, 0],
            ),
        ]
        for (
            layout,
            capture_input_lengths,
            capture_prefix_lengths,
            replay_input_lengths,
            replay_prefix_lengths,
        ) in test_cases:
            with self.subTest(layout=layout):
                head_num = 8
                head_num_kv = 2
                head_dim = 128
                tokens_per_block = 64

                attn_configs = self._create_config(
                    head_num=head_num,
                    head_num_kv=head_num_kv,
                    size_per_head=head_dim,
                    seq_size_per_block=tokens_per_block,
                )
                attn_configs.rope_config.style = RopeStyle.No
                capture_inputs = self._create_prefill_attention_inputs(
                    len(capture_input_lengths),
                    capture_input_lengths,
                    tokens_per_block,
                    prefix_lengths=capture_prefix_lengths,
                )
                replay_inputs = self._create_prefill_attention_inputs(
                    len(replay_input_lengths),
                    replay_input_lengths,
                    tokens_per_block,
                    prefix_lengths=replay_prefix_lengths,
                )
                capture_inputs.is_cuda_graph = True
                replay_inputs.is_cuda_graph = True

                static_qkv = self._create_qkv_tensor(
                    sum(capture_input_lengths),
                    head_num,
                    head_num_kv,
                    head_dim,
                    dtype=attn_configs.dtype,
                )
                replay_qkv = torch.randn_like(static_qkv)
                capture_inputs.dtype = get_typemeta(static_qkv)
                replay_inputs.dtype = get_typemeta(replay_qkv)
                cache_dtype = (
                    torch.float8_e4m3fn
                    if self.kv_cache_dtype == KvCacheDataType.FP8
                    else attn_configs.dtype
                )
                total_blocks = self._calculate_total_blocks(
                    [
                        prefix + length
                        for prefix, length in zip(
                            capture_prefix_lengths, capture_input_lengths
                        )
                    ],
                    tokens_per_block,
                )
                expect_cache, _, _ = self._create_kv_cache(
                    total_blocks,
                    tokens_per_block,
                    head_num_kv,
                    head_dim,
                    dtype=cache_dtype,
                )
                cache_snapshot = expect_cache.kv_cache_base.clone()
                graph_cache, _, _ = self._create_kv_cache(
                    total_blocks,
                    tokens_per_block,
                    head_num_kv,
                    head_dim,
                    dtype=cache_dtype,
                )
                graph_cache.kv_cache_base.copy_(cache_snapshot)

                expect_impl = FlashInferTRTLLMFMHAv2PagedPrefillImpl(
                    attn_configs, replay_inputs
                )
                expect_output = expect_impl.forward(replay_qkv, expect_cache, 0).clone()

                graph_impl = FlashInferTRTLLMFMHAv2PagedPrefillImpl(
                    attn_configs, capture_inputs
                )
                self.assertTrue(graph_impl.support_cuda_graph())

                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    graph_impl.forward(static_qkv, graph_cache, 0)
                torch.cuda.current_stream().wait_stream(warmup_stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = graph_impl.forward(static_qkv, graph_cache, 0)

                static_qkv.copy_(replay_qkv)
                graph_cache.kv_cache_base.copy_(cache_snapshot)
                capture_inputs.input_lengths.copy_(replay_inputs.input_lengths)
                capture_inputs.prefix_lengths.copy_(replay_inputs.prefix_lengths)
                capture_inputs.cu_seqlens_device.copy_(replay_inputs.cu_seqlens_device)
                capture_inputs.cu_kv_seqlens_device.copy_(
                    replay_inputs.cu_kv_seqlens_device
                )
                capture_inputs.padding_offset.copy_(replay_inputs.padding_offset)
                capture_inputs.kv_cache_kernel_block_id_device.copy_(
                    replay_inputs.kv_cache_kernel_block_id_device
                )
                graph_impl.prepare_cuda_graph(capture_inputs)
                graph.replay()
                torch.cuda.synchronize()

                torch.testing.assert_close(graph_output, expect_output, rtol=0, atol=0)

    def test_gqa(self):
        """Test paged prefill with grouped query attention and prefix cache"""
        print("\n=== Test FMHAv2PagedPrefill: GQA ===", flush=True)

        batch_size = 2
        input_lengths = [128, 256]
        prefix_lengths = [128, 256]
        head_num = 32
        head_num_kv = 4
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTLLMFMHAv2PagedPrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def test_long_sequence(self):
        """Test paged prefill with long sequences and long prefix cache"""
        print("\n=== Test FMHAv2PagedPrefill: Long Sequence ===", flush=True)

        batch_size = 2
        input_lengths = [512, 1024]
        prefix_lengths = [512, 1024]
        head_num = 32
        head_num_kv = 8
        size_per_head = 128
        seq_size_per_block = 64

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_prefill_attention_inputs(
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=prefix_lengths
        )

        attn_op = TRTLLMFMHAv2PagedPrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PagedPrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=prefix_lengths,
            use_padded=False,
        )

    def _run_impl_rope_correctness(self, rope_style):
        input_lengths = [24, 32]
        prefix_lengths = [48, 17]
        head_num = 8
        head_num_kv = 2
        head_dim = 128
        tokens_per_block = 64
        rope_base = 10000
        total_lengths = [
            prefix_length + input_length
            for prefix_length, input_length in zip(prefix_lengths, input_lengths)
        ]

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=tokens_per_block,
        )
        attn_configs.rope_config.style = rope_style
        attn_configs.rope_config.dim = head_dim
        attn_configs.rope_config.base = rope_base
        attn_configs.rope_config.max_pos = 128
        attn_configs.max_seq_len = 128
        attn_inputs = self._create_prefill_attention_inputs(
            len(input_lengths),
            input_lengths,
            tokens_per_block,
            prefix_lengths=prefix_lengths,
        )
        max_input_length = max(input_lengths)
        packed_offset = 0
        padding_offsets = []
        for batch_idx, input_length in enumerate(input_lengths):
            padding_offsets.extend(
                [batch_idx * max_input_length - packed_offset] * input_length
            )
            packed_offset += input_length
        attn_inputs.padding_offset = torch.tensor(
            padding_offsets, dtype=torch.int32, device=self.device
        )

        qkv_full = self._create_qkv_tensor(
            sum(total_lengths),
            head_num,
            head_num_kv,
            head_dim,
            dtype=attn_configs.dtype,
        )
        attn_inputs.dtype = get_typemeta(qkv_full)

        expected_qkv_full = qkv_full
        if rope_style != RopeStyle.No:
            expected_qkv_full = apply_base_rope_to_qkv_reference(
                qkv_full,
                total_lengths,
                head_num,
                head_num_kv,
                head_dim,
                rope_base,
            )
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            expected_qkv_full = expected_qkv_full.to(torch.float8_e4m3fn).to(
                qkv_full.dtype
            )

        qkv_new = []
        prefix_qkv = []
        offset = 0
        for prefix_length, input_length in zip(prefix_lengths, input_lengths):
            total_length = prefix_length + input_length
            prefix_qkv.append(expected_qkv_full[offset : offset + prefix_length])
            qkv_new.append(
                qkv_full[offset + prefix_length : offset + prefix_length + input_length]
            )
            offset += total_length
        qkv_new = torch.cat(qkv_new, dim=0)
        prefix_qkv = torch.cat(prefix_qkv, dim=0)

        cache_dtype = (
            torch.float8_e4m3fn
            if self.kv_cache_dtype == KvCacheDataType.FP8
            else attn_configs.dtype
        )
        total_blocks = self._calculate_total_blocks(total_lengths, tokens_per_block)
        kv_cache, _, _ = self._create_kv_cache(
            total_blocks,
            tokens_per_block,
            head_num_kv,
            head_dim,
            dtype=cache_dtype,
        )
        kv_cache.kv_cache_base.zero_()
        self._write_kv_cache(prefix_qkv, attn_inputs, prefix_lengths, kv_cache)
        kv_cache.kv_cache_base = kv_cache.kv_cache_base.flatten(1)

        impl = FlashInferTRTLLMFMHAv2PagedPrefillImpl(attn_configs, attn_inputs)
        actual = impl.forward(qkv_new.clone(), kv_cache, layer_idx=0)

        expected_full = compute_pytorch_prefill_reference(
            expected_qkv_full,
            total_lengths,
            head_num,
            head_num_kv,
            head_dim,
        )
        expected = []
        offset = 0
        for prefix_length, input_length in zip(prefix_lengths, input_lengths):
            expected.append(
                expected_full[
                    offset + prefix_length : offset + prefix_length + input_length
                ]
            )
            offset += prefix_length + input_length
        expected = torch.cat(expected, dim=0)

        if self.kv_cache_dtype == KvCacheDataType.FP8:
            torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)
        else:
            torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)

    def test_impl_without_rope_matches_torch(self):
        self._run_impl_rope_correctness(RopeStyle.No)

    def test_impl_with_rope_matches_torch(self):
        self._run_impl_rope_correctness(RopeStyle.Base)


class TestTRTLLMFMHAv2PagedPrefillOpFP8(TestTRTLLMFMHAv2PagedPrefillOpBF16):
    kv_cache_dtype = KvCacheDataType.FP8

    def setUp(self):
        if torch.cuda.is_available() and is_sm12x():
            self.skipTest("TRT-LLM FMHA v2 does not support FP8 on SM12x")
        super().setUp()


if __name__ == "__main__":
    unittest.main()
