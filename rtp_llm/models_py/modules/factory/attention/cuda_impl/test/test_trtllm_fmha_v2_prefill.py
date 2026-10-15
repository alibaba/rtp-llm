"""Tests for TRTLLMFMHAv2PrefillOp (non-padded mode)

Tests standard prefill without prefix cache via trtllm_fmha_v2_prefill.
MHA uses PACKED_QKV while GQA/MQA uses CONTIGUOUS_Q_KV.
This mode is used for dynamic batch processing.
"""

import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _is_fmha_impl_disabled,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl import trt
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    apply_base_rope_to_qkv_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTLLMFMHAv2TestBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.trt_test_utils import (
    compute_pytorch_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
    FlashInferNativePrefillImpl,
    FlashInferTRTLLMFMHAv2PrefillImpl,
    TRTLLMFMHAv2PrefillOp,
)
from rtp_llm.models_py.utils.arch import is_sm12x, is_sm90
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta


class TestFlashInferNativePrefillSupport(unittest.TestCase):
    @staticmethod
    def _config(**overrides) -> AttentionConfigs:
        config = AttentionConfigs()
        config.head_num = 16
        config.kv_head_num = 2
        config.size_per_head = 256
        config.tokens_per_block = 64
        config.kernel_tokens_per_block = 64
        config.dtype = torch.bfloat16
        config.kv_cache_dtype = KvCacheDataType.BASE
        config.use_mla = False
        config.is_causal = True
        for name, value in overrides.items():
            setattr(config, name, value)
        return config

    @staticmethod
    def _inputs(prefix_lengths=(0,), is_cuda_graph=False) -> PyAttentionInputs:
        inputs = PyAttentionInputs()
        inputs.prefix_lengths = torch.tensor(prefix_lengths, dtype=torch.int32)
        inputs.is_cuda_graph = is_cuda_graph
        return inputs

    def test_supports_validated_sm12x_shape(self):
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt.is_sm12x",
            return_value=True,
        ):
            self.assertTrue(
                FlashInferNativePrefillImpl.support(self._config(), self._inputs())
            )

    def test_rejects_unvalidated_configurations(self):
        cases = {
            "fp8_kv": ({"kv_cache_dtype": KvCacheDataType.FP8}, self._inputs()),
            "prefix": ({}, self._inputs((32,))),
            "cuda_graph": ({}, self._inputs(is_cuda_graph=True)),
            "fp16": ({"dtype": torch.float16}, self._inputs()),
            "fp32": ({"dtype": torch.float32}, self._inputs()),
            "head_dim": ({"size_per_head": 128}, self._inputs()),
            "mha": ({"head_num": 2, "kv_head_num": 2}, self._inputs()),
            "invalid_gqa": ({"head_num": 15, "kv_head_num": 2}, self._inputs()),
            "noncausal": ({"is_causal": False}, self._inputs()),
            "mla": ({"use_mla": True}, self._inputs()),
        }
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt.is_sm12x",
            return_value=True,
        ):
            for name, (overrides, inputs) in cases.items():
                with self.subTest(name=name):
                    self.assertFalse(
                        FlashInferNativePrefillImpl.support(
                            self._config(**overrides), inputs
                        )
                    )

    def test_rejects_sm90(self):
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt.is_sm12x",
            return_value=False,
        ):
            self.assertFalse(
                FlashInferNativePrefillImpl.support(self._config(), self._inputs())
            )

    def test_cuda_graph_falls_back_before_construction(self):
        config = self._config()
        inputs = self._inputs(is_cuda_graph=True)
        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt.is_sm12x",
            return_value=True,
        ), patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt._supports_trtllm_fmha_v2",
            return_value=True,
        ):
            self.assertFalse(FlashInferNativePrefillImpl.support(config, inputs))
            self.assertTrue(FlashInferTRTLLMFMHAv2PrefillImpl.support(config, inputs))

    def test_global_flashinfer_native_switch_disables_impl(self):
        config = FMHAConfig()
        config.disable_flashinfer_native = True
        self.assertTrue(_is_fmha_impl_disabled("FlashInferNativePrefillImpl", config))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTRTLLMFMHAv2PrefillQuantization(unittest.TestCase):
    def test_unit_scale_inputs_and_base_dtype_are_preserved(self):
        module = "rtp_llm.models_py.modules.factory.attention.cuda_impl.trt"
        for dtype in (torch.float16, torch.bfloat16):
            for heads, kv_heads in ((4, 4), (28, 4)):
                for fp8 in (False, True):
                    for strided in (False, True):
                        with self.subTest(
                            dtype=dtype, heads=heads, fp8=fp8, strided=strided
                        ):
                            config = AttentionConfigs()
                            config.head_num = heads
                            config.kv_head_num = kv_heads
                            config.size_per_head = 128
                            config.dtype = dtype
                            config.kv_cache_dtype = (
                                KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
                            )
                            config.fp8_kv_cache_mode = int(fp8)
                            op = TRTLLMFMHAv2PrefillOp(config)
                            op.attention_type = "mha" if heads == kv_heads else "gqa"
                            width = (heads + 2 * kv_heads) * 128
                            values = torch.tensor(
                                [
                                    0,
                                    1,
                                    -1,
                                    100,
                                    -100,
                                    448,
                                    -448,
                                    500,
                                    -500,
                                    1000,
                                    -1000,
                                    1e-4,
                                ],
                                dtype=dtype,
                                device="cuda",
                            )
                            qkv = values.repeat((4 * width + 11) // 12)[
                                : 4 * width
                            ].view(4, width)
                            if strided:
                                qkv = qkv.repeat_interleave(2, dim=1)[:, ::2]
                            original = qkv.clone()
                            expected = (
                                qkv.float().clamp(-448, 448).to(torch.float8_e4m3fn)
                                if fp8
                                else qkv
                            )
                            params = SimpleNamespace(
                                seq_lens=torch.tensor(
                                    [4], device="cuda", dtype=torch.int32
                                ),
                                max_q_len=4,
                                max_kv_len=4,
                                batch_size=1,
                                cu_seqlens=torch.tensor(
                                    [0, 4], device="cuda", dtype=torch.int32
                                ),
                            )
                            with patch(
                                f"{module}.trtllm_fmha_v2_prefill",
                                return_value=torch.zeros(
                                    4, heads, 128, device="cuda", dtype=dtype
                                ),
                            ) as fmha:
                                output = op.forward(qkv, None, params)
                            kwargs = fmha.call_args.kwargs
                            actual = kwargs["qkv"]
                            if heads == kv_heads:
                                self.assertEqual(kwargs["input_layout"], "PACKED_QKV")
                                self.assertEqual(actual.dtype, expected.dtype)
                                actual = actual.reshape(4, width)
                            else:
                                self.assertEqual(
                                    kwargs["input_layout"], "CONTIGUOUS_Q_KV"
                                )
                                for part in actual:
                                    self.assertEqual(part.dtype, expected.dtype)
                                actual = torch.cat(
                                    [part.float().reshape(4, -1) for part in actual],
                                    dim=1,
                                )
                            self.assertTrue(torch.isfinite(actual.float()).all().item())
                            torch.testing.assert_close(
                                actual.float(), expected.float(), rtol=0, atol=0
                            )
                            torch.testing.assert_close(qkv, original, rtol=0, atol=0)
                            self.assertEqual(output.dtype, dtype)
                            self.assertEqual(output.shape, (4, heads * 128))
                            self.assertEqual(kwargs["out_dtype"], dtype)
                            self.assertEqual(kwargs["bmm2_scale"], 1.0)


class TestTRTLLMWorkspaceReuse(unittest.TestCase):
    def test_workspace_reuse_after_inference_mode(self):
        with mock.patch.object(trt, "_g_trtllm_fmha_v2_workspace_pool", []):
            with torch.inference_mode():
                workspace = trt._get_trtllm_fmha_v2_workspace("cpu", 128)
                workspace.fill_(1)
                trt._release_trtllm_fmha_v2_workspace(workspace)
            reused = trt._get_trtllm_fmha_v2_workspace("cpu", 128)
            self.assertEqual(reused.data_ptr(), workspace.data_ptr())
            self.assertEqual(reused.count_nonzero().item(), 0)
            reused.fill_(2)
            self.assertTrue(torch.all(reused == 2).item())


class TestTRTLLMFMHAv2PrefillOpBF16(TRTLLMFMHAv2TestBase):
    """Test suite for TRTLLMFMHAv2PrefillOp in non-padded mode

    TRTLLMFMHAv2PrefillOp:
    - Standard prefill without prefix cache
    - Uses PACKED_QKV for MHA and CONTIGUOUS_Q_KV for GQA/MQA
    - prefix_lengths must be 0 or None
    - Only processes new input_lengths tokens
    - Non-padded mode: variable sequence lengths (no padding)
    """

    kv_cache_dtype = KvCacheDataType.BASE

    def _create_config(self, *args, **kwargs):
        kwargs["data_type"] = "bf16"
        attn_configs = super()._create_config(*args, **kwargs)
        attn_configs.kv_cache_dtype = self.kv_cache_dtype
        return attn_configs

    def run_correctness_test(self, *args, **kwargs):
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            kwargs.setdefault("rtol", 4e-2)
            kwargs.setdefault("atol", 4e-2)
            kwargs.setdefault("max_mismatch_rate", 1e-5)
        return super().run_correctness_test(*args, **kwargs)

    def test_basic(self):
        """Test basic prefill with single sequence"""
        print("\n=== Test FMHAv2Prefill: Basic ===", flush=True)

        batch_size = 1
        input_lengths = [128]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_short_sequence(self):
        batch_size = 1
        input_lengths = [8]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        self.run_correctness_test(
            attn_op=TRTLLMFMHAv2PrefillOp(attn_configs),
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_batch(self):
        """Test prefill with multiple sequences of variable lengths"""
        print("\n=== Test FMHAv2Prefill: Batch ===", flush=True)

        batch_size = 4
        input_lengths = [64, 128, 256, 512]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_non_causal(self):
        batch_size = 2
        input_lengths = [32, 48]
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
            prefix_lengths=None,
        )

        self.run_correctness_test(
            attn_op=TRTLLMFMHAv2PrefillOp(attn_configs),
            op_name="TRTLLMFMHAv2PrefillOp non-causal",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=tokens_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_cuda_graph(self):
        test_cases = [
            ("packed_qkv", 8, [48, 32], [40, 40]),
            ("contiguous_q_kv", 2, [32, 24], [24, 32]),
            ("replay_smaller_batch", 2, [32, 8, 8, 8], [24, 32, 0, 0]),
            ("capture_empty_batch", 2, [56, 0, 0, 0], [24, 32, 0, 0]),
        ]
        for layout, head_num_kv, capture_lengths, replay_lengths in test_cases:
            with self.subTest(layout=layout):
                head_num = 8
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
                    len(capture_lengths), capture_lengths, tokens_per_block
                )
                replay_inputs = self._create_prefill_attention_inputs(
                    len(replay_lengths), replay_lengths, tokens_per_block
                )
                capture_inputs.is_cuda_graph = True
                replay_inputs.is_cuda_graph = True

                static_qkv = self._create_qkv_tensor(
                    sum(capture_lengths),
                    head_num,
                    head_num_kv,
                    head_dim,
                    dtype=attn_configs.dtype,
                )
                replay_qkv = torch.randn_like(static_qkv)
                capture_inputs.dtype = get_typemeta(static_qkv)
                replay_inputs.dtype = get_typemeta(replay_qkv)
                expect_impl = FlashInferTRTLLMFMHAv2PrefillImpl(
                    attn_configs, replay_inputs
                )
                expect_output = expect_impl.forward(replay_qkv, None).clone()

                graph_impl = FlashInferTRTLLMFMHAv2PrefillImpl(
                    attn_configs, capture_inputs
                )
                self.assertTrue(graph_impl.support_cuda_graph())

                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    graph_impl.forward(static_qkv, None)
                torch.cuda.current_stream().wait_stream(warmup_stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = graph_impl.forward(static_qkv, None)

                static_qkv.copy_(replay_qkv)
                capture_inputs.input_lengths.copy_(replay_inputs.input_lengths)
                capture_inputs.cu_seqlens_device.copy_(replay_inputs.cu_seqlens_device)
                graph_impl.prepare_cuda_graph(capture_inputs)
                graph.replay()
                torch.cuda.synchronize()

                torch.testing.assert_close(graph_output, expect_output, rtol=0, atol=0)

    def test_prefill_cuda_graph_rope_kv_and_dynamic_batch(self):
        """Gate GQA and packed-MHA prefill graphs on SM90 and SM12x."""
        if self.kv_cache_dtype != KvCacheDataType.BASE:
            self.skipTest("generation-prefill CUDA graph requires BF16 KV cache")
        if not (is_sm90() or is_sm12x()):
            self.skipTest(
                "generation-prefill CUDA graph is allowlisted on SM90 and SM12x"
            )

        for head_num_kv in (2, 8):
            for token_capacity in (64, 256):
                with self.subTest(
                    head_num_kv=head_num_kv, token_capacity=token_capacity
                ):
                    self._run_prefill_cuda_graph_rope_kv_and_dynamic_batch(
                        head_num_kv, token_capacity
                    )

    def _run_prefill_cuda_graph_rope_kv_and_dynamic_batch(
        self, head_num_kv: int, token_capacity: int
    ):
        """Capture five fixed slots and replay dynamic request layouts.

        Capture five fixed sequence slots (four real slots plus one sentinel),
        then replay two real requests with a padded sentinel. This covers real
        RoPE, BF16 KV writes, dynamic zero-length slots, block-table refresh,
        and isolation of real KV pages in one graph. The multi-page case uses
        the production sentinel layout: every padding page aliases block 0.
        """
        max_requests = 4
        capture_lengths = [0] * max_requests + [token_capacity]
        head_num = 8
        head_dim = 64
        tokens_per_block = 64
        real_block_ids = [1, 2, 3, 4]
        scratch_block_id = 0 if token_capacity > tokens_per_block else 6
        total_blocks = 8

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=tokens_per_block,
        )
        attn_configs.rope_config.style = RopeStyle.Base
        attn_configs.rope_config.dim = head_dim
        attn_configs.rope_config.base = 10000
        attn_configs.rope_config.max_pos = token_capacity
        attn_configs.max_seq_len = token_capacity

        capture_inputs = self._create_prefill_attention_inputs(
            len(capture_lengths), capture_lengths, tokens_per_block
        )
        capture_inputs.is_cuda_graph = True
        capture_inputs.kv_cache_kernel_block_id.zero_()
        capture_inputs.kv_cache_kernel_block_id_device.zero_()
        capture_inputs.kv_cache_kernel_block_id[-1, 0] = scratch_block_id
        capture_inputs.kv_cache_kernel_block_id_device[-1, 0] = scratch_block_id

        static_qkv = self._create_qkv_tensor(
            token_capacity,
            head_num,
            head_num_kv,
            head_dim,
            dtype=attn_configs.dtype,
        )
        capture_inputs.dtype = get_typemeta(static_qkv)

        graph_cache, _, _ = self._create_kv_cache(
            total_blocks,
            tokens_per_block,
            head_num_kv,
            head_dim,
            dtype=attn_configs.dtype,
        )
        graph_impl = FlashInferTRTLLMFMHAv2PrefillImpl(attn_configs, capture_inputs)
        self.assertTrue(graph_impl.supports_generation_prefill_cuda_graph())

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            graph_impl.forward(static_qkv, graph_cache)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = graph_impl.forward(static_qkv, graph_cache)

        replay_layouts = ([24, 32], [64], [32, 32], [16, 16, 16, 16])
        for real_lengths in replay_layouts:
            with self.subTest(real_lengths=real_lengths):
                real_tokens = sum(real_lengths)
                sentinel_length = token_capacity - real_tokens
                replay_lengths = (
                    real_lengths
                    + [0] * (max_requests - len(real_lengths))
                    + [sentinel_length]
                )
                replay_qkv = torch.randn_like(static_qkv)
                static_qkv.copy_(replay_qkv)
                capture_inputs.input_lengths.copy_(
                    torch.tensor(replay_lengths, dtype=torch.int32, device=self.device)
                )

                cu_seqlens = [0]
                padding_offsets = []
                packed_offset = 0
                for slot, length in enumerate(replay_lengths):
                    padding_offsets.extend(
                        [slot * token_capacity - packed_offset] * length
                    )
                    packed_offset += length
                    cu_seqlens.append(packed_offset)
                replay_cu_seqlens = torch.tensor(
                    cu_seqlens, dtype=torch.int32, device=self.device
                )
                capture_inputs.cu_seqlens_device.copy_(replay_cu_seqlens)
                capture_inputs.cu_kv_seqlens_device.copy_(replay_cu_seqlens)
                capture_inputs.padding_offset.copy_(
                    torch.tensor(padding_offsets, dtype=torch.int32, device=self.device)
                )
                capture_inputs.kv_cache_kernel_block_id.zero_()
                capture_inputs.kv_cache_kernel_block_id_device.zero_()
                for row, block_id in enumerate(real_block_ids[: len(real_lengths)]):
                    capture_inputs.kv_cache_kernel_block_id[row, 0] = block_id
                    capture_inputs.kv_cache_kernel_block_id_device[row, 0] = block_id
                capture_inputs.kv_cache_kernel_block_id[-1, 0] = scratch_block_id
                capture_inputs.kv_cache_kernel_block_id_device[-1, 0] = scratch_block_id
                graph_impl.prepare_cuda_graph(capture_inputs)
                graph_cache.kv_cache_base.zero_()

                eager_inputs = self._create_prefill_attention_inputs(
                    len(real_lengths), real_lengths, tokens_per_block
                )
                eager_inputs.dtype = get_typemeta(replay_qkv)
                for row, block_id in enumerate(real_block_ids[: len(real_lengths)]):
                    eager_inputs.kv_cache_kernel_block_id[row, 0] = block_id
                    eager_inputs.kv_cache_kernel_block_id_device[row, 0] = block_id
                eager_cache, _, _ = self._create_kv_cache(
                    total_blocks,
                    tokens_per_block,
                    head_num_kv,
                    head_dim,
                    dtype=attn_configs.dtype,
                )
                eager_cache.kv_cache_base.zero_()
                eager_impl = FlashInferTRTLLMFMHAv2PrefillImpl(
                    attn_configs, eager_inputs
                )
                eager_output = eager_impl.forward(
                    replay_qkv[:real_tokens], eager_cache
                ).clone()

                graph.replay()
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    graph_output[:real_tokens], eager_output, rtol=5e-3, atol=5e-3
                )
                for block_id, length in zip(real_block_ids, real_lengths):
                    torch.testing.assert_close(
                        graph_cache.kv_cache_base[block_id, :, :, :length, :],
                        eager_cache.kv_cache_base[block_id, :, :, :length, :],
                        rtol=5e-3,
                        atol=5e-3,
                    )
                # Only real request pages and the sentinel page may be written.
                # In the multi-page case, do not compare sentinel bytes: those
                # writes intentionally alias, and no real request reads them.
                touched = set(real_block_ids[: len(real_lengths)]) | {scratch_block_id}
                for block_id in set(range(total_blocks)) - touched:
                    self.assertEqual(
                        graph_cache.kv_cache_base[block_id].count_nonzero().item(), 0
                    )
                scratch_nonzero = (
                    graph_cache.kv_cache_base[scratch_block_id].count_nonzero().item()
                )
                if sentinel_length:
                    self.assertGreater(scratch_nonzero, 0)
                else:
                    self.assertEqual(scratch_nonzero, 0)

    def test_gqa(self):
        """Test prefill with grouped query attention"""
        print("\n=== Test FMHAv2Prefill: GQA ===", flush=True)

        batch_size = 2
        input_lengths = [256, 512]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)
        attn_op.prepare(attn_inputs)
        self.assertEqual(attn_op.attention_type, "gqa")

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_mha(self):
        """Test prefill with packed multi-head attention input."""
        batch_size = 2
        input_lengths = [128, 256]
        head_num = 8
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )
        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)
        attn_op.prepare(attn_inputs)
        self.assertEqual(attn_op.attention_type, "mha")

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp MHA",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def test_long_sequence(self):
        """Test prefill with long sequences"""
        print("\n=== Test FMHAv2Prefill: Long Sequence ===", flush=True)

        batch_size = 2
        input_lengths = [1024, 2048]
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
            batch_size, input_lengths, seq_size_per_block, prefix_lengths=None
        )

        attn_op = TRTLLMFMHAv2PrefillOp(attn_configs)

        self.run_correctness_test(
            attn_op=attn_op,
            op_name="TRTLLMFMHAv2PrefillOp",
            batch_size=batch_size,
            input_lengths=input_lengths,
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            prefix_lengths=None,
            use_padded=False,
        )

    def _run_impl_rope_correctness(self, rope_style):
        input_lengths = [32, 47]
        head_num = 8
        head_num_kv = 2
        head_dim = 128
        tokens_per_block = 64
        rope_base = 10000

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
            prefix_lengths=None,
        )
        # Fused RoPE derives per-sequence positions from packed-to-padded offsets.
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
        attn_inputs.kv_cache_kernel_block_id = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id_device = torch.empty(
            0, dtype=torch.int32, device=self.device
        )

        qkv = self._create_qkv_tensor(
            sum(input_lengths),
            head_num,
            head_num_kv,
            head_dim,
            dtype=attn_configs.dtype,
        )
        attn_inputs.dtype = get_typemeta(qkv)
        expected_qkv = qkv
        if rope_style != RopeStyle.No:
            expected_qkv = apply_base_rope_to_qkv_reference(
                qkv,
                input_lengths,
                head_num,
                head_num_kv,
                head_dim,
                rope_base,
            )
        if self.kv_cache_dtype == KvCacheDataType.FP8:
            expected_qkv = expected_qkv.to(torch.float8_e4m3fn).to(qkv.dtype)
        expected = compute_pytorch_prefill_reference(
            expected_qkv,
            input_lengths,
            head_num,
            head_num_kv,
            head_dim,
        )

        impl = FlashInferTRTLLMFMHAv2PrefillImpl(attn_configs, attn_inputs)
        actual = impl.forward(qkv.clone(), None)

        if self.kv_cache_dtype == KvCacheDataType.FP8:
            torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)
        else:
            torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)

    def test_impl_without_rope_matches_torch(self):
        self._run_impl_rope_correctness(RopeStyle.No)

    def test_impl_with_rope_matches_torch(self):
        self._run_impl_rope_correctness(RopeStyle.Base)

    @unittest.skipUnless(is_sm12x(), "native FA2 rollout is SM12x-only")
    def test_native_fa2_matches_trt_with_custom_scale(self):
        input_lengths = [32, 47]
        head_num = 16
        head_num_kv = 2
        head_dim = 256
        tokens_per_block = 64

        for rope_style in (RopeStyle.No, RopeStyle.Base):
            with self.subTest(rope_style=rope_style):
                attn_configs = self._create_config(
                    head_num=head_num,
                    head_num_kv=head_num_kv,
                    size_per_head=head_dim,
                    seq_size_per_block=tokens_per_block,
                )
                attn_configs.is_causal = True
                attn_configs.q_scaling = 2.0
                attn_configs.softmax_extra_scale = 0.75
                attn_configs.rope_config.style = rope_style
                attn_configs.rope_config.dim = 64
                attn_configs.rope_config.base = 10000
                attn_configs.rope_config.max_pos = 128
                attn_configs.max_seq_len = 128

                attn_inputs = self._create_prefill_attention_inputs(
                    len(input_lengths),
                    input_lengths,
                    tokens_per_block,
                    prefix_lengths=None,
                )
                attn_inputs.sequence_lengths = attn_inputs.input_lengths
                attn_inputs.is_cuda_graph = False

                qkv = self._create_qkv_tensor(
                    sum(input_lengths),
                    head_num,
                    head_num_kv,
                    head_dim,
                    dtype=attn_configs.dtype,
                )
                attn_inputs.dtype = get_typemeta(qkv)

                trt_impl = FlashInferTRTLLMFMHAv2PrefillImpl(attn_configs, attn_inputs)
                native_impl = FlashInferNativePrefillImpl(attn_configs, attn_inputs)
                expected_scale = (
                    attn_configs.softmax_extra_scale
                    / attn_configs.q_scaling
                    * head_dim**-0.5
                )
                self.assertAlmostEqual(native_impl.fmha_impl.sm_scale, expected_scale)

                trt_output = trt_impl.forward(qkv.clone(), None)
                native_output = native_impl.forward(qkv.clone(), None)
                torch.testing.assert_close(
                    native_output, trt_output, rtol=5e-3, atol=2e-2
                )


class TestTRTLLMFMHAv2PrefillOpFP8(TestTRTLLMFMHAv2PrefillOpBF16):
    kv_cache_dtype = KvCacheDataType.FP8

    def setUp(self):
        if torch.cuda.is_available() and is_sm12x():
            self.skipTest("TRT-LLM FMHA v2 does not support FP8 on SM12x")
        super().setUp()


if __name__ == "__main__":
    unittest.main()
