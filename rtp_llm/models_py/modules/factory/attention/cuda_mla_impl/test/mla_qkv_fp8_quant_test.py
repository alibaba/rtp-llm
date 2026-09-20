"""Prefill Q/K/V quantization: numerical and real CUDA launch contracts."""

import unittest
from itertools import product

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    MlaFlashMLAPrefillOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import (
    quantize_fp8,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_qkv_fp8_quant import (
    quantize_qkv_fp8,
)
from rtp_llm.ops import KvCacheDataType


def cuda_kernel_names(function):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        function()
        torch.cuda.synchronize()
    return [
        event.name
        for event in trace.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not any(word in event.name.lower() for word in ("memcpy", "memset"))
    ]


class MlaQkvFp8QuantTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "requires CUDA; no passing by skip")
        torch.manual_seed(20260920)

    @staticmethod
    def _qkv(nq, nk, heads=12, packed=True):
        q = torch.randn(nq, heads, 192, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(nk, heads, 320, device="cuda", dtype=torch.bfloat16)
        k, v = kv[..., :192], kv[..., 192:]
        return (q, k, v) if packed else (q, k.contiguous(), v.contiguous())

    def assert_quantized_equal(self, sources, scale=1.0, actual=None):
        if actual is None:
            actual = quantize_qkv_fp8(*sources, scale)
        for i, (source, got) in enumerate(zip(sources, actual)):
            expected = quantize_fp8(source, scale if i == 0 else 1.0)
            self.assertEqual(got.shape, source.shape)
            self.assertEqual(got.dtype, torch.float8_e4m3fn)
            self.assertTrue(got.is_contiguous())
            self.assertTrue(
                torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
            )
        return actual

    def test_production_prefill_shapes_are_byte_exact(self):
        # Wrong head/feature addressing or sharing Q's scale with K/V breaks
        # the independent original-quantizer oracle, including real KV views.
        for heads, rows, packed in product(
            (12, 96), (32, 128, 512, 2048, 8192, 16384), (False, True)
        ):
            with self.subTest(heads=heads, rows=rows, packed=packed):
                self.assert_quantized_equal(self._qkv(rows, rows, heads, packed), 0.5)

    def test_ragged_empty_and_tail_inputs(self):
        # Q and KV counts need not match during prefix/chunked prefill. Empty
        # operands must not prevent the remaining operands from being written.
        for nq, nk in (
            (0, 0),
            (0, 17),
            (17, 0),
            (1, 1),
            (31, 129),
            (513, 2049),
            (2049, 513),
        ):
            with self.subTest(nq=nq, nk=nk):
                self.assert_quantized_equal(self._qkv(nq, nk), 0.3)

    def test_noncontiguous_layouts_and_supported_dtypes(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for layout in ("head_major", "stepped", "broadcast", "legacy_v"):
                with self.subTest(dtype=dtype, layout=layout):
                    q, k, v = (
                        torch.randn(3, rows, dim, device="cuda", dtype=dtype).transpose(
                            0, 1
                        )
                        for rows, dim in ((31, 65), (129, 65), (129, 33))
                    )
                    if layout == "stepped":
                        q, k, v = q[..., ::2], k[..., ::2], v[..., ::2]
                    elif layout == "broadcast":
                        q, k, v = (x[:, :1].expand(-1, 3, -1) for x in (q, k, v))
                    elif layout == "legacy_v":
                        q, k = q.contiguous(), k.contiguous()
                        v = torch.randn(129, 3, 66, device="cuda", dtype=dtype)[
                            ..., 33:
                        ]
                    self.assert_quantized_equal((q, k, v), 0.3)

    def test_rounding_signed_zero_nan_and_saturation(self):
        magnitudes = (0.0, 2**-10, 2**-9, 1, 1.0625, 1.1875, 432, 448, 449, 10000)
        values = [value * sign for value, sign in product(magnitudes, (1, -1))]
        values += [float("nan"), float("inf"), -float("inf"), 2**-10]
        # Aligned length exercises vectorized IO with NaNs and signed zeros.
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            source = torch.tensor(values, device="cuda", dtype=dtype).view(1, 1, -1)
            original = source.clone()
            for scale in (0.25, 0.3, 1.0, 2.0, 10.0):
                actual = self.assert_quantized_equal((source, source, source), scale)
                for i, got in enumerate(actual):
                    # A second, independently expressed oracle for finite
                    # values. NaN payload/sign is checked against the old kernel.
                    expected = (
                        (source.float() * (1.0 / scale if i == 0 else 1.0))
                        .clamp(-448, 448)
                        .to(torch.float8_e4m3fn)
                    )
                    finite = ~torch.isnan(source)
                    self.assertTrue(
                        torch.equal(
                            got.view(torch.uint8)[finite],
                            expected.view(torch.uint8)[finite],
                        )
                    )
            torch.testing.assert_close(source, original, rtol=0, atol=0, equal_nan=True)

    def test_aligned_lengths_do_not_assume_aligned_inputs(self):
        # A multiple-of-eight element count only proves mask alignment, not
        # pointer/stride alignment. Also exercise independently typed operands.
        for heads, dim in ((3, 64), (8, 65)):
            inputs = []
            for rows, dtype in zip(
                (31, 129, 17), (torch.bfloat16, torch.float16, torch.float32)
            ):
                storage = torch.randn(
                    rows * heads * (dim + 1) + 1, device="cuda", dtype=dtype
                )
                inputs.append(
                    storage.as_strided(
                        (rows, heads, dim), (heads * (dim + 1), dim + 1, 1), 1
                    )
                )
            with self.subTest(heads=heads, dim=dim):
                self.assert_quantized_equal(inputs, 0.3)

    def test_graph_replay_reads_new_strided_values(self):
        q = torch.randn(12, 31, 192, device="cuda", dtype=torch.bfloat16).transpose(
            0, 1
        )
        kv = torch.randn(129, 12, 320, device="cuda", dtype=torch.bfloat16)
        k, v = kv[..., :192], kv[..., 192:]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                quantize_qkv_fp8(q, k, v, 0.5)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = quantize_qkv_fp8(q, k, v, 0.5)
        pointers = [x.data_ptr() for x in actual]
        for value in (0.25, -17.0, 999.0):
            q.fill_(value)
            k.fill_(value * 2)
            v.fill_(-value)
            graph.replay()
            self.assert_quantized_equal((q, k, v), 0.5, actual)
            self.assertEqual([x.data_ptr() for x in actual], pointers)

    def test_large_token_and_head_strides_use_int64_addresses(self):
        # Only touch small valid regions in an ~8 GiB allocation. An int32
        # regression reads our zero sentinels, never an illegal CUDA address.
        storage = torch.empty(2**32 + 4096, device="cuda", dtype=torch.bfloat16)
        storage[:4096].zero_()
        for head_major in (False, True):
            inputs = []
            for i, dim in enumerate((192, 192, 128)):
                shape = (1, 3, dim) if head_major else (3, 1, dim)
                stride = (dim, 2**30 + 256, 1) if head_major else (2**30 + 256, dim, 1)
                view = storage.as_strided(shape, stride, 2**31 + i * 512)
                view.copy_(torch.randn(shape, device="cuda", dtype=torch.bfloat16))
                inputs.append(view)
            for _ in range(2):
                self.assert_quantized_equal(inputs, 0.5)

    def test_prefill_attention_uses_one_quantization_launch(self):
        # Replacing the fused call with three old quantizers must fail this
        # contract even if all three numerical outputs remain identical.
        op = MlaFlashMLAPrefillOp(
            12,
            512,
            64,
            128,
            128,
            128,
            128,
            1.0,
            True,
            [{}],
            kv_cache_dtype=KvCacheDataType.FP8,
            fp8_compute=True,
            q_scale=0.5,
            kv_scale=0.25,
        )
        q, k, v = self._qkv(32, 129)
        qi = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
        ki = torch.tensor([0, 129], device="cuda", dtype=torch.int32)

        def attention(causal=False, return_lse=True):
            return op._run_dense_attention(
                q,
                k,
                v,
                qo_indptr=qi,
                kv_indptr=ki,
                seq_lens=ki[1:] - ki[:-1],
                max_q_len=32,
                max_kv_len=129,
                causal=causal,
                return_lse=return_lse,
            )

        for _ in range(3):
            attention()
        names = cuda_kernel_names(attention)
        quantizers = [name for name in names if "_quantize" in name]
        self.assertEqual(len(quantizers), 1, names)

        # Exercise the real downstream attention with the original three-call
        # quantization baseline, not a mocked consumer of the new outputs.
        old_q, old_k, old_v = quantize_fp8(q, 0.5), quantize_fp8(k), quantize_fp8(v)
        for causal in (False, True):
            actual, actual_lse = attention(causal)
            expected, expected_lse = op.tokenspeed_prefill(
                query=old_q,
                key=old_k,
                value=old_v,
                seq_lens=ki[1:] - ki[:-1],
                cum_seq_lens=ki,
                max_seq_len=129,
                batch_size=1,
                softmax_scale=op.scale * op.q_scale,
                is_causal=causal,
                return_lse=True,
                cum_seq_lens_q=qi,
                max_seq_len_q=32,
                enable_pdl=False,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)

        without_lse, omitted_lse = attention(causal=True, return_lse=False)
        self.assertIsNone(omitted_lse)
        torch.testing.assert_close(without_lse, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
