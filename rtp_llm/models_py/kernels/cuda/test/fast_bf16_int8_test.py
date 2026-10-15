"""Accuracy tests for groupwise BF16/INT8 quantization and multi-source reduction."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.kernels.cuda.fast_bf16_int8 import (
    dequantize,
    dequantize_reduce,
    quantize,
)
from rtp_llm.models_py.kernels.cuda.test.fast_bf16_int8_test_utils import (
    rank_inputs,
    reference_dequantize,
    reference_quantize,
    reference_reduce,
)

GROUP_SIZES = (16, 32, 64, 128)


def allocate_codes(inputs, m, k, group_size, packed, scale_dtype=torch.bfloat16):
    """Allocate codes/scales for `inputs` sources, separately or as views of aligned packets."""
    if not packed:
        return (
            torch.empty((inputs, m, k), dtype=torch.int8, device="cuda"),
            torch.empty((inputs, m, k // group_size), dtype=scale_dtype, device="cuda"),
        )
    # [all codes][all scales][rank padding]; no padding within groups.
    n = m * k
    stride = ((n + 2 * n // group_size + 15) // 16) * 16
    packet = torch.empty((inputs, stride), dtype=torch.uint8, device="cuda")
    q = packet[:, :n].view(torch.int8).view(inputs, m, k)
    scales = packet[:, n : n + 2 * n // group_size].view(scale_dtype)
    return q, scales.view(inputs, m, k // group_size)


class FastBf16Int8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Check that the current device supports NVIDIA SM90+ CUDA kernels."""
        if (
            torch.version.hip is not None
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 9
        ):
            raise unittest.SkipTest("NVIDIA CUDA SM90+ required")

    def assert_bits_equal(self, actual, expected):
        """Compare tensor shapes, dtypes and raw bits, including signed zero."""
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        dtype = (
            torch.int16
            if actual.dtype in (torch.bfloat16, torch.float16)
            else torch.int8
        )
        self.assertTrue(torch.equal(actual.view(dtype), expected.view(dtype)))

    def quantize_checked(self, x, q, scales, group_size):
        """Check quantization against the original input and verify input preservation."""
        original_x = x.clone()
        expected_q, expected_s = reference_quantize(
            original_x, group_size, scales.dtype
        )
        quantize(x, q, scales, group_size=group_size)
        self.assert_bits_equal(x, original_x)
        self.assert_bits_equal(q, expected_q)
        self.assert_bits_equal(scales, expected_s)

    def test_scale_only_24_bitwise_cases(self):
        """Persist the 4 shapes x 3 amplitudes x 2 codecs as independent bitwise checks."""
        for m, factor, dtype in itertools.product(
            (1, 3, 8, 83), (0.01, 1.0, 100.0), (torch.bfloat16, torch.float16)
        ):
            with self.subTest(m=m, factor=factor, dtype=dtype):
                inputs = rank_inputs(m, 5120, factor)
                expected_q, expected_s, expected_out = reference_reduce(inputs, dtype)
                q, scales = allocate_codes(2, m, 5120, 16, False, dtype)
                for rank, x in enumerate(inputs):
                    quantize(x.cuda(), q[rank], scales[rank])
                out = torch.empty((m, 5120), device="cuda", dtype=torch.bfloat16)
                dequantize_reduce(q, scales, out)
                self.assert_bits_equal(q.cpu(), expected_q)
                self.assert_bits_equal(scales.cpu(), expected_s)
                self.assert_bits_equal(out.cpu(), expected_out)

    def dequantize_checked(self, q, scales, out, group_size):
        """Check decoding against the original codes/scales and verify input preservation."""
        original_q, original_s = q.clone(), scales.clone()
        expected = reference_dequantize(original_q, original_s, group_size)
        dequantize(q, scales, out, group_size=group_size)
        self.assert_bits_equal(q, original_q)
        self.assert_bits_equal(scales, original_s)
        self.assert_bits_equal(out, expected)

    def test_single_source_group_sizes(self):
        """Check groupwise quantization and single-source decoding across shapes and scale ranges."""
        cases = (
            (group_size, shape)
            for group_size in GROUP_SIZES
            for shape in (
                (1, group_size),
                (3, 3 * group_size),
                (1, 128),
                (7, 256),
                (127, 512),
                (4808, 5120),
            )
        )
        for group_size, (m, k) in cases:
            with self.subTest(group_size=group_size, shape=(m, k)):
                torch.manual_seed(9137)
                x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                groups = x.view(-1, group_size)
                if len(groups) > 1:
                    groups[0].zero_()
                    groups[1] *= 1e-12
                if len(groups) > 2:
                    groups[2, 0] = 100
                if len(groups) > 3:
                    groups[3].zero_()
                    groups[3, :5] = torch.tensor(
                        [127, 1.5, 2.5, -1.5, -2.5], device="cuda"
                    )
                q, scales = allocate_codes(1, m, k, group_size, packed=True)
                self.quantize_checked(x, q[0], scales[0], group_size)
                out = torch.empty_like(x)
                self.dequantize_checked(q[0], scales[0], out, group_size)
        for group_size in GROUP_SIZES:
            for max_index in (-1, 0, group_size // 2, group_size - 1):
                with self.subTest(group_size=group_size, max_index=max_index):
                    x = torch.zeros(
                        (1, group_size), dtype=torch.bfloat16, device="cuda"
                    )
                    if max_index >= 0:
                        x.copy_(
                            torch.tensor([1.5, 2.5, -1.5, -2.5], device="cuda")
                            .repeat(group_size // 4)
                            .view_as(x)
                        )
                        x[0, max_index] = 127 if max_index % 2 == 0 else -127
                    q, scales = allocate_codes(
                        1, 1, group_size, group_size, packed=True
                    )
                    self.quantize_checked(x, q[0], scales[0], group_size)
                    out = torch.empty_like(x)
                    self.dequantize_checked(q[0], scales[0], out, group_size)
        # Decoder accuracy covers every signed INT8 code and both scale paths.
        q = torch.arange(-128, 128, device="cuda").to(torch.int8).view(1, 256)
        out = torch.empty_like(q, dtype=torch.bfloat16)
        for group_size, scale in itertools.product(
            GROUP_SIZES, (0.0, 1e-38, 0.001, 1.0, 1000.0, 1e37)
        ):
            with self.subTest(group_size=group_size, scale=scale):
                scales = torch.full(
                    (1, 256 // group_size), scale, dtype=torch.bfloat16, device="cuda"
                )
                self.dequantize_checked(q, scales, out, group_size)

    def test_reduce_group_sizes(self):
        """Check ordered BF16 reduction across group sizes, source counts and storage layouts."""
        cases = itertools.product(
            GROUP_SIZES,
            (1, 2, 3, 4, 8, 16),
            ((1, 128), (7, 256), (127, 5120)),
            (False, True),
        )
        for group_size, inputs, (m, k), packed in cases:
            with self.subTest(
                group_size=group_size, inputs=inputs, shape=(m, k), packed=packed
            ):
                torch.manual_seed(9137)
                q, scales = allocate_codes(inputs, m, k, group_size, packed)
                expected = None
                for source in range(inputs):
                    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                    x *= 2.0 ** (source % 5 - 2)
                    x.view(-1)[:group_size].zero_()
                    # Exact cancellation plus independent scales on other groups.
                    x.view(-1)[group_size : 2 * group_size] = (-1.0) ** source
                    self.quantize_checked(x, q[source], scales[source], group_size)
                    decoded = reference_dequantize(
                        q[source], scales[source], group_size
                    )
                    expected = (
                        decoded
                        if expected is None
                        else (expected.float() + decoded.float()).to(torch.bfloat16)
                    )
                out = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
                original_q, original_s = q.clone(), scales.clone()
                dequantize_reduce(q, scales, out, group_size=group_size)
                self.assert_bits_equal(q, original_q)
                self.assert_bits_equal(scales, original_s)
                self.assert_bits_equal(out, expected)

    def test_fp16_scale_ranges_sources_and_padding(self):
        """Keep BF16 quantization/reduction semantics with the FP16 wire codec."""
        for group_size, inputs, packed in itertools.product(
            GROUP_SIZES, (1, 2, 3, 4, 8, 16), (False, True)
        ):
            with self.subTest(group=group_size, inputs=inputs, packed=packed):
                torch.manual_seed(1703)
                m, k = 3, 17 * group_size
                q, scales = allocate_codes(
                    inputs, m, k, group_size, packed, torch.float16
                )
                expected = None
                for source in range(inputs):
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    groups = x.view(-1, group_size)
                    groups[:6] = torch.tensor(
                        [0.0, 1e-8, 1e-5, 1e-3, 1e6, 1e8], device="cuda"
                    ).unsqueeze(-1)
                    groups[6].fill_((-1.0) ** source)
                    self.quantize_checked(x, q[source], scales[source], group_size)
                    decoded = reference_dequantize(
                        q[source], scales[source], group_size
                    )
                    expected = (
                        decoded
                        if expected is None
                        else (expected.float() + decoded.float()).to(torch.bfloat16)
                    )
                out = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
                original_q, original_s = q.clone(), scales.clone()
                dequantize_reduce(q, scales, out, group_size=group_size)
                self.assert_bits_equal(out, expected)
                self.assert_bits_equal(q, original_q)
                self.assert_bits_equal(scales, original_s)

    def test_fp16_decode_all_codes(self):
        """Cover signed INT8 endpoints and FP16 zero/subnormal/normal/maximum scales."""
        q = torch.arange(-128, 128, device="cuda").to(torch.int8).view(1, 256)
        out = torch.empty_like(q, dtype=torch.bfloat16)
        for group_size, value in itertools.product(
            GROUP_SIZES, (0.0, 2**-24, 2**-14, 0.001, 1.0, 65504.0)
        ):
            with self.subTest(group=group_size, value=value):
                scales = torch.full(
                    (1, 256 // group_size), value, dtype=torch.float16, device="cuda"
                )
                self.dequantize_checked(q, scales, out, group_size)

    def test_fp16_scale_cuda_graph_current_stream(self):
        """Replay on a nondefault stream and recompute scales from changed inputs."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.randn((2, 3, 5120), device="cuda", dtype=torch.bfloat16)
            q, scales = allocate_codes(2, 3, 5120, 16, False, torch.float16)
            out = torch.empty_like(x[0])

            def run():
                for source in range(2):
                    quantize(x[source], q[source], scales[source])
                dequantize_reduce(q, scales, out)

            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            for factor in (0.01, 100.0):
                x.copy_(torch.randn_like(x) * factor)
                graph.replay()
                expected = None
                for source in range(2):
                    rq, rs = reference_quantize(x[source], 16, torch.float16)
                    decoded = reference_dequantize(rq, rs, 16)
                    expected = (
                        decoded
                        if expected is None
                        else (expected.float() + decoded.float()).to(torch.bfloat16)
                    )
                    self.assert_bits_equal(q[source], rq)
                    self.assert_bits_equal(scales[source], rs)
                self.assert_bits_equal(out, expected)
        stream.synchronize()


class QwenInt8ScaleRoutingTest(unittest.TestCase):
    def test_deployment_switch_selects_dtype_for_both_reductions(self):
        from rtp_llm.models_py.model_desc import qwen3

        model_types = ("qwen_2", "qwen_agent", "qwen_tool", "qwen_3", "qwen3_vl")
        for model_type, enabled in itertools.product(model_types, (None, False, True)):
            with self.subTest(model_type=model_type, enabled=enabled):
                expected_dtype = torch.float16 if enabled else torch.bfloat16
                config = SimpleNamespace(
                    model_type=model_type,
                    activation_type=None,
                    layernorm_eps=1e-6,
                    enable_w4a16_sm120_dense_ffn=False,
                    getAttentionConfigs=lambda _: None,
                )
                if enabled is not None:
                    config.enable_int8_allreduce_fp16_scale = enabled
                parallelism = SimpleNamespace(
                    get_attn_tp_size=lambda: 2,
                    get_ffn_tp_size=lambda: 2,
                )
                weights = {
                    qwen3.W.pre_ln_gamma: SimpleNamespace(device=torch.device("cuda")),
                    qwen3.W.post_ln_gamma: object(),
                }
                attention = Mock(
                    tp_size=2, side_effect=lambda **kwargs: kwargs["hidden_states"]
                )
                mlp = Mock(side_effect=lambda x, **kwargs: x)
                with (
                    patch.object(qwen3, "CausalAttention", return_value=attention),
                    patch.object(qwen3, "DenseMLP", return_value=mlp),
                    patch.object(
                        qwen3,
                        "RMSNorm",
                        side_effect=lambda *a, **kw: torch.nn.Identity(),
                    ),
                    patch.object(
                        torch.cuda,
                        "get_device_name",
                        return_value="NVIDIA RTX PRO 5000 Blackwell",
                    ),
                    patch.object(torch.version, "hip", None),
                    patch.object(
                        qwen3,
                        "_quantized_all_reduce",
                        side_effect=lambda x, *a, **kw: x,
                    ) as reduce,
                ):
                    layer = qwen3.Qwen3DecoderLayer(config, parallelism, 0, weights)
                    layer(torch.ones((1, 16), dtype=torch.bfloat16), object())
                self.assertIs(layer.quant, qwen3.quantize)
                self.assertIs(layer.dequant, qwen3.dequantize_reduce)
                self.assertEqual(reduce.call_count, 2)
                for call in reduce.call_args_list:
                    self.assertIs(call.kwargs["scale_dtype"], expected_dtype)
                self.assertTrue(attention.call_args.kwargs["skip_allreduce"])
                self.assertTrue(mlp.call_args.kwargs["skip_allreduce"])

    def test_scale_dtype_reaches_both_gathers_and_decoder(self):
        from rtp_llm.models_py.model_desc import quantized_all_reduce as collective

        for requested_dtype in (None, torch.bfloat16, torch.float16):
            with self.subTest(scale_dtype=requested_dtype):
                hidden_states = torch.ones((3, 32), dtype=torch.bfloat16)
                quant, dequant = Mock(), Mock()
                pg = Mock()
                pg.size.return_value = 2
                kwargs = (
                    {} if requested_dtype is None else {"scale_dtype": requested_dtype}
                )
                expected_dtype = requested_dtype or torch.bfloat16
                with (
                    patch.object(collective, "support", return_value=True),
                    patch.object(collective, "_get_group", return_value=pg),
                    patch.object(torch.distributed, "all_gather_into_tensor") as gather,
                ):
                    result = collective._quantized_all_reduce(
                        hidden_states, quant, dequant, **kwargs
                    )
                self.assertIs(result, hidden_states)
                self.assertEqual(gather.call_count, 2)
                codes, scales = quant.call_args.args[1:3]
                self.assertIs(codes.dtype, torch.int8)
                self.assertIs(scales.dtype, expected_dtype)
                self.assertEqual(scales.shape, (3, 2))
                gathered_codes, gathered_scales, out = dequant.call_args.args
                self.assertIs(gathered_codes.dtype, torch.int8)
                self.assertIs(gathered_scales.dtype, expected_dtype)
                self.assertEqual(gathered_scales.shape, (2, 3, 2))
                self.assertIs(out, hidden_states)
                self.assertIs(gather.call_args_list[0].args[1], codes)
                self.assertIs(gather.call_args_list[1].args[1], scales)
                self.assertIs(gather.call_args_list[1].args[0].dtype, expected_dtype)


if __name__ == "__main__":
    unittest.main()
