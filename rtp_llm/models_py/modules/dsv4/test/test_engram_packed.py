"""Packed Engram byte/layout contract and explicitly enabled CUDA consumers."""

import importlib.util
import os
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "engram_packed_tested", SOURCE / "_engram_packed_triton.py"
)
packed_op = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packed_op)


def make_input(tokens, heads, dim, *, padding=0):
    width = heads * (dim + dim // 32)
    # A nonzero storage offset, padded token stride, and every possible byte.
    backing = torch.arange((tokens + 2) * (width + padding), dtype=torch.int64)
    backing = backing.to(torch.uint8).reshape(tokens + 2, width + padding)
    return backing[1 : tokens + 1, :width]


def reference(packed, heads, dim):
    """Independent byte-address oracle, including signed little-endian words."""
    source = packed.cpu().contiguous().numpy().tobytes()
    tokens = packed.shape[0]
    width = heads * dim
    groups = width // 32
    padded_tokens = (tokens + 3) // 4 * 4
    columns = (groups + 3) // 4
    value_bytes = bytearray(tokens * width)
    scale_bytes = bytearray(columns * padded_tokens * 4)
    record = dim + dim // 32
    for token in range(tokens):
        for head in range(heads):
            src = (token * heads + head) * record
            dst = (token * heads + head) * dim
            value_bytes[dst : dst + dim] = source[src : src + dim]
            for block in range(dim // 32):
                group = head * (dim // 32) + block
                dst = (group // 4 * padded_tokens + token) * 4 + group % 4
                scale_bytes[dst] = source[src + dim + block]
    values = torch.tensor(list(value_bytes), dtype=torch.uint8).reshape(tokens, width)
    words = [value[0] for value in struct.iter_unpack("<i", scale_bytes)]
    scales = torch.tensor(words, dtype=torch.int32).reshape(columns, padded_tokens)
    return values, scales


class CudaMetadataTensor(torch.Tensor):
    """CPU storage with CUDA metadata for allocation-free wrapper gate tests."""

    @property
    def is_cuda(self):
        return True


class EngramPackedContractTest(unittest.TestCase):
    def test_gate_before_allocation_and_supported_model_shape(self):
        valid = make_input(3, 24, 256, padding=11).as_subclass(CudaMetadataTensor)
        self.assertTrue(packed_op.is_supported(valid, 24, 256))
        bad = [
            (None, 24, 256),
            (valid.as_subclass(torch.Tensor), 24, 256),
            (valid.float(), 24, 256),
            (valid[:, ::2], 24, 256),
            (valid.unsqueeze(0), 24, 256),
            (valid, 23, 256),
            (valid, True, 256),
            (valid, 24, False),
            (valid, 24, 255),
            (valid, 0, 256),
            (valid.as_strided(valid.shape, (1, 1)), 24, 256),
            (torch._neg_view(valid), 24, 256),
        ]
        with patch.object(torch, "empty", side_effect=AssertionError("allocation")):
            for tensor, heads, dim in bad:
                with self.subTest(heads=heads, dim=dim):
                    self.assertIsNone(packed_op.unpack_packed_rows(tensor, heads, dim))

    def test_zero_rows_do_not_launch(self):
        value = torch.empty((0, 24 * 264), dtype=torch.uint8)
        with patch.object(packed_op, "is_supported", return_value=True), patch.object(
            packed_op, "_unpack_packed_rows_kernel"
        ) as kernel:
            q, scales = packed_op.unpack_packed_rows(value, 24, 256)
        self.assertEqual(q.shape, (0, 6144))
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.shape, (0, 48))
        self.assertEqual(scales.stride(), (1, 0))
        kernel.__getitem__.assert_not_called()

    def test_scale_byte_order_and_partial_words_oracle(self):
        source = torch.zeros((1, 5 * 33), dtype=torch.uint8)
        source.reshape(1, 5, 33)[0, :, 32] = torch.tensor(
            [0, 1, 128, 255, 254], dtype=torch.uint8
        )
        _, storage = reference(source, 5, 32)
        self.assertEqual(storage.shape, (2, 4))
        self.assertEqual(storage[0, 0].item(), -8388352)  # 0xff800100
        self.assertEqual(storage[1, 0].item(), 254)
        self.assertEqual(storage[:, 1:].count_nonzero().item(), 0)

    def test_original_bytes_do_not_imply_bf16_roundtrip_equivalence(self):
        # E4M3's smallest positive subnormal times exponent byte 1 underflows
        # BF16 to zero, while the source FP8 and exponent bytes remain nonzero.
        tiny = torch.tensor([1], dtype=torch.uint8).view(torch.float8_e4m3fn)
        scale = torch.tensor([1 << 23], dtype=torch.int32).view(torch.float32)
        self.assertGreater((tiny.float() * scale).item(), 0.0)
        self.assertEqual((tiny.float() * scale).bfloat16().item(), 0.0)
        self.assertEqual(tiny.view(torch.uint8).item(), 1)


@unittest.skipUnless(
    os.environ.get("TRITON_INTERPRET") == "1", "explicit CPU Triton interpreter only"
)
class EngramPackedInterpreterTest(unittest.TestCase):
    def test_actual_kernel_byte_layout_strides_padding_and_guards(self):
        cases = [(1, 1, 32), (3, 3, 32), (5, 3, 160), (7, 24, 256), (33, 2, 512)]
        for tokens, heads, dim in cases:
            with self.subTest(tokens=tokens, heads=heads, dim=dim):
                source = make_input(tokens, heads, dim, padding=13)
                expected_q, expected_s = reference(source, heads, dim)
                owner_q = torch.full(
                    (tokens * heads * dim + 32,), 0xA5, dtype=torch.uint8
                )
                q = (
                    owner_q[16:-16]
                    .reshape(tokens, heads * dim)
                    .view(torch.float8_e4m3fn)
                )
                owner_s = torch.full(
                    (expected_s.numel() + 8,), 0x12345678, dtype=torch.int32
                )
                s = owner_s[4:-4].reshape(expected_s.shape)
                work = expected_s.numel() * 128
                packed_op._unpack_packed_rows_kernel[((work + 4095) // 4096,)](
                    source,
                    q,
                    s,
                    tokens,
                    source.stride(0),
                    expected_s.shape[1],
                    HEADS=heads,
                    DIM=dim,
                    BLOCK=4096,
                    num_warps=4,
                )
                torch.testing.assert_close(
                    q.view(torch.uint8), expected_q, rtol=0, atol=0
                )
                torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
                self.assertTrue(bool((owner_q[:16] == 0xA5).all()))
                self.assertTrue(bool((owner_q[-16:] == 0xA5).all()))
                self.assertTrue(bool((owner_s[:4] == 0x12345678).all()))
                self.assertTrue(bool((owner_s[-4:] == 0x12345678).all()))
        self.assertFalse(torch.cuda.is_initialized())


@unittest.skipUnless(
    os.environ.get("RTP_ENGRAM_CPU_TEST_CUDA") == "1", "explicit CUDA oracle only"
)
class EngramPackedCudaTest(unittest.TestCase):
    def assert_matches(self, source, heads, dim, outputs=None):
        expected_q, expected_storage = reference(source, heads, dim)
        q, s = outputs or packed_op.unpack_packed_rows(source, heads, dim)
        tokens = source.shape[0]
        aligned_tokens = (tokens + 3) // 4 * 4
        self.assertTrue(q.is_contiguous())
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.int32)
        self.assertEqual(s.stride(), (1, aligned_tokens))
        torch.testing.assert_close(
            q.cpu().view(torch.uint8), expected_q, rtol=0, atol=0
        )
        storage = s.as_strided(expected_storage.shape, (aligned_tokens, 1))
        torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0)

    def test_bytes_layout_specials_strides_and_short_tails(self):
        for tokens in (0, 1, 3, 4, 7, 31, 33, 257):
            for heads, dim in ((1, 32), (3, 160), (24, 256)):
                with self.subTest(tokens=tokens, heads=heads, dim=dim):
                    cpu = make_input(tokens, heads, dim, padding=13)
                    backing = torch.full(
                        (tokens + 2, cpu.shape[1] + 13),
                        0xAA,
                        device="cuda",
                        dtype=torch.uint8,
                    )
                    source = backing[1 : tokens + 1, : cpu.shape[1]]
                    source.copy_(cpu)
                    self.assert_matches(source, heads, dim)

    def test_nondefault_stream_graph_replay_observes_new_bytes(self):
        heads, dim = 24, 256
        source = make_input(7, heads, dim).cuda()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                packed_op.unpack_packed_rows(source, heads, dim)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = packed_op.unpack_packed_rows(source, heads, dim)
        for fill in (0, 1, 128, 255):
            source.fill_(fill)
            graph.replay()
            self.assert_matches(source, heads, dim, output)

    def test_scale_layout_matches_cuda_quant_producer(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        heads, dim, tokens = 24, 256, 7
        # Setting one value to 448 fixes amax for each group. All chosen normal
        # exponents survive the BF16 producer exactly, so both scale values and
        # their packed byte positions must match (including high-bit bytes).
        exponents = torch.arange(heads * dim // 32, device="cuda") % 10 + 122
        row = torch.tensor([448.0], device="cuda").to(torch.float8_e4m3fn)
        source = torch.zeros(
            (tokens, heads, dim + dim // 32), device="cuda", dtype=torch.uint8
        )
        source[:, :, :dim].reshape(tokens, heads, dim // 32, 32)[:, :, :, 0] = row.view(
            torch.uint8
        )
        source[:, :, dim:] = exponents.to(torch.uint8).reshape(heads, dim // 32)
        q, s = packed_op.unpack_packed_rows(source.reshape(tokens, -1), heads, dim)
        scale = torch.pow(2.0, exponents.float() - 127).repeat_interleave(32)
        bf16 = (q.float() * scale).bfloat16()
        expected_q, expected_s = sgl_per_token_group_quant_fp8(
            bf16,
            32,
            eps=torch.finfo(torch.float32).tiny,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        self.assertEqual(s.stride(), expected_s.stride())
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
        torch.testing.assert_close(
            q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
        )

    def test_wkv_finite_range_against_bf16_roundtrip_and_reference(self):
        from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear

        torch.manual_seed(20261012)
        heads, dim, n = 24, 256, 256
        k = heads * dim
        weight = (torch.randn(n, k, device="cuda") * 0.02).to(torch.float8_e4m3fn)
        weight_scales = torch.ones(n // 32, k // 32, device="cuda").to(
            torch.float8_e8m0fnu
        )
        linear = V41MXFP8Linear(weight, weight_scales)
        for tokens in (1, 7, 33):
            source = torch.empty(
                (tokens, heads, dim + dim // 32), dtype=torch.uint8, device="cuda"
            )
            values = (torch.randn(tokens, heads, dim, device="cuda") * 20).to(
                torch.float8_e4m3fn
            )
            exponents = torch.randint(
                120, 135, (tokens, heads, dim // 32), device="cuda", dtype=torch.uint8
            )
            source[:, :, :dim] = values.view(torch.uint8)
            source[:, :, dim:] = exponents
            q, s = packed_op.unpack_packed_rows(source.reshape(tokens, -1), heads, dim)
            factors = torch.pow(2.0, exponents.float() - 127).repeat_interleave(32, -1)
            bf16 = (values.float() * factors).reshape(tokens, k).bfloat16()
            direct = linear.forward_quantized(q, s)
            legacy = linear(bf16)
            exact_input = bf16.float() @ weight.float().T
            torch.testing.assert_close(direct, legacy, rtol=0.016, atol=0.001)
            for actual in (direct, legacy):
                relative_rms = (
                    actual.float() - exact_input
                ).square().mean().sqrt() / exact_input.square().mean().sqrt()
                self.assertLess(relative_rms.item(), 0.005)


if __name__ == "__main__":
    unittest.main()
