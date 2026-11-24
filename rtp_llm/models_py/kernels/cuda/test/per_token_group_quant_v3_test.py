"""V4.1 group32 quantizer correctness, including legacy scale semantics."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
    sgl_per_token_group_quant_fp8_v3,
)
from rtp_llm.ops.compute_ops import per_token_group_quant_fp8_v3


def allocate(x):
    m, k = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((k // 128, (m + 3) // 4 * 4), device=x.device, dtype=torch.int32).T[
        :m
    ]
    return q, s


def reference(x):
    """Independent legacy group32 scale policy for finite BF16 input."""
    m, k = x.shape
    values = x.float().reshape(m, k // 32, 32)
    amax = values.abs().amax(dim=-1)
    exponent = torch.ceil(torch.log2((amax / 448.0).clamp_min(1e-10)))
    q = (
        (values / torch.exp2(exponent)[..., None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape(m, k)
    )
    codes = (exponent.to(torch.int32) + 127).reshape(m, k // 128, 4)
    scales = (
        codes[..., 0]
        | (codes[..., 1] << 8)
        | (codes[..., 2] << 16)
        | (codes[..., 3] << 24)
    )
    return q, scales


def assert_exact(actual, expected):
    q, scales = actual
    expected_q, expected_scales = expected
    torch.testing.assert_close(
        q.view(torch.uint8), expected_q.view(torch.uint8), atol=0, rtol=0
    )
    # K % 128 == 0: every byte in each logical scale word is defined.
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)


class QuantV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")

    def test_v3_entry_and_random_shapes(self):
        for m in (1, 2, 3, 4, 7, 8, 31, 32, 48, 127, 160, 192, 257, 768, 1536, 3072):
            for k in (128, 384, 1280, 2304, 4096, 5120, 6144, 8192, 15360):
                with self.subTest(m=m, k=k):
                    torch.manual_seed(123)
                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                    q, s = sgl_per_token_group_quant_fp8_v3(x)
                    assert_exact((q, s), reference(x))
                    self.assertEqual(s.stride(), (1, (m + 3) // 4 * 4))

    def test_all_finite_bf16_group_maxima(self):
        bits = torch.arange(0x7F80, device="cuda", dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        x = values[:, None].expand(-1, 128).contiguous()
        x[:, 1::2] *= -1
        assert_exact(sgl_per_token_group_quant_fp8_v3(x), reference(x))

    def test_group_outliers_and_zero(self):
        for multiplier in (0.0, 1e-30, 1e-10, 1.0, 1e10, 1e30):
            x = (torch.randn((33, 5120), device="cuda") * multiplier).bfloat16()
            x[:, ::32] = multiplier
            x[:, 1::32] = -multiplier
            assert_exact(sgl_per_token_group_quant_fp8_v3(x), reference(x))

    def test_stream_and_graph(self):
        x = torch.randn((192, 8192), device="cuda", dtype=torch.bfloat16)
        q, s = allocate(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            per_token_group_quant_fp8_v3(x, q, s)
        torch.cuda.current_stream().wait_stream(stream)
        assert_exact((q, s), reference(x))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            q, s = sgl_per_token_group_quant_fp8_v3(x)
        for _ in range(3):
            x.normal_()
            graph.replay()
            assert_exact((q, s), reference(x))

    def test_empty_and_invalid_inputs(self):
        x = torch.empty((0, 5120), device="cuda", dtype=torch.bfloat16)
        q, s = allocate(x)
        per_token_group_quant_fp8_v3(x, q, s)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(s.shape, (0, 40))
        x = torch.randn((7, 5120), device="cuda", dtype=torch.bfloat16)
        q, s = allocate(x)
        cases = [
            (x.float(), q, s),
            (x[:, ::2], q[:, ::2], s),
            (x, q, s.contiguous()),
            (x, q.float(), s),
            (x, q, s.float()),
            (x, q[:, :128], s),
            (x[:, :96].contiguous(), q[:, :96].contiguous(), s[:, :1]),
        ]
        for args in cases:
            with self.assertRaises(RuntimeError):
                per_token_group_quant_fp8_v3(*args)
        for args in ((x.cpu(), q, s), (x, q.cpu(), s), (x, q, s.cpu())):
            with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
                per_token_group_quant_fp8_v3(*args)
        alias = x.view(torch.float8_e4m3fn).view(-1)[: x.numel()].view_as(q)
        with self.assertRaises(RuntimeError):
            per_token_group_quant_fp8_v3(x, alias, s)

    def test_scale_alias_rejected(self):
        x = torch.randn((7, 5120), device="cuda", dtype=torch.bfloat16)
        q, _ = allocate(x)
        for storage in (x, q):
            with self.subTest(storage_dtype=storage.dtype):
                alias = storage.view(torch.int32).as_strided((7, 40), (1, 8))
                with self.assertRaises(RuntimeError):
                    per_token_group_quant_fp8_v3(x, q, alias)

    def test_explicit_python_entry(self):
        for m in (0, 1, 7, 192):
            x = torch.randn((m, 5120), device="cuda", dtype=torch.bfloat16)
            q, s = sgl_per_token_group_quant_fp8_v3(x)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(s.shape, (m, 40))
            assert_exact((q, s), reference(x))
        for shape in ((128,), (2, 0), (2, 96), (2, 2, 128)):
            with self.subTest(shape=shape), self.assertRaisesRegex(
                ValueError, "positive multiple of 128"
            ):
                sgl_per_token_group_quant_fp8_v3(torch.empty(shape))
        # Shape is checked before allocation; remaining contracts come from native.
        for x, message in (
            (torch.empty((2, 128), dtype=torch.bfloat16), "same CUDA device"),
            (torch.empty((2, 128), device="cuda"), "BF16"),
            (
                torch.empty((2, 256), device="cuda", dtype=torch.bfloat16)[:, ::2],
                "contiguous",
            ),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError, message
            ):
                sgl_per_token_group_quant_fp8_v3(x)

    def test_public_auto_keeps_legacy_scales_across_size_boundary(self):
        for m in (1, 819, 820, 3072):
            x = torch.zeros((m, 5120), device="cuda", dtype=torch.bfloat16)
            x[0, :32] = 1e-30
            for mode in ("auto", "legacy"):
                with self.subTest(m=m, mode=mode), patch.dict(
                    "os.environ", DSV4_FP8_QUANT_KERNEL=mode
                ):
                    actual = sgl_per_token_group_quant_fp8(
                        x,
                        group_size=32,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    assert_exact(actual, reference(x))

    def test_public_auto_thread_boundaries_and_batch_independence(self):
        # BF16 values below/at/above E4M3 rounding ties, with both signs.
        values = torch.tensor(
            [1.0546875, 1.0625, 1.0703125, 1.1796875, 1.1875, 1.1953125],
            device="cuda",
            dtype=torch.bfloat16,
        )
        for k, sizes in ((8192, (63, 64, 65, 16384)), (384, (1365, 1366, 16384))):
            rows = values.repeat((3 * k + 5) // 6)[: 3 * k].reshape(3, k)
            rows[:, 1::2] *= -1
            groups = rows.view(3, k // 32, 32)
            # Outliers bracket a power-of-two scale transition at 448.
            maxima = torch.tensor([446, 448, 450], device="cuda")[:, None]
            groups[:, :, 0] = maxima
            groups[:, :, 17] = -maxima
            exponents = torch.tensor([-40, -1, 0, 1, 10], device="cuda")
            factors = torch.exp2(exponents[torch.arange(k // 32, device="cuda") % 5])
            groups.mul_(factors[None, :, None])
            kwargs = dict(
                group_size=32,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            expected = reference(rows)
            with patch.dict("os.environ", DSV4_FP8_QUANT_KERNEL="legacy"):
                assert_exact(sgl_per_token_group_quant_fp8(rows, **kwargs), expected)
            with patch.dict("os.environ", DSV4_FP8_QUANT_KERNEL="auto"):
                small = sgl_per_token_group_quant_fp8(rows, **kwargs)
                assert_exact(small, expected)
                self.assertEqual(small[1].stride(), (1, 4))
                for m in sizes:
                    with self.subTest(m=m, k=k):
                        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                        positions = [0, m // 2, m - 1]
                        x[positions] = rows
                        q, scales = sgl_per_token_group_quant_fp8(x, **kwargs)
                        self.assertEqual(scales.stride(), (1, (m + 3) // 4 * 4))
                        assert_exact(
                            (
                                q.view(torch.uint8)[positions].view(q.dtype),
                                scales[positions],
                            ),
                            small,
                        )
                        if m < 16384:
                            assert_exact((q, scales), reference(x))

    def test_offsets_above_int32_elements(self):
        m, k = 262145, 8192
        required = m * k * 3 + m * (k // 32)
        free, _ = torch.cuda.mem_get_info()
        if free < required + (2 << 30):
            self.skipTest("Large-offset regression requires about 9 GiB free")
        x = torch.zeros((m, k), device="cuda", dtype=torch.bfloat16)
        x[-1].fill_(1)
        q, s = allocate(x)
        per_token_group_quant_fp8_v3(x, q, s)
        assert_exact((q[-1:], s[-1:]), reference(x[-1:]))
        assert_exact((q[:1], s[:1]), reference(x[:1]))

    def test_small_tails_and_16_byte_aligned_input(self):
        for m in (1, 3, 17, 33):
            for k in (128, 384, 1280, 2304, 5120, 8192):
                storage = torch.randn(m * k + 8, device="cuda", dtype=torch.bfloat16)
                x = storage[8:].view(m, k)
                self.assertEqual(x.data_ptr() % 32, 16)
                assert_exact(sgl_per_token_group_quant_fp8_v3(x), reference(x))


if __name__ == "__main__":
    unittest.main()
