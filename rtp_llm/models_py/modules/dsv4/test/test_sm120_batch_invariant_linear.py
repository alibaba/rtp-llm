"""Fixed-K projections preserve shared rows across prefill batch sizes."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.sm120_batch_invariant_linear import (
    batch_invariant_linear,
    is_supported,
)
from rtp_llm.models_py.utils.arch import is_sm120


class BatchInvariantLinearTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not is_sm120():
            self.skipTest("SM120 hardware is required")
        torch.manual_seed(73)

    def test_numerics_batch_repeat_and_graph(self):
        for n, k in ((17, 37), (64, 4096), (256, 4096), (512, 4096), (2048, 4096)):
            weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.03
            logical = torch.randn(7, k, device="cuda", dtype=torch.bfloat16)
            reference = (
                (logical.cpu().double() @ weight.cpu().double().T).float().cuda()
            )
            canonical = batch_invariant_linear(logical, weight, out_dtype=torch.float32)
            torch.testing.assert_close(canonical, reference, atol=1e-3, rtol=1e-4)
            for rows in (0, 1, 4, 17, 63, 64, 65, 129, 1028, 4096):
                with self.subTest(n=n, k=k, rows=rows):
                    ids = torch.arange(rows, device="cuda") % logical.shape[0]
                    x = logical.index_select(0, ids)
                    expected = canonical.index_select(0, ids)
                    for _ in range(3):
                        actual = batch_invariant_linear(
                            x, weight, out_dtype=torch.float32
                        )
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    rounded = batch_invariant_linear(
                        x, weight, out_dtype=torch.bfloat16
                    )
                    torch.testing.assert_close(
                        rounded, expected.bfloat16(), atol=0, rtol=0
                    )
                    if not rows:
                        continue
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = batch_invariant_linear(
                            x, weight, out_dtype=torch.float32
                        )
                    for _ in range(3):
                        captured.fill_(float("nan"))
                        graph.replay()
                        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
                    # Replay must consume changed inputs, not captured results.
                    x.neg_()
                    graph.replay()
                    torch.testing.assert_close(captured, -expected, atol=0, rtol=0)

    def test_strided_three_dimensional_input_and_weight(self):
        x = torch.randn(2, 17, 74, device="cuda", dtype=torch.bfloat16)[..., ::2]
        weight = torch.randn(37, 17, device="cuda", dtype=torch.bfloat16).T
        actual = batch_invariant_linear(x, weight, out_dtype=torch.float32)
        expected = batch_invariant_linear(
            x.contiguous(), weight.contiguous(), out_dtype=torch.float32
        )
        self.assertEqual(actual.shape, (2, 17, 17))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reference = (x.cpu().double() @ weight.cpu().double().T).float().cuda()
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-3)

    def test_rejects_unsupported_operands(self):
        x = torch.zeros(1, 37, device="cuda", dtype=torch.bfloat16)
        weight = torch.zeros(17, 37, device="cuda", dtype=torch.bfloat16)
        for bad_x, bad_weight in (
            (x.float(), weight),
            (x.cpu(), weight),
            (x, weight[:, :-1]),
            (x, weight[:0]),
        ):
            self.assertFalse(is_supported(bad_x, bad_weight))
            with self.assertRaisesRegex(ValueError, "SM120 BF16"):
                batch_invariant_linear(bad_x, bad_weight, out_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "BF16 or FP32"):
            batch_invariant_linear(x, weight, out_dtype=torch.float64)

        big_x = x[:, :1].expand(65536, 128)
        big_w = weight[:1, :1].expand(32768, 128)
        self.assertFalse(is_supported(big_x, big_w))
        with self.assertRaisesRegex(ValueError, "SM120 BF16"):
            batch_invariant_linear(big_x, big_w, out_dtype=torch.float32)

    def test_nondefault_device_and_stream(self):
        if torch.cuda.device_count() < 2 or not is_sm120(1):
            self.skipTest("Requires >= 2 GPUs with SM120 on device 1")
        prev = torch.cuda.current_device()
        dev1 = torch.device("cuda", 1)
        try:
            with torch.cuda.device(dev1):
                weight = torch.randn(64, 256, device=dev1, dtype=torch.bfloat16) * 0.03
                x = torch.randn(17, 256, device=dev1, dtype=torch.bfloat16)
                expected = batch_invariant_linear(x, weight, out_dtype=torch.float32)
                torch.cuda.synchronize(dev1)
            torch.cuda.set_device(0)
            actual = batch_invariant_linear(x, weight, out_dtype=torch.float32)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(actual.device, dev1)
            torch.cuda.synchronize(dev1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            stream1 = torch.cuda.Stream(device=dev1)
            with torch.cuda.device(dev1), torch.cuda.stream(stream1):
                x_stream = torch.empty_like(x)
                x_stream.copy_(x)
                stream_out = batch_invariant_linear(
                    x_stream, weight, out_dtype=torch.float32
                )
                done = torch.cuda.Event()
                done.record(stream1)
            torch.cuda.set_device(0)
            done.synchronize()
            self.assertEqual(stream_out.device, dev1)
            torch.testing.assert_close(stream_out, expected, rtol=0, atol=0)
        finally:
            torch.cuda.set_device(prev)


if __name__ == "__main__":
    unittest.main()
