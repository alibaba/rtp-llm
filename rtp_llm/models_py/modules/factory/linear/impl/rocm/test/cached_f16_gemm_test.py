"""ROCm correctness/cache tests; no service requests or profiler attachment."""

import concurrent.futures
import itertools
import unittest

import torch


@unittest.skipUnless(
    torch.version.hip and torch.cuda.is_available(), "ROCm GPU required"
)
class CachedF16GemmTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.ops.compute_ops import rtp_llm_ops
        from rtp_llm.utils.swizzle_utils import swizzle_tensor

        self.mm = rtp_llm_ops.rocm_cached_f16_gemm
        self.stats = rtp_llm_ops.rocm_f16_gemm_cache_stats
        self.swizzle = swizzle_tensor
        torch.manual_seed(42)

    def weight(self, weight, layout):
        if layout == "swizzle":
            return self.swizzle(weight.t(), False, MiM=16).t()
        if layout == "column":
            return weight.t().contiguous().t()
        return weight

    def test_layout_dtype_bias_and_dynamic_shapes(self):
        for dtype, layout, has_bias in itertools.product(
            (torch.float16, torch.bfloat16), ("row", "column", "swizzle"), (False, True)
        ):
            w = torch.randn(256, 512, device="cuda", dtype=dtype) * 0.02
            packed = self.weight(w, layout)
            for q in (1, 2, 3, 4, 14, 15, 16, 128):
                with self.subTest(dtype=dtype, layout=layout, bias=has_bias, q=q):
                    x = torch.randn(q, 256, device="cuda", dtype=dtype)
                    # Change bias and weight pointers on cache hits, as in distinct layers.
                    bias = (
                        torch.randn(512, device="cuda", dtype=dtype)
                        if has_bias
                        else None
                    )
                    for _ in range(2):
                        actual = self.mm(x, packed.clone(), bias, layout == "swizzle")
                        self.assertIsNotNone(actual)
                        expected = x.float() @ w.float()
                        if bias is not None:
                            expected += bias.float()
                        torch.testing.assert_close(
                            actual, expected.to(dtype), atol=0.02, rtol=0.02
                        )

    def test_trace_shapes_match_aiter_and_hit_cache(self):
        from aiter import hipb_create_extension, hipb_mm

        hipb_create_extension()
        for k, n in ((2560, 6144), (4096, 2560), (2560, 19456), (9728, 2560)):
            w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.01
            w = self.weight(w, "swizzle")
            for q in (1, 4, 14, 15, 16):
                x = torch.randn(q, k, device="cuda", dtype=w.dtype)
                expected = hipb_mm(x, w, -1, bpreshuffle=True)
                first = self.mm(x, w, None, True)
                before = self.stats()
                second = self.mm(x, w, None, True)
                after = self.stats()
                self.assertEqual(after[0], before[0] + 1)
                self.assertEqual(after[1:3], before[1:3])
                self.assertLessEqual(after[3], 256 * 1024 * 1024)
                torch.testing.assert_close(first, expected, atol=0, rtol=0)
                torch.testing.assert_close(second, expected, atol=0, rtol=0)

    def test_strides_pointer_changes_and_bias_isolation(self):
        for transposed in (False, True):
            for _ in range(3):
                x = torch.randn(7, 128, device="cuda", dtype=torch.float16)
                if transposed:
                    x = x.t().contiguous().t()
                else:
                    x = torch.cat((x, x), dim=1)[:, :128]  # padded leading dimension
                w = torch.randn(128, 64, device="cuda", dtype=x.dtype) * 0.03
                bias = torch.randn(64, device="cuda", dtype=x.dtype)
                actual = self.mm(x, w, bias, False)
                expected = (x.float() @ w.float() + bias.float()).to(x.dtype)
                torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)

    def test_capture_falls_back_without_mutating_cache(self):
        x = torch.randn(14, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, 512, device="cuda", dtype=x.dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.mm(x, w, None, False)
            before = self.stats()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = self.mm(x, w, None, False)
            output = x + 1  # keep the graph nonempty
        self.assertIsNone(result)
        with torch.cuda.stream(stream):
            self.assertEqual(before, self.stats())
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, x + 1)

    def test_unsupported_inputs_return_none(self):
        x = torch.randn(14, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, 512, device="cuda", dtype=x.dtype)
        self.assertIsNone(self.mm(x.float(), w.float(), None, False))
        self.assertIsNone(self.mm(x[:, ::2], w[:128], None, False))
        self.assertIsNone(self.mm(x[:0], w, None, False))
        self.assertIsNone(self.mm(x, w, None, True))  # not a packed column layout
        self.assertIsNone(self.mm(x.requires_grad_(), w, None, False))

    def test_thread_stream_and_device_isolation(self):
        def worker(device, seed):
            torch.cuda.set_device(device)
            for _ in range(2):
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    x = torch.full((15, 128), seed, device=device, dtype=torch.float16)
                    w = torch.full((128, 64), 0.01, device=device, dtype=x.dtype)
                    actual = self.mm(x, w, None, False)
                    torch.testing.assert_close(actual, x @ w, atol=0.01, rtol=0.01)
                stream.synchronize()

        devices = list(range(min(2, torch.cuda.device_count())))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [devices[0], devices[-1]], [1, 2]))

    def test_plan_capacity_is_bounded(self):
        # Fresh thread: no previous test has consumed its per-thread contexts.
        def worker():
            x = torch.ones((270, 64), device="cuda", dtype=torch.float16)
            w = torch.ones((64, 64), device="cuda", dtype=x.dtype)
            for q in range(1, 271):
                actual = self.mm(x[:q], w, None, False)
                if q <= 256:
                    self.assertIsNotNone(actual)
                else:
                    self.assertIsNone(actual)
            self.assertEqual(self.stats()[2], 256)
            torch.cuda.synchronize()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(worker).result()


if __name__ == "__main__":
    unittest.main()
