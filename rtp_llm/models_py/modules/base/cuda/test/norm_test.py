import itertools
import statistics
import time
from unittest import SkipTest, TestCase, main, mock

import torch
from rtp_llm.models_py.modules import RMSNorm, RMSNormTorch
from rtp_llm.ops.compute_ops import rtp_llm_ops
from torch import dtype as _dtype


class NormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_norm_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(hidden_size, dtype=dtype)
        rms_norm = RMSNorm(w)
        rms_norm_torch = RMSNormTorch(w)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        self.assertTrue(
            torch.allclose(rms_norm_torch(x), rms_norm(x), atol=1e-2, rtol=1e-2)
        )

    def test_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_rms_norm_test(*params)

    def test_current_stream_is_resolved_inside_binding(self):
        width = 4096
        x = torch.randn((97, width), dtype=torch.bfloat16)
        weight = torch.randn(width, dtype=torch.bfloat16)
        norm = RMSNorm(weight)
        output = torch.empty_like(x)
        stream = torch.cuda.Stream()

        with torch.cuda.stream(stream):
            with mock.patch.object(
                torch.cuda,
                "current_stream",
                side_effect=AssertionError("Python current_stream must not be called"),
            ):
                self.assertIs(norm(x, output), output)
            done = stream.record_event()

        torch.cuda.current_stream().wait_event(done)
        torch.testing.assert_close(
            output.float(), RMSNormTorch(weight)(x).float(), atol=1e-2, rtol=1e-2
        )

    def test_current_stream_binding_cpu_overhead(self):
        width = 4096
        x = torch.randn((419, width), dtype=torch.bfloat16)
        weight = torch.randn(width, dtype=torch.bfloat16)
        output = torch.empty_like(x)

        def baseline():
            rtp_llm_ops.rmsnorm(
                output, x, weight, 1e-6, torch.cuda.current_stream().cuda_stream
            )

        def candidate():
            rtp_llm_ops.rmsnorm_current_stream(output, x, weight, 1e-6)

        for _ in range(30):
            baseline()
            candidate()
        torch.cuda.synchronize()

        samples = {"baseline": [], "candidate": []}
        for _ in range(60):
            for name, fn in (("baseline", baseline), ("candidate", candidate)):
                torch.cuda.synchronize()
                begin = time.thread_time_ns()
                for _ in range(100):
                    fn()
                samples[name].append((time.thread_time_ns() - begin) / 100_000.0)
        torch.cuda.synchronize()

        baseline_us = statistics.median(samples["baseline"])
        candidate_us = statistics.median(samples["candidate"])
        print(
            "rmsnorm current-stream CPU us/call: "
            f"baseline={baseline_us:.3f}, candidate={candidate_us:.3f}"
        )
        self.assertLess(candidate_us, baseline_us)

    def test_inplace_output_exact_graph_and_cpu_overhead(self):
        width = 4096
        weight = torch.randn(width, dtype=torch.bfloat16)
        norm = RMSNorm(weight)
        for tokens in (1, 81, 419, 5103):
            with self.subTest(tokens=tokens):
                source = torch.randn((tokens, width), dtype=torch.bfloat16)
                expected = norm(source)
                inplace = source.clone()
                self.assertIs(norm(inplace, inplace), inplace)
                self.assertTrue(torch.equal(inplace, expected))

        stream = torch.cuda.Stream()
        source = torch.randn((5103, width), dtype=torch.bfloat16)
        expected = norm(source)
        inplace = source.clone()
        with torch.cuda.stream(stream):
            self.assertIs(norm(inplace, inplace), inplace)
            done = stream.record_event()
        torch.cuda.current_stream().wait_event(done)
        self.assertTrue(torch.equal(inplace, expected))

        static_input = torch.randn((419, width), dtype=torch.bfloat16)
        original = static_input.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = norm(static_input, static_input)
        expected = norm(original)
        static_input.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        self.assertIs(graph_output, static_input)
        self.assertTrue(torch.equal(graph_output, expected))

        source = torch.randn((419, width), dtype=torch.bfloat16)
        inplace = source.clone()

        def baseline():
            norm(source)

        def candidate():
            norm(inplace, inplace)

        for _ in range(30):
            baseline()
            candidate()
        torch.cuda.synchronize()
        samples = {"baseline": [], "candidate": []}
        for _ in range(60):
            for name, fn in (("baseline", baseline), ("candidate", candidate)):
                torch.cuda.synchronize()
                begin = time.thread_time_ns()
                for _ in range(100):
                    fn()
                samples[name].append((time.thread_time_ns() - begin) / 100_000.0)
        baseline_us = statistics.median(samples["baseline"])
        candidate_us = statistics.median(samples["candidate"])
        print(
            "rmsnorm inplace CPU us/call: "
            f"baseline={baseline_us:.3f}, candidate={candidate_us:.3f}"
        )
        self.assertLess(candidate_us, baseline_us)

    def test_rms_norm_large_row_offsets(self):
        # Exercise the actual >2**32 element address boundary, without
        # constructing a full FP32 reference of the 8 GiB activation.
        if torch.cuda.mem_get_info()[0] < 20 * 1024**3:
            raise SkipTest("large RMSNorm address test requires 20 GiB free HBM")
        rows, width = (1 << 20) + 3, 4096
        sample_rows = [0, (1 << 19), (1 << 20) - 1, 1 << 20, rows - 1]
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                x = torch.ones((rows, width), dtype=dtype)
                x[-1, : width // 2] = 2
                output = torch.full_like(x, float("nan"))
                norm = RMSNorm(torch.ones(width, dtype=dtype))
                self.assertIs(norm(x, output), output)
                reference = x[sample_rows].float()
                reference *= torch.rsqrt(
                    reference.square().mean(-1, keepdim=True) + norm.variance_epsilon
                )
                torch.testing.assert_close(
                    output[sample_rows].float(), reference, atol=1e-2, rtol=1e-2
                )
                # Capture both launches and change the tail after capture.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    norm(x, output)
                x[-1].fill_(3)
                graph.replay()
                torch.testing.assert_close(
                    output[-1], torch.ones_like(output[-1]), atol=1e-2, rtol=1e-2
                )
                del graph, x, output, norm
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
