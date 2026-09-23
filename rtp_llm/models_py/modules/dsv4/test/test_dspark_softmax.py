"""Numerical and stream isolation checks for the existing FlashInfer softmax."""

import json
import os
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.dspark_softmax import dspark_softmax


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DSparkSoftmaxTest(unittest.TestCase):
    def check_probabilities(self, logits, actual):
        expected = torch.softmax(logits, -1)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-8)
        torch.testing.assert_close(
            actual.sum(-1), torch.ones_like(actual[:, 0]), rtol=2e-6, atol=2e-6
        )
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue((actual >= 0).all().item())

    def test_shapes_and_extreme_finite_logits(self):
        torch.manual_seed(21)
        for batch in (1, 4, 8, 17):
            for vocab in (31, 32768, 129280, 129283):
                for scale in (0.0, 1.0, 100.0):
                    logits = torch.randn(batch, vocab, device="cuda") * scale
                    self.check_probabilities(logits, dspark_softmax(logits))

    def test_negative_infinity_mask_and_fallback(self):
        logits = torch.randn(4, 129280, device="cuda")
        logits[:, ::2] = -torch.inf
        self.check_probabilities(logits, dspark_softmax(logits))
        view = logits[:, ::2]
        view[:, 0] = 0
        self.check_probabilities(view, dspark_softmax(view))

    def test_concurrent_graphs_have_independent_scratch(self):
        inputs = [torch.randn(batch, 129280, device="cuda") for batch in (4, 8)]
        streams = [torch.cuda.Stream() for _ in inputs]
        graphs, outputs = [], []
        for stream, logits in zip(streams, inputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    dspark_softmax(logits)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = dspark_softmax(logits)
            graphs.append(graph)
            outputs.append(output)
        for scale in (0.0, 1.0, 20.0):
            for logits in inputs:
                logits.copy_(torch.randn_like(logits) * scale)
            for stream, graph in zip(streams, graphs):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(20):
                        graph.replay()
            for stream in streams:
                torch.cuda.current_stream().wait_stream(stream)
            for logits, output in zip(inputs, outputs):
                self.check_probabilities(logits, output)

    @unittest.skipUnless(os.environ.get("DSPARK_SOFTMAX_BENCH"), "opt-in benchmark")
    def test_benchmark(self):
        results = []
        for batch in (1, 4, 8, 17, 32):
            logits = torch.randn(batch, 129280, device="cuda")
            row = {"batch": batch}
            for name, fn in (
                ("torch", lambda: torch.softmax(logits, -1)),
                ("flashinfer", lambda: dspark_softmax(logits)),
            ):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        fn()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(50):
                        fn()
                for _ in range(10):
                    graph.replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                row[name + "_us"] = start.elapsed_time(end)
            results.append(row)
        content = json.dumps(results, indent=2)
        print(content)
        destination = os.environ.get("DSPARK_SOFTMAX_BENCH_JSON")
        if destination:
            Path(destination).write_text(content + "\n")


if __name__ == "__main__":
    unittest.main()
