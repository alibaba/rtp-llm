"""Synthetic BF16 gated-norm/layout correctness and path-level CUDA-event A/B."""

import hashlib
import inspect
import json
import math
import os
import socket
import statistics
import unittest
from collections import Counter
from pathlib import Path

import torch
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import FusedRMSNormGated


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inputs(natural_rows, physical_rows, heads):
    x = torch.randn(natural_rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    # Packed head dimensions with a non-packed token stride, as produced by KDA.
    storage = torch.randn(
        natural_rows, heads * 128 + 32, device="cuda", dtype=torch.bfloat16
    )
    gate = storage[:, : heads * 128].view(natural_rows, heads, 128)
    positions = torch.randperm(physical_rows, device="cuda")[:natural_rows]
    inverse = torch.full((physical_rows,), -1, device="cuda", dtype=torch.int64)
    inverse.index_copy_(
        0, positions, torch.arange(natural_rows, device="cuda", dtype=torch.int64)
    )
    # Valid token holes retain their mapping but have a masked, zero core output.
    x[::5].zero_()
    return x, gate, positions, inverse


def _baseline(norm, x, gate, positions, out):
    values = norm(x, gate)
    out.zero_()
    out.index_copy_(0, positions, values)
    return out


def _samples(function, count=24):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for start, end in zip(starts, ends):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _summary(samples):
    ordered = sorted(samples)
    return {
        "p50_us": statistics.median(ordered),
        "p90_us": ordered[math.ceil(len(ordered) * 0.9) - 1],
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


class KdaNormLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This Bazel test requires a CUDA GPU")
        torch.cuda.set_device(0)
        torch.manual_seed(20261104)
        cls.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        props = torch.cuda.get_device_properties(0)
        cls.report = {
            "workload": "synthetic KDA output gated RMSNorm and token-layout conversion",
            "dtype": "bfloat16",
            "head_dim": 128,
            "tp_size": 1,
            "execution": "eager CUDA events; CUDA Graph correctness checked separately",
            "timing_boundary": {
                "baseline": "production norm -> output zero_ -> index_copy_",
                "candidate": "production mapped norm into caller-owned output",
                "included": "all path kernels and per-call output allocation",
                "excluded": "input/map creation, initialization, JIT and warmup",
            },
            "hardware": {
                "host": socket.gethostname(),
                "gpu": props.name,
                "uuid": str(getattr(props, "uuid", "unknown")),
                "capability": [props.major, props.minor],
                "sm_count": props.multi_processor_count,
            },
            "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
            "source": {
                "commit": os.environ.get("RTP_TEST_SOURCE_SHA"),
                "diff_sha256": os.environ.get("RTP_TEST_SOURCE_DIFF_SHA256"),
                "test_sha256": _sha256(__file__),
                "kernel_sha256": _sha256(inspect.getsourcefile(FusedRMSNormGated)),
            },
            "quantiles": "p50 median; p90 nearest rank ceil(0.90*N)",
            "correctness": [],
            "benchmarks": [],
        }

    @classmethod
    def tearDownClass(cls):
        path = cls.output_dir / "kda_norm_layout.json"
        path.write_text(json.dumps(cls.report, indent=2) + "\n")
        print("KDA_NORM_LAYOUT_RESULT", path, flush=True)

    def _norm(self):
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        return FusedRMSNormGated(weight, 1e-6)

    def _check(self, norm, inputs, physical_rows, out=None):
        x, gate, positions, inverse = inputs
        shape = (physical_rows, x.shape[1], 128)
        if out is None:
            out = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        expected = _baseline(norm, x, gate, positions, torch.empty_like(out))
        out.fill_(float("nan"))
        actual = norm(x, gate, row_indices=inverse, out=out)
        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        padding = inverse < 0
        self.assertEqual(torch.count_nonzero(actual[padding]).item(), 0)
        return out

    def test_layout_correctness(self):
        for natural_rows, physical_rows, heads in (
            (0, 9, 6),
            (1, 5, 6),
            (17, 33, 12),
            (37, 64, 48),
            (67, 73, 6),
            (64, 64, 12),
        ):
            with self.subTest(rows=natural_rows, physical=physical_rows, heads=heads):
                inputs = _inputs(natural_rows, physical_rows, heads)
                self._check(self._norm(), inputs, physical_rows)
                self.report["correctness"].append(
                    {
                        "natural_rows": natural_rows,
                        "physical_rows": physical_rows,
                        "heads": heads,
                        "strided_gate": True,
                        "masked_core_rows": "every fifth natural row",
                        "comparison": "bitwise equality; all padding zero",
                    }
                )

    def test_reuse_and_cuda_graph(self):
        natural_rows, physical_rows, heads = 53, 79, 12
        norm = self._norm()
        inputs = _inputs(natural_rows, physical_rows, heads)
        x, gate, positions, inverse = inputs
        out = self._check(norm, inputs, physical_rows)
        for _ in range(3):
            norm(x, gate, row_indices=inverse, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            norm(x, gate, row_indices=inverse, out=out)
        for iteration in range(3):
            live = _inputs(natural_rows, physical_rows, heads)
            for destination, source in zip(inputs, live):
                destination.copy_(source)
            # Change every live input, including the map captured at a fixed address.
            out.fill_(float("nan"))
            graph.replay()
            expected = norm(x, gate, row_indices=inverse, out=torch.empty_like(out))
            torch.cuda.synchronize()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            baseline = _baseline(norm, x, gate, positions, torch.empty_like(out))
            torch.testing.assert_close(out, baseline, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(out).all().item())
            self.assertEqual(torch.count_nonzero(out[inverse < 0]).item(), 0)
            self._check(norm, inputs, physical_rows, out=out)
            self.report["correctness"].append(
                {
                    "cuda_graph_replay": iteration,
                    "changed_x_gate_and_map": True,
                    "comparison": "bitwise equality to fresh same-implementation eager and baseline",
                }
            )

    def test_ab_microbenchmark(self):
        for physical_rows in (16, 4096, 32768):
            for heads in (6, 12, 48):
                natural_rows = physical_rows - max(1, physical_rows // 16)
                inputs = _inputs(natural_rows, physical_rows, heads)
                x, gate, positions, inverse = inputs
                norm = self._norm()
                out = self._check(norm, inputs, physical_rows)
                baseline_out = torch.empty_like(out)
                paths = {
                    "baseline": lambda: _baseline(
                        norm, x, gate, positions, baseline_out
                    ),
                    "candidate": lambda: norm(x, gate, row_indices=inverse, out=out),
                }
                for function in paths.values():
                    for _ in range(8):
                        function()
                torch.cuda.synchronize()
                rounds = []
                for order in (("baseline", "candidate"), ("candidate", "baseline")):
                    raw = {label: _samples(paths[label]) for label in order}
                    summaries = {
                        label: _summary(values) for label, values in raw.items()
                    }
                    rounds.append(
                        {
                            "order": order,
                            "raw_us": raw,
                            "summary": summaries,
                            "speedup": summaries["baseline"]["p50_us"]
                            / summaries["candidate"]["p50_us"],
                        }
                    )
                self.report["benchmarks"].append(
                    {
                        "natural_rows": natural_rows,
                        "physical_rows": physical_rows,
                        "heads": heads,
                        "gate_stride": list(gate.stride()),
                        "rounds": rounds,
                    }
                )
        if os.environ.get("KDA_NORM_PROFILE", "0") == "1":
            self._profile()

    def _profile(self):
        norm = self._norm()
        x, gate, positions, inverse = _inputs(239, 257, 12)
        out = torch.empty(257, 12, 128, device="cuda", dtype=torch.bfloat16)
        paths = {
            "baseline": lambda: _baseline(norm, x, gate, positions, out),
            "candidate": lambda: norm(x, gate, row_indices=inverse, out=out),
        }
        result = {}
        for label, function in paths.items():
            function()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                with torch.profiler.record_function(label):
                    function()
                torch.cuda.synchronize()
            trace = self.output_dir / (label + "_norm_layout_trace.json")
            profile.export_chrome_trace(str(trace))
            events = Counter(
                event["name"]
                for event in json.loads(trace.read_text())["traceEvents"]
                if event.get("cat") == "kernel"
            )
            self.assertGreater(sum(events.values()), 0)
            result[label] = {"trace": trace.name, "cuda_events": dict(events)}
        self.report["profile"] = result


if __name__ == "__main__":
    unittest.main()
