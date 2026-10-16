"""Inference dispatch preserves the mHC kernel, aliasing, and training path."""

import importlib
import json
import os
import statistics
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

_ops = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.post"
)


class TestMhcPostInference(unittest.TestCase):
    def inputs(self, tokens, dim=4096):
        return (
            torch.randn(1, tokens, dim, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, tokens, 4, dim, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, tokens, 4, 1, device="cuda"),
            torch.randn(1, tokens, 4, 4, device="cuda"),
        )

    def test_exact_alias_and_async(self):
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(736)
        for mode in (torch.no_grad, torch.inference_mode):
            with mode():
                for tokens in (1, 81, 513, 5103, 9889):
                    values = self.inputs(tokens)
                    expected = _ops.MHCPost.apply(*values, None)
                    owned = _ops.mhc_post(*values)
                    torch.testing.assert_close(owned, expected, rtol=0, atol=0)
                    frozen = owned.clone()
                    for inplace in (False, True):
                        arguments = tuple(v.clone() for v in values)
                        output = (
                            arguments[1] if inplace else torch.empty_like(values[1])
                        )
                        with patch.object(
                            _ops.MHCPost,
                            "apply",
                            side_effect=AssertionError("autograd in inference"),
                        ):
                            result = _ops.mhc_post(*arguments, out=output)
                        self.assertIs(result, output)
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    torch.testing.assert_close(owned, frozen, rtol=0, atol=0)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    _ops.mhc_post(*values)
                    stream.synchronize()
                    torch.cuda._sleep(100_000_000)
                    pending = torch.cuda.Event()
                    pending.record()
                    actual = _ops.mhc_post(*values)
                    self.assertFalse(pending.query())
                stream.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_gradients_unchanged(self):
        torch.manual_seed(737)
        values = self.inputs(7, dim=256)
        results = []
        for fn in (_ops.MHCPost.apply, _ops.mhc_post):
            arguments = tuple(v.detach().clone().requires_grad_() for v in values)
            out = fn(*arguments, None)
            self.assertIsNotNone(out.grad_fn)
            out.float().sum().backward()
            results.append((out.detach(), *(v.grad for v in arguments)))
        for before, after in zip(*results):
            self.assertIsNotNone(before)
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    @torch.inference_mode()
    def test_wrapper_submission_benchmark(self):
        rows = []
        torch.manual_seed(738)
        for tokens in (1, 81, 513, 5103, 9889):
            values = self.inputs(tokens)
            output = torch.empty_like(values[1])
            arguments = (*values, output)
            functions = {"reference": _ops.MHCPost.apply, "candidate": _ops.mhc_post}
            for fn in functions.values():
                for _ in range(30):
                    fn(*arguments)
            samples = {name: [] for name in functions}
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for repeat in range(60):
                names = tuple(functions) if repeat % 2 else tuple(reversed(functions))
                for name in names:
                    torch.cuda.synchronize()
                    start.record()
                    begin = time.perf_counter_ns()
                    functions[name](*arguments)
                    cpu_us = (time.perf_counter_ns() - begin) / 1000
                    end.record()
                    end.synchronize()
                    samples[name].append(
                        dict(cpu_us=cpu_us, event_us=start.elapsed_time(end) * 1000)
                    )
            row = dict(tokens=tokens, hidden=4096, hc=4, samples=samples)
            rows.append(row)
            print(
                "MHC_POST_DISPATCH",
                tokens,
                {
                    name: {
                        key: statistics.median(v[key] for v in samples[name])
                        for key in ("cpu_us", "event_us")
                    }
                    for name in functions
                },
                flush=True,
            )
        destination = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "mhc_post_inference.json"
        )
        destination.write_text(
            json.dumps(
                {
                    "metric": "Full wrapper CPU submission and CUDA event stream latency, not kernel-only",
                    "warmup": 30,
                    "alternating_samples": 60,
                    "rows": rows,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
