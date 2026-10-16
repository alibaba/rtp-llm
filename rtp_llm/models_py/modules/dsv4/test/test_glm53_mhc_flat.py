"""Exercise the flat mHC path against the existing batched adapter."""

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
from rtp_llm.models_py.modules.dsv4.hc import tilelang_impl as impl

post_ops = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.post"
)


class TestMhcFlat(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        env = patch.dict(
            os.environ,
            {
                "DSV4_MHC_PRE_GEMM_BACKEND": "tilelang_single",
                "GLM53_MHC_PRECOMPUTE_FN": "1",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        enabled = patch.object(impl, "_FLAT_PREFILL", True)
        enabled.start()
        self.addCleanup(enabled.stop)
        torch.manual_seed(1217)

    def layer(self, dim):
        return impl.TileLangHCUnit(
            torch.randn(24, 4 * dim, device="cuda") * 0.02,
            torch.randn(24, device="cuda") * 0.02,
            torch.ones(3, device="cuda"),
            dim=dim,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )

    def assert_bits(self, old, new):
        for a, b in zip(old, new):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.stride(), b.stride())
            self.assertTrue(
                torch.equal(
                    a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
                )
            )

    @torch.inference_mode()
    def test_exact_layouts_and_alias(self):
        for dim in (256, 4096, 6144):
            layer = self.layer(dim)
            for leading in ((1,), (81,), (419,), (5103,), (9889,), (2, 13)):
                residual = torch.randn(
                    *leading, 4, dim, device="cuda", dtype=torch.bfloat16
                )
                frozen = residual.clone()
                with patch.object(impl, "_FLAT_PREFILL", False):
                    expected_pre = layer.pre(residual)
                    expected_post = layer.post(
                        expected_pre[0], residual.clone(), *expected_pre[1:]
                    )
                actual_pre = layer.pre(residual)
                self.assert_bits(expected_pre, actual_pre)
                self.assert_bits((frozen,), (residual,))
                saved_pre = tuple(t.clone() for t in actual_pre)
                result = layer.post(actual_pre[0], residual, *actual_pre[1:])
                self.assertEqual(result.data_ptr(), residual.data_ptr())
                self.assert_bits((expected_post,), (result,))
                self.assert_bits(saved_pre, actual_pre)
                # The standalone no-grad path also supports a separately owned result.
                owned = post_ops.mhc_post(actual_pre[0], frozen, *actual_pre[1:])
                self.assertNotEqual(owned.data_ptr(), frozen.data_ptr())
                self.assert_bits((expected_post,), (owned,))
                layer.pre(residual)
                self.assert_bits(saved_pre, actual_pre)
        layer = self.layer(256)
        flat = torch.randn(13, 4, 256, device="cuda", dtype=torch.bfloat16)
        with patch.object(
            impl, "wrap_hc_batch", side_effect=AssertionError("redundant batch view")
        ):
            values = layer.pre(flat)
            layer.post(values[0], flat, *values[1:])
        for bad in (flat[0], flat[..., :128], flat.transpose(0, 1)):
            with self.assertRaises(ValueError):
                layer.pre(bad)
        noncontiguous = torch.randn(13, 4, 512, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
        with self.assertRaises(ValueError):
            layer.pre(noncontiguous)

    @torch.inference_mode()
    def test_devices_stream_and_capture(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        for device in (0, 1):
            with torch.cuda.device(device):
                layer = self.layer(4096)
                residual = torch.randn(81, 4, 4096, device="cuda", dtype=torch.bfloat16)
                with patch.object(impl, "_FLAT_PREFILL", False):
                    expected = layer.pre(residual)
                    expected_post = layer.post(
                        expected[0], residual.clone(), *expected[1:]
                    )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    layer.pre(residual)
                    stream.synchronize()
                    scratch = residual.clone()
                    torch.cuda._sleep(100_000_000)
                    pending = torch.cuda.Event()
                    pending.record()
                    actual = layer.pre(scratch)
                    actual_post = layer.post(actual[0], scratch, *actual[1:])
                    self.assertFalse(pending.query())
                stream.synchronize()
                self.assert_bits(expected, actual)
                self.assert_bits((expected_post,), (actual_post,))
                scratch.copy_(residual)
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = layer.pre(scratch)
                    captured_post = layer.post(captured[0], scratch, *captured[1:])
                graph.replay()
                self.assert_bits(expected, captured)
                self.assert_bits((expected_post,), (captured_post,))

    def test_batched_training_and_flat_rejection(self):
        values = (
            torch.randn(1, 7, 256, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 7, 4, 256, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 7, 4, 1, device="cuda"),
            torch.randn(1, 7, 4, 4, device="cuda"),
        )
        results = []
        for function in (post_ops.MHCPost.apply, post_ops.mhc_post):
            args = tuple(v.detach().clone().requires_grad_() for v in values)
            output = function(*args, None)
            output.float().sum().backward()
            results.append((output.detach(), *(v.grad for v in args)))
        self.assert_bits(*results)
        with self.assertRaises(AssertionError):
            post_ops.mhc_post(*(v.squeeze(0).requires_grad_() for v in values))

    @torch.inference_mode()
    def test_wrapper_benchmark(self):
        layer = self.layer(4096)
        rows = []
        for tokens in (1, 81, 419, 5103, 9889):
            source = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
            scratch = source.clone()

            def call():
                values = layer.pre(scratch)
                return layer.post(values[0], scratch, *values[1:])

            for enabled in (False, True):
                impl._FLAT_PREFILL = enabled
                for _ in range(30):
                    scratch.copy_(source)
                    call()
            samples = {False: [], True: []}
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for repeat in range(60):
                for enabled in ((False, True) if repeat % 2 else (True, False)):
                    impl._FLAT_PREFILL = enabled
                    scratch.copy_(source)
                    torch.cuda.synchronize()
                    start.record()
                    begin = time.perf_counter_ns()
                    call()
                    cpu_us = (time.perf_counter_ns() - begin) / 1000
                    end.record()
                    end.synchronize()
                    samples[enabled].append(
                        dict(cpu_us=cpu_us, event_us=start.elapsed_time(end) * 1000)
                    )
            print(
                "MHC_FLAT",
                tokens,
                {
                    enabled: {
                        metric: statistics.median(row[metric] for row in values)
                        for metric in values[0]
                    }
                    for enabled, values in samples.items()
                },
                flush=True,
            )
            rows.append(dict(tokens=tokens, samples=samples))
        Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "mhc_flat.json").write_text(
            json.dumps(
                dict(
                    timing="complete HCUnit.pre + HCUnit.post CPU wall and CUDA-event stream latency; reset and input allocations excluded; no profiler",
                    warmup=30,
                    repeats=60,
                    rows=rows,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
