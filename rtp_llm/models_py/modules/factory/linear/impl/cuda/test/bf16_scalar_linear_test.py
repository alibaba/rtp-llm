"""Correctness and graph coverage for batch-invariant shared-expert gates."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.triton_kernels.common.scalar_linear import (
    maybe_bf16_scalar_linear,
    maybe_fp32_scalar_linear,
)
from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import fusion_phase


@contextmanager
def _tf32_setting(value):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = value
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class ScalarLinearTest(unittest.TestCase):
    def test_batch_invariance_graph_and_strides(self):
        torch.manual_seed(917)
        for width in (127, 4096, 8192):
            weight = torch.randn(1, width * 2, device="cuda", dtype=torch.bfloat16)[
                :, ::2
            ]
            rows = torch.randn(9, width * 2, device="cuda", dtype=torch.bfloat16)[
                :, ::2
            ]
            for bias in (
                None,
                torch.tensor([0.125], device="cuda", dtype=torch.bfloat16),
            ):
                ref = maybe_bf16_scalar_linear(rows, weight, bias)
                exact = F.linear(
                    rows.double(),
                    weight.double(),
                    None if bias is None else bias.double(),
                )
                torch.testing.assert_close(ref.double(), exact, rtol=0.004, atol=2e-5)
                for batch in (1, 2, 8, 32, 128, 512):
                    x = torch.randn(
                        batch, width * 2, device="cuda", dtype=torch.bfloat16
                    )[:, ::2]
                    target = batch // 2
                    x[target] = rows[0]
                    maybe_bf16_scalar_linear(x, weight, bias)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        y = maybe_bf16_scalar_linear(x, weight, bias)
                    for row in (0, 5, 8):
                        x[target] = rows[row]
                        graph.replay()
                        self.assertTrue(torch.equal(y[target], ref[row]))

    def test_large_prefill_keeps_original_projection(self):
        x = torch.randn(257, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(4096, 1, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "1"}):
            op = CudaF16Linear(w)
        self.assertTrue(torch.equal(op(x), F.linear(x, w.T)))

    def test_empty_and_fallback(self):
        w = torch.ones(1, 4096, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(maybe_bf16_scalar_linear(w[:0], w).shape, (0, 1))
        self.assertIsNone(maybe_bf16_scalar_linear(w.cpu(), w.cpu()))
        self.assertIsNone(maybe_bf16_scalar_linear(w.float(), w.float()))
        self.assertIsNone(maybe_bf16_scalar_linear(w, w.repeat(2, 1)))
        large = torch.ones(1, 8193, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(maybe_bf16_scalar_linear(large, large))

    def test_option_is_snapshotted_and_other_projections_unchanged(self):
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
        for outputs in (1, 64):
            w = torch.randn(4096, outputs, device="cuda", dtype=torch.bfloat16)
            with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "1"}):
                op = CudaF16Linear(w)
            with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "0"}):
                expected = (
                    maybe_bf16_scalar_linear(x, w.T)
                    if outputs == 1
                    else F.linear(x, w.T)
                )
                self.assertTrue(torch.equal(op(x), expected))
        with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "0"}):
            op = CudaF16Linear(w[:, :1])
        self.assertTrue(torch.equal(op(x), F.linear(x, w[:, :1].T)))


class Fp32SharedScalarDecodeTest(unittest.TestCase):
    def test_phase_scope_cache_and_unsupported_inputs(self):
        torch.manual_seed(303)
        weight = torch.randn(4096, 1, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
        for old_flag in ("0", "1"):
            with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": old_flag}):
                selected, original = CudaF16Linear(weight), CudaF16Linear(weight)
            self.assertTrue(selected.enable_fp32_shared_scalar_decode())
            cache = selected._shared_scalar_weight_fp32
            self.assertEqual(cache.dtype, torch.float32)
            self.assertTrue(torch.equal(cache.bfloat16(), selected.weight))
            self.assertIsNone(original._shared_scalar_weight_fp32)
            with fusion_phase(is_prefill=True):
                self.assertTrue(torch.equal(selected(x), original(x)))
            with _tf32_setting(False):
                expected = F.linear(x.float(), weight.T.float()).bfloat16()
                with fusion_phase(is_prefill=False):
                    self.assertTrue(torch.equal(selected(x), expected))
                    self.assertEqual(selected(x[:0]).shape, (0, 1))
                    strided = torch.randn(8, 8192, device="cuda", dtype=torch.bfloat16)[
                        :, ::2
                    ]
                    self.assertTrue(torch.equal(selected(strided), original(strided)))
            self.assertIs(selected._shared_scalar_weight_fp32, cache)
        for candidate in (
            CudaF16Linear(weight.cpu()),
            CudaF16Linear(weight.float()),
            CudaF16Linear(weight[:, :1].repeat(1, 2)),
            CudaF16Linear(
                weight, bias=torch.zeros(1, device="cuda", dtype=weight.dtype)
            ),
        ):
            self.assertFalse(candidate.enable_fp32_shared_scalar_decode())
        self.assertIsNone(maybe_fp32_scalar_linear(x.float(), cache))
        self.assertIsNone(maybe_fp32_scalar_linear(x.cpu(), cache.cpu()))

    def test_tf32_restored_after_success_and_exception(self):
        x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(1, 4096, device="cuda", dtype=torch.float32)
        real_linear = F.linear
        for initial in (False, True):
            with _tf32_setting(initial):

                def checked_linear(*args):
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    return real_linear(*args)

                with patch.object(F, "linear", side_effect=checked_linear):
                    maybe_fp32_scalar_linear(x, weight)
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, initial)
                with patch.object(
                    F, "linear", side_effect=RuntimeError("test failure")
                ):
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        maybe_fp32_scalar_linear(x, weight)
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, initial)

    def test_concurrent_calls_serialize_precision_switch(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock
        from time import sleep

        x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(1, 4096, device="cuda", dtype=torch.float32)
        state = {"active": 0, "peak": 0}
        lock = Lock()

        def checked_linear(value, *_args):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            sleep(0.01)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            with lock:
                state["active"] -= 1
            return value[:, :1]

        with _tf32_setting(True):
            with patch.object(F, "linear", side_effect=checked_linear):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [
                        pool.submit(maybe_fp32_scalar_linear, x, weight)
                        for _ in range(8)
                    ]
                    for future in futures:
                        self.assertEqual(future.result().dtype, torch.bfloat16)
            self.assertEqual(state["peak"], 1)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

    def test_original_13_graph_sizes_and_dynamic_inputs(self):
        torch.manual_seed(304)
        layer = CudaF16Linear(torch.randn(4096, 1, device="cuda", dtype=torch.bfloat16))
        self.assertTrue(layer.enable_fp32_shared_scalar_decode())
        cache = layer._shared_scalar_weight_fp32
        with fusion_phase(is_prefill=False):
            with _tf32_setting(True):
                for rows in (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256):
                    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            layer(x)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        y = layer(x)
                    self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                    for _ in range(2):
                        x.copy_(torch.randn_like(x))
                        graph.replay()
                        with _tf32_setting(False):
                            expected = F.linear(x.float(), cache).bfloat16()
                        self.assertTrue(torch.equal(y, expected), f"rows={rows}")
                    self.assertIs(layer._shared_scalar_weight_fp32, cache)


if __name__ == "__main__":
    unittest.main()
