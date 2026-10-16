"""Byte-exact, asynchronous MegaMoE packing and complete dispatch timing."""

import json
import os
import statistics
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.glm5_mega_moe import input_packer_triton as ops
from rtp_llm.models_py.modules.glm5_mega_moe.quant_layouts import (
    per_token_cast_to_fp8_packed_ue8m0,
)


class TestMoePackDispatch(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        enabled = patch.object(ops, "_COMPILED_DISPATCH", True)
        enabled.start()
        self.addCleanup(enabled.stop)
        self.vector = os.environ.get("GLM53_TEST_MOE_PACK_VECTOR", "0") == "1"
        vector = patch.object(ops, "_VECTOR_PACK", self.vector)
        vector.start()
        self.addCleanup(vector.stop)
        torch.manual_seed(917)

    def fixture(self, tokens, width=4096, topk=8, padded=False, unaligned=False):
        def tensor(columns, dtype):
            extra = 16 if padded else 0
            offset = int(unaligned)
            owner = torch.empty(
                (tokens + 2) * (columns + extra) + offset,
                dtype=dtype,
                device="cuda",
            )
            owner.fill_(3)
            view = owner[offset:].view(tokens + 2, columns + extra)[:tokens, :columns]
            return view, owner

        x, _ = tensor(width, torch.bfloat16)
        weights, _ = tensor(topk, torch.float32)
        indices, _ = tensor(topk, torch.int32 if padded else torch.int64)
        x.copy_(torch.randn_like(x) * 3)
        weights.copy_(torch.randn_like(weights))
        indices.copy_(torch.randint(384, indices.shape, device="cuda"))
        outputs = [
            tensor(width, torch.float8_e4m3fn),
            tensor(width // 128, torch.int32),
            tensor(topk, torch.int64),
            tensor(topk, torch.float32),
        ]
        return (x, weights, indices, *(v for v, _ in outputs)), tuple(
            owner for _, owner in outputs
        )

    def assert_bits(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for old, new in zip(expected, actual):
            self.assertEqual(old.dtype, new.dtype)
            self.assertEqual(old.shape, new.shape)
            self.assertTrue(
                torch.equal(
                    old.contiguous().view(torch.uint8),
                    new.contiguous().view(torch.uint8),
                )
            )

    def check(self, args, owners):
        inputs_before = tuple(t.clone() for t in args[:3])
        with patch.object(ops, "_COMPILED_DISPATCH", False), patch.object(
            ops, "_VECTOR_PACK", False
        ):
            ops.fused_pack_mega_moe_inputs(*args)
        expected = tuple(t.clone() for t in owners)
        for owner in owners:
            owner.fill_(3)
        ops.fused_pack_mega_moe_inputs(*args)
        self.assert_bits(expected, owners)
        self.assert_bits(inputs_before, args[:3])
        if args[0].shape[0]:
            fp8, sf = per_token_cast_to_fp8_packed_ue8m0(args[0].contiguous())
            self.assert_bits((fp8, sf), args[3:5])
            self.assert_bits((args[2].to(torch.int64), args[1]), args[5:])

    @torch.inference_mode()
    def test_shapes_layouts_and_block_settings(self):
        for tokens in (0, 1, 13, 81, 419, 2047, 2048, 5103, 9889):
            for width, topk, padded in (
                (128, 5, False),
                (4096, 8, False),
                (6144, 4, True),
            ):
                self.check(*self.fixture(tokens, width, topk, padded=padded))
        args, owners = self.fixture(81)
        for block_m in (1, 2, 4, 8, 3):
            with patch.dict(os.environ, {"GLM5_MEGA_MOE_PACK_BLOCK_M": str(block_m)}):
                self.check(args, owners)
        # Exact exponent boundaries, tiny values and signed zeros.
        values = torch.tensor(
            [0.0, -0.0, 1e-8, -1e-8, 0.0001, 0.5, 1.0, 2.0, 7.0, 448.0, 896.0],
            device="cuda",
            dtype=torch.bfloat16,
        )
        args[0].copy_(
            values[torch.arange(args[0].numel(), device="cuda") % len(values)].view_as(
                args[0]
            )
        )
        self.check(args, owners)
        self.assertGreater(ops._compiled_runner.cache_info().hits, 0)
        self.assertLessEqual(ops._compiled_runner.cache_info().currsize, 128)

    @torch.inference_mode()
    def test_devices_stream_and_capture_fallback(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        for device in (0, 1):
            with torch.cuda.device(device):
                args, owners = self.fixture(81)
                self.check(args, owners)
                expected = tuple(t.clone() for t in owners)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    ops.fused_pack_mega_moe_inputs(*args)
                    stream.synchronize()
                    torch.cuda._sleep(100_000_000)
                    pending = torch.cuda.Event()
                    pending.record()
                    ops.fused_pack_mega_moe_inputs(*args)
                    self.assertFalse(pending.query())
                stream.synchronize()
                self.assert_bits(expected, owners)
                with patch.object(
                    ops,
                    "_compiled_runner",
                    side_effect=AssertionError("must use original dispatch"),
                ):
                    self.check(*self.fixture(13, unaligned=True))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        ops.fused_pack_mega_moe_inputs(*args)
                for owner in owners:
                    owner.fill_(3)
                graph.replay()
                self.assert_bits(expected, owners)

    @torch.inference_mode()
    def test_wrapper_benchmark(self):
        rows = []
        for tokens in (1, 81, 419, 5103, 9889):
            args, owners = self.fixture(tokens)
            self.check(args, owners)
            for enabled in (False, True):
                ops._COMPILED_DISPATCH = enabled
                ops._VECTOR_PACK = enabled and self.vector
                for _ in range(30):
                    ops.fused_pack_mega_moe_inputs(*args)
            samples = {False: [], True: []}
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for repeat in range(60):
                for enabled in ((False, True) if repeat % 2 else (True, False)):
                    ops._COMPILED_DISPATCH = enabled
                    ops._VECTOR_PACK = enabled and self.vector
                    torch.cuda.synchronize()
                    start.record()
                    begin = time.perf_counter_ns()
                    ops.fused_pack_mega_moe_inputs(*args)
                    cpu_us = (time.perf_counter_ns() - begin) / 1000
                    end.record()
                    end.synchronize()
                    samples[enabled].append(
                        dict(cpu_us=cpu_us, event_us=start.elapsed_time(end) * 1000)
                    )
            print(
                "MOE_PACK_DISPATCH",
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
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "moe_pack_dispatch.json"
        ).write_text(
            json.dumps(
                dict(
                    timing="complete wrapper CPU wall time and CUDA-event stream latency, no profiler; allocations outside timing; same kernel arithmetic",
                    warmup=30,
                    repeats=60,
                    rows=rows,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
