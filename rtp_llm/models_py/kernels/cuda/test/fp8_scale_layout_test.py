"""Aligned FP8 scale views: full backing storage, exact writes and dispatch."""

import importlib
import json
import os
import statistics
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

quant = importlib.import_module("rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel")
allocate = quant.create_per_token_group_quant_fp8_output_scale


def original(
    x_shape, device, group_size, column_major_scales, scale_tma_aligned, scale_ue8m0
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        *batch, m, k = x_shape
        am, ak = quant.ceil_align(m, 4), quant.ceil_align(k // 128, 4)
        return torch.empty(
            (*batch, ak // 4, am), device=device, dtype=torch.int
        ).transpose(-1, -2)[..., :m, :]
    if column_major_scales:
        if scale_tma_aligned:
            am = quant.ceil_align(x_shape[-2], 4)
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, am),
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)[: x_shape[-2], :]
        return torch.empty(
            (x_shape[-1] // group_size,) + x_shape[:-1],
            device=device,
            dtype=torch.float32,
        ).permute(-1, -2)
    return torch.empty(
        x_shape[:-1] + (x_shape[-1] // group_size,), device=device, dtype=torch.float32
    )


def physical(t):
    return t.as_strided((t.untyped_storage().nbytes() // t.element_size(),), (1,), 0)


class ScaleLayoutTest(unittest.TestCase):
    def setUp(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        self.mode = mock.patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"})
        self.mode.start()
        self.addCleanup(self.mode.stop)
        torch.manual_seed(531018)

    def call(self, x, allocator, col=True, tma=True, ue8=True):
        with mock.patch.object(
            quant, "create_per_token_group_quant_fp8_output_scale", allocator
        ):
            return quant.sgl_per_token_group_quant_fp8(
                x, 128, column_major_scales=col, scale_tma_aligned=tma, scale_ue8m0=ue8
            )

    def bits(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        # Singleton dimensions can be contiguous with a non-unit final stride.
        # Same-width integer views preserve every bit without a byte-view stride
        # restriction or a floating-point equality comparison (NaN/signed zero).
        integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            a.element_size()
        ]
        self.assertTrue(torch.equal(a.view(integer), b.view(integer)))

    def test_original_unaligned_input_probe(self):
        # A CUDA fault poisons its context. Isolate the original native path
        # in a child process before attributing the earlier fixture failure.
        result = subprocess.run(
            [sys.executable, __file__, "--native-unaligned-probe"],
            env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
            capture_output=True,
            text=True,
            timeout=90,
        )
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "native_unaligned_probe.log"
        ).write_text(result.stdout + result.stderr)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("misaligned address", result.stderr)

    def test_layouts_storage_and_real_quant_writes(self):
        layouts = (
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        )
        for device in ("cpu", "cuda:0", "cuda:1"):
            for m in (0, 1, 3, 4, 5, 81, 419):
                for k in (128, 384, 4096):
                    for col, tma, ue8 in layouts:
                        a = original((m, k), device, 128, col, tma, ue8)
                        b = allocate((m, k), device, 128, col, tma, ue8)
                        self.assertEqual(
                            (
                                a.shape,
                                a.stride(),
                                a.storage_offset(),
                                a.dtype,
                                a.untyped_storage().nbytes(),
                            ),
                            (
                                b.shape,
                                b.stride(),
                                b.storage_offset(),
                                b.dtype,
                                b.untyped_storage().nbytes(),
                            ),
                        )
            for shape in ((2, 3, 384), (2, 0, 128), (0, 3, 128)):
                a = original(shape, device, 128, True, True, True)
                b = allocate(shape, device, 128, True, True, True)
                self.assertEqual(
                    (a.shape, a.stride(), a.untyped_storage().nbytes()),
                    (b.shape, b.stride(), b.untyped_storage().nbytes()),
                )
        for device in (0, 1):
            with torch.cuda.device(device):
                for m, k in (
                    (1, 128),
                    (3, 384),
                    (4, 4096),
                    (81, 384),
                    (419, 4096),
                    (5103, 4096),
                ):
                    x = torch.randn(m * k + 8, device="cuda", dtype=torch.bfloat16)[
                        8:
                    ].view(m, k)
                    for col, tma, ue8 in layouts:
                        guards = []

                        def guarded(fn):
                            def make(*args, **kwargs):
                                view = fn(*args, **kwargs)
                                count = physical(view).numel()
                                backing = torch.full(
                                    (count + 32,),
                                    123,
                                    dtype=view.dtype,
                                    device=view.device,
                                )
                                result = backing.as_strided(
                                    view.shape, view.stride(), 16
                                )
                                guards.append((backing, count))
                                return result

                            return make

                        a = self.call(x, guarded(original), col, tma, ue8)
                        b = self.call(x, guarded(allocate), col, tma, ue8)
                        self.bits(a[0], b[0])
                        self.bits(a[1], b[1])
                        self.bits(guards[0][0], guards[1][0])
                        for backing, count in guards:
                            self.assertTrue(torch.all(backing[:16] == 123).item())
                            self.assertTrue(
                                torch.all(backing[16 + count :] == 123).item()
                            )

    def test_owned_outputs_async_and_graph(self):
        for device in (0, 1):
            with torch.cuda.device(device):
                x = torch.randn(81, 384, device="cuda", dtype=torch.bfloat16)
                expected = self.call(x, original)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(50_000_000)
                    pending = torch.cuda.Event()
                    pending.record()
                    actual = self.call(x, allocate)
                    self.assertFalse(pending.query())
                stream.synchronize()
                for a, b in zip(actual, expected):
                    self.bits(a, b)
                saved = tuple(t.clone() for t in actual)
                other = self.call(x, allocate)
                for a, b, c in zip(actual, saved, other):
                    self.assertNotEqual(a.data_ptr(), c.data_ptr())
                    self.bits(a, b)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = self.call(x, allocate)
                x.mul_(0.73)
                expected = self.call(x, original)
                graph.replay()
                torch.cuda.synchronize()
                for a, b in zip(captured, expected):
                    self.bits(a, b)

    def test_complete_wrapper_timing(self):
        rows = []
        for m in (1, 81, 419, 5103, 9889):
            x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
            variants = (("original", original), ("one_view", allocate))
            for _, fn in variants:
                for _ in range(30):
                    self.call(x, fn)
            samples = {n: [] for n, _ in variants}
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for repeat in range(60):
                for name, fn in (variants if repeat % 2 else variants[::-1]):
                    with mock.patch.object(
                        quant, "create_per_token_group_quant_fp8_output_scale", fn
                    ):
                        torch.cuda.synchronize()
                        begin.record()
                        start = time.thread_time_ns()
                        result = quant.sgl_per_token_group_quant_fp8(
                            x,
                            128,
                            column_major_scales=True,
                            scale_tma_aligned=True,
                            scale_ue8m0=True,
                        )
                        cpu = (time.thread_time_ns() - start) / 1000
                        end.record()
                        end.synchronize()
                        samples[name].append(
                            dict(cpu_us=cpu, event_us=begin.elapsed_time(end) * 1000)
                        )
                        del result
            rows.append(
                dict(
                    tokens=m,
                    samples=samples,
                    medians={
                        n: {
                            k: statistics.median(v[k] for v in values)
                            for k in values[0]
                        }
                        for n, values in samples.items()
                    },
                )
            )
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "fp8_scale_layout_timing.json"
        ).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--native-unaligned-probe"]:
        os.environ["DSV4_FP8_QUANT_KERNEL"] = "v2"
        x = torch.randn(129, device="cuda", dtype=torch.bfloat16)[1:].view(1, 128)
        # Only the original allocator and native kernel are called here.
        with mock.patch.object(
            quant, "create_per_token_group_quant_fp8_output_scale", original
        ):
            quant.sgl_per_token_group_quant_fp8(x, 128)
        torch.cuda.synchronize()
    else:
        unittest.main()
