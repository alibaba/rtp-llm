"""Isolate low-rank unpack kernel, allocation and Python dispatch costs."""

import cProfile
import json
import os
import statistics
import time
import unittest
from collections import Counter
from pathlib import Path

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.triton_kernels.kimi_kda.lowrank_copy import (
    _copy_pair,
    copy_lowrank_pair,
)


@triton.jit
def _unpack_flat(
    x,
    f,
    g,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row, col = i // WIDTH, i % WIDTH
    source = row * STRIDE + OFFSET + col
    mask = row < ROWS
    tl.store(f + i, tl.load(x + source, mask, 0), mask)
    tl.store(g + i, tl.load(x + source + WIDTH, mask, 0), mask)


@triton.jit
def _unpack_tiled(
    x,
    f,
    g,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE).to(tl.int64)
    col = tl.arange(0, 2 * WIDTH).to(tl.int64)
    value = tl.load(
        x + row[:, None] * STRIDE + OFFSET + col[None, :], row[:, None] < ROWS, 0
    )
    tl.store(
        f + row[:, None] * WIDTH + col[None, :],
        value,
        (row[:, None] < ROWS) & (col[None, :] < WIDTH),
    )
    tl.store(
        g + row[:, None] * WIDTH + col[None, :] - WIDTH,
        value,
        (row[:, None] < ROWS) & (col[None, :] >= WIDTH),
    )


def launch(source, outputs, tile=0):
    if tile:
        _unpack_tiled[((source.shape[0] + tile - 1) // tile,)](
            source,
            *outputs,
            source.shape[0],
            source.stride(0),
            6160,
            128,
            tile,
            num_warps=4,
        )
    else:
        _unpack_flat[((source.shape[0] * 128 + 511) // 512,)](
            source, *outputs, source.shape[0], source.stride(0), 6160, 128, 512
        )
    return outputs


def direct_tiled(source):
    rows = source.shape[0]
    forget = source.new_empty((rows, 128))
    gate = source.new_empty((rows, 128))
    tile = 16 if rows >= 4096 else 4
    _unpack_tiled[((rows + tile - 1) // tile,)](
        source,
        forget,
        gate,
        rows,
        source.stride(0),
        6160,
        128,
        tile,
        num_warps=4,
    )
    return forget, gate


def direct_flat(source):
    rows = source.shape[0]
    forget = source.new_empty((rows, 128))
    gate = source.new_empty((rows, 128))
    _unpack_flat[((rows * 128 + 1023) // 1024,)](
        source,
        forget,
        gate,
        rows,
        source.stride(0),
        6160,
        128,
        1024,
    )
    return forget, gate


def measure(function):
    for _ in range(30):
        function()
    cpu, stream = [], []
    for _ in range(5):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        before = time.perf_counter()
        for _ in range(100):
            function()
        cpu.append((time.perf_counter() - before) * 1e4)
        end.record()
        end.synchronize()
        stream.append(start.elapsed_time(end) * 10)
    return {"cpu_us": statistics.median(cpu), "stream_us": statistics.median(stream)}


def gpu_measure(function):
    # Profile actual kernels. CUDA events around replay are a separate stream metric.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            function()
    graph.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        graph.replay()
        torch.cuda.synchronize()
    kernels = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    counts = Counter(event.name for event in kernels)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(50):
        graph.replay()
    end.record()
    end.synchronize()
    return {
        "kernel_sum_us_per_iter": sum(event.device_time_total for event in kernels)
        / 20,
        "launches_per_iter": len(kernels) / 20,
        "kernel_names": dict(counts),
        "graph_stream_us_per_iter": start.elapsed_time(end) * 1000 / 1000,
    }


class PackedLowrankTest(unittest.TestCase):
    def test_batch_size_does_not_compile_new_code(self):
        kernels = [
            _copy_pair.warmup(
                *([torch.bfloat16] * 4),
                rows,
                6528,
                6528,
                grid=((rows + 7) // 8, 1, 1),
                num_warps=4,
            )
            for rows in (13, 81, 419, 9889)
        ]
        for kernel in kernels[1:]:
            self.assertIs(kernel, kernels[0])

    def test_product_wrapper(self):
        records = []
        for tokens in (13, 81, 419, 1021, 4093, 9889):
            source = torch.arange(tokens * 6528, device="cuda", dtype=torch.int32)
            source = source.to(torch.int16).view(torch.bfloat16).reshape(tokens, 6528)
            forget, gate = source[:, 6160:6288], source[:, 6288:6416]
            expected = forget.contiguous(), gate.contiguous()

            def candidate():
                return copy_lowrank_pair(forget, gate)

            actual = candidate()
            self.assertIsNotNone(actual)
            for old, new in zip(expected, actual):
                self.assertTrue(new.is_contiguous())
                self.assertTrue(
                    torch.equal(old.view(torch.int16), new.view(torch.int16))
                )
            row = {
                "tokens": tokens,
                "eager": measure(lambda: (forget.contiguous(), gate.contiguous())),
                "wrapper": measure(candidate),
            }
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                for _ in range(20):
                    candidate()
                torch.cuda.synchronize()
            kernels = [
                e
                for e in profile.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            self.assertEqual(len(kernels), 20)
            row["wrapper"]["kernel_sum_us_per_iter"] = (
                sum(e.device_time_total for e in kernels) / 20
            )
            records.append(row)
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "lowrank_product_wrapper.json"
        ).write_text(json.dumps(records, indent=2))
        # Independent inputs, unequal row pitches, and more than one CUDA device.
        for device in range(min(2, torch.cuda.device_count())):
            with torch.cuda.device(device):
                left = torch.randn(81, 137, device="cuda", dtype=torch.bfloat16)[
                    :, :128
                ]
                right = torch.randn(81, 144, device="cuda", dtype=torch.bfloat16)[
                    :, :128
                ]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                # Compile before checking asynchronous execution.
                self.assertIsNotNone(copy_lowrank_pair(left, right))
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(300_000_000)
                    pending = torch.cuda.Event()
                    pending.record()
                    actual = copy_lowrank_pair(left, right)
                    self.assertFalse(pending.query())
                stream.synchronize()
                for old, new in zip((left, right), actual):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.assertIsNone(copy_lowrank_pair(left, right))
        self.assertIsNone(copy_lowrank_pair(forget[:1], gate[:1]))
        self.assertIsNone(copy_lowrank_pair(forget.float(), gate.float()))
        self.assertIsNone(copy_lowrank_pair(forget.cpu(), gate.cpu()))
        self.assertIsNone(copy_lowrank_pair(forget[:, :127], gate[:, :127]))
        unaligned = torch.empty(2 * 129 + 1, device="cuda", dtype=torch.bfloat16)[
            1:
        ].view(2, 129)[:, :128]
        self.assertIsNone(copy_lowrank_pair(unaligned, unaligned))

    def test_z_global_offsets_exceed_int32(self):
        # Only touch three short rows, but place their source addresses > 2^31
        # elements apart. This also checks dtype-only warmup's alignment contract.
        strides = ((1 << 31) + 128, (1 << 31) + 256)
        inputs = []
        expected = []
        for stride in strides:
            storage = torch.empty(2 * stride + 128, device="cuda", dtype=torch.bfloat16)
            value = storage.as_strided((3, 128), (stride, 1))
            data = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
            value.copy_(data)
            inputs.append(value)
            expected.append(data)
        actual = copy_lowrank_pair(*inputs)
        self.assertIsNotNone(actual)
        for old, new in zip(expected, actual):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_exact_async_and_decomposed_perf(self):
        records = []
        outdir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        for tokens in (1, 13, 81, 419, 1021, 4093, 9889):
            source = torch.arange(tokens * 6528, device="cuda", dtype=torch.int32)
            source = source.to(torch.int16).view(torch.bfloat16).reshape(tokens, 6528)
            views = source[:, 6160:6288], source[:, 6288:6416]
            outputs = tuple(
                torch.empty_like(v, memory_format=torch.contiguous_format)
                for v in views
            )
            row = {"tokens": tokens}

            def eager():
                return views[0].contiguous(), views[1].contiguous()

            def two_outputs():
                return tuple(
                    torch.empty_like(v, memory_format=torch.contiguous_format)
                    for v in views
                )

            def shared_output():
                return torch.empty(
                    (2, tokens, 128), device=source.device, dtype=source.dtype
                ).unbind(0)

            tile = 16 if tokens >= 4096 else 4
            grid = ((tokens + tile - 1) // tile, 1, 1)
            compiled = _unpack_tiled.warmup(
                source,
                *outputs,
                tokens,
                source.stride(0),
                6160,
                128,
                tile,
                grid=grid,
                num_warps=4,
            )
            runner = compiled[grid]

            def compiled_tiled():
                forget = source.new_empty((tokens, 128))
                gate = source.new_empty((tokens, 128))
                # Use CompiledKernel's public launch entry; it reads the current
                # stream on every call. Only code/grid are retained, no tensors.
                runner(source, forget, gate, tokens, source.stride(0), 6160, 128, tile)
                return forget, gate

            def compiled_empty_like():
                forget = torch.empty_like(
                    views[0], memory_format=torch.contiguous_format
                )
                gate = torch.empty_like(views[1], memory_format=torch.contiguous_format)
                runner(source, forget, gate, tokens, source.stride(0), 6160, 128, tile)
                return forget, gate

            functions = {
                "compiled_empty_like": compiled_empty_like,
                "compiled_tiled_alloc": compiled_tiled,
                "eager": eager,
                "direct_tiled_alloc": lambda: direct_tiled(source),
                "direct_flat1024_alloc": lambda: direct_flat(source),
                "flat_shared_alloc": lambda: launch(source, shared_output()),
                "flat_two_alloc": lambda: launch(source, two_outputs()),
                "flat_preallocated": lambda: launch(source, outputs),
                "tiled4_preallocated": lambda: launch(source, outputs, 4),
                "tiled16_preallocated": lambda: launch(source, outputs, 16),
                "stack": lambda: torch.stack(views).unbind(0),
                "foreach_preallocated": lambda: (
                    torch._foreach_copy_(outputs, views),
                    outputs,
                )[1],
            }
            expected = tuple(v.contiguous().view(torch.int16) for v in views)
            for name, function in functions.items():
                actual = function()
                for old, new in zip(expected, actual):
                    self.assertTrue(new.is_contiguous())
                    self.assertTrue(
                        torch.equal(old, new.view(torch.int16)), (tokens, name)
                    )
                row[name] = measure(function)
                row[name].update(gpu_measure(function))
                if name.startswith(("flat_", "tiled", "direct_", "compiled_")):
                    self.assertEqual(row[name]["launches_per_iter"], 1, row[name])
                if tokens == 419:
                    profile = cProfile.Profile()
                    profile.enable()
                    for _ in range(100):
                        function()
                    profile.disable()
                    profile.dump_stats(str(outdir / (name + ".pstats")))
            row["shared_allocation_only"] = measure(shared_output)
            row["two_allocation_only"] = measure(two_outputs)
            records.append(row)
            (outdir / "packed_lowrank_decomposed.json").write_text(
                json.dumps(records, indent=2)
            )
        # A cached public runner must accept fresh pointers and honor a new
        # current stream; no data pointer or stream may be captured by the cache.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fresh_source = torch.empty_like(source)
            fresh_outputs = tuple(torch.empty_like(v) for v in outputs)
            torch.cuda._sleep(300_000_000)
            pending = torch.cuda.Event()
            pending.record()
            fresh_source.copy_(source)
            runner(
                fresh_source,
                *fresh_outputs,
                tokens,
                fresh_source.stride(0),
                6160,
                128,
                tile,
            )
            self.assertFalse(pending.query())
        stream.synchronize()
        for old, new in zip(expected, fresh_outputs):
            self.assertTrue(torch.equal(old, new.view(torch.int16)))
        torch.cuda.synchronize()
        torch.cuda._sleep(300_000_000)
        pending = torch.cuda.Event()
        pending.record()
        launch(source, outputs, 4)
        self.assertFalse(pending.query())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
