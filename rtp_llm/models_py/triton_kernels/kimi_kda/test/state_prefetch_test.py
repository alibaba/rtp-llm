"""Actual product state prefetch: byte parity, fallback, full graph and regression suite."""

import json
import os
import statistics
import time
import unittest
from collections import Counter, OrderedDict
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import rtp_llm.models_py.triton_kernels.kimi_kda.chunk_copies as copies
import rtp_llm.models_py.triton_kernels.kimi_kda.chunk_delta_h as recurrence
import rtp_llm.models_py.triton_kernels.kimi_kda.state_update as updates
import torch
from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA
from rtp_llm.models_py.triton_kernels.kimi_kda.test import (
    full_reuse_graph_test as fixture,
)


def update_prepare(state, delta, gate, u, k, output=None):
    if output is None:
        output = torch.empty_like(state)
    result = copies.update_prepare_chunk(state, delta, gate, u, k, output)
    assert result is not None, "Eligible product path must not fall back"
    return result


@contextmanager
def variant(enabled):
    with mock.patch.object(copies, "_PREFETCH_STATE", enabled):
        yield


class StatePrefetchTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.FullReuseGraphTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        for owner, attr in (
            (copies, "_ENABLED"),
            (copies, "_PREFETCH_STATE"),
            (recurrence, "_REUSE_BIAS_OUTPUT"),
        ):
            patch = mock.patch.object(owner, attr, True)
            patch.start()
            self.addCleanup(patch.stop)
        self.directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])

    def assert_bits(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.stride(), b.stride())
        dtype = torch.int16 if a.element_size() == 2 else torch.int32
        self.assertTrue(
            torch.equal(a.contiguous().view(dtype), b.contiguous().view(dtype))
        )

    def test_0_state_and_prepared_bits(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        for device in (0, 1):
            with torch.cuda.device(device):
                for heads, length, offset in (
                    (2, 1, 1),
                    (16, 13, 1),
                    (16, 64, 0),
                    (64, 65, 0),
                ):
                    shape = (heads, 128, 128)

                    def fp32_bits():
                        return (
                            torch.randint(
                                -(2**31),
                                2**31,
                                (heads * 16384 + offset,),
                                device="cuda",
                                dtype=torch.int32,
                            )[offset:]
                            .view(shape)
                            .view(torch.float32)
                        )

                    state, delta = fp32_bits(), fp32_bits()
                    gate = (
                        torch.randn((heads, 128), device="cuda", dtype=torch.bfloat16)
                        * 10
                    )

                    def operand():
                        x = torch.arange(
                            (length + 2) * (heads + 1) * 128 + offset,
                            device="cuda",
                            dtype=torch.int32,
                        )
                        return (
                            x.to(torch.int16)
                            .view(torch.bfloat16)[offset:]
                            .view(length + 2, heads + 1, 128)[1 : 1 + length, :heads]
                        )

                    u, k = operand(), operand()
                    expected = updates.kda_state_update(state, delta, gate)
                    expected_copies = copies.prepare_chunk_copies(expected, u, k)
                    for use_output in (False, True):
                        target = torch.full(
                            (heads * 16384 + 2,), 9, device="cuda", dtype=torch.float32
                        )
                        output = target[1:-1].view(shape) if use_output else None
                        actual, prepared = update_prepare(
                            state, delta, gate, u, k, output
                        )
                        self.assert_bits(actual, expected)
                        for a, b in zip(prepared, expected_copies):
                            self.assert_bits(a, b)
                        if use_output:
                            self.assertEqual(actual.data_ptr(), output.data_ptr())
                            self.assertEqual(target[0].item(), 9)
                            self.assertEqual(target[-1].item(), 9)

    def compare_graphs(self, profile_only=False):
        rows = []
        for lengths in ((81,), (81,) * 63, (419,) * 3, (9889,)):
            values = self.fixture.inputs(lengths, heads=16)
            token = _FORWARD_METADATA.set({})
            caches, outputs, kernels = {}, {}, {}
            samples = {False: [], True: []}
            try:
                for enabled in (False, True):
                    torch.cuda.synchronize()
                    fixture._GRAPHS.clear()
                    fixture._BASE_LENGTHS.clear()
                    fixture._PREPARED_SMALL_PAIRS.clear()
                    with variant(enabled), torch.enable_grad():
                        outputs[enabled] = self.fixture.compare(values)
                        caches[enabled] = OrderedDict(fixture._GRAPHS)
                        for _ in range(3 if profile_only else 30):
                            fixture.chunk_kda_full_reuse_graph(**values)
                for a, b in zip(outputs[False], outputs[True]):
                    self.assert_bits(a, b)
                captures = fixture.graph_stats().get("captures", 0)
                if not profile_only:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    for repeat in range(60):
                        for enabled in ((False, True) if repeat % 2 else (True, False)):
                            torch.cuda.synchronize()
                            fixture._GRAPHS.clear()
                            fixture._GRAPHS.update(caches[enabled])
                            with variant(enabled), torch.enable_grad():
                                start.record()
                                begin = time.perf_counter_ns()
                                fixture.chunk_kda_full_reuse_graph(**values)
                                cpu = (time.perf_counter_ns() - begin) / 1000
                                end.record()
                                end.synchronize()
                            samples[enabled].append(
                                dict(
                                    cpu_us=cpu, event_us=start.elapsed_time(end) * 1000
                                )
                            )
                    self.assertEqual(fixture.graph_stats().get("captures", 0), captures)
                else:
                    for enabled in (False, True):
                        fixture._GRAPHS.clear()
                        fixture._GRAPHS.update(caches[enabled])
                        torch.cuda.synchronize()
                        with variant(
                            enabled
                        ), torch.enable_grad(), torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profile:
                            fixture.chunk_kda_full_reuse_graph(**values)
                            torch.cuda.synchronize()
                        path = (
                            self.directory
                            / f"prefetch_{len(lengths)}_{sum(lengths)}_{enabled}.json"
                        )
                        profile.export_chrome_trace(str(path))
                        events = json.loads(path.read_text())["traceEvents"]
                        gpu = [e for e in events if e.get("cat") == "kernel"]
                        kernels[enabled] = dict(
                            names_counts=dict(Counter(e["name"] for e in gpu)),
                            kernel_sum_us=sum(e["dur"] for e in gpu),
                            launch_count=len(gpu),
                            copies=dict(
                                Counter(
                                    e["args"]["bytes"]
                                    for e in events
                                    if e.get("cat") == "gpu_memcpy"
                                )
                            ),
                        )
                    old = Counter(kernels[False]["names_counts"])
                    new = Counter(kernels[True]["names_counts"])
                    fused = new.pop("_update_prepare", 0)
                    self.assertGreater(fused, 0)
                    new["_state_update"] += fused
                    new["_prepare"] += fused
                    self.assertEqual(old, new)
                    self.assertEqual(
                        kernels[False]["launch_count"] - kernels[True]["launch_count"],
                        fused,
                    )
                    self.assertEqual(kernels[False]["copies"], kernels[True]["copies"])
                rows.append(dict(lengths=lengths, samples=samples, kernels=kernels))
                print(
                    "KDA_STATE_PREFETCH",
                    len(lengths),
                    sum(lengths),
                    {
                        str(e): {
                            k: statistics.median(s[k] for s in ss)
                            for k in ("cpu_us", "event_us")
                        }
                        for e, ss in samples.items()
                        if ss
                    },
                    flush=True,
                )
            finally:
                torch.cuda.synchronize()
                fixture._GRAPHS.clear()
                fixture._BASE_LENGTHS.clear()
                fixture._PREPARED_SMALL_PAIRS.clear()
                caches.clear()
                outputs.clear()
                _FORWARD_METADATA.reset(token)
        (
            self.directory
            / ("prefetch_kernels.json" if profile_only else "prefetch_benchmark.json")
        ).write_text(json.dumps(rows, indent=2))

    def test_full_graph_timing(self):
        self.compare_graphs()

    def test_z_full_graph_kernels(self):
        self.compare_graphs(profile_only=True)

    def test_fallbacks_and_overlapping_outputs(self):
        state = torch.randn((16, 128, 128), device="cuda")
        delta = torch.randn_like(state)
        gate = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
        u = torch.randn((13, 16, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(u)
        output = torch.empty_like(state)
        args = [state, delta, gate, u, k, output]
        self.assertIsNotNone(copies.update_prepare_chunk(*args))
        for index in range(len(args)):
            bad = list(args)
            bad[index] = args[index].detach().requires_grad_(True)
            self.assertIsNone(copies.update_prepare_chunk(*bad))
        for bad_output in (
            None,
            state,
            delta,
            output.transpose(1, 2),
            output.to(torch.bfloat16),
        ):
            self.assertIsNone(copies.update_prepare_chunk(*args[:-1], bad_output))
        for index in (3, 4):
            bad = list(args)
            bad[index] = output.view(torch.bfloat16).flatten()[: u.numel()].view_as(u)
            self.assertIsNone(copies.update_prepare_chunk(*bad))
        bad = list(args)
        bad[2] = torch.empty((16, 256), device="cuda", dtype=torch.bfloat16)[:, ::2]
        self.assertIsNone(copies.update_prepare_chunk(*bad))
        with mock.patch.object(copies, "_PREFETCH_STATE", False):
            self.assertIsNone(copies.update_prepare_chunk(*args))
        # Adjacent views into the same checkpoint storage are safe.
        pair = torch.randn((2, 16, 128, 128), device="cuda")
        adjacent = [pair[0], delta, gate, u, k, pair[1]]
        expected = updates.kda_state_update(pair[0], delta, gate)
        actual = copies.update_prepare_chunk(*adjacent)
        self.assertIsNotNone(actual)
        self.assert_bits(actual[0], expected)


def load_tests(loader, tests, pattern):
    # Run all existing exact-layout/ownership/grad/two-device/stream/capture tests
    # with the new product flag enabled. Perf subcomparisons isolate older flags.
    combined = list(loader.loadTestsFromTestCase(StatePrefetchTest)) + list(
        loader.loadTestsFromTestCase(fixture.FullReuseGraphTest)
    )
    # Initialize CUPTI only after all clean timing methods across both classes.
    return unittest.TestSuite(
        sorted(combined, key=lambda test: test._testMethodName.startswith("test_z"))
    )


if __name__ == "__main__":
    unittest.main()
