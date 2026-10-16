"""Isolate host CPU spent waiting for identical GPU work; no serving changes."""

import ctypes
import json
import os
import statistics
import time
import unittest
from pathlib import Path

import torch


def current_context_api():
    # PyTorch/JIT dependencies can load multiple libcudart copies. The driver
    # API addresses the current CUDA context shared by those runtime copies.
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    driver.cuCtxGetFlags.restype = ctypes.c_int
    driver.cuCtxSetFlags.argtypes = [ctypes.c_uint]
    driver.cuCtxSetFlags.restype = ctypes.c_int
    return driver


class TestCudaWaitPolicy(unittest.TestCase):
    @torch.inference_mode()
    def test_wait_cpu_and_ordering(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        torch.cuda.init()
        runtime = current_context_api()

        def get_flags():
            flags = ctypes.c_uint()
            self.assertEqual(runtime.cuCtxGetFlags(ctypes.byref(flags)), 0)
            return flags.value

        def set_flags(flags):
            self.assertEqual(runtime.cuCtxSetFlags(flags), 0)
            self.assertEqual(get_flags(), flags)

        original_device = torch.cuda.current_device()
        original_flags = {}
        for device in (0, 1):
            with torch.cuda.device(device):
                # PyTorch can defer primary-context creation until allocation.
                # Driver context queries must follow real device initialization.
                torch.empty(1, device=device)
                torch.cuda.synchronize(device)
                original_flags[device] = get_flags()
        rows = []
        try:
            with torch.cuda.device(0):
                stream = torch.cuda.Stream()
                value = torch.zeros(256, dtype=torch.int32, device="cuda")
                host = torch.empty_like(value, device="cpu", pin_memory=True)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                blocking = torch.cuda.Event(blocking=True)
                torch.cuda.synchronize()
                # Calibrate GPU sleep once. Events here include stream launch
                # gaps; no result is represented as pure kernel execution time.
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        torch.cuda._sleep(1_000_000)
                    begin.record()
                    torch.cuda._sleep(10_000_000)
                    end.record()
                end.synchronize()
                cycles_per_ms = 10_000_000 / begin.elapsed_time(end)
                waiters = {
                    "device": torch.cuda.synchronize,
                    "stream": stream.synchronize,
                    "event_default": end.synchronize,
                    "event_blocking": blocking.synchronize,
                }
                update_count = 0

                def trial(target_ms, waiter):
                    nonlocal update_count
                    with torch.cuda.stream(stream):
                        begin.record()
                        torch.cuda._sleep(max(1, int(cycles_per_ms * target_ms)))
                        value.add_(1)
                        host.copy_(value, non_blocking=True)
                        end.record()
                        blocking.record()
                    cpu_start = time.thread_time_ns()
                    wall_start = time.perf_counter_ns()
                    waiter()
                    wall_us = (time.perf_counter_ns() - wall_start) / 1000
                    cpu_us = (time.thread_time_ns() - cpu_start) / 1000
                    update_count += 1
                    self.assertTrue(end.query())
                    # Reading pinned CPU memory adds no implicit CUDA wait.
                    self.assertEqual(host.tolist(), [update_count] * host.numel())
                    result = dict(
                        wait_wall_us=wall_us,
                        wait_thread_cpu_us=cpu_us,
                        stream_event_us=begin.elapsed_time(end) * 1000,
                    )
                    # Drain the later event outside timing so trials never
                    # overlap, even when waiting on the earlier default event.
                    stream.synchronize()
                    return result

                for target_ms in (0.1, 1.0, 10.0):
                    for kind, waiter in waiters.items():
                        samples = {"auto": [], "blocking": []}
                        for policy, scheduling_bits in (("auto", 0), ("blocking", 4)):
                            set_flags((original_flags[0] & ~7) | scheduling_bits)
                            for _ in range(30):
                                trial(target_ms, waiter)
                        for repeat in range(40):
                            order = (("auto", 0), ("blocking", 4))
                            if repeat % 2:
                                order = tuple(reversed(order))
                            for policy, scheduling_bits in order:
                                set_flags((original_flags[0] & ~7) | scheduling_bits)
                                samples[policy].append(trial(target_ms, waiter))
                        medians = {
                            policy: {
                                metric: statistics.median(r[metric] for r in values)
                                for metric in values[0]
                            }
                            for policy, values in samples.items()
                        }
                        rows.append(
                            dict(target_ms=target_ms, wait=kind, samples=samples)
                        )
                        print("CUDA_WAIT_POLICY", target_ms, kind, medians, flush=True)

                # A flag change belongs to this process/device; the other GPU
                # retains its original flags. Device-wide synchronization must
                # still cover both non-default streams and retain copy ordering.
                set_flags((original_flags[0] & ~7) | 4)
                with torch.cuda.device(1):
                    self.assertEqual(get_flags(), original_flags[1])
                other = torch.cuda.Stream()
                copy = torch.zeros(256, dtype=torch.int32, device="cuda")
                copy_cpu = torch.empty_like(copy, device="cpu", pin_memory=True)
                torch.cuda.synchronize()
                with torch.cuda.stream(other):
                    torch.cuda._sleep(int(cycles_per_ms * 20))
                    copy.add_(7)
                    copy_cpu.copy_(copy, non_blocking=True)
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(int(cycles_per_ms * 5))
                    value.add_(1)
                    host.copy_(value, non_blocking=True)
                torch.cuda.synchronize()
                self.assertEqual(copy_cpu.tolist(), [7] * copy_cpu.numel())
                self.assertEqual(host.tolist(), [update_count + 1] * host.numel())
                directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
                (directory / "cuda_wait_policy.json").write_text(
                    json.dumps(
                        dict(
                            torch=torch.__version__,
                            cuda=torch.version.cuda,
                            gpu=torch.cuda.get_device_name(),
                            cpu_affinity_count=len(os.sched_getaffinity(0)),
                            original_flags=original_flags,
                            cycles_per_ms=cycles_per_ms,
                            timing="wait-only wall/thread CPU; GPU events include submission gaps; no profiler",
                            rows=rows,
                        ),
                        indent=2,
                    )
                )
        finally:
            for device, flags in original_flags.items():
                with torch.cuda.device(device):
                    torch.cuda.synchronize()
                    set_flags(flags)
            torch.cuda.set_device(original_device)


if __name__ == "__main__":
    unittest.main()
