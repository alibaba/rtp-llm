"""
COPIED FROM DeepEP
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench(
    fn: Callable[[], None],
    num_warmups: int = 50,
    num_tests: int = 50,
    post_fn: Optional[Callable[[], None]] = None,
    trace_path: Optional[str] = None,
    suppress_kineto_output: bool = False,
    barrier_comm_profiling: bool = False,
) -> Tuple[float, float, float]:
    # 用256MB数据刷新L2缓存
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()
    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record(stream=torch.cuda.current_stream())
        fn()
        end_events[i].record(stream=torch.cuda.current_stream())
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()
    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]

    if trace_path is not None:
        # Flush L2
        cache.zero_()
        # Profile
        suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
        with suppress():
            schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
            ) as prof:
                for i in range(2):
                    # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                    if barrier_comm_profiling:
                        lhs = torch.randn(
                            (8192, 8192), dtype=torch.float, device="cuda"
                        )
                        rhs = torch.randn(
                            (8192, 8192), dtype=torch.float, device="cuda"
                        )
                        _ = lhs @ rhs
                        torch.distributed.all_reduce(
                            torch.ones(1, dtype=torch.float, device="cuda")
                        )
                    for _ in range(num_tests):
                        fn()
                    prof.step()
        prof.export_chrome_trace(trace_path)

    return float(np.average(times)), float(np.min(times)), float(np.max(times))


def bench_kineto(
    fn: Callable[[], None],
    kernel_names: Union[str, Tuple[str, ...]],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
) -> List[float]:
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        ) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    _ = lhs @ rhs
                    torch.distributed.all_reduce(
                        torch.ones(1, dtype=torch.float, device="cuda")
                    )
                for _ in range(num_tests):
                    fn()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        prof.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert (
            sum([name in line for line in prof_lines]) == 1
        ), f"Errors of the kernel {name} in the profiling table"

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(
                            float(time_str.replace(unit, "")) / scale
                        )
                        break
                break

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [
                event
                for event in profile_data["traceEvents"]
                if f"::{kernel_name}" in event["name"]
            ]
            events = sorted(events, key=lambda event: event["ts"])
            durations = [event["dur"] / 1e6 for event in events]
            # this assert may cause hang
            # assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [
                sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                for j in range(num_kernels_per_period)
            ]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def bench_compute_op(
    fn,
    num_warmups: int = 50,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    position_shift: Tuple[int, int] = (1, 1),
):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    # Warmup
    for _ in range(num_warmups):
        fn()
    # Add a large kernel to eliminate the CPU launch overhead
    lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
    rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        ) as prof:
            lhs @ rhs
            for _ in range(2):
                for _ in range(num_tests):
                    # Record
                    cache.zero_()
                    fn()
                prof.step()
            torch.cuda.synchronize()
    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Create a temporary trace file
    trace_path_to_use = trace_path
    if trace_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        trace_path_to_use = temp_file.name
        temp_file.close()
        prof.export_chrome_trace(trace_path_to_use)
    # Parse the trace events
    kernel_range_time = parse_trace_events(
        trace_path=trace_path_to_use, position_shift=position_shift
    )
    # Clean up the temporary trace file
    if trace_path is None:
        os.remove(trace_path_to_use)

    # Return execution durations
    return kernel_range_time


def parse_trace_events(
    trace_path: str, position_shift: Tuple[int, int] = (1, 1)
) -> float:
    seperated_kernel_name = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)"
    # Load the trace events
    profile_data = json.loads(Path(trace_path).read_text())
    trace_events = profile_data["traceEvents"]
    # Filter the trace events to only include the kernel events
    trace_events = [
        event
        for event in trace_events
        if (event["ph"] == "X" and event["cat"] == "kernel")
    ]
    # Sort the trace events by timestamp
    trace_events = sorted(trace_events, key=lambda event: event["ts"])
    # Strip the trace events to only include the full kernel events
    seperated_kernel_indices = [
        i
        for i, event in enumerate(trace_events)
        if (seperated_kernel_name in event["name"])
    ]
    assert (
        len(seperated_kernel_indices) > 1
    ), "There should be at least two matching indices"
    del trace_events[seperated_kernel_indices[-1] + 1 :]
    del trace_events[: seperated_kernel_indices[0]]
    # Find seperated kernel indices
    seperated_kernel_indices = [
        i
        for i, event in enumerate(trace_events)
        if (seperated_kernel_name in event["name"])
    ]
    # Calculate the duration of the kernel range
    start_kernel_event_indices = [
        seperated_kernel_indices[i] + position_shift[0]
        for i in range(0, len(seperated_kernel_indices) - 1)
    ]
    end_kernel_event_indices = [
        seperated_kernel_indices[i] - position_shift[1]
        for i in range(1, len(seperated_kernel_indices))
    ]
    kernel_range_durations = [
        trace_events[end_kernel_event_indices[i]]["ts"]
        + trace_events[end_kernel_event_indices[i]]["dur"]
        - trace_events[start_kernel_event_indices[i]]["ts"]
        for i in range(len(start_kernel_event_indices))
    ]
    return sum(kernel_range_durations) / len(kernel_range_durations)
