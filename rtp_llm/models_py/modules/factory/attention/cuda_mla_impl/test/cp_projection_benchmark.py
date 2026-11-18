import json
import statistics
import time

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import pynccl_cp
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import cp_bounded_gather


def benchmark_projection(rank: int, world: int) -> None:
    device = torch.device("cuda", rank)
    for rows in (8191, 131071):
        source = (
            torch.arange(rows * 576, device=device, dtype=torch.int32)
            .remainder(97)
            .to(torch.bfloat16)
            .view(rows, 576)
        )
        source.add_(rank)
        generator = torch.Generator().manual_seed(131)
        host_map = torch.randperm(rows * world, generator=generator)[
            : rows * world - 17
        ]
        mapping = host_map.to(device)
        inverse = cp_bounded_gather.prepare_restore_inverse(
            host_map, rows * world, device
        )
        assert inverse is not None
        calls = 0

        def gather(output: torch.Tensor, part: torch.Tensor, *, group=None) -> None:
            nonlocal calls
            calls += 1
            pynccl_cp.all_gather_into_tensor(output, part, group=dist.group.WORLD)

        def full(restored: bool) -> torch.Tensor:
            packed = torch.cat((source[:, :512], source[:, 512:]), dim=-1)
            output = torch.empty((rows * world, 576), device=device, dtype=source.dtype)
            gather(output, packed)
            return output.index_select(0, mapping) if restored else output

        def bounded(restored: bool, use_inverse: bool) -> torch.Tensor:
            return cp_bounded_gather.gather_projection_chunks(
                source[:, :512],
                source[:, 512:],
                world,
                mapping if restored else None,
                gather,
                inverse if use_inverse else None,
            )

        operations = (
            ("full_restore", lambda: full(True)),
            ("bounded_scan", lambda: bounded(True, False)),
            ("bounded_inverse", lambda: bounded(True, True)),
            ("full_rank_major", lambda: full(False)),
            ("bounded_rank_major", lambda: bounded(False, False)),
        )
        expected = full(True)
        for name, operation in operations:
            if "rank_major" not in name:
                assert torch.equal(operation(), expected), name
        del expected
        assert torch.equal(full(False), bounded(False, False))
        for _, operation in operations:
            for _ in range(5):
                operation()
        torch.cuda.synchronize()
        samples = {name: [] for name, _ in operations}
        for trial in range(10):
            ordered = operations if trial % 2 == 0 else tuple(reversed(operations))
            for name, operation in ordered:
                dist.barrier()
                torch.cuda.synchronize()
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                begin = time.perf_counter_ns()
                start.record()
                for _ in range(5):
                    operation()
                stop.record()
                stop.synchronize()
                samples[name].append(
                    (
                        start.elapsed_time(stop) * 200,
                        (time.perf_counter_ns() - begin) / 5000,
                    )
                )
        peaks = {}
        collective_counts = {}
        for name, operation in operations:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            calls = 0
            result = operation()
            torch.cuda.synchronize()
            peaks[name] = torch.cuda.max_memory_allocated() - before
            collective_counts[name] = calls
            del result
        plan_samples = []
        for _ in range(20):
            begin = time.perf_counter_ns()
            refreshed = cp_bounded_gather.prepare_restore_inverse(
                host_map, rows * world, device, current=inverse, forbid_realloc=True
            )
            torch.cuda.synchronize()
            assert refreshed.data_ptr() == inverse.data_ptr()
            plan_samples.append((time.perf_counter_ns() - begin) / 1000)
        print(
            "CP_PROJECTION_AB "
            + json.dumps(
                {
                    "rank": rank,
                    "world": world,
                    "local_rows": rows,
                    "stream_us": {
                        name: statistics.median(sample[0] for sample in values)
                        for name, values in samples.items()
                    },
                    "wall_us": {
                        name: statistics.median(sample[1] for sample in values)
                        for name, values in samples.items()
                    },
                    "samples": samples,
                    "incremental_peak_bytes": peaks,
                    "collectives": collective_counts,
                    "inverse_bytes": inverse.numel() * inverse.element_size(),
                    "inverse_refresh_wall_us": statistics.median(plan_samples),
                }
            ),
            flush=True,
        )
