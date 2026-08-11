"""CUDA-event benchmark for GLM5 prefill ``topk_v3_tie_break``.

The benchmark keeps allocations outside the timed region and compares against
both relevant baselines:

* ``topk_v3`` for uniform rows whose valid interval starts at zero.
* ``dsv4_top_k_per_row_prefill`` for ragged rows with non-zero request offsets.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    width: int
    valid_length: int
    k: int = 2048
    segments: int = 1
    prefix_ratio: float = 0.0
    compare_v3: bool = False
    pattern: str = "random"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _measure(
    fn: Callable[[], None], warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    values = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    return {
        "median_us": statistics.median(values),
        "p90_us": _percentile(values, 0.90),
        "min_us": min(values),
        "max_us": max(values),
    }


def _bounds(case: Case) -> tuple[torch.Tensor, torch.Tensor]:
    if case.segments == 1 and case.prefix_ratio == 0.0:
        starts = torch.zeros(case.rows, dtype=torch.int32, device="cuda")
        ends = torch.full(
            (case.rows,), case.valid_length, dtype=torch.int32, device="cuda"
        )
        return starts, ends

    assert case.width % case.segments == 0
    segment_width = case.width // case.segments
    assert case.valid_length <= segment_width
    row_ids = torch.arange(case.rows, dtype=torch.int64, device="cuda")
    segment_ids = row_ids % case.segments
    starts = (segment_ids * segment_width).to(torch.int32)
    prefix = min(
        case.valid_length,
        max(case.k, int(round(case.valid_length * case.prefix_ratio))),
    )
    growth = max(case.valid_length - prefix, 1)
    lengths = prefix + (row_ids // case.segments) % (growth + 1)
    lengths = lengths.clamp_max(case.valid_length).to(torch.int32)
    return starts.contiguous(), (starts + lengths).contiguous()


def _run_case(case: Case, warmup: int, iterations: int) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(
        2026081000 + case.rows + case.width
    )
    scores = torch.randn(
        case.rows,
        case.width,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    if case.pattern == "discrete":
        scores = torch.round(scores * 4.0) * 0.25
    elif case.pattern != "random":
        raise ValueError(f"unsupported score pattern: {case.pattern}")
    starts, ends = _bounds(case)
    output = torch.empty((case.rows, case.k), dtype=torch.int32, device="cuda")
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")

    def run_new() -> None:
        rtp_llm_ops.topk_v3_tie_break(
            scores,
            starts,
            ends,
            output,
            case.k,
            case.valid_length,
        )

    def run_per_row() -> None:
        rtp_llm_ops.dsv4_top_k_per_row_prefill(
            scores,
            starts,
            ends,
            output,
            case.rows,
            scores.stride(0),
            scores.stride(1),
            case.k,
            True,
        )

    measurements = {
        "topk_v3_tie_break": _measure(run_new, warmup, iterations),
        "dsv4_per_row": _measure(run_per_row, warmup, iterations),
    }
    if case.compare_v3:
        lengths = (ends - starts).contiguous()

        def run_v3() -> None:
            rtp_llm_ops.topk_v3(
                scores,
                lengths,
                output,
                workspace,
                case.k,
                case.valid_length,
            )

        measurements["topk_v3"] = _measure(run_v3, warmup, iterations)

    new_median = measurements["topk_v3_tie_break"]["median_us"]
    speedups = {
        f"vs_{name}": values["median_us"] / new_median
        for name, values in measurements.items()
        if name != "topk_v3_tie_break"
    }
    del scores, starts, ends, output, workspace
    torch.cuda.empty_cache()
    return {
        "case": asdict(case),
        "measurements": measurements,
        "speedups": speedups,
    }


def _cases(quick: bool) -> list[Case]:
    if quick:
        return [
            Case("uniform_register_4k", 104, 4096, 4096, compare_v3=True),
            Case("uniform_register_16k", 104, 16384, 16384, compare_v3=True),
            Case(
                "prefix80_ragged_16k",
                410,
                65536,
                16384,
                segments=4,
                prefix_ratio=0.8,
            ),
            Case(
                "prefix80_cp8_64k",
                1640,
                65536,
                65536,
                prefix_ratio=0.8,
            ),
        ]
    return [
        Case("uniform_register_4k", 104, 4096, 4096, compare_v3=True),
        Case("uniform_register_8k", 104, 8192, 8192, compare_v3=True),
        Case("uniform_register_16k", 104, 16384, 16384, compare_v3=True),
        Case("uniform_streaming_32k", 104, 32768, 32768, compare_v3=True),
        Case(
            "tie_dense_register_16k",
            104,
            16384,
            16384,
            compare_v3=True,
            pattern="discrete",
        ),
        Case(
            "tie_dense_streaming_64k",
            104,
            65536,
            65536,
            compare_v3=True,
            pattern="discrete",
        ),
        Case(
            "prefix80_ragged_4x16k",
            1640,
            65536,
            16384,
            segments=4,
            prefix_ratio=0.8,
        ),
        Case(
            "prefix80_cp8_64k",
            1640,
            65536,
            65536,
            prefix_ratio=0.8,
        ),
        Case(
            "prefix80_cp8_256k",
            6554,
            262144,
            262144,
            prefix_ratio=0.8,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    properties = torch.cuda.get_device_properties(0)
    result = {
        "benchmark": "topk_v3_tie_break",
        "device": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "multi_processor_count": properties.multi_processor_count,
            "total_memory": properties.total_memory,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "cases": [],
    }
    for case in _cases(args.quick):
        case_result = _run_case(case, args.warmup, args.iterations)
        result["cases"].append(case_result)
        measurements = case_result["measurements"]
        print(
            case.name,
            {name: round(values["median_us"], 3) for name, values in measurements.items()},
            case_result["speedups"],
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
